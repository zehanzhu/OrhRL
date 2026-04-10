from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from aiohttp import web
import ray

from ._support.renderer import ChatRenderer
from ._support.replay_cache import ReplayCache
from .backend import RolloutBackend
from .datatypes import InteractionRecord, ModelMappingEntry, ModelRequest, ModelResponse


def _validate_runtime_request(request: ModelRequest) -> None:
    if request.prompt_ids is None:
        raise ValueError("runtime prompt_ids are required on token-id paths")


def _validate_runtime_response(response: ModelResponse) -> None:
    if response.token_ids is None or not response.token_ids:
        raise ValueError("response token_ids must not be empty")
    if response.logprobs is not None and len(response.token_ids) != len(response.logprobs):
        raise ValueError("response token_ids/logprobs length mismatch")


def _build_drift_artifact(
    *,
    messages: list[dict[str, Any]],
    runtime_prompt_ids: list[int] | None,
    rerendered_prompt_ids: list[int] | None,
    response_ids: list[int] | None,
    response_logprobs: list[float] | None,
    render_fingerprint: dict[str, Any] | None,
    sampling_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "messages": messages,
        "runtime_prompt_ids": runtime_prompt_ids,
        "rerendered_prompt_ids": rerendered_prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "render_fingerprint": render_fingerprint or {},
        "sampling_fingerprint": sampling_fingerprint or {},
        "mismatch": runtime_prompt_ids != rerendered_prompt_ids,
    }


@ray.remote(num_cpus=0)
class MonitorActor:
    def __init__(
        self,
        *,
        backend: RolloutBackend,
        model_mapping: dict[str, ModelMappingEntry],
        host: str = "127.0.0.1",
        port: int = 0,
        renderer: ChatRenderer | Mapping[str, ChatRenderer] | None = None,
    ) -> None:
        self._backend = backend
        self._model_mapping = model_mapping
        self._host = host
        self._requested_port = port
        if isinstance(renderer, Mapping):
            self._renderer = dict(renderer)
        else:
            self._renderer = renderer

        self._buffer: list[InteractionRecord] = []
        self._turn_counters: dict[str, int] = {}
        self._buffer_generation = 0
        self._state_lock = threading.Lock()

        self._active_lease_id: str | None = None
        self._episode_id: str | None = None
        self._replay_cache: ReplayCache | None = None

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._port: int | None = None

    async def start_server_once(self) -> int:
        if self._runner is not None:
            if self._port is None:
                raise RuntimeError("MonitorActor server is in inconsistent started state")
            return self._port

        self._app = web.Application()
        self._app.router.add_post(
            "/leases/{lease_id}/v1/chat/completions",
            self._handle_chat_completions,
        )

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, host=self._host, port=self._requested_port)
        await self._site.start()

        server = self._site._server
        if server is None or not server.sockets:
            await self.stop_server()
            raise RuntimeError("failed to bind MonitorActor server socket")

        self._port = int(server.sockets[0].getsockname()[1])
        return self._port

    async def stop_server(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

        self._app = None
        self._runner = None
        self._site = None
        self._port = None

    async def begin_run(
        self,
        *,
        episode_id: str | None = None,
        replay_cache: ReplayCache | None = None,
    ) -> str:
        await self.start_server_once()
        with self._state_lock:
            if self._active_lease_id is not None:
                raise RuntimeError("MonitorActor already has an active lease")
            self._active_lease_id = uuid.uuid4().hex
            self._episode_id = episode_id or uuid.uuid4().hex
            self._replay_cache = replay_cache
            self._buffer.clear()
            self._turn_counters.clear()
            self._buffer_generation += 1
            return self._active_lease_id

    def get_server_base(self) -> str:
        if self._port is None:
            raise RuntimeError("MonitorActor server has not been started")
        return f"http://{self._host}:{self._port}"

    def get_lease_base_url(self, lease_id: str) -> str:
        return f"{self.get_server_base()}/leases/{lease_id}/v1"

    def snapshot_buffer(self, lease_id: str) -> list[InteractionRecord]:
        self._require_active_lease(lease_id)
        with self._state_lock:
            return list(self._buffer)

    def reset_run_state(self, lease_id: str) -> None:
        self._require_active_lease(lease_id)
        with self._state_lock:
            self._buffer.clear()
            self._turn_counters.clear()
            self._buffer_generation += 1
            self._active_lease_id = None
            self._episode_id = None
            self._replay_cache = None

    def is_idle(self) -> bool:
        with self._state_lock:
            return self._active_lease_id is None

    def _require_active_lease(self, lease_id: str) -> None:
        with self._state_lock:
            active_lease_id = self._active_lease_id
        if active_lease_id != lease_id:
            raise ValueError(f"inactive lease_id: {lease_id}")

    async def _handle_chat_completions(self, http_request: web.Request) -> web.Response:
        lease_id = http_request.match_info.get("lease_id")
        with self._state_lock:
            active_lease_id = self._active_lease_id
            episode_id = self._episode_id
        if active_lease_id is None or lease_id != active_lease_id or episode_id is None:
            return web.json_response({"error": "inactive lease"}, status=409)

        try:
            body = await http_request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": "request body must be a JSON object"}, status=400)

        agent_role = body.get("model")
        if not isinstance(agent_role, str) or agent_role not in self._model_mapping:
            return web.json_response({"error": f"unknown agent role: {agent_role}"}, status=400)
        mapping_entry = self._model_mapping[agent_role]

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return web.json_response({"error": "messages must be a list"}, status=400)

        generation_params = {k: v for k, v in body.items() if k not in {"model", "messages"}}
        if mapping_entry.actual_model is not None:
            generation_params["model"] = mapping_entry.actual_model

        with self._state_lock:
            turn_index = self._turn_counters.get(agent_role, 0)
            self._turn_counters[agent_role] = turn_index + 1
            generation_snapshot = self._buffer_generation
            replay_cache = self._replay_cache

        request_id = uuid.uuid4().hex
        model_request = ModelRequest(
            request_id=request_id,
            agent_role=agent_role,
            messages=messages,
            generation_params=generation_params,
            sampling_fingerprint=dict(generation_params),
        )
        renderer = self._resolve_renderer(agent_role)
        if renderer is not None and model_request.prompt_ids is None:
            prompt_ids, render_fingerprint = renderer.render(
                messages,
                add_generation_prompt=True,
            )
            model_request.prompt_ids = prompt_ids
            model_request.render_fingerprint = render_fingerprint

        response = None
        replayed = False
        if replay_cache is not None:
            response = replay_cache.lookup(agent_role, turn_index, messages)
            replayed = response is not None

        if response is None:
            try:
                if model_request.prompt_ids is not None:
                    _validate_runtime_request(model_request)
                response = await self._backend.generate(model_request)
                if model_request.prompt_ids is not None:
                    _validate_runtime_response(response)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=502)
        elif model_request.prompt_ids is not None:
            try:
                _validate_runtime_response(response)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=502)

        metadata = dict(getattr(response, "runtime_metadata", {}))
        if response.routed_experts is not None:
            metadata["routed_experts"] = response.routed_experts
        if model_request.prompt_ids is not None:
            metadata.setdefault(
                "drift_artifact",
                _build_drift_artifact(
                    messages=messages,
                    runtime_prompt_ids=response.prompt_ids or model_request.prompt_ids,
                    rerendered_prompt_ids=model_request.prompt_ids,
                    response_ids=response.token_ids,
                    response_logprobs=response.logprobs,
                    render_fingerprint=model_request.render_fingerprint,
                    sampling_fingerprint=model_request.sampling_fingerprint,
                ),
            )
        if replayed:
            metadata["replayed"] = True

        record = InteractionRecord(
            agent_role=agent_role,
            turn_index=turn_index,
            timestamp=time.time(),
            messages=messages,
            generation_params=generation_params,
            response_text=response.content,
            token_ids=response.token_ids,
            logprobs=response.logprobs,
            finish_reason=response.finish_reason,
            episode_id=episode_id,
            prompt_ids=response.prompt_ids or model_request.prompt_ids,
            metadata=metadata,
        )
        with self._state_lock:
            if generation_snapshot == self._buffer_generation and lease_id == self._active_lease_id:
                self._buffer.append(record)

        payload: dict[str, Any] = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent_role,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        return web.json_response(payload)

    def _resolve_renderer(self, agent_role: str) -> ChatRenderer | None:
        if self._renderer is None:
            return None
        if isinstance(self._renderer, ChatRenderer):
            return self._renderer
        return self._renderer.get(agent_role)
