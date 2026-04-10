from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .datatypes import ModelRequest, ModelResponse

MODEL_OVERRIDE_KEY = "model"

_SUPPORTED_SAMPLING_PARAM_KEYS = {
    "best_of",
    "detokenize",
    "early_stopping",
    "frequency_penalty",
    "ignore_eos",
    "include_stop_str_in_output",
    "length_penalty",
    "max_new_tokens",
    "max_tokens",
    "min_p",
    "min_tokens",
    "n",
    "presence_penalty",
    "prompt_logprobs",
    "repetition_penalty",
    "seed",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "stop",
    "stop_token_ids",
    "temperature",
    "top_k",
    "top_p",
    "truncate_prompt_tokens",
}

_CONTROL_PARAM_KEYS = {
    MODEL_OVERRIDE_KEY,
    "logprobs",
    "return_token_ids",
    "stream",
}


class RolloutBackend:
    """Rollout backend that sends prompt IDs directly to VERL actors."""

    def __init__(
        self,
        server_manager: Any | None = None,
        *,
        tokenizer: Any | None = None,
        decoder: Callable[[list[int]], str] | None = None,
        policy_to_manager: Mapping[str, Any] | None = None,
        policy_to_tokenizer: Mapping[str, Any] | None = None,
        policy_to_decoder: Mapping[str, Callable[[list[int]], str]] | None = None,
        policy_to_actual_model: Mapping[str, str] | None = None,
    ) -> None:
        self._server_manager = server_manager
        self._tokenizer = tokenizer
        self._decoder = decoder
        self._policy_to_manager = dict(policy_to_manager or {})
        self._policy_to_tokenizer = dict(policy_to_tokenizer or {})
        self._policy_to_decoder = dict(policy_to_decoder or {})
        self._actual_model_to_policy: dict[str, str] = {}

        if policy_to_actual_model is not None:
            for policy_name, actual_model in policy_to_actual_model.items():
                if isinstance(actual_model, str) and actual_model:
                    self._actual_model_to_policy[actual_model] = policy_name

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if request.prompt_ids is None:
            raise ValueError("RolloutBackend requires prompt_ids")

        manager, tokenizer, decoder = self._resolve_runtime_handles(request)
        output = await manager.generate(
            request_id=request.request_id,
            prompt_ids=list(request.prompt_ids),
            sampling_params=self._build_sampling_params(request.generation_params),
        )
        token_ids = self._normalize_token_sequence(getattr(output, "token_ids", None))
        logprobs = self._normalize_logprobs(getattr(output, "log_probs", None))
        routed_experts = getattr(output, "routed_experts", None)
        raw_stop_reason = getattr(output, "stop_reason", None)
        content = getattr(output, "text", None)
        if not isinstance(content, str) or not content:
            content = self._decode_response_text(token_ids, tokenizer=tokenizer, decoder=decoder)

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=self._normalize_finish_reason(raw_stop_reason),
            prompt_ids=list(request.prompt_ids),
            routed_experts=routed_experts,
            runtime_metadata={
                "raw_stop_reason": raw_stop_reason,
                "render_fingerprint": dict(request.render_fingerprint),
                "sampling_fingerprint": dict(request.sampling_fingerprint),
            },
        )

    def _resolve_runtime_handles(
        self,
        request: ModelRequest,
    ) -> tuple[Any, Any | None, Callable[[list[int]], str] | None]:
        if not self._policy_to_manager:
            if self._server_manager is None:
                raise ValueError("RolloutBackend is missing a server manager")
            return self._server_manager, self._tokenizer, self._decoder

        policy_name = self._resolve_policy_name(request)
        manager = self._policy_to_manager.get(policy_name)
        if manager is None:
            raise ValueError(f"No direct rollout manager configured for policy '{policy_name}'")

        tokenizer = self._policy_to_tokenizer.get(policy_name, self._tokenizer)
        decoder = self._policy_to_decoder.get(policy_name, self._decoder)
        return manager, tokenizer, decoder

    def _resolve_policy_name(self, request: ModelRequest) -> str:
        candidates: list[str] = []
        if isinstance(request.agent_role, str) and request.agent_role:
            candidates.append(request.agent_role)

        model_name = request.generation_params.get(MODEL_OVERRIDE_KEY)
        if isinstance(model_name, str) and model_name:
            candidates.append(model_name)

        for candidate in candidates:
            if candidate in self._policy_to_manager:
                return candidate

            policy_name = self._actual_model_to_policy.get(candidate)
            if policy_name is not None and policy_name in self._policy_to_manager:
                return policy_name

        if len(self._policy_to_manager) == 1:
            return next(iter(self._policy_to_manager))

        raise ValueError(
            f"Unable to resolve policy for RolloutBackend request from candidates={candidates}"
        )

    def _decode_response_text(
        self,
        token_ids: list[int] | None,
        *,
        tokenizer: Any | None,
        decoder: Callable[[list[int]], str] | None,
    ) -> str:
        if not token_ids:
            return ""
        if decoder is not None:
            return decoder(token_ids)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        raise ValueError(
            "RolloutBackend requires a tokenizer or decoder to recover text from token_ids"
        )

    @staticmethod
    def _normalize_token_sequence(raw_token_ids: Any) -> list[int] | None:
        if raw_token_ids is None:
            return None
        if hasattr(raw_token_ids, "tolist"):
            raw_token_ids = raw_token_ids.tolist()
        return [int(token_id) for token_id in raw_token_ids]

    @staticmethod
    def _normalize_logprobs(raw_log_probs: Any) -> list[float] | None:
        if raw_log_probs is None:
            return None
        if hasattr(raw_log_probs, "tolist"):
            raw_log_probs = raw_log_probs.tolist()
        return [float(value) for value in raw_log_probs]

    @staticmethod
    def _build_sampling_params(generation_params: Mapping[str, Any]) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {"logprobs": True}

        for key, value in generation_params.items():
            if key in _CONTROL_PARAM_KEYS:
                if key == "stream" and value:
                    raise ValueError("RolloutBackend does not support stream=True")
                continue

            if key not in _SUPPORTED_SAMPLING_PARAM_KEYS:
                raise ValueError(f"Unsupported generation parameter for RolloutBackend: {key}")

            if key == "n" and value != 1:
                raise ValueError("RolloutBackend currently supports only n=1")

            sampling_params[key] = value

        return sampling_params

    @staticmethod
    def _normalize_finish_reason(raw_stop_reason: Any) -> str:
        if raw_stop_reason in {"stop", "length", "content_filter", "tool_calls", "function_call"}:
            return str(raw_stop_reason)
        if raw_stop_reason in {"completed", "aborted", None}:
            return "stop"
        return "stop"
