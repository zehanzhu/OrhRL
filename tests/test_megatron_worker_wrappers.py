import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import sys
import unittest
from unittest import mock

from verl.single_controller.base.decorator import Dispatch, MAGIC_ATTR, register


class MegatronWorkerWrapperTests(unittest.TestCase):
    @contextmanager
    def _load_worker_module(self):
        fake_verl_module = ModuleType("verl.workers.megatron_workers")
        fake_mcore_compat = ModuleType("orchrl.utils.mcore_compat")
        fake_mcore_compat.bootstrap_mcore_runtime_compat = lambda: None
        fake_orchrl_pkg = ModuleType("orchrl")
        fake_orchrl_pkg.__path__ = []
        fake_orchrl_utils_pkg = ModuleType("orchrl.utils")
        fake_orchrl_utils_pkg.__path__ = []
        fake_orchrl_workers_pkg = ModuleType("orchrl.workers")
        fake_orchrl_workers_pkg.__path__ = []

        class _FakeBaseWorker:
            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def init_model(self):
                if not hasattr(self, "call_order"):
                    self.call_order = []
                self.call_order.append("base")
                return "base-init"

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def update_actor(self, data):
                if not hasattr(self, "call_order"):
                    self.call_order = []
                self.call_order.append("base-update")
                return "base-update"

        class _FakeAsyncActorRolloutRefWorker(_FakeBaseWorker):
            pass

        class _FakeCriticWorker(_FakeBaseWorker):
            pass

        fake_verl_module.AsyncActorRolloutRefWorker = _FakeAsyncActorRolloutRefWorker
        fake_verl_module.CriticWorker = _FakeCriticWorker

        module_name = "orchrl.workers.megatron_workers"
        module_path = Path(__file__).resolve().parents[1] / "orchrl" / "workers" / "megatron_workers.py"
        previous_module = sys.modules.pop(module_name, None)
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(module_spec)
        self.assertIsNotNone(module_spec.loader)

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "verl.workers.megatron_workers": fake_verl_module,
                    "orchrl": fake_orchrl_pkg,
                    "orchrl.utils": fake_orchrl_utils_pkg,
                    "orchrl.utils.mcore_compat": fake_mcore_compat,
                    "orchrl.workers": fake_orchrl_workers_pkg,
                },
            ):
                worker_module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_name] = worker_module
                module_spec.loader.exec_module(worker_module)
                yield worker_module
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module

    def test_init_model_overrides_preserve_dispatch_registration(self):
        with self._load_worker_module() as worker_module:
            actor_attrs = getattr(worker_module.AsyncActorRolloutRefWorker.init_model, MAGIC_ATTR, None)
            critic_attrs = getattr(worker_module.CriticWorker.init_model, MAGIC_ATTR, None)

        self.assertIsNotNone(actor_attrs)
        self.assertEqual(actor_attrs["dispatch_mode"], Dispatch.ONE_TO_ALL)
        self.assertIsNotNone(critic_attrs)
        self.assertEqual(critic_attrs["dispatch_mode"], Dispatch.ONE_TO_ALL)

    def test_actor_wrapper_bootstraps_runtime_compat_before_parent_init(self):
        with self._load_worker_module() as worker_module:
            worker = worker_module.AsyncActorRolloutRefWorker()
            worker.call_order = []

            with mock.patch.object(
                worker_module,
                "bootstrap_mcore_runtime_compat",
                side_effect=lambda: worker.call_order.append("bootstrap"),
            ) as bootstrap:
                result = worker.init_model()

        self.assertEqual(result, "base-init")
        self.assertEqual(worker.call_order, ["bootstrap", "base"])
        bootstrap.assert_called_once_with()

    def test_actor_update_actor_adjusts_invalid_mini_batch_size(self):
        with self._load_worker_module() as worker_module:
            worker = worker_module.AsyncActorRolloutRefWorker()

            class _Batch:
                batch_size = [40]

            class _Data:
                batch = _Batch()

            class _ActorCfg(dict):
                def __getattr__(self, key):
                    return self[key]

                def __setattr__(self, key, value):
                    self[key] = value

            worker.config = type("Cfg", (), {})()
            worker.config.actor = _ActorCfg(ppo_mini_batch_size=16)

            with mock.patch.object(
                worker_module.VerlAsyncActorRolloutRefWorker,
                "update_actor",
                return_value="updated",
            ) as parent_update:
                result = worker.update_actor(_Data())

        self.assertEqual(result, "updated")
        self.assertEqual(worker.config.actor.ppo_mini_batch_size, 10)
        parent_update.assert_called_once()

    def test_actor_update_actor_adjusts_inner_actor_mini_batch_size(self):
        with self._load_worker_module() as worker_module:
            worker = worker_module.AsyncActorRolloutRefWorker()

            class _Batch:
                batch_size = [40]

            class _Data:
                batch = _Batch()

            class _ActorCfg(dict):
                def __getattr__(self, key):
                    return self[key]

                def __setattr__(self, key, value):
                    self[key] = value

            worker.config = type("Cfg", (), {})()
            worker.config.actor = _ActorCfg(ppo_mini_batch_size=16)
            worker.actor = type("Actor", (), {})()
            worker.actor.config = _ActorCfg(ppo_mini_batch_size=16)

            with mock.patch.object(
                worker_module.VerlAsyncActorRolloutRefWorker,
                "update_actor",
                return_value="updated",
            ) as parent_update:
                result = worker.update_actor(_Data())

        self.assertEqual(result, "updated")
        self.assertEqual(worker.config.actor.ppo_mini_batch_size, 10)
        self.assertEqual(worker.actor.config.ppo_mini_batch_size, 10)
        parent_update.assert_called_once()

    def test_actor_update_actor_keeps_valid_mini_batch_size(self):
        with self._load_worker_module() as worker_module:
            worker = worker_module.AsyncActorRolloutRefWorker()

            class _Batch:
                batch_size = [40]

            class _Data:
                batch = _Batch()

            class _ActorCfg(dict):
                def __getattr__(self, key):
                    return self[key]

                def __setattr__(self, key, value):
                    self[key] = value

            worker.config = type("Cfg", (), {})()
            worker.config.actor = _ActorCfg(ppo_mini_batch_size=8)

            with mock.patch.object(
                worker_module.VerlAsyncActorRolloutRefWorker,
                "update_actor",
                return_value="updated",
            ) as parent_update:
                result = worker.update_actor(_Data())

        self.assertEqual(result, "updated")
        self.assertEqual(worker.config.actor.ppo_mini_batch_size, 8)
        parent_update.assert_called_once()
