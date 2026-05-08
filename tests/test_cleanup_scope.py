import os
import signal
from pathlib import Path
import unittest
from unittest import mock


class CleanupScopeTests(unittest.TestCase):
    def setUp(self):
        import orchrl.utils.clean_up as clean_up

        clean_up._CLEANED = False
        clean_up._TEMP_DIRS.clear()
        clean_up._RAY_PROCESS_MATCHERS.clear()

    def tearDown(self):
        import orchrl.utils.clean_up as clean_up

        clean_up._CLEANED = False
        clean_up._TEMP_DIRS.clear()
        clean_up._RAY_PROCESS_MATCHERS.clear()

    def test_cleanup_runtime_kills_only_registered_session_processes(self):
        import orchrl.utils.clean_up as clean_up

        clean_up.register_ray_process_matchers(
            "/tmp/verl_ray_123",
            "/tmp/verl_ray_123/session_abc",
        )

        with (
            mock.patch("orchrl.utils.clean_up.ray.is_initialized", return_value=False),
            mock.patch("orchrl.utils.clean_up.shutil.rmtree"),
            mock.patch("orchrl.utils.clean_up.subprocess.run") as run_mock,
        ):
            clean_up.cleanup_ray_runtime()

        issued_commands = [call.args[0] for call in run_mock.call_args_list]
        self.assertEqual(
            issued_commands,
            [
                ["pkill", "-9", "-f", "/tmp/verl_ray_123"],
                ["pkill", "-9", "-f", "/tmp/verl_ray_123/session_abc"],
            ],
        )

    def test_init_ray_registers_session_scoped_cleanup_targets(self):
        from orchrl.utils.ray_utils import init_ray_with_temp_dirs

        fake_context = mock.Mock(
            address_info={"session_dir": "/tmp/verl_ray_456/session_xyz"}
        )

        with (
            mock.patch("orchrl.utils.ray_utils.os.makedirs"),
            mock.patch("orchrl.utils.ray_utils.os.getpid", return_value=456),
            mock.patch("orchrl.utils.ray_utils.ray.is_initialized", return_value=False),
            mock.patch("orchrl.utils.ray_utils.ray.init", return_value=fake_context),
            mock.patch("orchrl.utils.clean_up.register_temp_dirs") as register_temp_dirs_mock,
            mock.patch(
                "orchrl.utils.clean_up.register_ray_process_matchers"
            ) as register_matchers_mock,
        ):
            init_ray_with_temp_dirs()

        register_temp_dirs_mock.assert_called_once_with(
            "/tmp/verl_ray_456",
            "/tmp/verl_spill_456",
        )
        register_matchers_mock.assert_called_once_with(
            "/tmp/verl_ray_456",
            "/tmp/verl_ray_456/session_xyz",
        )

    def test_init_ray_redirects_logs_to_writable_run_dir(self):
        from orchrl.utils.ray_utils import init_ray_with_temp_dirs

        fake_context = mock.Mock(address_info={"session_dir": "/tmp/verl_ray_789/session_xyz"})
        config = mock.Mock()
        config.resource = mock.Mock(n_gpus_per_node=1)
        config.training = mock.Mock(run_dir="outputs/training_runs/test_run")

        expected_dir = str(
            Path("/tmp/repo/outputs/training_runs/test_run/ray_logs").resolve()
        )

        with (
            mock.patch.dict(
                os.environ,
                {"BYTED_RAY_REDIRECT_LOG": "/var/log/tiger/ray/session_latest/logs"},
                clear=False,
            ),
            mock.patch("orchrl.utils.ray_utils.os.makedirs"),
            mock.patch("orchrl.utils.ray_utils.os.getpid", return_value=789),
            mock.patch("orchrl.utils.ray_utils.ray.is_initialized", return_value=False),
            mock.patch("orchrl.utils.ray_utils.ray.init", return_value=fake_context) as ray_init_mock,
            mock.patch("orchrl.utils.clean_up.register_temp_dirs"),
            mock.patch("orchrl.utils.clean_up.register_ray_process_matchers"),
            mock.patch("orchrl.utils.ray_utils._can_write_directory", side_effect=[False, True]),
            mock.patch("orchrl.utils.ray_utils.Path.cwd", return_value=Path("/tmp/repo")),
        ):
            init_ray_with_temp_dirs(config)
            self.assertEqual(os.environ["BYTED_RAY_REDIRECT_LOG"], expected_dir)

        self.assertTrue(ray_init_mock.called)

    def test_init_ray_local_runtime_disables_dashboard_and_forces_ipv4(self):
        from orchrl.utils.ray_utils import init_ray_with_temp_dirs

        fake_context = mock.Mock(
            address_info={"session_dir": "/tmp/verl_ray_321/session_xyz"}
        )

        with (
            mock.patch.dict(
                os.environ,
                {
                    "BYTED_RAY_POD_IP": "2605:340:cd51:602:d46a:ef3e:9ae4:cb96",
                    "RAY_IP": "2605:340:cd51:602:d46a:ef3e:9ae4:cb96",
                    "MY_HOST_IP": "10.122.123.189",
                },
                clear=False,
            ),
            mock.patch("orchrl.utils.ray_utils.os.makedirs"),
            mock.patch("orchrl.utils.ray_utils.os.getpid", return_value=321),
            mock.patch("orchrl.utils.ray_utils.ray.is_initialized", return_value=False),
            mock.patch("orchrl.utils.ray_utils.ray.init", return_value=fake_context) as ray_init_mock,
            mock.patch("orchrl.utils.clean_up.register_temp_dirs"),
            mock.patch("orchrl.utils.clean_up.register_ray_process_matchers"),
        ):
            init_ray_with_temp_dirs()
            ray_init_kwargs = ray_init_mock.call_args.kwargs
            self.assertFalse(ray_init_kwargs["include_dashboard"])
            self.assertEqual(ray_init_kwargs["_node_ip_address"], "10.122.123.189")
            self.assertEqual(os.environ["BYTED_RAY_POD_IP"], "10.122.123.189")
            self.assertEqual(os.environ["RAY_IP"], "10.122.123.189")

    def test_local_ray_process_env_overrides_force_ipv4_for_child_processes(self):
        from orchrl.utils.ray_utils import _init_local_ray_with_process_env_overrides
        import ray._private.services as ray_services

        observed_env_updates = {}

        def fake_start_ray_process(*args, **kwargs):
            observed_env_updates.update(kwargs.get("env_updates") or {})
            return object()

        def fake_ray_init(**kwargs):
            ray_services.start_ray_process(
                command=["fake-raylet"],
                process_type="raylet",
                fate_share=False,
            )
            return mock.Mock(address_info={})

        with (
            mock.patch.object(ray_services, "start_ray_process", side_effect=fake_start_ray_process),
            mock.patch("orchrl.utils.ray_utils.ray.init", side_effect=fake_ray_init),
        ):
            _init_local_ray_with_process_env_overrides(
                {"include_dashboard": False, "_node_ip_address": "10.122.123.189"},
                "10.122.123.189",
            )

        self.assertEqual(observed_env_updates["BYTED_RAY_POD_IP"], "10.122.123.189")
        self.assertEqual(observed_env_updates["RAY_IP"], "10.122.123.189")
        self.assertEqual(observed_env_updates["MY_HOST_IP"], "10.122.123.189")
        self.assertEqual(observed_env_updates["MY_POD_IP"], "10.122.123.189")
        self.assertEqual(observed_env_updates["MY_HOST_IPV6"], "")
        self.assertEqual(observed_env_updates["MY_POD_IPV6"], "")

    def test_init_ray_local_runtime_uses_system_cpu_count_by_default(self):
        from orchrl.utils.ray_utils import init_ray_with_temp_dirs

        fake_context = mock.Mock(
            address_info={"session_dir": "/tmp/verl_ray_654/session_xyz"}
        )
        config = mock.Mock()
        config.resource = mock.Mock(n_gpus_per_node=8, trainer_remote_num_cpus=None)
        config.resource.ray_address = None
        config.training = mock.Mock(run_dir="outputs/training_runs/test_run")
        config.ray_kwargs = None

        with (
            mock.patch("orchrl.utils.ray_utils.os.makedirs"),
            mock.patch("orchrl.utils.ray_utils.os.getpid", return_value=654),
            mock.patch("orchrl.utils.ray_utils.os.cpu_count", return_value=120),
            mock.patch("orchrl.utils.ray_utils.ray.is_initialized", return_value=False),
            mock.patch(
                "orchrl.utils.ray_utils._init_local_ray_with_process_env_overrides",
                return_value=fake_context,
            ) as init_local_mock,
            mock.patch("orchrl.utils.clean_up.register_temp_dirs"),
            mock.patch("orchrl.utils.clean_up.register_ray_process_matchers"),
        ):
            init_ray_with_temp_dirs(config)

        ray_init_kwargs = init_local_mock.call_args.args[0]
        self.assertEqual(ray_init_kwargs["num_cpus"], 120)

    def test_local_ray_process_env_overrides_disable_worker_prestart(self):
        from orchrl.utils.ray_utils import _build_local_ray_process_env_overrides

        env_overrides = _build_local_ray_process_env_overrides("10.122.123.189")

        self.assertEqual(env_overrides["RAY_enable_worker_prestart"], "false")
        self.assertEqual(env_overrides["RAY_prestart_worker_first_driver"], "false")

    def test_install_cleanup_hooks_use_runtime_cleanup_for_exit_and_signals(self):
        import orchrl.utils.clean_up as clean_up

        with (
            mock.patch("orchrl.utils.clean_up.atexit.register") as register_mock,
            mock.patch("orchrl.utils.clean_up.signal.signal") as signal_mock,
        ):
            clean_up.install_cleanup_hooks()

        register_mock.assert_called_once_with(clean_up.cleanup_ray_runtime)

        handlers = {call.args[0]: call.args[1] for call in signal_mock.call_args_list}

        with (
            mock.patch("orchrl.utils.clean_up.cleanup_ray_runtime") as cleanup_mock,
            mock.patch("orchrl.utils.clean_up.sys.exit") as exit_mock,
        ):
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        cleanup_mock.assert_called_once_with()
        exit_mock.assert_called_once_with(0)
