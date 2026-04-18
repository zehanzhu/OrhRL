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
