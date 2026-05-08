import unittest


class SiteCustomizeTests(unittest.TestCase):
    def test_should_skip_runtime_path_bootstrap_for_ray_runtime_env_agent(self):
        import sitecustomize

        should_bootstrap = sitecustomize._should_prepare_runtime_paths(
            "/usr/local/lib/python3.11/dist-packages/ray/_private/runtime_env/agent/main.py"
        )

        self.assertFalse(should_bootstrap)

    def test_should_skip_runtime_path_bootstrap_for_ray_dashboard_agent(self):
        import sitecustomize

        should_bootstrap = sitecustomize._should_prepare_runtime_paths(
            "/usr/local/lib/python3.11/dist-packages/ray/dashboard/agent.py"
        )

        self.assertFalse(should_bootstrap)

    def test_should_keep_runtime_path_bootstrap_for_ray_default_worker(self):
        import sitecustomize

        should_bootstrap = sitecustomize._should_prepare_runtime_paths(
            "/usr/local/lib/python3.11/dist-packages/ray/_private/workers/default_worker.py"
        )

        self.assertTrue(should_bootstrap)

    def test_should_keep_runtime_path_bootstrap_for_user_entrypoint(self):
        import sitecustomize

        should_bootstrap = sitecustomize._should_prepare_runtime_paths(
            "/mnt/bn/chenghao1026/zzh/OrhRL/orchrl/trainer/train.py"
        )

        self.assertTrue(should_bootstrap)

    def test_should_skip_heavy_runtime_compat_patches_for_ray_default_worker(self):
        import sitecustomize

        should_patch = sitecustomize._should_apply_runtime_compat_patches(
            "/usr/local/lib/python3.11/dist-packages/ray/_private/workers/default_worker.py"
        )

        self.assertFalse(should_patch)

    def test_should_keep_heavy_runtime_compat_patches_for_user_entrypoint(self):
        import sitecustomize

        should_patch = sitecustomize._should_apply_runtime_compat_patches(
            "/mnt/bn/chenghao1026/zzh/OrhRL/orchrl/trainer/train.py"
        )

        self.assertTrue(should_patch)
