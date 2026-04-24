import sys
import unittest
from pathlib import Path


class VerlResourcePoolPatchTests(unittest.TestCase):
    def test_importing_orchrl_patches_verl_resource_pool_manager(self):
        repo_root = Path(__file__).resolve().parents[2]
        orchrl_root = repo_root / "OrchRL"
        orchrl_root_str = str(orchrl_root)
        if orchrl_root_str not in sys.path:
            sys.path.insert(0, orchrl_root_str)

        import orchrl  # noqa: F401
        from verl.single_controller.ray import ResourcePoolManager
        from verl.single_controller.ray import base as ray_base

        self.assertTrue(
            getattr(ray_base, "_orchrl_resource_pool_patch_applied", False),
            "Importing orchrl should apply the VERL resource-pool patch",
        )
        self.assertIn(
            "bundle_resources",
            ResourcePoolManager.__dataclass_fields__,
            "Patched ResourcePoolManager should accept OrchRL bundle_resources",
        )
        self.assertIn(
            "n_gpus_per_node",
            ResourcePoolManager.__dataclass_fields__,
            "Patched ResourcePoolManager should accept OrchRL n_gpus_per_node",
        )


if __name__ == "__main__":
    unittest.main()
