from __future__ import annotations

"""OrchRL-specific monkey patches for VERL Ray resource pool scheduling.

This module keeps OrchRL's custom placement semantics out of the vendored VERL
source tree. Importing ``orchrl`` applies these patches so existing OrchRL code
can continue importing ``ResourcePoolManager`` from VERL while gaining:

- support for custom placement-group bundle resources
- support for carrying ``n_gpus_per_node`` through pool construction
- early validation that enough labeled Ray nodes exist for the requested pools

The patch is intentionally scoped to OrchRL's runtime. It should not be treated
as a generic VERL extension API.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

import ray
from ray.util.placement_group import placement_group


def patch_verl_resource_pool_manager() -> None:
    """Patch VERL resource-pool classes in-place for OrchRL scheduling needs."""
    from verl.single_controller.ray import base as ray_base_module
    import verl.single_controller.ray as ray_module

    if getattr(ray_base_module, "_orchrl_resource_pool_patch_applied", False):
        return

    original_init = ray_base_module.RayResourcePool.__init__
    original_check_resource_available = ray_base_module.ResourcePoolManager._check_resource_available

    def patched_init(
        self,
        process_on_nodes: Optional[list[int]] = None,
        use_gpu: bool = True,
        name_prefix: str = None,
        max_colocate_count: int = 10,
        detached: bool = False,
        accelerator_type: Optional[str] = None,
        bundle_resources: Optional[dict[str, float]] = None,
        n_gpus_per_node: int = 8,
    ) -> None:
        original_init(
            self,
            process_on_nodes=process_on_nodes,
            use_gpu=use_gpu,
            name_prefix=name_prefix,
            max_colocate_count=max_colocate_count,
            detached=detached,
            accelerator_type=accelerator_type,
        )
        self.n_gpus_per_node = n_gpus_per_node
        self.bundle_resources = dict(bundle_resources or {})

    def patched_get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = (
            name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        )
        if device_name == "npu":
            device_name = "NPU"
        elif device_name == "cuda":
            device_name = "GPU"

        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = 1
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        bundle.update(getattr(self, "bundle_resources", {}))
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None
        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        for idx, pg in enumerate(pgs):
            if not pg.wait(timeout_seconds=300):
                raise TimeoutError(f"Timed out waiting for placement group {pg_name_prefix}{idx} to become ready")

        self.pgs = ray_base_module.sort_placement_group_by_node_ip(pgs)
        return pgs

    @dataclass
    class OrchRLResourcePoolManager(ray_base_module.ResourcePoolManager):
        bundle_resources: dict[str, dict[str, float]] = field(default_factory=dict)
        n_gpus_per_node: int = 8

        def create_resource_pool(self):
            for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
                resource_pool = ray_base_module.RayResourcePool(
                    process_on_nodes=process_on_nodes,
                    use_gpu=True,
                    max_colocate_count=3,
                    name_prefix=resource_pool_name,
                    bundle_resources=self.bundle_resources.get(resource_pool_name),
                    n_gpus_per_node=self.n_gpus_per_node,
                )
                self.resource_pool_dict[resource_pool_name] = resource_pool

            self._check_resource_available()

        def _check_resource_available(self):
            original_check_resource_available(self)

            node_available_resources = ray._private.state.available_resources_per_node()
            for resource_pool_name, resource_pool in self.resource_pool_dict.items():
                requested_bundle_resources = getattr(resource_pool, "bundle_resources", {})
                if not requested_bundle_resources:
                    continue

                satisfiable_nodes = 0
                for node_info in node_available_resources.values():
                    available_gpus = node_info.get("GPU", node_info.get("NPU", 0))
                    if available_gpus < resource_pool.world_size:
                        continue

                    if all(node_info.get(name, 0) >= value for name, value in requested_bundle_resources.items()):
                        satisfiable_nodes += 1

                required_nodes = len(resource_pool.store)
                if satisfiable_nodes < required_nodes:
                    raise ValueError(
                        f"Resource pool '{resource_pool_name}' requires {required_nodes} node(s) with "
                        f"{resource_pool.world_size} GPUs and resources {requested_bundle_resources}, "
                        f"but only {satisfiable_nodes} satisfiable node(s) were found"
                    )

    ray_base_module.RayResourcePool.__init__ = patched_init
    ray_base_module.RayResourcePool.get_placement_groups = patched_get_placement_groups
    ray_base_module.ResourcePoolManager = OrchRLResourcePoolManager
    ray_module.ResourcePoolManager = OrchRLResourcePoolManager

    for module_name in (
        "verl.trainer.ppo.ray_trainer",
        "verl.workers.rollout.replica",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.ResourcePoolManager = OrchRLResourcePoolManager

    ray_base_module._orchrl_resource_pool_patch_applied = True
