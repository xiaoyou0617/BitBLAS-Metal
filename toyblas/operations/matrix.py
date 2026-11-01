from .config import MatmulConfig


def generate_matmul(config: MatmulConfig):
    """Compatibility entry point for explicitly building the selected Metal kernel."""
    from ..scheduler.dispatch import select_scheduler

    return select_scheduler(config, device="mps", backend="tilelang").build()
