from .config import MatmulConfig


def generate_matmul(config: MatmulConfig):
    """Compatibility entry point for explicitly building the tiled Metal kernel."""
    from ..scheduler.tiled import TiledMatmulScheduler

    return TiledMatmulScheduler(config).build()
