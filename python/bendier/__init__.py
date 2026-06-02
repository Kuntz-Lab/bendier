from pkgutil import extend_path

# Running scripts from the repo's `python/` folder shadows site-packages.
# Extend package search path so the installed `bendier._bendier` extension can be found.
__path__ = extend_path(__path__, __name__)

from ._bendier import *
from .plotting import (
    CosseratRodMeshManager,
    CosseratRodPlotter,
    ParallelRobotPlotter,
    TendonRobotPlotter,
    setup_plt,
)
