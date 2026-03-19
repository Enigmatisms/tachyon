"""Session context — holds loaded report data for Tool execution.

Tools need access to KernelReport objects, the SourceCorrelator, and
AnalyzerRegistry. SessionContext centralizes this shared state.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..models.kernel import KernelReport

if TYPE_CHECKING:
    from ..analyzers.base import AnalyzerRegistry
    from ..correlator.source_correlator import (
        ActionHandle,
        SourceCorrelator,
    )
    from ..correlator.source_mapper import NCUMappingSystem


class SessionContext:
    """Shared session state for Tool execution.

    Holds the loaded report's kernel data and provides lookup
    by index. All 12 Tools receive a reference to this context.
    """

    def __init__(
        self,
        kernels: list[KernelReport],
        action: ActionHandle | None = None,
        correlator: SourceCorrelator | None = None,
        registry: AnalyzerRegistry | None = None,
        mapper: NCUMappingSystem | None = None,
        allowed_source_paths: set[str] | None = None,
    ) -> None:
        self.kernels = kernels
        self.action = action
        self.correlator = correlator
        self.registry = registry
        self.mapper = mapper
        self.allowed_source_paths: set[str] = allowed_source_paths or set()

    @classmethod
    def build_allowed_source_paths(
        cls,
        kernels: list[KernelReport],
        mapper: NCUMappingSystem | None = None,
    ) -> set[str]:
        """Build source file whitelist from kernel metadata + mapper.

        Collects paths from kernel.source_files keys and
        mapper.get_mapped_sources(), keeping only existing files.
        """
        paths: set[str] = set()
        for k in kernels:
            paths.update(k.source_files.keys())
        if mapper is not None:
            try:
                paths.update(mapper.get_mapped_sources())
            except Exception:
                pass
        # Keep only existing files
        return {p for p in paths if os.path.isfile(p)}

    def get_kernel(self, kernel_id: int) -> KernelReport:
        """Get KernelReport by index.

        Raises:
            IndexError: If kernel_id is out of range.
        """
        if kernel_id < 0 or kernel_id >= len(self.kernels):
            raise IndexError(
                f"kernel_id={kernel_id} out of range. "
                f"Valid range: 0..{len(self.kernels) - 1}"
            )
        return self.kernels[kernel_id]

    @property
    def kernel_count(self) -> int:
        """Number of kernels in the session."""
        return len(self.kernels)
