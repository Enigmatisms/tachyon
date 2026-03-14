"""Session context — holds loaded report data for Tool execution.

Tools need access to KernelReport objects, the SourceCorrelator, and
AnalyzerRegistry. SessionContext centralizes this shared state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..models.kernel import KernelReport

if TYPE_CHECKING:
    from ..analyzers.base import AnalyzerRegistry
    from ..correlator.source_correlator import (
        ActionHandle,
        SourceCorrelator,
    )


class SessionContext:
    """Shared session state for Tool execution.

    Holds the loaded report's kernel data and provides lookup
    by index. All 9 Tools receive a reference to this context.
    """

    def __init__(
        self,
        kernels: list[KernelReport],
        action: ActionHandle | None = None,
        correlator: SourceCorrelator | None = None,
        registry: AnalyzerRegistry | None = None,
    ) -> None:
        self.kernels = kernels
        self.action = action
        self.correlator = correlator
        self.registry = registry

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
