"""NcuReportReader -- .ncu-rep binary report parser.

Wraps NVIDIA's ncu_report.py to extract kernel data into KernelReport models.

Runtime discovery of ncu_report.py follows a strict priority order:
1. TACHYON_NCU_REPORT_PATH environment variable
2. TachyonConfig tools.ncu_report_path (from config.toml or CLI)
3. Well-known installation paths with version globs
4. Already importable via PYTHONPATH / sys.path
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from tachyon.correlator.source_correlator import SourceInfo
from tachyon.errors.handler import ErrorCode, ToolResult
from tachyon.models.kernel import (
    DeviceInfo,
    InstancedMetricValue,
    KernelReport,
    LaunchParams,
    MetricValue,
    RuleResult,
)

logger = logging.getLogger(__name__)

# Well-known base directories where nsight-compute may be installed.
# Each entry is (base_dir, version_glob_needed).
# The discovery logic globs {base_dir}/{version_glob}/extras/python when
# version_glob_needed is True, or checks {base_dir}/extras/python directly.
_NCU_SEARCH_BASES: list[tuple[str, bool]] = [
    # Typical Linux system-wide: /opt/nvidia/nsight-compute/<version>/
    ("/opt/nvidia/nsight-compute", True),
    # CUDA toolkit bundled (no version subdir)
    ("/usr/local/cuda/nsight-compute", False),
    # CUDA toolkit bundled (with version subdir)
    ("/usr/local/cuda/nsight-compute", True),
    # Multi-CUDA: /usr/local/cuda-12.x/nsight-compute/<version>/
    ("/usr/local/cuda-{cuda_ver}/nsight-compute", True),
    # User-local installations
    ("{home}/.local/nvidia/nsight-compute", True),
]

_NCU_VERSION_GLOBS = ["2025.*", "2024.*", "2023.*"]
_CUDA_VERSION_GLOBS = ["12.*", "11.*"]


def _resolve_path_arg(raw: str) -> Path | None:
    """Resolve a user-supplied path that may point to a directory or file.

    Returns the *directory* containing ncu_report.py, or None if invalid.
    """
    p = Path(raw).expanduser().resolve()
    if p.is_dir() and (p / "ncu_report.py").exists():
        return p
    if p.is_file() and p.name == "ncu_report.py":
        return p.parent
    return None


def _discover_ncu_report_path(
    config_path: str | None = None,
) -> Path | None:
    """Search for ncu_report.py across known installation paths.

    Priority:
    1. TACHYON_NCU_REPORT_PATH environment variable  (highest)
    2. *config_path* argument (from TachyonConfig tools.ncu_report_path)
    3. Glob search through _NCU_SEARCH_PATHS x _NCU_VERSION_GLOBS
    4. Already importable (user added to PYTHONPATH)  (lowest)

    Returns the directory containing ``ncu_report.py``, or ``None`` when the
    module is already importable and no explicit path is needed.
    """
    # ── Priority 1: explicit env var ──
    env_path = os.environ.get("TACHYON_NCU_REPORT_PATH")
    if env_path:
        resolved = _resolve_path_arg(env_path)
        if resolved is not None:
            return resolved
        logger.warning(
            "TACHYON_NCU_REPORT_PATH=%s does not contain ncu_report.py; ignoring.",
            env_path,
        )

    # ── Priority 2: config file value ──
    if config_path:
        resolved = _resolve_path_arg(config_path)
        if resolved is not None:
            return resolved
        logger.warning(
            "tools.ncu_report_path=%s does not contain ncu_report.py; ignoring.",
            config_path,
        )

    # ── Priority 3: well-known paths with version globs ──
    home = str(Path.home())
    for base_template, needs_version in _NCU_SEARCH_BASES:
        # Expand {cuda_ver} templates into multiple concrete bases
        if "{cuda_ver}" in base_template:
            bases: list[Path] = []
            parent = Path(base_template.split("{cuda_ver}")[0].rstrip("/")).parent
            if parent.exists():
                for cuda_glob in _CUDA_VERSION_GLOBS:
                    expanded = base_template.format(home=home, cuda_ver=cuda_glob)
                    bases.extend(sorted(parent.glob(Path(expanded).name + "/nsight-compute"), reverse=True))
            # De-duplicate while preserving order
            seen: set[str] = set()
            unique_bases: list[Path] = []
            for b in bases:
                if str(b) not in seen:
                    seen.add(str(b))
                    unique_bases.append(b)
            bases = unique_bases
        else:
            bases = [Path(base_template.format(home=home))]

        for base_dir in bases:
            if not base_dir.exists():
                continue
            if needs_version:
                for version_glob in _NCU_VERSION_GLOBS:
                    for ver_dir in sorted(
                        base_dir.glob(version_glob), reverse=True
                    ):
                        candidate = ver_dir / "extras" / "python"
                        if (candidate / "ncu_report.py").exists():
                            return candidate
            else:
                candidate = base_dir / "extras" / "python"
                if (candidate / "ncu_report.py").exists():
                    return candidate

    # ── Priority 4: already importable ──
    try:
        importlib.import_module("ncu_report")
        return None  # None signals "already on sys.path"
    except ImportError:
        pass

    return None


def _load_ncu_module(config_path: str | None = None) -> Any:
    """Import the ``ncu_report`` module, adding to sys.path if needed.

    Args:
        config_path: Optional explicit path from
            ``TachyonConfig.tools.ncu_report_path``.
    """
    path = _discover_ncu_report_path(config_path=config_path)
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))
        logger.info("Added ncu_report.py path: %s", path)

    try:
        return importlib.import_module("ncu_report")
    except ImportError as e:
        raise ImportError(
            "Cannot import ncu_report. Install NVIDIA Nsight Compute and set "
            "TACHYON_NCU_REPORT_PATH, or add the extras/python directory to "
            "PYTHONPATH."
        ) from e


# ---------------------------------------------------------------------------
# NcuReportReader
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ActionHandle implementation
# ---------------------------------------------------------------------------


class NcuReportReader:
    """Reads ``.ncu-rep`` files and produces :class:`KernelReport` instances.

    This is the pipeline entry point -- every downstream Analyzer, Correlator,
    and Report layer consumes ``KernelReport`` objects produced here.

    Also implements ActionHandle protocol for source correlation:
    - source_info(pc): Map PC -> source file + line number
    - sass_by_pc(pc): Map PC -> SASS disassembly text
    - ptx_by_pc(pc): Map PC -> PTX intermediate representation text
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the reader, discovering and loading ncu_report.

        Args:
            config: Optional ``TachyonConfig`` instance or explicit path string.
                If a ``TachyonConfig``, extracts ``tools.ncu_report_path``.
                If a string, treated as a direct path to ncu_report.py's directory.
                Passed through to :func:`_load_ncu_module` for priority-2 lookup.
        """
        config_path: str | None = None
        if config is not None:
            # Accept TachyonConfig object — extract the ncu_report_path
            if hasattr(config, "tools") and hasattr(config.tools, "ncu_report_path"):
                config_path = config.tools.ncu_report_path
            elif isinstance(config, str):
                config_path = config
        self._ncu = _load_ncu_module(config_path=config_path)
        # Store action handles keyed by kernel name for source correlation
        self._actions: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> ToolResult[list[KernelReport]]:
        """Load all kernel launches from an ``.ncu-rep`` file.

        Returns :class:`ToolResult` to enable graceful error handling at
        the CLI layer.  On success ``result.data`` is a list of
        :class:`KernelReport`.
        """
        path = Path(path)

        if not path.exists():
            return ToolResult.fail(
                ErrorCode.INVALID_BINARY,
                f"File not found: {path}",
                suggestion="Check the file path and ensure the .ncu-rep file exists.",
            )

        if path.suffix != ".ncu-rep":
            return ToolResult.fail(
                ErrorCode.UNSUPPORTED_FORMAT,
                f"Unsupported file format: {path.suffix}",
                suggestion=(
                    "Tachyon only supports .ncu-rep files from NCU 2023.x-2025.x."
                ),
            )

        try:
            context = self._ncu.load_report(str(path))
        except Exception as e:
            return ToolResult.fail(
                ErrorCode.INVALID_BINARY,
                f"Failed to load report: {e}",
                suggestion=(
                    "The file may be corrupt or from an unsupported NCU version."
                ),
            )

        reports: list[KernelReport] = []
        self._actions.clear()
        for range_idx in range(context.num_ranges()):
            ncu_range = context.range_by_idx(range_idx)
            for action_idx in range(ncu_range.num_actions()):
                action = ncu_range.action_by_idx(action_idx)
                try:
                    report = self._build_kernel_report(action)
                    # Store action for source correlation by kernel name
                    self._actions[report.kernel_name] = action
                    reports.append(report)
                except Exception:
                    logger.warning(
                        "Skipping unparseable kernel action %d in range %d",
                        action_idx,
                        range_idx,
                        exc_info=True,
                    )

        logger.info("Loaded %d kernel launches from %s", len(reports), path.name)
        return ToolResult.ok(reports)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kernel_report(self, action: Any) -> KernelReport:
        """Convert a single IAction into a :class:`KernelReport`."""
        return KernelReport(
            kernel_name=action.name(),
            demangled_name=self._safe_demangled_name(action),
            launch_params=self._extract_launch_params(action),
            device_info=self._extract_device_info(action),
            metrics=self._extract_metrics(action),
            instanced_metrics=self._extract_instanced_metrics(action),
            source_files=self._extract_source_files(action),
            rule_results=self._extract_rule_results(action),
        )

    # -- name --------------------------------------------------------

    def _safe_demangled_name(self, action: Any) -> str:
        """Get demangled kernel name with fallback to mangled name."""
        try:
            return action.demangled_name()
        except (AttributeError, RuntimeError):
            return action.name()

    # -- launch params ------------------------------------------------

    def _extract_launch_params(self, action: Any) -> LaunchParams:
        """Extract CUDA launch configuration from action metrics."""

        def _int_metric(name: str, default: int = 0) -> int:
            try:
                m = action.metric_by_name(name)
                if m is not None and m.has_value():
                    return int(m.as_uint64())
            except (RuntimeError, ValueError, AttributeError):
                pass
            return default

        return LaunchParams(
            grid=(
                _int_metric("launch__grid_dim_x", 1),
                _int_metric("launch__grid_dim_y", 1),
                _int_metric("launch__grid_dim_z", 1),
            ),
            block=(
                _int_metric("launch__block_dim_x", 1),
                _int_metric("launch__block_dim_y", 1),
                _int_metric("launch__block_dim_z", 1),
            ),
            shared_mem_bytes=_int_metric("launch__shared_mem_per_block_dynamic"),
            registers_per_thread=_int_metric("launch__registers_per_thread"),
            static_shared_mem_bytes=_int_metric(
                "launch__shared_mem_per_block_static"
            ),
        )

    # -- device info --------------------------------------------------

    def _extract_device_info(self, action: Any) -> DeviceInfo:
        """Extract GPU device properties from action metrics."""

        def _int_metric(name: str, default: int = 0) -> int:
            try:
                m = action.metric_by_name(name)
                if m is not None and m.has_value():
                    return int(m.as_uint64())
            except (RuntimeError, ValueError, AttributeError):
                pass
            return default

        def _double_metric(name: str, default: float = 0.0) -> float:
            try:
                m = action.metric_by_name(name)
                if m is not None and m.has_value():
                    return m.as_double()
            except (RuntimeError, ValueError, AttributeError):
                pass
            return default

        peak_bw_bytes = _double_metric("dram__bytes.sum.peak_sustained")

        return DeviceInfo(
            name=self._get_string_metric(
                action, "device__attribute_display_name", "Unknown GPU"
            ),
            compute_capability=(
                _int_metric("device__attribute_compute_capability_major"),
                _int_metric("device__attribute_compute_capability_minor"),
            ),
            sm_count=_int_metric("device__attribute_multiprocessor_count"),
            max_clock_mhz=_int_metric("device__attribute_clock_rate") // 1000,
            memory_bus_width=_int_metric(
                "device__attribute_global_memory_bus_width"
            ),
            peak_memory_bandwidth_gbps=(
                peak_bw_bytes / 1e9 if peak_bw_bytes > 0 else 0.0
            ),
        )

    # -- string metric helper -----------------------------------------

    def _get_string_metric(
        self, action: Any, name: str, default: str = ""
    ) -> str:
        """Get string-valued metric with fallback."""
        try:
            m = action.metric_by_name(name)
            if m is not None and m.has_value():
                return str(m.value())
        except (AttributeError, RuntimeError):
            pass
        return default

    # -- scalar metrics -----------------------------------------------

    def _extract_metrics(self, action: Any) -> dict[str, MetricValue]:
        """Extract all scalar (non-instanced) metrics from the action.

        Iterates over every available metric name and captures those with
        scalar values.  For metrics with multiple instances (e.g. DRAM
        throughput has per-partition instances), the aggregate rollup value
        (``as_double()`` without instance index) is still captured as a
        scalar.  True per-PC instanced metrics are *also* handled by
        :meth:`_extract_instanced_metrics` for source-level attribution.
        """
        metrics: dict[str, MetricValue] = {}
        try:
            metric_names = action.metric_names()
        except (AttributeError, RuntimeError):
            return metrics

        # Prefixes that are truly per-PC and should ONLY go through
        # _extract_instanced_metrics (not stored as scalars).
        _PER_PC_PREFIXES = (
            "smsp__pcsamp_warps_issue_stalled_",
            "smsp__pcsamp_warp_stall_reason_",
            "inst_executed",
            "thread_inst_executed",
        )

        for name in metric_names:
            try:
                m = action.metric_by_name(name)
                if m is None or not m.has_value():
                    continue

                # True per-PC metrics: skip here, handled by _extract_instanced_metrics
                if m.num_instances() > 1 and any(
                    name.startswith(p) for p in _PER_PC_PREFIXES
                ):
                    continue

                # For all other metrics (including multi-instance aggregates
                # like dram__throughput), as_double() returns the rollup value.
                metrics[name] = MetricValue(
                    name=name,
                    value=m.as_double(),
                    unit=self._infer_unit(name),
                )
            except (RuntimeError, ValueError):
                # Some metrics may not support as_double(); skip gracefully
                continue
        return metrics

    # -- instanced metrics --------------------------------------------

    def _extract_instanced_metrics(
        self, action: Any
    ) -> dict[str, list[InstancedMetricValue]]:
        """Extract per-PC instanced metrics.

        These are metrics with ``num_instances() > 1``, where each instance
        corresponds to a specific PC address via ``correlation_ids()``.

        Extracts both warp-stall PC-sampling metrics AND execution count metrics
        required by SourceCorrelator for hotspot detection.
        """
        instanced: dict[str, list[InstancedMetricValue]] = {}

        # M1 scope: warp-stall PC-sampling metrics (both new and legacy formats)
        stall_prefixes = (
            "smsp__pcsamp_warps_issue_stalled_",
            "smsp__pcsamp_warp_stall_reason_",
        )
        # M2 scope: execution count metrics (inst_executed, thread_inst_executed_true)
        exec_names = ("inst_executed", "thread_inst_executed_true")

        try:
            metric_names = action.metric_names()
        except (AttributeError, RuntimeError):
            return instanced

        # Debug: log what metrics are available
        stall_candidates = [n for n in metric_names if any(n.startswith(p) for p in stall_prefixes)]
        exec_candidates = [n for n in metric_names if n in exec_names]
        if stall_candidates or exec_candidates:
            logger.info(
                "PC-sampling candidates: stall=%d, exec=%d",
                len(stall_candidates), len(exec_candidates)
            )
        else:
            # Log first few metrics for debugging
            logger.info("No PC-sampling metrics found. Sample metrics: %s", list(metric_names)[:10])

        for name in metric_names:
            # Check stall prefixes
            is_stall = any(name.startswith(p) for p in stall_prefixes)
            # Check execution metrics
            is_exec = name in exec_names

            if not (is_stall or is_exec):
                continue
            try:
                m = action.metric_by_name(name)
                if m is None or not m.has_value() or m.num_instances() <= 1:
                    # Log why we skip
                    if m is not None and m.has_value():
                        logger.debug(f"Skipping {name}: num_instances={m.num_instances()} <= 1")
                    continue
                corr_ids = m.correlation_ids()
                values: list[InstancedMetricValue] = []
                for i in range(corr_ids.num_instances()):
                    pc = corr_ids.as_uint64(i)
                    val = m.as_double(i)
                    # Source info lookup deferred to M2 SourceCorrelator
                    values.append(InstancedMetricValue(pc=pc, value=val))
                instanced[name] = values
                logger.info(f"Extracted instanced metric: {name} ({len(values)} PCs)")
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.warning("Failed to extract instanced metric %s: %s", name, e)
                continue

        return instanced

    # -- source files -------------------------------------------------

    def _extract_source_files(self, action: Any) -> dict[str, str]:
        """Extract embedded source file paths from the report.

        Returns a mapping of ``{file_path: content}``.  Content is left
        empty in M1; it will be populated on demand in M2 by
        :class:`SourceCorrelator`.
        """
        files: dict[str, str] = {}
        try:
            for src in action.source_files():
                files[str(src)] = ""  # Content populated on demand in M2
        except (AttributeError, RuntimeError):
            pass
        return files

    # -- rule results -------------------------------------------------

    def _extract_rule_results(self, action: Any) -> list[RuleResult]:
        """Extract NCU built-in rule analysis results."""
        results: list[RuleResult] = []
        try:
            for rule_dict in action.rule_results_as_dicts():
                msg = rule_dict.get("rule_message", {})
                results.append(
                    RuleResult(
                        rule_name=rule_dict.get(
                            "rule_identifier",
                            rule_dict.get("name", "unknown"),
                        ),
                        severity=self._map_rule_severity(msg.get("type")),
                        message=msg.get("message", ""),
                    )
                )
        except (AttributeError, RuntimeError):
            pass
        return results

    # ------------------------------------------------------------------
    # ActionHandle protocol implementation (for SourceCorrelator)
    # ------------------------------------------------------------------

    def source_info(self, pc: int, kernel_name: str | None = None) -> SourceInfo | None:
        """Map PC -> source file + line number.

        Args:
            pc: Program counter address.
            kernel_name: Optional kernel name to disambiguate. If not provided,
                uses the first available action.

        Returns:
            SourceInfo with file_name and line, or None if no debug info.
        """
        action = self._get_action(kernel_name)
        if action is None:
            return None
        try:
            src = action.source_info(pc)
            if src is not None:
                # NCU Python SWIG bindings return ISourceInfo object
                # with .file_name() and .line() methods (callable).
                # Older versions may use attributes (.src_file/.src_line)
                # or tuples/lists.
                if (
                    hasattr(src, "file_name")
                    and callable(getattr(src, "file_name", None))
                    and hasattr(src, "line")
                    and callable(getattr(src, "line", None))
                ):
                    try:
                        fn = src.file_name()
                        ln = src.line()
                        if isinstance(fn, str) and isinstance(ln, (int, str)):
                            return SourceInfo(
                                file_name=fn,
                                line=int(ln),
                            )
                    except (TypeError, AttributeError):
                        pass
                if hasattr(src, "src_file") and hasattr(src, "src_line"):
                    return SourceInfo(
                        file_name=str(src.src_file),
                        line=int(src.src_line),
                    )
                if isinstance(src, (tuple, list)) and len(src) >= 2:
                    return SourceInfo(
                        file_name=str(src[0]),
                        line=int(src[1]),
                    )
                # Fallback: try string parsing "filepath:line"
                s = str(src)
                colon = s.rfind(":")
                if colon > 0:
                    try:
                        return SourceInfo(
                            file_name=s[:colon],
                            line=int(s[colon + 1:]),
                        )
                    except ValueError:
                        pass
                logger.debug(
                    "Unable to parse source_info for pc=0x%x: type=%s",
                    pc, type(src),
                )
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug("source_info(pc=0x%x) failed: %s", pc, e)
        return None

    def sass_by_pc(self, pc: int, kernel_name: str | None = None) -> str | None:
        """Map PC -> SASS disassembly text.

        Args:
            pc: Program counter address.
            kernel_name: Optional kernel name to disambiguate. If not provided,
                uses the first available action.

        Returns:
            SASS instruction text, or None if unavailable.
        """
        action = self._get_action(kernel_name)
        if action is None:
            return None
        try:
            sass = action.sass_by_pc(pc)
            if sass is not None:
                return str(sass)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return None

    def ptx_by_pc(self, pc: int, kernel_name: str | None = None) -> str | None:
        """Map PC -> PTX intermediate representation text.

        Args:
            pc: Program counter address.
            kernel_name: Optional kernel name to disambiguate. If not provided,
                uses the first available action.

        Returns:
            PTX instruction text, or None if unavailable.
        """
        action = self._get_action(kernel_name)
        if action is None:
            return None
        try:
            ptx = action.ptx_by_pc(pc)
            if ptx is not None:
                return str(ptx)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return None

    def _get_action(self, kernel_name: str | None = None) -> Any | None:
        """Get action handle by kernel name, or first available if not specified."""
        if kernel_name is not None and kernel_name in self._actions:
            return self._actions[kernel_name]
        # Fallback to first available action
        if self._actions:
            return next(iter(self._actions.values()))
        return None

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _map_rule_severity(msg_type: Any) -> str:
        """Map NvRules MsgType enum value to a severity string.

        NvRules defines: OK=0, LOW=1, MED=2, HIGH=3.
        """
        type_map = {0: "OK", 1: "LOW", 2: "MED", 3: "HIGH"}
        if isinstance(msg_type, int):
            return type_map.get(msg_type, "OK")
        return str(msg_type) if msg_type else "OK"

    @staticmethod
    def _infer_unit(metric_name: str) -> str:
        """Infer the unit from NCU metric name suffix conventions.

        NCU metric names follow predictable patterns that encode the unit
        (e.g. ``dram__bytes.sum``, ``sm__throughput.avg.pct_of_peak_sustained``).
        """
        if "pct" in metric_name:
            return "%"
        if "time_duration" in metric_name or "time_active" in metric_name:
            return "ns"
        if metric_name.endswith(".sum") or metric_name.endswith(".avg"):
            if "bytes" in metric_name:
                return "byte"
            if "sectors" in metric_name:
                return "sector"
            if "requests" in metric_name:
                return "request"
            if "cycles" in metric_name or "cycle" in metric_name:
                return "cycle"
        return ""
