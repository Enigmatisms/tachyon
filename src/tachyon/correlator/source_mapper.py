"""NCU Source Mapper — pre-built bidirectional source <-> SASS mapping table.

Provides three capabilities:
1. Source <-> SASS bidirectional mapping (file:line <-> PC <-> SASS)
2. Mapped source file paths
3. Categorized performance hotspot report with SASS instruction mix

Architecture note: This module directly uses NCU SWIG bindings
(`ncu_report.load_report`), which requires NCU to be installed.
It is complementary to ``SourceCorrelator`` (which works on pre-extracted
instanced metrics from ``NcuReportReader``). The mapper pre-builds the full
mapping table at initialization time for O(1) lookups.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EMPTY_LIST: list = []


class NCUMappingSystem:
    """Bi-directional source <-> SASS mapping with bottleneck classification.

    Provides semantic analysis by grouping low-level NCU metrics into
    logical categories and classifying SASS instructions by type.
    """

    # Semantic mapping: Grouping NCU stall metrics into human-readable categories
    CATEGORY_MAP: dict[str, list[str]] = {
        "Memory (DRAM/L2/L1)": [
            "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count",
            "smsp__pcsamp_warp_stall_reason_l1_tex_throttle_sample_count",
            "smsp__pcsamp_warp_stall_reason_l2_throttle_sample_count",
            "smsp__pcsamp_warp_stall_reason_texture_sample_count",
            "smsp__pcsamp_warp_stall_reason_const_read_sample_count",
        ],
        "Async Copy / TMA": [
            "smsp__pcsamp_warp_stall_reason_async_copy_sample_count",
            "smsp__pcsamp_warp_stall_reason_tma_sample_count",
        ],
        "Compute (ALU/Tensor)": [
            "smsp__pcsamp_warp_stall_reason_math_pipe_sample_count",
            "smsp__pcsamp_warp_stall_reason_tensor_pipe_sample_count",
            "smsp__pcsamp_warp_stall_reason_bit_convert_pipe_sample_count",
        ],
        "Sync / Barrier": [
            "smsp__pcsamp_warp_stall_reason_sync_sample_count",
            "smsp__pcsamp_warp_stall_reason_barrier_sample_count",
            "smsp__pcsamp_warp_stall_reason_mbarrier_sample_count",
        ],
        "Instruction / Dependency": [
            "smsp__pcsamp_warp_stall_reason_dependency_sample_count",
            "smsp__pcsamp_warp_stall_reason_wait_sample_count",
            "smsp__pcsamp_warp_stall_reason_instruction_fetch_sample_count",
        ],
        "Scheduling / Resource": [
            "smsp__pcsamp_warp_stall_reason_not_selected_sample_count",
            "smsp__pcsamp_warp_stall_reason_pipe_busy_sample_count",
            "smsp__pcsamp_warp_stall_reason_throttle_sample_count",
            "smsp__pcsamp_warp_stall_reason_drain_sample_count",
        ],
    }

    # Legacy metric names (older NCU versions use these instead of CATEGORY_MAP)
    CATEGORY_MAP_LEGACY: dict[str, list[str]] = {
        "Memory (DRAM/L2/L1)": [
            "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
            "smsp__pcsamp_warps_issue_stalled_lg_throttle",
            "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
            "smsp__pcsamp_warps_issue_stalled_mio_throttle",
            "smsp__pcsamp_warps_issue_stalled_tex_throttle",
        ],
        "Compute (ALU/Tensor)": [
            "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
        ],
        "Sync / Barrier": [
            "smsp__pcsamp_warps_issue_stalled_barrier",
            "smsp__pcsamp_warps_issue_stalled_membar",
        ],
        "Instruction / Dependency": [
            "smsp__pcsamp_warps_issue_stalled_wait",
            "smsp__pcsamp_warps_issue_stalled_branch_resolving",
            "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
            "smsp__pcsamp_warps_issue_stalled_no_instructions",
        ],
        "Scheduling / Resource": [
            "smsp__pcsamp_warps_issue_stalled_not_selected",
            "smsp__pcsamp_warps_issue_stalled_sleeping",
            "smsp__pcsamp_warps_issue_stalled_drain",
            "smsp__pcsamp_warps_issue_stalled_imc_miss",
            "smsp__pcsamp_warps_issue_stalled_misc",
            "smsp__pcsamp_warps_issue_stalled_selected",
        ],
    }

    # SASS instruction prefix -> category (for source-line SASS mix analysis)
    _SASS_PREFIXES: list[tuple[str, tuple[str, ...]]] = [
        ("Memory Load", (
            "LDG", "LDL", "LDS", "LDC", "LD", "LDE",
            "RED.E", "ATOM.G", "ATOM.E",
        )),
        ("Memory Store", (
            "STG", "STL", "STS", "ST", "STE",
            "RED", "ATOM",
        )),
        ("Async Copy / TMA", (
            "CP",
            "LDGSTS",
        )),
        ("Float Compute", (
            "FFMA", "FMUL", "FADD", "FSUB", "FDIV",
            "HADD2", "HMUL2", "HFMA2",
            "FNEG", "FABS", "FSET", "FMNMX", "FCMP", "FSEL",
            "MUFU", "RRO",
        )),
        ("Int Compute", (
            "IMAD", "IMUL", "IADD", "IADD3", "ISCADD",
            "ISUB", "IABS", "IAND", "IOR", "IXOR",
            "SHR", "SHL", "SHF", "LOP3", "BFE", "BFI",
            "PRMT", "POPC", "FLO",
        )),
        ("Tensor/Matrix", (
            "HMMA", "IMMA", "DMMA", "BMMA",
            "WMMA", "WGMMA",
        )),
        ("Control Flow", (
            "BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT",
            "SSY", "BSSY", "BSYNC", "BREAK", "CONT",
        )),
        ("Sync", (
            "BAR", "MEMBAR", "DEPBAR", "WARPSYNC", "MBAR",
        )),
        ("Predicate", (
            "PSETP", "P2R", "R2P", "CSET", "CSETP",
            "ISETP", "FSETP", "PLOP3",
        )),
        ("Register/Misc", (
            "MOV", "MOVZ", "S2R", "CS2R", "R2S",
            "LEA", "SEL", "NOP", "BPT",
        )),
    ]

    __slots__ = [
        "ncu_report", "report", "_s2as_flat", "_as2s_flat",
        "_kernels", "_get_sass_fast", "_get_src_fast", "_processed_configs",
        "_total_samples_per_kernel", "_total_exec_per_kernel",
        "_include_tree", "_fwd_include", "_include_chains",
        "_active_category_map",
    ]

    def __init__(self, report_path: str | Path) -> None:
        """Load .ncu-rep and build the full source <-> SASS mapping table.

        Args:
            report_path: Path to .ncu-rep file.

        Raises:
            ImportError: If ncu_report module is not available.
            RuntimeError: If the report cannot be loaded.
        """
        ncu = self._import_ncu_report()
        self.ncu_report = ncu
        self.report = ncu.load_report(str(report_path))

        self._s2as_flat: dict[tuple[str, str, int], list[dict]] = {}
        self._as2s_flat: dict[tuple[str, int], dict] = {}
        self._kernels: list[str] = []
        self._processed_configs: set[tuple[str, tuple]] = set()
        self._total_samples_per_kernel: dict[str, int] = defaultdict(int)
        self._total_exec_per_kernel: dict[str, int] = defaultdict(int)

        self._get_sass_fast = self._s2as_flat.get
        self._get_src_fast = self._as2s_flat.get

        self._include_tree: dict[str, list[str]] = {}
        self._fwd_include: dict[str, list[str]] = {}
        self._include_chains: dict[str, list[str]] = {}

        self._index_all_kernels()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_ncu_report() -> Any:
        """Import ncu_report module (already on sys.path from reader init).

        Raises:
            ImportError: If ncu_report is not available.
        """
        try:
            import ncu_report
            return ncu_report
        except ImportError:
            # Fallback: try discovering and loading the module
            from tachyon.reader.ncu_reader import _load_ncu_module
            return _load_ncu_module()

    # ------------------------------------------------------------------
    # Mapping table builder
    # ------------------------------------------------------------------

    def _build_action_maps(
        self,
        action: Any,
        kernel_name: str,
        temp_s2as: defaultdict,
    ) -> None:
        """Build per-PC metric entries for one action (kernel launch config)."""
        base_metric = (
            action.metric_by_name("inst_executed")
            or action.metric_by_name("smsp__inst_executed")
        )
        if not base_metric:
            return

        # Detect metric name format: try new-style first, fall back to old-style
        if not hasattr(self, '_active_category_map'):
            new_sample = (
                list(self.CATEGORY_MAP.values())[0][0]
                if self.CATEGORY_MAP else ""
            )
            if new_sample and action.metric_by_name(new_sample):
                self._active_category_map = self.CATEGORY_MAP
            else:
                self._active_category_map = self.CATEGORY_MAP_LEGACY

        # Collect all metrics defined in active category map plus sample count
        all_metric_names = [
            m for sub in self._active_category_map.values() for m in sub
        ]
        all_metric_names.append("smsp__pcsamp_sample_count")
        m_objs: dict[str, Any] = {
            n: action.metric_by_name(n) for n in all_metric_names
        }
        # Track base metric for per-PC storage
        m_objs["inst_executed"] = base_metric

        pcs = base_metric.correlation_ids()
        for i in range(pcs.num_instances()):
            pc = pcs.as_uint64(i)

            # Accumulate totals (using instance index, NOT pc)
            sample_obj = m_objs.get("smsp__pcsamp_sample_count")
            self._total_samples_per_kernel[kernel_name] += (
                sample_obj.as_uint64(i) if sample_obj else 0
            )
            self._total_exec_per_kernel[kernel_name] += base_metric.as_uint64(i)

            src_info = action.source_info(pc)
            if not src_info:
                continue
            info_obj = (
                src_info[0] if isinstance(src_info, (list, tuple)) else src_info
            )

            try:
                f_path = sys.intern(info_obj.file_name())
                l_num = info_obj.line()
                pc_metrics: dict[str, int] = {}
                for n, obj in m_objs.items():
                    try:
                        pc_metrics[n] = obj.as_uint64(i) if obj else 0
                    except (RuntimeError, IndexError):
                        pc_metrics[n] = 0

                entry = {
                    "pc": hex(pc),
                    "sass": action.sass_by_pc(pc).strip(),
                    "file": f_path,
                    "line": l_num,
                    "metrics": pc_metrics,
                }
                temp_s2as[(kernel_name, f_path, l_num)].append(entry)
                self._as2s_flat[(kernel_name, pc)] = entry
            except Exception:
                continue

    # ------------------------------------------------------------------
    # SASS instruction classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_sass(sass: str) -> str:
        """Return the category name for a single SASS instruction."""
        mnemonic = sass.split()[0].split(".")[0].split("@")[0].rstrip(":")
        for cat_name, prefixes in NCUMappingSystem._SASS_PREFIXES:
            for pfx in prefixes:
                if mnemonic.startswith(pfx):
                    return cat_name
        return "Other"

    @staticmethod
    def classify_line_sass(inst_list: list[dict]) -> dict[str, int]:
        """Aggregate SASS categories for all instructions on a source line.

        Returns dict like ``{"Memory Load": 3, "Register/Misc": 2, ...}``.
        Only includes categories with count > 0.
        """
        counts: dict[str, int] = defaultdict(int)
        for entry in inst_list:
            cat = NCUMappingSystem.classify_sass(entry["sass"])
            counts[cat] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # #include tree (file-level context for agent)
    # ------------------------------------------------------------------

    _INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

    @staticmethod
    def _short_path(fpath: str) -> str:
        """Extract the last 2 path segments for display/matching."""
        norm = fpath.replace("\\", "/")
        parts = norm.split("/")
        return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    @staticmethod
    def _is_project_file(fpath: str) -> bool:
        """Check if a path is a project source file (not system header)."""
        norm = fpath.replace("\\", "/")
        first = norm.split("/")[1] if len(norm.split("/")) > 1 else ""
        return first not in ("usr", "opt", "lib", "etc", "home", "tmp")

    def build_include_tree(
        self, source_files: dict[str, str],
    ) -> dict[str, list[str]]:
        """Parse #include directives and build include maps.

        Returns reverse map: included_file -> [includers].
        Also populates self._fwd_include: includer -> [included files].

        Note: Only includes files present in source_files. Intermediate
        headers may be missing from NCU's source_files if no PC maps
        directly to them, causing chain gaps.
        """
        all_project_shorts: set[str] = set()
        for fp in source_files:
            if self._is_project_file(fp):
                all_project_shorts.add(self._short_path(fp))

        fwd: dict[str, list[str]] = defaultdict(list)
        for fpath, content in source_files.items():
            if not self._is_project_file(fpath):
                continue
            sp = self._short_path(fpath)
            for line in content.split("\n"):
                m = self._INCLUDE_RE.match(line)
                if m:
                    inc_path = m.group(1)
                    inc_parts = inc_path.replace("\\", "/").split("/")
                    for ps in all_project_shorts:
                        ps_parts = ps.split("/")
                        if (
                            len(inc_parts) <= len(ps_parts)
                            and ps_parts[-len(inc_parts) :] == inc_parts
                        ):
                            fwd[sp].append(ps)
                            break

        rev: dict[str, list[str]] = defaultdict(list)
        for includer, included_list in fwd.items():
            for inc in included_list:
                rev[inc].append(includer)

        self._fwd_include = dict(fwd)
        return dict(rev)

    def _build_include_chains(self) -> None:
        """BFS from each .cu file to trace shortest include chain.

        Populates self._include_chains: {file_short: [cu, ..., file_short]}.
        """
        cu_files = [
            sp for sp in self._fwd_include if sp.endswith(".cu")
        ]
        if not cu_files:
            return

        for cu in cu_files:
            visited: set[str] = {cu}
            queue: list[tuple[str, list[str]]] = [(cu, [cu])]
            for cur, path in queue:
                for inc in self._fwd_include.get(cur, []):
                    if inc not in visited:
                        visited.add(inc)
                        new_path = path + [inc]
                        if (
                            inc not in self._include_chains
                            or len(new_path) < len(self._include_chains[inc])
                        ):
                            self._include_chains[inc] = new_path
                        queue.append((inc, new_path))

    # ------------------------------------------------------------------
    # Stall computation (shared by SPI and stall_profile)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stalls(
        inst_list: list[dict],
        category_map: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, int], int]:
        """Compute per-category stall breakdown and total stall count.

        Single-pass over *category_map*. Returns (stall_profile, total_stalls).
        Defaults to CATEGORY_MAP (new-format) when *category_map* is None.
        """
        if category_map is None:
            category_map = NCUMappingSystem.CATEGORY_MAP
        stall_profile: dict[str, int] = {}
        total_stalls = 0
        for cat_name, metric_names in category_map.items():
            cat_val = 0
            for e in inst_list:
                for m in metric_names:
                    cat_val += e["metrics"].get(m, 0)
            if cat_val > 0:
                stall_profile[cat_name] = cat_val
                total_stalls += cat_val
        return stall_profile, total_stalls

    # ------------------------------------------------------------------
    # Focus hint generation (for agent guidance)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_focus_hint(
        spi: float,
        severity: float,
        dominant_stall: str,
        dominant_sass: str,
        include_chain: list[str] | None,
        line_exec: int,
        line_samples: int,
    ) -> str:
        """Generate one-line optimization guidance for the agent."""
        hints: list[str] = []
        if spi == -1.0:
            hints.append("pure stall point (no execution, only stalls)")
        elif spi > 10:
            hints.append("extreme stall bottleneck")
        elif spi > 3:
            hints.append("stall-bound")
        if include_chain and len(include_chain) > 2:
            hints.append("deep inlined utility")
        if severity > 5 and dominant_stall != "N/A":
            if "Memory" in dominant_stall:
                hints.append("memory-bound hotspot")
            elif "Sync" in dominant_stall or "Barrier" in dominant_stall:
                hints.append("sync overhead")
            elif "Compute" in dominant_stall:
                hints.append("compute-intensive")
        if severity > 5 and line_samples > 0 and line_exec == 0:
            hints.append("stall-only hotspot (no exec data)")
        return "; ".join(hints) if hints else ""

    # ------------------------------------------------------------------
    # Bottleneck / Hotspot Report
    # ------------------------------------------------------------------

    def get_bottleneck_report(self, top_n: int = 10) -> list[dict]:
        """Source-level performance hotspot report with categorized metrics.

        Each entry contains:
          severity       — native % of kernel total (pc_sample or exec_freq)
          spi            — stalls per instruction; -1.0 = pure stall point
          focus_hint     — one-line optimization guidance for the agent
          include_chain  — [cu, ..., file] full include path (None if unavailable)
          stall_profile  — {category: count} per-category stall breakdown
          dominant_stall — stall category with highest count
          sass_mix       — {category: count} instruction type breakdown
          dominant_sass  — instruction category with highest count
          sass_preview   — up to 3 representative SASS instructions

        Returns list of dicts sorted by severity descending.
        """
        raw_reports: list[dict] = []
        for (k_name, f_path, l_num), inst_list in self._s2as_flat.items():
            kernel_total_samples = self._total_samples_per_kernel[k_name]
            kernel_total_exec = self._total_exec_per_kernel[k_name]

            line_samples = sum(
                e["metrics"].get("smsp__pcsamp_sample_count", 0)
                for e in inst_list
            )
            line_exec = sum(
                e["metrics"].get("inst_executed", 0) for e in inst_list
            )

            # Severity: native only, no propagation
            if kernel_total_samples > 0 and line_samples > 0:
                severity = round(
                    (line_samples / kernel_total_samples) * 100, 2
                )
                severity_metric = "pc_sample"
            elif kernel_total_exec > 0 and line_exec > 0:
                severity = round((line_exec / kernel_total_exec) * 100, 2)
                severity_metric = "exec_freq"
            else:
                severity = 0.0
                severity_metric = "none"

            # Stalls: single-pass computation
            stall_profile, total_stalls = self._compute_stalls(
                inst_list,
                getattr(self, '_active_category_map', self.CATEGORY_MAP),
            )

            # SPI: stalls per instruction
            if line_exec > 0:
                spi = round(total_stalls / line_exec, 2)
            elif total_stalls > 0:
                spi = -1.0  # pure stall point
            else:
                spi = 0.0

            dominant_stall = (
                max(stall_profile, key=stall_profile.get)
                if stall_profile else "N/A"
            )

            # SASS instruction mix
            sass_mix = self.classify_line_sass(inst_list)
            dominant_sass = (
                max(sass_mix, key=sass_mix.get) if sass_mix else "N/A"
            )

            # Representative SASS instructions (top 3 unique)
            seen_sass: set[str] = set()
            sass_preview: list[str] = []
            for e in inst_list:
                m = e["sass"].strip()
                if m not in seen_sass:
                    seen_sass.add(m)
                    sass_preview.append(m)
                if len(sass_preview) >= 3:
                    break

            # Include chain
            sp = self._short_path(f_path)
            include_chain = self._include_chains.get(sp)

            # Focus hint
            focus_hint = self._compute_focus_hint(
                spi, severity, dominant_stall, dominant_sass,
                include_chain, line_exec, line_samples,
            )

            raw_reports.append({
                "file": f_path,
                "line": l_num,
                "kernel": k_name,
                "severity": severity,
                "severity_metric": severity_metric,
                "spi": spi,
                "focus_hint": focus_hint,
                "num_insts": len(inst_list),
                "line_exec": line_exec,
                "line_samples": line_samples,
                "dominant_stall": dominant_stall,
                "dominant_sass": dominant_sass,
                "stall_profile": stall_profile,
                "stall_total": total_stalls,
                "sass_mix": sass_mix,
                "sass_preview": sass_preview,
                "include_chain": include_chain,
            })

        raw_reports.sort(
            key=lambda x: (x["severity"], x["num_insts"]), reverse=True
        )
        return raw_reports[:top_n]

    def format_hotspot_report(self, top_n: int = 10) -> str:
        """Human-readable categorized source-level hotspot report.

        Groups results by kernel, then ranks source lines by severity.
        Each entry shows: severity% | SPI | file:line, chain, hint, insts, stalls.
        """
        report = self.get_bottleneck_report(top_n=top_n)
        if not report:
            return "No source-mapped instructions found in report."

        lines: list[str] = []
        current_kernel: str | None = None

        for rank, entry in enumerate(report, 1):
            k_name = entry["kernel"]

            if k_name != current_kernel:
                current_kernel = k_name
                total_s = self._total_samples_per_kernel.get(k_name, 0)
                total_e = self._total_exec_per_kernel.get(k_name, 0)
                sep = "=" * 100
                lines.append("")
                lines.append(sep)
                lines.append(
                    f"  PERFORMANCE HOTSPOT REPORT - {k_name}"
                )
                lines.append(
                    f"  PC-sampling: {total_s:,} samples  |  "
                    f"inst_executed: {total_e:,}  |  "
                    f"{'Ranked by sampling severity' if total_s > 0 else 'Ranked by execution frequency'}"
                )
                lines.append(sep)

            f_full = entry["file"]
            f_short = "/".join(f_full.replace("\\", "/").split("/")[-2:])
            loc = f"{f_short}:{entry['line']}"

            sev = entry["severity"]
            metric_tag = entry["severity_metric"]
            if metric_tag == "pc_sample":
                sev_label = f"{sev}%"
            elif metric_tag == "exec_freq":
                sev_label = f"{sev}% (exec)"
            else:
                sev_label = "-"

            # SPI label
            spi = entry["spi"]
            if spi == -1.0:
                spi_str = "SPI INF"
            elif spi > 0:
                spi_str = f"SPI {spi}"
            else:
                spi_str = ""

            lines.append("")
            lines.append(
                f"  #{rank:<3} {sev_label:<12} {spi_str:<10} | {loc}"
            )

            # Include chain
            chain = entry.get("include_chain")
            if chain and len(chain) > 1:
                lines.append(f"       chain: {' -> '.join(chain)}")

            # Focus hint
            hint = entry.get("focus_hint", "")
            if hint:
                lines.append(f"       hint:  {hint}")

            mix_parts = [
                f"{v} {k.split('/')[0].split('(')[0].lower()}"
                for k, v in sorted(
                    entry["sass_mix"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            sass_summary = ", ".join(mix_parts) if mix_parts else "N/A"

            if entry["stall_profile"]:
                total_st = entry["stall_total"]
                stall_parts = [
                    f"{k.split('(')[0].strip()}: {v / total_st * 100:.0f}%"
                    for k, v in sorted(
                        entry["stall_profile"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]
                stall_summary = "Stalls: " + ", ".join(stall_parts)
            else:
                stall_summary = "Stalls: none (no PC-sampling data)"

            lines.append(
                f"       {entry['num_insts']} insts | {sass_summary}"
            )
            lines.append(
                f"       {stall_summary} | dominant: {entry['dominant_stall']}"
            )
            for sp in entry["sass_preview"]:
                lines.append(f"         {sp}")

        lines.append("")
        lines.append("=" * 100)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Launch config helpers
    # ------------------------------------------------------------------

    def _get_launch_config(self, action: Any) -> tuple:
        """Extracts grid and block dimensions for kernel fingerprinting."""
        dims: list[int] = []
        for dim in [
            "grid_dim_x", "grid_dim_y", "grid_dim_z",
            "block_dim_x", "block_dim_y", "block_dim_z",
        ]:
            m = (
                action.metric_by_name(f"launch__{dim}")
                or action.metric_by_name(dim)
            )
            dims.append(int(m.as_uint64()) if m else 0)
        return tuple(dims)

    def _index_all_kernels(self) -> None:
        """Process report ranges and build de-duplicated mapping indexes."""
        temp_s2as: defaultdict = defaultdict(list)

        for r_idx in range(self.report.num_ranges()):
            report_range = self.report.range_by_idx(r_idx)
            for a_idx in range(report_range.num_actions()):
                action = report_range.action_by_idx(a_idx)
                k_name = sys.intern(action.name())

                config = self._get_launch_config(action)
                config_id = (k_name, config)

                if config_id in self._processed_configs:
                    continue

                self._processed_configs.add(config_id)
                if k_name not in self._kernels:
                    self._kernels.append(k_name)

                # Build include tree once (first action with source files)
                if not self._include_tree:
                    try:
                        sf = action.source_files()
                        if sf:
                            self._include_tree = self.build_include_tree(sf)
                            self._build_include_chains()
                    except Exception:
                        pass

                self._build_action_maps(action, k_name, temp_s2as)

        for tuple_key, inst_list in temp_s2as.items():
            inst_list.sort(key=lambda x: int(x["pc"], 16))
            self._s2as_flat[tuple_key] = inst_list

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_mapped_sources(self) -> list[str]:
        """Return sorted list of source file paths with mapped instructions."""
        return sorted({key[1] for key in self._s2as_flat.keys()})

    def get_sass_by_line(
        self, kernel_name: str, file_path: str, line_num: int,
    ) -> list[dict]:
        """Get all SASS instructions for a source line."""
        return self._get_sass_fast(
            (kernel_name, file_path, line_num), _EMPTY_LIST,
        )

    def get_source_by_pc(self, kernel_name: str, pc: int | str) -> dict | None:
        """Map a PC address to source location and SASS."""
        addr = int(pc, 16) if isinstance(pc, str) else pc
        return self._get_src_fast((kernel_name, addr))

    def get_kernels(self) -> list[str]:
        """Return list of kernel names in the report."""
        return self._kernels

    def get_include_tree(self) -> dict[str, list[str]]:
        """Return the include tree: {included_file_short: [includer_short, ...]}."""
        return self._include_tree

    def get_include_chain(self, fpath: str) -> list[str] | None:
        """Return the full include chain from kernel .cu to this file, or None."""
        return self._include_chains.get(self._short_path(fpath))
