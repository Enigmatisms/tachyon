"""Unit tests for NCUMappingSystem — SASS classification and hotspot report.

The mapper requires NCU SWIG bindings, so we only test the pure-logic parts:
- SASS instruction classification
- Hotspot report formatting from pre-built data
- SPI, include chain, focus_hint computation
- Query API
"""
from unittest.mock import MagicMock

import pytest

from tachyon.correlator.source_mapper import NCUMappingSystem


def _make_empty_mapper() -> NCUMappingSystem:
    """Create a mapper with all required slots initialized (no NCU needed)."""
    mapper = NCUMappingSystem.__new__(NCUMappingSystem)
    mapper._s2as_flat = {}
    mapper._as2s_flat = {}
    mapper._kernels = []
    mapper._total_samples_per_kernel = {}
    mapper._total_exec_per_kernel = {}
    mapper._include_tree = {}
    mapper._fwd_include = {}
    mapper._include_chains = {}
    mapper._get_sass_fast = mapper._s2as_flat.get
    mapper._get_src_fast = mapper._as2s_flat.get
    return mapper


class TestClassifySass:
    """Test SASS instruction classification by mnemonic prefix."""

    def test_memory_load(self):
        assert NCUMappingSystem.classify_sass("LDG.E R0, [R2]") == "Memory Load"
        assert NCUMappingSystem.classify_sass("LDL.LU R4, [R1+0x10]") == "Memory Load"
        assert NCUMappingSystem.classify_sass("LD.E.128 R0, [R4]") == "Memory Load"
        # RED.E is classified as Memory Store (reduction is read-modify-write)
        assert NCUMappingSystem.classify_sass("RED.E R0, [R2], R4") == "Memory Store"

    def test_memory_store(self):
        assert NCUMappingSystem.classify_sass("STG.E R0, [R2]") == "Memory Store"
        assert NCUMappingSystem.classify_sass("STL.128 [R1+0x20], R4") == "Memory Store"

    def test_float_compute(self):
        assert NCUMappingSystem.classify_sass("FFMA R2, R0, R1, R3") == "Float Compute"
        assert NCUMappingSystem.classify_sass("FMUL.RZ R4, R0, R2") == "Float Compute"
        assert NCUMappingSystem.classify_sass("HFMA2.RN.F16x2") == "Float Compute"
        assert NCUMappingSystem.classify_sass("MUFU.RCP R2, R0") == "Float Compute"

    def test_int_compute(self):
        assert NCUMappingSystem.classify_sass("IMAD R0, R1, R2, R3") == "Int Compute"
        assert NCUMappingSystem.classify_sass("IADD3 R0, R1, R2, RZ") == "Int Compute"
        assert NCUMappingSystem.classify_sass("LOP3.LUT R0, R1, R2, R3, 0xff") == "Int Compute"
        assert NCUMappingSystem.classify_sass("SHR.U32 R0, R1, 0x5") == "Int Compute"

    def test_tensor_matrix(self):
        assert NCUMappingSystem.classify_sass("HMMA.16816.F32.TF32") == "Tensor/Matrix"
        assert NCUMappingSystem.classify_sass("WGMMA.MMA_ASYNC") == "Tensor/Matrix"

    def test_async_copy(self):
        assert NCUMappingSystem.classify_sass("CP.ASYNC.CA") == "Async Copy / TMA"
        # LDGSTS starts with LDG, so it matches Memory Load first.
        # This is a known limitation of prefix-based classification.
        assert NCUMappingSystem.classify_sass("LDGSTS.E.128") == "Memory Load"

    def test_control_flow(self):
        assert NCUMappingSystem.classify_sass("BRA `label`") == "Control Flow"
        assert NCUMappingSystem.classify_sass("EXIT") == "Control Flow"
        assert NCUMappingSystem.classify_sass("BSSY B0, `target`") == "Control Flow"
        assert NCUMappingSystem.classify_sass("CALL `func`") == "Control Flow"

    def test_sync(self):
        assert NCUMappingSystem.classify_sass("BAR.SYNC 0") == "Sync"
        assert NCUMappingSystem.classify_sass("MEMBAR.CTA") == "Sync"

    def test_predicate(self):
        assert NCUMappingSystem.classify_sass("PSETP.AND PT1, PT, PT, !PT") == "Predicate"
        assert NCUMappingSystem.classify_sass("ISETP.GE.AND P0, PT, R0, R1, PT") == "Predicate"

    def test_register_misc(self):
        assert NCUMappingSystem.classify_sass("MOV R0, R1") == "Register/Misc"
        assert NCUMappingSystem.classify_sass("S2R R0, SR_TID.X") == "Register/Misc"
        assert NCUMappingSystem.classify_sass("NOP") == "Register/Misc"

    def test_unknown(self):
        assert NCUMappingSystem.classify_sass("ULDC.64 UR36, c[0x0][0x118]") == "Other"
        assert NCUMappingSystem.classify_sass("R2UR UR38, R1") == "Other"
        assert NCUMappingSystem.classify_sass("UIADD3 UR38, UR38, 0x20") == "Other"


class TestClassifyLineSass:
    """Test SASS classification aggregation for a list of instructions."""

    def test_aggregate_categories(self):
        inst_list = [
            {"sass": "LDG.E R0, [R2]"},
            {"sass": "LDG.E R1, [R2+0x40]"},
            {"sass": "FFMA R4, R0, R1, R3"},
            {"sass": "MOV R5, R6"},
        ]
        result = NCUMappingSystem.classify_line_sass(inst_list)
        assert result["Memory Load"] == 2
        assert result["Float Compute"] == 1
        assert result["Register/Misc"] == 1
        # "Other" and zero-count categories should not appear
        assert "Other" not in result
        assert "Memory Store" not in result

    def test_empty_list(self):
        assert NCUMappingSystem.classify_line_sass([]) == {}


class TestBottleneckReport:
    """Test bottleneck report generation from pre-built mapping data."""

    @pytest.fixture()
    def mapper_with_data(self) -> NCUMappingSystem:
        """Create a mapper with pre-populated mapping data (no NCU needed)."""
        mapper = _make_empty_mapper()
        mapper._s2as_flat = {
            ("test_kernel", "/path/to/gemm.cu", 100): [
                {
                    "pc": "0x1000",
                    "sass": "LDG.E R0, [R2]",
                    "file": "/path/to/gemm.cu",
                    "line": 100,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 5000,
                        "inst_executed": 100000,
                        "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 3000,
                        "smsp__pcsamp_warp_stall_reason_math_pipe_sample_count": 1000,
                    },
                },
            ],
            ("test_kernel", "/path/to/gemm.cu", 120): [
                {
                    "pc": "0x2000",
                    "sass": "FFMA R4, R0, R1, R3",
                    "file": "/path/to/gemm.cu",
                    "line": 120,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 2000,
                        "inst_executed": 50000,
                        "smsp__pcsamp_warp_stall_reason_math_pipe_sample_count": 1500,
                        "smsp__pcsamp_warp_stall_reason_sync_sample_count": 500,
                    },
                },
            ],
        }
        mapper._as2s_flat = {}
        mapper._kernels = ["test_kernel"]
        mapper._total_samples_per_kernel = {"test_kernel": 10000}
        mapper._total_exec_per_kernel = {"test_kernel": 200000}
        return mapper

    def test_report_returns_entries_sorted_by_severity(self, mapper_with_data):
        report = mapper_with_data.get_bottleneck_report(top_n=10)
        assert len(report) == 2
        # Line 100: 5000/10000 = 50%
        # Line 120: 2000/10000 = 20%
        assert report[0]["line"] == 100
        assert report[0]["severity"] == 50.0
        assert report[1]["line"] == 120
        assert report[1]["severity"] == 20.0

    def test_report_has_categorized_stall_profile(self, mapper_with_data):
        report = mapper_with_data.get_bottleneck_report(top_n=10)
        entry = report[0]  # Line 100
        # Memory stall: 3000, Compute stall: 1000
        assert "Memory (DRAM/L2/L1)" in entry["stall_profile"]
        assert entry["stall_profile"]["Memory (DRAM/L2/L1)"] == 3000
        assert "Compute (ALU/Tensor)" in entry["stall_profile"]
        assert entry["dominant_stall"] == "Memory (DRAM/L2/L1)"

    def test_report_has_sass_mix(self, mapper_with_data):
        report = mapper_with_data.get_bottleneck_report(top_n=10)
        # Line 100 has LDG → Memory Load
        entry = report[0]
        assert "sass_mix" in entry
        assert "Memory Load" in entry["sass_mix"]
        assert entry["dominant_sass"] == "Memory Load"

    def test_report_top_n_limit(self, mapper_with_data):
        report = mapper_with_data.get_bottleneck_report(top_n=1)
        assert len(report) == 1
        assert report[0]["line"] == 100

    def test_report_empty_data(self):
        mapper = _make_empty_mapper()
        report = mapper.get_bottleneck_report()
        assert report == []

    def test_format_hotspot_report(self, mapper_with_data):
        text = mapper_with_data.format_hotspot_report(top_n=5)
        assert "test_kernel" in text
        assert "gemm.cu" in text
        assert "50.0%" in text
        assert "memory load" in text  # formatted output uses lowercase

    def test_format_empty_report(self):
        mapper = _make_empty_mapper()
        text = mapper.format_hotspot_report()
        assert "No source-mapped instructions" in text


class TestMapperQueryAPI:
    """Test public query API."""

    def test_get_mapped_sources(self):
        mapper = _make_empty_mapper()
        mapper._s2as_flat = {
            ("k1", "/a/b.cu", 10): [],
            ("k1", "/a/c.cu", 20): [],
            ("k2", "/a/b.cu", 10): [],
        }
        sources = mapper.get_mapped_sources()
        assert sources == ["/a/b.cu", "/a/c.cu"]

    def test_get_sass_by_line(self):
        mapper = _make_empty_mapper()
        insts = [
            {"pc": "0x1000", "sass": "LDG.E R0", "file": "a.cu", "line": 10, "metrics": {}},
        ]
        mapper._s2as_flat[("k1", "a.cu", 10)] = insts
        result = mapper.get_sass_by_line("k1", "a.cu", 10)
        assert len(result) == 1
        assert result[0]["sass"] == "LDG.E R0"

    def test_get_sass_by_line_empty(self):
        mapper = _make_empty_mapper()
        result = mapper.get_sass_by_line("k1", "missing.cu", 99)
        assert result == []

    def test_get_source_by_pc(self):
        mapper = _make_empty_mapper()
        entry = {"pc": "0x1000", "sass": "LDG", "file": "a.cu", "line": 10, "metrics": {}}
        mapper._as2s_flat[("k1", 0x1000)] = entry
        result = mapper.get_source_by_pc("k1", 0x1000)
        assert result["file"] == "a.cu"

    def test_get_source_by_pc_hex_string(self):
        mapper = _make_empty_mapper()
        entry = {"pc": "0x2000", "sass": "FFMA", "file": "b.cu", "line": 20, "metrics": {}}
        mapper._as2s_flat[("k1", 0x2000)] = entry
        result = mapper.get_source_by_pc("k1", "0x2000")
        assert result is not None
        assert result["line"] == 20

    def test_get_kernels(self):
        mapper = _make_empty_mapper()
        mapper._kernels = ["kernel_a", "kernel_b"]
        assert mapper.get_kernels() == ["kernel_a", "kernel_b"]

    def test_get_include_tree(self):
        mapper = _make_empty_mapper()
        mapper._include_tree = {"utils.cuh": ["kernel.cu", "main.cu"]}
        assert mapper.get_include_tree() == {"utils.cuh": ["kernel.cu", "main.cu"]}

    def test_get_include_chain(self):
        mapper = _make_empty_mapper()
        mapper._include_chains = {"path/utils.cuh": ["kernel.cu", "helper.cuh", "utils.cuh"]}
        assert mapper.get_include_chain("/some/path/utils.cuh") == [
            "kernel.cu", "helper.cuh", "utils.cuh",
        ]

    def test_get_include_chain_not_found(self):
        mapper = _make_empty_mapper()
        assert mapper.get_include_chain("/some/missing.cuh") is None


class TestSPI:
    """Test SPI (Stalls Per Instruction) computation."""

    @pytest.fixture()
    def mapper_with_stall_data(self) -> NCUMappingSystem:
        """Create a mapper with entries covering all SPI cases."""
        mapper = _make_empty_mapper()
        mapper._s2as_flat = {
            # Normal: 4000 stalls / 1000 exec → SPI = 4.0
            ("k1", "/a/b.cu", 10): [
                {
                    "pc": "0x100", "sass": "LDG.E R0, [R2]",
                    "file": "/a/b.cu", "line": 10,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 100,
                        "inst_executed": 1000,
                        "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 4000,
                    },
                },
            ],
            # Pure stall: 500 stalls, 0 exec → SPI = -1.0
            ("k1", "/a/b.cu", 20): [
                {
                    "pc": "0x200", "sass": "BAR.SYNC 0",
                    "file": "/a/b.cu", "line": 20,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 50,
                        "inst_executed": 0,
                        "smsp__pcsamp_warp_stall_reason_sync_sample_count": 500,
                    },
                },
            ],
            # No stalls: 0 stalls / 2000 exec → SPI = 0.0
            ("k1", "/a/b.cu", 30): [
                {
                    "pc": "0x300", "sass": "FFMA R0, R1, R2, R3",
                    "file": "/a/b.cu", "line": 30,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 0,
                        "inst_executed": 2000,
                    },
                },
            ],
        }
        mapper._kernels = ["k1"]
        mapper._total_samples_per_kernel = {"k1": 150}
        mapper._total_exec_per_kernel = {"k1": 3000}
        return mapper

    def test_spi_normal(self, mapper_with_stall_data):
        report = mapper_with_stall_data.get_bottleneck_report(top_n=10)
        entry_10 = next(e for e in report if e["line"] == 10)
        assert entry_10["spi"] == 4.0
        assert entry_10["stall_total"] == 4000
        assert entry_10["line_exec"] == 1000

    def test_spi_pure_stall(self, mapper_with_stall_data):
        report = mapper_with_stall_data.get_bottleneck_report(top_n=10)
        entry_20 = next(e for e in report if e["line"] == 20)
        assert entry_20["spi"] == -1.0
        assert entry_20["stall_total"] == 500
        assert entry_20["line_exec"] == 0

    def test_spi_no_stalls(self, mapper_with_stall_data):
        report = mapper_with_stall_data.get_bottleneck_report(top_n=10)
        entry_30 = next(e for e in report if e["line"] == 30)
        assert entry_30["spi"] == 0.0
        assert entry_30["stall_total"] == 0

    def test_spi_in_report_dict(self, mapper_with_stall_data):
        """All report entries must have 'spi' field."""
        report = mapper_with_stall_data.get_bottleneck_report(top_n=10)
        for entry in report:
            assert "spi" in entry

    def test_format_shows_spi(self, mapper_with_stall_data):
        text = mapper_with_stall_data.format_hotspot_report(top_n=10)
        assert "SPI 4.0" in text
        assert "SPI INF" in text


class TestComputeStalls:
    """Test _compute_stalls static method."""

    def test_single_category(self):
        inst_list = [
            {"metrics": {
                "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 100,
            }},
            {"metrics": {
                "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 200,
            }},
        ]
        profile, total = NCUMappingSystem._compute_stalls(inst_list)
        assert total == 300
        assert profile["Memory (DRAM/L2/L1)"] == 300

    def test_multiple_categories(self):
        inst_list = [
            {"metrics": {
                "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 100,
                "smsp__pcsamp_warp_stall_reason_math_pipe_sample_count": 0,
            }},
        ]
        profile, total = NCUMappingSystem._compute_stalls(inst_list)
        assert total == 100
        assert len(profile) == 1

    def test_empty_inst_list(self):
        profile, total = NCUMappingSystem._compute_stalls([])
        assert total == 0
        assert profile == {}

    def test_no_stall_metrics(self):
        inst_list = [{"metrics": {"inst_executed": 1000}}]
        profile, total = NCUMappingSystem._compute_stalls(inst_list)
        assert total == 0
        assert profile == {}

    def test_profile_sum_equals_total(self):
        """Sum of all stall_profile values must equal stall_total."""
        inst_list = [
            {"metrics": {
                "smsp__pcsamp_warp_stall_reason_memory_pipe_sample_count": 100,
                "smsp__pcsamp_warp_stall_reason_math_pipe_sample_count": 50,
                "smsp__pcsamp_warp_stall_reason_sync_sample_count": 30,
            }},
        ]
        profile, total = NCUMappingSystem._compute_stalls(inst_list)
        assert sum(profile.values()) == total

    def test_legacy_metric_format(self):
        """_compute_stalls works with legacy CATEGORY_MAP_LEGACY."""
        inst_list = [
            {"metrics": {
                "smsp__pcsamp_warps_issue_stalled_long_scoreboard": 100,
                "smsp__pcsamp_warps_issue_stalled_barrier": 50,
            }},
        ]
        profile, total = NCUMappingSystem._compute_stalls(
            inst_list, NCUMappingSystem.CATEGORY_MAP_LEGACY,
        )
        assert total == 150
        assert "Memory (DRAM/L2/L1)" in profile
        assert profile["Memory (DRAM/L2/L1)"] == 100
        assert "Sync / Barrier" in profile
        assert profile["Sync / Barrier"] == 50


class TestFocusHint:
    """Test _compute_focus_hint static method."""

    def test_pure_stall_point(self):
        hint = NCUMappingSystem._compute_focus_hint(
            -1.0, 10.0, "N/A", "Memory Load", None, 0, 100,
        )
        assert "pure stall point" in hint

    def test_extreme_stall(self):
        hint = NCUMappingSystem._compute_focus_hint(
            15.0, 0.5, "N/A", "Memory Load", None, 100, 50,
        )
        assert "extreme stall bottleneck" in hint

    def test_stall_bound(self):
        hint = NCUMappingSystem._compute_focus_hint(
            5.0, 0.5, "N/A", "Memory Load", None, 100, 50,
        )
        assert "stall-bound" in hint

    def test_deep_inlined_utility(self):
        chain = ["kernel.cu", "helper.cuh", "utils.cuh", "math.cuh"]
        hint = NCUMappingSystem._compute_focus_hint(
            0.0, 3.0, "N/A", "N/A", chain, 100, 0,
        )
        assert "deep inlined utility" in hint

    def test_memory_bound_hotspot(self):
        hint = NCUMappingSystem._compute_focus_hint(
            0.5, 10.0, "Memory (DRAM/L2/L1)", "Memory Load", None, 1000, 500,
        )
        assert "memory-bound hotspot" in hint

    def test_sync_overhead(self):
        hint = NCUMappingSystem._compute_focus_hint(
            0.5, 8.0, "Sync / Barrier", "Control Flow", None, 500, 200,
        )
        assert "sync overhead" in hint

    def test_compute_intensive(self):
        hint = NCUMappingSystem._compute_focus_hint(
            0.1, 6.0, "Compute (ALU/Tensor)", "Float Compute", None, 2000, 100,
        )
        assert "compute-intensive" in hint

    def test_stall_only_hotspot(self):
        hint = NCUMappingSystem._compute_focus_hint(
            0.0, 8.0, "N/A", "N/A", None, 0, 100,
        )
        assert "stall-only hotspot" in hint

    def test_empty_hint(self):
        hint = NCUMappingSystem._compute_focus_hint(
            0.0, 1.0, "N/A", "N/A", None, 100, 0,
        )
        assert hint == ""

    def test_multiple_hints(self):
        chain = ["kernel.cu", "deep/util.cuh", "math.cuh"]
        hint = NCUMappingSystem._compute_focus_hint(
            5.0, 8.0, "Memory (DRAM/L2/L1)", "Memory Load", chain, 100, 50,
        )
        assert "stall-bound" in hint
        assert "deep inlined utility" in hint
        assert "memory-bound hotspot" in hint


class TestIncludeTree:
    """Test #include tree building and chain tracing."""

    def test_short_path(self):
        assert NCUMappingSystem._short_path("/a/b/c.cu") == "b/c.cu"
        assert NCUMappingSystem._short_path("single.c") == "single.c"
        assert NCUMappingSystem._short_path("/a/b/c/d.cuh") == "c/d.cuh"

    def test_is_project_file(self):
        assert NCUMappingSystem._is_project_file("/myproject/src/kernel.cu") is True
        assert NCUMappingSystem._is_project_file("/workspace/project/file.cuh") is True
        assert NCUMappingSystem._is_project_file("/usr/include/stdio.h") is False
        assert NCUMappingSystem._is_project_file("/opt/nvidia/something.cuh") is False

    def test_build_include_tree(self):
        mapper = _make_empty_mapper()
        source_files = {
            "/project/kernel.cu": '#include "helper.cuh"\n#include "math.cuh"\n',
            "/project/helper.cuh": '#include "types.h"\n',
            "/project/math.cuh": '#include "types.h"\n',
            "/project/types.h": "",
        }
        rev = mapper.build_include_tree(source_files)
        # types.h is included by helper.cuh and math.cuh
        assert "project/types.h" in rev
        assert set(rev["project/types.h"]) == {"project/helper.cuh", "project/math.cuh"}
        # helper.cuh is included by kernel.cu
        assert "project/helper.cuh" in rev
        assert rev["project/helper.cuh"] == ["project/kernel.cu"]

    def test_build_include_tree_ignores_system_headers(self):
        mapper = _make_empty_mapper()
        source_files = {
            "/project/kernel.cu": '#include <stdio.h>\n#include "local.h"\n',
            "/project/local.h": "",
        }
        rev = mapper.build_include_tree(source_files)
        assert "stdio.h" not in rev
        assert "project/local.h" in rev

    def test_include_chains(self):
        mapper = _make_empty_mapper()
        source_files = {
            "/project/kernel.cu": '#include "mid.cuh"\n',
            "/project/mid.cuh": '#include "deep.cuh"\n',
            "/project/deep.cuh": "",
        }
        mapper.build_include_tree(source_files)
        mapper._build_include_chains()

        assert mapper._include_chains.get("project/mid.cuh") == ["project/kernel.cu", "project/mid.cuh"]
        assert mapper._include_chains.get("project/deep.cuh") == ["project/kernel.cu", "project/mid.cuh", "project/deep.cuh"]

    def test_include_chain_public_api(self):
        mapper = _make_empty_mapper()
        source_files = {
            "/project/kernel.cu": '#include "util.cuh"\n',
            "/project/util.cuh": "",
        }
        mapper.build_include_tree(source_files)
        mapper._build_include_chains()

        # get_include_chain works with full path
        chain = mapper.get_include_chain("/project/util.cuh")
        assert chain == ["project/kernel.cu", "project/util.cuh"]

    def test_include_chain_shortest_path(self):
        """When multiple paths exist, chain should use shortest."""
        mapper = _make_empty_mapper()
        source_files = {
            "/project/kernel.cu": '#include "a.cuh"\n#include "b.cuh"\n',
            "/project/a.cuh": '#include "deep.cuh"\n',
            "/project/b.cuh": '#include "mid.cuh"\n',
            "/project/mid.cuh": '#include "deep.cuh"\n',
            "/project/deep.cuh": "",
        }
        mapper.build_include_tree(source_files)
        mapper._build_include_chains()

        chain = mapper._include_chains.get("project/deep.cuh")
        # Two paths: kernel->a->deep (len 3) and kernel->b->mid->deep (len 4)
        # Should prefer shortest
        assert len(chain) == 3
        assert chain == ["project/kernel.cu", "project/a.cuh", "project/deep.cuh"]


class TestReportIncludeChain:
    """Test that bottleneck report includes include_chain field."""

    @pytest.fixture()
    def mapper_with_chain(self) -> NCUMappingSystem:
        mapper = _make_empty_mapper()
        mapper._s2as_flat = {
            ("k1", "/project/utils.cuh", 50): [
                {
                    "pc": "0x100", "sass": "LDG.E R0, [R2]",
                    "file": "/project/utils.cuh", "line": 50,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 500,
                        "inst_executed": 10000,
                    },
                },
            ],
        }
        mapper._kernels = ["k1"]
        mapper._total_samples_per_kernel = {"k1": 1000}
        mapper._total_exec_per_kernel = {"k1": 20000}
        mapper._include_chains = {
            "project/utils.cuh": ["project/kernel.cu", "project/helper.cuh", "project/utils.cuh"],
        }
        return mapper

    def test_report_has_include_chain(self, mapper_with_chain):
        report = mapper_with_chain.get_bottleneck_report(top_n=10)
        assert len(report) == 1
        assert report[0]["include_chain"] == ["project/kernel.cu", "project/helper.cuh", "project/utils.cuh"]

    def test_report_chain_none_when_missing(self):
        mapper = _make_empty_mapper()
        mapper._s2as_flat = {
            ("k1", "/project/standalone.cu", 10): [
                {
                    "pc": "0x100", "sass": "FFMA R0, R1, R2, R3",
                    "file": "/project/standalone.cu", "line": 10,
                    "metrics": {
                        "smsp__pcsamp_sample_count": 100,
                        "inst_executed": 1000,
                    },
                },
            ],
        }
        mapper._kernels = ["k1"]
        mapper._total_samples_per_kernel = {"k1": 200}
        mapper._total_exec_per_kernel = {"k1": 2000}
        report = mapper.get_bottleneck_report(top_n=10)
        assert report[0]["include_chain"] is None

    def test_report_has_focus_hint(self, mapper_with_chain):
        report = mapper_with_chain.get_bottleneck_report(top_n=10)
        assert "focus_hint" in report[0]
        assert isinstance(report[0]["focus_hint"], str)

    def test_format_shows_chain(self, mapper_with_chain):
        text = mapper_with_chain.format_hotspot_report(top_n=10)
        assert "chain:" in text
        assert "project/kernel.cu" in text
        assert "project/utils.cuh" in text
