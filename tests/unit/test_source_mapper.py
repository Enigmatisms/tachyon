"""Unit tests for NCUMappingSystem — SASS classification and hotspot report.

The mapper requires NCU SWIG bindings, so we only test the pure-logic parts:
- SASS instruction classification
- Hotspot report formatting from pre-built data
- Query API
"""
from unittest.mock import MagicMock

import pytest

from tachyon.correlator.source_mapper import NCUMappingSystem


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
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        from collections import defaultdict
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
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._s2as_flat = {}
        mapper._as2s_flat = {}
        mapper._kernels = []
        mapper._total_samples_per_kernel = {}
        mapper._total_exec_per_kernel = {}
        report = mapper.get_bottleneck_report()
        assert report == []

    def test_format_hotspot_report(self, mapper_with_data):
        text = mapper_with_data.format_hotspot_report(top_n=5)
        assert "test_kernel" in text
        assert "gemm.cu" in text
        assert "50.0%" in text
        assert "memory load" in text  # formatted output uses lowercase

    def test_format_empty_report(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._s2as_flat = {}
        mapper._as2s_flat = {}
        mapper._kernels = []
        mapper._total_samples_per_kernel = {}
        mapper._total_exec_per_kernel = {}
        text = mapper.format_hotspot_report()
        assert "No source-mapped instructions" in text


class TestMapperQueryAPI:
    """Test public query API."""

    def test_get_mapped_sources(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._s2as_flat = {
            ("k1", "/a/b.cu", 10): [],
            ("k1", "/a/c.cu", 20): [],
            ("k2", "/a/b.cu", 10): [],
        }
        sources = mapper.get_mapped_sources()
        assert sources == ["/a/b.cu", "/a/c.cu"]

    def test_get_sass_by_line(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._s2as_flat = {}
        mapper._get_sass_fast = mapper._s2as_flat.get
        insts = [
            {"pc": "0x1000", "sass": "LDG.E R0", "file": "a.cu", "line": 10, "metrics": {}},
        ]
        mapper._s2as_flat[("k1", "a.cu", 10)] = insts
        result = mapper.get_sass_by_line("k1", "a.cu", 10)
        assert len(result) == 1
        assert result[0]["sass"] == "LDG.E R0"

    def test_get_sass_by_line_empty(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._s2as_flat = {}
        mapper._get_sass_fast = mapper._s2as_flat.get
        result = mapper.get_sass_by_line("k1", "missing.cu", 99)
        assert result == []

    def test_get_source_by_pc(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._as2s_flat = {}
        mapper._get_src_fast = mapper._as2s_flat.get
        entry = {"pc": "0x1000", "sass": "LDG", "file": "a.cu", "line": 10, "metrics": {}}
        mapper._as2s_flat[("k1", 0x1000)] = entry
        result = mapper.get_source_by_pc("k1", 0x1000)
        assert result["file"] == "a.cu"

    def test_get_source_by_pc_hex_string(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._as2s_flat = {}
        mapper._get_src_fast = mapper._as2s_flat.get
        entry = {"pc": "0x2000", "sass": "FFMA", "file": "b.cu", "line": 20, "metrics": {}}
        mapper._as2s_flat[("k1", 0x2000)] = entry
        result = mapper.get_source_by_pc("k1", "0x2000")
        assert result is not None
        assert result["line"] == 20

    def test_get_kernels(self):
        mapper = NCUMappingSystem.__new__(NCUMappingSystem)
        mapper._kernels = ["kernel_a", "kernel_b"]
        assert mapper.get_kernels() == ["kernel_a", "kernel_b"]
