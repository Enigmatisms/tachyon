"""Unit tests for tachyon.utils.kernel_filter."""
from __future__ import annotations

import pytest

from tachyon.utils.kernel_filter import (
    filter_kernels,
    match_kernel_name,
    to_ncu_regex,
)


class TestMatchKernelName:

    def test_glob_match_star(self):
        assert match_kernel_name("matmul_kernel_f32", None, "matmul*")

    def test_glob_match_question(self):
        assert match_kernel_name("reduce_v2", None, "reduce_v?")

    def test_glob_no_match(self):
        assert not match_kernel_name("softmax_kernel", None, "matmul*")

    def test_regex_match(self):
        assert match_kernel_name("matmul_kernel_f32", None, r"matmul_kernel_\w+")

    def test_regex_no_match(self):
        assert not match_kernel_name("softmax_kernel", None, r"^matmul_\d+$")

    def test_substring_match_case_insensitive(self):
        assert match_kernel_name("MyMatMulKernel", None, "matmul")

    def test_substring_no_match(self):
        assert not match_kernel_name("softmax_kernel", None, "matmul")

    def test_demangled_name_match(self):
        assert match_kernel_name("_Z7matmulPfS_S_", "matmul<float>", "matmul*")

    def test_demangled_name_fallback(self):
        # Only mangled name matches
        assert match_kernel_name("_Z7matmulPfS_S_", "softmax<float>", "_Z7matmul*")

    def test_none_demangled_name(self):
        assert match_kernel_name("matmul_kernel", None, "matmul*")


class TestFilterKernels:

    @pytest.fixture
    def sample_kernels(self):
        from tachyon.models.kernel import (
            DeviceInfo,
            KernelReport,
            LaunchParams,
        )

        def _make(name, demangled=None):
            return KernelReport(
                kernel_name=name,
                demangled_name=demangled or name,
                launch_params=LaunchParams(
                    grid=(1, 1, 1), block=(256, 1, 1),
                    shared_mem_bytes=0, registers_per_thread=32,
                ),
                device_info=DeviceInfo(
                    name="A100", compute_capability=(8, 0),
                    sm_count=108, max_clock_mhz=1410,
                    memory_bus_width=5120, peak_memory_bandwidth_gbps=2039.0,
                ),
                metrics={},
            )

        return [
            _make("matmul_f32", "matmul<float>"),
            _make("matmul_f16", "matmul<half>"),
            _make("softmax_kernel", "softmax<float>"),
            _make("reduce_sum", "reduce_sum<float, 256>"),
        ]

    def test_filter_glob(self, sample_kernels):
        result = filter_kernels(sample_kernels, "matmul*")
        assert len(result) == 2
        assert all("matmul" in k.kernel_name for k in result)

    def test_filter_regex(self, sample_kernels):
        result = filter_kernels(sample_kernels, r"matmul_f\d+")
        assert len(result) == 2

    def test_filter_substring(self, sample_kernels):
        result = filter_kernels(sample_kernels, "reduce")
        assert len(result) == 1
        assert result[0].kernel_name == "reduce_sum"

    def test_filter_no_match(self, sample_kernels):
        result = filter_kernels(sample_kernels, "convolution*")
        assert len(result) == 0

    def test_filter_all_match(self, sample_kernels):
        result = filter_kernels(sample_kernels, "*")
        assert len(result) == 4

    def test_preserves_order(self, sample_kernels):
        result = filter_kernels(sample_kernels, "*")
        assert [k.kernel_name for k in result] == [
            "matmul_f32", "matmul_f16", "softmax_kernel", "reduce_sum"
        ]


class TestToNcuRegex:

    def test_simple_glob_star(self):
        result = to_ncu_regex("matmul*")
        assert result.startswith("matmul")
        assert ".*" in result

    def test_glob_question(self):
        result = to_ncu_regex("reduce_v?")
        assert "." in result  # ? -> .

    def test_regex_passthrough(self):
        assert to_ncu_regex(r"matmul_kernel_\d+") == r"matmul_kernel_\d+"

    def test_regex_with_anchors(self):
        assert to_ncu_regex("^matmul") == "^matmul"

    def test_regex_with_groups(self):
        assert to_ncu_regex("(matmul|softmax)") == "(matmul|softmax)"

    def test_plain_string(self):
        # No glob or regex chars — passed through as-is
        assert to_ncu_regex("matmul_kernel") == "matmul_kernel"
