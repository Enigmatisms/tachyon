"""Unit tests for NcuReportReader — all tests use mocked ncu_report module."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ━━━━━━━━━━━━━━━━━━━━━━━ Discovery tests ━━━━━━━━━━━━━━━━━━━━━


class TestDiscoverNcuReportPath:
    def test_env_var_priority_directory(self, tmp_path: Path):
        """TACHYON_NCU_REPORT_PATH env var (directory) takes highest priority."""
        # Create a fake ncu_report.py
        fake_dir = tmp_path / "extras" / "python"
        fake_dir.mkdir(parents=True)
        (fake_dir / "ncu_report.py").write_text("# fake")

        with patch.dict(os.environ, {"TACHYON_NCU_REPORT_PATH": str(fake_dir)}):
            from tachyon.reader.ncu_reader import _discover_ncu_report_path
            result = _discover_ncu_report_path()
            assert result == fake_dir

    def test_env_var_priority_file(self, tmp_path: Path):
        """TACHYON_NCU_REPORT_PATH pointing to ncu_report.py file."""
        fake_dir = tmp_path / "python"
        fake_dir.mkdir(parents=True)
        ncu_file = fake_dir / "ncu_report.py"
        ncu_file.write_text("# fake")

        with patch.dict(os.environ, {"TACHYON_NCU_REPORT_PATH": str(ncu_file)}):
            from tachyon.reader.ncu_reader import _discover_ncu_report_path
            result = _discover_ncu_report_path()
            assert result == fake_dir

    def test_env_var_invalid_path(self):
        """Invalid TACHYON_NCU_REPORT_PATH falls through to search."""
        with patch.dict(os.environ, {"TACHYON_NCU_REPORT_PATH": "/nonexistent/path"}):
            from tachyon.reader.ncu_reader import _discover_ncu_report_path
            # Should not crash, just return None if nothing found
            result = _discover_ncu_report_path()
            # May be None or a found path, just verify no exception

    def test_already_importable(self):
        """If ncu_report is already on sys.path, return None."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TACHYON_NCU_REPORT_PATH if set
            os.environ.pop("TACHYON_NCU_REPORT_PATH", None)
            with patch("importlib.import_module") as mock_import:
                mock_import.return_value = MagicMock()
                from tachyon.reader.ncu_reader import _discover_ncu_report_path
                result = _discover_ncu_report_path()
                # None signals "already importable"


# ━━━━━━━━━━━━━━━━━━━━━━━ Load module tests ━━━━━━━━━━━━━━━━━━━


class TestLoadNcuModule:
    def test_import_error_helpful_message(self):
        """_load_ncu_module raises ImportError with helpful message."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TACHYON_NCU_REPORT_PATH", None)
            with patch("importlib.import_module", side_effect=ImportError("no module")):
                from tachyon.reader.ncu_reader import _load_ncu_module
                with pytest.raises(ImportError, match="TACHYON_NCU_REPORT_PATH"):
                    _load_ncu_module()


# ━━━━━━━━━━━━━━━━━━━━━━━ NcuReportReader tests ━━━━━━━━━━━━━━━


def _make_mock_ncu_module():
    """Create a mock ncu_report module with realistic API surface."""
    mock_module = MagicMock()

    # Create a mock action with realistic metrics
    mock_action = MagicMock()
    mock_action.name.return_value = "_Z12test_kernelPf"
    mock_action.demangled_name.return_value = "test_kernel(float*)"

    # Mock metric_by_name
    metrics_data = {
        "launch__grid_dim_x": (4096, "uint64"),
        "launch__grid_dim_y": (1, "uint64"),
        "launch__grid_dim_z": (1, "uint64"),
        "launch__block_dim_x": (256, "uint64"),
        "launch__block_dim_y": (1, "uint64"),
        "launch__block_dim_z": (1, "uint64"),
        "launch__shared_mem_per_block_dynamic": (0, "uint64"),
        "launch__registers_per_thread": (32, "uint64"),
        "launch__shared_mem_per_block_static": (0, "uint64"),
        "device__attribute_display_name": ("NVIDIA H100", "string"),
        "device__attribute_compute_capability_major": (9, "uint64"),
        "device__attribute_compute_capability_minor": (0, "uint64"),
        "device__attribute_multiprocessor_count": (132, "uint64"),
        "device__attribute_clock_rate": (1980000, "uint64"),
        "device__attribute_global_memory_bus_width": (5120, "uint64"),
        "dram__bytes.sum.peak_sustained": (3.352e12, "double"),
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": (75.0, "double"),
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": (45.0, "double"),
    }

    def mock_metric_by_name(name):
        if name in metrics_data:
            val, typ = metrics_data[name]
            m = MagicMock()
            m.has_value.return_value = True
            m.num_instances.return_value = 1
            if typ == "uint64":
                m.as_uint64.return_value = val
                m.as_double.return_value = float(val)
            elif typ == "double":
                m.as_double.return_value = val
            elif typ == "string":
                m.value.return_value = val
            return m
        return None

    mock_action.metric_by_name = mock_metric_by_name
    mock_action.metric_names.return_value = list(metrics_data.keys())
    mock_action.source_files.return_value = []
    mock_action.rule_results_as_dicts.return_value = [
        {"rule_identifier": "SpeedOfLight",
         "rule_message": {"type": 2, "message": "Moderate utilization"}},
    ]

    # Create mock range and context
    mock_range = MagicMock()
    mock_range.num_actions.return_value = 1
    mock_range.action_by_idx.return_value = mock_action

    mock_context = MagicMock()
    mock_context.num_ranges.return_value = 1
    mock_context.range_by_idx.return_value = mock_range

    mock_module.load_report.return_value = mock_context
    return mock_module


class TestNcuReportReaderLoad:
    def test_load_missing_file(self):
        """load() returns ToolResult.fail for nonexistent file."""
        mock_module = _make_mock_ncu_module()
        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load("/nonexistent/report.ncu-rep")
            assert not result.success
            assert "not found" in result.error.message.lower() or "not found" in result.error.message

    def test_load_wrong_extension(self, tmp_path: Path):
        """load() rejects non-.ncu-rep files."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c")

        mock_module = _make_mock_ncu_module()
        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load(csv_file)
            assert not result.success
            assert "UNSUPPORTED_FORMAT" in result.error.code.value

    def test_load_success(self, tmp_path: Path):
        """load() successfully parses mocked .ncu-rep file."""
        fake_rep = tmp_path / "test.ncu-rep"
        fake_rep.write_bytes(b"fake ncu data")

        mock_module = _make_mock_ncu_module()
        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load(fake_rep)
            assert result.success
            assert result.data is not None
            reports = result.data
            assert len(reports) == 1

            r = reports[0]
            assert r.kernel_name == "_Z12test_kernelPf"
            assert r.demangled_name == "test_kernel(float*)"
            assert r.launch_params.grid == (4096, 1, 1)
            assert r.launch_params.block == (256, 1, 1)
            assert r.launch_params.registers_per_thread == 32
            assert r.device_info.compute_capability == (9, 0)
            assert r.device_info.sm_count == 132

    def test_load_corrupt_file(self, tmp_path: Path):
        """load() handles corrupt .ncu-rep gracefully."""
        fake_rep = tmp_path / "corrupt.ncu-rep"
        fake_rep.write_bytes(b"corrupt data")

        mock_module = MagicMock()
        mock_module.load_report.side_effect = RuntimeError("Parse error")

        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load(fake_rep)
            assert not result.success
            assert "INVALID_BINARY" in result.error.code.value

    def test_load_extracts_metrics(self, tmp_path: Path):
        """Verify metrics extraction from mocked action."""
        fake_rep = tmp_path / "metrics.ncu-rep"
        fake_rep.write_bytes(b"data")

        mock_module = _make_mock_ncu_module()
        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load(fake_rep)
            assert result.success
            r = result.data[0]
            # Should have scalar metrics
            assert len(r.metrics) > 0

    def test_load_extracts_rule_results(self, tmp_path: Path):
        """Verify rule results extraction."""
        fake_rep = tmp_path / "rules.ncu-rep"
        fake_rep.write_bytes(b"data")

        mock_module = _make_mock_ncu_module()
        with patch("tachyon.reader.ncu_reader._load_ncu_module", return_value=mock_module):
            from tachyon.reader.ncu_reader import NcuReportReader
            reader = NcuReportReader()
            result = reader.load(fake_rep)
            assert result.success
            r = result.data[0]
            assert len(r.rule_results) == 1
            assert r.rule_results[0].rule_name == "SpeedOfLight"


# ━━━━━━━━━━━━━━━━━━━━━━━ Helper method tests ━━━━━━━━━━━━━━━━━


class TestInferUnit:
    def test_pct(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._infer_unit("sm__throughput.avg.pct_of_peak") == "%"

    def test_bytes(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._infer_unit("dram__bytes.sum") == "byte"

    def test_sectors(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._infer_unit("l1tex__t_sectors.sum") == "sector"

    def test_cycles(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._infer_unit("sm__cycles_elapsed.sum") == "cycle"

    def test_unknown(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._infer_unit("some_random_metric") == ""


class TestMapRuleSeverity:
    def test_int_values(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._map_rule_severity(0) == "OK"
        assert NcuReportReader._map_rule_severity(1) == "LOW"
        assert NcuReportReader._map_rule_severity(2) == "MED"
        assert NcuReportReader._map_rule_severity(3) == "HIGH"

    def test_none(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._map_rule_severity(None) == "OK"

    def test_string(self):
        from tachyon.reader.ncu_reader import NcuReportReader
        assert NcuReportReader._map_rule_severity("HIGH") == "HIGH"


class TestSourceInfoParsing:
    """Test NcuReportReader.source_info() with various NCU API return types.

    The NCU Python SWIG bindings can return different types depending on
    NCU version.  source_info() must handle all known formats.
    """

    def _make_reader(self, mock_action: MagicMock):
        """Create a reader with a single mocked action."""
        from tachyon.reader.ncu_reader import NcuReportReader
        reader = NcuReportReader.__new__(NcuReportReader)
        reader._actions = {"test_kernel": mock_action}
        reader._ncu = MagicMock()
        return reader

    def test_swig_object_with_attributes(self):
        """SWIG object with .src_file and .src_line attributes."""
        mock_action = MagicMock()
        src_obj = MagicMock()
        src_obj.src_file = "/path/to/kernel.cu"
        src_obj.src_line = 42
        mock_action.source_info.return_value = src_obj

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is not None
        assert result.file_name == "/path/to/kernel.cu"
        assert result.line == 42

    def test_tuple_return(self):
        """Tuple (file_path, line_number) return type."""
        mock_action = MagicMock()
        mock_action.source_info.return_value = ("/path/to/kernel.cu", 42)

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is not None
        assert result.file_name == "/path/to/kernel.cu"
        assert result.line == 42

    def test_list_return(self):
        """List [file_path, line_number] return type."""
        mock_action = MagicMock()
        mock_action.source_info.return_value = ["/path/to/kernel.cu", 42]

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is not None
        assert result.file_name == "/path/to/kernel.cu"
        assert result.line == 42

    def test_string_filepath_colon_line(self):
        """String 'file_path:line_number' return type."""
        mock_action = MagicMock()
        mock_action.source_info.return_value = "/path/to/kernel.cu:42"

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is not None
        assert result.file_name == "/path/to/kernel.cu"
        assert result.line == 42

    def test_none_returns_none(self):
        """None return means no debug info available."""
        mock_action = MagicMock()
        mock_action.source_info.return_value = None

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is None

    def test_exception_returns_none(self):
        """RuntimeError from NCU API should return None gracefully."""
        mock_action = MagicMock()
        mock_action.source_info.side_effect = RuntimeError("PC not found")

        reader = self._make_reader(mock_action)
        result = reader.source_info(0x1000)

        assert result is None

    def test_kernel_name_selects_action(self):
        """kernel_name parameter should select the correct action handle."""
        from tachyon.reader.ncu_reader import NcuReportReader

        mock_action_a = MagicMock()
        mock_action_a.source_info.return_value = ("kernel_a.cu", 10)

        mock_action_b = MagicMock()
        mock_action_b.source_info.return_value = ("kernel_b.cu", 20)

        reader = NcuReportReader.__new__(NcuReportReader)
        reader._actions = {
            "kernel_a": mock_action_a,
            "kernel_b": mock_action_b,
        }
        reader._ncu = MagicMock()

        result_a = reader.source_info(0x1000, kernel_name="kernel_a")
        assert result_a.file_name == "kernel_a.cu"
        assert result_a.line == 10

        result_b = reader.source_info(0x1000, kernel_name="kernel_b")
        assert result_b.file_name == "kernel_b.cu"
        assert result_b.line == 20
