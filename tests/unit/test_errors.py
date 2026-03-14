"""Unit tests for error handling infrastructure."""
from tachyon.errors.handler import ERROR_SUGGESTIONS, ErrorCode, ErrorInfo, ToolResult


class TestErrorCode:
    def test_values(self):
        assert ErrorCode.UNSUPPORTED_FORMAT == "UNSUPPORTED_FORMAT"
        assert ErrorCode.INVALID_BINARY == "INVALID_BINARY"
        assert ErrorCode.LLM_UNAVAILABLE == "LLM_UNAVAILABLE"

    def test_is_string_enum(self):
        assert isinstance(ErrorCode.UNKNOWN, str)
        assert ErrorCode.UNKNOWN == "UNKNOWN"


class TestErrorInfo:
    def test_creation(self):
        ei = ErrorInfo(
            code=ErrorCode.INVALID_BINARY,
            message="File corrupt",
            suggestion="Re-download the file",
        )
        assert ei.code == ErrorCode.INVALID_BINARY
        assert ei.message == "File corrupt"
        assert ei.context == {}

    def test_with_context(self):
        ei = ErrorInfo(
            code=ErrorCode.METRIC_NOT_FOUND,
            message="Missing metric",
            suggestion="Re-profile",
            context={"metric": "sm__throughput"},
        )
        assert ei.context["metric"] == "sm__throughput"


class TestToolResult:
    def test_ok(self):
        result = ToolResult.ok([1, 2, 3])
        assert result.success is True
        assert result.data == [1, 2, 3]
        assert result.error is None

    def test_fail(self):
        result = ToolResult.fail(
            ErrorCode.INVALID_BINARY,
            "File not found",
            suggestion="Check the path",
        )
        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.code == ErrorCode.INVALID_BINARY
        assert "not found" in result.error.message

    def test_fail_with_context(self):
        result = ToolResult.fail(
            ErrorCode.TOOL_NOT_FOUND,
            "ncu not found",
            context={"searched": ["/usr/local/cuda"]},
        )
        assert result.error.context["searched"] == ["/usr/local/cuda"]

    def test_ok_with_none_data(self):
        result = ToolResult.ok(None)
        assert result.success is True
        assert result.data is None


class TestErrorSuggestions:
    def test_all_codes_have_suggestions(self):
        """Verify all error codes in suggestions dict are valid."""
        for code in ERROR_SUGGESTIONS:
            assert isinstance(code, ErrorCode)
            assert len(ERROR_SUGGESTIONS[code]) > 0

    def test_key_suggestions_exist(self):
        assert ErrorCode.UNSUPPORTED_FORMAT in ERROR_SUGGESTIONS
        assert ErrorCode.LLM_UNAVAILABLE in ERROR_SUGGESTIONS
        assert ErrorCode.TOOL_NOT_FOUND in ERROR_SUGGESTIONS
