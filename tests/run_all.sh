#!/usr/bin/env bash
# Tachyon 全量回归测试脚本
# 用法: bash tests/run_all.sh          # 默认运行全部测试
#       bash tests/run_all.sh unit      # 只跑 unit tests
#       bash tests/run_all.sh integ     # 只跑 integration tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

MODE="${1:-all}"

echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}  Tachyon 全量回归测试${RESET}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

FAILED=0

run_suite() {
    local label="$1"
    local path="$2"

    if [ ! -d "$path" ]; then
        echo -e "${YELLOW}[SKIP]${RESET} $label — 目录不存在: $path"
        return
    fi

    # 检查目录下是否有 test_*.py 文件
    if ! find "$path" -maxdepth 1 -name 'test_*.py' -print -quit | grep -q .; then
        echo -e "${YELLOW}[SKIP]${RESET} $label — 无测试文件"
        return
    fi

    echo -e "${BOLD}▶ $label${RESET}"
    if python -m pytest "$path" -v --tb=short -q 2>&1; then
        echo -e "${GREEN}[PASS]${RESET} $label"
    else
        echo -e "${RED}[FAIL]${RESET} $label"
        FAILED=1
    fi
    echo ""
}

case "$MODE" in
    unit)
        run_suite "Unit Tests" "tests/unit"
        ;;
    integ|integration)
        run_suite "Integration Tests" "tests/integration"
        ;;
    all)
        run_suite "Unit Tests" "tests/unit"
        run_suite "Integration Tests" "tests/integration"
        ;;
    *)
        echo -e "${RED}未知模式: $MODE${RESET}"
        echo "用法: $0 [all|unit|integ]"
        exit 1
        ;;
esac

echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  全部通过 ✓${RESET}"
else
    echo -e "${RED}${BOLD}  存在失败 ✗${RESET}"
    exit 1
fi
