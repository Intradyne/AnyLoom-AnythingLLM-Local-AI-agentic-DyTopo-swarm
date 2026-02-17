#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# dev_run.sh
# ============================================================================
# Full stack validation for AnyLoom development environment
#
# This script checks:
# - Qdrant accessibility on port 6333
# - LLM server accessibility on port 8008
# - Python imports (dytopo, qdrant_client, openai)
# - Configuration loading
#
# Usage:
#   bash scripts/dev_run.sh
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
# ============================================================================

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QDRANT_PORT=6333
LLM_PORT=8008

# Tracking
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  AnyLoom Stack Validation${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# ============================================================================
# Helper functions
# ============================================================================

check_pass() {
    echo -e "${GREEN}[✓]${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

check_fail() {
    echo -e "${RED}[✗]${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

check_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# ============================================================================
# Check 1: Qdrant
# ============================================================================
echo -e "${BLUE}[1/4] Checking Qdrant...${NC}"

if curl -s --max-time 3 "http://localhost:${QDRANT_PORT}" > /dev/null 2>&1; then
    # Get version info if possible
    VERSION_INFO=$(curl -s --max-time 2 "http://localhost:${QDRANT_PORT}" 2>/dev/null | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    check_pass "Qdrant is accessible on port ${QDRANT_PORT} (version: ${VERSION_INFO})"

    # Check if our collection exists
    COLLECTIONS=$(curl -s --max-time 2 "http://localhost:${QDRANT_PORT}/collections" 2>/dev/null || echo "{}")
    if echo "${COLLECTIONS}" | grep -q "anyloom_docs"; then
        check_pass "Collection 'anyloom_docs' exists"
    else
        check_warn "Collection 'anyloom_docs' not found (run: python src/reindex.py)"
    fi
else
    check_fail "Qdrant is NOT accessible on port ${QDRANT_PORT}"
    echo -e "         ${RED}Run: docker start anyloom-qdrant${NC}"
fi
echo ""

# ============================================================================
# Check 2: LLM server
# ============================================================================
echo -e "${BLUE}[2/4] Checking LLM server...${NC}"

LLM_AVAILABLE=false

if curl -s --max-time 3 "http://localhost:${LLM_PORT}/v1/models" > /dev/null 2>&1; then
    MODEL_INFO=$(curl -s --max-time 2 "http://localhost:${LLM_PORT}/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4 || echo "unknown")
    check_pass "LLM server is accessible on port ${LLM_PORT} (model: ${MODEL_INFO})"
    LLM_AVAILABLE=true
else
    check_fail "LLM server is NOT accessible on port ${LLM_PORT}"
    echo -e "         ${RED}Run: docker compose up -d llm${NC}"
fi
echo ""

# ============================================================================
# Check 3: Python imports
# ============================================================================
echo -e "${BLUE}[3/4] Checking Python environment...${NC}"

# Add src to Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    check_fail "Python not found in PATH"
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}  Validation Failed${NC}"
    echo -e "${RED}======================================${NC}"
    exit 1
fi

PYTHON_VERSION=$(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')
check_pass "Python ${PYTHON_VERSION} found"

# Test critical imports
IMPORT_RESULTS=$(${PYTHON_CMD} << 'EOF'
import sys
results = []

# Test dytopo
try:
    import dytopo
    results.append(("dytopo", "PASS", ""))
except ImportError as e:
    results.append(("dytopo", "FAIL", str(e)))

# Test dytopo.config
try:
    from dytopo.config import load_config
    results.append(("dytopo.config", "PASS", ""))
except ImportError as e:
    results.append(("dytopo.config", "FAIL", str(e)))

# Test openai
try:
    import openai
    results.append(("openai", "PASS", openai.__version__))
except ImportError as e:
    results.append(("openai", "FAIL", str(e)))

# Test qdrant_client
try:
    import qdrant_client
    results.append(("qdrant_client", "PASS", qdrant_client.__version__))
except ImportError as e:
    results.append(("qdrant_client", "FAIL", str(e)))

# Test yaml
try:
    import yaml
    results.append(("pyyaml", "PASS", ""))
except ImportError as e:
    results.append(("pyyaml", "FAIL", str(e)))

# Test networkx
try:
    import networkx
    results.append(("networkx", "PASS", ""))
except ImportError as e:
    results.append(("networkx", "FAIL", str(e)))

# Test sentence_transformers
try:
    import sentence_transformers
    results.append(("sentence_transformers", "PASS", ""))
except ImportError as e:
    results.append(("sentence_transformers", "FAIL", str(e)))

for name, status, info in results:
    print(f"{name}|{status}|{info}")
EOF
)

IMPORT_FAILED=false
while IFS='|' read -r name status info; do
    if [[ "${status}" == "PASS" ]]; then
        if [[ -n "${info}" ]]; then
            check_pass "Import ${name} (version: ${info})"
        else
            check_pass "Import ${name}"
        fi
    else
        check_fail "Import ${name} failed: ${info}"
        IMPORT_FAILED=true
    fi
done <<< "${IMPORT_RESULTS}"

if [[ "${IMPORT_FAILED}" == "true" ]]; then
    echo -e "         ${RED}Run: pip install -r requirements-dytopo.txt${NC}"
fi
echo ""

# ============================================================================
# Check 4: Configuration
# ============================================================================
echo -e "${BLUE}[4/4] Checking configuration...${NC}"

CONFIG_CHECK=$(${PYTHON_CMD} << 'EOF'
import sys
try:
    from dytopo.config import load_config
    config = load_config()

    # Check required sections
    required_sections = ['llm', 'routing', 'orchestration', 'logging', 'concurrency']
    for section in required_sections:
        if section not in config:
            print(f"FAIL|Missing section: {section}")
            sys.exit(1)

    # Extract key config values
    backend = config['concurrency']['backend']
    base_url = config['llm']['base_url']
    model = config['llm']['model']
    max_concurrent = config['concurrency']['max_concurrent']
    log_dir = config['logging']['log_dir']

    print(f"PASS|backend={backend}")
    print(f"PASS|base_url={base_url}")
    print(f"PASS|model={model}")
    print(f"PASS|max_concurrent={max_concurrent}")
    print(f"PASS|log_dir={log_dir}")

except Exception as e:
    print(f"FAIL|{e}")
    sys.exit(1)
EOF
)

if [[ $? -eq 0 ]]; then
    check_pass "Configuration file loaded successfully"

    # Parse and display config details
    while IFS='|' read -r status message; do
        if [[ "${status}" == "PASS" ]]; then
            echo -e "         ${message}"
        fi
    done <<< "${CONFIG_CHECK}"

    # Validate backend matches available services
    BACKEND=$(echo "${CONFIG_CHECK}" | grep "backend=" | cut -d'=' -f2)
    BASE_URL=$(echo "${CONFIG_CHECK}" | grep "base_url=" | cut -d'=' -f2)

    if [[ "${BACKEND}" == "llama-cpp" ]]; then
        if [[ "${LLM_AVAILABLE}" == "true" ]]; then
            check_pass "Backend configured for llama-cpp and service is available"
        else
            check_warn "Backend configured for llama-cpp but service is NOT available"
            echo -e "         ${YELLOW}Run: docker compose up -d llm${NC}"
        fi
    fi
else
    check_fail "Configuration validation failed"
    echo -e "${RED}${CONFIG_CHECK}${NC}"
    echo -e "         ${RED}Check dytopo_config.yaml for syntax errors${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}======================================${NC}"

if [[ ${CHECKS_FAILED} -eq 0 ]] && [[ ${WARNINGS} -eq 0 ]]; then
    echo -e "${GREEN}  All Systems Ready!${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo -e "${GREEN}✓ ${CHECKS_PASSED} checks passed${NC}"
    echo ""
    echo -e "${BLUE}You can now:${NC}"
    echo -e "  - Run DyTopo examples: ${YELLOW}python examples/*.py${NC}"
    echo -e "  - Start development: ${YELLOW}python src/your_script.py${NC}"
    echo -e "  - Initialize RAG: ${YELLOW}python src/reindex.py${NC}"
    echo ""
    exit 0
elif [[ ${CHECKS_FAILED} -eq 0 ]] && [[ ${WARNINGS} -gt 0 ]]; then
    echo -e "${YELLOW}  Validation Complete with Warnings${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo -e "${GREEN}✓ ${CHECKS_PASSED} checks passed${NC}"
    echo -e "${YELLOW}⚠ ${WARNINGS} warnings${NC}"
    echo ""
    echo -e "${YELLOW}System is functional but some optional components are missing.${NC}"
    echo -e "${YELLOW}Review warnings above to optimize your setup.${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}  Validation Failed${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    echo -e "${GREEN}✓ ${CHECKS_PASSED} checks passed${NC}"
    echo -e "${RED}✗ ${CHECKS_FAILED} checks failed${NC}"
    if [[ ${WARNINGS} -gt 0 ]]; then
        echo -e "${YELLOW}⚠ ${WARNINGS} warnings${NC}"
    fi
    echo ""
    echo -e "${RED}Please fix the failures above before proceeding.${NC}"
    echo ""
    echo -e "${BLUE}Quick fixes:${NC}"
    echo -e "  - Qdrant: ${YELLOW}bash scripts/bootstrap_dev.sh${NC}"
    echo -e "  - LLM: ${YELLOW}docker compose up -d llm${NC}"
    echo -e "  - Python deps: ${YELLOW}pip install -r requirements-dytopo.txt${NC}"
    echo ""
    exit 1
fi
