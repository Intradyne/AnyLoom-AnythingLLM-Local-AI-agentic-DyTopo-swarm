#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# bootstrap_dev.sh
# ============================================================================
# One-shot development environment setup for AnyLoom
#
# This script:
# - Installs Python dependencies from requirements-dytopo.txt
# - Checks/starts Qdrant container on port 6333
# - Creates log directories
# - Validates configuration with Python
#
# Usage:
#   bash scripts/bootstrap_dev.sh
# ============================================================================

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QDRANT_PORT=6333
QDRANT_CONTAINER_NAME="anyloom-qdrant"
QDRANT_VOLUME="qdrant_anyloom"
REQUIREMENTS_FILE="requirements-dytopo.txt"
LOG_DIR="${HOME}/dytopo-logs"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  AnyLoom Development Bootstrap${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# ============================================================================
# Step 1: Check project structure
# ============================================================================
echo -e "${YELLOW}[1/5] Checking project structure...${NC}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo -e "${RED}ERROR: ${REQUIREMENTS_FILE} not found.${NC}"
    echo -e "${RED}Please run this script from the project root directory.${NC}"
    exit 1
fi

if [[ ! -f "dytopo_config.yaml" ]]; then
    echo -e "${YELLOW}⚠ Warning: dytopo_config.yaml not found.${NC}"
    echo -e "${YELLOW}The default configuration will be used.${NC}"
fi

if [[ ! -d "src/dytopo" ]]; then
    echo -e "${RED}ERROR: src/dytopo directory not found.${NC}"
    echo -e "${RED}Please run this script from the project root directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Project structure valid${NC}"
echo ""

# ============================================================================
# Step 2: Install Python dependencies
# ============================================================================
echo -e "${YELLOW}[2/5] Installing Python dependencies...${NC}"

if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)
PYTHON_VERSION=$(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}Using Python ${PYTHON_VERSION}${NC}"

# Check pip
if ! ${PYTHON_CMD} -m pip --version &> /dev/null; then
    echo -e "${RED}ERROR: pip not found. Please install pip.${NC}"
    exit 1
fi

echo -e "${BLUE}Installing dependencies from ${REQUIREMENTS_FILE}...${NC}"
if ${PYTHON_CMD} -m pip install -r "${REQUIREMENTS_FILE}" --quiet; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}ERROR: Failed to install Python dependencies${NC}"
    echo -e "${RED}Try manually: pip install -r ${REQUIREMENTS_FILE}${NC}"
    exit 1
fi

echo -e "${BLUE}Installing additional RAG and MCP dependencies...${NC}"
if ${PYTHON_CMD} -m pip install qdrant-client "sentence-transformers[onnx]" onnxruntime --quiet; then
    echo -e "${GREEN}✓ Additional dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Failed to install some additional dependencies${NC}"
    echo -e "${YELLOW}You may need to install them manually later.${NC}"
fi
echo ""

# ============================================================================
# Step 3: Check/start Qdrant
# ============================================================================
echo -e "${YELLOW}[3/5] Setting up Qdrant...${NC}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker Desktop.${NC}"
    exit 1
fi

# Check if Docker daemon is running
if ! docker ps &> /dev/null; then
    echo -e "${RED}ERROR: Docker daemon not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is available${NC}"

# Check if Qdrant container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
    echo -e "${BLUE}Qdrant container '${QDRANT_CONTAINER_NAME}' exists${NC}"

    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        echo -e "${GREEN}✓ Qdrant container is already running${NC}"
    else
        echo -e "${BLUE}Starting existing Qdrant container...${NC}"
        if docker start "${QDRANT_CONTAINER_NAME}" &> /dev/null; then
            echo -e "${GREEN}✓ Qdrant container started${NC}"
        else
            echo -e "${RED}ERROR: Failed to start Qdrant container${NC}"
            exit 1
        fi
    fi
else
    echo -e "${BLUE}Creating new Qdrant container...${NC}"

    # Create and start Qdrant container
    if docker run -d --name "${QDRANT_CONTAINER_NAME}" \
        -p "${QDRANT_PORT}:6333" \
        -v "${QDRANT_VOLUME}:/qdrant/storage" \
        --restart always \
        --memory=4g \
        --cpus=4 \
        qdrant/qdrant:latest &> /dev/null; then
        echo -e "${GREEN}✓ Qdrant container created and started${NC}"
    else
        echo -e "${RED}ERROR: Failed to create Qdrant container${NC}"
        exit 1
    fi
fi

# Wait for Qdrant to be ready
echo -e "${BLUE}Waiting for Qdrant to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [[ ${RETRY_COUNT} -lt ${MAX_RETRIES} ]]; do
    if curl -s --max-time 2 "http://localhost:${QDRANT_PORT}" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Qdrant is accessible on port ${QDRANT_PORT}${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

if [[ ${RETRY_COUNT} -eq ${MAX_RETRIES} ]]; then
    echo -e "${RED}ERROR: Qdrant did not become ready within 30 seconds${NC}"
    echo -e "${RED}Check Docker logs: docker logs ${QDRANT_CONTAINER_NAME}${NC}"
    exit 1
fi

echo -e "${BLUE}Qdrant dashboard: http://localhost:${QDRANT_PORT}${NC}"
echo ""

# ============================================================================
# Step 4: Create log directories
# ============================================================================
echo -e "${YELLOW}[4/5] Creating log directories...${NC}"

if [[ -d "${LOG_DIR}" ]]; then
    echo -e "${GREEN}✓ Log directory already exists: ${LOG_DIR}${NC}"
else
    if mkdir -p "${LOG_DIR}"; then
        echo -e "${GREEN}✓ Created log directory: ${LOG_DIR}${NC}"
    else
        echo -e "${RED}ERROR: Failed to create log directory: ${LOG_DIR}${NC}"
        exit 1
    fi
fi

# Create a test log file to verify write access
if touch "${LOG_DIR}/.test" 2>/dev/null; then
    rm -f "${LOG_DIR}/.test"
    echo -e "${GREEN}✓ Log directory is writable${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Log directory may not be writable${NC}"
fi
echo ""

# ============================================================================
# Step 5: Validate configuration
# ============================================================================
echo -e "${YELLOW}[5/5] Validating configuration...${NC}"

# Add src to Python path for imports
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

# Test Python imports
echo -e "${BLUE}Testing Python imports...${NC}"

IMPORT_TEST=$(${PYTHON_CMD} << 'EOF'
import sys
try:
    import dytopo
    print(f"dytopo: OK (from {dytopo.__file__})")
except ImportError as e:
    print(f"dytopo: FAILED ({e})")
    sys.exit(1)

try:
    from dytopo.config import load_config
    print("dytopo.config: OK")
except ImportError as e:
    print(f"dytopo.config: FAILED ({e})")
    sys.exit(1)

try:
    import openai
    print(f"openai: OK (version {openai.__version__})")
except ImportError as e:
    print(f"openai: FAILED ({e})")
    sys.exit(1)

try:
    import qdrant_client
    print(f"qdrant_client: OK (version {qdrant_client.__version__})")
except ImportError as e:
    print(f"qdrant_client: FAILED ({e})")
    sys.exit(1)

try:
    import yaml
    print("pyyaml: OK")
except ImportError as e:
    print(f"pyyaml: FAILED ({e})")
    sys.exit(1)
EOF
)

if [[ $? -eq 0 ]]; then
    echo "${IMPORT_TEST}" | while IFS= read -r line; do
        echo -e "${GREEN}  ✓ ${line}${NC}"
    done
else
    echo -e "${RED}${IMPORT_TEST}${NC}"
    echo -e "${RED}ERROR: Python import validation failed${NC}"
    exit 1
fi

# Test configuration loading
echo -e "${BLUE}Testing configuration loading...${NC}"

CONFIG_TEST=$(${PYTHON_CMD} << 'EOF'
import sys
try:
    from dytopo.config import load_config
    config = load_config()
    print(f"Backend: {config['concurrency']['backend']}")
    print(f"LLM URL: {config['llm']['base_url']}")
    print(f"Model: {config['llm']['model']}")
    print(f"Max Concurrent: {config['concurrency']['max_concurrent']}")
    print(f"Log Dir: {config['logging']['log_dir']}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
EOF
)

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓ Configuration loaded successfully${NC}"
    echo "${CONFIG_TEST}" | while IFS= read -r line; do
        echo -e "  ${line}"
    done
else
    echo -e "${RED}${CONFIG_TEST}${NC}"
    echo -e "${RED}ERROR: Configuration validation failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Bootstrap Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo -e "  ✓ Python dependencies installed"
echo -e "  ✓ Qdrant running on port ${QDRANT_PORT}"
echo -e "  ✓ Log directory created: ${LOG_DIR}"
echo -e "  ✓ Configuration validated"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Start Docker stack: ${YELLOW}docker compose up -d${NC}"
echo -e "  2. Validate full stack: ${YELLOW}bash scripts/dev_run.sh${NC}"
echo -e "  3. Initialize Qdrant collection: ${YELLOW}python src/reindex.py${NC}"
echo ""
echo -e "${BLUE}Access points:${NC}"
echo -e "  Qdrant Dashboard: ${YELLOW}http://localhost:${QDRANT_PORT}${NC}"
echo -e "  LLM API (after setup): ${YELLOW}http://localhost:8008/v1${NC}"
echo ""
