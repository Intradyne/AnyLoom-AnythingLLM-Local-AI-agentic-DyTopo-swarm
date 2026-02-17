#!/usr/bin/env bash
# install_skills.sh — Install AnyLoom custom agent skills into AnythingLLM container
# Usage: bash scripts/install_skills.sh
#
# NOTE: This script is also integrated into configure_anythingllm.py.
# You can run either this script standalone or the full configure script:
#   python scripts/configure_anythingllm.py

set -euo pipefail

CONTAINER="anyloom-anythingllm"
SKILLS_DEST="/app/server/storage/plugins/agent-skills"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_SRC="$PROJECT_ROOT/skills"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo " AnyLoom Agent Skills Installer"
echo "========================================="
echo ""

# --- Step 1: Check if container is running ---
echo -e "${YELLOW}[1/4]${NC} Checking if container '${CONTAINER}' is running..."

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo -e "${RED}ERROR:${NC} Container '${CONTAINER}' is not running."
    echo ""
    echo "Start it with:"
    echo "  docker compose up -d"
    echo ""
    echo "Or check container name with:"
    echo "  docker ps --format '{{.Names}}'"
    exit 1
fi

echo -e "${GREEN}OK${NC} — Container '${CONTAINER}' is running."
echo ""

# --- Step 2: Copy skill directories into container ---
echo -e "${YELLOW}[2/4]${NC} Copying skill directories into container..."

for skill_dir in "$SKILLS_SRC"/*/; do
    skill_name="$(basename "$skill_dir")"

    if [ ! -f "$skill_dir/plugin.json" ] || [ ! -f "$skill_dir/handler.js" ]; then
        echo -e "  ${YELLOW}SKIP${NC} — ${skill_name} (missing plugin.json or handler.js)"
        continue
    fi

    echo "  Copying ${skill_name}..."
    docker exec "${CONTAINER}" rm -rf "${SKILLS_DEST}/${skill_name}" 2>/dev/null || true
    docker cp "$skill_dir" "${CONTAINER}:${SKILLS_DEST}/${skill_name}"
done

echo -e "${GREEN}OK${NC} — Skills copied."
echo ""

# --- Step 3: Install npm dependencies inside container ---
echo -e "${YELLOW}[3/4]${NC} Installing npm dependencies inside container..."

docker exec "${CONTAINER}" npm install \
    --prefix /app/server --legacy-peer-deps \
    yahoo-finance2@3 defuddle jsdom @mozilla/readability turndown \
    2>&1 | tail -5

echo -e "${GREEN}OK${NC} — Dependencies installed."
echo ""

# --- Step 4: Verify skill directories exist in container ---
echo -e "${YELLOW}[4/4]${NC} Verifying installed skills..."

VERIFY_FAILED=0

for skill_dir in "$SKILLS_SRC"/*/; do
    skill_name="$(basename "$skill_dir")"

    if [ ! -f "$skill_dir/plugin.json" ]; then
        continue
    fi

    if docker exec "${CONTAINER}" test -f "${SKILLS_DEST}/${skill_name}/plugin.json"; then
        echo -e "  ${GREEN}OK${NC}  ${skill_name}/plugin.json"
    else
        echo -e "  ${RED}FAIL${NC}  ${skill_name}/plugin.json not found in container"
        VERIFY_FAILED=1
    fi

    if docker exec "${CONTAINER}" test -f "${SKILLS_DEST}/${skill_name}/handler.js"; then
        echo -e "  ${GREEN}OK${NC}  ${skill_name}/handler.js"
    else
        echo -e "  ${RED}FAIL${NC}  ${skill_name}/handler.js not found in container"
        VERIFY_FAILED=1
    fi
done

echo ""

if [ "$VERIFY_FAILED" -eq 1 ]; then
    echo -e "${RED}Some skills failed verification. Check the output above.${NC}"
    exit 1
fi

# --- Done ---
echo "========================================="
echo -e "${GREEN} Skills installed successfully!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Open AnythingLLM in your browser"
echo "  2. Go to Workspace Settings > Agent Configuration > Custom Skills"
echo "  3. Enable the installed skills"
echo "  4. Start chatting with @agent to use the skills"
echo ""
