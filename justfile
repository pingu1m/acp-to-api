# acp-to-api — all CLI tools run via uvx (no install needed)

# Shared list of test dependencies for uvx pytest invocations
_test_deps := "--from . --with pytest --with pytest-asyncio --with httpx --with openai --with 'ag2[openai]' --with langchain --with langchain-openai --with pydantic-ai"

# Install dependencies (only needed for local dev)
install:
    uv sync --dev

# Lint with ruff
lint:
    uvx ruff check .

# Format with ruff
fmt:
    uvx ruff format .

# Check lint + format (CI-style, no changes)
check:
    uvx ruff check .
    uvx ruff format --check .

# Start the server with TOML config
serve:
    uvx acp-to-api serve --config acp-to-api.toml

# Start with raw ACP + REST logging
serve-debug:
    PYTHONLOGLEVEL=DEBUG uvx acp-to-api serve --config acp-to-api.toml --raw-acp --raw-rest

# Start with inline provider JSON
serve-inline provider='{"name":"cursor","command":"agent","args":["acp"]}':
    uvx acp-to-api serve --provider '{{ provider }}'

# Run E2E tests (basic)
test-e2e:
    uvx {{ _test_deps }} pytest tests/test_e2e_cursor.py -q

# Run LangChain agentic E2E tests
test-langchain:
    uvx {{ _test_deps }} pytest tests/test_e2e_langchain_agent.py -v

# Run all tests
test:
    uvx {{ _test_deps }} pytest -q

# Run LangGraph experiment (interactive model selection)
exp-langgraph question='Is Rust a good choice for agentic coding?' backend='cursor' base_url='':
    bash -lc 'set -euo pipefail; QUESTION="$1"; BACKEND="$2"; BASE_URL="$3"; if [ -z "$BASE_URL" ]; then BASE_URL="http://127.0.0.1:11434/api/v1/${BACKEND}/openai"; fi; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-exp-langgraph.log 2>&1 & SERVER_PID=$!; STARTED=1; cleanup() { if [ "$STARTED" -eq 1 ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-exp-langgraph.log" >&2; exit 1; fi; fi; uv run experiments/langgraph_basic_research.py --question "$QUESTION" --backend "$BACKEND" --base-url "$BASE_URL"' -- '{{ question }}' '{{ backend }}' '{{ base_url }}'

# Convenience wrappers for explicit backend runs
exp-langgraph-cursor question='Is Rust a good choice for agentic coding?':
    just exp-langgraph '{{ question }}' 'cursor' ''

exp-langgraph-kiro question='Is Rust a good choice for agentic coding?':
    just exp-langgraph '{{ question }}' 'kiro' ''

# Run LangGraph experiment with explicit model IDs (non-interactive)
exp-langgraph-models sonnet opus question='Is Rust a good choice for agentic coding?' backend='cursor' base_url='':
    bash -lc 'set -euo pipefail; SONNET="$1"; OPUS="$2"; QUESTION="$3"; BACKEND="$4"; BASE_URL="$5"; if [ -z "$BASE_URL" ]; then BASE_URL="http://127.0.0.1:11434/api/v1/${BACKEND}/openai"; fi; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-exp-langgraph-models.log 2>&1 & SERVER_PID=$!; STARTED=1; cleanup() { if [ "$STARTED" -eq 1 ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-exp-langgraph-models.log" >&2; exit 1; fi; fi; uv run experiments/langgraph_basic_research.py --question "$QUESTION" --backend "$BACKEND" --base-url "$BASE_URL" --sonnet-model "$SONNET" --opus-model "$OPUS" --non-interactive' -- '{{ sonnet }}' '{{ opus }}' '{{ question }}' '{{ backend }}' '{{ base_url }}'

# Run PydanticAI experiment (interactive model selection)
exp-pydantic-ai question='Is Rust a good choice for agentic coding?' base_url='http://127.0.0.1:11434/api/v1/cursor/openai':
    uv run experiments/pydantic_ai_basic_research.py --question '{{ question }}' --base-url '{{ base_url }}'

# Run PydanticAI experiment with explicit model IDs (non-interactive)
exp-pydantic-ai-models sonnet opus question='Is Rust a good choice for agentic coding?' base_url='http://127.0.0.1:11434/api/v1/cursor/openai':
    uv run experiments/pydantic_ai_basic_research.py --question '{{ question }}' --base-url '{{ base_url }}' --sonnet-model '{{ sonnet }}' --opus-model '{{ opus }}' --non-interactive

# Run Claude Code SDK hello-world experiment
exp-claude-sdk-hello base_url='http://127.0.0.1:11434/api/v1/cursor/anthropic' model='sonnet':
    bash -lc 'set -euo pipefail; BASE_URL="$1"; MODEL="$2"; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-exp-claude-sdk.log 2>&1 & SERVER_PID=$!; STARTED=1; cleanup() { if [ "$STARTED" -eq 1 ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-exp-claude-sdk.log" >&2; exit 1; fi; fi; uv run experiments/claude_code_sdk_hello_world.py --base-url "$BASE_URL" --model "$MODEL"' -- '{{ base_url }}' '{{ model }}'

# Run OpenAI SDK image-explain experiment (downloads public image, uploads as data URL)
exp-openai-image-explain image_url='https://raw.githubusercontent.com/github/explore/main/topics/python/python.png' base_url='http://127.0.0.1:11434/api/v1/cursor/openai' model='auto':
    bash -lc 'set -euo pipefail; IMAGE_URL="$1"; BASE_URL="$2"; MODEL="$3"; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-exp-openai-image.log 2>&1 & SERVER_PID=$!; STARTED=1; cleanup() { if [ "$STARTED" -eq 1 ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-exp-openai-image.log" >&2; exit 1; fi; fi; uv run experiments/openai_image_explain.py --image-url "$IMAGE_URL" --base-url "$BASE_URL" --model "$MODEL"' -- '{{ image_url }}' '{{ base_url }}' '{{ model }}'

# Run OpenAI SDK PDF-explain experiment (downloads public PDF, uploads as data URL)
exp-openai-pdf-explain pdf_url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf' base_url='http://127.0.0.1:11434/api/v1/cursor/openai' model='auto':
    bash -lc 'set -euo pipefail; PDF_URL="$1"; BASE_URL="$2"; MODEL="$3"; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-exp-openai-pdf.log 2>&1 & SERVER_PID=$!; STARTED=1; cleanup() { if [ "$STARTED" -eq 1 ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-exp-openai-pdf.log" >&2; exit 1; fi; fi; uv run experiments/openai_pdf_explain.py --pdf-url "$PDF_URL" --base-url "$BASE_URL" --model "$MODEL"' -- '{{ pdf_url }}' '{{ base_url }}' '{{ model }}'

# Smoke test Codex CLI against local backend
smoke-codex base_url='http://127.0.0.1:11434/api/v1/cursor/openai':
    bash -lc 'set -euo pipefail; BASE_URL="$1"; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; TMP_HOME="$(mktemp -d)"; cleanup() { rm -rf "$TMP_HOME"; if [ "$STARTED" -eq 1 ] && [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-smoke-codex.log 2>&1 & SERVER_PID=$!; STARTED=1; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-smoke-codex.log" >&2; exit 1; fi; fi; printf "%s\n" "[profiles.acp_local]" "model = \"cursor\"" "model_provider = \"acp_local\"" "approval_policy = \"never\"" "" "[model_providers.acp_local]" "name = \"ACP Local\"" "base_url = \"$BASE_URL\"" "env_key = \"OPENAI_API_KEY\"" "wire_api = \"responses\"" > "$TMP_HOME/config.toml"; OUTPUT="$(printf "Reply with exactly: codex-acp-ok\n" | CODEX_HOME="$TMP_HOME" OPENAI_API_KEY=dummy codex -p acp_local -a never -s read-only exec --skip-git-repo-check -C "$(pwd)" - 2>/tmp/codex-smoke.stderr)"; printf "%s\n" "$OUTPUT"; printf "%s\n" "$OUTPUT" | rg -q "codex-acp-ok"' -- '{{ base_url }}'
    echo "Codex smoke test passed."

# Smoke test Claude Code CLI against local backend
smoke-claude base_url='http://127.0.0.1:11434/api/v1/cursor/anthropic':
    bash -lc 'set -euo pipefail; BASE_URL="$1"; HEALTH_URL="${BASE_URL%%/api/v1/*}/health"; STARTED=0; cleanup() { if [ "$STARTED" -eq 1 ] && [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }; trap cleanup EXIT INT TERM; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Starting local acp-to-api server..."; uvx acp-to-api serve --config acp-to-api.toml >/tmp/acp-to-api-smoke-claude.log 2>&1 & SERVER_PID=$!; STARTED=1; for i in $(seq 1 30); do if curl -fsS "$HEALTH_URL" >/dev/null; then break; fi; sleep 1; done; if ! curl -fsS "$HEALTH_URL" >/dev/null; then echo "Failed to start acp-to-api. See /tmp/acp-to-api-smoke-claude.log" >&2; exit 1; fi; fi; OUTPUT="$(printf "Reply with exactly: claude-acp-ok\n" | ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL="$BASE_URL" claude -p --model sonnet --permission-mode dontAsk 2>/tmp/claude-smoke.stderr)"; printf "%s\n" "$OUTPUT"; printf "%s\n" "$OUTPUT" | rg -q "^claude-acp-ok$"' -- '{{ base_url }}'
    echo "Claude smoke test passed."

# Smoke test both Codex and Claude CLIs
smoke-clis:
    just smoke-codex
    just smoke-claude

# -- Daemon management --------------------------------------------------------

# Start the daemon (background)
daemon-start:
    uvx acp-to-api start --config acp-to-api.toml

# Stop the daemon
daemon-stop:
    uvx acp-to-api stop

# Show daemon status
daemon-status:
    uvx acp-to-api status

# Restart the daemon
daemon-restart:
    uvx acp-to-api restart --config acp-to-api.toml

# Install as system service (auto-start on boot)
setup:
    uvx acp-to-api setup --config acp-to-api.toml

# Remove system service
setup-uninstall:
    uvx acp-to-api setup --uninstall

# -- Testing ------------------------------------------------------------------

# Run daemon E2E tests
test-daemon:
    uvx {{ _test_deps }} pytest tests/test_e2e_daemon.py -v

# Run all tests including daemon
test-all:
    uvx {{ _test_deps }} pytest -v

# -- Build --------------------------------------------------------------------

# Build the package
build:
    uv build

# Publish to PyPI
publish: build
    uv publish
