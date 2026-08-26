#!/usr/bin/env bash
# =============================================================================
# diri-cyrex standalone setup
#
# Default: install OS-level deps for local Cyrex builds.
# --run:           Cyrex engine + cyrex-interface + messaging + realtime-gateway
#                  (live chat / Socket.IO path usable from the interface)
# --run --headless: engine only — no messaging / RTG / api-gateway
#
# Platform root is resolved as:
#   1) $DEEPIRI_PLATFORM_ROOT if set
#   2) parent repo (../docker-compose.dev.yml) when Cyrex is a submodule
#   3) sister repo (../deepiri-platform/docker-compose.dev.yml)
#
# Usage:
#   ./setup.sh
#   ./setup.sh --run
#   ./setup.sh --run --headless
#   ./setup.sh --run --build
#   ./setup.sh --help
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="docker-compose.dev.yml"

# Engine-only (matches HOW_TO_USE_CYREX Option B; postgres → postgres-cyrex).
readonly CYREX_HEADLESS_SERVICES=(
    postgres-cyrex
    redis
    influxdb
    etcd
    minio
    milvus
    cyrex
    cyrex-interface
    ollama
    synapse
    synapse-sugar-glider
)

# Interface + live delivery + speech (livekit/speech need compose defs).
# See docs/architecture/DEEPIRI_SPEECH_INTEGRATION.md
readonly CYREX_RUN_SERVICES=(
    "${CYREX_HEADLESS_SERVICES[@]}"
    postgres-core
    api-gateway
    messaging-service
    realtime-gateway
    livekit
    speech
)

DO_RUN=0
DO_BUILD=0
DO_DEPS=1
DO_HEADLESS=0

usage() {
    cat <<'EOF'
diri-cyrex setup.sh — standalone Cyrex setup / bring-up

Usage:
  ./setup.sh                      Install OS deps (poppler, tesseract, etc.)
  ./setup.sh --run                Engine + interface + messaging + RTG
  ./setup.sh --run --headless     Engine + interface only (no messaging/RTG)
  ./setup.sh --run --build        Build images, then start (works with --headless)
  ./setup.sh --help

Env:
  DEEPIRI_PLATFORM_ROOT   Override path to deepiri-platform
  INSTALL_PYTHON_DEPS=1   Also run poetry install (with --deps / default)
  POETRY_DEVICE_EXTRA     gpu|rocm|mps|cpu|auto (default: auto)

--run looks for deepiri-platform as:
  - parent of this repo (submodule layout), or
  - sister ../deepiri-platform (standalone clone layout)

--run (default):
  headless services plus postgres-core api-gateway messaging-service realtime-gateway
  plus livekit/speech when those services exist in platform compose
  (missing optional services are skipped with a warning; --run still succeeds)

--run --headless:
  postgres-cyrex redis influxdb etcd minio milvus
  cyrex cyrex-interface ollama synapse synapse-sugar-glider
EOF
}

find_platform_root() {
    local candidate

    if [[ -n "${DEEPIRI_PLATFORM_ROOT:-}" ]]; then
        candidate="$(cd "$DEEPIRI_PLATFORM_ROOT" && pwd)"
        if [[ -f "${candidate}/${COMPOSE_FILE}" ]]; then
            echo "$candidate"
            return 0
        fi
        echo "setup.sh: DEEPIRI_PLATFORM_ROOT=${DEEPIRI_PLATFORM_ROOT} has no ${COMPOSE_FILE}" >&2
        return 1
    fi

    # Parent: diri-cyrex is a submodule of deepiri-platform
    candidate="$(cd "${SCRIPT_DIR}/.." && pwd)"
    if [[ -f "${candidate}/${COMPOSE_FILE}" ]]; then
        echo "$candidate"
        return 0
    fi

    # Sister: diri-cyrex and deepiri-platform side-by-side
    candidate="$(cd "${SCRIPT_DIR}/../deepiri-platform" 2>/dev/null && pwd)" || true
    if [[ -n "${candidate:-}" && -f "${candidate}/${COMPOSE_FILE}" ]]; then
        echo "$candidate"
        return 0
    fi

    echo "setup.sh: could not find deepiri-platform (${COMPOSE_FILE})." >&2
    echo "  Expected parent of this repo, or sister ../deepiri-platform," >&2
    echo "  or set DEEPIRI_PLATFORM_ROOT." >&2
    return 1
}

# Return service names defined in the platform compose file (one per line).
list_compose_services() {
    local platform_root="$1"
    (cd "$platform_root" && docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null) || true
}

# Keep only services present in compose; print warnings for skipped ones.
# Optional services (livekit/speech) are skipped with a warning.
# Required services missing from compose cause a hard error.
filter_available_services() {
    local platform_root="$1"
    shift
    local -a requested=("$@")
    local -a available=()
    local -a selected=()
    local -a skipped=()
    local -a missing_required=()
    local svc
    local avail_set

    is_optional_service() {
        case "$1" in
            livekit | speech) return 0 ;;
            *) return 1 ;;
        esac
    }

    mapfile -t available < <(list_compose_services "$platform_root")
    if [[ ${#available[@]} -eq 0 ]]; then
        echo "setup.sh: warning: could not list compose services; starting requested names as-is" >&2
        printf '%s\n' "${requested[@]}"
        return 0
    fi

    avail_set=" $(printf '%s ' "${available[@]}") "
    for svc in "${requested[@]}"; do
        if [[ "$avail_set" == *" ${svc} "* ]]; then
            selected+=("$svc")
        else
            skipped+=("$svc")
            if ! is_optional_service "$svc"; then
                missing_required+=("$svc")
            fi
        fi
    done

    if [[ ${#skipped[@]} -gt 0 ]]; then
        echo "setup.sh: warning: skipping services not defined in ${COMPOSE_FILE}:" >&2
        printf '  - %s\n' "${skipped[@]}" >&2
        if [[ " ${skipped[*]} " == *" livekit "* ]] || [[ " ${skipped[*]} " == *" speech "* ]]; then
            echo "  (livekit/speech are optional until platform compose defines them)" >&2
        fi
    fi

    if [[ ${#missing_required[@]} -gt 0 ]]; then
        echo "setup.sh: error: required services missing from ${COMPOSE_FILE}:" >&2
        printf '  - %s\n' "${missing_required[@]}" >&2
        return 1
    fi

    if [[ ${#selected[@]} -eq 0 ]]; then
        echo "setup.sh: error: no requested services are available in ${COMPOSE_FILE}" >&2
        return 1
    fi

    printf '%s\n' "${selected[@]}"
}

run_cyrex_stack() {
    local platform_root compose_args=()
    local -a services=()
    local -a requested=()
    local mode_label
    local has_livekit=0
    local has_speech=0
    local svc

    if ! command -v docker >/dev/null 2>&1; then
        echo "setup.sh: docker not found." >&2
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "setup.sh: docker compose not found." >&2
        exit 1
    fi

    platform_root="$(find_platform_root)"

    if [[ "$DO_HEADLESS" -eq 1 ]]; then
        requested=("${CYREX_HEADLESS_SERVICES[@]}")
        mode_label="headless (engine + cyrex-interface only)"
    else
        requested=("${CYREX_RUN_SERVICES[@]}")
        mode_label="full (engine + messaging + realtime-gateway; speech if available)"
    fi

    local filtered_output
    if ! filtered_output="$(filter_available_services "$platform_root" "${requested[@]}")"; then
        exit 1
    fi
    mapfile -t services <<< "$filtered_output"
    # Drop empty line mapfile can add when output is empty
    if [[ ${#services[@]} -eq 1 && -z "${services[0]:-}" ]]; then
        services=()
    fi
    if [[ ${#services[@]} -eq 0 ]]; then
        echo "setup.sh: error: no services to start after compose filtering" >&2
        exit 1
    fi

    for svc in "${services[@]}"; do
        [[ "$svc" == "livekit" ]] && has_livekit=1
        [[ "$svc" == "speech" ]] && has_speech=1
    done

    echo "setup.sh: platform root → ${platform_root}"
    echo "setup.sh: starting Cyrex stack — ${mode_label}"
    echo "         services: ${services[*]}"
    echo ""

    compose_args=(-f "$COMPOSE_FILE" up -d)
    if [[ "$DO_BUILD" -eq 1 ]]; then
        compose_args+=(--build)
    else
        compose_args+=(--no-build)
    fi
    # Only start what we list (avoid pulling the entire platform via depends_on).
    compose_args+=(--no-deps "${services[@]}")

    (cd "$platform_root" && docker compose "${compose_args[@]}")

    echo ""
    echo "setup.sh: Cyrex stack up (${mode_label})."
    echo "  Cyrex:           http://localhost:8000"
    echo "  Cyrex Interface: http://localhost:5175"
    echo "  Ollama:          http://localhost:11434"
    echo "  Synapse:         http://localhost:8002"
    if [[ "$DO_HEADLESS" -eq 0 ]]; then
        echo "  Realtime GW:     http://localhost:5008"
        echo "  Messaging:       http://localhost:5010"
        echo "  API Gateway:     http://localhost:5100"
        if [[ "$has_livekit" -eq 1 ]]; then
            echo "  LiveKit:         http://localhost:7880"
        fi
        if [[ "$has_speech" -eq 1 ]]; then
            echo "  Speech:          http://localhost:5020"
        fi
    fi
    echo ""
    echo "  Health: curl -s http://localhost:8000/health"
    echo "  Docs:   http://localhost:8000/docs"
    if [[ "$DO_HEADLESS" -eq 1 ]]; then
        echo ""
        echo "  Tip: omit --headless to also start messaging + realtime-gateway."
    fi
}

# ---------- OS deps (existing behavior) ------------------------------------

readonly CYREX_APT_PACKAGES=(
    curl
    git
    gcc
    g++
    poppler-utils
    tesseract-ocr
)

readonly CYREX_BREW_PACKAGES=(
    curl
    git
    poppler
    tesseract
)

readonly CYREX_APK_PACKAGES=(
    curl
    git
    gcc
    g++
    musl-dev
    poppler-utils
    tesseract-ocr
)

readonly CYREX_DNF_PACKAGES=(
    curl
    git
    gcc
    gcc-c++
    poppler-utils
    tesseract
)

detect_platform() {
    case "$(uname -s)" in
        Linux*)
            if [ -f /etc/alpine-release ]; then
                echo alpine
            elif command -v apt-get >/dev/null 2>&1; then
                echo debian
            elif command -v dnf >/dev/null 2>&1; then
                echo fedora
            elif command -v yum >/dev/null 2>&1; then
                echo rhel
            else
                echo linux-unknown
            fi
            ;;
        Darwin*)
            echo macos
            ;;
        *)
            echo unknown
            ;;
    esac
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "setup.sh: root privileges required to install system packages." >&2
        exit 1
    fi
}

install_debian() {
    echo "setup.sh: installing Debian/Ubuntu packages..."
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends "${CYREX_APT_PACKAGES[@]}"
    if [ "$(id -u)" -eq 0 ]; then
        rm -rf /var/lib/apt/lists/*
    fi
}

install_alpine() {
    echo "setup.sh: installing Alpine packages..."
    run_as_root apk add --no-cache "${CYREX_APK_PACKAGES[@]}"
}

install_fedora() {
    echo "setup.sh: installing Fedora packages..."
    run_as_root dnf install -y "${CYREX_DNF_PACKAGES[@]}"
}

install_rhel() {
    echo "setup.sh: installing RHEL/CentOS packages..."
    run_as_root yum install -y "${CYREX_DNF_PACKAGES[@]}"
}

install_macos() {
    if ! command -v brew >/dev/null 2>&1; then
        echo "setup.sh: Homebrew not found. Install from https://brew.sh then re-run." >&2
        exit 1
    fi
    echo "setup.sh: installing macOS packages via Homebrew..."
    brew install "${CYREX_BREW_PACKAGES[@]}"
}

install_python_deps() {
    if [ "${INSTALL_PYTHON_DEPS:-0}" != "1" ]; then
        echo "setup.sh: skip Python deps (set INSTALL_PYTHON_DEPS=1 to run poetry install)."
        return 0
    fi
    if ! command -v poetry >/dev/null 2>&1; then
        echo "setup.sh: poetry not found; install Poetry 1.8+ then re-run." >&2
        return 1
    fi
    local extra
    extra="${POETRY_DEVICE_EXTRA:-auto}"
    if [ "$extra" = "auto" ]; then
        if command -v deepiri-gpu >/dev/null 2>&1; then
            extra="$(deepiri-gpu detect --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('backend','cpu'))" 2>/dev/null || echo cpu)"
        else
            extra="cpu"
        fi
    fi
    case "$extra" in
        cuda | nvidia | gpu) extra="gpu" ;;
        amd | rocm) extra="rocm" ;;
        mps | macos | darwin) extra="mps" ;;
        cpu | *) extra="cpu" ;;
    esac
    echo "setup.sh: poetry install --extras ${extra}"
    (cd "$SCRIPT_DIR" && poetry install --no-ansi --extras "$extra")
}

install_system_deps() {
    local platform
    platform="$(detect_platform)"
    echo "setup.sh: detected platform: ${platform}"

    case "${platform}" in
        debian) install_debian ;;
        alpine) install_alpine ;;
        fedora) install_fedora ;;
        rhel) install_rhel ;;
        macos) install_macos ;;
        linux-unknown | unknown)
            echo "setup.sh: unsupported platform; install these manually if needed:"
            printf '  - %s\n' "${CYREX_APT_PACKAGES[@]}"
            return 0
            ;;
    esac

    echo "setup.sh: system dependencies ready."
    install_python_deps
}

# ---------- args -----------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            DO_RUN=1
            DO_DEPS=0
            shift
            ;;
        --headless)
            DO_HEADLESS=1
            shift
            ;;
        --build)
            DO_BUILD=1
            shift
            ;;
        --deps)
            DO_DEPS=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "setup.sh: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$DO_HEADLESS" -eq 1 && "$DO_RUN" -eq 0 ]]; then
    echo "setup.sh: --headless requires --run" >&2
    usage >&2
    exit 1
fi

if [[ "$DO_RUN" -eq 1 ]]; then
    run_cyrex_stack
fi

if [[ "$DO_DEPS" -eq 1 ]]; then
    install_system_deps
fi
