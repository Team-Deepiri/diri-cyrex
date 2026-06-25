#!/usr/bin/env bash
set -euo pipefail

# OS-level dependencies for diri-cyrex (local dev and Docker image builds).
# Python packages: Poetry (`pyproject.toml`). Optional host install:
#   INSTALL_PYTHON_DEPS=1 ./setup.sh
#   POETRY_DEVICE_EXTRA=gpu|rocm|mps|cpu|auto ./setup.sh

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
        exit 0
        ;;
esac

echo "setup.sh: system dependencies ready."

install_python_deps() {
    if [ "${INSTALL_PYTHON_DEPS:-0}" != "1" ]; then
        echo "setup.sh: skip Python deps (set INSTALL_PYTHON_DEPS=1 to run poetry install)."
        return 0
    fi
    if ! command -v poetry >/dev/null 2>&1; then
        echo "setup.sh: poetry not found; install Poetry 1.8+ then re-run." >&2
        return 1
    fi
    local script_dir extra
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    (cd "$script_dir" && poetry install --no-ansi --extras "$extra")
}

install_python_deps
