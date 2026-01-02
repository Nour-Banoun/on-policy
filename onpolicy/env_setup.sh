#!/usr/bin/env bash
set -euo pipefail

# env_setup.sh - install Miniconda (if missing), create conda env from environment.yaml and activate it.
# Place this script next to environment.yaml and run it as:
#   source ./env_setup.sh
# (activation only persists if sourced)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/environment.yaml"
MINICONDA_DIR="$HOME/miniconda3"
INSTALLER="/tmp/miniconda_installer.sh"

# helper: detect platform for Miniconda installer
detect_installer_url() {
    local os arch url
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Linux) os="Linux" ;;
        Darwin) os="MacOSX" ;;
        *) echo "Unsupported OS: $os" >&2; return 1 ;;
    esac
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "Unsupported arch: $arch" >&2; return 1 ;;
    esac
    if [[ "$os" == "MacOSX" && "$arch" == "arm64" ]]; then
        url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        url="https://repo.anaconda.com/miniconda/Miniconda3-latest-${os}-${arch}.sh"
    fi
    echo "$url"
}

# 1) Ensure environment.yaml exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment.yaml not found at: $ENV_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

# 2) Install Miniconda if conda is missing
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found — installing Miniconda to $MINICONDA_DIR ..."
    URL="$(detect_installer_url)"
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        echo "curl or wget is required to download Miniconda." >&2
        return 1 2>/dev/null || exit 1
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "$INSTALLER"
    else
        wget -qO "$INSTALLER" "$URL"
    fi
    chmod +x "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$INSTALLER"
fi

# 3) Initialize conda for this shell session
if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
    # prefer explicit conda.sh so activation works in scripts/sourced shells
    # shellcheck source=/dev/null
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
    export PATH="$MINICONDA_DIR/bin:$PATH"
fi

# 4) Create or update environment from environment.yaml
# Try to get env name from file (if present)
ENV_NAME="$(grep -m1 '^name:' "$ENV_FILE" | sed -E 's/name:[[:space:]]*//g' || true)"
if [[ -n "$ENV_NAME" ]]; then
    if conda env list | awk '{print $1}' | grep -xq "$ENV_NAME"; then
        echo "Updating existing conda environment: $ENV_NAME"
        conda env update -f "$ENV_FILE" --prune
    else
        echo "Creating conda environment: $ENV_NAME"
        conda env create -f "$ENV_FILE"
    fi
else
    echo "No 'name:' found in environment.yaml — creating environment as specified in file"
    conda env create -f "$ENV_FILE" || { echo "Environment create failed" >&2; return 1 2>/dev/null || exit 1; }
fi

# 5) Activate the environment (works only if this script is sourced)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    if [[ -n "$ENV_NAME" ]]; then
        conda activate "$ENV_NAME"
    else
        # try to parse env name from conda env created (fallback)
        echo "Environment created. Please run 'conda activate <env-name>' to activate it."
    fi
else
    if [[ -n "$ENV_NAME" ]]; then
        echo "To activate the environment in your current shell run:"
        echo "  source \"$SCRIPT_DIR/$(basename "$0")\""
        echo "or run:"
        echo "  conda activate $ENV_NAME"
    else
        echo "Environment created. Run 'conda env list' to see name and activate it."
    fi
fi