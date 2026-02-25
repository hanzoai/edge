#!/bin/sh
# Hanzo Edge installer
# Usage: curl -sSL https://edge.hanzo.ai/install.sh | sh
set -e

REPO="hanzoai/edge"
INSTALL_DIR="$HOME/.hanzo/bin"
BINARY="hanzo-edge"

# --- Helpers ---

info() {
  printf '  \033[1m%s\033[0m\n' "$1"
}

err() {
  printf '  \033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

# --- Detect platform ---

detect_os() {
  case "$(uname -s)" in
    Linux*)   echo "linux" ;;
    Darwin*)  echo "darwin" ;;
    MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
    *)        err "Unsupported operating system: $(uname -s)" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)   echo "x86_64" ;;
    aarch64|arm64)   echo "aarch64" ;;
    *)               err "Unsupported architecture: $(uname -m)" ;;
  esac
}

# --- Fetch latest release tag ---

latest_tag() {
  if command -v curl >/dev/null 2>&1; then
    curl -sSf "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' \
      | head -1 \
      | sed 's/.*"tag_name": *"//;s/".*//'
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
      | grep '"tag_name"' \
      | head -1 \
      | sed 's/.*"tag_name": *"//;s/".*//'
  else
    err "Neither curl nor wget found. Install one and retry."
  fi
}

# --- Download ---

download() {
  url="$1"
  dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -sSfL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dest"
  else
    err "Neither curl nor wget found."
  fi
}

# --- Main ---

main() {
  printf '\n'
  info "Hanzo Edge Installer"
  printf '\n'

  OS="$(detect_os)"
  ARCH="$(detect_arch)"
  info "Detected: ${OS} / ${ARCH}"

  TAG="$(latest_tag)"
  if [ -z "$TAG" ]; then
    err "Could not determine latest release. Check https://github.com/${REPO}/releases"
  fi
  info "Latest release: ${TAG}"

  # Construct download URL
  # Release asset naming: hanzo-edge-{tag}-{os}-{arch}.tar.gz
  ASSET="${BINARY}-${TAG}-${OS}-${ARCH}.tar.gz"
  URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

  TMPDIR="$(mktemp -d)"
  ARCHIVE="${TMPDIR}/${ASSET}"

  info "Downloading ${URL}"
  download "$URL" "$ARCHIVE"

  # Extract
  info "Extracting to ${INSTALL_DIR}"
  mkdir -p "$INSTALL_DIR"
  tar -xzf "$ARCHIVE" -C "$TMPDIR"

  # Find the binary in extracted contents
  if [ -f "${TMPDIR}/${BINARY}" ]; then
    mv "${TMPDIR}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
  elif [ -f "${TMPDIR}/${BINARY}-${TAG}-${OS}-${ARCH}/${BINARY}" ]; then
    mv "${TMPDIR}/${BINARY}-${TAG}-${OS}-${ARCH}/${BINARY}" "${INSTALL_DIR}/${BINARY}"
  else
    # Try to find it anywhere in the temp dir
    FOUND="$(find "$TMPDIR" -name "$BINARY" -type f | head -1)"
    if [ -n "$FOUND" ]; then
      mv "$FOUND" "${INSTALL_DIR}/${BINARY}"
    else
      err "Could not locate ${BINARY} binary in release archive."
    fi
  fi

  chmod +x "${INSTALL_DIR}/${BINARY}"

  # Cleanup
  rm -rf "$TMPDIR"

  # Check PATH
  case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
      info ""
      info "Add Hanzo Edge to your PATH:"
      printf '\n'
      printf '    export PATH="%s:$PATH"\n' "$INSTALL_DIR"
      printf '\n'
      info "Add that line to your ~/.bashrc, ~/.zshrc, or equivalent."
      ;;
  esac

  # Verify
  if [ -x "${INSTALL_DIR}/${BINARY}" ]; then
    printf '\n'
    info "Installed: ${INSTALL_DIR}/${BINARY}"
    VERSION="$("${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || echo "${TAG}")"
    info "Version: ${VERSION}"
  else
    err "Installation failed."
  fi

  printf '\n'
  info "Run your first model:"
  printf '\n'
  printf '    %s run zenlm/zen4-mini-gguf -p "Hello, world!"\n' "$BINARY"
  printf '\n'
}

main
