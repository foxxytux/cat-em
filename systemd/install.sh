#!/bin/bash
set -euo pipefail
# CodeAgent-RWKV installer — systemd user service
# Run: bash install.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_BIN="${HOME}/.local/bin"
INSTALL_SHARE="${HOME}/.local/share/codeagent"
INSTALL_SYSTEMD="${HOME}/.config/systemd/user"

echo "=== CodeAgent-RWKV Installer ==="

mkdir -p "${INSTALL_BIN}" "${INSTALL_SHARE}" "${INSTALL_SYSTEMD}"

# Copy server
echo "Installing server to ${INSTALL_BIN}/codeagent-server"
cp "${PROJECT_DIR}/api/python/server.py" "${INSTALL_BIN}/codeagent-server"
chmod +x "${INSTALL_BIN}/codeagent-server"

# Copy chatbot CLI
echo "Installing CLI to ${INSTALL_BIN}/codeagent-chat"
cp "${PROJECT_DIR}/chatbot/cli.py" "${INSTALL_BIN}/codeagent-chat"
chmod +x "${INSTALL_BIN}/codeagent-chat"

# Copy checkpoint if available
if [ -d "${PROJECT_DIR}/checkpoints" ] && [ -f "${PROJECT_DIR}/checkpoints/model.safetensors" ]; then
    echo "Copying checkpoint..."
    cp -r "${PROJECT_DIR}/checkpoints" "${INSTALL_SHARE}/"
fi

# Install systemd service
echo "Installing systemd user service..."
cp "${SCRIPT_DIR}/codeagent.service" "${INSTALL_SYSTEMD}/"
systemctl --user daemon-reload
systemctl --user enable codeagent

echo ""
echo "Done! Commands:"
echo "  systemctl --user start codeagent     # Start server"
echo "  systemctl --user stop codeagent      # Stop server"
echo "  journalctl --user -u codeagent -f    # View logs"
echo "  codeagent-chat                       # CLI chatbot"
echo ""
echo "API: http://127.0.0.1:8080"
echo "Docs: http://127.0.0.1:8080/docs"
