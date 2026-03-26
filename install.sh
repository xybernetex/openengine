#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenEngine — Ubuntu install script
# Usage: curl -fsSL https://raw.githubusercontent.com/xybernetex/openengine/main/install.sh | bash
# Or:    chmod +x install.sh && ./install.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="https://github.com/xybernetex/openengine.git"
INSTALL_DIR="/opt/openengine"
SERVICE_USER="openengine"
SERVICE_NAME="openengine"
WEIGHTS_URL=""   # set this once weights are hosted publicly

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▶ $*${RESET}"; }
success() { echo -e "${GREEN}✔ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
die()     { echo -e "${RED}✖ $*${RESET}"; exit 1; }

echo -e "${BOLD}"
echo "  ╔═══════════════════════════════════╗"
echo "  ║      OpenEngine Installer         ║"
echo "  ╚═══════════════════════════════════╝"
echo -e "${RESET}"

# ── Root check ───────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && die "Run as root: sudo bash install.sh"

# ── Collect config ───────────────────────────────────────────────────────────
read -rp "$(echo -e ${BOLD})Domain (e.g. api.yourdomain.com): $(echo -e ${RESET})" DOMAIN
[[ -z "$DOMAIN" ]] && die "Domain is required."

echo ""
info "LLM fallback credentials — used when callers don't pass their own keys."
info "Leave blank to require callers to always supply keys per-request."
echo ""
read -rp "  LLM Provider (cloudflare/openai/anthropic/gemini/mistral) [cloudflare]: " LLM_PROVIDER
LLM_PROVIDER="${LLM_PROVIDER:-cloudflare}"
read -rp "  LLM API Key (leave blank for Cloudflare): " LLM_API_KEY
read -rp "  CF Account ID (Cloudflare only): " CF_ACCOUNT_ID
read -rp "  CF API Token  (Cloudflare only): " CF_API_TOKEN
read -rp "  Tavily API Key (web search fallback): " TAVILY_API_KEY
read -rp "  Resend API Key (email fallback): " RESEND_API_KEY

# Webhook secret — auto-generate
WEBHOOK_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)

echo ""

# ── System packages ──────────────────────────────────────────────────────────
info "Updating apt and installing system packages..."
apt-get update -q
apt-get install -y -q \
    git curl wget build-essential \
    python3 python3-pip python3-venv python3-dev \
    sqlite3 libsqlite3-dev \
    debian-keyring debian-archive-keyring apt-transport-https

# ── Caddy ────────────────────────────────────────────────────────────────────
if ! command -v caddy &>/dev/null; then
    info "Installing Caddy..."
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -q
    apt-get install -y -q caddy
    success "Caddy installed."
else
    success "Caddy already installed."
fi

# ── Service user ─────────────────────────────────────────────────────────────
if ! id "$SERVICE_USER" &>/dev/null; then
    info "Creating system user: $SERVICE_USER"
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

# ── Clone / update repo ──────────────────────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Repo already exists — pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    info "Cloning OpenEngine..."
    git clone "$REPO" "$INSTALL_DIR"
fi
success "Repo at $INSTALL_DIR"

# ── Python venv ──────────────────────────────────────────────────────────────
info "Setting up Python virtual environment..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip -q
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
success "Python dependencies installed."

# ── Model weights ────────────────────────────────────────────────────────────
WEIGHTS_PATH="$INSTALL_DIR/models/policy_v3.pt"
mkdir -p "$INSTALL_DIR/models"

if [[ -f "$WEIGHTS_PATH" ]]; then
    success "Model weights already present."
elif [[ -n "$WEIGHTS_URL" ]]; then
    info "Downloading model weights..."
    wget -q --show-progress -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
    success "Model weights downloaded."
else
    warn "Model weights not found at $WEIGHTS_PATH"
    warn "Upload them manually:"
    warn "  scp policy_v3.pt root@$DOMAIN:$WEIGHTS_PATH"
    warn "Then restart: systemctl restart $SERVICE_NAME"
fi

# ── Database directory ───────────────────────────────────────────────────────
mkdir -p /var/lib/openengine
chown "$SERVICE_USER:$SERVICE_USER" /var/lib/openengine

# ── .env ─────────────────────────────────────────────────────────────────────
info "Writing .env..."
cat > "$INSTALL_DIR/.env" <<EOF
# Auto-generated by install.sh — edit as needed

# ── LLM Provider (fallback — callers can override per-request) ───────────────
LLM_PROVIDER=${LLM_PROVIDER}
LLM_API_KEY=${LLM_API_KEY}
LLM_MODEL=

# Cloudflare Workers AI
CF_ACCOUNT_ID=${CF_ACCOUNT_ID}
CF_API_TOKEN=${CF_API_TOKEN}

# ── Tools ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY=${TAVILY_API_KEY}
RESEND_API_KEY=${RESEND_API_KEY}

# ── ToolServer ───────────────────────────────────────────────────────────────
XYBER_TOOL_WEBHOOK_URL=http://localhost:9000/invoke
XYBER_TOOL_WEBHOOK_SECRET=${WEBHOOK_SECRET}
XYBER_CAPABILITY_MANIFEST_PATH=examples/manifests/toolserver.json

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_PATH=/var/lib/openengine/working_memory.db

# ── Redis (optional) ─────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Local ────────────────────────────────────────────────────────────────────
XYBER_LOCAL_USER_ID=local-dev
EOF
chmod 600 "$INSTALL_DIR/.env"
chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
success ".env written."

# ── File ownership ───────────────────────────────────────────────────────────
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# ── systemd service ───────────────────────────────────────────────────────────
info "Installing systemd service..."
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=OpenEngine API
After=network.target
Wants=network-online.target

[Service]
User=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF
success "systemd service written."

# ── Caddy config ─────────────────────────────────────────────────────────────
info "Configuring Caddy for $DOMAIN..."
cat > /etc/caddy/Caddyfile <<EOF
${DOMAIN} {
    reverse_proxy 127.0.0.1:8000 {
        header_up X-Real-IP {remote_host}
    }

    # Rate limit: 30 requests/minute per IP
    rate_limit {remote_host} 30r/m

    log {
        output file /var/log/caddy/openengine.log
        format json
    }
}
EOF
mkdir -p /var/log/caddy
success "Caddyfile written."

# ── Enable WAL mode on SQLite at first run ───────────────────────────────────
info "Enabling SQLite WAL mode..."
DB_PATH="/var/lib/openengine/working_memory.db"
if command -v sqlite3 &>/dev/null; then
    sudo -u "$SERVICE_USER" sqlite3 "$DB_PATH" "PRAGMA journal_mode=WAL;" 2>/dev/null || true
fi

# ── Start services ───────────────────────────────────────────────────────────
info "Starting services..."
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" caddy
systemctl restart "$SERVICE_NAME" caddy

# ── Health check ─────────────────────────────────────────────────────────────
info "Waiting for API to come up..."
sleep 4
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/v1/health" || echo "000")
if [[ "$HTTP_STATUS" == "200" ]]; then
    success "API is live at http://127.0.0.1:8000/v1/health"
else
    warn "Health check returned HTTP $HTTP_STATUS — check logs: journalctl -u $SERVICE_NAME -n 50"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  OpenEngine is installed.${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════${RESET}"
echo ""
echo -e "  API endpoint:  ${CYAN}https://${DOMAIN}/v1/run${RESET}"
echo -e "  Health check:  ${CYAN}https://${DOMAIN}/v1/health${RESET}"
echo -e "  Swagger UI:    ${CYAN}https://${DOMAIN}/docs${RESET}"
echo ""
echo -e "  Logs:     ${YELLOW}journalctl -u ${SERVICE_NAME} -f${RESET}"
echo -e "  Restart:  ${YELLOW}systemctl restart ${SERVICE_NAME}${RESET}"
echo -e "  Config:   ${YELLOW}${INSTALL_DIR}/.env${RESET}"
echo ""

if [[ ! -f "$WEIGHTS_PATH" ]]; then
    echo -e "${RED}${BOLD}  ⚠ Model weights missing — upload before use:${RESET}"
    echo -e "  ${YELLOW}scp policy_v3.pt root@${DOMAIN}:${WEIGHTS_PATH}${RESET}"
    echo -e "  ${YELLOW}systemctl restart ${SERVICE_NAME}${RESET}"
    echo ""
fi
