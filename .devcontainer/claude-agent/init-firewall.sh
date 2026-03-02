#!/bin/bash
# Firewall initialization for Claude Code agent container.
# Runs as root (via sudo) before handing off to the claude-agent user.
#
# Network policy:
#   ALLOW  loopback
#   ALLOW  DNS (UDP/TCP 53)
#   ALLOW  HTTPS (TCP 443) — Anthropic API via Cloudflare
#   ALLOW  MCP SSE server on host (TCP $MCP_PORT → host gateway)
#   DROP   Direct access to game server port ($GAME_SERVER_PORT)
#   DROP   everything else

set -e

MCP_PORT="${MCP_PORT:-8001}"
GAME_SERVER_PORT="${GAME_SERVER_PORT:-8000}"

# Claude Code expects ~/.claude.json in the home directory, not inside ~/.claude/
if [ -f /home/claude-agent/.claude/.claude.json ]; then
    cp /home/claude-agent/.claude/.claude.json /home/claude-agent/.claude.json
    chown claude-agent:claude-agent /home/claude-agent/.claude.json
fi

# Mounted volumes are owned by the host user — grant claude-agent read/write access.
# /workspace: agent scratch space (needs write for temp files, session state)
# /home/claude-agent/.claude: agent memory with credentials (needs read for OAuth tokens)
chown -R claude-agent:claude-agent /workspace 2>/dev/null || chmod -R 777 /workspace 2>/dev/null || true
chown -R claude-agent:claude-agent /home/claude-agent/.claude 2>/dev/null || chmod -R 755 /home/claude-agent/.claude 2>/dev/null || true

# Wrapper script propagates env vars that su drops (su resets the environment).
# Uses exec to preserve the PTY chain from Docker -t.
cat > /tmp/run_claude.sh << 'WRAPPER_EOF'
#!/bin/sh
export HOME=/home/claude-agent
export CLAUDE_CONFIG_DIR=/home/claude-agent/.claude
DIAG=/workspace/.container_diagnostics.log
{
  echo "=== Container Diagnostics ==="
  echo "timestamp: $(date)"
  echo "user: $(whoami)"
  echo "HOME=$HOME"
  echo "CLAUDE_CONFIG_DIR=$CLAUDE_CONFIG_DIR"
  echo "full_args: $*"
  echo "arg_count: $#"
  echo ""
  echo "=== Credential Files ==="
  echo "~/.claude.json exists: $(test -f "$HOME/.claude.json" && echo YES || echo NO)"
  echo "~/.claude.json size: $(wc -c < "$HOME/.claude.json" 2>/dev/null || echo 0)"
  echo "$CLAUDE_CONFIG_DIR/ contents:"
  ls -la "$CLAUDE_CONFIG_DIR/" 2>&1
  echo ".credentials.json size: $(wc -c < "$CLAUDE_CONFIG_DIR/.credentials.json" 2>/dev/null || echo 0)"
  echo "settings.json size: $(wc -c < "$CLAUDE_CONFIG_DIR/settings.json" 2>/dev/null || echo 0)"
  echo ""
  echo "=== TTY Status ==="
  test -t 0 && echo "stdin: TTY" || echo "stdin: NOT_TTY"
  test -t 1 && echo "stdout: TTY" || echo "stdout: NOT_TTY"
  test -t 2 && echo "stderr: TTY" || echo "stderr: NOT_TTY"
  echo ""
  echo "=== DNS Test ==="
  getent hosts api.anthropic.com 2>&1 || echo "DNS_FAILED"
  echo ""
  echo "=== MCP Config ==="
  cat /workspace/.mcp_config.json 2>&1 || echo "NO_MCP_CONFIG"
  echo ""
  echo "=== Node/Claude Version ==="
  node --version 2>&1 || echo "NO_NODE"
  which claude 2>&1 || echo "NO_CLAUDE_BIN"
  echo ""
  echo "=== About to exec claude ==="
} > "$DIAG" 2>&1
exec "$@"
WRAPPER_EOF
chmod +x /tmp/run_claude.sh

# Skip firewall when using host network (would modify host iptables)
if [ "${SKIP_FIREWALL:-0}" = "1" ]; then
    echo "⏭️  Skipping firewall (host network mode)"
    exec su claude-agent -c "/tmp/run_claude.sh $*"
fi

echo "🔒 Initializing container firewall..."

# ── IPv4 ──────────────────────────────────────────────────────────────────────
iptables -F && iptables -X
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

iptables -A OUTPUT -p udp --dport 53 -j ACCEPT   # DNS
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT   # Anthropic API (HTTPS)

# Allow MCP SSE server on the host gateway.
# host.docker.internal is mapped to the gateway by --add-host in the docker run command.
HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
fi
iptables -A OUTPUT -p tcp -d "$HOST_IP" --dport "$MCP_PORT" -j ACCEPT
echo "   ✓ MCP server allowed: $HOST_IP:$MCP_PORT"

iptables -A OUTPUT -p tcp --dport "$GAME_SERVER_PORT" -j DROP
echo "   ✓ Game server blocked: port $GAME_SERVER_PORT"

# ── IPv6 ──────────────────────────────────────────────────────────────────────
ip6tables -F 2>/dev/null && ip6tables -X 2>/dev/null || true
ip6tables -P INPUT  DROP 2>/dev/null || true
ip6tables -P FORWARD DROP 2>/dev/null || true
ip6tables -P OUTPUT DROP 2>/dev/null || true
ip6tables -A INPUT  -i lo -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
ip6tables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p udp --dport 53 -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p tcp --dport 53 -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true

echo "✅ Firewall initialized"

# Switch to claude-agent user and exec the agent command.
exec su claude-agent -c "/tmp/run_claude.sh $*"
