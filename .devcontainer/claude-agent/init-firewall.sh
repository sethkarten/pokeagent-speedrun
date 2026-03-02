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

# Mounted volumes are owned by the host user — fix permissions so claude-agent can access them.
# /workspace: needs read+write (temp files, session state)
# /home/claude-agent/.claude: needs read+write (OAuth credentials + Claude session files)
# Use chmod (not chown) so the host user retains ownership for backup operations.
chmod -R a+rwX /workspace 2>/dev/null || true
chmod -R a+rwX /home/claude-agent/.claude 2>/dev/null || true

# Wrapper script propagates env vars that su drops (su resets the environment).
# Uses exec to preserve the PTY chain from Docker -t.
cat > /tmp/run_claude.sh << 'WRAPPER_EOF'
#!/bin/sh
export HOME=/home/claude-agent
export CLAUDE_CONFIG_DIR=/home/claude-agent/.claude
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
