#!/bin/bash
# Firewall initialization for Hermes agent container.

set -e

MCP_PORT="${MCP_PORT:-8001}"
GAME_SERVER_PORT="${GAME_SERVER_PORT:-8000}"

cat > /tmp/run_hermes.sh << 'WRAPPER_EOF'
#!/bin/sh
export HOME=/home/hermes-agent
export HERMES_HOME=/home/hermes-agent/.hermes
# Hermes must come first so its utils.py (atomic_json_write) is found; our utils package would shadow it
export PYTHONPATH="/opt/hermes-agent:/opt/pokeagent-src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
WRAPPER_EOF
chmod +x /tmp/run_hermes.sh

if [ "${SKIP_FIREWALL:-0}" = "1" ]; then
    echo "⏭️  Skipping firewall (host network mode)"
    exec su hermes-agent -c "/tmp/run_hermes.sh $*"
fi

echo "🔒 Initializing container firewall..."

iptables -F && iptables -X
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
fi
iptables -A OUTPUT -p tcp -d "$HOST_IP" --dport "$MCP_PORT" -j ACCEPT
echo "   ✓ MCP server allowed: $HOST_IP:$MCP_PORT"

iptables -A OUTPUT -p tcp --dport "$GAME_SERVER_PORT" -j DROP
echo "   ✓ Game server blocked: port $GAME_SERVER_PORT"

ip6tables -F 2>/dev/null && ip6tables -X 2>/dev/null || true
ip6tables -P INPUT DROP 2>/dev/null || true
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

exec su hermes-agent -c "/tmp/run_hermes.sh $*"
