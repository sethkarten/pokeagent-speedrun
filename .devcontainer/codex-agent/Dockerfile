# Minimal containerized environment for Codex CLI agent
# Analogous to claude-agent and gemini-agent with OpenAI/OpenRouter support

FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    iptables \
    iproute2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Override at build time so container UID/GID matches host (avoids permission errors on bind mounts):
#   docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) ...
ARG USERNAME=codex-agent
ARG USER_UID=1001
ARG USER_GID=1001

RUN if getent group $USER_GID > /dev/null 2>&1; then \
        groupdel $(getent group $USER_GID | cut -d: -f1); \
    fi && \
    if getent passwd $USER_UID > /dev/null 2>&1; then \
        userdel -f $(getent passwd $USER_UID | cut -d: -f1); \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Install Codex CLI globally via npm
# Auth: OPENROUTER_API_KEY or OPENAI_API_KEY passed at runtime; or codex login credentials in ~/.codex
RUN npm install -g @openai/codex

COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

USER $USERNAME

WORKDIR /workspace

# The entrypoint runs the firewall initialization, then execs the command
# sudo -E preserves environment variables (MCP_PORT, GAME_SERVER_PORT, OPENROUTER_API_KEY)
ENTRYPOINT ["sudo", "-E", "/usr/local/bin/init-firewall.sh"]
