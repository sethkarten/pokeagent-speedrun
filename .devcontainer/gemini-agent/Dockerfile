# Minimal containerized environment for Gemini CLI agent
# Analogous to claude-agent container with Google-specific modifications

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
ARG USERNAME=gemini-agent
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

# Install Gemini CLI globally via npm
# Auth: GEMINI_API_KEY is passed as an env var at runtime (no login flow needed)
RUN npm install -g @google/gemini-cli

COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

USER $USERNAME

WORKDIR /workspace

# The entrypoint runs the firewall initialization, then execs the command
# sudo -E preserves environment variables (MCP_PORT, GAME_SERVER_PORT, GEMINI_API_KEY)
ENTRYPOINT ["sudo", "-E", "/usr/local/bin/init-firewall.sh"]
