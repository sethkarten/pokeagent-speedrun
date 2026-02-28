# Minimal containerized environment for Claude Code CLI agent
# Based on Anthropic's reference devcontainer with security-focused modifications

FROM node:20-bookworm-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    iptables \
    iproute2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for running the agent
# Override at build time so container UID/GID matches host (avoids permission errors on bind mounts):
#   docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) ...
# Defaults below are fallbacks only; they may not match your host UID/GID.
ARG USERNAME=claude-agent
ARG USER_UID=1001
ARG USER_GID=1001

# If the requested GID already exists, delete the group to free it up
# Then create the new user/group with the specified ID
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

# Install Claude Code CLI globally via npm (same as Anthropic's reference devcontainer: 
# @see https://github.com/anthropics/claude-code/tree/main/.devcontainer)
# Auth: Host's ~/.claude/ subscription credentials are mounted at runtime
RUN npm install -g @anthropic-ai/claude-code

# Copy and setup firewall script
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

# Switch to non-root user
USER $USERNAME

# Set working directory to the mounted agent scratch space
WORKDIR /workspace

# The entrypoint runs the firewall initialization, then execs the command
# Use sudo -E to preserve environment variables (MCP_PORT, GAME_SERVER_PORT)
ENTRYPOINT ["sudo", "-E", "/usr/local/bin/init-firewall.sh"]
