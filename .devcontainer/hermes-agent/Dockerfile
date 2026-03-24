# Minimal containerized environment for Hermes Agent

FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y \
    git \
    iptables \
    iproute2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=hermes-agent
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

RUN git clone https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent && \
    cd /opt/hermes-agent && \
    git checkout 934fc9df221cb24402bac442db46318cf083800b && \
    git submodule update --init mini-swe-agent && \
    pip install --no-cache-dir ".[mcp]" && \
    pip install --no-cache-dir -e "./mini-swe-agent"

COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

USER $USERNAME
WORKDIR /workspace

ENTRYPOINT ["sudo", "-E", "/usr/local/bin/init-firewall.sh"]
