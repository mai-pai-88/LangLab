FROM python:3.12-slim

ARG USERNAME=labuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl git nodejs sudo unzip \
	&& curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
	&& unzip -q /tmp/awscliv2.zip -d /tmp \
	&& /tmp/aws/install \
	&& rm -rf /tmp/aws /tmp/awscliv2.zip \
	&& groupadd --gid ${USER_GID} ${USERNAME} \
	&& useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
	&& echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
	&& chmod 0440 /etc/sudoers.d/${USERNAME} \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

ENV UV_INSTALL_DIR=/usr/local/bin
ENV UV_NO_MODIFY_PATH=1
COPY requirements.txt /tmp/requirements.txt
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
	&& uv pip install --system --requirement /tmp/requirements.txt

WORKDIR /workspaces/LangLab
COPY .claude/skills/archify /home/${USERNAME}/.claude/skills/archify
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.claude
USER ${USERNAME}
