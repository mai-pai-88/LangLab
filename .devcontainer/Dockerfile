FROM python:3.12-slim

ARG USERNAME=labuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl git sudo unzip \
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

WORKDIR /workspaces/LangLab
USER ${USERNAME}
