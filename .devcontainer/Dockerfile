FROM python:3.12-slim

ARG USERNAME=labuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN apt-get update \
	&& apt-get install -y --no-install-recommends sudo \
	&& groupadd --gid ${USER_GID} ${USERNAME} \
	&& useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
	&& echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
	&& chmod 0440 /etc/sudoers.d/${USERNAME} \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/LangLab
USER ${USERNAME}
