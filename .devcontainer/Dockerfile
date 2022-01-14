# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.191.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Choice] Node.js version: none, lts, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

RUN pip install alpaca_trade_api==1.4.3 \
    boto3 \
    jupyterlab \
    mplfinance \
    numpy \
    pandas \
    pandas_ta \
    pyarrow \
    pytest \
    pytest-cov


# from https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html#installing-h2o-3
RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

RUN mkdir -p /usr/share/man/man1 /usr/share/man/man2

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-11-jre