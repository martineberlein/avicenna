FROM ubuntu:22.04
LABEL author="Martin Eberlein"

USER root

# Set environment variables
ENV TZ=Europe/Berlin \
    PYENV_ROOT=/root/.pyenv \
    PATH=/root/.pyenv/shims:/root/.pyenv/bin:$PATH

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install \
    git \
    bash \
    fish \
    gcc \
    g++ \
    make \
    cmake \
    clang \
    wget \
    python3 \
    python3-pip \
    vim \
    curl

# Dependencies for pyenv
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y tzdata
RUN apt-get -y install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# Install pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $PYENV_ROOT


# RUN useradd -ms /bin/bash avicenna
# USER avicenna
WORKDIR /home/avicenna

RUN pip3 install --upgrade pip wheel setuptools
# RUN pip3 install scipy fuzzingbook==1.1

WORKDIR /home/avicenna
# Clone and install Tests4Py
RUN git clone https://github.com/smythi93/Tests4Py.git && \
    cd Tests4Py && \
    git pull && \
    pip3 install .

WORKDIR /home/avicenna
# Clone and install avicenna
RUN git clone https://github.com/martineberlein/avicenna.git avicenna && \
    cd avicenna && \
    git pull && \
    git checkout dev && \
    pip3 install -e .[dev]

CMD ["fish"]
