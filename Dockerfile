FROM ubuntu:20.04
LABEL maintainer="baung.dinh@veriserve-vietnam.com"
LABEL version="1.0"

ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PATH="/opt/pyenv/shims:/opt/pyenv/bin:$PATH" \
    PYENV_ROOT="/opt/pyenv" \
    PYENV_SHELL="bash" \
    TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install sudo gnupg software-properties-common --no-install-recommends \
        build-essential -y \
        ca-certificates -y \
        curl -y \
        git -y \
        libbz2-dev -y \
        libffi-dev -y \
        libncurses5-dev -y \
        libncursesw5-dev -y \
        libreadline-dev -y \
        libsqlite3-dev -y \
        #libssl1.1-dev \
        liblzma-dev -y \
        libssl-dev -y \
        llvm -y \
        make -y \
        netbase -y \
        pkg-config -y \
        tk-dev -y \
        wget -y \
        xz-utils -y \
        zlib1g-dev -y \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyenv-version.txt python-versions.txt /

RUN git clone -b `cat /pyenv-version.txt` --single-branch --depth 1 https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && for version in `cat /python-versions.txt`; do pyenv install $version; done \
    && pyenv global `cat /python-versions.txt` \
    && find $PYENV_ROOT/versions -type d '(' -name '__pycache__' -o -name 'test' -o -name 'tests' ')' -exec rm -rf '{}' + \
    && find $PYENV_ROOT/versions -type f '(' -name '*.pyo' -o -name '*.exe' ')' -exec rm -f '{}' + \
 && rm -rf /tmp/*

# Set Japanese environment
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen ja_JP.UTF-8 && \
    echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Set alias for python3
RUN echo "alias python=python3" >> $HOME/.bashrc && \
    echo "alias pip=pip3" >> $HOME/.bashrc

RUN pip --no-cache-dir install \
    h5py \
    typing-extensions\
    wheel

# CUDA 11.3
RUN pip --no-cache-dir install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -U torchtext==0.10.0

RUN pip --no-cache-dir install \
    pytest \
    opencv-python \
    graphviz \
    gpustat

# Install python modules.
COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt \
    && find $PYENV_ROOT/versions -type d '(' -name '__pycache__' -o -name 'test' -o -name 'tests' ')' -exec rm -rf '{}' + \
    && find $PYENV_ROOT/versions -type f '(' -name '*.pyo' -o -name '*.exe' ')' -exec rm -f '{}' + \
    && rm -rf /tmp/*

WORKDIR /work

#CMD ["/bin/bash"]
CMD [ "python", "app.py"]
