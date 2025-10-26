Bootstrap: docker
From: python:3.12-slim-bookworm

%labels
    AppName        xestools
    Description    "I20 XES/RXES Viewer"
    Maintainer     "luke.higgins@diamond.ac.uk"

%environment
    export MPLBACKEND=QtAgg
    export QT_QPA_PLATFORM=xcb
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PYTHONUNBUFFERED=1
    export PATH=/opt/venv/bin:$PATH
    export PYTHONPATH=/app:$PYTHONPATH

%post
    set -e
    export DEBIAN_FRONTEND=noninteractive

    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-util1 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-render0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libsm6 \
        libice6 \
        libfontconfig1 \
        fonts-dejavu-core \
        libgl1 \
        libglx0 \
        libegl1 \
        libopengl0 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libfreetype6 \
        libpng16-16 \
        libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

    python -m venv /opt/venv
    . /opt/venv/bin/activate

    pip install --upgrade pip setuptools wheel
    pip install numpy scipy matplotlib PySide6 h5py lmfit

    mkdir -p /app

%files
    . /app

%runscript
    . /opt/venv/bin/activate
    cd /app
    exec python main.py "$@"

%test
    . /opt/venv/bin/activate
    # Force an offscreen platform during test to dodge X/GL requirements:
    QT_QPA_PLATFORM=offscreen python -c "import PySide6; print('PySide6 import OK')"
