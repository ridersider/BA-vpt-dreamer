# x86_64 Ubuntu 20.04 (kein CUDA nötig – Training läuft auf CPU)
# --platform=linux/amd64 ist zwingend: Minecraft 1.11.2 braucht LWJGL2-Natives,
# die nur für x86/x86_64 existieren, nicht für ARM64.
FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV DISPLAY=:99
ENV MINEDOJO_HEADLESS=1

# ── System-Pakete ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    python3.9 python3.9-dev python3-pip \
    git wget curl \
    xvfb x11-utils x11-xserver-utils \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libgl1-mesa-glx libegl1-mesa \
    libgl1-mesa-dri \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3

# ── pip vorbereiten ────────────────────────────────────────────
RUN pip install --upgrade "setuptools==65.5.0" "pip==21.3.1" "wheel==0.38.4"

# ── PyTorch (CPU – kein GPU im Container) ─────────────────────
RUN pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── MineDojo + VPT-Abhängigkeiten ─────────────────────────────
RUN pip install minedojo
RUN pip install \
    gym==0.19.0 \
    gym3 \
    opencv-python-headless \
    einops \
    x-transformers \
    numpy==1.23.5

# ── r2dreamer-Abhängigkeiten ───────────────────────────────────
RUN pip install hydra-core omegaconf tensorboard

# ── Eigener Code + Repos ins Image ────────────────────────────
WORKDIR /workspace
COPY repos/r2dreamer /workspace/r2dreamer
COPY repos/vpt       /workspace/vpt

# ── MineDojo patchen & Malmo bauen ────────────────────────────
# Dieser Schritt dauert beim ersten Build ~5–15 Min (Gradle lädt Minecraft herunter).
# Das Ergebnis ist im Image gecacht – spätere Starts sind sofort fertig.
COPY docker/setup_minedojo.sh /tmp/setup_minedojo.sh
RUN chmod +x /tmp/setup_minedojo.sh && /tmp/setup_minedojo.sh

# ── MineRL (nach MineDojo, damit /opt/local-maven-repo existiert) ─
COPY docker/setup_minerl.sh /tmp/setup_minerl.sh
RUN chmod +x /tmp/setup_minerl.sh && /tmp/setup_minerl.sh

# ── Arbeitsverzeichnisse anlegen ──────────────────────────────
RUN mkdir -p /workspace/src /workspace/models /workspace/logs

WORKDIR /workspace

# Xvfb beim Start (für Minecraft-Rendering nötig).
# ENTRYPOINT statt CMD, damit Dev Containers es nicht überschreiben.
ENTRYPOINT ["bash", "-c", "Xvfb :99 -screen 0 1280x720x24 +extension GLX & sleep 1 && exec \"$@\"", "--"]
CMD ["bash"]
