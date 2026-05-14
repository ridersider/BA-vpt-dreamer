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

# ── Gradle-Cache vorbefüllen ──────────────────────────────────
# Verhindert dass Gradle Minecraft-Assets vom Netz lädt (auf dem Cluster geblockt).
# Erstellt mit: docker run --rm vpt-dreamer:v2 tar -czf - /root/.gradle/caches/minecraft \
#               > docker/gradle_minecraft_cache.tar.gz
COPY docker/gradle_minecraft_cache.tar.gz /tmp/gradle_minecraft_cache.tar.gz
RUN tar -xzf /tmp/gradle_minecraft_cache.tar.gz -C / && \
    rm -f /tmp/gradle_minecraft_cache.tar.gz

# ── MineDojo patchen & Malmo bauen ────────────────────────────
# downloadAssets wird übersprungen: der Cluster blockt resources.download.minecraft.net,
# was 1196 parallele fehlschlagende Connections und "Too many open files" verursacht.
# Die benötigten Assets (Core-Texturen) sind bereits im gradle_minecraft_cache.tar.gz.
COPY docker/setup_minedojo.sh /tmp/setup_minedojo.sh
RUN chmod +x /tmp/setup_minedojo.sh && \
    sed -i 's|./gradlew setupDecompWorkspace$|./gradlew setupDecompWorkspace -x getAssets|' /tmp/setup_minedojo.sh && \
    /tmp/setup_minedojo.sh

# ── MineRL (nach MineDojo, damit /opt/local-maven-repo existiert) ─
COPY docker/setup_minerl.sh /tmp/setup_minerl.sh
RUN chmod +x /tmp/setup_minerl.sh && /tmp/setup_minerl.sh

# ── Arbeitsverzeichnisse anlegen ──────────────────────────────
RUN mkdir -p /workspace/src /workspace/models /workspace/logs

WORKDIR /workspace

ENTRYPOINT ["bash", "-c", "Xvfb :99 -screen 0 1280x720x24 +extension GLX & sleep 1 && exec \"$@\"", "--"]
CMD ["bash"]
