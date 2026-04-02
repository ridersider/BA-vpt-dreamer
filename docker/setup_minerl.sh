#!/bin/bash
# Klont MineRL, patcht build.gradle (lokales MixinGradle-Maven-Repo),
# und installiert das Paket. Muss nach setup_minedojo.sh laufen,
# damit /opt/local-maven-repo bereits existiert.
set -e

echo "=== [1/3] MineRL klonen ==="
git clone --depth 1 https://github.com/minerllabs/minerl.git /opt/minerl
cd /opt/minerl

echo "=== [2/3] build.gradle patchen (lokales MixinGradle-Repo) ==="
python3 - << 'EOF'
import pathlib

path = pathlib.Path('/opt/minerl/minerl/Malmo/Minecraft/build.gradle')
c = path.read_text()
c = c.replace(
    'buildscript {\n    repositories {\n',
    'buildscript {\n    repositories {\n        maven { url "file:///opt/local-maven-repo" }\n'
)
path.write_text(c)
print("build.gradle patched")
EOF

echo "=== [3/3] MineRL installieren ==="
pip install .

echo "=== MineRL Setup abgeschlossen ==="
