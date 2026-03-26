#!/bin/bash
# Patches alle bekannten MineDojo-Kompatibilitätsprobleme und baut Malmo.
# Wird einmalig beim Docker-Build ausgeführt.
set -e

MINEDOJO_SIM=/usr/local/lib/python3.9/dist-packages/minedojo/sim
MC_DIR=$MINEDOJO_SIM/Malmo/Minecraft
VPT_DIR=/workspace/vpt

echo "=== [1/6] numpy 2.0 Patch ==="
sed -i 's/np\.unicode_/np.str_/g' $MINEDOJO_SIM/spaces.py

echo "=== [2/6] gym Dict .keys() + __setitem__ Patch ==="
python3 - << 'EOF'
import re, pathlib

wrappers = pathlib.Path('/usr/local/lib/python3.9/dist-packages/minedojo/sim/wrappers/ar_nn')
for f in wrappers.glob('*.py'):
    t = f.read_text()
    # observation_space.keys() / action_space.keys() -> .spaces.keys()
    t = t.replace('.observation_space.keys()', '.observation_space.spaces.keys()')
    t = t.replace('.action_space.keys()',       '.action_space.spaces.keys()')
    # observation_space["x"].keys() -> observation_space["x"].spaces.keys()
    t = re.sub(r'(observation_space\[.*?\])\.keys\(\)', r'\1.spaces.keys()', t)
    # obs_space["x"] = ... -> obs_space.spaces["x"] = ...  (gym 0.19 hat kein __setitem__)
    t = re.sub(r'\bobs_space\["', 'obs_space.spaces["', t)
    # obs_space.keys() -> obs_space.spaces.keys()
    t = t.replace('obs_space.keys()', 'obs_space.spaces.keys()')
    # del obs_space["x"] -> del obs_space.spaces["x"]
    t = re.sub(r'del obs_space\[', 'del obs_space.spaces[', t)
    f.write_text(t)
print("wrappers patched")
EOF

echo "=== [3/6] VPT actions.py: minerl-Import optional machen ==="
sed -i 's/^import minerl.herobraine.hero.mc as mc$/try:\n    import minerl.herobraine.hero.mc as mc\nexcept ImportError:\n    mc = None/' \
    $VPT_DIR/lib/actions.py

echo "=== [4/6] MixinGradle aus Source bauen ==="
cd /tmp
git clone --depth 200 https://github.com/SpongePowered/MixinGradle.git
cd MixinGradle
git checkout dcfaf61

# Gradle-Wrapper aus dem Minecraft-Verzeichnis übernehmen (Version 4.10.2)
cp -r $MC_DIR/gradle .
cp    $MC_DIR/gradlew .
chmod +x gradlew

GRADLE_USER_HOME=/root/.gradle ./gradlew jar
JAR=$(ls build/libs/*.jar)

# In lokales Maven-Repo legen
REPO=/opt/local-maven-repo/com/github/SpongePowered/MixinGradle/dcfaf61
mkdir -p "$REPO"
cp "$JAR" "$REPO/MixinGradle-dcfaf61.jar"
cat > "$REPO/MixinGradle-dcfaf61.pom" << 'POM'
<?xml version="1.0" encoding="UTF-8"?>
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.github.SpongePowered</groupId>
  <artifactId>MixinGradle</artifactId>
  <version>dcfaf61</version>
</project>
POM

echo "=== [5/6] Malmo build.gradle patchen ==="
python3 - << 'EOF'
import pathlib

path = pathlib.Path('/usr/local/lib/python3.9/dist-packages/minedojo/sim/Malmo/Minecraft/build.gradle')
c = path.read_text()

# 5a) Lokales Maven-Repo als erste Repository-Quelle hinzufügen
c = c.replace(
    'buildscript {\n    repositories {\n',
    'buildscript {\n    repositories {\n        maven { url "file:///opt/local-maven-repo" }\n'
)

# 5b) schemas.index: relativen Pfad durch projektbezogenen Pfad ersetzen
c = c.replace(
    "def schemaIndexFile = new File('src/main/resources/schemas.index')",
    "def schemaIndexFile = new File(projectDir, 'src/main/resources/schemas.index')"
)

# 5c) runClient Task: headless + DISPLAY setzen
c = c.replace(
    'exec.jvmArgs(["-Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin","-Xmx2G", "-Xdebug", "-Dmixin.debug=true"])',
    'exec.jvmArgs(["-Djava.awt.headless=true","-Dfml.ignoreMissingCert=true","-Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin","-Xmx2G"])'
)
c = c.replace(
    'exec.args(["--tweakClass", "com.microsoft.Malmo.Launcher.tweakers.CoremodTweaker"])',
    'exec.args(["--tweakClass", "com.microsoft.Malmo.Launcher.tweakers.CoremodTweaker"])\nexec.environment("DISPLAY", System.getenv("DISPLAY") ?: ":99")'
)

path.write_text(c)
print("build.gradle patched")
EOF

echo "=== [5d] launchClient.sh patchen ==="
python3 - << 'EOF'
import pathlib

path = pathlib.Path('/usr/local/lib/python3.9/dist-packages/minedojo/sim/Malmo/Minecraft/launchClient.sh')
c = path.read_text()

# headless + ignoreMissingCert zur fat.jar-Kommandozeile hinzufügen
c = c.replace(
    'cmd="java -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin',
    'cmd="java -Djava.awt.headless=true -Dfml.ignoreMissingCert=true -Dfml.coreMods.load=com.microsoft.Malmo.OverclockingPlugin'
)

# xvfb-run durch direktes Starten mit gesetztem DISPLAY ersetzen
c = c.replace(
    'if [ "$MINEDOJO_HEADLESS" == "1" ]; then\n  xvfb-run -a -s "-screen 0 1024x768x24" $cmd\nelse\n  $cmd\nfi',
    'export DISPLAY=${DISPLAY:-:99}\n$cmd'
)

path.write_text(c)
print("launchClient.sh patched")
EOF

echo "=== [6/6] Malmo bauen (setupDecompWorkspace + shadowJar) ==="
cd $MC_DIR
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
./gradlew setupDecompWorkspace
./gradlew shadowJar

echo "=== [7/7] Gradle-Cache symlinken + instance.py patchen ==="
# srg-mcp.srg liegt in ~/.gradle/caches/minecraft, aber GradleStart sucht es
# unter {minecraft_dir}/run/gradle/caches/minecraft -> Symlink erstellen
mkdir -p $MC_DIR/run/gradle/caches
ln -sfn /root/.gradle/caches/minecraft $MC_DIR/run/gradle/caches/minecraft

# instance.py: copytree mit symlinks=True damit der Symlink ins Temp-Dir
# kopiert wird statt 76MB Dateien bei jedem env.reset()
python3 - << 'EOF'
import pathlib
f = pathlib.Path('/usr/local/lib/python3.9/dist-packages/minedojo/sim/bridge/mc_instance/instance.py')
t = f.read_text()
t = t.replace(
    'ignore=shutil.ignore_patterns("**.lock"),\n            )',
    'ignore=shutil.ignore_patterns("**.lock"),\n                symlinks=True,\n            )'
)
f.write_text(t)
print("instance.py patched")
EOF

echo "=== Setup abgeschlossen ==="
ls -lah $MC_DIR/build/libs/
