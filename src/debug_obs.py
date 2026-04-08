import sys
sys.path.insert(0, '/workspace/vpt')

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import ENV_KWARGS

env = HumanSurvival(**ENV_KWARGS).make()
obs = env.reset()

print("obs keys:", list(obs.keys()))
if "inventory" in obs:
    print("inventory keys:", list(obs["inventory"].keys()))
env.close()
