from typing import List

import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.server.world import FlatWorldGenerator


class DrawingDecorator(Handler):
    """DrawingDecorator that bypasses Jinja2 autoescape by overriding xml()."""

    def __init__(self, to_draw: str):
        self.to_draw = to_draw

    def xml(self) -> str:
        return f"<DrawingDecorator>{self.to_draw}</DrawingDecorator>"

    def xml_template(self) -> str:
        return "<DrawingDecorator>{{ to_draw }}</DrawingDecorator>"

    def to_string(self) -> str:
        return "drawing_decorator"

# ── Tree definitions ───────────────────────────────────────────────────────────
# Each entry: (log_type, leaf_type)
TREE_TYPES = [
    ("minecraft:oak_log",      "minecraft:oak_leaves"),
    ("minecraft:birch_log",    "minecraft:birch_leaves"),
    ("minecraft:spruce_log",   "minecraft:spruce_leaves"),
    ("minecraft:jungle_log",   "minecraft:jungle_leaves"),
    ("minecraft:acacia_log",   "minecraft:acacia_leaves"),
    ("minecraft:dark_oak_log", "minecraft:dark_oak_leaves"),
    ("minecraft:dirt")
]

# Superflat: bedrock at y=0, 2x dirt y=1-2, grass at y=3 → trees start at y=4
GROUND_Y = 3
TRUNK_HEIGHT = 5   # y=4 .. y=8
LEAF_RADIUS = 2    # leaves extend ±2 blocks in X/Z
LEAF_Y_START = 6   # leaves start 2 below treetop
LEAF_Y_END   = GROUND_Y + TRUNK_HEIGHT + 1  # y=9

TREE_SPACING = 7   # blocks between trunks
GRID_RADIUS  = 4   # trees from -GRID_RADIUS to +GRID_RADIUS  → 9×9 = 81 trees


def _cuboid(x1, y1, z1, x2, y2, z2, block: str) -> str:
    return (
        f'<DrawCuboid x1="{x1}" y1="{y1}" z1="{z1}"'
        f' x2="{x2}" y2="{y2}" z2="{z2}" type="{block}"/>'
    )


def _build_tree_xml() -> str:
    parts = []
    tree_index = 0
    for gx in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for gz in range(-GRID_RADIUS, GRID_RADIUS + 1):
            # Skip centre so the agent spawns in a clearing
            if gx == 0 and gz == 0:
                continue

            x = gx * TREE_SPACING
            z = gz * TREE_SPACING
            log, leaf = TREE_TYPES[tree_index % len(TREE_TYPES)]
            tree_index += 1

            trunk_top = GROUND_Y + TRUNK_HEIGHT
            # Trunk
            parts.append(_cuboid(x, GROUND_Y + 1, z, x, trunk_top, z, log))
            # Leaves
            parts.append(_cuboid(
                x - LEAF_RADIUS, LEAF_Y_START, z - LEAF_RADIUS,
                x + LEAF_RADIUS, LEAF_Y_END,   z + LEAF_RADIUS,
                leaf,
            ))

    return "".join(parts)


class TreeFlatEnv(HumanSurvival):
    """
    Superflat world densely planted with all six wood types on a regular grid.
    Designed as a simplified curriculum environment for log-harvesting tasks.
    """

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "MineRLTreeFlat-v0"
        super().__init__(**kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        # Empty generatorString → Malmo default flat (bedrock/dirt/grass)
        return [FlatWorldGenerator(force_reset=True, generatorString="")]

    def create_server_decorators(self) -> List[Handler]:
        return [DrawingDecorator(_build_tree_xml())]

    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + [
            handlers.InventoryAgentStart({
                0: {"type": "iron_shovel", "quantity": 1},
            })
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=6000),
            handlers.SpawningInitialCondition(allow_spawning=False),  # no hostile mobs
        ]
