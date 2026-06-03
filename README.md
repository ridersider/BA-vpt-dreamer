# VPT Finetuning

Fine-tuning of OpenAI's Video PreTraining (VPT) agent on Minecraft tasks using PPO, with support for curriculum learning, single-item reward, and navigation training.

---

## 1. Setup

### Prerequisites

- Docker and Docker Compose (for local development)
- Podman + SLURM (for cluster training)
- VPT model files (`.model` + `.weights`) placed in `./models/`

### Download the foundation model

The project uses the 1x-width foundational model by default (`foundation-model-1x`). Download both the architecture file and the weights from OpenAI's public storage:

```bash
mkdir -p models
wget -P models https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model
wget -P models https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights
```

The paths are referenced in `PPOConfig` via `model_path`, `weights_path`, and `pretrained_weights_path` (all default to `/workspace/models/foundation-model-1x.*`). To use a larger variant (2x or 3x), download the corresponding `.model` and `.weights` files and point the config to them.

Other available variants (see `repos/vpt/README.md` for the full list):

| Variant | `.model` file | `.weights` file |
|---------|--------------|-----------------|
| 1x Foundation (default) | `foundation-model-1x.model` | `foundation-model-1x.weights` |
| 2x Foundation | `2x.model` | `foundation-model-2x.weights` |
| 3x Foundation | `foundation-model-3x.model` | `foundation-model-3x.weights` |
| 2x RL from Foundation | `2x.model` | `rl-from-foundation-2x.weights` |

### Build the container

```bash
docker compose build
```

The image is based on Ubuntu 20.04 (`linux/amd64` — required for Minecraft's LWJGL2 natives). The build compiles Minecraft/Malmo via Gradle and takes **5–15 minutes on first run**; subsequent builds use the layer cache.

### Start the container (interactive shell)

```bash
docker compose up -d
docker compose exec vpt-dreamer bash
```

The following directories are bind-mounted into the container:

| Host path   | Container path      | Purpose                          |
|-------------|---------------------|----------------------------------|
| `./models/` | `/workspace/models` | VPT `.model` and `.weights` files |
| `./src/`    | `/workspace/src`    | Training source code (live reload) |
| `./logs/`   | `/workspace/logs`   | TensorBoard logs and checkpoints  |

### Run training (inside container)

```bash
cd /workspace/src
python train.py                                          # sequential curriculum (default)
python train.py --mode single --single_task_items log   # single-item reward
python train.py --mode navigation                        # XZ navigation
```

TensorBoard:

```bash
tensorboard --logdir /workspace/logs --port 6006        # accessible at localhost:6006
```

A live video stream (MJPEG) is available at `http://localhost:8080` while training runs.

---

## 2. Experiments

All training behaviour is controlled by `src/config.py` and can be overridden via CLI flags to `train.py`. The central class is `PPOConfig`.

### Training modes (`mode`)

| Mode | Description |
|------|-------------|
| `sequential` | Trains the agent through the ordered task curriculum defined in `TASKS`. When a task is completed the agent advances to the next one automatically. Per-task weights are saved and reloaded on resumption. |
| `single` | Flat reward for a fixed set of items. Items are specified via `single_task_items` (must exist in `MASTER_REWARD_TABLE`). |
| `navigation` | Distance-delta reward for reaching a random XZ target. No item collection. The target is sampled each episode from a disk of radius `nav_target_radius` around spawn. A red wool tower marks the goal. |

### Curriculum (`TASKS`)

The curriculum is the ordered list `TASKS` in `config.py`. Each `TaskSpec` defines:

| Field | Description                                                                                                             |
|-------|-------------------------------------------------------------------------------------------------------------------------|
| `name` | Short identifier, used in filenames and logs                                                                            |
| `display_name` | Human-readable label                                                                                                    |
| `item_reward_keys` | Items from `MASTER_REWARD_TABLE` to reward                                                                              |
| `goal_key` | `"reward_based"` → task ends when max reward is accumulated; otherwise → task ends when the named item is first obtained |
| `action_reward_key` | Optional: key press (e.g. `"inventory"`) that gives a small reward per step while the UI is open                        |
| `disable_movement` | If `True`, movement action weights are set to negative values (for crafting tasks)                                   |

Default curriculum:

1. **Obtain Wood** — collect 8 logs
2. **Craft Wooden Items** — craft sticks and a crafting table (movement disabled)
3. **Craft Wooden Pickaxe** — craft a wooden pickaxe (movement disabled)
4. **Obtain Stone** — collect 11 cobblestone
5. **Craft Stone Pickaxe** — craft a stone pickaxe

To modify the curriculum, add,remove or reorder `TaskSpec` entries in `TASKS`. Restart from a specific task using `start_task`.

### Rollout collection (`update_mode`)

| Mode | Behaviour |
|------|-----------|
| `fixed` | Collect exactly `n_steps` environment steps per update, always run PPO update |
| `reward_triggered` | Collect until reward > 0 or `max_steps_no_reward` steps; skip update if no reward was seen |

### PPO hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 256 | Steps per rollout (fixed mode) |
| `n_epochs` | 1 | PPO epochs per update |
| `batch_size` | 64 | Mini-batch size |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip ratio |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 5.0 | Gradient clipping |
| `target_kl` | 0.02 | Early-stop KL threshold per update |
| `learning_rate` | 2e-5 | Adam learning rate |
| `weight_decay` | 0.001 | Adam weight decay |

### KL regularisation towards pretrained policy

To prevent catastrophic forgetting, a frozen copy of the pretrained VPT weights is kept and a KL penalty between the current and the pretrained policy is added to the PPO loss. The coefficient decays over training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kl_pretrained_coef` | 0.1 | Initial KL penalty weight |
| `kl_pretrained_coef_decay` | 0.9995 | Multiplicative decay per update |

### Navigation-specific parameters

Only active when `mode="navigation"`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nav_target_radius` | 100.0 | Max distance of random target from spawn (blocks) |
| `nav_success_radius` | 5.0 | Distance threshold that counts as success (blocks) |
| `nav_reward_scale` | 1.0 | Multiplier on the distance-delta reward |
| `nav_success_bonus` | 100.0 | One-time bonus on reaching the target |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VPT_MODELS` | `/workspace/models` | Directory with `.model` and `.weights` files |
| `VPT_LOGS` | `/workspace/logs` | TensorBoard log and checkpoint directory |
| `VPT_TASK_INVENTORIES` | `/workspace/task_inventories` | Saved inventory snapshots per completed task |

### Example: reproducing a specific experiment

```bash
# Sequential curriculum starting from the stone-collection task
python train.py \
    --mode sequential \
    --update_mode reward_triggered \
    --max_steps_no_reward 4096 \
    --total_timesteps 1000000

# Navigation with a smaller target radius to speed up early learning
python train.py \
    --mode navigation \
    --nav_target_radius 30 \
    --nav_success_radius 5 \
    --nav_success_bonus 50 \
    --total_timesteps 500000
```

---

## 3. Environments

### `SurvivalEnv` (`src/envs/survival_env.py`)

Base environment. Extends MineRL `HumanSurvival` with:

- **Fixed world seed** (`seed=42`) so all runs start in the same world.
- **Configurable start inventory** — pass `start_inventory` as a tuple of `(item_id, quantity)` pairs; used by the curriculum to hand the agent items from the previous task.
- **Drawing decorator support** — allows placing arbitrary blocks at mission start (used by navigation to place a visible goal tower).

### `SurvivalRewardEnv` (`src/envs/survival_env.py`)

Reward wrapper around `SurvivalEnv`. Drives item-based rewards from `MASTER_REWARD_TABLE`:

- Rewards new item pickups up to each item's `max_qty` cap at `reward_per_item` per unit.
- Optionally **keeps inventory across deaths** — the agent's inventory at death is restored at the next reset.
- Logs every pickup event to `item_log` for post analysis.

### `SequentialRewardEnv` (`src/envs/sequential_env.py`)

Curriculum wrapper. Stacks on top of `SurvivalEnv` (without `SurvivalRewardEnv`) and advances through `TASKS` automatically:

- **obtain tasks** complete when accumulated reward reaches `task.max_reward`.
- **craft tasks** complete when the goal item appears in the inventory.
- On task advance, per-task accumulators reset; items already in inventory count as newly collected, so skills carry over.
- `repeat=True` keeps the agent on the same task after completion (used during checkpoint collection).

### `NavigationRewardEnv` (`src/envs/navigation_env.py`)

Reward wrapper for XZ-plane navigation (stacks on top of `SurvivalEnv`):

- **Continuous reward**: potential-based distance shaping `(d_{t-1} - d_t) × reward_scale` (positive when approaching, negative when retreating).
- **Success bonus**: one-time reward when within `success_radius` blocks; episode terminates.
- Target is sampled uniformly in a disk of `target_radius` around spawn each episode.
- Each episode's full trajectory is saved to `<log_dir>/nav_episodes/episode_<N>.json`.
- From episode 2 onward, a red wool tower is placed at the target position (requires `SurvivalEnv`).

### `TreeFlatEnv` (`src/envs/tree_flat_env.py`)

Specialised flat-world environment for isolated log-harvesting experiments:

- Superflat world (bedrock / dirt / grass) — no terrain variation.
- A 9×9 grid of trees (spacing 7 blocks, radius 4 grid cells) is planted at mission start, covering all six wood types in rotation.
- Agent spawns in a clearing at the centre.
- No hostile mob spawning; daytime is fixed at 6000 ticks.
- Agent starts with an iron axe in slot 0.

### Reward table (`MASTER_REWARD_TABLE` in `src/config.py`)

All rewarded items and their caps (based on VPT paper, Table 7):

| Item | Max qty | Reward per unit |
|------|---------|-----------------|
| Log (all variants) | 8 | 0.125 |
| Planks | 20 | 0.05 |
| Stick | 16 | 0.0625 |
| Crafting Table | 1 | 1.0 |
| Cobblestone | 11 | ~0.091 |
| Wooden Pickaxe | 1 | 1.0 |
| Stone Pickaxe | 1 | 1.0 |
| Furnace | 1 | 1.0 |
| Coal | 5 | 0.4 |
| Torch | 16 | 0.125 |
| Iron Ore | 3 | ~1.333 |
| Iron Ingot | 3 | ~1.333 |
| Iron Pickaxe | 1 | 4.0 |
| Diamond | uncapped | ~2.667 |
| Diamond Pickaxe | uncapped | 8.0 |
