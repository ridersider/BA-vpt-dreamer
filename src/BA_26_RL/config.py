import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PPOConfig:
    # Rollout
    n_steps: int = 512

    # PPO hyperparameters
    n_epochs: int = 3
    batch_size: int = 64
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 5.0
    target_kl: float = 0.02

    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.0

    # Model paths (inside Docker container)
    model_path: str = "/workspace/models/foundation-model-1x.model"
    weights_path: str = "/workspace/models/ppo_finetuned.weights"
    out_weights_path: str = "/workspace/models/ppo_finetuned.weights"

    # Training length
    total_timesteps: int = 50_000

    # Checkpointing
    checkpoint_every_n_updates: int = 10

    # Live stream
    stream_port: int = 8080
    stream_fps: int = 10

    # Video / logging
    record_every_n_updates: int = 10
    video_fps: int = 20
    log_dir: str = field(default_factory=lambda: f"/workspace/logs/run_{int(time.time())}")
