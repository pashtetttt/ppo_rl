# PPO vs REINFORCE on Atari Pong

Reinforcement learning project comparing **REINFORCE** (vanilla policy gradient) and **PPO** (Proximal Policy Optimization) on the classic Atari game **Pong**, with **Pygame** visualization.

## Environment

- **Game**: Atari Pong — the agent controls **one paddle**; the **other paddle is the built-in game AI** (the “computer”). So the agent is always playing against the same opponent.
- **Observation**: Stack of the last **4 grayscale frames** resized to **84×84** (DQN-style preprocessing). The agent never sees the full-color game; it sees only this small grayscale image.
- **Actions**: 6 discrete actions (NOOP, FIRE, RIGHT, LEFT, etc.).
- **Episode**: Runs until one player scores **21 points** (standard Pong rules).
- **Reward**: Sparse and delayed: **+1** when the agent wins a rally (scores a point), **−1** when it loses a rally. Hundreds of steps can pass with zero reward, making credit assignment difficult.

### How training works (who plays whom)

During training, **the agent plays against the Atari game’s built-in AI**. Each “step” is a few frames (frame skip 4). The agent only gets a reward when a point is scored: +1 if the agent scored, −1 if the opponent scored. There is no human in the loop; the policy is updated from these rewards (REINFORCE or PPO).

### Why grayscale (and why no ball coordinates)?

- **Grayscale**: The agent is trained “from pixels” on purpose: it gets a small 84×84 image (and a 4-frame stack for motion). Using grayscale keeps the input small and matches the usual Atari/RL setup (e.g. DQN). Color doesn’t help for Pong (paddle, ball, walls are clear in grayscale), and the CNN learns to spot the ball and paddles from these pixels.
- **No ball coordinates**: We do **not** feed (x, y) of the ball. The task is **pixel-based**: the policy must learn from the image. Giving coordinates would be a different, easier task (state-based RL). Here the network has to learn a visual representation itself, which is part of the challenge and the point of the project.

## Why PPO over REINFORCE?

- **REINFORCE** uses full-episode returns and a baseline; it suffers from **high variance** and poor credit assignment with long delayed reward, so it often fails to learn Pong reliably.
- **PPO** uses:
  - **Clipped surrogate objective** to limit policy updates and avoid instability.
  - **Generalized Advantage Estimation (GAE)** to reduce variance while keeping bias manageable.
  - **Multiple epochs** over minibatches of data for better sample efficiency.

Together, this makes PPO more **stable** and **sample-efficient**, and it can learn to play Pong where naive REINFORCE typically fails.

### What the training visualization shows

When you run with `--visualize`, the Pygame window shows:

- **Left area (“game” box)**: The **agent’s input** — one 84×84 grayscale frame (from the 4-frame stack), scaled up. This is the preprocessed observation the policy uses, **not** the full-color Atari game. So you see a small, gray version of the screen.
- **Right panel**: Score (agent vs opponent), episode/step, **reward (return)** over recent episodes, and **policy loss**. So you are monitoring learning (returns and loss), not watching the literal game.

The real Atari game is not rendered during training (to keep it fast and simple). To **watch the trained agent play the actual Pong game**, use `play_pong.py` (see below).

**Display fix:** If the left “game” box looked almost black, that was because the env returns observations in [0, 1]; the viewer now scales that to [0, 255] so the agent’s grayscale view is visible.

### Why is average return still about -20 after 80k steps?

- **Return** = (agent points − opponent points) per game (to 21). So −20 means the agent is losing almost every point and is normal early in training.
- **Pong from pixels is hard:** The agent must learn from sparse reward (+1/−1 only when a point is scored), with hundreds of zero-reward steps per game. Credit assignment is difficult.
- **80k steps is still early.** Many setups need **1–2 million steps** (or more) before average return turns positive. So slow progress at 80k is expected; keep training (e.g. `--timesteps 2000000`).
- If progress stays flat for a long time, you can try: slightly lower learning rate (e.g. `--lr 1e-4`), longer runs, or checking that the training window’s “agent view” now shows a visible grayscale game (so the pipeline is correct).

## Setup

```bash
cd "your_path/atari"
pip install -r requirements.txt
```

**Note**: `ale-py` includes the Atari ROMs. On first use, the environment may need to fetch or use bundled ROMs.

## Project structure

| File | Description |
|------|-------------|
| `env_utils.py` | Builds Pong env with 84×84 grayscale and 4-frame stack. |
| `models.py` | CNN feature extractor and actor-critic heads (PyTorch). |
| `agents.py` | REINFORCE and PPO agents (REINFORCE with baseline; PPO with GAE + clipping). |
| `visualize.py` | Pygame window: game view, score, episode returns, loss curves. |
| `train.py` | Training script: REINFORCE and/or PPO with optional live visualization. |
| `run_eval.py` | Load a checkpoint and run evaluation with Pygame (shows agent’s 84×84 input + stats). |
| `play_pong.py` | **Watch the trained agent in the real Atari Pong game** (native game window). |

## Usage

### Train with live visualization (Pygame)

```bash
# Train PPO with on-screen visualization
python train.py --algorithm ppo --visualize --timesteps 500000

# Train REINFORCE with visualization
python train.py --algorithm reinforce --visualize --episodes 500

# Train both (REINFORCE first, then PPO) and compare
python train.py --algorithm both --visualize --episodes 300 --timesteps 300000
```

### Train without display (e.g. headless server)

```bash
python train.py --algorithm ppo --timesteps 500000 --no-viz
```

### Evaluate a saved model (Pygame stats + agent’s view)

```bash
python run_eval.py --checkpoint ppo_pong.pt --algorithm ppo --episodes 5
```

This shows the **agent’s 84×84 grayscale input** (what the policy “sees”), plus score and return graphs. It is not the full Atari game screen.

### Watch the trained agent in the real Atari Pong game

After training, you can run the policy inside the **actual Atari Pong** window (the real game, not the small grayscale view):

```bash
python play_pong.py --checkpoint ppo_pong.pt --algorithm ppo --episodes 3
```

A separate window will open with the classic Pong game; the agent controls the right paddle and plays until the chosen number of episodes (games to 21 points) finish.

Checkpoints are saved as `ppo_pong.pt` and `reinforce_pong.pt` when using `--algorithm both`.

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `ppo` | `reinforce`, `ppo`, or `both` |
| `--visualize` | off | Show Pygame window during training |
| `--episodes` | 500 | Number of episodes (REINFORCE) |
| `--timesteps` | 500000 | Total environment steps (PPO) |
| `--ppo-steps` | 2048 | Rollout length per PPO update |
| `--lr` | 2.5e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--device` | cuda/cpu | PyTorch device |

## Implementation details

- **CNN**: 3 conv layers (32, 64, 64 filters) + MLP (512) for policy and value.
- **REINFORCE**: One update per episode; baseline = value function; gradient clipping.
- **PPO**: GAE (λ=0.95), clip ε=0.1, 4 epochs per rollout, minibatch size 64.
- **Optimizer**: Adam with default β₁, β₂.

Enjoy training and watching the agent learn to play Pong.
