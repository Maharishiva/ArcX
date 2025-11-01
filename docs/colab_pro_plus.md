# Colab Pro+ Quickstart (ARC-AGI PPO)

## 1. Launch a GPU Runtime
- Open https://colab.research.google.com and create a new notebook.
- `Runtime ▶ Change runtime type` → set **Hardware accelerator** to **GPU** and pick an **A100** (available on Pro+).
- (Optional) Increase **GPU RAM** if the option appears.

## 2. Install Dependencies
```bash
%%capture
!sudo apt-get update
!pip install -U pip
!pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install flax optax orbax tensorstore imageio einops
```
> **Tip:** Restart the runtime after installing to ensure JAX sees the GPU.

## 3. Fetch the Project
```bash
!git clone https://github.com/<your-account>/arc-agi.git
%cd arc-agi
```
If you are testing this branch before merging, swap the repository/branch accordingly.

## 4. Run the A100 Training Preset
```bash
!PYTHONPATH=. python scripts/ppo_train.py \
    --preset a100 \
    --device cuda \
    --total-updates 1000 \
    --eval-interval 50
```
- `--preset a100` expands the rollout batch (64 envs × 128 steps), uses 1e-4 LR, 16 minibatches, and checkpoints under `checkpoints/ppo_a100`.
- `--device cuda` ensures JAX uses the attached GPU; omit or set `auto` to rely on environment defaults.
- You can further tune `--total-updates`, `--rollout-length`, etc., by passing explicit values (which will override the preset).

## 5. Qualitative Evaluation (GIFs)
After a run finishes (or mid-training if checkpoints exist):
```bash
!PYTHONPATH=. python scripts/ppo_eval_viz.py \
    --checkpoint checkpoints/ppo_a100/ppo_1000 \
    --device cuda \
    --num-episodes 4 \
    --rollout-horizon 64 \
    --output-dir artifacts/ppo_eval
```
Downloads: `from google.colab import files; files.download('artifacts/ppo_eval/episode_000.gif')`.

## 6. Debug-Scale Smoke Test
To ensure the environment is wired up before long runs:
```bash
!PYTHONPATH=. python scripts/ppo_train.py --preset debug --total-updates 2 --num-envs 2 --rollout-length 4 --device cuda
```

## 7. Pushing Results
- Commit checkpoints/GIFs sparingly; prefer uploading to cloud storage if they are large.
- Use `git status` to verify only intended files (typically under `docs/`, `scripts/`, or summary notebooks) are staged.
- When satisfied, follow the PR guide in the main repository (e.g., `git checkout -b feature/ppo-a100`, `git push -u origin feature/ppo-a100`).
