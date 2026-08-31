# GPU Training Preflight

Date: 2026-08-03

This document records the reproducible calibration performed before any learned
policy is trained. It is not a learning result.

## Completed CPU measurement

The command below uses the full health wrapper, random actions, and separate
MuJoCo worker processes.

```bash
python scripts/calibrate_training_stack.py \
  --environment-id Pusher-v5 --workers 8 12 16 20 --steps-per-worker 5000
```

| Workers | Pusher aggregate steps/s |
|---:|---:|
| 8 | 52,703 |
| 12 | 60,345 |
| 16 | 72,209 |
| 20 | 77,487 |

The initial learner setting is **8 workers**. Increase to 12 only after the
first learner run confirms that system RAM, rollout latency, and unrelated host
workloads remain healthy. The CPU-only peak at 20 workers is not a training
recommendation because the learner and operating system also need CPU capacity.

### GPU-isolated host confirmation

The workstation subsequently ran the same preflight with
`CUDA_VISIBLE_DEVICES=""`, while another project occupied the GPU. This is the
relevant shared-workload measurement:

| Workers | Pusher aggregate steps/s |
|---:|---:|
| 8 | 40,745 |
| 12 | 46,667 |

PyTorch `2.11.0+cu128` reported CUDA runtime `12.8`. CUDA was intentionally
hidden, so `torch_cuda_unavailable` is the expected result and confirms that
this CPU-only measurement created no CUDA context. Although 12 workers was
14.5% faster in this snapshot, keep **8 workers** as the initial learner value
while the workstation is shared.

Reacher-v5 reached 13,388, 18,846, 35,882, and 65,467 aggregate steps/s at 1,
2, 4, and 8 workers respectively. Pusher is the conservative primary timing
reference.

## Current block on GPU smoke training

In the current Codex execution environment:

- `nvidia-smi` cannot communicate with the NVIDIA driver;
- `.venv-mujoco` has no PyTorch installation;
- therefore CUDA availability and GPU matrix throughput cannot be measured, and
  no learner has been started.

This may mean that the sandbox lacks GPU passthrough even if the workstation
itself has a working RTX 4090. It is not evidence that the physical GPU is
faulty.

## Required pass condition

Run the following from the GPU-visible workstation environment after confirming
that a short calibration job is acceptable alongside other GPU users:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/calibrate_training_stack.py \
  --environment-id Pusher-v5 --workers 8 12 --steps-per-worker 5000
```

The preflight passes only when the final command reports `cuda_probe.available:
true`. Its default CUDA matrix probe is intentionally small (two 2048-square
operands, approximately tens of MB of tensor storage) and performs no learning.

## Next action after pass

Create the GPU learner environment, run a 1--3 hour PPO/SAC smoke training at
8 workers, and record environment steps/s, peak VRAM, success/return trace,
checkpoint-resume behavior, and per-lifetime logging integrity. Do not launch a
full hyperparameter sweep before this smoke run passes.
