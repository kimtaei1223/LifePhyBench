# Free Simulator Plan

Decision checkpoint: 2026-08-03.

## Selected stack

The MVP uses **MuJoCo 3.11.0 + Gymnasium 1.3.0 + Python 3.11**.  MuJoCo is
Apache-2.0 and Gymnasium is MIT licensed.  No paid software is required.

Reasons:

- `mjModel` exposes runtime arrays for contact friction, joint damping and
  friction loss, actuator gain/gear, and actuator force range;
- `mj_resetData` resets simulation state without reconstructing the model, so
  an explicitly managed health state and changed physical parameters can survive
  task resets;
- a separate model per vector worker makes lifetime isolation auditable;
- CPU simulation plus a GPU learner fits the available i7-14700KF, RTX 4090, and
  32 GB RAM better than beginning with a large GPU simulator.

Official references:

- [MuJoCo simulation/programming documentation](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)
- [MuJoCo model arrays](https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html)
- [MuJoCo 3.11.0 release](https://github.com/google-deepmind/mujoco/releases/tag/3.11.0)
- [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/)

## MVP sequence

1. Wrap `Pusher-v5` with `reset_episode` and `reset_lifetime`.
2. Add actuator-efficiency wear from squared control dose.
3. Audit friction wear from tangential contact work. **Pusher rejected:** its
   slider-constrained object did not respond to `geom_friction` mutation.
4. Add recoverable thermal degradation.
5. Verify vector-worker independence and physics-response direction.
6. Train state-based baselines.
7. Extend one conclusion to a Panda/Fetch manipulation task.
8. Reproduce one central result in ManiSkill or Isaac Lab as an engine check.

## Runtime mutations

Candidate physical mappings are:

```text
surface health  -> model.geom_friction[..., 0]
joint aging     -> model.dof_damping / model.dof_frictionloss
actuator health -> model.actuator_forcerange / actuator_gainprm / actuator_gear
```

For force-range degradation, the model must enable actuator force limiting.
Every mutation receives a directionality regression test; changing an array is
not sufficient evidence that the effective dynamics changed as intended.

The Pusher contact-friction audit is a concrete example: because controlled
object velocity did not change with mutated `geom_friction`, no contact-friction
mechanism is claimed in this task. A later contact-rich task must pass the same
test before adding this channel.

## Hardware policy

- Benchmark 8, 12, 16, and 20 CPU workers; begin with 12.
- Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` per worker.
- Reserve the RTX 4090 for policy/sequence-model learning.
- Store state trajectories and scalar logs; render only evaluation subsets.
- Keep RGB-D as a representative secondary experiment because system RAM is
  32 GB.

## Deferred stacks

### ManiSkill

Useful for contact-rich manipulation and RGB-D.  Code and SAPIEN are permissive,
but included assets can have CC BY-NC terms.  Runtime per-environment material
mutation in GPU-vectorized simulation needs explicit validation, so it is not the
MVP dependency.

### Isaac Lab

Offers clear environment-indexed APIs for joint damping, friction, and effort
limits, but has a larger installation/RAM burden and NVIDIA EULA.  It is suitable
for later cross-engine validation rather than the first implementation.
