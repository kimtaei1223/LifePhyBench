# Frame-Stack PPO Development Results

This is a development configuration comparison, not a final benchmark result.
It uses Pusher-v5 with persistent endogenous actuator wear, a lifetime-preserved
finite observation history, PPO, one million training transitions, and five
development seeds. Each checkpoint was deterministically evaluated for 200 task
episodes (10 physical lifetimes).

| Stack size | Seed 1000 | Seed 1001 | Seed 1002 | Seed 1003 | Seed 1004 | Mean across seeds |
|---|---:|---:|---:|---:|---:|---:|
| 4 | -27.18 | -26.06 | -26.70 | -27.92 | -25.67 | -26.71 |
| 8 | -28.33 | -26.40 | -29.55 | -28.87 | -28.98 | -28.43 |

For the paired seed comparison, stack 4 exceeds stack 8 by 1.72 reward points
on average (sample standard deviation 1.29). Stack 4 is therefore selected as
the development finite-history baseline for the next comparison. This selection
does not establish that finite history is sufficient, nor does it support a
paper-level claim before held-out degradation laws and complete-lifetime
uncertainty analysis.

The immediately following training task is to complete the endogenous recurrent
comparison for seeds 1002--1004. The earlier episode/lifetime RNN runs cover
only seeds 1000 and 1001, whereas the selected frame-stack baseline has five
development seeds.
