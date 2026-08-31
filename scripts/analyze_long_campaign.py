#!/usr/bin/env python3
"""Aggregate the 2M thermal confirmatory campaigns with paired bootstrap CIs."""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parent.parent

def load(pattern: str):
    out = {}
    for p in glob.glob(str(ROOT / pattern)):
        seed = p.split("seed")[1].split("-")[0]
        x = json.loads(Path(p).read_text())["task_episode_evaluation"]
        out[seed] = x
    return out

def load_reevaluation(pattern: str, expected_task_episodes: int):
    out = load(pattern)
    if len(out) != 5:
        raise SystemExit(f"expected 5 reevaluated seeds for {pattern}, found {len(out)}")
    invalid = {
        seed: value.get("task_episodes")
        for seed, value in out.items()
        if value.get("task_episodes") != expected_task_episodes
    }
    if invalid:
        raise SystemExit(
            f"evaluation protocol mismatch for {pattern}: {invalid}; "
            f"expected {expected_task_episodes} task episodes"
        )
    return out

def bootstrap_mean(x, rng, n=20000):
    x = np.asarray(x, dtype=float)
    draws = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

def group_summary(d, rng):
    reward = np.array([v["mean_task_episode_reward"] for v in d.values()])
    load = np.array([v["mean_episode_end_thermal_load"] for v in d.values()])
    eff = np.array([v["mean_episode_end_efficiency"] for v in d.values()])
    return {"n": len(reward), "reward_mean": float(reward.mean()),
            "reward_sd": float(reward.std(ddof=1)), "reward_ci95": bootstrap_mean(reward, rng),
            "thermal_load_mean": float(load.mean()), "thermal_load_sd": float(load.std(ddof=1)),
            "efficiency_mean": float(eff.mean()), "efficiency_sd": float(eff.std(ddof=1)),
            "seeds": sorted(d)}

def main():
    rng = np.random.default_rng(20260812)
    expected_task_episodes = 1000
    groups = {
        "endogenous_episode": load_reevaluation("outputs/thermal_endogenous_long/*episode-seed*-steps2000k/task_episode_evaluation.json", expected_task_episodes),
        "endogenous_lifetime": load_reevaluation("outputs/thermal_endogenous_long/*lifetime-seed*-steps2000k/task_episode_evaluation.json", expected_task_episodes),
        "matched_exogenous_episode": load_reevaluation("outputs/thermal_exogenous_matched_long/*episode-seed*-steps2000k/task_episode_evaluation.json", expected_task_episodes),
        "matched_exogenous_lifetime": load_reevaluation("outputs/thermal_exogenous_matched_long/*lifetime-seed*-steps2000k/task_episode_evaluation.json", expected_task_episodes),
    }
    summary = {k: group_summary(v, rng) for k, v in groups.items()}
    seeds = sorted(set.intersection(*(set(v) for v in groups.values())))
    end = np.array([groups["endogenous_lifetime"][s]["mean_task_episode_reward"] - groups["endogenous_episode"][s]["mean_task_episode_reward"] for s in seeds])
    exo = np.array([groups["matched_exogenous_lifetime"][s]["mean_task_episode_reward"] - groups["matched_exogenous_episode"][s]["mean_task_episode_reward"] for s in seeds])
    interaction = end - exo
    contrasts = {}
    for name, x in [("endogenous_lifetime_minus_episode", end), ("matched_exogenous_lifetime_minus_episode", exo), ("interaction_endogenous_minus_exogenous", interaction)]:
        contrasts[name] = {"values": x.tolist(), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci95": bootstrap_mean(x, rng), "t_p": float(ttest_1samp(x, 0).pvalue), "wilcoxon_p": float(wilcoxon(x).pvalue)}
    report = {"phase": "thermal_2m_integrated_analysis", "evaluation_protocol": f"independent reevaluation ({expected_task_episodes} task episodes per run)", "groups": summary, "paired_seeds": seeds, "contrasts": contrasts}
    out = ROOT / "outputs/thermal_2m_integrated_analysis.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    md = ["# Thermal 2M integrated analysis", "", "| Group | Reward mean | SD | 95% bootstrap CI | Thermal load | Efficiency |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in summary.items(): md.append(f"| {k} | {v['reward_mean']:.3f} | {v['reward_sd']:.3f} | [{v['reward_ci95'][0]:.3f}, {v['reward_ci95'][1]:.3f}] | {v['thermal_load_mean']:.4f} | {v['efficiency_mean']:.4f} |")
    md += ["", "| Contrast | Mean | SD | 95% bootstrap CI | t-test p | Wilcoxon p |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in contrasts.items(): md.append(f"| {k} | {v['mean']:.3f} | {v['sd']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['t_p']:.4f} | {v['wilcoxon_p']:.4f} |")
    (ROOT / "outputs/thermal_2m_integrated_analysis.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
