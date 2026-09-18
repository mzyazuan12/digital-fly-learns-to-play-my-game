"""Evaluate a frozen or plastic agent against control policies.

Do not describe weight changes alone as evidence of learning. A real claim
needs the intact connectome to beat random, shuffled, rewired, and frozen
controls on held-out episodes of the same stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.train import build_agent, make_env, run_episode
import random

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MaleCNS Shiritori controls.")
    parser.add_argument("--stage", default="tiny")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--control", default="frozen")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=96)
    parser.add_argument("--sequences", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval.json")
    args = parser.parse_args(argv)

    class NS:
        pass

    ns = NS()
    for k, v in vars(args).items():
        setattr(ns, k, v)

    rng = random.Random(args.seed)
    agent, connectome = build_agent(ns)
    agent.plastic = False  # evaluation does not update weights
    env = make_env(ns)
    returns = []
    wins = 0
    valid = 0
    for ep in range(args.episodes):
        summary = run_episode(
            agent, env, window_ms=args.window_ms, epsilon=args.epsilon, rng=rng, max_steps=args.max_steps
        )
        returns.append(summary["return"])
        if summary.get("winner") == "player":
            wins += 1
        if summary.get("valid_event") == "valid_word":
            valid += 1
        print(f"eval ep {ep:4d}  return={summary['return']:7.3f}")

    report = {
        "stage": args.stage,
        "control": args.control,
        "episodes": args.episodes,
        "mean_return": sum(returns) / max(1, len(returns)),
        "win_rate": wins / max(1, args.episodes),
        "neurons": connectome.n,
        "edges": connectome.n_edges,
        "learning_demonstrated": False,
        "note": "Compare this file against random/shuffled/rewired/frozen siblings before any learning claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
