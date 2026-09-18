"""Closed-loop training. The fly connectome is the policy substrate.

Exploration epsilon, if enabled, is an engineering wrapper around the
decoder — it is not a second neural network choosing words.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from flybrain.loader import load_connectome, synthetic_connectome
from flybrain.network import LIFNetwork, LIFParams
from flybrain.plasticity import RewardModulatedPlasticity
from flybrain.populations import ACTIONS, build_interface
from flybrain.sensory import SensoryEncoder
from shiritori.action_decoder import KeyboardDecoder
from shiritori.environment import ShiritoriEnv
from training.controls import RandomPolicy, apply_control
from training.curriculum import tiny_words
from training.letter_env import LetterEnv
from training.runlog import JsonlLogger

ROOT = Path(__file__).resolve().parents[1]


def _maybe_explore(action: str, epsilon: float, rng: random.Random) -> str:
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return action


def run_episode(agent, env, *, window_ms: float, epsilon: float, rng: random.Random, max_steps: int):
    obs = env.reset() if hasattr(env, "reset") else None
    if obs is None:
        obs = env.observe()
    total = 0.0
    steps = 0
    info_last = {}
    done = False
    while not done and steps < max_steps:
        agent.net.clear_drive()
        agent.encoder.apply(agent.net, obs)
        duration_s = window_ms / 1000.0
        counts = agent.net.run_ms(window_ms)
        raw = agent.decoder.decode(counts, duration_s)
        action = _maybe_explore(raw, epsilon, rng)
        result = env.step(action)
        if isinstance(result, tuple):
            obs, reward, done, info = result
        else:
            obs, reward, done, info = result.observation, result.reward, result.done, result.info
        agent.plasticity.observe_activity()
        if agent.plastic:
            agent.plasticity.apply_reward(reward)
        total += reward
        steps += 1
        info_last = info
    return {
        "return": total,
        "steps": steps,
        "done": done,
        "info": {k: v for k, v in info_last.items() if isinstance(v, (str, int, float, bool))},
        "score": getattr(env, "score", None),
        "winner": getattr(env, "winner", None),
        "valid_event": info_last.get("event"),
    }


class FlyAgent:
    def __init__(self, net: LIFNetwork, interface, plastic: bool):
        self.net = net
        self.interface = interface
        self.encoder = SensoryEncoder(interface)
        self.decoder = KeyboardDecoder(interface)
        self.plasticity = RewardModulatedPlasticity(net)
        self.plastic = plastic


def build_agent(args) -> tuple[FlyAgent, object]:
    if args.synthetic:
        connectome = synthetic_connectome(n=args.synthetic_n, seed=args.seed)
    else:
        connectome = load_connectome(progress=True)
        connectome = apply_control(connectome, args.control, args.seed)
    params = LIFParams(dt=args.dt)
    net = LIFNetwork(connectome, params=params, seed=args.seed)
    interface = build_interface(connectome, seed=args.seed)
    plastic = args.control != "frozen" and args.control != "random"
    return FlyAgent(net, interface, plastic=plastic), connectome


def make_env(args):
    if args.stage == "letters":
        return LetterEnv(seed=args.seed, sequences=args.sequences)
    if args.stage in {"tiny", "tiny20", "fly1"}:
        return ShiritoriEnv.from_words(tiny_words(20), difficulty="casual", seed=args.seed)
    difficulty = "casual"
    if args.stage in {"pro", "fly3"}:
        difficulty = "pro"
    if args.stage in {"featherine", "fly4"}:
        difficulty = "featherine"
    return ShiritoriEnv.from_words(difficulty=difficulty, seed=args.seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a MaleCNS LIF agent on Shiritori stages.")
    parser.add_argument("--stage", default="letters", help="letters|tiny|casual|pro|featherine")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0, help="Training dt in ms (modeling choice).")
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.05, help="Decoder-wrapper exploration.")
    parser.add_argument("--control", default="none", help="none|frozen|shuffled|rewired|random")
    parser.add_argument("--synthetic", action="store_true", help="Use a tiny toy graph, not MaleCNS.")
    parser.add_argument("--synthetic-n", type=int, default=96)
    parser.add_argument("--sequences", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "train.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    logger = JsonlLogger(args.out)
    logger.write(
        {
            "event": "run_start",
            "stage": args.stage,
            "control": args.control,
            "episodes": args.episodes,
            "dt": args.dt,
            "window_ms": args.window_ms,
            "epsilon": args.epsilon,
            "synthetic": args.synthetic,
            "disclaimer": "This run does not by itself demonstrate learning.",
        }
    )

    if args.control == "random":
        env = make_env(args)
        policy = RandomPolicy(args.seed)
        for ep in range(args.episodes):
            obs = env.reset()
            total = 0.0
            done = False
            steps = 0
            while not done and steps < args.max_steps:
                result = env.step(policy.act(obs))
                obs, reward, done, info = result.observation, result.reward, result.done, result.info
                total += reward
                steps += 1
            logger.write({"event": "episode", "episode": ep, "return": total, "steps": steps, "control": "random"})
            print(f"ep {ep:4d}  return={total:7.3f}  steps={steps}")
        logger.close()
        return 0

    agent, connectome = build_agent(args)
    env = make_env(args)
    t0 = time.perf_counter()
    for ep in range(args.episodes):
        summary = run_episode(
            agent,
            env,
            window_ms=args.window_ms,
            epsilon=args.epsilon,
            rng=rng,
            max_steps=args.max_steps,
        )
        snap = agent.plasticity.snapshot() if agent.plastic else {}
        logger.write(
            {
                "event": "episode",
                "episode": ep,
                "neurons": connectome.n,
                "edges": connectome.n_edges,
                **summary,
                **snap,
            }
        )
        print(
            f"ep {ep:4d}  return={summary['return']:7.3f}  steps={summary['steps']:3d}  "
            f"changed_synapses={snap.get('changed_edges', 0)}"
        )
        if args.checkpoint and (ep + 1) % 10 == 0:
            agent.net.save(args.checkpoint)
    logger.write({"event": "run_end", "seconds": round(time.perf_counter() - t0, 3)})
    logger.close()
    if args.checkpoint:
        agent.net.save(args.checkpoint)
        meta = {
            "interface": agent.interface.notes,
            "learning_demonstrated": False,
            "control": args.control,
            "stage": args.stage,
        }
        args.checkpoint.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
