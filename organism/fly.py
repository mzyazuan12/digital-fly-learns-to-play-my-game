"""One persistent virtual Drosophila.

The fly is not a Shiritori agent. It is an individual with a MaleCNS-topology
brain, a body, and a sensorimotor loop. Tasks are worlds. Weights survive
environment changes unless you explicitly clear plasticity.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome, load_connectome, synthetic_connectome
from flybrain.network import LIFNetwork, LIFParams
from flybrain.plasticity import RewardModulatedPlasticity
from organism.assumptions import as_dict as assumption_dict
from organism.body import Body, Pose, build_body
from organism.bridge import MotorBridge
from organism.channels import SensorimotorChannels
from organism.developer import DeveloperControls
from organism.loop import SensorimotorLoop
from organism.physiology import Physiology
from organism.provenance import ProvenanceLog
from organism.toy import locomotor_connectome


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FlyIdentity:
    fly_id: str
    created_at: str
    seed: int
    connectome_dataset: str
    connectome_n: int
    connectome_edges: int
    body_kind: str = "unattached"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "fly_id": self.fly_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "connectome_dataset": self.connectome_dataset,
            "connectome_n": self.connectome_n,
            "connectome_edges": self.connectome_edges,
            "body_kind": self.body_kind,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FlyIdentity":
        return cls(
            fly_id=payload["fly_id"],
            created_at=payload["created_at"],
            seed=int(payload["seed"]),
            connectome_dataset=payload["connectome_dataset"],
            connectome_n=int(payload["connectome_n"]),
            connectome_edges=int(payload["connectome_edges"]),
            body_kind=payload.get("body_kind", "unattached"),
            notes=list(payload.get("notes", [])),
        )


class VirtualFly:
    """A reusable individual. Put it in any world. Do not reset it for a new task."""

    def __init__(
        self,
        connectome: Connectome,
        *,
        seed: int = 0,
        params: LIFParams | None = None,
        fly_id: str | None = None,
        created_at: str | None = None,
        channels: SensorimotorChannels | None = None,
    ):
        params = params or LIFParams(dt=1.0)
        self.connectome = connectome
        self.net = LIFNetwork(connectome, params=params, seed=seed)
        self.channels = channels or SensorimotorChannels.from_connectome(connectome, seed=seed)
        self.plasticity = RewardModulatedPlasticity(self.net)
        self.bridge = MotorBridge(connectome)
        self.physiology = Physiology(seed=seed)
        self.identity = FlyIdentity(
            fly_id=fly_id or str(uuid.uuid4()),
            created_at=created_at or _utc_now(),
            seed=seed,
            connectome_dataset=str(connectome.report.get("dataset_id", "unknown")),
            connectome_n=connectome.n,
            connectome_edges=connectome.n_edges,
        )
        self.history: list[dict] = []
        self.provenance = ProvenanceLog()
        self.body: Body | None = None
        self.world = None
        self.loop: SensorimotorLoop | None = None
        self.developer = DeveloperControls(self)
        self.history.append(
            {
                "event": "created",
                "at": self.identity.created_at,
                "dataset": self.identity.connectome_dataset,
            }
        )

    @classmethod
    def hatch(
        cls,
        *,
        seed: int = 0,
        connectome: str | Connectome = "synthetic",
        params: LIFParams | None = None,
    ) -> "VirtualFly":
        if isinstance(connectome, Connectome):
            graph = connectome
        elif connectome in {"synthetic", "toy", "synthetic_locomotor"}:
            graph = locomotor_connectome(seed=seed)
        elif connectome in {"synthetic_chain", "unit"}:
            graph = synthetic_connectome(n=64, extra_edges=20, seed=seed)
        elif connectome in {"malecns", "malecns_v1", "full"}:
            graph = load_connectome(progress=True)
        else:
            raise ValueError(f"Unknown connectome '{connectome}'")
        return cls(graph, seed=seed, params=params)

    def inhabit(
        self,
        world,
        *,
        physics: bool = False,
        spawn: Pose | None = None,
        brain_ticks: int = 10,
        physics_substeps: int | None = None,
    ) -> SensorimotorLoop:
        """Place this individual in a world without clearing neural state."""
        spawn = spawn or Pose()
        self.body = build_body(physics=physics, name="virtual_fly", spawn=spawn)
        self.world = world
        if hasattr(self.body, "world"):
            self.body.world = world
        self.identity.body_kind = self.body.kind
        window_s = brain_ticks * self.net.params.dt / 1000.0
        if hasattr(self.body, "dt_s"):
            self.body.dt_s = window_s
        substeps = physics_substeps
        if substeps is None:
            substeps = 1
            if physics and hasattr(self.body, "sim"):
                substeps = max(1, int(round(window_s / float(self.body.sim.timestep))))
        self.loop = SensorimotorLoop(
            self, brain_ticks=brain_ticks, physics_substeps=substeps
        )
        self.history.append(
            {
                "event": "inhabit",
                "world": world.name,
                "body": self.body.kind,
                "at": _utc_now(),
            }
        )
        return self.loop

    def detach(self) -> None:
        """Leave the world. Brain, plasticity, and identity stay."""
        if self.world is not None:
            self.history.append(
                {"event": "detach", "world": self.world.name, "at": _utc_now()}
            )
        self.body = None
        self.world = None
        self.loop = None

    def step(self, **kwargs):
        if self.loop is None:
            raise RuntimeError("Call inhabit(world) before stepping")
        return self.loop.step(**kwargs)

    def run(self, n_steps: int):
        if self.loop is None:
            raise RuntimeError("Call inhabit(world) before running")
        return self.loop.run(n_steps)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "identity.json").write_text(
            json.dumps(
                {
                    **self.identity.to_json(),
                    "lif": {
                        "dt": self.net.params.dt,
                        "v_rest": self.net.params.v_rest,
                        "v_threshold": self.net.params.v_threshold,
                        "tau_m": self.net.params.tau_m,
                        "tau_g": self.net.params.tau_g,
                        "t_ref": self.net.params.t_ref,
                        "delay": self.net.params.delay,
                        "contact_gain": self.net.params.contact_gain,
                    },
                    "assumptions": assumption_dict(),
                    "channels": self.channels.notes,
                    "motor_bridge": self.bridge.notes,
                    "physiology": self.physiology.state.snapshot(),
                    "plasticity": self.plasticity.snapshot(),
                    "history": self.history,
                    "provenance_summary": self.provenance.summary(),
                },
                indent=2,
            )
            + "\n"
        )
        (path / "channels.json").write_text(json.dumps(self.channels.to_jsonable()) + "\n")
        self.net.save(path / "brain.npz")
        np.savez_compressed(
            path / "plasticity.npz",
            eligibility=self.plasticity.eligibility,
            n_updates=np.int64(self.plasticity.n_updates),
            last_reward=np.float64(self.plasticity.last_reward),
            changed_edges=np.int64(self.plasticity.changed_edges),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path | str,
        connectome: Connectome | str | None = None,
    ) -> "VirtualFly":
        path = Path(path)
        identity = json.loads((path / "identity.json").read_text())
        dataset = identity.get("connectome_dataset", "synthetic_locomotor")
        if connectome is None:
            if dataset in {"malecns_v1", "malecns"}:
                graph = load_connectome(progress=False)
            elif dataset == "synthetic":
                graph = synthetic_connectome(
                    n=identity["connectome_n"], extra_edges=20, seed=identity["seed"]
                )
            else:
                graph = locomotor_connectome(seed=identity["seed"])
        elif isinstance(connectome, Connectome):
            graph = connectome
        else:
            graph = VirtualFly.hatch(seed=identity["seed"], connectome=connectome).connectome

        if graph.n != identity["connectome_n"] or graph.n_edges != identity["connectome_edges"]:
            raise ValueError(
                "Connectome shape does not match this individual: "
                f"saved n={identity['connectome_n']} edges={identity['connectome_edges']}, "
                f"loaded n={graph.n} edges={graph.n_edges}"
            )

        lif = identity.get("lif", {})
        params = LIFParams(
            dt=float(lif.get("dt", 1.0)),
            v_rest=float(lif.get("v_rest", -52.0)),
            v_threshold=float(lif.get("v_threshold", -45.0)),
            tau_m=float(lif.get("tau_m", 20.0)),
            tau_g=float(lif.get("tau_g", 5.0)),
            t_ref=float(lif.get("t_ref", 2.2)),
            delay=float(lif.get("delay", 1.8)),
            contact_gain=float(lif.get("contact_gain", 0.275)),
        )
        channels = SensorimotorChannels.from_jsonable(
            json.loads((path / "channels.json").read_text())
        )
        fly = cls(
            graph,
            seed=identity["seed"],
            params=params,
            fly_id=identity["fly_id"],
            created_at=identity["created_at"],
            channels=channels,
        )
        fly.identity = FlyIdentity.from_json(identity)
        fly.history = list(identity.get("history", []))
        fly.net.load(path / "brain.npz")
        plastic = np.load(path / "plasticity.npz")
        fly.plasticity.eligibility = plastic["eligibility"]
        fly.plasticity.n_updates = int(plastic["n_updates"])
        fly.plasticity.last_reward = float(plastic["last_reward"])
        fly.plasticity.changed_edges = int(plastic["changed_edges"])
        if identity.get("physiology"):
            fly.physiology.state.load(identity["physiology"])
        fly.history.append({"event": "loaded", "from": str(path), "at": _utc_now()})
        return fly
