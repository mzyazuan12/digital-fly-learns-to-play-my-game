"""One persistent virtual Drosophila.

Birth instantiates a NEW digital individual from measured anatomy. MaleCNS
does not contain the specimen's live physiological state, so this is not
resurrection of that animal's mind.

The fly is not a Shiritori agent. Tasks are worlds. Neural, synaptic,
neuromodulatory and metabolic state survive environment changes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome, load_connectome, synthetic_connectome
from flybrain.mushroom_body import MushroomBodyLearning
from flybrain.network import LIFNetwork, LIFParams
from organism.assumptions import as_dict as assumption_dict
from organism.body import Body, Pose, build_body
from organism.bridge import MotorBridge
from organism.channels import SensorimotorChannels
from organism.config import (
    BIRTH_DEFINITION,
    DEFAULT_MOTOR_MODE,
    LEGACY_SCAFFOLD,
    MODEL_VERSION,
    MotorMode,
)
from organism.developer import DeveloperControls
from organism.loop import SensorimotorLoop
from organism.motor_map import MotorNeuronMuscleMap
from organism.physiology import Physiology
from organism.provenance import ProvenanceLog
from organism.toy import locomotor_connectome


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


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
    model_version: str = MODEL_VERSION
    motor_mode: str = DEFAULT_MOTOR_MODE.value
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
            "model_version": self.model_version,
            "motor_mode": self.motor_mode,
            "birth_definition": BIRTH_DEFINITION,
            "consciousness_claimed": False,
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
            model_version=str(payload.get("model_version", MODEL_VERSION)),
            motor_mode=str(payload.get("motor_mode", DEFAULT_MOTOR_MODE.value)),
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
        motor_mode: MotorMode | str | None = None,
        legacy_scaffold: bool | None = None,
        save_dir: Path | str | None = None,
    ):
        params = params or LIFParams(dt=1.0)
        self.connectome = connectome
        self.net = LIFNetwork(connectome, params=params, seed=seed)
        self.channels = channels or SensorimotorChannels.from_connectome(connectome, seed=seed)
        self.learning = MushroomBodyLearning(self.net)
        self.plasticity = self.learning
        self.legacy_scaffold = LEGACY_SCAFFOLD if legacy_scaffold is None else bool(legacy_scaffold)
        self.bridge = MotorBridge(connectome, legacy_scaffold=self.legacy_scaffold)
        self.physiology = Physiology(seed=seed, legacy_scaffold=self.legacy_scaffold)
        self.motor_map = MotorNeuronMuscleMap(connectome)
        if motor_mode is None:
            self.motor_mode = DEFAULT_MOTOR_MODE
        elif isinstance(motor_mode, MotorMode):
            self.motor_mode = motor_mode
        else:
            self.motor_mode = MotorMode(motor_mode)
        self.motor_report: dict = {}
        self.identity = FlyIdentity(
            fly_id=fly_id or str(uuid.uuid4()),
            created_at=created_at or _utc_now(),
            seed=seed,
            connectome_dataset=str(connectome.report.get("dataset_id", "unknown")),
            connectome_n=connectome.n,
            connectome_edges=connectome.n_edges,
            motor_mode=self.motor_mode.value,
        )
        self.history: list[dict] = []
        self.provenance = ProvenanceLog()
        self.body: Body | None = None
        self.world = None
        self.loop: SensorimotorLoop | None = None
        self.developer = DeveloperControls(self)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.history.append(
            {
                "event": "created",
                "at": self.identity.created_at,
                "dataset": self.identity.connectome_dataset,
                "birth": BIRTH_DEFINITION,
            }
        )
        if self.save_dir is not None:
            self._init_life_history()

    @classmethod
    def hatch(
        cls,
        *,
        seed: int = 0,
        connectome: str | Connectome = "synthetic",
        params: LIFParams | None = None,
        motor_mode: MotorMode | str | None = None,
        legacy_scaffold: bool | None = None,
    ) -> "VirtualFly":
        graph = cls._load_graph(connectome, seed=seed)
        return cls(
            graph,
            seed=seed,
            params=params,
            motor_mode=motor_mode,
            legacy_scaffold=legacy_scaffold,
        )

    @classmethod
    def birth(
        cls,
        individual_id: str,
        *,
        seed: int = 0,
        connectome: str | Connectome = "synthetic",
        directory: Path | str | None = None,
        motor_mode: MotorMode | str | None = None,
        legacy_scaffold: bool = False,
    ) -> "VirtualFly":
        """Create a new persistent individual. Does not resurrect the EM specimen."""
        root = Path(directory) if directory is not None else Path("individuals") / individual_id
        graph = cls._load_graph(connectome, seed=seed)
        fly = cls(
            graph,
            seed=seed,
            fly_id=individual_id,
            motor_mode=motor_mode,
            legacy_scaffold=legacy_scaffold,
            save_dir=root,
        )
        fly.save(root)
        fly.log_life(
            "birth",
            {
                "seed": seed,
                "dataset": fly.identity.connectome_dataset,
                "n": fly.connectome.n,
                "model_version": MODEL_VERSION,
            },
        )
        return fly

    @staticmethod
    def _load_graph(connectome: str | Connectome, *, seed: int) -> Connectome:
        if isinstance(connectome, Connectome):
            return connectome
        if connectome in {"synthetic", "toy", "synthetic_locomotor"}:
            return locomotor_connectome(seed=seed)
        if connectome in {"synthetic_chain", "unit"}:
            return synthetic_connectome(n=64, extra_edges=20, seed=seed)
        if connectome in {"malecns", "malecns_v1", "full"}:
            return load_connectome(progress=True)
        raise ValueError(f"Unknown connectome '{connectome}'")

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
        self.log_life("inhabit", {"world": world.name, "body": self.body.kind})
        return self.loop

    def detach(self) -> None:
        """Leave the world. Brain, plasticity, and identity stay."""
        if self.world is not None:
            self.history.append(
                {"event": "detach", "world": self.world.name, "at": _utc_now()}
            )
            self.log_life("detach", {"world": self.world.name})
        self.body = None
        self.world = None
        self.loop = None

    def lesion(self, pathway: str, silent: bool = True) -> np.ndarray:
        """Silence an identified pathway. Restoring it is the scientific control."""
        names = {
            "walk": self.bridge.walk_indices,
            "walk_initiation": self.bridge.walk_indices,
            "DNp09": self.bridge.walk_indices,
            "steer_left": self.bridge.steer_left,
            "steer_right": self.bridge.steer_right,
        }
        if pathway not in names:
            raise KeyError(f"Unknown pathway '{pathway}'")
        idx = names[pathway]
        self.net.lesion(idx, silent=silent)
        if silent:
            self.bridge.walk_trace = 0.0 if pathway in {"walk", "walk_initiation", "DNp09"} else self.bridge.walk_trace
        self.log_life("lesion", {"pathway": pathway, "silent": silent, "n": int(idx.size)})
        return idx

    def step(self, **kwargs):
        if self.loop is None:
            raise RuntimeError("Call inhabit(world) before stepping")
        return self.loop.step(**kwargs)

    def run(self, n_steps: int):
        if self.loop is None:
            raise RuntimeError("Call inhabit(world) before running")
        return self.loop.run(n_steps)

    def _life_db(self) -> Path | None:
        if self.save_dir is None:
            return None
        return self.save_dir / "life_history.sqlite"

    def _init_life_history(self) -> None:
        db = self._life_db()
        if db is None:
            return
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, at TEXT, t_ms REAL, kind TEXT, payload TEXT)"
        )
        conn.commit()
        conn.close()

    def log_life(self, kind: str, payload: dict | None = None) -> None:
        db = self._life_db()
        if db is None:
            return
        db.parent.mkdir(parents=True, exist_ok=True)
        if not db.exists():
            self._init_life_history()
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO events(at, t_ms, kind, payload) VALUES (?, ?, ?, ?)",
            (_utc_now(), float(self.net.sim_ms), kind, json.dumps(_json_ready(payload or {}))),
        )
        conn.commit()
        conn.close()

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else self.save_dir
        if path is None:
            raise ValueError("No save path")
        path.mkdir(parents=True, exist_ok=True)
        self.save_dir = path
        (path / "genome_config.json").write_text(
            json.dumps(
                _json_ready(
                    {
                    "individual_id": self.identity.fly_id,
                    "seed": self.identity.seed,
                    "model_version": MODEL_VERSION,
                    "connectome_dataset": self.identity.connectome_dataset,
                    "motor_mode": self.motor_mode.value,
                    "legacy_scaffold": self.legacy_scaffold,
                    "birth_definition": BIRTH_DEFINITION,
                    "consciousness_claimed": False,
                    "neuron_models": self.net.models.snapshot(),
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
                    }
                ),
                indent=2,
            )
            + "\n"
        )
        anatomy = path / "anatomy"
        anatomy.mkdir(exist_ok=True)
        (anatomy / "MaleCNS_v1.json").write_text(
            json.dumps(
                {
                    "dataset_id": self.identity.connectome_dataset,
                    "n": self.connectome.n,
                    "edges": self.connectome.n_edges,
                    "pointer": "data/malecns_v1" if "malecns" in self.identity.connectome_dataset else "organism.toy",
                    "copied": False,
                },
                indent=2,
            )
            + "\n"
        )
        (path / "identity.json").write_text(
            json.dumps(
                _json_ready(
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
                    "motor_map": self.motor_map.notes,
                    "physiology": self.physiology.state.snapshot(),
                    "neuromodulation": self.physiology.neuromodulation.state.snapshot(),
                    "plasticity": self.plasticity.snapshot(),
                    "history": self.history,
                    "provenance_summary": self.provenance.summary(),
                    "legacy_scaffold": self.legacy_scaffold,
                    }
                ),
                indent=2,
            )
            + "\n"
        )
        (path / "metabolic_state.json").write_text(
            json.dumps(_json_ready(self.physiology.state.snapshot()), indent=2) + "\n"
        )
        (path / "neuromodulatory_state.json").write_text(
            json.dumps(_json_ready(self.physiology.neuromodulation.state.snapshot()), indent=2) + "\n"
        )
        (path / "channels.json").write_text(json.dumps(self.channels.to_jsonable()) + "\n")
        self.net.save(path / "neural_state.npz")
        self.net.save(path / "brain.npz")
        np.savez_compressed(
            path / "synaptic_state.npz",
            anatomical=self.net.anatomical,
            functional_gain=self.net.functional_gain,
            plastic_component=self.net.plastic_component,
        )
        memories = path / "memories"
        memories.mkdir(exist_ok=True)
        np.savez_compressed(
            memories / "mushroom_body.npz",
            eligibility=self.learning.eligibility,
            edge_mask=self.learning.edge_mask,
            n_updates=np.int64(self.learning.n_updates),
            last_dopamine=np.float64(self.learning.last_dopamine),
            changed_edges=np.int64(self.learning.changed_edges),
        )
        np.savez_compressed(
            path / "plasticity.npz",
            eligibility=self.learning.eligibility,
            n_updates=np.int64(self.learning.n_updates),
            last_reward=np.float64(self.learning.last_dopamine),
            changed_edges=np.int64(self.learning.changed_edges),
        )
        if self.body is not None:
            (path / "body_state.json").write_text(json.dumps(self.body.snapshot(), indent=2) + "\n")
        self._init_life_history()
        self.log_life("save", {"path": str(path)})
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
            motor_mode=identity.get("motor_mode"),
            legacy_scaffold=bool(identity.get("legacy_scaffold", False)),
            save_dir=path,
        )
        fly.identity = FlyIdentity.from_json(identity)
        fly.history = list(identity.get("history", []))
        brain_path = path / "neural_state.npz"
        if not brain_path.exists():
            brain_path = path / "brain.npz"
        fly.net.load(brain_path)
        syn = path / "synaptic_state.npz"
        if syn.exists():
            syn_data = np.load(syn)
            fly.net.functional_gain = syn_data["functional_gain"]
            fly.net.plastic_component = syn_data["plastic_component"]
            fly.net._rebuild_weights()
        plastic = np.load(path / "plasticity.npz")
        fly.learning.eligibility = plastic["eligibility"]
        fly.learning.n_updates = int(plastic["n_updates"])
        fly.learning.last_dopamine = float(plastic["last_reward"])
        fly.learning.changed_edges = int(plastic["changed_edges"])
        mb = path / "memories" / "mushroom_body.npz"
        if mb.exists():
            mb_data = np.load(mb)
            fly.learning.eligibility = mb_data["eligibility"]
            fly.learning.n_updates = int(mb_data["n_updates"])
            fly.learning.last_dopamine = float(mb_data["last_dopamine"])
            fly.learning.changed_edges = int(mb_data["changed_edges"])
        if identity.get("physiology"):
            fly.physiology.state.load(identity["physiology"])
        if identity.get("neuromodulation"):
            fly.physiology.neuromodulation.state.load(identity["neuromodulation"])
        elif (path / "neuromodulatory_state.json").exists():
            fly.physiology.neuromodulation.state.load(
                json.loads((path / "neuromodulatory_state.json").read_text())
            )
        fly.history.append({"event": "loaded", "from": str(path), "at": _utc_now()})
        fly.log_life("load", {"from": str(path)})
        return fly
