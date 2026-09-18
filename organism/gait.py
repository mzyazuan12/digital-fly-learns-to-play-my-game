"""Brain-disconnected NeuroMechFly walking.

MaleCNS is not used here. The official FlyGym 2.1 NeuroMechFly (micro-CT,
orange chitin, compound eyes) owns the articulated MuJoCo body. FlyGym's
HybridTurningController is an engineered VNC/muscle surrogate: preprogrammed
steps plus a CPG, not recovered motor circuitry.

Walking is produced only by leg actuators and contact physics. This module
never writes the free-root pose as a substitute for gait.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# Engineered descending drives for the HybridTurningController.
# left/right amplitudes modulate the two sides of the CPG; they are not
# neuron indices. Higher amplitude on the right legs typically turns left.
GAIT_COMMANDS: dict[str, dict[str, float | str]] = {
    "stand": {"left": 0.0, "right": 0.0, "mode": "rest"},
    "stop": {"left": 0.0, "right": 0.0, "mode": "rest"},
    "walk": {"left": 1.0, "right": 1.0, "mode": "walk"},
    "walk_slow": {"left": 0.4, "right": 0.4, "mode": "walk"},
    "walk_fast": {"left": 1.2, "right": 1.2, "mode": "walk"},
    "turn_left": {"left": 0.2, "right": 1.0, "mode": "walk"},
    "turn_right": {"left": 1.0, "right": 0.2, "mode": "walk"},
    "groom": {"left": 0.0, "right": 0.0, "mode": "groom"},
    "fly": {"left": 0.0, "right": 0.0, "mode": "fly"},
}

BATTERY = (
    ("stand", 0.25),
    ("walk", 0.55),
    ("walk_slow", 0.35),
    ("walk_fast", 0.35),
    ("turn_left", 0.40),
    ("turn_right", 0.40),
    ("stop", 0.25),
)

# MuJoCo stays at 0.1 ms. The CPG/contact readout does not need 10 kHz.
# 4 ms control still samples a 10–20 Hz tripod many times per step, and
# walking speed stays ~11 mm/s (adult Drosophila 10–25 mm/s).
CONTROL_DECIMATION = 40


def dof_name(dof: object) -> str:
    name = getattr(dof, "name", None)
    if isinstance(name, str) and "-" in name:
        return name
    parent = dof.parent.name if hasattr(dof.parent, "name") else str(dof.parent)
    child = dof.child.name if hasattr(dof.child, "name") else str(dof.child)
    axis = dof.axis.value if hasattr(dof.axis, "value") else str(dof.axis)
    return f"{parent}-{child}-{axis}"


def remap_angles(angles: np.ndarray, src_order: list, dst_order: list) -> np.ndarray:
    by_name = {dof_name(dof): float(angles[i]) for i, dof in enumerate(src_order)}
    missing = [dof_name(dof) for dof in dst_order if dof_name(dof) not in by_name]
    if missing:
        raise KeyError(f"No controller angle for DoFs: {missing[:8]}")
    return np.array([by_name[dof_name(dof)] for dof in dst_order], dtype=float)


def _heading_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _segment_name(seg: object) -> str:
    return seg.name if hasattr(seg, "name") else str(seg)


@dataclass
class Proprioception:
    """Physical feedback for a nervous-system interface.

    World x/y are intentionally absent. Pose for logging lives on the body,
    not in this packet.
    """

    joint_names: list[str]
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    contact_found: np.ndarray
    contact_forces: np.ndarray
    actuator_forces: np.ndarray
    thorax_quat_wxyz: np.ndarray
    angular_velocity: np.ndarray
    tarsus5_z_mm: np.ndarray


@dataclass
class BodyTransform:
    segment: str
    mujoco_name: str
    pos_mm: np.ndarray
    quat_wxyz: np.ndarray


@dataclass
class GaitSnapshot:
    t_s: float
    command: str
    left: float
    right: float
    mode: str
    thorax_xyz_mm: np.ndarray
    heading_rad: float
    joint_angles: np.ndarray
    joint_names: list[str]
    contact_found: np.ndarray
    contact_force_mag: np.ndarray
    root_written: bool


class RootMotionForbidden(RuntimeError):
    """Raised when code tries to walk by writing the free-root pose."""


class ArticulatedFly:
    """Untethered NeuroMechFly in MuJoCo. Brain disconnected.

    The thorax translates because stance-leg forces act on the body.
    """

    kind = "neuromechfly"
    controller_kind = "flygym_hybrid_turning_controller"
    controller_role = "engineered_vnc_muscle_surrogate"

    def __init__(
        self,
        *,
        name: str = "nmf",
        spawn_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.6),
        spawn_yaw_rad: float = 0.0,
        render: bool = False,
        camera_res: tuple[int, int] = (1080, 1920),
        buffer_frames: bool = True,
        ground_half_mm: float = 6000.0,
    ):
        from flygym import Simulation
        from flygym.anatomy import ContactBodiesPreset
        from flygym.compose import ActuatorType, FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import (
            HybridTurningController,
            LocomotionAction,
            PreprogrammedSteps,
            apply_locomotion_action,
            get_default_locomotion_dof_order,
        )
        from flygym_demo.complex_terrain.common import make_locomotion_fly

        self.name = name
        self._root_writes = 0
        self._allow_root_write = True
        self.last_command_name = "stand"
        self.last_left = 0.0
        self.last_right = 0.0
        self.last_mode = "rest"
        self.last_groom_hz = 0.0
        self.last_flight_hz = 0.0
        self._physics_steps = 0

        fly = make_locomotion_fly(name=name, colorize=True)
        # LEGS_ONLY leaves c_head unwelded. fusestatic would swallow it into
        # the thorax; xpos[-1] then poses the head mesh on a hind tarsus.
        fly.mjcf_root.compiler.fusestatic = False
        fly.add_vision()
        # FlyGym 2.1 default follow-cam: thorax frame, above and to the
        # right, looking down at the fly. Do not put this at thorax height;
        # that is a worm's-eye belly shot.
        fly.add_tracking_camera(
            name="gait_cam",
            mode="track",
            pos_offset=(-0.5, -7.5, 5.0),
            rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
            fovy=30.0,
        )

        world = FlatGroundWorld(half_size=float(ground_half_mm))
        grid = world.mjcf_root.material("grid")
        if grid is not None:
            grid.reflectance = 0.0
        yaw = spawn_yaw_rad
        world.add_fly(
            fly,
            list(spawn_xyz_mm),
            Rotation3D("quat", [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=True,
        )
        # add_fly reapplies mujoco_globals.yaml (fusestatic: true). Keep c_head.
        world.mjcf_root.compiler.fusestatic = False
        world.mjcf_root.worldbody.add_camera(
            name="gait_world",
            pos=(-2.0, -14.0, 8.0),
            xyaxes=(1.0, 0.0, 0.0, 0.0, 0.6, 0.8),
            fovy=32.0,
        )

        sim = Simulation(world)
        fly_dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        ctrl_dofs = get_default_locomotion_dof_order()
        steps = PreprogrammedSteps()
        ctrl_dt = float(sim.timestep) * CONTROL_DECIMATION
        controller = HybridTurningController(
            timestep=ctrl_dt,
            preprogrammed_steps=steps,
            output_dof_order=ctrl_dofs,
        )
        controller.retraction_persistence_steps = max(
            1, int(round(20 * float(sim.timestep) / ctrl_dt))
        )
        sim.reset()
        controller.reset(seed=0)
        stand = LocomotionAction(
            joint_angles=remap_angles(
                steps.default_pose_by_dof_order(ctrl_dofs), ctrl_dofs, fly_dofs
            ),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(sim, fly.name, stand)
        sim.warmup(0.08)

        self.fly = fly
        self.sim = sim
        self.controller = controller
        self.world = world
        self._fly_dofs = fly_dofs
        self._ctrl_dofs = ctrl_dofs
        self._stand_action = stand
        self._apply = apply_locomotion_action
        self._LocomotionAction = LocomotionAction
        self._ActuatorType = ActuatorType
        self._control_decimation = CONTROL_DECIMATION
        self._steps_since_ctrl = CONTROL_DECIMATION
        self._last_mapped = stand
        self._HybridObs = None
        from flygym_demo.complex_terrain import HybridControllerObservation

        self._HybridObs = HybridControllerObservation

        self.joint_names = [dof_name(dof) for dof in fly.get_jointdofs_order()]
        self.actuated_names = [dof_name(dof) for dof in fly_dofs]
        self.body_names = [_segment_name(seg) for seg in fly.get_bodysegs_order()]
        self.leg_names = list(fly.get_legs_order())
        thorax = next(i for i, n in enumerate(self.body_names) if n.endswith("c_thorax") or n == "c_thorax")
        self._thorax_idx = thorax
        self._tarsus5_idx = [
            i for i, n in enumerate(self.body_names) if n.endswith("_tarsus5")
        ]
        self._internal_body_ids = np.asarray(
            sim._internal_bodyids_by_fly[fly.name], dtype=np.int32
        )
        self.mujoco_body_names = [
            _mj_name(sim.mj_model, "BODY", int(i)) for i in self._internal_body_ids
        ]
        if any(int(i) < 0 for i in self._internal_body_ids):
            missing = [
                self.body_names[i]
                for i, bid in enumerate(self._internal_body_ids)
                if int(bid) < 0
            ]
            raise RuntimeError(f"NeuroMechFly body ids missing after compile: {missing}")
        self.camera_names = [
            _mj_name(sim.mj_model, "CAMERA", i) for i in range(sim.mj_model.ncam)
        ]
        self.render_enabled = False
        self._jpeg: bytes | None = None
        self._last_rgb = None
        if render:
            self.enable_renderer(camera_res=camera_res, buffer_frames=buffer_frames)

        self._allow_root_write = False
        self._spawn_qpos = np.array(sim.mj_data.qpos[:7], dtype=np.float64)
        self._vision_ok = True

    def enable_renderer(
        self,
        *,
        camera_res: tuple[int, int] = (1080, 1920),
        playback_speed: float = 0.2,
        output_fps: int = 25,
        buffer_frames: bool = True,
    ) -> str:
        cam = self._pick_camera()
        self.sim.set_renderer(
            cam,
            camera_res=camera_res,
            playback_speed=playback_speed,
            output_fps=output_fps,
            buffer_frames=buffer_frames,
        )
        self.render_enabled = True
        self._last_rgb = None
        return cam

    def _pick_camera(self) -> str:
        # Tracking camera keeps all six legs in frame. Thorax translation is
        # the checkerboard sliding under the feet, plus the logged xy path.
        for name in self.camera_names:
            if name and "gait_cam" in name:
                return name
        for name in self.camera_names:
            if name and "gait_world" in name:
                return name
        if not self.camera_names:
            raise RuntimeError("NeuroMechFly compiled without cameras")
        return self.camera_names[0]

    def set_command(self, command: str | tuple[float, float] | np.ndarray) -> None:
        if isinstance(command, str):
            if command not in GAIT_COMMANDS:
                raise KeyError(f"Unknown gait command '{command}'")
            spec = GAIT_COMMANDS[command]
            self.last_command_name = command
            self.last_left = float(spec["left"])
            self.last_right = float(spec["right"])
            self.last_mode = str(spec["mode"])
            if self.last_mode == "rest":
                self.controller.reset(seed=0)
            return
        vec = np.asarray(command, dtype=np.float64).reshape(2)
        self.last_command_name = "custom"
        self.last_left = float(vec[0])
        self.last_right = float(vec[1])
        self.last_mode = "walk" if float(np.mean(np.abs(vec))) > 0.05 else "rest"
        if self.last_mode == "rest":
            self.controller.reset(seed=0)

    def step(self, command: str | tuple[float, float] | np.ndarray | None = None) -> GaitSnapshot:
        if command is not None:
            self.set_command(command)
        if self.last_mode == "groom":
            self._apply(self.sim, self.name, self._groom_action())
            self._steps_since_ctrl = self._control_decimation
        elif self.last_mode == "fly":
            self._apply(self.sim, self.name, self._flight_action())
            self._steps_since_ctrl = self._control_decimation
        elif self.last_mode == "rest":
            self._apply(self.sim, self.name, self._stand_action)
            self._steps_since_ctrl = self._control_decimation
        else:
            self._steps_since_ctrl += 1
            if self._steps_since_ctrl >= self._control_decimation:
                self._steps_since_ctrl = 0
                obs = self._HybridObs.from_sim(self.sim, self.name)
                action = self.controller.step(
                    np.array([self.last_left, self.last_right], dtype=float), obs
                )
                self._last_mapped = self._LocomotionAction(
                    joint_angles=remap_angles(
                        action.joint_angles, self._ctrl_dofs, self._fly_dofs
                    ),
                    adhesion_onoff=action.adhesion_onoff,
                )
                self._apply(self.sim, self.name, self._last_mapped)
        self.sim.step()
        self._physics_steps += 1
        if self.render_enabled:
            self.sim.render_as_needed()
        return self.snapshot()

    def render_frame(self) -> np.ndarray:
        renderer = self.sim.renderer
        if renderer is None:
            self.enable_renderer(buffer_frames=False)
            renderer = self.sim.renderer
        if renderer is None:
            raise RuntimeError("FlyGym renderer did not attach")
        cam_id = next(iter(renderer._cameras_names2id.values()))
        renderer.mj_renderer.update_scene(
            self.sim.mj_data, cam_id, renderer.scene_option
        )
        rgb = np.asarray(renderer.mj_renderer.render())
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"MuJoCo render returned unexpected shape {rgb.shape}")
        self._last_rgb = rgb
        return rgb

    def save_png(self, path: Path) -> Path:
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.asarray(self.render_frame(), dtype=np.uint8)
        Image.fromarray(rgb[..., :3]).save(path)
        return path

    def _groom_action(self):
        """Front-leg antenna-groom CPG. Engineered VNC surrogate, not recovered circuitry.

        Frequency and amplitude follow descending groom rate, not a named clip.
        """
        angles = np.array(self._stand_action.joint_angles, dtype=float, copy=True)
        t = self._physics_steps * float(self.sim.timestep)
        hz = float(np.clip(2.2 + 0.28 * self.last_groom_hz, 1.6, 6.0))
        amp = float(np.clip(0.28 + 0.06 * self.last_groom_hz, 0.22, 0.72))
        phase = 2.0 * np.pi * hz * t
        touched = 0
        for i, name in enumerate(self.actuated_names):
            low = name.lower()
            if "lf_coxa" in low or "lf_femur" in low or "lf_trochanter" in low:
                angles[i] += amp * np.sin(phase)
                touched += 1
            elif "rf_coxa" in low or "rf_femur" in low or "rf_trochanter" in low:
                angles[i] += amp * np.sin(phase + 1.1)
                touched += 1
        if touched == 0:
            n = min(6, len(angles))
            for i in range(n):
                angles[i] += 0.7 * amp * np.sin(phase + 0.4 * i)
        return self._LocomotionAction(
            joint_angles=angles,
            adhesion_onoff=np.array([False, True, True, False, True, True], dtype=bool),
        )

    def _flight_action(self):
        """Wingbeat overlay on a standing pose. No free-root takeoff.

        Beat rate follows descending flight rate. Legs stay on the HybridTurning
        rest pose; the living-room renderer also oscillates the wing meshes.
        """
        angles = np.array(self._stand_action.joint_angles, dtype=float, copy=True)
        t = self._physics_steps * float(self.sim.timestep)
        hz = float(np.clip(12.0 + 0.8 * self.last_flight_hz, 8.0, 28.0))
        amp = float(np.clip(0.35 + 0.04 * self.last_flight_hz, 0.28, 0.85))
        phase = 2.0 * np.pi * hz * t
        for i, name in enumerate(self.actuated_names):
            low = name.lower()
            if "wing" in low:
                angles[i] += amp * np.sin(phase if "l_wing" in low or "-l_" in low else phase + np.pi)
        return self._LocomotionAction(
            joint_angles=angles,
            adhesion_onoff=np.ones(6, dtype=bool),
        )

    def teleport(self, *_args: Any, **_kwargs: Any) -> None:
        raise RootMotionForbidden(
            "Biological/embodied mode forbids writing the fly root pose. "
            "Walking must come from leg actuation and contact."
        )

    def proprioception(self) -> Proprioception:
        sim = self.sim
        angles = np.asarray(sim.get_joint_angles(self.name), dtype=np.float64)
        vels = np.asarray(sim.get_joint_velocities(self.name), dtype=np.float64)
        found, forces, *_ = sim.get_ground_contact_info(self.name)
        act = np.asarray(
            sim.get_actuator_forces(self.name, self._ActuatorType.POSITION),
            dtype=np.float64,
        )
        pos = np.asarray(sim.get_body_positions(self.name), dtype=np.float64)
        quat = np.asarray(sim.get_body_rotations(self.name), dtype=np.float64)
        thorax_id = int(self._internal_body_ids[self._thorax_idx])
        ang = np.asarray(sim.mj_data.cvel[thorax_id, :3], dtype=np.float64)
        tarsus_z = pos[self._tarsus5_idx, 2] if self._tarsus5_idx else np.zeros(6)
        return Proprioception(
            joint_names=list(self.joint_names),
            joint_angles=angles,
            joint_velocities=vels,
            contact_found=np.asarray(found, dtype=np.float64),
            contact_forces=np.asarray(forces, dtype=np.float64),
            actuator_forces=act,
            thorax_quat_wxyz=quat[self._thorax_idx],
            angular_velocity=ang,
            tarsus5_z_mm=np.asarray(tarsus_z, dtype=np.float64),
        )

    def body_transforms(self) -> list[BodyTransform]:
        data = self.sim.mj_data
        thorax_id = int(self._internal_body_ids[self._thorax_idx])
        out = []
        for i, segment in enumerate(self.body_names):
            bid = int(self._internal_body_ids[i])
            if bid < 0:
                bid = thorax_id
            out.append(
                BodyTransform(
                    segment=segment,
                    mujoco_name=self.mujoco_body_names[i],
                    pos_mm=np.asarray(data.xpos[bid], dtype=np.float64).copy(),
                    quat_wxyz=np.asarray(data.xquat[bid], dtype=np.float64).copy(),
                )
            )
        return out

    def geom_transforms(self) -> list[BodyTransform]:
        """World pose of each visual mesh geom.

        Static segments (eyes, wings) can be fusestatic-welded into a parent
        body. Body xpos then aliases the parent, so both eyes sit on one point
        and the GLB mesh is applied at the wrong origin. geom_xpos/geom_xmat
        keep the authored offset. Rendering must follow geoms, not bodies.
        """
        import mujoco as mj

        model = self.sim.mj_model
        data = self.sim.mj_data
        known = set(self.body_names)
        mesh_type = int(mj.mjtGeom.mjGEOM_MESH)
        out: list[BodyTransform] = []
        for i in range(int(model.ngeom)):
            if int(model.geom_type[i]) != mesh_type:
                continue
            raw = _mj_name(model, "GEOM", i)
            name = raw.rsplit("/", 1)[-1]
            if name not in known:
                continue
            quat = np.empty(4, dtype=np.float64)
            mat = np.ascontiguousarray(data.geom_xmat[i], dtype=np.float64)
            mj.mju_mat2Quat(quat, mat)
            out.append(
                BodyTransform(
                    segment=name,
                    mujoco_name=raw,
                    pos_mm=np.asarray(data.geom_xpos[i], dtype=np.float64).copy(),
                    quat_wxyz=quat,
                )
            )
        return out or self.body_transforms()

    def render_map(self) -> dict[str, Any]:
        """MuJoCo geom name → render node. Meshes follow geom poses."""
        return {
            "kind": self.kind,
            "controller": self.controller_kind,
            "controller_role": self.controller_role,
            "units": "mm",
            "quat_order": "wxyz",
            "bodies": [
                {
                    "segment": t.segment,
                    "mujoco_body": t.mujoco_name,
                    "render_node": t.segment,
                    "pos_mm": t.pos_mm.tolist(),
                    "quat_wxyz": t.quat_wxyz.tolist(),
                }
                for t in self.geom_transforms()
            ],
            "joints": self.joint_names,
            "actuators": self.actuated_names,
            "cameras": self.camera_names,
            "nbody": int(self.sim.mj_model.nbody),
            "njnt": int(self.sim.mj_model.njnt),
            "nu": int(self.sim.mj_model.nu),
            "ngeom": int(self.sim.mj_model.ngeom),
        }

    def thorax_xyz(self) -> np.ndarray:
        return np.asarray(self.sim.get_body_positions(self.name)[self._thorax_idx], dtype=np.float64)

    def heading_rad(self) -> float:
        quat = np.asarray(self.sim.get_body_rotations(self.name)[self._thorax_idx], dtype=np.float64)
        return _heading_from_quat(quat)

    def snapshot(self) -> GaitSnapshot:
        prop = self.proprioception()
        force_mag = np.linalg.norm(prop.contact_forces.reshape(6, -1), axis=1)
        return GaitSnapshot(
            t_s=self._physics_steps * float(self.sim.timestep),
            command=self.last_command_name,
            left=self.last_left,
            right=self.last_right,
            mode=self.last_mode,
            thorax_xyz_mm=self.thorax_xyz(),
            heading_rad=self.heading_rad(),
            joint_angles=prop.joint_angles.copy(),
            joint_names=list(self.joint_names),
            contact_found=prop.contact_found.copy(),
            contact_force_mag=force_mag,
            root_written=self._root_writes > 0,
        )

    def latest_frame(self) -> np.ndarray | None:
        if getattr(self, "_last_rgb", None) is not None:
            return self._last_rgb
        renderer = self.sim.renderer
        if renderer is None or not renderer.frames:
            return None
        cam = next(iter(renderer.frames))
        frames = renderer.frames[cam]
        if not frames:
            return None
        return frames[-1]

    def save_video(self, path: Path) -> Path:
        if self.sim.renderer is None:
            raise RuntimeError("Renderer was not enabled")
        path = Path(path)
        self.sim.renderer.save_video(path)
        return path

    def inspect_model(self) -> dict[str, Any]:
        model = self.sim.mj_model
        hinge = int(sum(1 for i in range(model.njnt) if int(model.jnt_type[i]) == 3))
        free = int(sum(1 for i in range(model.njnt) if int(model.jnt_type[i]) == 0))
        return {
            "kind": self.kind,
            "name": self.name,
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "hinge_joints": hinge,
            "free_joints": free,
            "nu": int(model.nu),
            "ngeom": int(model.ngeom),
            "timestep_s": float(self.sim.timestep),
            "actuated_dofs": len(self.actuated_names),
            "joint_dofs": len(self.joint_names),
            "bodies": self.body_names,
            "mujoco_bodies": self.mujoco_body_names,
            "cameras": self.camera_names,
            "controller": self.controller_kind,
            "controller_role": self.controller_role,
            "brain": None,
            "root_motion": False,
        }


def _mj_name(model, kind: str, index: int) -> str:
    import mujoco as mj

    if int(index) < 0:
        return ""
    obj = getattr(mj.mjtObj, f"mjOBJ_{kind}")
    name = mj.mj_id2name(model, obj, int(index))
    return name or f"{kind.lower()}_{index}"


def _leg_joint_mask(names: list[str]) -> dict[str, np.ndarray]:
    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    out = {}
    for leg in legs:
        out[leg] = np.array(
            [n.startswith(f"{leg}_") or f"-{leg}_" in n or n.startswith(f"c_thorax-{leg}_") for n in names],
            dtype=bool,
        )
        if not out[leg].any():
            out[leg] = np.array([f"{leg}_" in n for n in names], dtype=bool)
    return out


def _summarize_phase(command: str, angles: np.ndarray, names: list[str], xyz: np.ndarray, headings: np.ndarray, contacts: np.ndarray) -> dict[str, Any]:
    std = np.std(angles, axis=0)
    masks = _leg_joint_mask(names)
    per_leg = {}
    for leg, mask in masks.items():
        per_leg[leg] = float(np.max(std[mask])) if mask.any() else 0.0
    start = xyz[0]
    end = xyz[-1]
    disp = float(np.linalg.norm(end[:2] - start[:2]))
    heading_delta = float(headings[-1] - headings[0])
    heading_delta = float((heading_delta + np.pi) % (2 * np.pi) - np.pi)
    return {
        "command": command,
        "steps": int(angles.shape[0]),
        "joint_std_max": float(np.max(std)) if std.size else 0.0,
        "joint_std_mean": float(np.mean(std)) if std.size else 0.0,
        "leg_joint_std_max": per_leg,
        "legs_articulating": int(sum(v > 0.02 for v in per_leg.values())),
        "thorax_xy_disp_mm": disp,
        "thorax_z_mean_mm": float(np.mean(xyz[:, 2])),
        "thorax_z_std_mm": float(np.std(xyz[:, 2])),
        "heading_delta_rad": heading_delta,
        "contact_frac": float(np.mean(contacts > 0.5)),
        "start_xy_mm": start[:2].tolist(),
        "end_xy_mm": end[:2].tolist(),
    }


def evaluate_battery(report: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    phases = {p["command"]: p for p in report["phases"]}
    walk = phases.get("walk")
    stand = phases.get("stand")
    if walk is None:
        return False, ["missing walk phase"]
    if walk["legs_articulating"] < 6:
        errors.append(
            f"walk must articulate all 6 legs; got {walk['leg_joint_std_max']}"
        )
    if walk["thorax_xy_disp_mm"] < 1.0:
        errors.append(
            f"walk thorax xy displacement {walk['thorax_xy_disp_mm']:.3f} mm is too small"
        )
    if walk["contact_frac"] < 0.15:
        errors.append(f"walk contact fraction {walk['contact_frac']:.3f} is too low")
    if walk["thorax_z_mean_mm"] < 0.15 or walk["thorax_z_mean_mm"] > 4.0:
        errors.append(f"walk thorax z {walk['thorax_z_mean_mm']:.3f} mm is not on the floor")
    if stand and stand["thorax_xy_disp_mm"] > max(1.2, 0.4 * walk["thorax_xy_disp_mm"]):
        errors.append(
            f"stand slid {stand['thorax_xy_disp_mm']:.3f} mm; expected planted stance"
        )
    left = phases.get("turn_left")
    right = phases.get("turn_right")
    if left and abs(left["heading_delta_rad"]) < 0.08:
        errors.append(f"turn_left heading change {left['heading_delta_rad']:.3f} rad is too small")
    if right and abs(right["heading_delta_rad"]) < 0.08:
        errors.append(f"turn_right heading change {right['heading_delta_rad']:.3f} rad is too small")
    if report.get("root_written"):
        errors.append("free-root pose was written during the trial")
    if report.get("brain") is not None:
        errors.append("MaleCNS was connected; gait test must disconnect it")
    return not errors, errors


def run_battery(
    *,
    video: Path | None = None,
    report_path: Path | None = None,
    durations: tuple[tuple[str, float], ...] = BATTERY,
    sample_every: int = 10,
) -> dict[str, Any]:
    fly = ArticulatedFly(render=video is not None)
    model = fly.inspect_model()
    traces: dict[str, dict[str, list]] = {}
    stills: dict[str, str] = {}
    for command, duration_s in durations:
        n_steps = max(1, int(round(duration_s / float(fly.sim.timestep))))
        angles = []
        xyz = []
        headings = []
        contacts = []
        for i in range(n_steps):
            snap = fly.step(command)
            if i % sample_every == 0:
                angles.append(snap.joint_angles)
                xyz.append(snap.thorax_xyz_mm)
                headings.append(snap.heading_rad)
                contacts.append(snap.contact_found)
        traces[command] = {
            "angles": np.stack(angles),
            "xyz": np.stack(xyz),
            "heading": np.asarray(headings),
            "contact": np.stack(contacts),
        }
        if fly.render_enabled and command in {"stand", "walk", "walk_fast", "turn_left"}:
            stills[command] = str(fly.save_png(OUTPUTS / f"gait_{command}.png"))

    phases = [
        _summarize_phase(
            command,
            traces[command]["angles"],
            fly.joint_names,
            traces[command]["xyz"],
            traces[command]["heading"],
            traces[command]["contact"],
        )
        for command, _ in durations
    ]
    report = {
        "ok": False,
        "brain": None,
        "body": model,
        "controller": {
            "name": fly.controller_kind,
            "role": fly.controller_role,
            "note": "FlyGym HybridTurningController + PreprogrammedSteps. Not VNC motor neurons.",
        },
        "root_written": fly._root_writes > 0,
        "phases": phases,
        "joint_names": fly.joint_names,
        "render_map": fly.render_map() if video is None else {"cameras": fly.camera_names},
    }
    ok, errors = evaluate_battery(report) if any(p["command"] == "walk" for p in phases) else (True, [])
    report["ok"] = ok
    report["errors"] = errors
    if stills:
        report["stills"] = stills
    if video is not None:
        report["video"] = str(fly.save_video(video))
    path = report_path or (OUTPUTS / "gait_battery.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = json.loads(json.dumps(report, default=_json_default))
    path.write_text(json.dumps(serial, indent=2) + "\n")
    report["report_path"] = str(path)
    _write_joint_plot(fly.joint_names, traces, path.with_suffix(".joints.npz"))
    return report


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def _write_joint_plot(names: list[str], traces: dict, npz_path: Path) -> None:
    payload = {"names": np.array(names)}
    for command, data in traces.items():
        payload[f"{command}_angles"] = data["angles"]
        payload[f"{command}_xyz"] = data["xyz"]
        payload[f"{command}_heading"] = data["heading"]
        payload[f"{command}_contact"] = data["contact"]
    np.savez_compressed(npz_path, **payload)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    walk = traces.get("walk")
    if walk is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    t = np.arange(walk["angles"].shape[0])
    for i, name in enumerate(names):
        if "coxa" in name or "trochanter" in name or name.endswith("tibia-pitch"):
            axes[0].plot(t, walk["angles"][:, i], lw=0.8, label=name)
    axes[0].set_ylabel("joint angle (rad)")
    axes[0].set_title("Walk: actuated leg joints (must change)")
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    xyz = walk["xyz"]
    axes[1].plot(xyz[:, 0], xyz[:, 1], color="black")
    axes[1].scatter(xyz[0, 0], xyz[0, 1], color="green", label="start")
    axes[1].scatter(xyz[-1, 0], xyz[-1, 1], color="red", label="end")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("thorax x (mm)")
    axes[1].set_ylabel("thorax y (mm)")
    axes[1].set_title("Walk: thorax path from physics (not root motion)")
    axes[1].legend()
    axes[2].plot(t, walk["contact"], lw=0.9)
    axes[2].set_ylabel("foot contact")
    axes[2].set_xlabel("samples")
    axes[2].set_title("Walk: ground contact per leg")
    fig.tight_layout()
    fig.savefig(npz_path.with_suffix(".png"), dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Brain-disconnected NeuroMechFly walking test. MaleCNS is not loaded."
    )
    parser.add_argument("--video", type=Path, default=None, help="Write a MuJoCo MP4")
    parser.add_argument("--report", type=Path, default=OUTPUTS / "gait_battery.json")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer after the battery")
    parser.add_argument("--seconds", type=float, default=None, help="Override every phase duration")
    parser.add_argument("--command", choices=sorted(GAIT_COMMANDS), default=None, help="Run one command instead of the battery")
    args = parser.parse_args(argv)
    print("Building FlyGym NeuroMechFly (MaleCNS disconnected)…")
    durations = BATTERY
    if args.command:
        durations = ((args.command, args.seconds or 0.6),)
    elif args.seconds is not None:
        durations = tuple((name, args.seconds) for name, _ in BATTERY)
    report = run_battery(video=args.video, report_path=args.report, durations=durations)
    print(json.dumps({k: report[k] for k in ("ok", "errors", "controller", "root_written")}, indent=2))
    for phase in report["phases"]:
        print(
            f"  {phase['command']:11} legs={phase['legs_articulating']}/6  "
            f"xy={phase['thorax_xy_disp_mm']:.2f} mm  "
            f"z={phase['thorax_z_mean_mm']:.2f} mm  "
            f"Δheading={phase['heading_delta_rad']:+.3f} rad  "
            f"contact={phase['contact_frac']:.2f}  "
            f"joint_std={phase['joint_std_max']:.3f}"
        )
    print(f"report: {report['report_path']}")
    if args.video:
        print(f"video:  {report.get('video')}")
    if not report["ok"]:
        print("MILESTONE FAILED: articulated physics walking is not verified.")
        return 1
    print("MILESTONE PASSED: six-leg articulated gait moved the untethered NeuroMechFly.")
    if args.viewer:
        from flygym.rendering import launch_interactive_viewer

        fly = ArticulatedFly()
        fly.set_command("walk")
        for _ in range(int(0.3 / fly.sim.timestep)):
            fly.step()
        launch_interactive_viewer(fly.sim.mj_model, fly.sim.mj_data, init_keyframe=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
