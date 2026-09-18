"""Closed sensorimotor loop.

environment → biological receptors → MaleCNS → identified DNs → body → physics

No task branch. No world.reward. No play_shiritori.
Walking start/stop come from identified DN rates unless LEGACY_SCAFFOLD is on.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from organism.config import MotorMode
from organism.motor_map import command_for_mode
from organism.provenance import BehaviorSource, StepRecord
from organism.sensory import SensoryObservation, SensorySystem


class SensorimotorLoop:
    def __init__(
        self,
        fly: "VirtualFly",  # noqa: F821
        *,
        brain_ticks: int = 10,
        physics_substeps: int = 1,
    ):
        self.fly = fly
        self.brain_ticks = brain_ticks
        self.physics_substeps = physics_substeps
        self.senses = SensorySystem(fly.channels)
        self.bridge = fly.bridge
        self.physiology = fly.physiology
        self._observation: SensoryObservation | None = None
        self._duration_s = 0.0

    def _paused_record(self) -> StepRecord:
        fly = self.fly
        pose = fly.body.pose
        return StepRecord(
            t_ms=fly.net.sim_ms,
            sources=(BehaviorSource.DEVELOPER_OVERRIDE,),
            descending=(0.0, 0.0),
            x_mm=pose.x_mm,
            y_mm=pose.y_mm,
            heading_rad=pose.heading_rad,
            left_eye=0.0,
            right_eye=0.0,
            total_spikes=int(fly.net.total_spikes),
            world=fly.world.name,
            notes="paused",
            developer=True,
            motor_mode=fly.motor_mode.value,
            motor_fidelity_level=fly.motor_fidelity_level,
        )

    def infer_command(
        self,
        observation: SensoryObservation | None = None,
        *,
        apply_senses: bool = True,
        extra_drive: list[tuple] | None = None,
    ) -> StepRecord:
        """Sense + neural dynamics + descending command. Does not step MuJoCo."""
        fly = self.fly
        if fly.body is None or fly.world is None:
            raise RuntimeError("VirtualFly must inhabit a world before stepping")
        if fly.developer.paused:
            return self._paused_record()

        if not apply_senses:
            observation = observation or SensoryObservation()
            fly.net.clear_drive()
        elif observation is None:
            observation = fly.body.sense(fly.world)
        self._observation = observation
        if apply_senses:
            self.senses.apply(fly.net, observation)
        self.physiology.modulate(fly.net, self.bridge)
        external = "NONE"
        if extra_drive:
            for indices, current, source in extra_drive:
                fly.net.add_drive(indices, float(current), source=str(source))
                if str(source).startswith("experiment."):
                    external = str(source)
        counts = fly.net.step(self.brain_ticks)
        duration_s = self.brain_ticks * fly.net.params.dt / 1000.0
        self._duration_s = duration_s
        command = self.bridge.read(
            counts,
            duration_s,
            walking_drive=self.physiology.state.walking_drive,
            walking_bout_s=self.physiology.state.walking_bout_s,
            grooming_drive=self.physiology.state.grooming_drive,
            flight_drive=self.physiology.state.flight_drive,
            graded_output=fly.net.graded_output,
            net=fly.net,
            external_command=external,
            developer_override="NONE",
            privileged_observation="NONE",
            motor_mode=fly.motor_mode.value,
        )
        fly.physiology.state.walking_drive = float(command.walk_trace / 40.0) if command.walk_trace else 0.0
        fly.motor_report = command_for_mode(
            fly.motor_mode,
            left=command.left,
            right=command.right,
            walk_mode=command.mode,
            mn_activity=fly.motor_map.activity(counts, duration_s),
        )
        fly.motor_report["motor_fidelity_level"] = command.motor_fidelity_level

        sources = [BehaviorSource.BIOLOGICAL_CONNECTOME]
        if command.mode in {"walk", "reverse"} and fly.policy.allow_pretrained_low_level_gait:
            if getattr(fly.body, "kind", "") == "neuromechfly":
                sources.append(BehaviorSource.PRETRAINED_LOCOMOTION_CONTROLLER)
        if command.scaffold_used:
            sources.append(BehaviorSource.HAND_IMPLEMENTED_TRANSITION)
        if fly.learning.changed_edges:
            sources.append(BehaviorSource.LEARNED_PLASTICITY)
        pose = fly.body.pose
        return StepRecord(
            t_ms=fly.net.sim_ms,
            sources=tuple(dict.fromkeys(sources)),
            descending=(float(command.left), float(command.right)),
            x_mm=pose.x_mm,
            y_mm=pose.y_mm,
            heading_rad=pose.heading_rad,
            left_eye=observation.left_eye,
            right_eye=observation.right_eye,
            total_spikes=int(counts.sum()),
            world=fly.world.name,
            mode=command.mode,
            walking_drive=fly.physiology.state.walking_drive,
            walk_hz=command.walk_hz,
            walk_trace=command.walk_trace,
            scaffold_used=command.scaffold_used,
            motor_mode=fly.motor_mode.value,
            motor_fidelity_level=command.motor_fidelity_level,
        )

    def step_body(self, n: int | None = None) -> None:
        """Advance MuJoCo with the last descending command."""
        fly = self.fly
        command = self.bridge.last_command
        steps = self.physics_substeps if n is None else n
        fly.body.apply_descending(command)
        for _ in range(max(1, int(steps))):
            fly.body.step_physics()
        fly.world.step()

    def step(self, *, reward: float | None = None, apply_senses: bool = True, extra_drive: list[tuple] | None = None) -> StepRecord:
        fly = self.fly
        if fly.body is None or fly.world is None:
            raise RuntimeError("VirtualFly must inhabit a world before stepping")
        if fly.developer.paused:
            return self._paused_record()

        record = self.infer_command(apply_senses=apply_senses, extra_drive=extra_drive)
        self.step_body()
        observation = self._observation
        duration_s = self._duration_s
        command = self.bridge.last_command
        self.physiology.step(
            duration_s,
            walking=command.mode in {"walk", "reverse"},
            contact=float(np.max(observation.contact)) if observation is not None and observation.contact.size else 0.0,
            odor=0.0 if observation is None else observation.odor,
            vision=(
                0.0
                if observation is None
                else 0.5 * (observation.left_eye + observation.right_eye)
            ),
        )

        fly.learning.observe_activity()
        if reward:
            fly.learning.apply_reward(reward)
        elif fly.learning.dan.size:
            dan_hz = float(fly.net.counts[fly.learning.dan].sum()) / max(duration_s, 1e-6)
            if dan_hz > 0.5:
                fly.learning.apply_dopamine(fly.physiology.neuromodulation.state.dopamine)

        pose = fly.body.pose
        record = replace(record, x_mm=pose.x_mm, y_mm=pose.y_mm, heading_rad=pose.heading_rad)
        fly.provenance.append(record)
        return record

    def run(self, n_steps: int) -> list[StepRecord]:
        return [self.step() for _ in range(n_steps)]
