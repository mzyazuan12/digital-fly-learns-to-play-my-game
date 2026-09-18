"""Closed sensorimotor loop.

environment → senses → MaleCNS → identified DNs → body controller → physics

No task branch. No world.reward. No play_shiritori.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

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
        )

    def infer_command(self, observation: SensoryObservation | None = None) -> StepRecord:
        """Sense + LIF + descending command. Does not step MuJoCo."""
        fly = self.fly
        if fly.body is None or fly.world is None:
            raise RuntimeError("VirtualFly must inhabit a world before stepping")
        if fly.developer.paused:
            return self._paused_record()

        if observation is None:
            observation = fly.body.sense(fly.world)
        self._observation = observation
        self.senses.apply(fly.net, observation)
        self.physiology.modulate(fly.net, self.bridge)
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
        )

        sources = [BehaviorSource.BIOLOGICAL_CONNECTOME]
        if command.mode in {"walk", "reverse", "groom", "fly"}:
            if getattr(fly.body, "kind", "") == "neuromechfly":
                sources.append(BehaviorSource.PRETRAINED_LOCOMOTION_CONTROLLER)
            sources.append(BehaviorSource.HAND_IMPLEMENTED_TRANSITION)
        else:
            sources.append(BehaviorSource.HAND_IMPLEMENTED_TRANSITION)
        if fly.plasticity.changed_edges:
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

    def step(self, *, reward: float | None = None) -> StepRecord:
        fly = self.fly
        if fly.body is None or fly.world is None:
            raise RuntimeError("VirtualFly must inhabit a world before stepping")
        if fly.developer.paused:
            return self._paused_record()

        record = self.infer_command()
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

        fly.plasticity.observe_activity()
        if reward:
            fly.plasticity.apply_reward(reward)

        pose = fly.body.pose
        record = replace(record, x_mm=pose.x_mm, y_mm=pose.y_mm, heading_rad=pose.heading_rad)
        fly.provenance.append(record)
        return record

    def run(self, n_steps: int) -> list[StepRecord]:
        return [self.step() for _ in range(n_steps)]
