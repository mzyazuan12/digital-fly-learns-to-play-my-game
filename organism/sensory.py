"""World → currents. The brain never receives privileged task fields.

No `target_position`, `keyboard_key`, or `correct_answer` is written into
the network. Intensities come from simulated receptors on the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flybrain.network import LIFNetwork
from organism.channels import SensorimotorChannels


@dataclass
class SensoryObservation:
    """Receptor-level snapshot. Pose is for the body/developer, not a BCI label."""

    left_eye: float = 0.0
    right_eye: float = 0.0
    ommatidia: np.ndarray | None = None
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    contact: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    odor: float = 0.0
    taste: float = 0.0
    angular_velocity: float = 0.0
    heading_rad: float = 0.0


@dataclass(frozen=True)
class SensoryGains:
    """Hand-set current scales. Not photoreceptor biophysics."""

    vision: float = 28.0
    proprioception: float = 6.0
    contact: float = 10.0
    antenna: float = 8.0
    olfaction: float = 8.0
    gustation: float = 8.0


class SensorySystem:
    def __init__(self, channels: SensorimotorChannels, gains: SensoryGains | None = None):
        self.channels = channels
        self.gains = gains or SensoryGains()

    def apply(self, net: LIFNetwork, observation: SensoryObservation) -> None:
        net.clear_drive()
        g = self.gains
        self._inject(net, self.channels.left_eye, g.vision * _clip01(observation.left_eye))
        self._inject(net, self.channels.right_eye, g.vision * _clip01(observation.right_eye))
        if observation.ommatidia is not None:
            self._inject_ommatidia(net, observation.ommatidia)
        self._inject_vector(net, self.channels.proprioception, observation.joint_angles, g.proprioception)
        self._inject_vector(net, self.channels.contact, observation.contact, g.contact)
        self._inject(net, self.channels.antenna_left, g.antenna * _clip01(observation.antenna_left))
        self._inject(net, self.channels.antenna_right, g.antenna * _clip01(observation.antenna_right))
        self._inject(net, self.channels.olfaction, g.olfaction * _clip01(observation.odor))
        self._inject(net, self.channels.gustation, g.gustation * _clip01(observation.taste))

    def _inject_ommatidia(self, net: LIFNetwork, ommatidia: np.ndarray) -> None:
        """Map (2, n, c) ommatidia onto left/right visual cells if both exist."""
        if ommatidia.ndim < 2 or ommatidia.shape[0] < 2:
            return
        left_mean = float(np.mean(ommatidia[0]))
        right_mean = float(np.mean(ommatidia[1]))
        # Replace the coarse brightness with the actual retinal means.
        self._inject(net, self.channels.left_eye, self.gains.vision * _clip01(left_mean))
        self._inject(net, self.channels.right_eye, self.gains.vision * _clip01(right_mean))
        left_cells = self.channels.left_eye
        right_cells = self.channels.right_eye
        self._paint_retina(net, left_cells, ommatidia[0])
        self._paint_retina(net, right_cells, ommatidia[1])

    def _paint_retina(self, net: LIFNetwork, cells: np.ndarray, eye: np.ndarray) -> None:
        if cells.size == 0:
            return
        flat = np.asarray(eye, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return
        bins = np.array_split(flat, min(cells.size, flat.size))
        n = min(len(bins), cells.size)
        for i in range(n):
            value = float(np.mean(bins[i])) if bins[i].size else 0.0
            if value > 0:
                net.inject([int(cells[i])], self.gains.vision * _clip01(value))

    @staticmethod
    def _inject(net: LIFNetwork, indices: np.ndarray, current: float) -> None:
        if current != 0.0 and indices.size:
            net.inject(indices, current)

    @staticmethod
    def _inject_vector(
        net: LIFNetwork, indices: np.ndarray, values: np.ndarray, gain: float
    ) -> None:
        if indices.size == 0:
            return
        vec = np.asarray(values, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return
        scaled = np.clip(np.abs(vec), 0.0, 1.0)
        n = min(indices.size, scaled.size)
        for i in range(n):
            if scaled[i] > 0:
                net.inject([int(indices[i])], float(gain * scaled[i]))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def geometric_eyes(
    x_mm: float,
    y_mm: float,
    heading_rad: float,
    lights: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Two cosine eyes. `lights` are (x, y, intensity) in mm.

    This is a photoreceptor stand-in, not a privileged coordinate channel.
    """
    left = 0.0
    right = 0.0
    for lx, ly, intensity in lights:
        dx = lx - x_mm
        dy = ly - y_mm
        dist2 = dx * dx + dy * dy
        d0 = 280.0
        atten = 1.0 / (1.0 + dist2 / (d0 * d0))
        bearing = np.arctan2(dy, dx) - heading_rad
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        left += max(0.0, float(intensity) * np.cos(bearing - 0.7)) * atten
        right += max(0.0, float(intensity) * np.cos(bearing + 0.7)) * atten
    return _clip01(left), _clip01(right)
