"""Physical simulation engine for cross-layer workload auditing.

This module owns pulse-level Hamiltonian construction and bounded concurrent
stress definitions. We simulate a three-qubit line q0-Q-q2 where Q is the
active logical qubit for one-qubit characterization and (Q, q2) is the active
pair for echoed CR-CNOT workload replay.

Author: Anonymous Authors
Institution: Blinded for Review
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import qutip as qt


@dataclass(frozen=True)
class StressProfile:
    """Hardware stress profile used for bounded concurrent interference tests."""

    name: str
    J1Q: float
    JQ2: float
    A1: float
    A2: float
    Aro: float
    delta1: float
    delta2: float
    delta_ro: float

    @classmethod
    def from_mapping(cls, name: str, values: Dict[str, float]) -> "StressProfile":
        return cls(
            name=name,
            J1Q=float(values.get("J1Q", 0.0)),
            JQ2=float(values.get("JQ2", 0.0)),
            A1=float(values.get("A1", 0.0)),
            A2=float(values.get("A2", 0.0)),
            Aro=float(values.get("Aro", 0.0)),
            delta1=float(values.get("delta1", 0.0)),
            delta2=float(values.get("delta2", 0.0)),
            delta_ro=float(values.get("delta_ro", 0.0)),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "J1Q": self.J1Q,
            "JQ2": self.JQ2,
            "A1": self.A1,
            "A2": self.A2,
            "Aro": self.Aro,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta_ro": self.delta_ro,
        }


class PhysicsEngine:
    """Pulse-level physics backend based on QuTiP mesolve.

    The engine intentionally mirrors the exploratory notebook assumptions while
    providing reusable object-oriented methods for QPT and workload replay.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.I = qt.qeye(2)
        self.zero = qt.basis(2, 0)
        self.one = qt.basis(2, 1)

    @staticmethod
    def rx(theta: float) -> qt.Qobj:
        return qt.Qobj(
            [
                [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
                [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ]
        )

    @staticmethod
    def ry(theta: float) -> qt.Qobj:
        return qt.Qobj(
            [
                [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                [np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ]
        )

    @staticmethod
    def rz(theta: float) -> qt.Qobj:
        return qt.Qobj(
            [
                [np.exp(-1j * theta / 2.0), 0.0],
                [0.0, np.exp(1j * theta / 2.0)],
            ]
        )

    @staticmethod
    def pulse_shape(
        shape: str = "cos",
        A: float = 1.0,
        delta: float = 0.0,
        sigma: float = 0.1,
        alpha: float = 0.5,
        chirp_rate: float = 0.0,
        t_duration: float = 1.0,
    ) -> Callable[[float, Dict], float]:
        """Build a time-domain pulse envelope closure used by QuTiP."""
        t_center = t_duration / 2.0
        if shape == "cos":
            return lambda t, args: A * np.cos(delta * t)
        if shape == "gaussian":
            return lambda t, args: A * np.exp(-((t - t_center) ** 2) / (2.0 * sigma**2))
        if shape == "square":
            return (
                lambda t, args: A
                if (t_center - 0.2 * t_duration) <= t <= (t_center + 0.2 * t_duration)
                else 0.0
            )
        if shape == "chirp":
            return lambda t, args: A * np.cos((delta + chirp_rate * t) * t)
        if shape == "drag":
            return lambda t, args: A * (
                np.exp(-((t - t_center) ** 2) / (2.0 * sigma**2))
                - alpha
                * (t - t_center)
                / sigma**2
                * np.exp(-((t - t_center) ** 2) / (2.0 * sigma**2))
            )
        raise ValueError(f"Unknown pulse shape: {shape}")

    def stress_profiles(self) -> Dict[str, StressProfile]:
        """Load bounded stress regimes from configuration."""
        profiles = self.config.get("stress_profiles", {})
        return {
            name: StressProfile.from_mapping(name, values)
            for name, values in profiles.items()
        }

    def _background_terms_3q(
        self, stress: StressProfile, t_duration: float
    ) -> Tuple[qt.Qobj, List[List]]:
        h_couple_1q = stress.J1Q * qt.tensor(qt.sigmay(), qt.sigmax(), self.I)
        h_couple_q2 = stress.JQ2 * qt.tensor(self.I, qt.sigmax(), qt.sigmay())
        h_static = h_couple_1q + h_couple_q2

        coeff1 = self.pulse_shape(
            shape="chirp",
            A=stress.A1,
            delta=stress.delta1,
            chirp_rate=0.5,
            t_duration=t_duration,
        )
        coeff2 = self.pulse_shape(
            shape="chirp",
            A=stress.A2,
            delta=stress.delta2,
            chirp_rate=0.5,
            t_duration=t_duration,
        )
        coeff_ro = self.pulse_shape(
            shape="cos",
            A=stress.Aro,
            delta=stress.delta_ro,
            t_duration=t_duration,
        )

        h_drive1 = [qt.tensor(qt.sigmax(), self.I, self.I), coeff1]
        h_drive2 = [qt.tensor(self.I, self.I, qt.sigmax()), coeff2]
        h_ro = [qt.tensor(self.I, qt.sigmax(), self.I), coeff_ro]
        return h_static, [h_drive1, h_drive2, h_ro]

    def simulate_probe_state(
        self,
        qubit_q_input_dm: qt.Qobj,
        theta_gate: float,
        stress: StressProfile,
        nsteps: int = 50,
    ) -> qt.Qobj:
        """Run one-qubit probe evolution in the three-qubit environment."""
        state_init_3q = qt.tensor(self.zero.proj(), qubit_q_input_dm, self.zero.proj())

        gate_cfg = self.config.get("single_qubit_gate", {})
        t_duration = float(gate_cfg.get("duration", 1.0))
        sigma_gate = float(gate_cfg.get("sigma", 0.2))

        tlist = np.linspace(0.0, t_duration, nsteps)
        h_static, h_background = self._background_terms_3q(stress, t_duration)

        # Area calibration for H(t)=A f(t) sigma_y, enforcing angle theta_gate.
        a_q = (theta_gate / 2.0) / (sigma_gate * np.sqrt(2.0 * np.pi))
        coeff_q = self.pulse_shape(
            shape="gaussian", A=a_q, sigma=sigma_gate, t_duration=t_duration
        )
        h_gate_q = [qt.tensor(self.I, qt.sigmay(), self.I), coeff_q]

        h_total = [h_static] + h_background + [h_gate_q]
        sol = qt.mesolve(h_total, state_init_3q, tlist, c_ops=[], e_ops=[])
        return sol.states[-1].ptrace(1)

    def evolve_echoed_cr_cnot(
        self,
        rho_init_3q: qt.Qobj,
        stress: StressProfile,
        theta_zx: float,
        rx_target_corr: float,
        rz_control_corr: float,
        beta_ix: float,
        gamma_zi: float,
        sigma: float,
        t_duration: float,
        nsteps: int,
        apply_corrections: bool = True,
    ) -> qt.Qobj:
        """Evolve an echoed CR-CNOT pulse sequence on (Q, q2)."""
        rho = rho_init_3q.proj() if rho_init_3q.isket else rho_init_3q

        h_couple_1q = stress.J1Q * qt.tensor(qt.sigmay(), qt.sigmax(), self.I)
        h_couple_q2 = stress.JQ2 * qt.tensor(self.I, qt.sigmax(), qt.sigmay())
        h_static = h_couple_1q + h_couple_q2

        zqx2 = qt.tensor(self.I, qt.sigmaz(), qt.sigmax())
        ix2 = qt.tensor(self.I, self.I, qt.sigmax())
        zq = qt.tensor(self.I, qt.sigmaz(), self.I)
        h_cr_op = 0.5 * (zqx2 + beta_ix * ix2 + gamma_zi * zq)

        half_t = t_duration / 2.0
        omega_area_half = theta_zx
        omega_a = omega_area_half / (sigma * np.sqrt(2.0 * np.pi))

        def apply_1q(rho_in: qt.Qobj, U: qt.Qobj, which: int) -> qt.Qobj:
            ops = [self.I, self.I, self.I]
            ops[which] = U
            u_tot = qt.tensor(ops)
            return u_tot * rho_in * u_tot.dag()

        def evolve_segment(rho_in: qt.Qobj, seg_t: float, omega_sign: float) -> qt.Qobj:
            tlist = np.linspace(0.0, seg_t, nsteps)
            coeff_cr = self.pulse_shape(
                shape="gaussian", A=omega_sign * omega_a, sigma=sigma, t_duration=seg_t
            )
            h_cr = [h_cr_op, coeff_cr]
            h_bg_static, h_bg = self._background_terms_3q(stress, seg_t)
            h_total = [h_bg_static] + h_bg + [h_cr]
            sol = qt.mesolve(h_total, rho_in, tlist, c_ops=[], e_ops=[])
            return sol.states[-1]

        rho = evolve_segment(rho, half_t, omega_sign=+1.0)
        rho = apply_1q(rho, self.rx(np.pi), which=1)
        rho = evolve_segment(rho, half_t, omega_sign=-1.0)
        rho = apply_1q(rho, self.rx(np.pi), which=1)

        if apply_corrections:
            rho = apply_1q(rho, self.rx(rx_target_corr), which=2)
            rho = apply_1q(rho, self.rz(rz_control_corr), which=1)

        return rho
