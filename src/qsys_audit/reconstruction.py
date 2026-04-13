"""Reconstruction methods for cross-layer channel analysis.

This module provides three reconstruction pathways:
1. Process tomography (QPT) and Choi assembly.
2. Isometry-plus-noise fitting from a physical Choi matrix.
3. Shadow-assisted output-state estimation for lightweight audits.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import qutip as qt
from scipy.linalg import polar

from .physics_engine import PhysicsEngine, StressProfile


class ReconstructionEngine:
    """Cross-layer reconstruction helper built around a shared physics engine."""

    def __init__(self, physics: PhysicsEngine, config: Dict):
        self.physics = physics
        self.config = config
        self.i2 = qt.qeye(2)
        self.x2 = qt.sigmax()
        self.y2 = qt.sigmay()
        self.z2 = qt.sigmaz()
        self.zero = qt.basis(2, 0)
        self.one = qt.basis(2, 1)
        self.plus = (self.zero + self.one).unit()
        self.minus = (self.zero - self.one).unit()
        self.plus_i = (self.zero + 1j * self.one).unit()
        self.minus_i = (self.zero - 1j * self.one).unit()
        self.shadow_projectors = {
            "X": [self.plus.proj(), self.minus.proj()],
            "Y": [self.plus_i.proj(), self.minus_i.proj()],
            "Z": [self.zero.proj(), self.one.proj()],
        }

    def perform_qpt_pipeline(self, theta_gate: float, stress: StressProfile) -> qt.Qobj:
        """Run four-probe one-qubit QPT and return a 4x4 Choi matrix."""
        rho_in_list = [
            self.zero.proj(),
            self.one.proj(),
            self.plus.proj(),
            self.plus_i.proj(),
        ]

        rho_out_list = [
            self.physics.simulate_probe_state(rho_in, theta_gate, stress)
            for rho_in in rho_in_list
        ]
        rho0_out, rho1_out, rho_plus_out, rho_plus_i_out = rho_out_list

        a_out = rho0_out
        b_out = rho1_out
        s_term = a_out + b_out
        x_term = 2.0 * rho_plus_out - s_term
        y_term = 2.0 * rho_plus_i_out - s_term

        c_out = 0.5 * (x_term + 1j * y_term)
        d_out = 0.5 * (x_term - 1j * y_term)

        proj00 = self.zero * self.zero.dag()
        proj11 = self.one * self.one.dag()
        proj01 = self.zero * self.one.dag()
        proj10 = self.one * self.zero.dag()

        return (
            qt.tensor(proj00, a_out)
            + qt.tensor(proj01, c_out)
            + qt.tensor(proj10, d_out)
            + qt.tensor(proj11, b_out)
        )

    def ideal_choi_ry(self, theta: float) -> qt.Qobj:
        """Construct ideal Choi matrix for Ry(theta)."""
        u = self.physics.ry(theta)
        proj00 = self.zero * self.zero.dag()
        proj11 = self.one * self.one.dag()
        proj01 = self.zero * self.one.dag()
        proj10 = self.one * self.zero.dag()

        def evolve(rho: qt.Qobj) -> qt.Qobj:
            return u * rho * u.dag()

        a_out = evolve(proj00)
        b_out = evolve(proj11)
        c_out = evolve(proj01)
        d_out = evolve(proj10)
        return (
            qt.tensor(proj00, a_out)
            + qt.tensor(proj01, c_out)
            + qt.tensor(proj10, d_out)
            + qt.tensor(proj11, b_out)
        )

    @staticmethod
    def kraus_operators_from_choi(c_choi: qt.Qobj, tol: float = 1e-8) -> List[qt.Qobj]:
        """Extract trace-preserving Kraus operators from a Choi matrix."""
        c_mat = c_choi.full()
        eigvals, eigvecs = np.linalg.eigh(c_mat)

        kraus_ops_raw: List[qt.Qobj] = []
        for idx, lam in enumerate(eigvals):
            if np.real(lam) > tol:
                v = eigvecs[:, idx]
                k_mat = np.sqrt(lam) * v.reshape((2, 2), order="F")
                kraus_ops_raw.append(qt.Qobj(k_mat))

        if not kraus_ops_raw:
            return [qt.Qobj(np.zeros((2, 2), dtype=complex))]

        sum_ops = sum((k.dag() * k for k in kraus_ops_raw), qt.Qobj(np.zeros((2, 2))))
        trace_val = np.real(sum_ops.tr()) / 2.0
        norm_factor = 1.0 if trace_val <= 0.0 else 1.0 / np.sqrt(trace_val)
        return [k * norm_factor for k in kraus_ops_raw]

    @staticmethod
    def choi_from_kraus(kraus_ops: List[qt.Qobj]) -> qt.Qobj:
        """Assemble a Choi matrix from Kraus operators."""
        d = kraus_ops[0].shape[0]
        c = np.zeros((d * d, d * d), dtype=complex)
        for k in kraus_ops:
            v = k.full().reshape(d * d, order="F")
            c += np.outer(v, v.conj())
        return qt.Qobj(c)

    @staticmethod
    def su2_axis_angle(u_qobj: qt.Qobj, tol: float = 1e-10) -> Tuple[float, np.ndarray, qt.Qobj]:
        """Return axis-angle representation of a 2x2 unitary."""
        u = u_qobj.full()
        det_u = np.linalg.det(u)
        phase = np.angle(det_u) / 2.0
        u_su2 = u * np.exp(-1j * phase)

        tr_u = np.trace(u_su2)
        cos_half_theta = np.clip(np.real(tr_u) / 2.0, -1.0, 1.0)
        theta = 2.0 * np.arccos(cos_half_theta)

        if abs(theta) < tol:
            return 0.0, np.array([0.0, 0.0, 1.0]), qt.Qobj(u_su2)

        sigmas = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        axis = []
        for sigma in sigmas:
            val = np.trace(sigma.full().conj().T @ u_su2)
            axis_k = (1j * val) / (2.0 * np.sin(theta / 2.0))
            axis.append(np.real(axis_k))

        axis = np.array(axis)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > tol:
            axis = axis / axis_norm
        else:
            axis = np.array([0.0, 0.0, 1.0])

        return float(theta), axis, qt.Qobj(u_su2)

    def fit_unitary_plus_noise_from_choi(self, c_choi: qt.Qobj, tol: float = 1e-8) -> Dict:
        """Factor E_phys ~= N o U_eff and estimate Pauli-channel coordinates of N."""
        kraus_ops = self.kraus_operators_from_choi(c_choi, tol=tol)
        fro_norms = [np.linalg.norm(k.full(), "fro") for k in kraus_ops]
        k0 = kraus_ops[int(np.argmax(fro_norms))]

        svals = np.linalg.svd(k0.full(), compute_uv=False)
        u_eff_mat, _ = polar(k0.full())
        u_eff = qt.Qobj(u_eff_mat)

        theta_eff, n_axis, _ = self.su2_axis_angle(u_eff)

        l_ops = [k * u_eff.dag() for k in kraus_ops]
        paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]

        lambdas = []
        for j in range(1, 4):
            sigma_j = paulis[j]
            n_sigma_j = sum((l * sigma_j * l.dag() for l in l_ops), qt.Qobj(np.zeros((2, 2))))
            lam_j = 0.5 * np.trace(sigma_j.full().conj().T @ n_sigma_j.full()).real
            lambdas.append(float(lam_j))

        lam_x, lam_y, lam_z = lambdas
        p_i = (1.0 + lam_x + lam_y + lam_z) / 4.0
        p_x = (1.0 + lam_x - lam_y - lam_z) / 4.0
        p_y = (1.0 - lam_x + lam_y - lam_z) / 4.0
        p_z = (1.0 - lam_x - lam_y + lam_z) / 4.0

        probs = np.clip(np.array([p_i, p_x, p_y, p_z]), 0.0, 1.0)
        if probs.sum() > 0.0:
            probs = probs / probs.sum()

        return {
            "U_eff": u_eff,
            "theta_eff": float(theta_eff),
            "n_axis": n_axis,
            "dominant_singular_value": float(np.max(svals)),
            "pauli_lambdas": (lam_x, lam_y, lam_z),
            "pauli_probs": probs,
            "kraus_noise": l_ops,
        }

    @staticmethod
    def process_distance_d(c_ideal: qt.Qobj, c_phys: qt.Qobj) -> float:
        d = 2
        f_proc = (1.0 / (d**2)) * np.trace(c_ideal.full().conj().T @ c_phys.full())
        return float(1.0 - np.real(f_proc))

    @staticmethod
    def delta_log(u_ideal: qt.Qobj, u_eff: qt.Qobj) -> float:
        m = u_ideal.dag() * u_eff
        val = m.tr()
        return float(np.sqrt(max(0.0, 1.0 - (abs(val) ** 2) / 4.0)))

    @staticmethod
    def delta_inc(pauli_probs: np.ndarray) -> float:
        return float(1.0 - pauli_probs[0])

    def choi_partial_trace_output(self, c_choi: qt.Qobj) -> qt.Qobj:
        m = c_choi.full()
        pt = np.array(
            [
                [np.trace(m[0:2, 0:2]), np.trace(m[0:2, 2:4])],
                [np.trace(m[2:4, 0:2]), np.trace(m[2:4, 2:4])],
            ],
            dtype=complex,
        )
        return qt.Qobj(pt)

    @staticmethod
    def project_psd_qobj(x: qt.Qobj, tol: float = 1e-12) -> qt.Qobj:
        m = 0.5 * (x.full() + x.full().conj().T)
        vals, vecs = np.linalg.eigh(m)
        vals = np.where(vals > tol, vals, 0.0)
        m_psd = vecs @ np.diag(vals) @ vecs.conj().T
        return qt.Qobj(m_psd, dims=x.dims)

    def project_tp_choi_1q(self, c_choi: qt.Qobj) -> qt.Qobj:
        d = 2
        p = self.choi_partial_trace_output(c_choi).full()
        correction = np.kron((np.eye(d) - p) / d, np.eye(d))
        return qt.Qobj(c_choi.full() + correction, dims=c_choi.dims)

    def project_choi_to_cptp_1q(self, c_choi: qt.Qobj, n_iter: int = 20) -> qt.Qobj:
        c = qt.Qobj(c_choi.full(), dims=c_choi.dims)
        for _ in range(n_iter):
            c = self.project_psd_qobj(c)
            c = self.project_tp_choi_1q(c)
        c = self.project_psd_qobj(c)
        c = self.project_tp_choi_1q(c)
        c = qt.Qobj(0.5 * (c.full() + c.full().conj().T), dims=c.dims)
        return c

    def process_distance_d_physical(self, c_ideal: qt.Qobj, c_phys: qt.Qobj) -> Tuple[float, qt.Qobj]:
        d = 2
        c_proj = self.project_choi_to_cptp_1q(c_phys)
        f_proc = (1.0 / (d**2)) * np.trace(c_ideal.full().conj().T @ c_proj.full())
        f_proc = float(np.clip(np.real(f_proc), 0.0, 1.0))
        return 1.0 - f_proc, c_proj

    @staticmethod
    def median_of_means(values: List[float], k_blocks: int = 10) -> float:
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        if n == 0:
            raise ValueError("Need at least one sample.")

        k_blocks = max(1, min(k_blocks, n))
        block = n // k_blocks
        usable = arr[: k_blocks * block]
        block_means = usable.reshape(k_blocks, block).mean(axis=1)
        return float(np.median(block_means))

    @staticmethod
    def project_density_matrix(rho: qt.Qobj) -> qt.Qobj:
        m = 0.5 * (rho + rho.dag())
        vals, vecs = np.linalg.eigh(m.full())
        vals = np.clip(np.real(vals), 0.0, None)

        if vals.sum() <= 0:
            vals = np.array([1.0, 0.0], dtype=float)
        vals = vals / vals.sum()

        rho_phys = qt.Qobj(np.zeros((2, 2), dtype=complex))
        for idx, val in enumerate(vals):
            vec = vecs[:, idx]
            rho_phys += val * qt.Qobj(np.outer(vec, vec.conj()))
        return rho_phys

    def single_qubit_pauli_shadow_samples(
        self, rho: qt.Qobj, n_shots: int, seed: int = 0
    ) -> Tuple[List[qt.Qobj], List[Tuple[str, int]]]:
        """Generate one-qubit Pauli shadow samples and shot records."""
        rng = np.random.default_rng(seed)
        bases = ["X", "Y", "Z"]
        shadows: List[qt.Qobj] = []
        records: List[Tuple[str, int]] = []

        for _ in range(n_shots):
            basis = bases[int(rng.integers(0, 3))]
            projs = self.shadow_projectors[basis]

            probs = np.array([float((p * rho).tr().real) for p in projs], dtype=float)
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum() if probs.sum() > 0 else np.array([0.5, 0.5])

            bit = int(rng.choice([0, 1], p=probs))
            p_b = projs[bit]
            rho_hat = 3.0 * p_b - self.i2
            shadows.append(rho_hat)
            records.append((basis, bit))

        return shadows, records

    def estimate_rho_from_shadows(self, shadows: List[qt.Qobj], k_blocks: int = 10) -> qt.Qobj:
        """Estimate a physical one-qubit density matrix from Pauli shadows."""
        ex = self.median_of_means([np.real((self.x2 * s).tr()) for s in shadows], k_blocks)
        ey = self.median_of_means([np.real((self.y2 * s).tr()) for s in shadows], k_blocks)
        ez = self.median_of_means([np.real((self.z2 * s).tr()) for s in shadows], k_blocks)
        rho_est = 0.5 * (self.i2 + ex * self.x2 + ey * self.y2 + ez * self.z2)
        return self.project_density_matrix(rho_est)
