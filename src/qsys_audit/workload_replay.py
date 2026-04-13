"""Workload replay from algorithmic circuits to pulse-aware physical fidelity.

We map two-qubit workloads into a supported gate basis and replay them on a
hybrid model where one-qubit gates are ideal and CNOT uses the echoed CR-CNOT
physical sequence. This mirrors the original notebook's cross-layer framing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import qutip as qt
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit, transpile

from .physics_engine import PhysicsEngine, StressProfile


@dataclass(frozen=True)
class WorkloadMetrics:
    workload: str
    stress: str
    f_expected: float
    f_worst: float
    f_spread: float
    n_cx: int
    n_1q: int
    depth: int
    size: int


class WorkloadReplay:
    """Replay engine for two-qubit algorithmic workloads."""

    def __init__(self, physics: PhysicsEngine, config: Dict):
        self.physics = physics
        self.config = config
        self.i2 = qt.qeye(2)
        self.zero = qt.basis(2, 0)
        self.one = qt.basis(2, 1)
        self.plus = (self.zero + self.one).unit()
        self.h2 = qt.Qobj([[1, 1], [1, -1]], dims=[[2], [2]]) / np.sqrt(2.0)
        self.u_cnot = qt.Qobj(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            ),
            dims=[[2, 2], [2, 2]],
        )
        self._workload_aliases = {
            "ghz": "ghz",
            "graph_state": "graphstate",
            "graphstate": "graphstate",
            "w_state": "wstate",
            "wstate": "wstate",
            "qft": "qft",
            "qaoa": "qaoa",
            "vqe_su2": "vqe_su2",
            "vqe_two_local": "vqe_two_local",
        }

    @staticmethod
    def _proj(psi: qt.Qobj) -> qt.Qobj:
        return psi * psi.dag()

    def available_workloads(self) -> List[str]:
        return sorted(self._workload_aliases.keys())

    def resolve_workload_name(self, name: str) -> str:
        key = name.strip().lower()
        if key not in self._workload_aliases:
            raise ValueError(
                f"Unsupported workload '{name}'. Supported aliases: {sorted(self._workload_aliases.keys())}"
            )
        return self._workload_aliases[key]

    def load_workload_circuit(self, workload_name: str) -> QuantumCircuit:
        """Load a two-qubit MQT-Bench algorithm-level workload circuit."""
        canonical_name = self.resolve_workload_name(workload_name)
        qc = get_benchmark(
            benchmark=canonical_name,
            level=BenchmarkLevel.ALG,
            circuit_size=2,
            random_parameters=True,
            generate_mirror_circuit=False,
        )
        try:
            return qc.remove_final_measurements(inplace=False)
        except Exception:
            return qc

    def normalize_2q_circuit_for_replay(self, qc_nom: QuantumCircuit) -> QuantumCircuit:
        qc_work = qc_nom
        for _ in range(6):
            try:
                qc_work = qc_work.decompose()
            except Exception:
                break
        return transpile(
            qc_work,
            basis_gates=["rx", "ry", "rz", "cx"],
            optimization_level=0,
        )

    @staticmethod
    def qiskit_circuit_to_oplist(qc: QuantumCircuit) -> List[Dict]:
        ops: List[Dict] = []
        for item in qc.data:
            inst = item.operation
            qinds = [qc.find_bit(q).index for q in item.qubits]
            params = [float(p) for p in inst.params]
            ops.append({"name": inst.name, "qubits": qinds, "params": params})
        return ops

    def apply_1q_to_3q_dm(self, rho3: qt.Qobj, u: qt.Qobj, which_phys: int) -> qt.Qobj:
        ops = [self.i2, self.i2, self.i2]
        ops[which_phys] = u
        u_tot = qt.tensor(ops)
        return u_tot * rho3 * u_tot.dag()

    def apply_1q_to_2q_ket(self, psi2: qt.Qobj, u: qt.Qobj, which_logical: int) -> qt.Qobj:
        if which_logical == 0:
            u_tot = qt.tensor(u, self.i2)
        elif which_logical == 1:
            u_tot = qt.tensor(self.i2, u)
        else:
            raise ValueError("which_logical must be 0 or 1")
        return (u_tot * psi2).unit()

    def apply_cnot_ideal_2q(self, psi2: qt.Qobj, control: int, target: int) -> qt.Qobj:
        if (control, target) == (0, 1):
            return (self.u_cnot * psi2).unit()
        if (control, target) == (1, 0):
            hh = qt.tensor(self.h2, self.h2)
            return (hh * self.u_cnot * hh * psi2).unit()
        raise ValueError(f"Unsupported CX direction: {(control, target)}")

    def simulate_2q_workload_ideal(self, oplist: Sequence[Dict], psi_in_2q: qt.Qobj) -> qt.Qobj:
        psi = psi_in_2q
        for op in oplist:
            name = op["name"]
            qs = op["qubits"]
            ps = op["params"]
            if name == "rx":
                psi = self.apply_1q_to_2q_ket(psi, self.physics.rx(ps[0]), qs[0])
            elif name == "ry":
                psi = self.apply_1q_to_2q_ket(psi, self.physics.ry(ps[0]), qs[0])
            elif name == "rz":
                psi = self.apply_1q_to_2q_ket(psi, self.physics.rz(ps[0]), qs[0])
            elif name == "cx":
                psi = self.apply_cnot_ideal_2q(psi, qs[0], qs[1])
            else:
                raise NotImplementedError(f"Unsupported ideal gate: {name}")
        return psi.unit()

    def simulate_2q_workload_physical(
        self,
        oplist: Sequence[Dict],
        psi_in_2q: qt.Qobj,
        stress: StressProfile,
    ) -> qt.Qobj:
        """Replay workload on physical q0-Q-q2 model and return rho(Q,q2)."""
        cnot_cfg = self.config.get("cnot", {})
        rho3 = qt.tensor(self._proj(self.zero), self._proj(psi_in_2q))

        for op in oplist:
            name = op["name"]
            qs = op["qubits"]
            ps = op["params"]

            if name in {"rx", "ry", "rz"}:
                gate = {
                    "rx": self.physics.rx,
                    "ry": self.physics.ry,
                    "rz": self.physics.rz,
                }[name]
                u = gate(ps[0])
                phys = 1 if qs[0] == 0 else 2
                rho3 = self.apply_1q_to_3q_dm(rho3, u, phys)
                continue

            if name == "cx":
                control, target = qs
                if (control, target) == (0, 1):
                    rho3 = self.physics.evolve_echoed_cr_cnot(
                        rho3,
                        stress=stress,
                        theta_zx=float(cnot_cfg["theta_zx"]),
                        rx_target_corr=float(cnot_cfg["rx_target_corr"]),
                        rz_control_corr=float(cnot_cfg["rz_control_corr"]),
                        beta_ix=float(cnot_cfg.get("beta_ix", 0.2)),
                        gamma_zi=float(cnot_cfg.get("gamma_zi", 0.05)),
                        sigma=float(cnot_cfg.get("sigma", 0.25)),
                        t_duration=float(cnot_cfg.get("duration", 1.5)),
                        nsteps=int(cnot_cfg.get("nsteps", 120)),
                    )
                elif (control, target) == (1, 0):
                    rho3 = self.apply_1q_to_3q_dm(rho3, self.h2, 1)
                    rho3 = self.apply_1q_to_3q_dm(rho3, self.h2, 2)
                    rho3 = self.physics.evolve_echoed_cr_cnot(
                        rho3,
                        stress=stress,
                        theta_zx=float(cnot_cfg["theta_zx"]),
                        rx_target_corr=float(cnot_cfg["rx_target_corr"]),
                        rz_control_corr=float(cnot_cfg["rz_control_corr"]),
                        beta_ix=float(cnot_cfg.get("beta_ix", 0.2)),
                        gamma_zi=float(cnot_cfg.get("gamma_zi", 0.05)),
                        sigma=float(cnot_cfg.get("sigma", 0.25)),
                        t_duration=float(cnot_cfg.get("duration", 1.5)),
                        nsteps=int(cnot_cfg.get("nsteps", 120)),
                    )
                    rho3 = self.apply_1q_to_3q_dm(rho3, self.h2, 1)
                    rho3 = self.apply_1q_to_3q_dm(rho3, self.h2, 2)
                else:
                    raise NotImplementedError(f"Unsupported CX direction: {(control, target)}")
                continue

            raise NotImplementedError(f"Unsupported physical gate: {name}")

        return rho3.ptrace([1, 2])

    def benchmark_workload(self, workload_name: str, stress: StressProfile) -> WorkloadMetrics:
        """Evaluate expected and worst-case fidelity for one workload/stress pair."""
        qc_nom = self.load_workload_circuit(workload_name)
        tqc = self.normalize_2q_circuit_for_replay(qc_nom)
        oplist = self.qiskit_circuit_to_oplist(tqc)

        n_cx = sum(1 for op in oplist if op["name"] == "cx")
        n_1q = sum(1 for op in oplist if op["name"] in {"rx", "ry", "rz"})

        input_states: List[Tuple[str, qt.Qobj]] = [
            ("00", qt.tensor(self.zero, self.zero)),
            ("10", qt.tensor(self.one, self.zero)),
            ("+0", qt.tensor(self.plus, self.zero)),
            ("0+", qt.tensor(self.zero, self.plus)),
        ]

        fvals: List[float] = []
        for _, psi_in in input_states:
            psi_id = self.simulate_2q_workload_ideal(oplist, psi_in)
            rho_phys = self.simulate_2q_workload_physical(oplist, psi_in, stress)
            fidelity = float((self._proj(psi_id) * rho_phys).tr().real)
            fvals.append(fidelity)

        fvals_np = np.asarray(fvals, dtype=float)
        return WorkloadMetrics(
            workload=workload_name,
            stress=stress.name,
            f_expected=float(np.mean(fvals_np)),
            f_worst=float(np.min(fvals_np)),
            f_spread=float(np.max(fvals_np) - np.min(fvals_np)),
            n_cx=n_cx,
            n_1q=n_1q,
            depth=int(tqc.depth()),
            size=int(tqc.size()),
        )

    def benchmark_table(self, workload_names: Sequence[str], stress_names: Sequence[str]) -> pd.DataFrame:
        """Convenience API for multi-workload summary tables."""
        profiles = self.physics.stress_profiles()
        rows = []
        for workload in workload_names:
            for s_name in stress_names:
                metric = self.benchmark_workload(workload, profiles[s_name])
                rows.append(
                    {
                        "Workload": metric.workload,
                        "Stress": metric.stress,
                        "F_expected": metric.f_expected,
                        "F_worst": metric.f_worst,
                        "F_spread": metric.f_spread,
                        "n_cx": metric.n_cx,
                        "n_1q": metric.n_1q,
                        "depth": metric.depth,
                        "size": metric.size,
                    }
                )
        return pd.DataFrame(rows)
