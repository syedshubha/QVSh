"""Reproduce the shadow convergence study for the single-qubit gate suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import qutip as qt
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsys_audit.physics_engine import PhysicsEngine, StressProfile
from qsys_audit.reconstruction import ReconstructionEngine


def load_config() -> Dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_gate_specs(physics: PhysicsEngine) -> Dict[str, Dict]:
    sigma = float(physics.config.get("single_qubit_gate", {}).get("sigma", 0.2))
    duration = float(physics.config.get("single_qubit_gate", {}).get("duration", 1.0))
    return {
        "Rx_pi_2": {
            "ideal_U": physics.rx(np.pi / 2.0),
            "segments": [{"axis": "x", "theta": np.pi / 2.0, "sigma": sigma, "duration": duration}],
            "virtual_post": [],
        },
        "Ry_pi_2": {
            "ideal_U": physics.ry(np.pi / 2.0),
            "segments": [{"axis": "y", "theta": np.pi / 2.0, "sigma": sigma, "duration": duration}],
            "virtual_post": [],
        },
        "Rz_pi_2": {
            "ideal_U": physics.rz(np.pi / 2.0),
            "segments": [],
            "virtual_post": [physics.rz(np.pi / 2.0)],
        },
        "T": {
            "ideal_U": physics.rz(np.pi / 4.0),
            "segments": [],
            "virtual_post": [physics.rz(np.pi / 4.0)],
        },
    }


def ideal_choi_from_unitary(U: qt.Qobj) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)

    proj00 = zero * zero.dag()
    proj11 = one * one.dag()
    proj01 = zero * one.dag()
    proj10 = one * zero.dag()

    def evolve(rho: qt.Qobj) -> qt.Qobj:
        return U * rho * U.dag()

    A_out = evolve(proj00)
    B_out = evolve(proj11)
    C_out = evolve(proj01)
    D_out = evolve(proj10)

    return (
        qt.tensor(proj00, A_out)
        + qt.tensor(proj01, C_out)
        + qt.tensor(proj10, D_out)
        + qt.tensor(proj11, B_out)
    )


def _gaussian_theta_to_amplitude(theta: float, sigma: float) -> float:
    return (theta / 2.0) / (sigma * np.sqrt(2.0 * np.pi))


def _gate_axis_op(physics: PhysicsEngine, axis: str) -> qt.Qobj:
    if axis == "x":
        return qt.tensor(physics.I, qt.sigmax(), physics.I)
    if axis == "y":
        return qt.tensor(physics.I, qt.sigmay(), physics.I)
    if axis == "z":
        return qt.tensor(physics.I, qt.sigmaz(), physics.I)
    raise ValueError(f"Unsupported gate axis: {axis}")


def get_final_state_for_gate(
    physics: PhysicsEngine,
    rho_q_in: qt.Qobj,
    gate_spec: Dict,
    stress: StressProfile,
    nsteps: int,
) -> qt.Qobj:
    rho_init_3q = qt.tensor(physics.zero.proj(), rho_q_in, physics.zero.proj())

    seg_durations = [seg.get("duration", 1.0) for seg in gate_spec.get("segments", [])]
    duration = max([1.0] + seg_durations)

    tlist = np.linspace(0.0, duration, nsteps)
    h_static, h_background = physics._background_terms_3q(stress, duration)
    h_total = [h_static] + h_background

    for seg in gate_spec.get("segments", []):
        sigma = float(seg.get("sigma", 0.2))
        amplitude = _gaussian_theta_to_amplitude(float(seg["theta"]), sigma)
        coeff = physics.pulse_shape(shape="gaussian", A=amplitude, sigma=sigma, t_duration=duration)
        h_total.append([_gate_axis_op(physics, seg["axis"]), coeff])

    sol = qt.mesolve(h_total, rho_init_3q, tlist, c_ops=[], e_ops=[])
    rho_q = sol.states[-1].ptrace(1)

    for u_post in gate_spec.get("virtual_post", []):
        rho_q = u_post * rho_q * u_post.dag()
    return rho_q


def _choi_from_four_outputs(outputs: List[qt.Qobj]) -> qt.Qobj:
    rho0_out, rho1_out, rho_plus_out, rho_plus_i_out = outputs
    A_out = rho0_out
    B_out = rho1_out
    S = A_out + B_out
    X = 2.0 * rho_plus_out - S
    Y = 2.0 * rho_plus_i_out - S
    C_out = 0.5 * (X + 1j * Y)
    D_out = 0.5 * (X - 1j * Y)

    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    proj00 = zero * zero.dag()
    proj11 = one * one.dag()
    proj01 = zero * one.dag()
    proj10 = one * zero.dag()

    return (
        qt.tensor(proj00, A_out)
        + qt.tensor(proj01, C_out)
        + qt.tensor(proj10, D_out)
        + qt.tensor(proj11, B_out)
    )


def perform_exact_process(
    physics: PhysicsEngine,
    gate_spec: Dict,
    stress: StressProfile,
    nsteps: int,
) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    plus = (zero + one).unit()
    plus_i = (zero + 1j * one).unit()

    probes = [zero.proj(), one.proj(), plus.proj(), plus_i.proj()]
    outputs = [get_final_state_for_gate(physics, rho_in, gate_spec, stress, nsteps) for rho_in in probes]
    return _choi_from_four_outputs(outputs)


def perform_shadow_process(
    physics: PhysicsEngine,
    recon: ReconstructionEngine,
    gate_spec: Dict,
    stress: StressProfile,
    n_shots: int,
    k_blocks: int,
    seed: int,
    nsteps: int,
) -> qt.Qobj:
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    plus = (zero + one).unit()
    plus_i = (zero + 1j * one).unit()

    probes = [zero.proj(), one.proj(), plus.proj(), plus_i.proj()]
    outputs = []

    for idx, rho_in in enumerate(probes):
        rho_exact = get_final_state_for_gate(physics, rho_in, gate_spec, stress, nsteps)
        shadows, _ = recon.single_qubit_pauli_shadow_samples(rho_exact, n_shots=n_shots, seed=seed + 1000 * idx)
        rho_est = recon.estimate_rho_from_shadows(shadows, k_blocks=k_blocks)
        outputs.append(rho_est)

    return _choi_from_four_outputs(outputs)


def classify_verdict(d_val: float, dlog: float, dinc: float, cfg: Dict) -> str:
    th = cfg.get("verification_thresholds", {})
    if d_val <= th.get("eps_d_pass", 0.02) and dlog <= th.get("eps_log_pass", 0.05) and dinc <= th.get("eps_inc_pass", 0.05):
        return "PASS"
    if d_val >= th.get("d_hard", 0.30) or dlog >= th.get("log_hard", 0.90) or dinc >= th.get("inc_hard", 0.50):
        return "HARD FAIL"
    return "SOFT FAIL"


def choose_mom_k(n_shots: int, k_max: int = 10, min_block: int = 20) -> int:
    return max(1, min(k_max, n_shots // min_block))


def analyze_exact(recon: ReconstructionEngine, gate_spec: Dict, c_raw: qt.Qobj, cfg: Dict) -> Dict:
    c_ideal = ideal_choi_from_unitary(gate_spec["ideal_U"])
    d_val, c_phys = recon.process_distance_d_physical(c_ideal, c_raw)
    fit = recon.fit_unitary_plus_noise_from_choi(c_phys)
    dlog = recon.delta_log(gate_spec["ideal_U"], fit["U_eff"])
    dinc = recon.delta_inc(fit["pauli_probs"])
    return {
        "D": float(d_val),
        "verdict": classify_verdict(float(d_val), float(dlog), float(dinc), cfg),
    }


def run() -> None:
    cfg = load_config()
    physics = PhysicsEngine(cfg)
    recon = ReconstructionEngine(physics, cfg)

    gate_specs = build_gate_specs(physics)
    stresses = physics.stress_profiles()

    shot_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    n_trials = 20
    nsteps = 50

    rows = []

    for gate_name, gate_spec in gate_specs.items():
        for stress_name in ["calibrated", "moderate", "aggressive"]:
            stress = stresses[stress_name]

            c_exact = perform_exact_process(physics, gate_spec, stress, nsteps=nsteps)
            exact = analyze_exact(recon, gate_spec, c_exact, cfg)

            for n_shots in shot_list:
                k_eff = choose_mom_k(n_shots)
                d_vals = []
                verdict_matches = []

                for trial in range(n_trials):
                    c_shadow = perform_shadow_process(
                        physics,
                        recon,
                        gate_spec,
                        stress,
                        n_shots=n_shots,
                        k_blocks=k_eff,
                        seed=10000 + n_shots + 111 * trial,
                        nsteps=nsteps,
                    )
                    shadow = analyze_exact(recon, gate_spec, c_shadow, cfg)
                    d_vals.append(shadow["D"])
                    verdict_matches.append(shadow["verdict"] == exact["verdict"])

                d_vals_np = np.asarray(d_vals, dtype=float)
                rows.append(
                    {
                        "Gate": gate_name,
                        "Stress": stress_name,
                        "Shots": n_shots,
                        "K_eff": k_eff,
                        "D_mean": float(np.mean(d_vals_np)),
                        "D_var": float(np.var(d_vals_np, ddof=1)) if len(d_vals_np) > 1 else 0.0,
                        "D_bias_abs": float(abs(np.mean(d_vals_np) - exact["D"])),
                        "Verdict_stability_pct": 100.0 * float(np.mean(verdict_matches)),
                        "D_exact": float(exact["D"]),
                        "Exact_verdict": exact["verdict"],
                    }
                )

    df = pd.DataFrame(rows)
    print("We evaluate shadow convergence for our single-qubit gate suite.")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
