# qsys_audit

We present an offline workload auditor for quantum operating systems and schedulers. Our framework bridges pulse-level physical simulation and workload-level risk profiling by combining Hamiltonian evolution, process reconstruction, and replay of algorithmic workloads under bounded concurrent stress.

## Why This Tool

We designed this artifact to answer a systems question: how much scheduler-visible workload reliability degrades when low-level hardware stress changes from calibrated to aggressive conditions.

Our cross-layer method includes:

1. Pulse-level simulation of a three-qubit line model with static coupling, neighbor drives, and readout crosstalk.
2. Single-qubit process reconstruction with Choi-based diagnostics and coherent/noise decomposition.
3. Workload replay that maps two-qubit algorithmic circuits onto ideal single-qubit gates plus a physical echoed CR-CNOT implementation.
4. Fidelity risk reporting with expected fidelity, worst-case fidelity, and fidelity spread.

## Repository Layout

- src/qsys_audit/physics_engine.py: QuTiP mesolve logic, Hamiltonian builders, and bounded concurrent stress definitions.
- src/qsys_audit/reconstruction.py: QPT, isometry fit, and shadow-assisted estimation utilities.
- src/qsys_audit/workload_replay.py: Workload replay with ideal single-qubit gates and physical echoed CR-CNOT execution.
- config.yaml: Hardware limits, detunings, stress profiles, and calibration constants.
- audit_cli.py: Command-line interface for workload risk audits.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run a workload audit under aggressive stress:

```bash
python audit_cli.py --workload w_state --stress aggressive
```

3. Expected output fields include:

- Expected Fidelity
- Worst-Case Fidelity
- Fidelity Spread
- Verdict

## CLI Usage

```bash
python audit_cli.py --workload <name> --stress <calibrated|moderate|aggressive>
```

Optional JSON output:

```bash
python audit_cli.py --workload ghz --stress moderate --json
```

Supported workload aliases include: ghz, graph_state, w_state, qft, qaoa, vqe_su2, and vqe_two_local.

## Methodology Summary

We model physical execution in a three-qubit topology q1-Q-q2:

- q1 is an idling neighbor.
- Q and q2 are the active logical pair.
- One-qubit probes and workloads are executed under configurable concurrent interference.

For reconstruction, we perform one-qubit QPT and fit an isometry-plus-noise factorization $E_{phys} \approx N \circ U_{eff}$. We then report process distance and coherent/incoherent deviation metrics. For workload-level risk, we replay algorithmic circuits after decomposition into $\{rx, ry, rz, cx\}$, where $cx$ is realized by the echoed CR-CNOT pulse model.

