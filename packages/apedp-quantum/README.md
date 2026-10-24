# APEDP-Quantum

Autonomous Quantum Many-Body Discovery, Matrix Product Operator (MPO) Hamiltonians, and Density Matrix Renormalization Group (DMRG) Engine.

## Installation

```bash
pip install -e packages/apedp-quantum
```

## Features

- **DMRG Solver**: Variational 1D quantum spin-chain solver with MPS/MPO representation.
- **Exact Diagonalization Parity**: Benchmarked against exact sparse Hamiltonian diagonalization to machine precision ($< 10^{-12}$).
- **Quantum Vacuum Sandboxes**: False vacuum decay and zero-point fluctuation analysis.
