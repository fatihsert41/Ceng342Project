# Mixed-Precision Pipelined BiCGSTAB Solver

**CENG342 — Parallel Programming Project**
**Group 11**

---

## Overview

This project implements a **mixed-precision pipelined BiCGSTAB** (Biconjugate Gradient Stabilized) solver for large sparse linear systems `Ax = b`. Three parallel versions are provided:

1. **Serial** — reference implementation.
2. **OpenMP** — shared-memory parallelisation.
3. **MPI + OpenMP** — distributed-memory parallelisation, pipelined with `MPI_Iallreduce` for non-blocking global reductions.

Matrix values are stored in **FP32** while vector arithmetic stays in **FP64**, halving the matrix memory footprint without affecting convergence.

## Repository Contents

| File | Description |
|------|-------------|
| `Group11_Project_Code.zip` | Full source tree (`src/`, `include/`, `Makefile`, `bench.sh`) |
| `Group11_Project_Report.pdf` | Compiled project report (PDF) |
| `rapor.tex` | LaTeX source of the report |
| `figures/` | Screenshots used in the report |

## Build

Unzip `Group11_Project_Code.zip`, then on a Linux / WSL machine with `gcc` and OpenMPI:

```bash
make all
```

This produces three binaries:

- `bicgstab_serial`
- `bicgstab_omp`
- `bicgstab_mpi`

## Run

```bash
# Serial
./bicgstab_serial 500000 1000 1e-10

# OpenMP (4 threads)
OMP_NUM_THREADS=4 ./bicgstab_omp 500000 1000 1e-10

# MPI (4 ranks, 1 thread per rank)
OMP_NUM_THREADS=1 mpirun --oversubscribe -np 4 ./bicgstab_mpi 500000 1000 1e-10
```

Arguments: `N max_iter tolerance`.

## Results Summary

Strong-scaling on WSL2 Ubuntu (4 logical cores, OpenMPI 4.1.6, gcc -O2):

| N | Serial | OMP-4 | MPI-4 | S<sub>OMP</sub> | S<sub>MPI</sub> |
|---:|---:|---:|---:|:---:|:---:|
| 100,000 | 31.0 ms | 17.7 ms | 36.2 ms | **1.75×** | 0.86× |
| 500,000 | 165.1 ms | 126.9 ms | 226.0 ms | **1.30×** | 0.73× |
| 2,000,000 | 660.5 ms | 496.4 ms | 825.9 ms | **1.33×** | 0.80× |

All three versions converge to **bit-level identical** numerical solutions
(`|r|/|b| < 1e-10` in every test).

## Key Techniques

- **CSR** sparse-matrix storage with parallel FP32 mirror.
- **OpenMP** reduction (`reduction(+:s)`) for inner products.
- **MPI** row-block distribution with `MPI_Allgatherv` for SpMV.
- **Pipelining**: non-blocking `MPI_Iallreduce` overlapped with the next `Allgatherv`.
- **Mixed precision** SpMV: FP32 matrix × FP64 vector → FP64 accumulator.

See `Group11_Project_Report.pdf` for the full methodology and discussion.
