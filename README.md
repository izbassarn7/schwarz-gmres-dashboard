# schwarz-gmres

Parallel Schwarz preconditioners (ASM, RAS, two-level) for GMRES on large sparse SLAE, hybrid CPU-GPU.

## Build

```bash
docker compose -f docker/docker-compose.yml up -d dev
docker exec -it <container> bash

cd /workspace
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

### With PETSc / Trilinos baselines

```bash
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DUSE_PETSC=ON -DUSE_TRILINOS=ON
ninja
```

## Run

```bash
# Sequential GMRES + ILU(0)
./src/schwarz_solve --matrix ../data/thermal1.mtx --precond ilu0

# GMRES + RAS (4 subdomains, overlap 2)
./src/schwarz_solve --matrix ../data/thermal1.mtx --precond ras \
    --overlap 2 --output ../results/run.jsonl
```

## Tests

```bash
cd build
ctest --output-on-failure
```

## Benchmarks

```bash
# Download SuiteSparse matrices
bash scripts/download_matrices.sh

# Run scaling tests
bash scripts/run_scaling.sh

# Generate plots
python3 scripts/plot_scaling.py results/
python3 scripts/compare.py results/
```

## Preconditioners

| Type | Class | Description |
|------|-------|-------------|
| `jacobi` | `JacobiPrecond` | Diagonal scaling |
| `ilu0` | `ILU0Precond` | ILU(0) incomplete factorization |
| `asm` | `ASMPrecond` | Additive Schwarz with overlap |
| `ras` | `RASPrecond` | Restricted Additive Schwarz |
| `twolevel_asm` | `TwoLevelSchwarzPrecond` | 2-level ASM + coarse grid |
| `twolevel_ras` | `TwoLevelSchwarzPrecond` | 2-level RAS + coarse grid |

## Project structure

```
src/core/       - CSR matrix, GMRES(m), BLAS-1 ops
src/precond/    - All preconditioners (Jacobi, ILU0, ASM, RAS, two-level)
src/decomp/     - METIS domain decomposition, overlap generation
src/gpu/        - CUDA kernels (cuSPARSE SpMV, batched ILU, GPU GMRES)
src/comm/       - MPI halo exchange, CUDA-aware MPI
src/mesh/       - Poisson generators (2D, 3D)
src/io/         - MatrixMarket reader, JSON metrics logger
tests/          - Unit tests
bench/          - Scaling benchmarks, PETSc/Trilinos baselines
scripts/        - Automation (download, run, plot, compare)
```
