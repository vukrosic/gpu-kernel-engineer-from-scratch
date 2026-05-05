# Month 09: PyTorch Integration

## Goal

Connect custom kernels to the ecosystem ML engineers actually use.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 33 | PyTorch baselines | Compare custom kernels to PyTorch ops. | Baseline benchmark script exists. |
| 34 | Custom op wrapper | Wrap one kernel in a clean Python API. | API is usable from an example. |
| 35 | Testing GPU code | Shape, dtype, and tolerance tests. | Test matrix is documented. |
| 36 | Debugging GPU kernels | Narrow failed GPU tests by shape, dtype, masks, indexing, and launch setup. | You can debug one failing kernel case methodically. |

## Minimum Viable Month

One custom kernel callable from a PyTorch-style workflow.

## Portfolio Note

Explain how custom kernels fit into an ML workflow instead of living as isolated demos.
