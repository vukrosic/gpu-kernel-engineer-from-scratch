# Month 05: Softmax And Normalization

## Goal

Build transformer-adjacent kernels that appear in real AI systems.

## Weeks

| Week | Topic | Main Build | Done When |
| --- | --- | --- | --- |
| 17 | Softmax math | Numerically stable row-wise softmax. | Matches NumPy or PyTorch. |
| 18 | Fused softmax | Combine max, exp, sum, and divide. | Benchmark compares fused and unfused. |
| 19 | LayerNorm | Forward LayerNorm. | Correctness and timing are documented. |
| 20 | Checkpoint | Normalization systems note. | You can explain why normalization appears in transformers. |

## Minimum Viable Month

Correct row-wise softmax and a numerical-stability explanation.

## Portfolio Note

Connect softmax and normalization to transformer workloads.
