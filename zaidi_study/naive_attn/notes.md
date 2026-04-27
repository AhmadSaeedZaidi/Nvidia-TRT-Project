# Naive Attention Notes

## Data Layout
- $K, Q, V$ are $N \times d_k$ matrices.
- Data is passed in row-major order.
- To access the dimensions of the $i$-th token's query vector: `Q[i * d_k]` to `Q[i * d_k + d_k - 1]`.

## Math Basics
- Overall formula: $\text{attention} = \text{Softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$
- $\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$
- Find $\exp(z_i)$ for all $k$ and sum them up for the denominator of the softmax.

## super naive
- 3 explicit kernels (Matrix Multiplication and Softmax).
- Materializes the (`seq_len` $\times$ `seq_len`) attention score matrix in global memory.
- `Kernel 1`: Multiplies $Q$ and $K^T$, scaling by $\sqrt{d_k}$.
- `Kernel 2`: Applies Softmax row-wise to the $S$ matrix to get attention probabilities $P$.
- `Kernel 3`: Multiplies $P$ and $V$ to get the final output. We launch `seq_len * d_k` threads, each responsible for computing one element of the output matrix ($N \times d_k$).

## v2
- `Kernel 1`
  - Uses **Shared Memory Tiling**
  - Uses **Thread Coarsening**
- `Kernel 2`
  - Abandons the "1 thread per row" model which was too slow.
  - Now, an entire thread block cooperatively computes a single row.
  - Uses highly optimized warp-shuffle intrinsics (`__shfl_down_sync`) and shared memory to perform parallel tree-reductions to find the `max` and `sum` across the row simultaneously.
- `Kernel 3`
  - Applies the same advanced Matrix Multiplication techniques (Shared Memory Tiling + Thread Coarsening) from Kernel 1 to compute the final output embeddings efficiently. 

