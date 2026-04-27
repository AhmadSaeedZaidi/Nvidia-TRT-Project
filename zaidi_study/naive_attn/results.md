========= CORRECTNESS =========
K1 (QK^T) Max Difference   : 0
K2 (Softmax) Max Difference: 6.51926e-09
K3 (P*V) Max Difference    : 0

========= PERFORMANCE (ms) =========
K1 (Q*K^T)     : v1 = 0.1498 ms | v2 = 0.1464 ms | Speedup = 1.0232x
K2 (Softmax)   : v1 = 0.1910 ms | v2 = 0.1229 ms | Speedup = 1.5542x
K3 (P*V)       : v1 = 0.1416 ms | v2 = 0.0901 ms | Speedup = 1.5717x