========= CORRECTNESS =========
Max diff (V1 vs V2):    2.026558e-06
Max diff (V1 vs FA_1):  2.503395e-06
Max diff (V1 vs FA_2):  2.503395e-06

========= PERFORMANCE (ms) =========
Naive Pipeline (V1)      : 19.5180 ms
Optimized Pipeline (V2)  : 5.0852 ms
Flash Attention v1       : 4.0465 ms
Flash Attn (Improved)    : 2.2733 ms

Speedup (FA Imp vs V1): 8.5857x
Speedup (FA Imp vs V2): 2.2369x
Speedup (FA Imp vs FA1): 1.7800x