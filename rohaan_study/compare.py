import sys
import os

def load_output(filename):
    with open(filename, 'r') as f:
        content = f.read().strip()
        values = [float(x) for x in content.split()]
    return values

def main():
    naive = load_output('naive_out.txt')
    fa1 = load_output('fa1/fa1_out.txt')
    
    if len(naive) != len(fa1):
        print(f"Length mismatch: naive={len(naive)}, fa1={len(fa1)}")
        return
    
    max_diff = 0.0
    max_idx = 0
    zero_count_fa1 = 0
    for i in range(len(naive)):
        diff = abs(naive[i] - fa1[i])
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if fa1[i] == 0.0:
            zero_count_fa1 += 1
    
    print(f"Max difference: {max_diff:.8e} at index {max_idx}")
    print(f"naive[{max_idx}] = {naive[max_idx]:.8e}")
    print(f"fa1[{max_idx}] = {fa1[max_idx]:.8e}")
    print(f"Query {max_idx // 64}, Dim {max_idx % 64}")
    print(f"Zero count in fa1: {zero_count_fa1}")
    
    # Also check for inf/nan in fa1
    inf_count = sum(1 for x in fa1 if not (x == x))  # NaN check
    neg_inf_count = sum(1 for x in fa1 if x == float('-inf'))
    pos_inf_count = sum(1 for x in fa1 if x == float('inf'))
    print(f"FA1 stats: NaN={inf_count}, -Inf={neg_inf_count}, +Inf={pos_inf_count}")

if __name__ == '__main__':
    main()
