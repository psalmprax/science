# SPDX-License-Identifier: Apache-2.0
# Kozykorpesh public V3/run361677445, https://www.kaggle.com/code/kozykappa/quantum-ai-lb-score-6-68108?scriptVersionId=361677445
# Source-only adaptation: explicit coefficients supplied by caller; no Toolbox or binary input.
# Original notices and credited Sumit/Rudra/Alex/Kirill/Artem lineage retained below.
# Optimized shift adapted from Dmitrii Gluzdov V6/run361592338; original arithmetic credits retained.
import numpy as np

def optimized_shift(b,flags,word,L,inverse,prune=True):
 stages=[(1,1),(2,2),(3,4),(4,8)];occ=[];n=L
 for k,d in stages:
  occ.append(min(n,L+2) if prune and k==4 else n);n=min(n+d,len(word))
 seq=list(enumerate(stages));seq=seq[::-1] if inverse else seq
 for j,(k,d) in seq:
  indices=range(occ[j]) if inverse else reversed(range(occ[j]))
  for i in indices:
   if i+d<len(word):
    zero_boundary = i+d >= L+sum(amount for _,amount in stages[:j])
    if zero_boundary:
     if inverse:b.cx(word[i+d],word[i]);b.del_and(flags[k],word[i],word[i+d])
     else:b.init_and(flags[k],word[i],word[i+d]);b.cx(word[i+d],word[i])
    else:b.cswap(flags[k],word[i],word[i+d])


def mobius_transform(table):
    """Packed-output Boolean truth table <-> ANF coefficient table over GF(2)."""
    result = list(map(int, table))
    if not result or len(result) & (len(result) - 1):
        raise ValueError("truth-table length must be a power of two")
    n = (len(result) - 1).bit_length()
    for bit in range(n):
        step = 1 << bit
        for start in range(0, len(result), 2 * step):
            for offset in range(step):
                result[start + step + offset] ^= result[start + offset]
    return result
