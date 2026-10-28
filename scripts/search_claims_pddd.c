/*
 * search_claims_pddd.c — exhaustive search for the shared multi-output PDDD parameters
 * used by the 89Q invsqrt champion (11-node coarse6 + 28-node curve4..6).
 *
 * Build and run (needs GCC and OpenMP):
 *     gcc -O2 -fopenmp -o /tmp/search_claims_pddd scripts/search_claims_pddd.c
 *     /tmp/search_claims_pddd
 *
 * Runtime on the development machine: ~30 s for Test 1, ~26 s for Test 2.
 *
 * What a "shared node" is here: an intermediate AND node of the Davio (recursive
 * xor) decomposition that more than one of the evaluated outputs needs. The search
 * enumerates all 8! variable orders x 256 polarities and reports the permutation
 * with the fewest such nodes. Both tests are pure searches over hardcoded truth
 * tables; the tables are the curve/coarse bits of the invsqrt residual split and
 * are not derived in this file.
 *
 * Test 2's result (36 nodes, order [7,4,3,2,1,0,6,5], polarity 32) is the input that
 * scripts/test_viable_optimization_path.py consumes and that
 * scripts/test_shared_pddd_circuit.py's build_shared_plan reproduces as 36 nodes.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

typedef struct {
    uint8_t level;
    uint64_t w0;
    uint64_t w1;
} NodeKey;

typedef struct {
    NodeKey keys[128];
    int count;
} NodeSet;

static inline void add_node(NodeSet *set, uint8_t level, uint64_t w0, uint64_t w1) {
    for (int i = 0; i < set->count; i++) {
        if (set->keys[i].level == level && set->keys[i].w0 == w0 && set->keys[i].w1 == w1) {
            return;
        }
    }
    if (set->count < 128) {
        set->keys[set->count].level = level;
        set->keys[set->count].w0 = w0;
        set->keys[set->count].w1 = w1;
        set->count++;
    }
}

static void rec_lvl2(uint32_t t, NodeSet *set) {
    if (t == 0 || t == 0xF) return;
    uint32_t lo = t & 3;
    uint32_t hi = t >> 2;
    uint32_t r = lo ^ hi;
    if (r != 0 && r != 3) {
        add_node(set, 1, r, 0);
    }
}

static void rec_lvl3(uint32_t t, NodeSet *set) {
    if (t == 0 || t == 0xFF) return;
    uint32_t lo = t & 0xF;
    uint32_t hi = t >> 4;
    uint32_t r = lo ^ hi;
    if (r != 0 && r != 0xF) {
        add_node(set, 2, r, 0);
    }
    rec_lvl2(lo, set);
    rec_lvl2(r, set);
}

static void rec_lvl4(uint32_t t, NodeSet *set) {
    if (t == 0 || t == 0xFFFF) return;
    uint32_t lo = t & 0xFF;
    uint32_t hi = t >> 8;
    uint32_t r = lo ^ hi;
    if (r != 0 && r != 0xFF) {
        add_node(set, 3, r, 0);
    }
    rec_lvl3(lo, set);
    rec_lvl3(r, set);
}

static void rec_lvl5(uint32_t t, NodeSet *set) {
    if (t == 0 || t == 0xFFFFFFFFU) return;
    uint32_t lo = t & 0xFFFF;
    uint32_t hi = t >> 16;
    uint32_t r = lo ^ hi;
    if (r != 0 && r != 0xFFFF) {
        add_node(set, 4, r, 0);
    }
    rec_lvl4(lo, set);
    rec_lvl4(r, set);
}

static void rec_lvl6(uint64_t t, NodeSet *set) {
    if (t == 0 || t == 0xFFFFFFFFFFFFFFFFULL) return;
    uint64_t lo = t & 0xFFFFFFFFULL;
    uint64_t hi = t >> 32;
    uint64_t r = lo ^ hi;
    if (r != 0 && r != 0xFFFFFFFFULL) {
        add_node(set, 5, r, 0);
    }
    rec_lvl5((uint32_t)lo, set);
    rec_lvl5((uint32_t)r, set);
}

static void rec_lvl7(uint64_t t0, uint64_t t1, NodeSet *set) {
    if ((t0 == 0 && t1 == 0) || (t0 == ~0ULL && t1 == ~0ULL)) return;
    uint64_t lo = t0;
    uint64_t hi = t1;
    uint64_t r = lo ^ hi;
    if (r != 0 && r != ~0ULL) {
        add_node(set, 6, r, 0);
    }
    rec_lvl6(lo, set);
    rec_lvl6(r, set);
}

static void add_truth_nodes(const uint64_t t[4], NodeSet *set) {
    if ((t[0] == 0 && t[1] == 0 && t[2] == 0 && t[3] == 0) ||
        (t[0] == ~0ULL && t[1] == ~0ULL && t[2] == ~0ULL && t[3] == ~0ULL)) {
        return;
    }
    uint64_t lo0 = t[0], lo1 = t[1];
    uint64_t hi0 = t[2], hi1 = t[3];
    uint64_t r0 = lo0 ^ hi0;
    uint64_t r1 = lo1 ^ hi1;
    if ((r0 != 0 || r1 != 0) && (r0 != ~0ULL || r1 != ~0ULL)) {
        add_node(set, 7, r0, r1);
    }
    rec_lvl7(lo0, lo1, set);
    rec_lvl7(r0, r1, set);
}

static void generate_permutations(uint8_t perms[40320][8]) {
    int p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int count = 0;
    int c[8] = {0};
    
    for (int j = 0; j < 8; j++) perms[count][j] = p[j];
    count++;

    int i = 0;
    while (i < 8) {
        if (c[i] < i) {
            if (i % 2 == 0) {
                int tmp = p[0]; p[0] = p[i]; p[i] = tmp;
            } else {
                int tmp = p[c[i]]; p[c[i]] = p[i]; p[i] = tmp;
            }
            for (int j = 0; j < 8; j++) perms[count][j] = p[j];
            count++;
            c[i]++;
            i = 0;
        } else {
            c[i] = 0;
            i++;
        }
    }
}

int main(int argc, char **argv) {
    uint64_t t3[4] = { 0x000000007ffff800ULL, 0xfffffffffffffff8ULL, 0x003fffff8000ffe0ULL, 0xfffffffe00000000ULL };
    uint64_t t4[4] = { 0xfffffffffff00000ULL, 0x0000007fffffffffULL, 0xfffffe00000003ffULL, 0xffffffffffffffffULL };
    uint64_t t5[4] = { 0x00003fffffffffffULL, 0x0000000000000000ULL, 0xffffffffffffffffULL, 0x00000000000000ffULL };
    uint64_t t6[4] = { 0x0000000000000007ULL, 0x0000000000000000ULL, 0x00000000007fffffULL, 0x0000000000000000ULL };
    uint64_t t_coarse6[4] = { 0x00000001ffffffffULL, 0x0000000000000000ULL, 0xffffffffffffffffULL, 0x00000000000007ffULL };

    static uint8_t perms[40320][8];
    generate_permutations(perms);

    printf("=================================================================\n");
    printf("TEST 1: Exhaustive search for 4-output {curve3, curve4, curve5, curve6}\n");
    printf("=================================================================\n");
    double t0 = omp_get_wtime();
    int min_c3_to_c6 = 999;
    uint8_t best_order_c3[8];
    uint8_t best_pol_c3 = 0;

    #pragma omp parallel
    {
        int local_min = 999;
        uint8_t local_best_order[8];
        uint8_t local_best_pol = 0;

        #pragma omp for schedule(dynamic, 64)
        for (int p_idx = 0; p_idx < 40320; p_idx++) {
            const uint8_t *order = perms[p_idx];
            uint8_t basis[256];
            for (int i = 0; i < 256; i++) {
                int orig = 0;
                for (int j = 0; j < 8; j++) {
                    if ((i >> j) & 1) orig |= (1 << order[j]);
                }
                basis[i] = (uint8_t)orig;
            }

            for (int pol = 0; pol < 256; pol++) {
                uint64_t tr3[4] = {0}, tr4[4] = {0}, tr5[4] = {0}, tr6[4] = {0};
                for (int i = 0; i < 256; i++) {
                    int orig = basis[i] ^ pol;
                    int word = orig >> 6;
                    int bit_shift = orig & 63;
                    tr3[i >> 6] |= (((t3[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr4[i >> 6] |= (((t4[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr5[i >> 6] |= (((t5[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr6[i >> 6] |= (((t6[word] >> bit_shift) & 1ULL) << (i & 63));
                }

                NodeSet set;
                set.count = 0;
                add_truth_nodes(tr3, &set);
                add_truth_nodes(tr4, &set);
                add_truth_nodes(tr5, &set);
                add_truth_nodes(tr6, &set);

                if (set.count < local_min) {
                    local_min = set.count;
                    memcpy(local_best_order, order, 8);
                    local_best_pol = (uint8_t)pol;
                }
            }
        }

        #pragma omp critical
        {
            if (local_min < min_c3_to_c6) {
                min_c3_to_c6 = local_min;
                memcpy(best_order_c3, local_best_order, 8);
                best_pol_c3 = local_best_pol;
            }
        }
    }
    double t1 = omp_get_wtime();
    printf("Result Test 1 ({c3, c4, c5, c6}): Minimum Shared Nodes = %d in %.2f s\n", min_c3_to_c6, t1 - t0);
    printf("  Best order: [%d, %d, %d, %d, %d, %d, %d, %d], polarity: %d\n\n",
           best_order_c3[0], best_order_c3[1], best_order_c3[2], best_order_c3[3],
           best_order_c3[4], best_order_c3[5], best_order_c3[6], best_order_c3[7], best_pol_c3);

    printf("=================================================================\n");
    printf("TEST 2: Exhaustive search for 4-output {coarse6, curve4, curve5, curve6}\n");
    printf("=================================================================\n");
    double t2 = omp_get_wtime();
    int min_coarse_and_curve = 999;
    uint8_t best_order_coarse[8];
    uint8_t best_pol_coarse = 0;

    #pragma omp parallel
    {
        int local_min = 999;
        uint8_t local_best_order[8];
        uint8_t local_best_pol = 0;

        #pragma omp for schedule(dynamic, 64)
        for (int p_idx = 0; p_idx < 40320; p_idx++) {
            const uint8_t *order = perms[p_idx];
            uint8_t basis[256];
            for (int i = 0; i < 256; i++) {
                int orig = 0;
                for (int j = 0; j < 8; j++) {
                    if ((i >> j) & 1) orig |= (1 << order[j]);
                }
                basis[i] = (uint8_t)orig;
            }

            for (int pol = 0; pol < 256; pol++) {
                uint64_t tr_c6[4] = {0}, tr4[4] = {0}, tr5[4] = {0}, tr6[4] = {0};
                for (int i = 0; i < 256; i++) {
                    int orig = basis[i] ^ pol;
                    int word = orig >> 6;
                    int bit_shift = orig & 63;
                    tr_c6[i >> 6] |= (((t_coarse6[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr4[i >> 6] |= (((t4[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr5[i >> 6] |= (((t5[word] >> bit_shift) & 1ULL) << (i & 63));
                    tr6[i >> 6] |= (((t6[word] >> bit_shift) & 1ULL) << (i & 63));
                }

                NodeSet set;
                set.count = 0;
                add_truth_nodes(tr_c6, &set);
                add_truth_nodes(tr4, &set);
                add_truth_nodes(tr5, &set);
                add_truth_nodes(tr6, &set);

                if (set.count < local_min) {
                    local_min = set.count;
                    memcpy(local_best_order, order, 8);
                    local_best_pol = (uint8_t)pol;
                }
            }
        }

        #pragma omp critical
        {
            if (local_min < min_coarse_and_curve) {
                min_coarse_and_curve = local_min;
                memcpy(best_order_coarse, local_best_order, 8);
                best_pol_coarse = local_best_pol;
            }
        }
    }
    double time_end = omp_get_wtime();
    printf("Result Test 2 ({coarse6, c4, c5, c6}): Minimum Shared Nodes = %d in %.2f s\n", min_coarse_and_curve, time_end - t2);
    printf("  Best order: [%d, %d, %d, %d, %d, %d, %d, %d], polarity: %d\n",
           best_order_coarse[0], best_order_coarse[1], best_order_coarse[2], best_order_coarse[3],
           best_order_coarse[4], best_order_coarse[5], best_order_coarse[6], best_order_coarse[7], best_pol_coarse);
    printf("=================================================================\n");

    return 0;
}
