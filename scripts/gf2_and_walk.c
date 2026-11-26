// Status: deprecated (retired 27 Nov 2026 on the owner's word; kept to replicate its record). Development, not a result: a flip walk over AND gates rather than bilinear products, driven by
// scripts/search_gf2_and_walk.py; nothing it prints counts until the driver has checked it.
//
// Inputs z = (x, y), 2n bits (x in bits 0..n-1, y in bits n..2n-1). A gate (a, b, c) is the AND (a.z)(b.z), added
// into the outputs of c. Its quadratic part is the 2-form a ^ b, which depends only on the plane spanned by a and b;
// its linear part (the diagonal a_i b_i z_i) is corrected by XORs, which cost nothing. Gates (a, b, c) compute
// GF(2^n) multiplication with XORs alone besides when sum_t c_t (x) (a_t ^ b_t) equals the 2-forms of the product,
// sum_ij T_ijk x_i ^ y_j for output k (Mirwald and Schnorr's view of multiplicative complexity; the bilinear products
// are the gates with a in x and b in y). Moves:
//   shared vector v (in both planes): v^w (x) c + v^w' (x) c' = v^(w+w') (x) c + v^w' (x) (c+c')
//   shared output c:                  v^w (x) c + v'^w' (x) c = (v+v')^w (x) c + v'^(w+w') (x) c, for any bases
//   reductions: equal planes merge their outputs; equal outputs with planes meeting in v merge into v^(w+w');
//   a zero 2-form or output goes; plus: v^w (x) c + v'^w' (x) c' = v^w (x) (c+c') + (v+v')^w (x) c' + v'^(w+w') (x) c'.
// Input: "n r0" then r0 lines "a b c". Output: "best flips pluses resets seconds" then the gates. Args as gf2_flip_walk.c.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXT 512
static int n, len;
static uint32_t A[MAXT], B[MAXT], C[MAXT];
static uint64_t rs;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

// canonical basis of the plane of (a, b): the two smallest of {a, b, a^b}; zero when a, b are dependent
static inline void canon(uint32_t a, uint32_t b, uint32_t *p, uint32_t *q) {
    uint32_t c = a ^ b;
    if (!a || !b || !c) { *p = *q = 0; return; }
    uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
    if (c < lo) { *p = c; *q = lo; } else if (c < hi) { *p = lo; *q = c; } else { *p = lo; *q = hi; }
}
static inline int in_plane(uint32_t v, int t) { return v && (v == A[t] || v == B[t] || v == (A[t] ^ B[t])); }
static inline int same_plane(int t, int s) {
    uint32_t p1, q1, p2, q2; canon(A[t], B[t], &p1, &q1); canon(A[s], B[s], &p2, &q2); return p1 == p2 && q1 == q2;
}
// another vector w of t's plane besides v (v must be in it)
static inline uint32_t other(int t, uint32_t v) { return v == A[t] ? B[t] : A[t]; }
static void remove_term(int t) { len--; if (t != len) { A[t] = A[len]; B[t] = B[len]; C[t] = C[len]; } }

static int pend[4096]; static int npend;
static inline void push(int t) { if (npend < 4096) pend[npend++] = t; }
static void fix_pending(int gone, int moved_from) {
    for (int i = 0; i < npend; i++) { if (pend[i] == gone) pend[i] = -1; else if (pend[i] == moved_from) pend[i] = gone; }
}

static int settle(void) {
    int removed = 0;
    while (npend) {
        int t = pend[--npend];
        if (t < 0 || t >= len) continue;
        uint32_t p, q; canon(A[t], B[t], &p, &q);
        if (!p || !C[t]) { int last = len - 1; remove_term(t); fix_pending(t, last); removed++; continue; }
        int done = 0;
        for (int s = 0; s < len && !done; s++) {
            if (s == t) continue;
            if (same_plane(t, s)) {                                   // equal planes: outputs add
                C[t] ^= C[s];
                int last = len - 1; remove_term(s); fix_pending(s, last); if (t == last) t = s;
                push(t); removed++; done = 1;
            } else if (C[s] == C[t]) {                                // equal outputs and planes meeting in v
                uint32_t vs[3] = {A[t], B[t], A[t] ^ B[t]}; uint32_t v = 0;
                for (int i = 0; i < 3; i++) if (in_plane(vs[i], s)) { v = vs[i]; break; }
                if (!v) continue;
                uint32_t w = other(t, v), w2 = other(s, v);
                A[t] = v; B[t] = w ^ w2;
                int last = len - 1; remove_term(s); fix_pending(s, last); if (t == last) t = s;
                push(t); removed++; done = 1;
            }
        }
    }
    return removed;
}

// a random flip; -1 when no gate shares a vector or an output with another
static int flip(void) {
    for (int tries = 0; tries < 2 * len; tries++) {
        int t = rnd() % len;
        uint32_t vs[3] = {A[t], B[t], A[t] ^ B[t]};
        int cand[MAXT], kind[MAXT]; uint32_t sv[MAXT]; int nc = 0;
        for (int s = 0; s < len; s++) {
            if (s == t) continue;
            for (int i = 0; i < 3; i++) if (in_plane(vs[i], s)) { cand[nc] = s; kind[nc] = 0; sv[nc] = vs[i]; nc++; }
            if (C[s] == C[t]) { cand[nc] = s; kind[nc] = 1; sv[nc] = 0; nc++; }
        }
        if (!nc) continue;
        int k = rnd() % nc, s = cand[k];
        if (kind[k] == 0) {
            uint32_t v = sv[k], w = other(t, v), w2 = other(s, v);
            if (rnd() & 1) { int x = t; t = s; s = x; uint32_t y = w; w = w2; w2 = y; }
            // v^w (x) C[t] + v^w2 (x) C[s] -> v^(w+w2) (x) C[t] + v^w2 (x) (C[s]+C[t])
            A[t] = v; B[t] = w ^ w2; A[s] = v; B[s] = w2; C[s] ^= C[t];
        } else {
            uint32_t pt[3] = {A[t], B[t], A[t] ^ B[t]}, ps[3] = {A[s], B[s], A[s] ^ B[s]};
            int i = rnd() % 3, j = rnd() % 3;
            uint32_t v = pt[i], w = pt[(i + 1 + rnd() % 2) % 3], v2 = ps[j], w2 = ps[(j + 1 + rnd() % 2) % 3];
            // v^w + v2^w2 = (v+v2)^w + v2^(w+w2)
            A[t] = v ^ v2; B[t] = w; A[s] = v2; B[s] = w ^ w2;
        }
        npend = 0; push(t); push(s);
        return settle();
    }
    return -1;
}

static int plus(void) {
    if (len + 1 >= MAXT) return 0;
    for (int tries = 0; tries < 64; tries++) {
        int t = rnd() % len, s = rnd() % len;
        if (t == s || C[t] == C[s]) continue;
        uint32_t pt[3] = {A[t], B[t], A[t] ^ B[t]}, ps[3] = {A[s], B[s], A[s] ^ B[s]};
        int i = rnd() % 3, j = rnd() % 3;
        uint32_t v = pt[i], w = pt[(i + 1) % 3], v2 = ps[j], w2 = ps[(j + 1) % 3];
        if (in_plane(v2, t) || in_plane(v, s)) continue;
        // v^w (x) c + v2^w2 (x) c2 = v^w (x) (c+c2) + (v+v2)^w (x) c2 + v2^(w+w2) (x) c2: no two of these merge at once
        uint32_t c = C[t], c2 = C[s];
        C[t] = c ^ c2;
        int k = len++; A[k] = v ^ v2; B[k] = w; C[k] = c2;
        A[s] = v2; B[s] = w ^ w2;
        npend = 0; push(t); push(s); push(k); settle();
        return 1;
    }
    return 0;
}

#define POOL 16
static uint32_t pool[POOL][MAXT][3]; static int poollen[POOL], npool;
static void save(int k) { poollen[k] = len; for (int t = 0; t < len; t++) { pool[k][t][0] = A[t]; pool[k][t][1] = B[t]; pool[k][t][2] = C[t]; } }
static void restore(int k) { len = poollen[k]; for (int t = 0; t < len; t++) { A[t] = pool[k][t][0]; B[t] = pool[k][t][1]; C[t] = pool[k][t][2]; } }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: seconds seed [plateau slack reset target]\n"); return 1; }
    double seconds = atof(argv[1]); rs = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)atoll(argv[2]) * 0xD1B54A32D192ED03ULL); if (!rs) rs = 1;
    long plateau = argc > 3 ? atol(argv[3]) : 100; int slack = argc > 4 ? atoi(argv[4]) : 2;
    long reset = argc > 5 ? atol(argv[5]) : 1000000; int target = argc > 6 ? atoi(argv[6]) : 0;
    int r0; if (scanf("%d %d", &n, &r0) != 2 || r0 > MAXT) return 2;
    for (int t = 0; t < r0; t++) { unsigned a, b, c; if (scanf("%u %u %u", &a, &b, &c) != 3) return 2; A[t] = a; B[t] = b; C[t] = c; }
    len = r0;
    npend = 0; for (int t = 0; t < len; t++) push(t); settle();
    int best = len; npool = 1; save(0);
    fprintf(stderr, "start %d\n", best);
    clock_t c0 = clock(); long long flips = 0, pluses = 0, resets = 0, it = 0; long since_change = 0, since_best = 0;
    while (1) {
        if ((++it & 0x3FF) == 0 && (double)(clock() - c0) / CLOCKS_PER_SEC > seconds) break;
        if (target && best <= target) break;
        int gone = flip();
        if (gone < 0) {
            if (len < best + slack && plus()) pluses++; else { restore(rnd() % npool); resets++; }
            since_change = 0; continue;
        }
        flips++; since_change++; since_best++;
        if (gone) since_change = 0;
        if (len < best) {
            best = len; npool = 1; save(0); since_best = 0;
            fprintf(stderr, "best %d at %.1fs flips %lld\n", best, (double)(clock() - c0) / CLOCKS_PER_SEC, flips);
        } else if (len == best && gone && npool < POOL) { save(npool); npool++; }
        if (since_best >= reset) { restore(rnd() % npool); resets++; since_best = since_change = 0; }
        else if (since_change >= plateau && len < best + slack) { if (plus()) pluses++; since_change = 0; }
    }
    restore(0);
    printf("%d %lld %lld %lld %.1f\n", best, flips, pluses, resets, (double)(clock() - c0) / CLOCKS_PER_SEC);
    for (int t = 0; t < len; t++) printf("%u %u %u\n", A[t], B[t], C[t]);
    return 0;
}
