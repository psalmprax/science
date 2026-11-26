// Exhaustive search over symmetric decompositions of GF(2^12) multiplication that are invariant under a power of
// the Frobenius, by orbits of forms (development, audit-only; see scripts/gf2_invariant_search.py).
// Each orbit brings a few vectors (its outputs' degrees of freedom, as bilinear maps on the pairs (u^i, u^j));
// a choice of orbits is a decomposition exactly when the multiplication (the target) lies in their span.
// Usage: gf2_invariant_search FILE [--shard K N] SIZE SIZE ...   (the orbit sizes of the profile)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXW 16
#define MAXO 4096
#define MAXV 12
#define MAXBASIS 160

static int n, norb, W, npairs;
static uint64_t target[MAXW];
static int lam[MAXO], dsz[MAXO], nv[MAXO];
static uint64_t vecs[MAXO][MAXV][MAXW];
static uint64_t basis[MAXBASIS][MAXW]; static int nb; static int pv[MAXW * 64];
static int prof[64], nprof, chosen[64];
static long long leaves, found;
static int shard = 0, shards = 1;

static inline int top(const uint64_t *v) { for (int w = W - 1; w >= 0; w--) if (v[w]) return w * 64 + 63 - __builtin_clzll(v[w]); return -1; }
static void add(const uint64_t *src) {
    uint64_t v[MAXW]; memcpy(v, src, sizeof(uint64_t) * W);
    for (;;) {
        int b = top(v); if (b < 0) return;
        int i = pv[b];
        if (i < 0) { memcpy(basis[nb], v, sizeof(uint64_t) * W); pv[b] = nb; nb++; return; }
        for (int w = 0; w < W; w++) v[w] ^= basis[i][w];
    }
}
static void undo(int to) { while (nb > to) { nb--; pv[top(basis[nb])] = -1; } }
static int in_span(const uint64_t *src) {
    uint64_t v[MAXW]; memcpy(v, src, sizeof(uint64_t) * W);
    for (;;) { int b = top(v); if (b < 0) return 1; int i = pv[b]; if (i < 0) return 0; for (int w = 0; w < W; w++) v[w] ^= basis[i][w]; }
}
static void rec(int k, int start) {
    if (k == nprof) {
        leaves++;
        if (in_span(target)) {
            found++;
            if (found <= 20) { printf("FOUND:"); for (int i = 0; i < nprof; i++) printf(" %d", lam[chosen[i]]); printf("\n"); fflush(stdout); }
        }
        return;
    }
    int d = prof[k];
    int from = (k > 0 && prof[k - 1] == d) ? start : 0;
    for (int o = from; o < norb; o++) {
        if (dsz[o] != d) continue;
        if (k == 0 && (o % shards) != shard) continue;
        int save = nb;
        for (int e = 0; e < nv[o]; e++) add(vecs[o][e]);
        chosen[k] = o;
        rec(k + 1, o + 1);
        undo(save);
    }
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: FILE [--shard K N] SIZES...\n"); return 1; }
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    int hdr[4]; if (fread(hdr, 4, 4, f) != 4) return 2;
    n = hdr[0]; norb = hdr[1]; W = hdr[2]; npairs = hdr[3];
    if (W > MAXW || norb > MAXO) return 3;
    if (fread(target, 8, W, f) != (size_t)W) return 2;
    for (int o = 0; o < norb; o++) {
        int h[3]; if (fread(h, 4, 3, f) != 3) return 2; lam[o] = h[0]; dsz[o] = h[1]; nv[o] = h[2];
        if (nv[o] > MAXV) return 3;
        for (int e = 0; e < nv[o]; e++) if (fread(vecs[o][e], 8, W, f) != (size_t)W) return 2;
    }
    fclose(f);
    for (int i = 0; i < W * 64; i++) pv[i] = -1;
    int first = 2;
    if (argc > 4 && !strcmp(argv[2], "--shard")) { shard = atoi(argv[3]); shards = atoi(argv[4]); first = 5; }
    if (shards < 1 || shard < 0 || shard >= shards) return 4;
    nprof = argc - first; for (int i = 0; i < nprof; i++) prof[i] = atoi(argv[i + first]);
    for (int i = 0; i < nprof; i++) for (int j = i + 1; j < nprof; j++) if (prof[j] > prof[i]) { int t = prof[i]; prof[i] = prof[j]; prof[j] = t; }
    rec(0, 0);
    printf("profile"); for (int i = 0; i < nprof; i++) printf(" %d", prof[i]);
    printf(" shard %d/%d: leaves %lld found %lld\n", shard, shards, leaves, found);
    return 0;
}
