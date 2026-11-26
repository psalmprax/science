// Status: audit-only. Development, not a result: the flip walk over GF(4), driven by scripts/search_gf4_mul_flip.py.
// The moves of scripts/gf2_flip_walk.c with GF(4) scalars. Nothing it prints counts until the driver has checked it.
// Flip-graph walk for bilinear decompositions over GF(4) (vectors of length n <= 8, 2 bits per coordinate).
// Element encoding: b0 + 2*b1 means b0 + b1*w, w^2 = w + 1. Addition is XOR.
// Term = A (x) B (x) C, vectors unnormalised; sharing is judged up to a scalar (normalised keys).
// Moves: flips with a random scalar, merges of terms sharing two factors, dependent groups, zero factors,
// plus transitions, resets to a pool. Input: "n r0" then r0 lines "A B C" (16-bit ints). Args as flip.c.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXT 256
#define MAXV 65536
#define MAXB 96

static int n, len;
static uint16_t V[3][MAXT];          // actual vectors
static uint16_t K[3][MAXT];          // normalised keys
static uint8_t cnt[3][MAXV];
static uint8_t bucket[3][MAXV][MAXB];
static uint8_t pos[3][MAXT];
static uint16_t shared[3][MAXV]; static int nshared[3]; static int32_t where[3][MAXV];
static uint16_t SC[4][MAXV];         // SC[s][v] = s * v
static uint8_t MUL[4][4] = {{0,0,0,0},{0,1,2,3},{0,2,3,1},{0,3,1,2}};
static uint8_t INV[4] = {0,1,3,2};
static uint64_t rs;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static inline uint16_t lead(uint16_t v) { for (int i = 0; i < n; i++) { int e = (v >> (2 * i)) & 3; if (e) return e; } return 0; }
static inline uint16_t norm(uint16_t v) { int l = lead(v); return l ? SC[INV[l]][v] : 0; }

static void badd(int m, int t) {
    uint16_t k = K[m][t];
    if (cnt[m][k] >= MAXB) { fprintf(stderr, "bucket overflow\n"); exit(3); }
    pos[m][t] = cnt[m][k]; bucket[m][k][cnt[m][k]++] = (uint8_t)t;
    if (cnt[m][k] == 2) { where[m][k] = nshared[m]; shared[m][nshared[m]++] = k; }
}
static void bdel(int m, int t) {
    uint16_t k = K[m][t]; int p = pos[m][t]; int last = --cnt[m][k];
    uint8_t mv = bucket[m][k][last]; bucket[m][k][p] = mv; pos[m][mv] = (uint8_t)p;
    if (last == 1) { int w = where[m][k]; uint16_t mvk = shared[m][--nshared[m]]; shared[m][w] = mvk; where[m][mvk] = w; }
}
static void setv(int m, int t, uint16_t v) { bdel(m, t); V[m][t] = v; K[m][t] = norm(v); badd(m, t); }
static int append_term(uint16_t a, uint16_t b, uint16_t c) {
    int t = len++; V[0][t] = a; V[1][t] = b; V[2][t] = c;
    for (int m = 0; m < 3; m++) { K[m][t] = norm(V[m][t]); badd(m, t); }
    return t;
}
static int remove_term(int t) {
    for (int m = 0; m < 3; m++) bdel(m, t);
    int last = len - 1;
    if (t != last) for (int m = 0; m < 3; m++) { bdel(m, last); V[m][t] = V[m][last]; K[m][t] = K[m][last]; badd(m, t); }
    len--; return t != last ? last : -1;
}
// scalar s such that V[m][t] = s * V[m][u] (same key)
static inline int ratio(int m, int t, int u) { return MUL[lead(V[m][t])][INV[lead(V[m][u])]]; }

static int pend[2048]; static int npend;
static inline void push(int t) { if (npend < 2048) pend[npend++] = t; }

// dependent subset: members j of bucket (mode m, key k), term j = X (x) Yhat_j (x) Z_j with X = V[m][ref],
// Yhat_j = ratio_j * V[p][j]. Finds coefficients co_j (nonzero) with sum co_j Yhat_j = 0. Returns count.
static int dependent(int m, uint16_t k, int p, int *idx, int *coef) {
    int c = cnt[m][k];
    int ref = bucket[m][k][0];
    uint16_t bv[8]; uint8_t bc[8][MAXB]; int have[8] = {0};
    for (int i = 0; i < c; i++) {
        int t = bucket[m][k][i];
        uint16_t y = SC[ratio(m, t, ref)][V[p][t]];
        uint8_t co[MAXB]; memset(co, 0, c); co[i] = 1;
        while (y) {
            int top = -1, e = 0;
            for (int q = n - 1; q >= 0; q--) { e = (y >> (2 * q)) & 3; if (e) { top = q; break; } }
            if (!have[top]) {
                int inv = INV[e];
                bv[top] = SC[inv][y]; for (int j = 0; j < c; j++) bc[top][j] = MUL[inv][co[j]];
                have[top] = 1; break;
            }
            y ^= SC[e][bv[top]]; for (int j = 0; j < c; j++) co[j] ^= MUL[e][bc[top][j]];
        }
        if (!y) {
            int ns = 0;
            for (int j = 0; j < c; j++) if (co[j]) { idx[ns] = bucket[m][k][j]; coef[ns] = co[j]; ns++; }
            return ns;
        }
    }
    return 0;
}

static int settle(void) {
    int gone = 0;
    while (npend) {
        int t = pend[--npend];
        if (t < 0 || t >= len) continue;
        int drop = -1; int changed[MAXB]; int nch = 0;
        if (!V[0][t] || !V[1][t] || !V[2][t]) drop = t;
        else {
            int found = -1, m3 = -1;
            for (int mm = 0; mm < 3 && found < 0; mm++) {
                int m1 = (mm + 1) % 3, m2 = (mm + 2) % 3;
                uint16_t k1 = K[m1][t], k2 = K[m2][t];
                if (cnt[m1][k1] < 2 || cnt[m2][k2] < 2) continue;
                for (int i = 0; i < cnt[m1][k1]; i++) { int s = bucket[m1][k1][i]; if (s != t && K[m2][s] == k2) { found = s; m3 = mm; break; } }
            }
            if (found >= 0) {
                int m1 = (m3 + 1) % 3, m2 = (m3 + 2) % 3;
                // found = r1 * r2 * (V[m1][t]) (x) (V[m2][t]) (x) V[m3][found]/... : scalar of found relative to t
                int s = MUL[ratio(m1, found, t)][ratio(m2, found, t)];
                setv(m3, t, V[m3][t] ^ SC[s][V[m3][found]]);
                drop = found; changed[nch++] = t;
            } else {
                int idx[MAXB], coef[MAXB]; int ns = 0, q = -1;
                for (int m = 0; m < 3 && !ns; m++) {
                    if (cnt[m][K[m][t]] < 3) continue;
                    for (int p = 0; p < 3 && !ns; p++) if (p != m) { ns = dependent(m, K[m][t], p, idx, coef); if (ns) q = 3 - m - p; }
                }
                if (!ns) continue;
                // sum co_j Yhat_j = 0: Yhat_r = co_r^-1 sum_{j != r} co_j Yhat_j, so term r folds into the others:
                // Z_j += (co_r^-1 co_j) Z_r, then term r goes.
                int r = idx[0]; int cr = coef[0];
                for (int j = 1; j < ns; j++) {
                    int u = idx[j]; int g = MUL[INV[cr]][coef[j]];
                    setv(q, u, V[q][u] ^ SC[g][V[q][r]]);
                    changed[nch++] = u;
                }
                drop = r;
            }
        }
        for (int i = 0; i < npend; i++) if (pend[i] == drop) pend[i] = -1;
        int moved = remove_term(drop); gone++;
        if (moved >= 0) { for (int i = 0; i < npend; i++) if (pend[i] == moved) pend[i] = drop; for (int i = 0; i < nch; i++) if (changed[i] == moved) changed[i] = drop; }
        for (int i = 0; i < nch; i++) push(changed[i]);
    }
    return gone;
}

static int flip(void) {
    int total = nshared[0] + nshared[1] + nshared[2];
    if (!total) return -1;
    int k = rnd() % total, m = 0;
    while (k >= nshared[m]) { k -= nshared[m]; m++; }
    uint16_t key = shared[m][k]; int c = cnt[m][key];
    int i = rnd() % c, j = rnd() % (c - 1); if (j >= i) j++;
    int t = bucket[m][key][i], s = bucket[m][key][j];
    int p = (m + 1) % 3, q = (m + 2) % 3;
    if (rnd() & 1) { int tmp = p; p = q; q = tmp; }
    // make t's mode-m vector equal to s's: V[m][t] = a * V[m][s]  => move a into t's p-vector
    int a = ratio(m, t, s);
    setv(m, t, V[m][s]); setv(p, t, SC[a][V[p][t]]);
    int lam = 1 + rnd() % 3;
    // t: x (x) (Yt + lam*Ys) (x) Zt ; s: x (x) Ys (x) (Zs + lam*Zt)
    setv(p, t, V[p][t] ^ SC[lam][V[p][s]]);
    setv(q, s, V[q][s] ^ SC[lam][V[q][t]]);
    npend = 0; push(t); push(s);
    return settle();
}

static int plus(void) {
    for (int tries = 0; tries < 64; tries++) {
        int i = rnd() % len, j = rnd() % len;
        if (i == j) continue;
        if (K[0][i] == K[0][j] || K[1][i] == K[1][j] || K[2][i] == K[2][j]) continue;
        int o[3] = {0, 1, 2};
        for (int k = 2; k > 0; k--) { int r = rnd() % (k + 1); int tmp = o[k]; o[k] = o[r]; o[r] = tmp; }
        uint16_t fi[3] = {V[0][i], V[1][i], V[2][i]}, fj[3] = {V[0][j], V[1][j], V[2][j]};
        // a(x)b(x)c + a'(x)b'(x)c' = (a+a')(x)b(x)c + a'(x)(b+b')(x)c ... in char 2 with signs irrelevant:
        // (a - a') b c + a' (b - b') c + a' b' (c' - c)... char 2 -> same as GF(2) identity
        uint16_t first[3], second[3], third[3];
        first[o[0]] = fi[o[0]] ^ fj[o[0]]; first[o[1]] = fi[o[1]]; first[o[2]] = fi[o[2]];
        second[o[0]] = fj[o[0]]; second[o[1]] = fi[o[1]] ^ fj[o[1]]; second[o[2]] = fi[o[2]];
        third[o[0]] = fj[o[0]]; third[o[1]] = fj[o[1]]; third[o[2]] = fi[o[2]] ^ fj[o[2]];
        for (int m = 0; m < 3; m++) { setv(m, i, first[m]); setv(m, j, second[m]); }
        int k = append_term(third[0], third[1], third[2]);
        npend = 0; push(i); push(j); push(k); settle();
        return 1;
    }
    return 0;
}

static void load(uint16_t (*src)[3], int count) {
    while (len) remove_term(len - 1);
    for (int t = 0; t < count; t++) append_term(src[t][0], src[t][1], src[t][2]);
    for (int rep = 0; rep < 4; rep++) { npend = 0; for (int t = 0; t < len; t++) push(t); settle(); }
}

#define POOL 16
static uint16_t pool[POOL][MAXT][3]; static int poollen[POOL]; static int npool;
static uint16_t start[MAXT][3];

int main(int argc, char **argv) {
    double seconds = atof(argv[1]); rs = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)atoll(argv[2]) * 0xD1B54A32D192ED03ULL); if (!rs) rs = 1;
    long plateau = argc > 3 ? atol(argv[3]) : 100; int slack = argc > 4 ? atoi(argv[4]) : 2;
    long reset = argc > 5 ? atol(argv[5]) : 1000000; int target = argc > 6 ? atoi(argv[6]) : 0;
    for (int s = 0; s < 4; s++) for (int v = 0; v < MAXV; v++) { uint16_t out = 0; for (int i = 0; i < 8; i++) out |= (uint16_t)MUL[s][(v >> (2 * i)) & 3] << (2 * i); SC[s][v] = out; }
    int r0; if (scanf("%d %d", &n, &r0) != 2) return 2;
    for (int t = 0; t < r0; t++) { unsigned a, b, c; if (scanf("%u %u %u", &a, &b, &c) != 3) return 2; start[t][0] = a; start[t][1] = b; start[t][2] = c; }
    for (int i = 0; i < 20; i++) rnd();
    load(start, r0);
    int best = len; npool = 1; poollen[0] = len;
    for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[0][t][m] = V[m][t];
    fprintf(stderr, "start %d\n", len);
    clock_t c0 = clock(); long long flips = 0, pluses = 0, resets = 0, it = 0; long since_change = 0, since_best = 0;
    while (1) {
        if ((++it & 0x3FF) == 0 && (double)(clock() - c0) / CLOCKS_PER_SEC > seconds) break;
        if (target && best <= target) break;
        int gone = flip();
        if (gone < 0) {
            if (len < best + slack && plus()) pluses++;
            else { int k = rnd() % npool; uint16_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint16_t) * 3 * poollen[k]); load(tmp, poollen[k]); resets++; }
            since_change = 0; continue;
        }
        flips++; since_change++; since_best++;
        if (gone) since_change = 0;
        if (len < best) {
            best = len; npool = 1; poollen[0] = len;
            for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[0][t][m] = V[m][t];
            since_best = 0;
            fprintf(stderr, "best %d at %.1fs flips %lld\n", best, (double)(clock() - c0) / CLOCKS_PER_SEC, flips);
        } else if (len == best && gone && npool < POOL) {
            poollen[npool] = len; for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[npool][t][m] = V[m][t]; npool++;
        }
        if (since_best >= reset) { int k = rnd() % npool; uint16_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint16_t) * 3 * poollen[k]); load(tmp, poollen[k]); resets++; since_best = since_change = 0; }
        else if (since_change >= plateau && len < best + slack) { if (plus()) pluses++; since_change = 0; }
    }
    printf("%d %lld %lld %lld %.1f\n", best, flips, pluses, resets, (double)(clock() - c0) / CLOCKS_PER_SEC);
    for (int t = 0; t < poollen[0]; t++) printf("%u %u %u\n", pool[0][t][0], pool[0][t][1], pool[0][t][2]);
    return 0;
}
