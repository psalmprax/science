// Flip-graph random walk for bilinear decompositions over GF(2) (n <= 16), the same moves as
// apedp_dyn/quantum/flip_search.py written in C for speed (about 14 million flips a second on one core
// against 150 thousand in Python). Development, audit-only: nothing it prints counts until
// scripts/search_gf2_mul_flip_c.py has checked it against every entry of the tensor.
// Moves: flips (Kauers-Moosbauer), reductions (zero factor, two shared factors, dependent group),
// plus transitions (Arai et al.), resets to a pool of shortest schemes.
// Input (stdin): first line "n r0", then r0 lines "a b c" (decimal ints, bit i = coefficient i).
// Output: "best flips pluses resets seconds", then the best scheme's terms.
// Args: seconds seed [plateau slack reset target]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXT 512
#define MAXV 65536
#define MAXB 128

static int n, len;
static uint32_t F[3][MAXT];
static uint16_t cnt[3][MAXV];
static uint16_t bucket[3][MAXV][MAXB];
static uint16_t pos[3][MAXT];   // position of term t in its bucket for mode m
static uint32_t shared[3][MAXV]; static int nshared[3]; static int32_t where[3][MAXV];   // values held by 2+ terms
static uint64_t rs;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void bucket_add(int m, int t) {
    uint32_t v = F[m][t];
    if (cnt[m][v] >= MAXB) { fprintf(stderr, "bucket overflow\n"); exit(3); }
    pos[m][t] = cnt[m][v];
    bucket[m][v][cnt[m][v]++] = (uint16_t)t;
    if (cnt[m][v] == 2) { where[m][v] = nshared[m]; shared[m][nshared[m]++] = v; }
}
static void bucket_del(int m, int t) {
    uint32_t v = F[m][t];
    int p = pos[m][t];
    int last = --cnt[m][v];
    uint16_t moved = bucket[m][v][last];
    bucket[m][v][p] = moved;
    pos[m][moved] = (uint16_t)p;
    if (last == 1) { int w = where[m][v]; uint32_t mv = shared[m][--nshared[m]]; shared[m][w] = mv; where[m][mv] = w; }
}
static void set_factor(int m, int t, uint32_t v) { bucket_del(m, t); F[m][t] = v; bucket_add(m, t); }
static int append_term(uint32_t a, uint32_t b, uint32_t c) {
    int t = len++;
    F[0][t] = a; F[1][t] = b; F[2][t] = c;
    for (int m = 0; m < 3; m++) bucket_add(m, t);
    return t;
}
// remove term t; last term moves to t. returns old index of the moved term (or -1)
static int remove_term(int t) {
    for (int m = 0; m < 3; m++) bucket_del(m, t);
    int last = len - 1;
    if (t != last) {
        for (int m = 0; m < 3; m++) { bucket_del(m, last); F[m][t] = F[m][last]; bucket_add(m, t); }
    }
    len--;
    return t != last ? last : -1;
}

// pending work list
static int pend[4096]; static int npend;
static inline void push(int t) { if (npend < 4096) pend[npend++] = t; }
static void remap(int from, int to) { for (int i = 0; i < npend; i++) if (pend[i] == from) pend[i] = to; }
static void drop_pending(int t) { for (int i = 0; i < npend; i++) if (pend[i] == t) pend[i] = -1; }

// dependent subset among terms sharing factor v in mode m, judged on mode p. returns size or 0
static int dependent(int m, uint32_t v, int p, int *out) {
    int k = cnt[m][v];
    uint32_t vec[MAXB]; uint64_t used[MAXB][2];   // up to 128 members
    int piv[32]; for (int i = 0; i < 32; i++) piv[i] = -1;
    uint32_t bv[32]; uint64_t bu[32][2];
    for (int i = 0; i < k; i++) {
        int t = bucket[m][v][i];
        uint32_t x = F[p][t]; uint64_t u0 = 0, u1 = 0;
        if (i < 64) u0 = 1ULL << i; else u1 = 1ULL << (i - 64);
        while (x) {
            int top = 31 - __builtin_clz(x);
            if (piv[top] < 0) { piv[top] = 1; bv[top] = x; bu[top][0] = u0; bu[top][1] = u1; break; }
            x ^= bv[top]; u0 ^= bu[top][0]; u1 ^= bu[top][1];
        }
        if (!x) {
            int c = 0;
            for (int j = 0; j < k; j++) if ((j < 64 ? (u0 >> j) : (u1 >> (j - 64))) & 1) out[c++] = bucket[m][v][j];
            (void)vec; (void)used;
            return c;
        }
    }
    return 0;
}

static int settle(void) {
    int gone = 0;
    while (npend) {
        int t = pend[--npend];
        if (t < 0 || t >= len) continue;
        int drop = -1; int changed[MAXB]; int nchanged = 0;
        if (!F[0][t] || !F[1][t] || !F[2][t]) { drop = t; }
        else {
            // partner sharing two factors
            int found = -1, m3 = -1;
            for (int mm = 0; mm < 3 && found < 0; mm++) {
                int m1 = (mm + 1) % 3, m2 = (mm + 2) % 3;
                uint32_t v1 = F[m1][t], v2 = F[m2][t];
                int k1 = cnt[m1][v1], k2 = cnt[m2][v2];
                if (k1 < 2 || k2 < 2) continue;
                int mm1 = k1 <= k2 ? m1 : m2; uint32_t vv = k1 <= k2 ? v1 : v2; int other = k1 <= k2 ? m2 : m1; uint32_t ov = k1 <= k2 ? v2 : v1;
                for (int i = 0; i < cnt[mm1][vv]; i++) { int s = bucket[mm1][vv][i]; if (s != t && F[other][s] == ov) { found = s; m3 = mm; break; } }
            }
            if (found >= 0) {
                set_factor(m3, t, F[m3][t] ^ F[m3][found]);
                drop = found; changed[nchanged++] = t;
            } else {
                int sub[MAXB]; int ns = 0, q = -1;
                for (int m = 0; m < 3 && !ns; m++) {
                    if (cnt[m][F[m][t]] < 3) continue;
                    for (int p = 0; p < 3 && !ns; p++) if (p != m) { ns = dependent(m, F[m][t], p, sub); if (ns) q = 3 - m - p; }
                }
                if (!ns) continue;
                drop = sub[0];
                for (int i = 1; i < ns; i++) { set_factor(q, sub[i], F[q][sub[i]] ^ F[q][drop]); changed[nchanged++] = sub[i]; }
            }
        }
        drop_pending(drop);
        int moved = remove_term(drop);   // term formerly at `moved` now at `drop`
        gone++;
        if (moved >= 0) { remap(moved, drop); for (int i = 0; i < nchanged; i++) if (changed[i] == moved) changed[i] = drop; }
        for (int i = 0; i < nchanged; i++) if (changed[i] != drop || moved >= 0) push(changed[i]);
    }
    return gone;
}

static int flip(void) {
    int total = nshared[0] + nshared[1] + nshared[2];
    if (!total) return -1;   // no two terms share a factor: no flip leaves this point
    int k = rnd() % total, m = 0;
    while (k >= nshared[m]) { k -= nshared[m]; m++; }
    uint32_t v = shared[m][k];
    int c = cnt[m][v];
    int i = rnd() % c, j = rnd() % (c - 1); if (j >= i) j++;
    int t = bucket[m][v][i], s = bucket[m][v][j];
    int p = (m + 1) % 3, q = (m + 2) % 3;
    if (rnd() & 1) { int tmp = p; p = q; q = tmp; }
    set_factor(p, t, F[p][t] ^ F[p][s]);
    set_factor(q, s, F[q][s] ^ F[q][t]);
    npend = 0; push(t); push(s);
    return settle();
}

static int plus(void) {
    for (int tries = 0; tries < 64; tries++) {
        int i = rnd() % len, j = rnd() % len;
        if (i == j) continue;
        if (F[0][i] == F[0][j] || F[1][i] == F[1][j] || F[2][i] == F[2][j]) continue;
        int o[3] = {0, 1, 2};
        for (int k = 2; k > 0; k--) { int r = rnd() % (k + 1); int tmp = o[k]; o[k] = o[r]; o[r] = tmp; }
        uint32_t fi[3] = {F[0][i], F[1][i], F[2][i]}, fj[3] = {F[0][j], F[1][j], F[2][j]};
        uint32_t first[3], second[3], third[3];
        first[o[0]] = fi[o[0]] ^ fj[o[0]]; first[o[1]] = fi[o[1]]; first[o[2]] = fi[o[2]];
        second[o[0]] = fj[o[0]]; second[o[1]] = fi[o[1]] ^ fj[o[1]]; second[o[2]] = fi[o[2]];
        third[o[0]] = fj[o[0]]; third[o[1]] = fj[o[1]]; third[o[2]] = fi[o[2]] ^ fj[o[2]];
        for (int m = 0; m < 3; m++) { set_factor(m, i, first[m]); set_factor(m, j, second[m]); }
        int k = append_term(third[0], third[1], third[2]);
        npend = 0; push(i); push(j); push(k);
        settle();
        return 1;
    }
    return 0;
}

static void load(uint32_t (*src)[3], int count) {
    while (len) remove_term(len - 1);
    for (int t = 0; t < count; t++) append_term(src[t][0], src[t][1], src[t][2]);
    npend = 0; for (int t = 0; t < len; t++) push(t);
    while (settle()) { npend = 0; for (int t = 0; t < len; t++) push(t); }
}

#define POOL 16
static uint32_t pool[POOL][MAXT][3]; static int poollen[POOL]; static int npool;
static uint32_t start[MAXT][3];

int main(int argc, char **argv) {
    double seconds = atof(argv[1]); rs = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)atoll(argv[2]) * 0xD1B54A32D192ED03ULL); if (!rs) rs = 1;
    long plateau = argc > 3 ? atol(argv[3]) : 100; int slack = argc > 4 ? atoi(argv[4]) : 2;
    long reset = argc > 5 ? atol(argv[5]) : 300000; int target = argc > 6 ? atoi(argv[6]) : 0;
    int r0; if (scanf("%d %d", &n, &r0) != 2) return 2;
    for (int t = 0; t < r0; t++) if (scanf("%u %u %u", &start[t][0], &start[t][1], &start[t][2]) != 3) return 2;
    for (int i = 0; i < 20; i++) rnd();
    load(start, r0);
    int best = len;
    npool = 1; poollen[0] = len; for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[0][t][m] = F[m][t];
    fprintf(stderr, "start %d (after settling)\n", len);
    clock_t c0 = clock(); long long flips = 0, pluses = 0, resets = 0, it = 0;
    long since_change = 0, since_best = 0;
    while (1) {
        if ((++it & 0x3FF) == 0 && (double)(clock() - c0) / CLOCKS_PER_SEC > seconds) break;
        if (target && best <= target) break;
        int gone = flip();
        if (gone < 0) {
            if (len < best + slack && plus()) pluses++;
            else { int k = rnd() % npool; uint32_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint32_t) * 3 * poollen[k]); load(tmp, poollen[k]); resets++; }
            since_change = 0; continue;
        }
        flips++; since_change++; since_best++;
        if (gone) since_change = 0;
        if (len < best) {
            best = len; npool = 1; poollen[0] = len;
            for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[0][t][m] = F[m][t];
            since_best = 0;
            fprintf(stderr, "best %d at %.1fs flips %lld\n", best, (double)(clock() - c0) / CLOCKS_PER_SEC, flips);
        } else if (len == best && gone && npool < POOL) {
            poollen[npool] = len; for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[npool][t][m] = F[m][t]; npool++;
        }
        if (since_best >= reset) {
            int k = rnd() % npool; uint32_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint32_t) * 3 * poollen[k]); load(tmp, poollen[k]);
            resets++; since_best = since_change = 0;
        } else if (since_change >= plateau && len < best + slack) {
            if (plus()) pluses++;
            since_change = 0;
        }
    }
    printf("%d %lld %lld %lld %.1f\n", best, flips, pluses, resets, (double)(clock() - c0) / CLOCKS_PER_SEC);
    for (int t = 0; t < poollen[0]; t++) printf("%u %u %u\n", pool[0][t][0], pool[0][t][1], pool[0][t][2]);
    return 0;
}
