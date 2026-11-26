// Status: audit-only. Development, not a result: the flip walk of scripts/gf2_flip_walk.c with its moves chosen by
// look-ahead instead of at random, after the non-neural program search's studies (psalmprax/non-neural-program-search,
// study 3: looking one step ahead through the interpreter was the largest single gain per candidate). Driven by
// scripts/search_gf2_guided_walk.py; nothing it prints counts until the driver has checked it.
// At every step K candidate flips are drawn as the plain walk draws one; each is scored without being applied,
// score = w . phi, from what it would do one step ahead:
//   phi0 a new factor would be zero (the product goes)       phi1 a changed product would share two factors (a merge)
//   phi2 how many products already hold the new values       phi3 sharing the changed products would lose
//   phi4 change in the number of bits of the two factors     phi5 log2 of the size of the shared factor's bucket
//   phi6 1 (a constant)
// and one is chosen with probability proportional to exp(score / temperature). With all weights zero it is the plain
// walk's choice among K. Args: seconds seed plateau slack reset target K temperature w0 .. w6. With LOG=path in the
// environment the steps that led to each new best are written there (every candidate's phi, its probability under the
// walk's own choice, and the one taken), for learning the weights from the walk's own descents.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

#define NF 7
static int K = 1; static double temperature = 1.0; static double W[NF];
static FILE *logfile;
typedef struct { int m, t, s, p, q; double phi[NF]; } Cand;

// the last RING steps: every candidate's phi and log probability under the walk's own choice, and which was taken.
// When they lead to a new best (or a new solution at the best count) they are written out, for a conditional logit
// that compares the move taken with the moves passed over in the same state.
#define RING 64
#define MAXK 64
typedef struct { int nc, pick; double logpi[MAXK]; double phi[MAXK][NF]; } Step;
static Step ring[RING]; static long long nring;
static void flush_ring(void) {
    long long n = nring < RING ? nring : RING;
    fprintf(logfile, "B %lld\n", n);
    for (long long i = 0; i < n; i++) {
        Step *st = &ring[(nring - 1 - i) % RING];
        fprintf(logfile, "S %d %d\n", st->nc, st->pick);
        for (int k = 0; k < st->nc; k++) {
            fprintf(logfile, "%.6g", st->logpi[k]);
            for (int f = 0; f < NF - 1; f++) fprintf(logfile, " %g", st->phi[k][f]);
            fprintf(logfile, "\n");
        }
    }
    nring = 0;
}

// does (a, b, c) share two factors with a product other than t and s?
static int shares_two(uint32_t a, uint32_t b, uint32_t c, int t, int s) {
    uint32_t f[3] = {a, b, c};
    for (int mm = 0; mm < 3; mm++) {
        int m1 = (mm + 1) % 3, m2 = (mm + 2) % 3;
        uint32_t v1 = f[m1], v2 = f[m2];
        if (v1 >= MAXV || v2 >= MAXV) continue;
        int k1 = cnt[m1][v1], k2 = cnt[m2][v2];
        if (!k1 || !k2) continue;
        int mb = k1 <= k2 ? m1 : m2; uint32_t vb = k1 <= k2 ? v1 : v2; int mo = k1 <= k2 ? m2 : m1; uint32_t vo = k1 <= k2 ? v2 : v1;
        for (int i = 0; i < cnt[mb][vb]; i++) { int r = bucket[mb][vb][i]; if (r != t && r != s && F[mo][r] == vo) return 1; }
    }
    return 0;
}

static void features(Cand *c) {
    int m = c->m, t = c->t, s = c->s, p = c->p, q = c->q;
    uint32_t tp = F[p][t] ^ F[p][s], sq = F[q][s] ^ F[q][t];
    uint32_t tn[3] = {F[0][t], F[1][t], F[2][t]}, sn[3] = {F[0][s], F[1][s], F[2][s]};
    tn[p] = tp; sn[q] = sq;
    c->phi[0] = (!tp || !sq) ? 1.0 : 0.0;
    c->phi[1] = (tp && sq && (shares_two(tn[0], tn[1], tn[2], t, s) || shares_two(sn[0], sn[1], sn[2], t, s))) ? 1.0 : 0.0;
    c->phi[2] = (double)((tp < MAXV ? cnt[p][tp] : 0) + (sq < MAXV ? cnt[q][sq] : 0));
    c->phi[3] = (double)((cnt[p][F[p][t]] == 2) + (cnt[q][F[q][s]] == 2));
    c->phi[4] = (double)(__builtin_popcount(tp) - __builtin_popcount(F[p][t]) + __builtin_popcount(sq) - __builtin_popcount(F[q][s]));
    c->phi[5] = log2((double)cnt[m][F[m][t]]);
    c->phi[6] = 1.0;
}

static int draw(Cand *c) {
    int total = nshared[0] + nshared[1] + nshared[2];
    if (!total) return 0;
    int k = rnd() % total, m = 0;
    while (k >= nshared[m]) { k -= nshared[m]; m++; }
    uint32_t v = shared[m][k];
    int n_in = cnt[m][v];
    int i = rnd() % n_in, j = rnd() % (n_in - 1); if (j >= i) j++;
    c->m = m; c->t = bucket[m][v][i]; c->s = bucket[m][v][j];
    c->p = (m + 1) % 3; c->q = (m + 2) % 3;
    if (rnd() & 1) { int tmp = c->p; c->p = c->q; c->q = tmp; }
    return 1;
}

static int flip(void) {
    Cand cand[MAXK]; double score[MAXK]; int nc = 0;
    for (int k = 0; k < K && k < MAXK; k++) { if (!draw(&cand[nc])) return -1; features(&cand[nc]); nc++; }
    double top = -1e300;
    for (int k = 0; k < nc; k++) { score[k] = 0; for (int f = 0; f < NF; f++) score[k] += W[f] * cand[k].phi[f]; if (score[k] > top) top = score[k]; }
    double z = 0; for (int k = 0; k < nc; k++) { score[k] = exp((score[k] - top) / temperature); z += score[k]; }
    double u = (double)(rnd() >> 11) / 9007199254740992.0 * z; int pick = nc - 1;
    for (int k = 0; k < nc; k++) { u -= score[k]; if (u <= 0) { pick = k; break; } }
    Cand *c = &cand[pick];
    int t = c->t, s = c->s, p = c->p, q = c->q;
    set_factor(p, t, F[p][t] ^ F[p][s]);
    set_factor(q, s, F[q][s] ^ F[q][t]);
    npend = 0; push(t); push(s);
    int gone = settle();
    if (logfile) {
        Step *st = &ring[nring % RING]; st->nc = nc; st->pick = pick;
        for (int k = 0; k < nc; k++) { st->logpi[k] = log(score[k] / z); memcpy(st->phi[k], cand[k].phi, sizeof(double) * NF); }
        nring++;
    }
    return gone;
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
    K = argc > 7 ? atoi(argv[7]) : 1; temperature = argc > 8 ? atof(argv[8]) : 1.0;
    for (int f = 0; f < NF; f++) W[f] = argc > 9 + f ? atof(argv[9 + f]) : 0.0;
    if (temperature <= 0) temperature = 1e-9;
    if (getenv("LOG")) logfile = fopen(getenv("LOG"), "w");
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
            else { int k = rnd() % npool; uint32_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint32_t) * 3 * poollen[k]); load(tmp, poollen[k]); resets++; nring = 0; }
            since_change = 0; continue;
        }
        flips++; since_change++; since_best++;
        if (gone) since_change = 0;
        if (len < best) {
            best = len; npool = 1; poollen[0] = len;
            for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[0][t][m] = F[m][t];
            since_best = 0;
            if (logfile) flush_ring();
            fprintf(stderr, "best %d at %.1fs flips %lld\n", best, (double)(clock() - c0) / CLOCKS_PER_SEC, flips);
        } else if (len == best && gone && npool < POOL) {
            if (logfile) flush_ring();
            poollen[npool] = len; for (int t = 0; t < len; t++) for (int m = 0; m < 3; m++) pool[npool][t][m] = F[m][t]; npool++;
        }
        if (since_best >= reset) {
            int k = rnd() % npool; uint32_t tmp[MAXT][3]; memcpy(tmp, pool[k], sizeof(uint32_t) * 3 * poollen[k]); load(tmp, poollen[k]);
            resets++; since_best = since_change = 0; nring = 0;
        } else if (since_change >= plateau && len < best + slack) {
            if (plus()) pluses++;
            since_change = 0;
        }
    }
    if (logfile) fclose(logfile);
    printf("%d %lld %lld %lld %.1f\n", best, flips, pluses, resets, (double)(clock() - c0) / CLOCKS_PER_SEC);
    for (int t = 0; t < poollen[0]; t++) printf("%u %u %u\n", pool[0][t][0], pool[0][t][1], pool[0][t][2]);
    return 0;
}
