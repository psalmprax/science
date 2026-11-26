// Status: audit-only. Development, not a result: the flip walk restricted to decompositions of GF(2^n) multiplication that
// are invariant under a power of the Frobenius, phi: x -> x^(2^step) (after Moosbauer and Poole, "Flip graphs with
// symmetry", ISSAC 2025). Driven by scripts/search_gf2_orbit_walk.py; nothing it prints counts until the driver checks it.
//
// A product (a, b, c) is (a.x)(b.y) c. phi maps a decomposition to a decomposition: tau(a, b, c) = (P^T a, P^T b, F c),
// F the matrix of phi and P its inverse. An invariant decomposition is a union of tau-orbits. Every move here is applied
// to whole orbits, so the scheme stays invariant: a flip of t (orbit O1) and s (orbit O2 != O1) sharing a factor is the
// flip of every pair (tau^j t, tau^j s); two orbits of one size whose members share two factors merge; an orbit whose
// representative has two factors fixed by tau^e collapses to size e (its d / e members add up there); members that come
// out equal cancel in pairs; plus transitions take two orbits of one size to three.
// Input: "n step modulus r0" then r0 lines "a b c" (the whole invariant decomposition). Output as gf2_flip_walk.c:
// "best flips pluses resets seconds" then the best decomposition, one product a line.
// Args: seconds seed [plateau slack reset target]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXO 256
#define MAXF 1024
static int n, step, order;
static unsigned modulus;
static uint16_t TA[4096], TC[4096];      // tau on forms and on outputs

static int no;                           // orbits
static uint16_t R[MAXO][3]; static int D[MAXO]; static int dirty[MAXO];
static uint16_t FA[MAXF][3]; static int FO[MAXF], FJ[MAXF]; static int nf;
static uint64_t rs;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static unsigned gmul(unsigned a, unsigned b) {
    unsigned r = 0;
    while (b) { if (b & 1) r ^= a; b >>= 1; a <<= 1; if (a >> n & 1) a ^= modulus; }
    return r;
}
static unsigned gfrob(unsigned a, int k) { for (int i = 0; i < k; i++) a = gmul(a, a); return a; }
static inline uint16_t tau1(int m, uint16_t v) { return m < 2 ? TA[v] : TC[v]; }
static inline void tau3(const uint16_t *in, uint16_t *out) { out[0] = TA[in[0]]; out[1] = TA[in[1]]; out[2] = TC[in[2]]; }
static void taupow(const uint16_t *in, int j, uint16_t *out) {
    uint16_t t[3] = {in[0], in[1], in[2]};
    for (int i = 0; i < j; i++) { uint16_t u[3]; tau3(t, u); memcpy(t, u, sizeof t); }
    memcpy(out, t, sizeof t);
}
static int period(const uint16_t *rep) {
    uint16_t t[3] = {rep[0], rep[1], rep[2]};
    for (int e = 1; e <= order; e++) {
        uint16_t u[3]; tau3(t, u); memcpy(t, u, sizeof t);
        if (t[0] == rep[0] && t[1] == rep[1] && t[2] == rep[2]) return e;
    }
    return order;
}
static int rank_now(void) { int r = 0; for (int o = 0; o < no; o++) r += D[o]; return r; }
static void expand(void) {
    nf = 0;
    for (int o = 0; o < no; o++) {
        uint16_t t[3] = {R[o][0], R[o][1], R[o][2]};
        for (int j = 0; j < D[o]; j++) {
            memcpy(FA[nf], t, sizeof t); FO[nf] = o; FJ[nf] = j; nf++;
            uint16_t u[3]; tau3(t, u); memcpy(t, u, sizeof t);
        }
    }
}
static void drop_orbit(int o) { no--; if (o != no) { memcpy(R[o], R[no], sizeof R[o]); D[o] = D[no]; dirty[o] = dirty[no]; } }
// set orbit o to the sum of the `mult` tau-images (j < mult) of rep: as an orbit with the rep's period, or nothing
static void set_orbit(int o, const uint16_t *rep, int mult) {
    dirty[o] = 1;
    if (!rep[0] || !rep[1] || !rep[2]) { D[o] = 0; return; }
    int e = period(rep);                  // e divides mult when mult is a multiple of the rep's period under the move
    memcpy(R[o], rep, sizeof R[o]);
    if (mult % e) { fprintf(stderr, "internal: period %d does not divide %d\n", e, mult); exit(6); }
    D[o] = ((mult / e) & 1) ? e : 0;
}
static int add_orbit(const uint16_t *rep, int mult) {
    if (no >= MAXO) return -1;
    int o = no++; set_orbit(o, rep, mult);
    if (!D[o]) { no--; return -1; }
    return o;
}
static void clean(void) { for (int o = 0; o < no;) { if (!D[o]) drop_orbit(o); else o++; } }

static int fold_dependent(int o);
// one orbit's local reductions: collapse, a merge with another orbit, a dependent group through one of its members
static int reduce_orbit(int o) {
    int d = D[o];
    for (int e = 1; e < d; e++) {                               // collapse
        if (d % e) continue;
        uint16_t t[3]; taupow(R[o], e, t);
        int fixed = (t[0] == R[o][0]) + (t[1] == R[o][1]) + (t[2] == R[o][2]);
        if (fixed < 2) continue;
        int q = t[0] != R[o][0] ? 0 : t[1] != R[o][1] ? 1 : 2;
        uint16_t rep[3] = {R[o][0], R[o][1], R[o][2]}; uint16_t acc = 0, v = R[o][q];
        for (int i = 0; i < d / e; i++) { acc ^= v; for (int k = 0; k < e; k++) v = tau1(q, v); }
        rep[q] = acc;
        set_orbit(o, rep, e); clean();
        return 1;
    }
    expand();
    for (int i = 0; i < nf; i++) {                              // merge: member i of o with a member k of another orbit
        if (FO[i] != o) continue;
        for (int k = 0; k < nf; k++) {
            int o2 = FO[k];
            if (o2 == o || D[o2] != d) continue;
            int eq = (FA[i][0] == FA[k][0]) + (FA[i][1] == FA[k][1]) + (FA[i][2] == FA[k][2]);
            if (eq < 2) continue;
            int q = FA[i][0] != FA[k][0] ? 0 : FA[i][1] != FA[k][1] ? 1 : 2;
            int j = ((FJ[k] - FJ[i]) % d + d) % d;
            uint16_t sv[3]; taupow(R[o2], j, sv);
            uint16_t rep[3] = {R[o][0], R[o][1], R[o][2]};
            if (eq == 3) rep[2] = 0; else rep[q] ^= sv[q];
            set_orbit(o, rep, d); D[o2] = 0; clean();
            return 1;
        }
    }
    return fold_dependent(o);
}

// settle: local reductions of every changed orbit until none applies; returns products removed
static int settle(void) {
    int before = rank_now();
    for (int guard = 0; guard < 100000; guard++) {
        int o = -1;
        for (int k = 0; k < no; k++) if (dirty[k]) { o = k; break; }
        if (o < 0) break;
        dirty[o] = 0;
        reduce_orbit(o);                                    // marks every orbit it changes
    }
    return before - rank_now();
}

// dependent group through a member of orbit o: members of distinct orbits of one size sharing its factor of mode m,
// whose factors of mode p add up to zero; one of them is the sum of the others there, so its mode-q factor folds into
// theirs and its orbit goes. 1 if applied.
static int fold_dependent(int o) {
    for (int t = 0; t < nf; t++) {
        if (FO[t] != o) continue;
        for (int m = 0; m < 3; m++) {
            int mem[64], nm = 0; unsigned char seen_orbit[MAXO]; memset(seen_orbit, 0, (size_t)no);
            mem[nm++] = t; seen_orbit[o] = 1;
            for (int s = 0; s < nf && nm < 64; s++)
                if (FA[s][m] == FA[t][m] && D[FO[s]] == D[o] && !seen_orbit[FO[s]]) { seen_orbit[FO[s]] = 1; mem[nm++] = s; }
            if (nm < 3) continue;
            for (int pp = 1; pp <= 2; pp++) {
                int p = (m + pp) % 3, q = 3 - m - p;
                uint16_t bv[16]; uint64_t bu[16]; int have[16] = {0};
                for (int i = 0; i < nm; i++) {
                    uint16_t x = FA[mem[i]][p]; uint64_t u = 1ULL << i;
                    while (x) {
                        int top = 31 - __builtin_clz(x);
                        if (!have[top]) { have[top] = 1; bv[top] = x; bu[top] = u; break; }
                        x ^= bv[top]; u ^= bu[top];
                    }
                    if (x) continue;
                    int r = mem[i], d = D[FO[r]], orr = FO[r];
                    for (int k = 0; k < nm; k++) {
                        if (!((u >> k) & 1) || mem[k] == r) continue;
                        int ok = FO[mem[k]];
                        int j = ((FJ[r] - FJ[mem[k]]) % d + d) % d;
                        uint16_t z[3]; taupow(R[orr], j, z);
                        uint16_t rep[3] = {R[ok][0], R[ok][1], R[ok][2]}; rep[q] ^= z[q];
                        set_orbit(ok, rep, d);
                    }
                    D[orr] = 0; clean();
                    return 1;
                }
            }
        }
    }
    return 0;
}

// synchronized flip; returns products removed, or -1 when no flip was found
static int flip(void) {
    expand();
    {
        // a random member t with a partner s in another orbit sharing a factor (mode m)
        int t = -1, s = -1, m = -1;
        for (int tries = 0; tries < 2 * nf && t < 0; tries++) {
            int i = rnd() % nf; long seen = 0;
            for (int k = 0; k < nf; k++) {
                if (FO[k] == FO[i]) continue;
                for (int mm = 0; mm < 3; mm++) if (FA[i][mm] == FA[k][mm]) { seen++; if (rnd() % seen == 0) { t = i; s = k; m = mm; } }
            }
        }
        if (t < 0) return -1;
        int p = (m + 1) % 3, q = (m + 2) % 3;
        if (rnd() & 1) { int x = p; p = q; q = x; }
        int o1 = FO[t], o2 = FO[s], d1 = D[o1], d2 = D[o2];
        int g = d1, h = d2; while (h) { int x = g % h; g = h; h = x; }
        int L = d1 / g * d2;
        // deltas on orbit representatives: rep1 member index FJ[t] + j, rep2 member FJ[s] + j, j < L
        uint16_t tt[3], ss[3]; memcpy(tt, FA[t], sizeof tt); memcpy(ss, FA[s], sizeof ss);
        uint16_t delta1 = 0, delta2 = 0;    // accumulated at member index 0 of each orbit
        for (int j = 0; j < L; j++) {
            int j1 = (FJ[t] + j) % d1, j2 = (FJ[s] + j) % d2;
            if (j1 == 0) delta1 ^= ss[p];     // member 0 of o1 receives s-image's mode p
            if (j2 == 0) delta2 ^= tt[q];
            uint16_t u[3]; tau3(tt, u); memcpy(tt, u, sizeof u); tau3(ss, u); memcpy(ss, u, sizeof u);
        }
        uint16_t r1[3] = {R[o1][0], R[o1][1], R[o1][2]}, r2[3] = {R[o2][0], R[o2][1], R[o2][2]};
        r1[p] ^= delta1; r2[q] ^= delta2;
        int before = rank_now();
        set_orbit(o1, r1, d1); set_orbit(o2, r2, d2); clean();
        int gone = before - rank_now();
        return gone + settle();
    }
}

static int plus(void) {
    for (int tries = 0; tries < 64; tries++) {
        int o1 = rnd() % no, o2 = rnd() % no;
        if (o1 == o2 || D[o1] != D[o2]) continue;
        int d = D[o1];
        uint16_t a[3], b[3]; memcpy(a, R[o1], sizeof a); taupow(R[o2], rnd() % d, b);
        if (a[0] == b[0] || a[1] == b[1] || a[2] == b[2]) continue;
        int o[3] = {0, 1, 2};
        for (int k = 2; k > 0; k--) { int r = rnd() % (k + 1); int x = o[k]; o[k] = o[r]; o[r] = x; }
        uint16_t f1[3], f2[3], f3[3];
        f1[o[0]] = a[o[0]] ^ b[o[0]]; f1[o[1]] = a[o[1]]; f1[o[2]] = a[o[2]];
        f2[o[0]] = b[o[0]]; f2[o[1]] = a[o[1]] ^ b[o[1]]; f2[o[2]] = a[o[2]];
        f3[o[0]] = b[o[0]]; f3[o[1]] = b[o[1]]; f3[o[2]] = a[o[2]] ^ b[o[2]];
        if (no + 1 >= MAXO) return 0;
        set_orbit(o1, f1, d); set_orbit(o2, f2, d); add_orbit(f3, d); clean();
        settle();
        return 1;
    }
    return 0;
}

// the tensor of the current scheme, bit (i, j, k) of n^3; used by the check mode (ORBIT_CHECK in the environment)
static uint64_t T0[4096 / 64 * 12 * 12], T1[4096 / 64 * 12 * 12];
static void tensor_of(uint64_t *T) {
    memset(T, 0, sizeof T0); expand();
    for (int t = 0; t < nf; t++)
        for (int i = 0; i < n; i++) if (FA[t][0] >> i & 1)
            for (int j = 0; j < n; j++) if (FA[t][1] >> j & 1)
                for (int k = 0; k < n; k++) if (FA[t][2] >> k & 1) { int b = (i * n + j) * n + k; T[b / 64] ^= 1ULL << (b % 64); }
}
static void check_tensor(const char *where) {
    tensor_of(T1);
    if (memcmp(T0, T1, sizeof T0)) { fprintf(stderr, "tensor changed after %s\n", where); exit(7); }
}

#define POOL 16
static uint16_t pool[POOL][MAXO][3]; static int poolD[POOL][MAXO], poolN[POOL], npool;
static void save(int k) { poolN[k] = no; memcpy(pool[k], R, sizeof(uint16_t) * 3 * no); memcpy(poolD[k], D, sizeof(int) * no); }
static void restore(int k) { no = poolN[k]; memcpy(R, pool[k], sizeof(uint16_t) * 3 * no); memcpy(D, poolD[k], sizeof(int) * no); memset(dirty, 0, sizeof dirty); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: seconds seed [plateau slack reset target]\n"); return 1; }
    double seconds = atof(argv[1]); rs = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)atoll(argv[2]) * 0xD1B54A32D192ED03ULL); if (!rs) rs = 1;
    long plateau = argc > 3 ? atol(argv[3]) : 100; int slack = argc > 4 ? atoi(argv[4]) : 2;
    long reset = argc > 5 ? atol(argv[5]) : 1000000; int target = argc > 6 ? atoi(argv[6]) : 0;
    int r0;
    if (scanf("%d %d %u %d", &n, &step, &modulus, &r0) != 4) return 2;
    order = n; for (int g = 1; g <= n; g++) if ((g * step) % n == 0) { order = g; break; }
    for (unsigned v = 0; v < (1u << n); v++) {
        TC[v] = (uint16_t)gfrob(v, step);
        unsigned out = 0;                                     // P^T a: (a.(P x)), P = phi^-1 = x -> x^(2^(n-step))
        for (int j = 0; j < n; j++) out |= (unsigned)__builtin_parity(v & gfrob(1u << j, n - step)) << j;
        TA[v] = (uint16_t)out;
    }
    static uint16_t flat[MAXF][3]; static int used[MAXF];
    if (r0 > MAXF) return 2;
    for (int t = 0; t < r0; t++) { unsigned a, b, c; if (scanf("%u %u %u", &a, &b, &c) != 3) return 2; flat[t][0] = a; flat[t][1] = b; flat[t][2] = c; }
    no = 0;
    for (int t = 0; t < r0; t++) {                            // orbits of the start; it must be invariant
        if (used[t]) continue;
        int d = period(flat[t]);
        uint16_t x[3] = {flat[t][0], flat[t][1], flat[t][2]};
        for (int j = 0; j < d; j++) {
            int hit = -1;
            for (int k = 0; k < r0; k++) if (!used[k] && flat[k][0] == x[0] && flat[k][1] == x[1] && flat[k][2] == x[2]) { hit = k; break; }
            if (hit < 0) { fprintf(stderr, "start not invariant under x -> x^(2^%d)\n", step); return 5; }
            used[hit] = 1; uint16_t u[3]; tau3(x, u); memcpy(x, u, sizeof u);
        }
        memcpy(R[no], flat[t], sizeof R[no]); D[no] = d; no++;
    }
    int checking = getenv("ORBIT_CHECK") != NULL;
    tensor_of(T0);
    for (int o = 0; o < no; o++) dirty[o] = 1;
    settle();
    if (checking) check_tensor("the first settle");
    int best = rank_now(); npool = 1; save(0);
    fprintf(stderr, "start %d in %d orbits (order %d)\n", best, no, order);
    clock_t c0 = clock(); long long flips = 0, pluses = 0, resets = 0, it = 0; long since_change = 0, since_best = 0;
    while (1) {
        if ((++it & 0xFF) == 0 && (double)(clock() - c0) / CLOCKS_PER_SEC > seconds) break;
        if (target && best <= target) break;
        int gone = flip();
        if (checking) check_tensor(gone < 0 ? "a failed flip" : "a flip");
        if (gone < 0) {
            if (rank_now() < best + slack && plus()) pluses++;
            else { restore(rnd() % npool); resets++; }
            if (checking) check_tensor("a plus transition after no flip");
            since_change = 0; continue;
        }
        flips++; since_change++; since_best++;
        if (gone) since_change = 0;
        int r = rank_now();
        if (r < best) {
            best = r; npool = 1; save(0); since_best = 0;
            fprintf(stderr, "best %d at %.1fs flips %lld\n", best, (double)(clock() - c0) / CLOCKS_PER_SEC, flips);
        } else if (r == best && gone && npool < POOL) { save(npool); npool++; }
        if (since_best >= reset) { restore(rnd() % npool); resets++; since_best = since_change = 0; }
        else if (since_change >= plateau && r < best + slack) { if (plus()) pluses++; since_change = 0; }
        if (checking) check_tensor("a plus transition or reset");
    }
    restore(0); expand();
    printf("%d %lld %lld %lld %.1f\n", best, flips, pluses, resets, (double)(clock() - c0) / CLOCKS_PER_SEC);
    for (int t = 0; t < nf; t++) printf("%u %u %u\n", FA[t][0], FA[t][1], FA[t][2]);
    return 0;
}
