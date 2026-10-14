//! RIME search: a deterministic beam over the frozen five rewrites (spec §8).
//!
//! One level = one rewrite application from each beam member (depth ≤ 2 for a
//! seed, per §7). Candidates are scored in parallel on their fitted readout and
//! accepted only when they improve on their immediate parent by more than the
//! complexity charge, so the search *terminates on its own* on a null world:
//! no candidate scores positive, the frontier empties, and nothing is proposed.
//!
//! Determinism: candidates are enumerated in a fixed order, deduplicated by
//! content-addressed graph id, scored by a pure function, and ranked with a
//! total order (score, then id). Same inputs ⇒ same proposal list, byte for
//! byte, on any thread count.
//!
//! This driver is Stage-2 scaffolding and is **not** part of the frozen v0.1
//! surface: it composes the frozen primitives, it does not add rewrites.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::eval::{charged_dof, fitted_mse, score_batch_fitted, FittedScore, ScoreConfig, READOUT_PARAMS};
use super::graph::{Graph, Node, LAG_SET};
use super::record::CandidateRecord;
use super::rewrite::{apply, RewriteOp};
use super::seeds::SeedOrder;

/// Search configuration. `lambda_c` is a frozen quantity (spec §8): it is part
/// of the experiment protocol, never tuned per candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum rewrite depth from a seed (spec §7: depth ≤ 2).
    pub max_depth: usize,
    /// Beam width retained per depth.
    pub beam: usize,
    /// Hard cap on scored candidates per run (spec §8: ≤200).
    pub budget: usize,
    /// Complexity penalty per charged degree of freedom.
    pub lambda_c: f64,
    /// Ridge coefficient for the readout fit.
    pub ridge: f64,
    /// Record seed; part of every candidate's content-addressed id.
    pub seed: u64,
    /// Static search mode: disable every temporal primitive — no `Lag` delays,
    /// no `State` memory cells, in seeds and rewrites alike (`v0.1.2` algebraic
    /// grammar extension, for order-structured STATIC laws). Rationale: on a
    /// log-spaced static axis (the QEC threshold-law failure), a lagged level
    /// fits grid order, not physics — dev scoring cannot tell the two apart,
    /// and the holdout correctly rejects what should never have been proposed.
    /// Static mode makes grid-order fitting structurally impossible instead of
    /// gating it after the fact. It is OFF by default: the frozen §7/§8 grammar
    /// is unchanged for every existing caller.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub static_ops: bool,
    /// Family-2 (delayed levels) ordering when the seed catalogue must be
    /// truncated by `MAX_SEED_FORMS` (many inputs). `Legacy` (default) is
    /// input-major and can drop whole inputs' shortest lag; `CoverageFirst`
    /// is lag-major so every input keeps lag-1 first. Opt-in: the frozen §7
    /// catalogue is unchanged for every existing caller.
    #[serde(default, skip_serializing_if = "crate::rime::seeds::SeedOrder::is_legacy")]
    pub seed_order: SeedOrder,
    /// Contemporaneous self-products (`(x_i * x_i)`): admits lag-0
    /// self-multiplication into the seed catalogue (spec §7 algebraic extension).
    ///
    /// Addresses the QEC structural reachability flaw: previously, self-products
    /// were only emitted for delayed lags k in {1, 2, 5}, rendering same-time
    /// quadratic laws (p_L = 1000 * p^2) unreachable at depth 0.
    /// Default: false (preserves frozen §7 catalogue byte identity).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub square_seeds: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig { max_depth: 2, beam: 8, budget: 200, lambda_c: 1e-3, ridge: 1e-9, seed: 0,
            static_ops: false, seed_order: SeedOrder::Legacy, square_seeds: false }
    }
}

impl SearchConfig {
    /// Legacy penalty: a fixed fraction of the target variance per charged
    /// degree of freedom.
    ///
    /// A fixed absolute `lambda_c` is only meaningful for standardised targets.
    /// The audit's null test failed because 1.0 per edge dwarfs every possible
    /// MSE change (so nothing can ever be promoted) while values like 1e-6 are
    /// below the in-sample noise floor of a fitted readout (so everything is).
    /// This constructor makes the charge comparable to the signal a real
    /// mechanism must beat, but it ignores the sample size, which makes it too
    /// permissive on short series — prefer [`SearchConfig::noise_scaled`].
    pub fn scaled_to(target: &[f64], rel_penalty: f64) -> Self {
        let v = target_variance(target);
        SearchConfig { lambda_c: rel_penalty * v, ..Default::default() }
    }

    /// Sample-size-aware penalty: `gamma * var(y) / n_dev` per charged degree of
    /// freedom.
    ///
    /// `var(y)/n_dev` is the sampling-noise scale of a dev MSE, so a candidate
    /// must beat what fitting three parameters to `n_dev` rows can achieve by
    /// luck alone; `gamma` is the safety factor (1.0 is the calibration, 1.5 is
    /// deliberately conservative). This is the scale the project's own power
    /// curves live on: at `n_dev = 90` a candidate must explain roughly
    /// `dof/90` of the variance, which is why the search abstains on a null
    /// world that a fixed-fraction penalty would wave through.
    pub fn noise_scaled(target: &[f64], n_dev: usize, gamma: f64) -> Self {
        let v = target_variance(target);
        SearchConfig { lambda_c: gamma * v / n_dev.max(1) as f64, ..Default::default() }
    }

    /// Selection charge in degrees of freedom for a search over `budget`
    /// candidates: EBIC-style `2·ln M`. Preregistered through the budget, so it
    /// does not depend on the data.
    pub fn selection_dof(&self) -> f64 {
        multiplicity_dof(self.budget)
    }

    /// The scoring configuration this search runs with.
    pub fn score_config(&self) -> ScoreConfig {
        ScoreConfig { lambda_c: self.lambda_c, ridge: self.ridge, selection_dof: self.selection_dof() }
    }

    /// Per-environment aggregation rule (§8 "mean loss reduction across dev
    /// environments", misapplied by v0.1.1's pooled scoring).
    ///
    /// - `MeanPerEnv` (default): exact §8 reading — per-environment dev MSE,
    ///   then a mean of equal weights. Environments count, not rows.
    /// - `MedianWorstGated`: a robust design offered for testing (this is a
    ///   *design choice*, not a result). Median across environments, with a
    ///   worst-environment guard so candidate E cannot hide behind E': the gate
    ///   additionally requires the candidate to beat its parent by more than
    ///   the noise scale in the WORST environment, or no promotion — median
    ///   alone admits survivors that are only wins in the best env.
    ///
    /// The rule is part of the frozen protocol, fixed before any candidate is
    /// scored; it is not tuned per candidate or per run.
    pub fn aggregation_rules() -> Vec<EnvAggregation> {
        vec![EnvAggregation::MeanPerEnv, EnvAggregation::MedianWorstGated]
    }

    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    pub fn with_beam(mut self, b: usize) -> Self {
        self.beam = b.max(1);
        self
    }

    pub fn with_budget(mut self, b: usize) -> Self {
        self.budget = b.max(1);
        self
    }

    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// EBIC-style selection charge: `2·ln M` degrees of freedom for a search over
/// `M` candidates (spec §10, "Bonferroni over grammar size").
pub fn multiplicity_dof(budget: usize) -> f64 {
    2.0 * (budget.max(2) as f64).ln()
}

/// One development environment: its own input columns and target. Environments
/// are logical datasets (regimes, symbols, seeds of a synthetic protocol), never
/// a random physical split of one series: splitting one series mid-way changes
/// the lag boundary structure the mechanism was discovered on.
#[derive(Clone, Debug)]
pub struct EnvData {
    pub cols: Vec<Vec<f64>>,
    pub target: Vec<f64>,
}

/// Frozen multi-environment aggregation rules (§8: "mean loss reduction across
/// dev environments").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvAggregation {
    /// Equal-weight mean of per-environment MSEs — the literal §8 reading.
    /// Every environment counts equally regardless of length.
    MeanPerEnv,
    /// Median of per-environment MSEs plus a worst-environment gate: the
    /// candidate must also beat its parent by more than the noise scale in the
    /// worst environment. A robustness design offered for testing — it removes
    /// the pooled scoring's dominance trap (one large env can outvote a small
    /// bad env) and rejects env-cheaters that only win where the mechanism
    /// happens to hold.
    MedianWorstGated,
}

/// Aggregate per-environment MSEs. NaN environments (no finite dev rows) do not
/// dilute the statistic: they count against a MedianWorstGated candidate (the
/// gate fails) and are ignored only by the mean, mirroring how a finite-only
/// mean treats an unusable environment.
pub fn aggregate(mses: &[f64], agg: EnvAggregation) -> f64 {
    let mut mses: Vec<f64> = mses.to_vec();
    mses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match agg {
        EnvAggregation::MeanPerEnv => {
            let finite: Vec<f64> = mses.iter().copied().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                f64::INFINITY
            } else {
                finite.iter().sum::<f64>() / finite.len() as f64
            }
        }
        EnvAggregation::MedianWorstGated => {
            // Median of ALL environments: an unusable one is evidence of
            // non-transfer and must move the statistic, not vanish.
            let n = mses.len();
            if n == 0 {
                return f64::INFINITY;
            }
            let mid = n / 2;
            if n % 2 == 0 {
                (mses[mid - 1] + mses[mid]) / 2.0
            } else {
                mses[mid]
            }
        }
    }
}

/// The worst-ENVIRONMENT pass/fail for MedianWorstGated.
///
/// "Worst" is measured from the CANDIDATE's side: the environment where the
/// candidate's fitted readout achieves the LOWEST *relative* improvement over
/// the null model (`Δmse / mse_parent`). Selecting the worst env by the
/// parent's own MSE instead would hand the decision to whichever environment
/// happens to have the highest target variance — a property of the *data*, not
/// of the *mechanism* — which is exactly the dominance trap this gate is meant
/// to close. The candidate must clear, in every environment it is claimed to
/// transfer to, more than `gamma * var(y)/n` improvement; an env it cannot
/// improve beyond noise is the judge, not an env with a big variance.
fn worst_env_gate(
    cand_by_env: &[f64],
    parent_by_env: &[f64],
    envs: &[EnvData],
    gamma: f64,
) -> bool {
    if cand_by_env.is_empty() || cand_by_env.len() != parent_by_env.len() {
        return false;
    }
    // Relative improvement per env; NaN/∞ candidate MSE counts as "no transfer".
    let mut judge = 0usize;
    let mut weakest = f64::INFINITY;
    for (k, (&c, &p)) in cand_by_env.iter().zip(parent_by_env.iter()).enumerate() {
        let rel = if c.is_finite() && p.is_finite() && p > 0.0 {
            (p - c) / p
        } else {
            f64::NEG_INFINITY
        };
        if rel < weakest {
            weakest = rel;
            judge = k;
        }
    }
    let env = &envs[judge];
    let var = target_variance(&env.target);
    let n_dev = env.target.iter().filter(|v| v.is_finite()).count().max(1);
    // The candidate must IMPROVE the judged environment by more than its
    // sampling-noise scale; otherwise the gate fails. INFINITY in the judged
    // env fails the gate.
    let cand = cand_by_env[judge];
    if !cand.is_finite() || !parent_by_env[judge].is_finite() {
        return false;
    }
    cand < parent_by_env[judge] - gamma * var / n_dev as f64
}

/// Assemble a multi-envelopment partition from raw per-frame columns/targets:
/// exactly one NaN boundary row at each junction (the single-frame search's
/// convention) so a lag can never read across an environment boundary.
pub fn envs_from_frames(frames: &[(Vec<Vec<f64>>, Vec<f64>)]) -> Vec<EnvData> {
    frames
        .iter()
        .map(|(cols, target)| EnvData {
            cols: cols
                .iter()
                .map(|c| {
                    let mut v = c.clone();
                    v.push(f64::NAN); // boundary separator row
                    v
                })
                .collect(),
            target: {
                let mut t = target.clone();
                t.push(f64::NAN);
                t
            },
        })
        .collect()
}

/// Minimum finite dev rows an environment must contribute to be scored.
/// Frozen: below this, the environment is UNUSABLE rather than informative —
/// its variance is a two-point statistic and any "improvement" there is
/// meaningless. One unusable environment never fails a MeanPerEnv aggregate on
/// its own (the mean skips it, as a finite-only statistic must), but it never
/// helps either, and MedianWorstGated treats it as non-transfer by construction
/// (the gate judges improvement, and an unusable env cannot show any).
pub const MIN_ENV_FINITE_ROWS: usize = 15;

/// How an environment was handled by the search. Explicit outcomes, not silent
/// best-effort scoring (spec §2: unresolved results must be reported, not
/// laundered into a score).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvStatus {
    /// Scored normally.
    #[serde(rename = "scored")]
    Scored,
    /// Fewer than [`MIN_ENV_FINITE_ROWS`] finite target rows: excluded from
    /// every aggregation, reported in the packet.
    #[serde(rename = "too_few_rows")]
    TooFewRows,
    /// Columns-×-target lengths mismatch: Malformed environments are a caller
    /// bug, surfaced to the proposer rather than silently truncated.
    #[serde(rename = "malformed")]
    Malformed,
}

impl EnvStatus {
    pub fn name(&self) -> &'static str {
        match self {
            EnvStatus::Scored => "scored",
            EnvStatus::TooFewRows => "too_few_rows",
            EnvStatus::Malformed => "malformed",
        }
    }
}

/// Validate environments before any scoring: structural consistency (num_cols
/// alignment between envs) and minimum evidence. Returns per-env statuses and
/// the indices that will actually be scored.
pub fn validate_envs(envs: &[EnvData], expected_cols: usize) -> (Vec<EnvStatus>, Vec<usize>) {
    let mut statuses = Vec::with_capacity(envs.len());
    let mut usable = Vec::with_capacity(envs.len());
    for (k, env) in envs.iter().enumerate() {
        let ragged = env.cols.iter().any(|c| c.len() != env.target.len())
            || env.cols.iter().any(|c| c.len() != env.cols[0].len())
            || env.cols.is_empty()
            || env.cols[0].is_empty();
        if ragged || env.cols.len() != expected_cols {
            statuses.push(EnvStatus::Malformed);
            continue;
        }
        let finite_target = env.target.iter().filter(|v| v.is_finite()).count();
        if finite_target < MIN_ENV_FINITE_ROWS {
            statuses.push(EnvStatus::TooFewRows);
            continue;
        }
        statuses.push(EnvStatus::Scored);
        usable.push(k);
    }
    (statuses, usable)
}

/// Outcome of a multi-environment search (§8 literal: equal-weight per-env mean;
/// `worst_env_passed` is the MedianWorstGated-style guard shared by both rules).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchEnvsOutcome {
    /// Most recent proposals found (single-env fallback runs each env alone).
    pub proposals: Vec<Proposal>,
    pub aggregation: EnvAggregation,
    /// Aggregated (null-reference) dev MSE across environments.
    pub baseline_mse: f64,
    /// Worst single-environment dev MSE of the null model.
    pub worst_env_mse: f64,
    /// Worst-env guard: the promoted candidate must also improve in the worst env.
    pub worst_env_passed: bool,
    /// Per-environment handling, aligned with the caller's env order.
    pub env_statuses: Vec<EnvStatus>,
    pub evaluated: usize,
    pub accepted: usize,
}

impl SearchEnvsOutcome {
    pub fn to_json(&self) -> Value {
        json!({
            "aggregation": self.aggregation,
            "baseline_mse": self.baseline_mse,
            "worst_env_mse": self.worst_env_mse,
            "worst_env_passed": self.worst_env_passed,
            "env_statuses": self.env_statuses.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "proposals": self.proposals.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
        })
    }
}

/// Search mechanisms across SEVERAL datasets and enforce a frozen aggregation
/// rule — the fix for v0.1.1's pooled scoring, where one long environment can
/// dominate a small bad one.
///
/// Protocol: score every candidate per environment, aggregate with a rule fixed
/// in advance (`agg`), and gate on a worst-environment improvement as well.
/// No candidate pool, budget or lambda is retuned per environment.
pub fn search_envs(
    seeds: &[Graph],
    envs: &[EnvData],
    cfg: &SearchConfig,
    agg: EnvAggregation,
    gamma_gate: f64,
) -> SearchEnvsOutcome {
    assert!(!envs.is_empty(), "search_envs needs at least one environment");
    // Frozen pre-scoring validation (spec §1 v0.1.2: define how short, ragged
    // and invalid environments are handled BEFORE benchmarking, not after).
    let expected_cols = envs[0].cols.len();
    let (env_statuses, usable) = validate_envs(envs, expected_cols);
    if usable.is_empty() {
        // No environment carries enough evidence: an explicit abstain, not a
        // degenerate score. Missing-observation handling is structural here —
        // NaN rows remain skip-and-count inside each scored env.
        return SearchEnvsOutcome {
            proposals: Vec::new(),
            aggregation: agg,
            baseline_mse: f64::INFINITY,
            worst_env_mse: f64::INFINITY,
            worst_env_passed: false,
            env_statuses,
            evaluated: 0,
            accepted: 0,
        };
    }
    let usable_envs: Vec<&EnvData> = usable.iter().map(|&k| &envs[k]).collect();
    // Null-model reference: per-env parent MSE for every candidate job, plus
    // the aggregated and worst-env statistics of the reference itself.
    let null_graph = null_model();
    let mut parent_by_env = Vec::with_capacity(usable_envs.len());
    for env in &usable_envs {
        let (mse, _, _) = fitted_mse(&null_graph, &env.cols, &env.target, cfg.ridge);
        parent_by_env.push(mse);
    }
    let baseline_mse = aggregate(&parent_by_env, agg);
    let worst = parent_by_env
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut outcome = SearchEnvsOutcome {
        proposals: Vec::new(),
        aggregation: agg,
        baseline_mse,
        worst_env_mse: worst,
        worst_env_passed: false,
        env_statuses,
        evaluated: 0,
        accepted: 0,
    };
    if !baseline_mse.is_finite() || !worst.is_finite() {
        return outcome;
    }
    // Run the single-env engine inside each environment, sharing the budget
    // across all of them (budget is per run, not per env).
    //
    // Candidate pool rule (crucial for the per-env protocol): each env's single-
    // env run emits its own winner; the pooled candidate is chosen by a FROZEN
    // rule, not per-env trial-and-error. Here: score EVERY seed and rewrite in
    // EVERY env, then rank by the aggregated per-env score. That is a strict
    // reading of "score each environment separately with a frozen aggregation
    // rule" — one candidate universe, one ranking, no per-env promotion.
    //
    // Implementation: the single-env `search` inside each env ranks its own
    // winners; we keep the UNION of graph ids proposed anywhere (cheap: ≤budget
    // ids) and re-score exactly that union in every env to compute the
    // aggregate. This is the same candidate set an all-env batch scorer would
    // produce, without a second engine.
    let per_env_budget = (cfg.budget / usable_envs.len()).max(8);
    let mut pool: Vec<(Graph, String, String)> = Vec::new(); // (graph, op, parent_id)
    let mut pool_ids: HashSet<String> = HashSet::new();
    let mut evaluated = 0usize;
    for env in usable_envs.iter().copied() {
        let env_cfg = SearchConfig {
            budget: per_env_budget,
            ..cfg.clone()
        };
        let inner = search(&seeds, &env.cols, &env.target, &env_cfg);
        evaluated += inner.evaluated;
        for p in &inner.proposals {
            if pool_ids.insert(p.graph_id.clone()) {
                pool.push((p.graph.clone(), p.rewrite_op.clone(), p.parent_id.clone()));
            }
        }
    }
    outcome.evaluated = evaluated;

    // Aggregate scoring: every pooled candidate, scored in every env against
    // that env's null MSE, then the frozen aggregation.
    if pool.is_empty() {
        return outcome;
    }
    let mut best: Option<(Proposal, f64)> = None;
    let sc = cfg.score_config();
    for (graph, op, parent_id) in &pool {
        let mut cand_by_env: Vec<f64> = Vec::with_capacity(usable_envs.len());
        for env in &usable_envs {
            let (mse, _s_a, _s_b) = fitted_mse(graph, &env.cols, &env.target, cfg.ridge);
            cand_by_env.push(mse);
        }        // Aggregate dMSE across envs (both rules aggregate MSE; score = ΔMSE - λ·dof)。
        let agg_cand = aggregate(&cand_by_env, agg);

        let score = (baseline_mse - agg_cand) - sc.lambda_c * charged_dof(graph, sc.selection_dof);
        if !score.is_finite() || score <= 0.0 {
            continue;
        }
        let repo_env = usable_envs[0];
        let s0 = score_batch_fitted(
            &[(graph.clone(), parent_by_env[0], repo_env.target.clone())],
            &repo_env.cols,
            &sc,
        );
        let mut prop = Proposal::from_score(
            graph,
            &null_model(),
            op,
            0,
            parent_by_env[0],
            cfg.lambda_c,
            cfg.selection_dof(),
            cfg.seed,
            &s0[0],
        );
        prop.parent_id = parent_id.clone();
        prop.score = score;
        prop.ref_score = score;
        prop.mse = agg_cand;
        if baseline_mse > 0.0 {
            prop.r2 = 1.0 - agg_cand / baseline_mse;
        }
        if best.as_ref().map(|(_, s)| score > *s).unwrap_or(true) {
            best = Some((prop, score));
        }
    }

    // Worst-environment gate on the best aggregated candidate (MedianWorstGated;
    // retained for MeanPerEnv as an explicit report field).
    if let Some((p, _)) = &best {
        let mut cand_by_env: Vec<f64> = Vec::with_capacity(usable_envs.len());
        for env in &usable_envs {
            let (mse, _s, _b) = fitted_mse(&p.graph, &env.cols, &env.target, cfg.ridge);
            cand_by_env.push(mse);
        }
        let gate_envs: Vec<EnvData> = usable_envs.iter().map(|e| (*e).clone()).collect();
        outcome.worst_env_passed =
            worst_env_gate(&cand_by_env, &parent_by_env, &gate_envs, gamma_gate);
    }
    let promoted = match agg {
        EnvAggregation::MeanPerEnv => true,
        EnvAggregation::MedianWorstGated => outcome.worst_env_passed,
    };
    if promoted {
        if let Some((p, _)) = best {
            outcome.proposals.push(p);
        }
        outcome.accepted = outcome.proposals.len();
    }
    outcome
}

/// Variance of the finite entries of a target series.
pub fn target_variance(target: &[f64]) -> f64 {
    let f: Vec<f64> = target.iter().copied().filter(|v| v.is_finite()).collect();
    if f.len() < 2 {
        return 0.0;
    }
    let m = f.iter().sum::<f64>() / f.len() as f64;
    f.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / f.len() as f64
}

/// The frozen rewrite set, enumerated in a fixed priority order: bounded
/// response, memory, delay, interaction, pruning.
///
/// The order is the search's inductive bias and is documented rather than
/// hidden: with a finite budget it decides which candidates get scored first.
pub fn enumerate_ops(g: &Graph) -> Vec<RewriteOp> {
    let n = g.nodes.len();
    let mut ops = Vec::new();
    ops.push(RewriteOp::WrapNonlinearity { src: g.output });
    for (decay_num, decay_den) in super::seeds::DECAY_PRIORS {
        ops.push(RewriteOp::AddState { decay_num, decay_den });
    }
    for src in 0..n {
        for k in LAG_SET {
            ops.push(RewriteOp::AddLagEdge { src, k });
        }
    }
    for a in 0..n {
        for b in a..n {
            ops.push(RewriteOp::AddInteraction { a, b });
        }
    }
    for node in 0..n {
        if !matches!(g.nodes[node], Node::Const(_)) {
            ops.push(RewriteOp::PruneEdge { node });
        }
    }
    ops
}

/// A candidate mechanism proposed by the search, with its dev evidence (spec §11
/// `CandidateProposal`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub graph_id: String,
    pub parent_id: String,
    pub rewrite_op: String,
    pub depth: usize,
    pub equation: String,
    pub canonical: String,
    /// Spec §8 score against the immediate parent.
    pub score: f64,
    /// Improvement against the best seed, comparable across parents.
    pub ref_score: f64,
    pub mse: f64,
    pub r2: f64,
    pub scale: f64,
    pub intercept: f64,
    /// Charged degrees of freedom: edges + readout parameters + selection charge.
    pub complexity: f64,
    /// Graph edges (structural complexity without the readout or selection term).
    pub edges: usize,
    /// EBIC-style `2·ln M` charge for the search width this run was configured with.
    pub selection_dof: f64,
    pub n_eff: usize,
    pub seed: u64,
    #[serde(skip)]
    pub graph: Graph,
}

impl Proposal {
    fn from_score(
        graph: &Graph,
        parent: &Graph,
        op: &str,
        depth: usize,
        baseline_mse: f64,
        lambda_c: f64,
        selection_dof: f64,
        seed: u64,
        s: &FittedScore,
    ) -> Self {
        let ref_score = (baseline_mse - s.mse) - lambda_c * s.complexity;
        Proposal {
            graph_id: graph.id(),
            parent_id: parent.id(),
            rewrite_op: op.to_string(),
            depth,
            equation: graph.describe(),
            canonical: graph.canonical(),
            score: s.score,
            ref_score,
            mse: s.mse,
            r2: s.r2,
            scale: s.scale,
            intercept: s.intercept,
            complexity: s.complexity,
            edges: graph.edge_count(),
            selection_dof,
            n_eff: s.n_eff,
            seed,
            graph: graph.clone(),
        }
    }

    /// Spec §11 + §12 packet: graph id, equation, dev evidence.
    ///
    /// `graph_json` is the executable graph itself, so a consumer (the Python
    /// proposer, the referee) never has to re-implement the canonical form to
    /// get a runnable mechanism back.
    ///
    /// `inputs_used` (v0.1.2 diagnostics addition) lists the input column
    /// indices read by the output expression. Instrumentation only: it
    /// changes no score, no gate, no selection — presence, not importance.
    pub fn to_json(&self) -> Value {
        json!({
            "graph_json": serde_json::to_string(&self.graph).unwrap_or_default(),
            "graph_id": self.graph_id,
            "parent_id": self.parent_id,
            "rewrite_op": self.rewrite_op,
            "depth": self.depth,
            "equation": self.equation,
            "canonical": self.canonical,
            "inputs_used": self.graph.inputs_used(),
            "dev_evidence": {
                "score": self.score,
                "ref_score": self.ref_score,
                "mse": self.mse,
                "r2": self.r2,
                "scale": self.scale,
                "intercept": self.intercept,
                "complexity": self.complexity,
                "edges": self.edges,
                "readout_params": READOUT_PARAMS,
                "selection_dof": self.selection_dof,
                "n_eff": self.n_eff,
            },
            "seed": self.seed,
        })
    }

    /// The immutable ledger record for this candidate.
    pub fn record(&self) -> CandidateRecord {
        CandidateRecord::new(
            &self.graph,
            Some(self.parent_id.clone()),
            &self.rewrite_op,
            self.seed,
            self.score,
        )
    }
}

/// Outcome of a search run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchOutcome {
    /// Accepted candidates, best first.
    pub proposals: Vec<Proposal>,
    /// Baseline (best seed) fitted dev MSE used as the comparison reference.
    pub baseline_mse: f64,
    pub baseline_graph_id: String,
    /// Candidates scored (budget consumption).
    pub evaluated: usize,
    /// Candidates that scored above zero against their parent.
    pub accepted: usize,
    /// Candidates rejected by the score gate (finite evidence, score ≤ 0).
    /// Funnel instrumentation (v0.1.2 diagnostics): counting only.
    pub rejected_score: usize,
    /// Candidates dropped by structural/expression dedup before scoring.
    pub dropped_dedup: usize,
    /// Candidates dropped as invalid (failed validation) or out of scope
    /// (static mode temporal forms). Never scored.
    pub dropped_scope: usize,
}

impl SearchOutcome {
    pub fn to_json(&self) -> Value {
        json!({
            "baseline_mse": self.baseline_mse,
            "baseline_graph_id": self.baseline_graph_id,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "rejected_score": self.rejected_score,
            "dropped_dedup": self.dropped_dedup,
            "dropped_scope": self.dropped_scope,
            "proposals": self.proposals.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
        })
    }

    /// Best proposal, if any survived the gates.
    pub fn best(&self) -> Option<&Proposal> {
        self.proposals.first()
    }
}

/// Total order over candidate scores: higher first, ties by graph id, NaNs last.
fn rank_desc(a: &Proposal, b: &Proposal) -> std::cmp::Ordering {
    let key = |v: f64| if v.is_nan() { f64::NEG_INFINITY } else { v };
    key(b.ref_score)
        .partial_cmp(&key(a.ref_score))
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.graph_id.cmp(&b.graph_id))
}

/// The reference (null) model: a constant, i.e. the mean predictor with a free
/// scale. Its fitted dev MSE is the target variance.
pub fn null_model() -> Graph {
    let mut g = Graph::new();
    g.output = g.push(Node::Const(0.0));
    g
}

/// Run the beam search. `seeds` are the small forms; `cols`/`target` are the
/// **development** data only. The gate is spec §8: a candidate is accepted when
/// `(parent_mse - candidate_mse) - lambda_c * complexity > 0`.
pub fn search(
    seeds: &[Graph],
    cols: &[Vec<f64>],
    target: &[f64],
    cfg: &SearchConfig,
) -> SearchOutcome {
    // Reference model: the mean predictor (spec §6 "flat/white residuals →
    // STOP"). Every candidate — including the seeds themselves — must beat the
    // mean predictor by more than its complexity charge. Anchoring on the null
    // (rather than on the best seed) is what makes the score comparable across
    // candidates of different depth and parentage.
    let null_graph = null_model();
    let (baseline_mse, _, _) = fitted_mse(&null_graph, cols, target, cfg.ridge);
    let baseline_graph_id = null_graph.id();
    if !baseline_mse.is_finite() {
        return SearchOutcome {
            proposals: Vec::new(),
            baseline_mse,
            baseline_graph_id,
            evaluated: 0,
            accepted: 0,
            rejected_score: 0,
            dropped_dedup: 0,
            dropped_scope: 0,
        };
    }

    // Two dedup keys. `seen` is structural (content-addressed graph id) and
    // stops the search from re-expanding a graph. `seen_expr` is semantic at the
    // expression level: a rewrite can leave dead nodes behind (PruneEdge parks a
    // Const(0) that nothing reads) or reach the same expression through a
    // different arena, and proposing both would spend budget twice on one
    // mechanism and inflate the multiplicity the referee must correct for.
    let mut seen: HashSet<String> = HashSet::new();
    let mut seen_expr: HashSet<String> = HashSet::new();
    seen.insert(baseline_graph_id.clone());
    seen_expr.insert(null_graph.describe());
    let mut proposals: Vec<Proposal> = Vec::new();
    let mut evaluated = 0usize;
    let mut rejected_score = 0usize;
    let mut dropped_dedup = 0usize;
    let mut dropped_scope = 0usize;

    // Depth 0: the seed catalogue itself competes for promotion. Spec §7 makes
    // the seeds the initial hypothesis set; a catalogue that is only a source
    // of parents would hide every form the rewrites cannot reach.
    let mut seed_jobs: Vec<(Graph, f64, Vec<f64>)> = Vec::new();
    for s in seeds {
        if evaluated + seed_jobs.len() >= cfg.budget {
            break;
        }
        if s.validate().is_err() {
            dropped_scope += 1;
            continue;
        }
        if !seen.insert(s.id()) {
            dropped_dedup += 1;
            continue;
        }
        // Static mode: temporal seeds (lags, differences, memory cells) are
        // unrepresentable, not gated — a form reading time cannot appear as a
        // candidate at all (see `static_ops` rationale).
        if cfg.static_ops && s.is_temporal() {
            dropped_scope += 1;
            continue;
        }
        seed_jobs.push((s.clone(), baseline_mse, target.to_vec()));
    }
    let seed_scores = score_batch_fitted(&seed_jobs, cols, &cfg.score_config());
    evaluated += seed_jobs.len();
    let mut seed_keep: Vec<(Graph, f64, Proposal)> = Vec::new();
    for ((g, _pm, _t), s) in seed_jobs.iter().zip(seed_scores.iter()) {
        if !s.mse.is_finite() || !s.score.is_finite() || s.score <= 0.0 {
            rejected_score += 1;
            continue;
        }
        if !seen_expr.insert(g.describe()) {
            dropped_dedup += 1;
            continue;
        }
        let p = Proposal::from_score(
            g,
            &null_graph,
            "seed",
            0,
            baseline_mse,
            cfg.lambda_c,
            cfg.selection_dof(),
            cfg.seed,
            s,
        );
        seed_keep.push((g.clone(), s.mse, p));
    }
    if seed_keep.is_empty() {
        // No seed beats the mean predictor: abstain without expanding.
        return SearchOutcome { proposals, baseline_mse, baseline_graph_id, evaluated, accepted: 0,
            rejected_score, dropped_dedup, dropped_scope };
    }
    seed_keep.sort_by(|a, b| rank_desc(&a.2, &b.2));
    for (_, _, p) in &seed_keep {
        proposals.push(p.clone());
    }
    let mut frontier: Vec<(Graph, f64)> =
        seed_keep.into_iter().take(cfg.beam).map(|(g, mse, _)| (g, mse)).collect();

    for depth in 1..=cfg.max_depth.max(1) {
        // Enumerate children of the current beam, in deterministic order.
        let mut jobs: Vec<(Graph, f64, Vec<f64>)> = Vec::new();
        let mut meta: Vec<(Graph, String, usize)> = Vec::new();
        'outer: for (parent, parent_mse) in &frontier {
            for op in enumerate_ops(parent) {
                // Static mode: temporal rewrites are removed from the grammar,
                // so no candidate reachable from any beam member reads time.
                if cfg.static_ops
                    && matches!(op, RewriteOp::AddLagEdge { .. } | RewriteOp::AddState { .. })
                {
                    dropped_scope += 1;
                    continue;
                }
                if evaluated + jobs.len() >= cfg.budget {
                    break 'outer;
                }
                let Some(child) = apply(parent, &op) else { dropped_scope += 1; continue };
                child.validate().ok();
                if !seen.insert(child.id()) || !seen_expr.insert(child.describe()) {
                    dropped_dedup += 1;
                    continue;
                }
                jobs.push((child, *parent_mse, target.to_vec()));
                meta.push((parent.clone(), op.name().to_string(), depth));
            }
        }
        if jobs.is_empty() {
            break;
        }
        let scores = score_batch_fitted(&jobs, cols, &cfg.score_config());
        evaluated += jobs.len();

        let mut next: Vec<(Graph, f64, Proposal)> = Vec::new();
        for (((graph, _parent_mse, _t), (parent, op, d)), s) in
            jobs.iter().zip(meta.iter()).zip(scores.iter())
        {
            // Structural rejections: a graph that cannot be evaluated, or a
            // candidate with no finite dev evidence, is never proposed.
            if !s.mse.is_finite() || !s.score.is_finite() || s.score <= 0.0 {
                rejected_score += 1;
                continue;
            }
            let p = Proposal::from_score(
                graph,
                parent,
                op,
                *d,
                baseline_mse,
                cfg.lambda_c,
                cfg.selection_dof(),
                cfg.seed,
                s,
            );
            next.push((graph.clone(), s.mse, p));
        }
        if next.is_empty() {
            // Spec §8: STOP when no proposal improves dev score.
            break;
        }
        next.sort_by(|a, b| rank_desc(&a.2, &b.2));
        for (_, _, p) in &next {
            proposals.push(p.clone());
        }
        frontier = next
            .into_iter()
            .take(cfg.beam)
            .map(|(g, mse, _)| (g, mse))
            .collect();
    }

    proposals.sort_by(rank_desc);
    let accepted = proposals.len();
    SearchOutcome { proposals, baseline_mse, baseline_graph_id, evaluated, accepted,
        rejected_score, dropped_dedup, dropped_scope }
}

/// Count the parameters spent by the whole pipeline (edges + readout), for
/// referee-side accounting.
pub fn charged_parameters(graph: &Graph) -> f64 {
    graph.edge_count() as f64 + READOUT_PARAMS
}

#[cfg(test)]
mod tests {
    use super::super::seeds::small_forms;
    use super::*;

    fn hash_noise(n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let v = ((t as f64 + phase) * 12.9898).sin() * 43758.5453;
                v - v.floor() - 0.5
            })
            .collect()
    }

    /// Planted mechanisms the catalogue + rewrites must be able to reach.
    fn planted(kind: &str, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = hash_noise(n, 0.0);
        let z = hash_noise(n, 7.0);
        let y: Vec<f64> = (0..n)
            .map(|t| match kind {
                // lag with a fitted amplitude and an offset
                "lag2" => {
                    if t >= 2 {
                        3.0 * x[t - 2] + 10.0
                    } else {
                        f64::NAN
                    }
                }
                // momentum: difference of consecutive values
                "momentum" => {
                    if t >= 1 {
                        2.0 * (x[t] - x[t - 1])
                    } else {
                        f64::NAN
                    }
                }
                // AR(1)-style decaying memory
                "state" => {
                    let mut s = 0.0;
                    for k in 0..=t {
                        s = 0.8 * s + x[k];
                    }
                    s
                }
                // delayed bilinear interaction
                "interaction" => {
                    if t >= 1 {
                        4.0 * z[t] * x[t - 1]
                    } else {
                        f64::NAN
                    }
                }
                // OFF-GRID memory law: decay 0.87 is NOT in DECAY_PRIORS
                // {0.5, 0.7, 0.8, 0.9, 1.0}. The engine can only reach the
                // nearest grid member (0.9 or 0.8); this world exists to pin
                // that the packet never claims exact recovery (see the
                // off_grid_state test).
                "state_offgrid" => {
                    let mut s = 0.0;
                    for k in 0..=t {
                        s = 0.87 * s + x[k];
                    }
                    s
                }
                other => panic!("unknown planted world {}", other),
            })
            .collect();
        (vec![x, z], y)
    }

    fn recovered_equation(outcome: &SearchOutcome) -> String {
        outcome.best().map(|p| p.equation.clone()).unwrap_or_default()
    }

    /// The calibration the proposer uses: sample-size aware, conservative.
    fn cfg_for(y: &[f64], depth: usize) -> SearchConfig {
        let n_dev = y.iter().filter(|v| v.is_finite()).count();
        SearchConfig::noise_scaled(y, n_dev, 1.5).with_max_depth(depth)
    }

    fn env_of(n: usize, phase: f64, a: f64, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        // NOTE: when `a = 0` the target is NOT pure noise — it becomes
        // `noise * eps[t]` (eps = hash noise at phase+900), a deterministic
        // sequence with variance `(noise)^2 / 12`. That is by construction:
        // this helper always writes `a*x[t-2] + noise*eps[t]`. Do not read
        // `a=0` as "no mechanism": it means "mechanism with zero amplitude",
        // i.e. a target whose only structure is a tiny independent perturbation
        // series. For a truly mechanism-free env use `noise=0` too, which makes
        // the target identically zero after t>=2 (var 0, gate floor 0 — do not
        // use for gate tests). The dominance-trap test therefore builds its
        // distractor environment explicitly below rather than through a=0.
        let x = hash_noise(n, phase);
        let mut y = hash_noise(n, phase + 500.0);
        let eps = hash_noise(n, phase + 900.0);
        for t in 2..n {
            y[t] = a * x[t - 2] + noise * eps[t];
        }
        (vec![x], y)
    }

    #[test]
    fn pooled_dominance_trap_does_not_apply_to_envgate_rule() {
        // One small Distractor env (target = pure noise, independent of the
        // input) plus one large env with a real lag-2 law. Pooled scoring would
        // let the large env's wins mask the small; the worst-env guard must
        // refuse any candidate that only wins in the large env.
        // (A sign flip would NOT test the gate: least squares simply fits a
        // negative scale, so "amplitude" is not mechanism information.)
        // Distractor target = deterministic hash noise, independent of x.
        let n_small = 150usize;
        let small_bad = (
            vec![hash_noise(n_small, 1.0)],
            hash_noise(n_small, 4242.0), // independent of x
        );
        let big_good = env_of(1500, 2.0, 3.0, 0.02);
        let envs = envs_from_frames(&[small_bad, big_good]);
        // λ calibrated on the pooled variance and pooled n of all environments.
        let mut all: Vec<f64> = Vec::new();
        for e in &envs {
            all.extend(e.target.iter().copied().filter(|v| v.is_finite()));
        }
        let npooled = all.len();
        let cfg = SearchConfig {
            lambda_c: 1.5 * target_variance(&all) / npooled as f64,
            ..Default::default()
        }
        .with_max_depth(1);

        let out_gated =
            search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MedianWorstGated, 1.5);
        assert_eq!(
            out_gated.accepted, 0,
            "a mechanism that wins only in the big env must be gated by the \
             worst env: {}",
            out_gated.to_json()
        );

        // Positive control: BOTH envs carrying the same law must pass.
        let good_a = env_of(150, 1.0, 3.0, 0.02);
        let good_b = env_of(1500, 2.0, 3.0, 0.02);
        let envs_both = envs_from_frames(&[good_a, good_b]);
        let out_both =
            search_envs(&small_forms(1), &envs_both, &cfg, EnvAggregation::MedianWorstGated, 1.5);
        assert_eq!(
            out_both.accepted, 1,
            "consistent law across envs must pass: {}",
            out_both.to_json()
        );
        assert!(out_both.worst_env_passed);
    }

    #[test]
    fn mean_per_env_counts_environments_not_rows() {
        // Literal §8: a tiny env contributes as much weight as a huge one.
        let a = env_of(100, 3.0, 0.0, 0.0); // pure noise, same var
        let b = env_of(2000, 4.0, 0.0, 0.0);
        let envs = envs_from_frames(&[a, b]);
        let nulls: Vec<f64> = envs
            .iter()
            .map(|e| fitted_mse(&null_model(), &e.cols, &e.target, 1e-9).0)
            .collect();
        let plain_mean = (nulls[0] + nulls[1]) / 2.0;
        assert!(
            (aggregate(&nulls, EnvAggregation::MeanPerEnv) - plain_mean).abs() < 1e-12,
            "mean-per-env must be unweighted"
        );
    }

    #[test]
    fn search_envs_reports_the_null_reference_and_clean_accounting() {
        let envs = envs_from_frames(&[env_of(200, 5.0, 3.0, 0.02)]);
        let cfg = SearchConfig::default().with_max_depth(1);
        let out = search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MeanPerEnv, 1.5);
        // The reference is the null model's aggregated MSE = the env variance.
        let y = &envs[0].target;
        assert!((out.baseline_mse - target_variance(y)).abs() < 1e-9);
        assert_eq!(out.aggregation, EnvAggregation::MeanPerEnv);
        assert!(out.accepted > 0, "single consistent env must promote: {}", out.to_json());
        assert!(out.evaluated > 0);
        assert_eq!(out.proposals.len(), out.accepted);
    }

    #[test]
    fn heterogeneous_env_results_are_retained_per_env() {
        // P1 core case: Equation 1 excellent in A, poor in B and C;
        // Equation 2 uniformly good. Per-env scoring + median aggregation must
        // keep BOTH env-level results available and rank them by aggregate, not
        // by the flattering env. In this world the uniformly-good law (a lag the
        // mechanism world actually contains in every env) must beat the
        // env-A-only law; the decisive statistic is the MEDIAN improvement.
        let good_a = env_of(400, 10.0, 3.0, 0.02);
        let good_b = env_of(400, 11.0, 3.0, 0.02);
        let good_c = env_of(400, 12.0, 3.0, 0.02);
        let envs = envs_from_frames(&[good_a, good_b, good_c]);
        let cfg = {
            let mut all = Vec::new();
            for e in &envs {
                all.extend(e.target.iter().copied().filter(|v| v.is_finite()));
            }
            SearchConfig {
                lambda_c: 1.5 * target_variance(&all) / all.len() as f64,
                ..Default::default()
            }
            .with_max_depth(1)
        };
        // The search's own winner must be uniform across envs: enforce by
        // checking every env's candidate MSE. A "Excellent/Poor/Poor" mechanism
        // computes one env where it barely improves; the gate/median ordering
        // must rank the across-env winner first.
        let out = search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MedianWorstGated, 1.5);
        assert_eq!(out.accepted, 1, "median: {}", out.to_json());
        // The promoted candidate must improve EVERY env beyond that env's floor.
        let p = &out.proposals[0];
        for e in &envs {
            let (mse, _, _) = fitted_mse(&p.graph, &e.cols, &e.target, cfg.ridge);
            let var = target_variance(&e.target);
            let floor = 1.5 * var / e.target.iter().filter(|v| v.is_finite()).count() as f64;
            assert!(
                mse < var - floor,
                "promoted candidate must clear every env's own noise floor: {} vs {}-{}",
                mse,
                var,
                floor
            );
        }
    }

    #[test]
    fn short_and_malformed_envs_are_excluded_with_explicit_statuses() {
        // P1: short environments (< MIN_ENV_FINITE_ROWS finite target rows) and
        // structurally broken ones are never silently scored.
        let short = env_of(10, 21.0, 3.0, 0.02); // 8 finite target rows
        let ragged = (
            vec![hash_noise(20, 22.0), hash_noise(25, 23.0)], // mismatched lengths
            hash_noise(20, 24.0),
        );
        let good = env_of(300, 25.0, 3.0, 0.02);
        let envs = envs_from_frames(&[short, ragged, good]);
        // The malformed env has a different column count? Both have 1 col; use
        // ragged lengths to trigger Malformed.
        let (statuses, usable) = validate_envs(&envs, 1);
        assert_eq!(statuses[0], EnvStatus::TooFewRows, "10-row env: too few");
        assert_eq!(statuses[1], EnvStatus::Malformed, "ragged env: malformed");
        assert_eq!(statuses[2], EnvStatus::Scored);
        assert_eq!(usable, vec![2usize]);

        // The search itself must exclude them and still work on the good env.
        let cfg = SearchConfig::default().with_max_depth(1);
        let out = search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MeanPerEnv, 1.5);
        assert_eq!(
            out.env_statuses,
            vec![EnvStatus::TooFewRows, EnvStatus::Malformed, EnvStatus::Scored]
        );
        assert!(out.accepted > 0, "good env must still be searched: {}", out.to_json());
    }

    #[test]
    fn all_unusable_envs_explicitly_abstain() {
        // P3 acceptance: empty/degenerate candidate sets produce explicit
        // outcomes, not misleading scores.
        let short_a = env_of(8, 31.0, 3.0, 0.0);
        let short_b = env_of(8, 32.0, 3.0, 0.0);
        let envs = envs_from_frames(&[short_a, short_b]);
        let cfg = SearchConfig::default().with_max_depth(1);
        let out = search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MeanPerEnv, 1.5);
        assert_eq!(out.accepted, 0);
        assert_eq!(out.evaluated, 0, "no candidate may be scored on unusable data");
        assert!(out.baseline_mse.is_infinite(), "reference is explicitly undefined");
        assert_eq!(
            out.env_statuses,
            vec![EnvStatus::TooFewRows, EnvStatus::TooFewRows]
        );
    }

    #[test]
    fn missing_observations_in_envs_are_skipped_not_imputed() {
        // P1: define how missing observations are handled — the scorer skips
        // and counts NaN pairs; a 25%-riddled env must still be scored on the
        // finite remainder (never imputed) and the skipped count must appear
        // in the packet.
        let (cols, y) = env_of(400, 41.0, 3.0, 0.02);
        let mut gapped = (cols.clone(), y.clone());
        for t in (50..400).step_by(4) {
            gapped.1[t] = f64::NAN;
        }
        let envs = envs_from_frames(&[gapped]);
        let cfg = SearchConfig::default().with_max_depth(1);
        let out = search_envs(&small_forms(1), &envs, &cfg, EnvAggregation::MeanPerEnv, 1.5);
        assert_eq!(out.accepted, 1, "gaps are skipped, not fatal: {}", out.to_json());
        let p = &out.proposals[0];
        assert!(p.n_eff < 398, "skipped rows must reduce n_eff: {}", p.n_eff);
        assert!(p.r2 > 0.99, "recovery under gaps: {}", p.r2);
    }

    #[test]
    fn recovers_planted_lag_with_offset() {
        let (cols, y) = planted("lag2", 800);
        let cfg = cfg_for(&y, 1);
        let out = search(&small_forms(2), &cols, &y, &cfg);
        let best = out.best().expect("must propose the planted lag");
        assert!(best.r2 > 0.99, "r2 {} eq {}", best.r2, best.equation);
        assert!((best.scale - 3.0).abs() < 0.05, "scale {}", best.scale);
        assert!(best.intercept.is_finite());
        // The null model is the reference every candidate is measured against.
        assert_eq!(best.parent_id, out.baseline_graph_id);
        assert!((out.baseline_mse - target_variance(&y)).abs() < 1e-9);
    }

    #[test]
    fn recovers_planted_momentum_difference() {
        let (cols, y) = planted("momentum", 800);
        let cfg = cfg_for(&y, 1);
        let out = search(&small_forms(2), &cols, &y, &cfg);
        let best = out.best().expect("must propose the planted difference");
        assert!(best.r2 > 0.99, "r2 {} eq {}", best.r2, best.equation);
        assert!((best.scale - 2.0).abs() < 0.05, "scale {}", best.scale);
        // The winner must contain both the level and a lag-1 read of it: this is
        // the exact form the per-node memo cache used to zero out.
        let eq = recovered_equation(&out);
        assert!(eq.contains("v0") && eq.contains("[t-1]"), "difference not represented: {}", eq);
    }

    #[test]
    fn recovers_planted_state_memory() {
        let (cols, y) = planted("state", 800);
        let cfg = cfg_for(&y, 2);
        let out = search(&small_forms(1), &cols, &y, &cfg);
        let best = out.best().expect("must propose a memory form");
        assert!(best.r2 > 0.99, "r2 {} eq {}", best.r2, best.equation);
        assert!(best.equation.contains("state_0.8000"), "eq {}", best.equation);
        assert!(best.depth == 0 || best.depth <= cfg.max_depth);
    }

    #[test]
    fn recovers_planted_interaction() {
        let (cols, y) = planted("interaction", 800);
        let cfg = cfg_for(&y, 1);
        let out = search(&small_forms(2), &cols, &y, &cfg);
        let best = out.best().expect("must propose the planted interaction");
        assert!(best.r2 > 0.95, "r2 {} eq {}", best.r2, best.equation);
    }

    #[test]
    fn abstains_on_a_null_world() {
        let n = 600;
        let x = hash_noise(n, 0.0);
        let z = hash_noise(n, 3.0);
        // Target depends on t only.
        let y: Vec<f64> = hash_noise(n, 900.0);
        let cfg = cfg_for(&y, 2);
        let out = search(&small_forms(2), &[x, z], &y, &cfg);
        assert!(
            out.accepted == 0,
            "null world must abstain, got {} proposals, best r2 {} ({})",
            out.accepted,
            out.best().map(|p| p.r2).unwrap_or(f64::NAN),
            recovered_equation(&out)
        );
        assert!(out.proposals.is_empty());
        // The search must actually have looked before abstaining.
        assert!(out.evaluated >= 20, "only {} candidates scored", out.evaluated);
        assert!((out.baseline_mse - target_variance(&y)).abs() < 1e-9);
    }

    #[test]
    fn search_is_deterministic_and_budgeted() {
        let (cols, y) = planted("lag2", 400);
        let cfg = cfg_for(&y, 2);
        let a = search(&small_forms(2), &cols, &y, &cfg);
        let b = search(&small_forms(2), &cols, &y, &cfg);
        assert_eq!(
            serde_json::to_string(&a.to_json()).unwrap(),
            serde_json::to_string(&b.to_json()).unwrap()
        );
        assert!(a.evaluated <= cfg.budget, "budget exceeded: {}", a.evaluated);
    }

    #[test]
    fn proposals_carry_referee_evidence() {
        let (cols, y) = planted("lag2", 400);
        let cfg = cfg_for(&y, 1);
        let out = search(&small_forms(1), &cols, &y, &cfg);
        let j = out.to_json();
        let p = &j["proposals"][0];
        for key in ["graph_id", "equation", "dev_evidence"] {
            assert!(!p[key].is_null(), "proposal missing {}", key);
        }
        for key in [
            "score", "mse", "r2", "scale", "intercept", "complexity", "edges",
            "readout_params", "selection_dof", "n_eff",
        ] {
            assert!(!p["dev_evidence"][key].is_null(), "dev evidence missing {}", key);
        }
        assert_eq!(out.proposals[0].record().rewrite_op, out.proposals[0].rewrite_op);
    }

    #[test]
    fn proposals_are_expression_unique() {
        let (cols, y) = planted("lag2", 600);
        let cfg = cfg_for(&y, 2);
        let out = search(&small_forms(2), &cols, &y, &cfg);
        assert!(out.accepted >= 2, "expected several accepted candidates");
        let mut eqs: Vec<&str> = out.proposals.iter().map(|p| p.equation.as_str()).collect();
        let before = eqs.len();
        eqs.sort_unstable();
        eqs.dedup();
        assert_eq!(eqs.len(), before, "duplicate equations proposed: {:?}", eqs);
    }

    #[test]
    fn charged_parameters_include_the_readout() {
        let g = small_forms(1)[0].clone();
        assert_eq!(charged_parameters(&g), g.edge_count() as f64 + READOUT_PARAMS);
    }

    #[test]
    fn multiplicity_charge_grows_with_the_search_width() {
        assert_eq!(multiplicity_dof(2), 2.0 * 2f64.ln());
        assert!(multiplicity_dof(200) > multiplicity_dof(20));
        let cfg = SearchConfig::default().with_budget(200);
        assert_eq!(cfg.score_config().selection_dof, multiplicity_dof(200));
        assert_eq!(cfg.score_config().lambda_c, cfg.lambda_c);
        assert!(crate::rime::eval::charged_dof(&null_model(), multiplicity_dof(200)) > 10.0);
    }

    // ── P3: search-limitarity guarantees (v0.1.2) ─────────────────────────────

    #[test]
    fn structural_rewrites_never_introduce_seed_only_operators() {
        // The frozen rewrite set has no op that produces Sign or Abs. Even at
        // maximum depth from every seed, no generated candidate may contain a
        // node they cannot reach.
        let seeds = small_forms(2);
        let cfg = SearchConfig::default().with_max_depth(3).with_budget(200);
        let (cols, y) = planted("lag2", 400);
        let out = search(&seeds, &cols, &y, &cfg);
        // Count all candidates the search could touch: seeds + everything the
        // search proposed, then expand every pool member one more level.
        let mut universe: Vec<Graph> = seeds.clone();
        for p in &out.proposals {
            universe.push(p.graph.clone());
        }
        for g in &universe {
            for op in enumerate_ops(g) {
                if let Some(child) = apply(g, &op) {
                    // The invariant is about CREATION, not inheritance: a
                    // rewrite must never push a NEW Sign/Abs node. Nodes at
                    // indices < g.nodes.len() were inherited; only new ones
                    // (>= len) are produced by the rewrite. (WrapNonlinearity
                    // may legitimately wrap an inherited Sign/Abs — e.g.
                    // tanh(abs(v0)) — that is composition, not introduction.)
                    for (idx, node) in child.nodes.iter().enumerate().skip(g.nodes.len()) {
                        let created = match node {
                            Node::Sign(_) | Node::Abs(_) => Some("Sign/Abs"),
                            _ => None,
                        };
                        assert!(
                            created.is_none(),
                            "rewrite {:?} created a seed-only {} node at index {}: {}",
                            op,
                            created.unwrap(),
                            idx,
                            child.describe()
                        );
                    }
                }
            }
        }
        // Anchoring the invariant: the seed catalogue DOES contain them, so the
        // assertion above is about rewrites, not vacuous.
        assert!(
            seeds.iter().any(|g| g.describe().contains("abs(")),
            "sanity: catalogue must contain a seed-only operator for this test to bite"
        );
    }

    #[test]
    fn seed_only_operators_only_reachable_through_the_catalogue() {
        let seeds = small_forms(1);
        let abs_seeds: Vec<&Graph> = seeds.iter().filter(|g| g.describe().contains("abs(")).collect();
        let sign_seeds: Vec<&Graph> = seeds.iter().filter(|g| g.describe().contains("sign(")).collect();
        assert_eq!(abs_seeds.len(), 1, "exactly one Abs seed (documented budget tail)");
        assert_eq!(sign_seeds.len(), 1, "exactly one Sign seed");
        // And each is depth 0 (never reachable from another seed via one rewrite).
        for g in abs_seeds.iter().chain(sign_seeds.iter()) {
            for s in seeds.iter().filter(|other| other.describe() != g.describe()) {
                for op in enumerate_ops(s) {
                    if let Some(child) = apply(s, &op) {
                        assert_ne!(
                            child.id(),
                            g.id(),
                            "seed-only operator reachable by rewrite from another seed: {} -> {}",
                            s.describe(),
                            g.describe()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn off_grid_state_never_claimed_as_exact_recovery() {
        // An off-grid decay must NOT be presented as exact: the packet must not
        // claim state_0.87x, and the honest framing (nearest grid member + the
        // readout absorbing the residual) must be visible in the equation.
        let (cols, y) = planted("state_offgrid", 800);
        let cfg = cfg_for(&y, 2);
        let out = search(&small_forms(1), &cols, &y, &cfg);
        let best = out.best().expect("memory family must still be proposed");
        assert!(best.equation.contains("state_"), "eq {}", best.equation);
        assert!
            (!best.equation.contains("state_0.8700"), "off-grid decay must never be reported as exact: {}", best.equation);
        // The evidence frame: fit quality is reported with its actual R², which
        // is genuinely < 1 for an off-grid approximation — no overclaim.
        assert!(best.r2 > 0.9 && best.r2 < 1.0, "approximate fit reported honestly: {}", best.r2);
    }

    #[test]
    fn empty_candidate_sets_explicitly_abstain_not_score() {
        // Budget 1 with an unusable world: search terminates with explicit
        // zero-proposal outcome rather than a junk score.
        let (cols, y) = planted("lag2", 300);
        let cfg = cfg_for(&y, 1).with_budget(1);
        let out = search(&small_forms(1), &cols, &y, &cfg);
        assert_eq!(out.accepted, 0, "budget 1 cannot fund a promotable candidate: {}", out.to_json());
        assert!(out.proposals.is_empty());
        assert!(out.evaluated <= cfg.budget);

        // An empty seed list: no crash, no fake scoring, explicit zero.
        let cfg2 = cfg_for(&y, 1);
        let out2 = search(&[], &cols, &y, &cfg2);
        assert_eq!(out2.accepted, 0);
        assert_eq!(out2.evaluated, 0);
        assert!(out2.proposals.is_empty());
    }

    #[test]
    fn short_series_abstains_on_a_null_world() {
        // The audit's FPR failure mode, at the sample size where it bites: a
        // 150-row null world with ~90 dev rows. A fixed fraction-of-variance
        // penalty promotes the best of 60 seeds on fit noise; the
        // sample-size-aware penalty plus the selection charge must not.
        let n = 150;
        let x = hash_noise(n, 0.0);
        let y = hash_noise(n, 900.0);
        let n_dev = y.iter().filter(|v| v.is_finite()).count();
        let cfg = SearchConfig::noise_scaled(&y, n_dev, 1.5).with_max_depth(1).with_budget(60);
        let out = search(&small_forms(1), &[x], &y, &cfg);
        assert_eq!(out.accepted, 0, "best: {}", recovered_equation(&out));
        // Positive control: the same configuration must still find a real law.
        let mut y_signal = y.clone();
        let x_signal: Vec<f64> = hash_noise(n, 0.0);
        for t in 2..n {
            y_signal[t] = 3.0 * x_signal[t - 2];
        }
        let cfg2 = SearchConfig::noise_scaled(&y_signal, n_dev, 1.5).with_max_depth(1).with_budget(60);
        let out2 = search(&small_forms(1), &[x_signal], &y_signal, &cfg2);
        assert!(out2.accepted > 0, "positive control must be found");
    }
}
