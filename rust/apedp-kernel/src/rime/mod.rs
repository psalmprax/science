//! RIME Stage-1 core: typed mechanism graphs, frozen rewrites, batch
//! evaluation, immutable records. See docs/RIME_V01_MASTER_SPEC.md.
//!
//! `search` and `seeds` are Stage-2 scaffolding built *from* the frozen
//! primitives (spec §7/§8); they add no rewrites and change no node type.
pub mod additive;
pub mod d3;
pub mod eval;
pub mod graph;
pub mod record;
pub mod residual;
pub mod rewrite;
pub mod search;
// NOTE (kernel restore): `referee` targets the v0.1-scaffolding search API
// (SearchPacket/SearchConfig) and does not compile against the restored
// v0.1.2 search API. Nothing imports it (Python referee is APEDP-X side).
// File kept on disk; re-attach only with a port to the v0.1.2 packet types.
// pub mod referee;
pub mod coupled_system;
pub mod diff_pade;
pub mod enumerate;
pub mod high_dim_pde;
pub mod implicit;
pub mod invariant_transfer;
pub mod koopman;
pub mod lie_symmetry;
pub mod number_theory;
pub mod pde;
pub mod seeds;
pub mod total_variation;
