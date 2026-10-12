//! RIME Stage-1 core: typed mechanism graphs, frozen rewrites, batch
//! evaluation, immutable records. See docs/RIME_V01_MASTER_SPEC.md.
//!
//! `search` and `seeds` are Stage-2 scaffolding built *from* the frozen
//! primitives (spec §7/§8); they add no rewrites and change no node type.
pub mod eval;
pub mod graph;
pub mod record;
pub mod rewrite;
pub mod search;
pub mod seeds;
