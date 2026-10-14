// D3 disagreement monitor – shadow‑mode logging for RIME candidates.
//! This module records candidate information after each search run and can be
//! read by a downstream process to compute disagreement metrics against other
//! proposers (e.g. PySR).  It does not alter search or promotion decisions.

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::io::Write;
use serde::{Serialize, Deserialize};
use std::path::Path;

/// Path helper: creates a timestamp‑named log file under the given directory.
/// Returns the full path to the log file.
pub fn make_log_path(base_dir: &Path) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = format!("d3_candidates_{}.log", ts);
    base_dir.join(name)
}

/// A record of one candidate produced by a search run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateRecord {
    /// Unique identifier for the graph (hash of graph_id + seed).
    pub graph_id: String,
    /// Final MSE of the fitted readout.
    pub residual_mse: f64,
    /// Timestamp of when the candidate was produced.
    pub timestamp: std::time::SystemTime,
}

// Append a `CandidateRecord` to the run‑specific log file.
// The file is created if it does not exist and appended to otherwise.
pub fn log_candidate(log_path: &Path, record: &CandidateRecord) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        if let Ok(line) = serde_json::to_string(record) {
            let _ = writeln!(file, "{}", line);
        }
    }
}

// Convenience: log a candidate directly using a path constructed via
/// make_log_path.
pub fn log_candidate_from_run(run_dir: &Path, graph_id: &str, residual_mse: f64) {
    let path = make_log_path(run_dir);
    let record = CandidateRecord {
        graph_id: graph_id.to_string(),
        residual_mse,
        timestamp: std::time::SystemTime::now(),
    };
    log_candidate(&path, &record);
}