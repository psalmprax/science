//! Native ZX-Calculus and Clifford+T circuit optimization.
//!
//! Evaluates and optimizes quantum circuits via:
//! - Spider fusion & phase addition modulo 8 (T + T^dagger -> I, T + T -> S, T^4 -> Z, T^8 -> I).
//! - Commuting Z-phase gates through CX controls and CZ ports.
//! - Commuting X-phase gates through CX targets.
//! - Two-qubit gate adjacent cancellations (CX . CX -> I, CZ . CZ -> I, SWAP . SWAP -> I).
//! - Single-qubit self-inverse cancellations (H . H -> I, X . X -> I, Z . Z -> I).

use serde_json::{json, Value};

use crate::stabilizer::VqcError;

#[derive(Clone, Debug, PartialEq)]
pub struct GateOp {
    pub name: String,
    pub targets: Vec<usize>,
    pub params: Vec<f64>,
    pub condition: Vec<(usize, bool)>,
}

fn is_z_phase(name: &str) -> bool {
    matches!(name, "t" | "tdg" | "s" | "sdg" | "z")
}

fn phase_eighths(name: &str) -> Option<u8> {
    match name {
        "t" => Some(1),
        "s" => Some(2),
        "z" => Some(4),
        "sdg" => Some(6),
        "tdg" => Some(7),
        _ => None,
    }
}

fn eighths_to_gates(phase: u8) -> Vec<&'static str> {
    match phase % 8 {
        0 => vec![],
        1 => vec!["t"],
        2 => vec!["s"],
        3 => vec!["s", "t"],
        4 => vec!["z"],
        5 => vec!["z", "t"],
        6 => vec!["sdg"],
        7 => vec!["tdg"],
        _ => unreachable!(),
    }
}

fn can_commute(a: &GateOp, b: &GateOp) -> bool {
    let disjoint = !a.targets.iter().any(|t| b.targets.contains(t));
    if disjoint {
        return true;
    }
    let name_a = a.name.to_lowercase();
    let name_b = b.name.to_lowercase();
    let is_z_a = is_z_phase(&name_a);
    let is_z_b = is_z_phase(&name_b);

    // Both Z-phase on the same target qubit
    if is_z_a && is_z_b && a.targets == b.targets {
        return true;
    }
    // Z-phase on control qubit of CX
    if is_z_a && name_b == "cx" && a.targets.len() == 1 && b.targets.len() >= 2 && a.targets[0] == b.targets[0] {
        return true;
    }
    if is_z_b && name_a == "cx" && b.targets.len() == 1 && a.targets.len() >= 2 && b.targets[0] == a.targets[0] {
        return true;
    }
    // Z-phase on either qubit of CZ
    if is_z_a && name_b == "cz" && a.targets.len() == 1 && b.targets.contains(&a.targets[0]) {
        return true;
    }
    if is_z_b && name_a == "cz" && b.targets.len() == 1 && a.targets.contains(&b.targets[0]) {
        return true;
    }
    // X on target of CX
    if name_a == "x" && name_b == "cx" && a.targets.len() == 1 && b.targets.len() >= 2 && a.targets[0] == b.targets[1] {
        return true;
    }
    if name_b == "x" && name_a == "cx" && b.targets.len() == 1 && a.targets.len() >= 2 && b.targets[0] == a.targets[1] {
        return true;
    }

    false
}

/// Optimize a circuit's operation sequence using ZX spider fusion and commutation.
pub fn optimize_circuit_json(circuit_json: &str, max_iterations: usize) -> Result<String, VqcError> {
    let payload: Value = serde_json::from_str(circuit_json)
        .map_err(|e| VqcError::Value(format!("invalid circuit JSON: {e}")))?;

    let num_qubits = payload.get("num_qubits").and_then(Value::as_u64).unwrap_or(1) as usize;
    let num_bits = payload.get("num_bits").and_then(Value::as_u64).unwrap_or(0) as usize;
    let ancillas = payload.get("ancillas").cloned().unwrap_or_else(|| json!([]));
    let metadata = payload.get("metadata").cloned().unwrap_or_else(|| json!({}));
    let schema_version = payload.get("schema_version").and_then(Value::as_u64).unwrap_or(1);

    let Some(raw_ops) = payload.get("ops").and_then(Value::as_array) else {
        return Ok(circuit_json.to_string());
    };

    let mut current_ops: Vec<GateOp> = Vec::new();
    for entry in raw_ops {
        let op_type = entry.get("op").and_then(Value::as_str).unwrap_or("");
        if op_type != "gate" {
            // Non-gate ops (measure, reset, noise) preserved as dummy ops
            current_ops.push(GateOp {
                name: entry.get("name").and_then(Value::as_str).unwrap_or(op_type).to_string(),
                targets: entry.get("targets").and_then(Value::as_array).map(|arr| {
                    arr.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()
                }).unwrap_or_default(),
                params: vec![],
                condition: vec![],
            });
            continue;
        }

        let name = entry.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        let targets: Vec<usize> = entry.get("targets").and_then(Value::as_array).map(|arr| {
            arr.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()
        }).unwrap_or_default();

        let params: Vec<f64> = entry.get("params").and_then(Value::as_array).map(|arr| {
            arr.iter().filter_map(Value::as_f64).collect()
        }).unwrap_or_default();

        current_ops.push(GateOp {
            name,
            targets,
            params,
            condition: vec![],
        });
    }

    for _ in 0..max_iterations {
        let mut changed = false;

        // Step 1: Commutation / bubbling
        let ops_len = current_ops.len();
        let mut j = 0;
        while j < ops_len {
            let op_a = &current_ops[j];
            let name_a = op_a.name.to_lowercase();
            let is_z_a = is_z_phase(&name_a);

            for k in (j + 1)..std::cmp::min(ops_len, j + 24) {
                let op_b = &current_ops[k];

                // Check commutation between j and k
                let mut can_b_commute = true;
                for m in ((j + 1)..=k - 1).rev() {
                    if !can_commute(&current_ops[m], op_b) {
                        can_b_commute = false;
                        break;
                    }
                }

                if !can_b_commute {
                    let mut can_a_commute = true;
                    for m in (j + 1)..k {
                        if !can_commute(op_a, &current_ops[m]) {
                            can_a_commute = false;
                            break;
                        }
                    }
                    if !can_a_commute {
                        continue;
                    }
                }

                let name_b = op_b.name.to_lowercase();
                let is_z_b = is_z_phase(&name_b);

                // Case A: Two Z-phase gates on same target qubit
                if is_z_a && is_z_b && op_a.targets == op_b.targets {
                    let commuted = current_ops.remove(k);
                    current_ops.insert(j + 1, commuted);
                    changed = true;
                    break;
                }

                // Case B: Identical 2-qubit gates
                if matches!(name_a.as_str(), "cx" | "cz" | "swap") && name_a == name_b {
                    let match_targets = if matches!(name_a.as_str(), "cz" | "swap") {
                        let mut s_a = op_a.targets.clone();
                        let mut s_b = op_b.targets.clone();
                        s_a.sort();
                        s_b.sort();
                        s_a == s_b
                    } else {
                        op_a.targets == op_b.targets
                    };
                    if match_targets {
                        let commuted = current_ops.remove(k);
                        current_ops.insert(j + 1, commuted);
                        changed = true;
                        break;
                    }
                }

                // Case C: Adjacent self-inverses
                if matches!(name_a.as_str(), "h" | "x" | "z") && name_a == name_b && op_a.targets == op_b.targets {
                    let commuted = current_ops.remove(k);
                    current_ops.insert(j + 1, commuted);
                    changed = true;
                    break;
                }
            }
            j += 1;
        }

        // Step 2: Linear fusion & cancellations
        let mut new_ops = Vec::new();
        let mut i = 0;
        let ops_len = current_ops.len();
        while i < ops_len {
            let op = &current_ops[i];
            let name = op.name.to_lowercase();

            if i + 1 < ops_len {
                let next_op = &current_ops[i + 1];
                let next_name = next_op.name.to_lowercase();

                // Rule 1: Spider fusion of adjacent Z-phase gates on same qubit
                if is_z_phase(&name) && is_z_phase(&next_name) && op.targets == next_op.targets {
                    if let (Some(p1), Some(p2)) = (phase_eighths(&name), phase_eighths(&next_name)) {
                        let fused_phase = (p1 + p2) % 8;
                        let replacement = eighths_to_gates(fused_phase);
                        for r_name in replacement {
                            new_ops.push(GateOp {
                                name: r_name.to_string(),
                                targets: op.targets.clone(),
                                params: vec![],
                                condition: vec![],
                            });
                        }
                        i += 2;
                        changed = true;
                        continue;
                    }
                }

                // Rule 2: Two-qubit gate cancellation
                if matches!(name.as_str(), "cx" | "cz" | "swap") && name == next_name {
                    let match_targets = if matches!(name.as_str(), "cz" | "swap") {
                        let mut s_a = op.targets.clone();
                        let mut s_b = next_op.targets.clone();
                        s_a.sort();
                        s_b.sort();
                        s_a == s_b
                    } else {
                        op.targets == next_op.targets
                    };
                    if match_targets {
                        i += 2;
                        changed = true;
                        continue;
                    }
                }

                // Rule 3: Single-qubit self-inverse cancellation
                if matches!(name.as_str(), "h" | "x" | "z") && name == next_name && op.targets == next_op.targets {
                    i += 2;
                    changed = true;
                    continue;
                }
            }

            new_ops.push(op.clone());
            i += 1;
        }

        current_ops = new_ops;
        if !changed {
            break;
        }
    }

    // Reconstruct canonical JSON
    let serialized_ops: Vec<Value> = current_ops
        .into_iter()
        .map(|op| {
            json!({
                "op": "gate",
                "name": op.name,
                "targets": op.targets,
                "params": op.params,
                "condition": []
            })
        })
        .collect();

    let output = json!({
        "schema_version": schema_version,
        "num_qubits": num_qubits,
        "num_bits": num_bits,
        "ancillas": ancillas,
        "metadata": metadata,
        "ops": serialized_ops
    });

    Ok(output.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_zx_t_fusion_and_cx_cancellation() {
        let input_json = json!({
            "schema_version": 1,
            "num_qubits": 2,
            "num_bits": 0,
            "ops": [
                {"op": "gate", "name": "t", "targets": [0]},
                {"op": "gate", "name": "cx", "targets": [0, 1]},
                {"op": "gate", "name": "t", "targets": [0]},
                {"op": "gate", "name": "cx", "targets": [0, 1]},
                {"op": "gate", "name": "tdg", "targets": [0]}
            ]
        }).to_string();

        let opt_json = optimize_circuit_json(&input_json, 12).unwrap();
        let val: Value = serde_json::from_str(&opt_json).unwrap();
        let ops = val.get("ops").unwrap().as_array().unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].get("name").unwrap().as_str().unwrap(), "t");
        assert_eq!(ops[0].get("targets").unwrap().as_array().unwrap(), &vec![json!(0)]);
    }

    #[test]
    fn test_native_zx_t_t_to_s() {
        let input_json = json!({
            "schema_version": 1,
            "num_qubits": 1,
            "num_bits": 0,
            "ops": [
                {"op": "gate", "name": "t", "targets": [0]},
                {"op": "gate", "name": "t", "targets": [0]}
            ]
        }).to_string();

        let opt_json = optimize_circuit_json(&input_json, 12).unwrap();
        let val: Value = serde_json::from_str(&opt_json).unwrap();
        let ops = val.get("ops").unwrap().as_array().unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].get("name").unwrap().as_str().unwrap(), "s");
    }

    #[test]
    fn test_native_zx_t_cz_commutation() {
        let input_json = json!({
            "schema_version": 1,
            "num_qubits": 2,
            "num_bits": 0,
            "ops": [
                {"op": "gate", "name": "t", "targets": [0]},
                {"op": "gate", "name": "cz", "targets": [0, 1]},
                {"op": "gate", "name": "t", "targets": [0]},
                {"op": "gate", "name": "cz", "targets": [0, 1]}
            ]
        }).to_string();

        let opt_json = optimize_circuit_json(&input_json, 12).unwrap();
        let val: Value = serde_json::from_str(&opt_json).unwrap();
        let ops = val.get("ops").unwrap().as_array().unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].get("name").unwrap().as_str().unwrap(), "s");
    }
}
