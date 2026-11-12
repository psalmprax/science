//! The circuit IR this crate executes: the same canonical JSON
//! `apedp_dyn.vqc.circuit.Circuit.to_json()` already emits.
//!
//! There is one circuit format, not two. Rust does not get its own builder, its
//! own gate naming or its own schema version: it parses the Python canonical
//! serialisation and refuses everything it cannot represent, by name. That is
//! the "one canonical circuit specification" rule from the review this module
//! answers, implemented as a parser rather than as a convention.
//!
//! Refusal is deliberately *late* for gates: a non-Clifford gate lives in the
//! IR and is refused when it executes, which is when `stabilizer.py` refuses it
//! too. A conditional gate whose condition is false is skipped by both, so the
//! two implementations agree even in that corner.

use serde_json::Value;

use crate::pauli::Clifford;
use crate::stabilizer::VqcError;

/// One operation, in the order the circuit declares it.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Gate {
        /// `None` for a gate this tier cannot execute; the name is kept so the
        /// refusal can name it.
        gate: Option<Clifford>,
        name: String,
        targets: Vec<usize>,
        params: Vec<f64>,
        condition: Vec<(usize, bool)>,
    },
    Measure {
        qubit: usize,
        bit: usize,
        condition: Vec<(usize, bool)>,
    },
    Reset {
        qubit: usize,
        condition: Vec<(usize, bool)>,
    },
    Noise {
        kind: String,
        qubit: usize,
        probability: f64,
        condition: Vec<(usize, bool)>,
    },
}

impl Op {
    /// The condition, empty when the operation is unconditional.
    pub fn condition(&self) -> &[(usize, bool)] {
        match self {
            Op::Gate { condition, .. }
            | Op::Measure { condition, .. }
            | Op::Reset { condition, .. }
            | Op::Noise { condition, .. } => condition,
        }
    }
}

/// A parsed circuit: what the Python side calls a `Circuit`.
#[derive(Clone, Debug)]
pub struct CircuitIr {
    pub num_qubits: usize,
    pub num_bits: usize,
    pub ancillas: Vec<usize>,
    pub ops: Vec<Op>,
}

fn value_error(message: impl Into<String>) -> VqcError {
    VqcError::Value(message.into())
}

fn unsupported(message: impl Into<String>) -> VqcError {
    VqcError::Unsupported(message.into())
}

fn integer(entry: &Value, key: &str) -> Result<usize, VqcError> {
    let raw = entry
        .get(key)
        .ok_or_else(|| value_error(format!("operation is missing {key:?}")))?;
    let parsed = raw
        .as_i64()
        .ok_or_else(|| value_error(format!("{key} is not an integer: {raw}")))?;
    usize::try_from(parsed).map_err(|_| value_error(format!("{key} is negative: {parsed}")))
}

/// Python's canonical form is a list of `{"bit": i, "value": bool}` objects, not
/// a list of pairs. Reading the *format the writer emits* is the whole point of
/// this parser; a parity test caught the first version of it guessing.
fn condition(entry: &Value) -> Result<Vec<(usize, bool)>, VqcError> {
    let Some(list) = entry.get("condition") else {
        return Ok(Vec::new());
    };
    let Some(items) = list.as_array() else {
        return Ok(Vec::new());
    };
    let mut pairs = Vec::with_capacity(items.len());
    for item in items {
        let pair = item
            .as_object()
            .ok_or_else(|| value_error(format!("condition entry {item} is not an object")))?;
        let bit = pair
            .get("bit")
            .and_then(Value::as_u64)
            .ok_or_else(|| value_error(format!("condition entry {item} has no integer bit")))?;
        let wanted = pair
            .get("value")
            .and_then(Value::as_bool)
            .ok_or_else(|| value_error(format!("condition entry {item} has no boolean value")))?;
        pairs.push((bit as usize, wanted));
    }
    Ok(pairs)
}

/// Parse the canonical circuit JSON.
pub fn parse(text: &str) -> Result<CircuitIr, VqcError> {
    let payload: Value =
        serde_json::from_str(text).map_err(|error| value_error(format!("not JSON: {error}")))?;
    if !payload.is_object() {
        return Err(value_error("expected an object at the top level"));
    }
    let num_qubits = integer(&payload, "num_qubits")?;
    if num_qubits < 1 {
        return Err(value_error(format!(
            "num_qubits must be positive, got {num_qubits}"
        )));
    }
    let num_bits = match payload.get("num_bits") {
        None | Some(Value::Null) => 0,
        Some(_) => integer(&payload, "num_bits")?,
    };
    let mut ancillas = Vec::new();
    if let Some(list) = payload.get("ancillas").and_then(Value::as_array) {
        for entry in list {
            let qubit = entry
                .as_u64()
                .ok_or_else(|| value_error(format!("ancilla {entry} is not an integer")))?;
            ancillas.push(qubit as usize);
        }
    }

    let mut ops = Vec::new();
    let Some(entries) = payload.get("ops").and_then(Value::as_array) else {
        return Ok(CircuitIr {
            num_qubits,
            num_bits,
            ancillas,
            ops,
        });
    };
    for entry in entries {
        let tag = entry
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| value_error(format!("operation {entry} has no tag")))?;
        let condition = condition(entry)?;
        match tag {
            "gate" => {
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| value_error("gate is missing its name"))?
                    .to_string();
                let mut targets = Vec::new();
                if let Some(list) = entry.get("targets").and_then(Value::as_array) {
                    for target in list {
                        let qubit = target.as_u64().ok_or_else(|| {
                            value_error(format!("target {target} is not an integer"))
                        })?;
                        targets.push(qubit as usize);
                    }
                }
                let mut params = Vec::new();
                if let Some(list) = entry.get("params").and_then(Value::as_array) {
                    for param in list {
                        let value = param.as_f64().ok_or_else(|| {
                            value_error(format!("parameter {param} is not a number"))
                        })?;
                        params.push(value);
                    }
                }
                ops.push(Op::Gate {
                    gate: Clifford::from_name(&name),
                    name,
                    targets,
                    params,
                    condition,
                });
            }
            "measure" => ops.push(Op::Measure {
                qubit: integer(entry, "qubit")?,
                bit: integer(entry, "bit")?,
                condition,
            }),
            "reset" => ops.push(Op::Reset {
                qubit: integer(entry, "qubit")?,
                condition,
            }),
            "noise" => ops.push(Op::Noise {
                kind: entry
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                qubit: integer(entry, "qubit")?,
                probability: entry
                    .get("probability")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                condition,
            }),
            other => return Err(value_error(format!("unknown operation tag {other:?}"))),
        }
    }

    Ok(CircuitIr {
        num_qubits,
        num_bits,
        ancillas,
        ops,
    })
}

/// Whether a condition is satisfied by the bits collected so far.
pub fn condition_holds(condition: &[(usize, bool)], bits: &[bool]) -> bool {
    condition
        .iter()
        .all(|(bit, wanted)| bits.get(*bit).copied().unwrap_or(false) == *wanted)
}

/// Resolve a gate to something executable, refusing by name the way
/// `stabilizer.py::_clifford_spec` does.
pub fn executable(
    gate: &Option<Clifford>,
    name: &str,
    params: &[f64],
) -> Result<Clifford, VqcError> {
    let Some(gate) = gate else {
        return Err(unsupported(format!("unknown gate {name:?}")));
    };
    if !params.is_empty() {
        return Err(unsupported(format!(
            "{name:?} takes parameters; a parameterised rotation is not Clifford"
        )));
    }
    Ok(*gate)
}

/// The refusal a `Noise` operation earns from this tier.
pub fn refuse_noise(kind: &str) -> VqcError {
    unsupported(format!(
        "noise op {kind:?} turns a pure stabilizer state into a mixture; use the statevector tier"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::Gate1;

    const CIRCUIT: &str = r#"{"ancillas":[2],"metadata":{},"num_bits":2,"num_qubits":3,
        "ops":[
        {"condition":null,"name":"h","op":"gate","params":[],"targets":[0]},
        {"condition":null,"name":"cx","op":"gate","params":[],"targets":[0,1]},
        {"condition":[{"bit":1,"value":true}],"name":"x","op":"gate","params":[],"targets":[2]},
        {"bit":0,"condition":null,"op":"measure","qubit":0},
        {"condition":null,"op":"reset","qubit":1}
        ],
        "schema_version":1}"#;

    #[test]
    fn the_canonical_python_json_parses_into_ops() {
        let circuit = parse(CIRCUIT).unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.num_bits, 2);
        assert_eq!(circuit.ancillas, vec![2]);
        assert_eq!(circuit.ops.len(), 5);
        match &circuit.ops[0] {
            Op::Gate { gate, targets, .. } => {
                assert_eq!(*gate, Some(Clifford::One(Gate1::H)));
                assert_eq!(targets, &vec![0]);
            }
            other => panic!("{other:?}"),
        }
        match &circuit.ops[2] {
            Op::Gate { condition, .. } => assert_eq!(condition, &vec![(1, true)]),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            circuit.ops[3],
            Op::Measure {
                qubit: 0,
                bit: 0,
                ..
            }
        ));
        assert!(matches!(circuit.ops[4], Op::Reset { qubit: 1, .. }));
    }

    #[test]
    fn a_non_clifford_gate_is_parsed_and_refused_when_it_executes() {
        let text = r#"{"num_qubits":1,"num_bits":0,"ops":[
            {"condition":null,"name":"rx","op":"gate","params":[0.5],"targets":[0]}]}"#;
        let circuit = parse(text).unwrap();
        match &circuit.ops[0] {
            Op::Gate {
                gate, name, params, ..
            } => {
                assert_eq!(*gate, None);
                let error = executable(gate, name, params).unwrap_err();
                assert!(error.message().contains("unknown gate"), "{error:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_parameterised_rotation_that_exists_is_refused_as_not_clifford() {
        let error = executable(&None, "rx", &[0.5]).unwrap_err();
        assert!(error.message().contains("rx"));
    }

    #[test]
    fn conditions_are_evaluated_against_collected_bits() {
        let bits = vec![false, true];
        assert!(condition_holds(&[(1, true)], &bits));
        assert!(!condition_holds(&[(0, true)], &bits));
        // An unmeasured bit reads as false, which is what the Python executor does.
        assert!(condition_holds(&[(7, false)], &bits));
    }
}
