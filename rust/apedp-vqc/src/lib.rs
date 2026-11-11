//! The `apedp_vqc` Python extension module: the native Clifford / stabilizer
//! execution core, called Rust-first by `apedp_dyn/vqc/native.py`.
//!
//! The surface is deliberately small and JSON-shaped. Every structured answer is
//! one string the Python bridge parses, so the FFI contract is a schema rather
//! than a tuple order, and a change cannot silently reorder somebody's
//! arguments. The scalars stay scalars.
//!
//! Nothing here decides a verdict. A Rust refusal (`NotImplementedError`) means
//! "this execution path could not answer", never "this candidate is bad": the
//! caller falls back to `apedp_dyn/vqc/stabilizer.py` and, in the equivalence
//! case, Python referees whatever the fast path produced.

pub mod bits;
pub mod equivalence;
pub mod ir;
pub mod pauli;
pub mod runtime;
pub mod stabilizer;

#[cfg(feature = "python")]
mod bindings {
    use pyo3::exceptions::{PyNotImplementedError, PyValueError};
    use pyo3::prelude::*;
    use serde_json::json;

    use crate::equivalence;
    use crate::ir;
    use crate::pauli::{self, LABEL_CHARS};
    use crate::runtime;
    use crate::stabilizer::{Stabilizer, VqcError};

    impl From<VqcError> for PyErr {
        fn from(error: VqcError) -> PyErr {
            match error {
                VqcError::Value(message) => PyValueError::new_err(message),
                VqcError::Unsupported(message) => PyNotImplementedError::new_err(message),
            }
        }
    }

    fn parse_circuit(text: &str) -> PyResult<ir::CircuitIr> {
        Ok(ir::parse(text)?)
    }

    fn parse_labels(text: &str, num_qubits: usize) -> PyResult<Vec<u8>> {
        if text.len() != num_qubits {
            return Err(PyValueError::new_err(format!(
                "expected {num_qubits} labels, got {}",
                text.len()
            )));
        }
        text.chars()
            .map(|character| {
                LABEL_CHARS
                    .iter()
                    .position(|candidate| *candidate == character)
                    .map(|index| index as u8)
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("{character:?} is not one of I, X, Y, Z"))
                    })
            })
            .collect()
    }

    /// Execute a circuit: counts, per-shot bitstrings and ancilla excitations.
    #[pyfunction]
    #[pyo3(signature = (circuit_json, shots, seed = 0))]
    fn run_report(circuit_json: &str, shots: usize, seed: u64) -> PyResult<String> {
        let circuit = parse_circuit(circuit_json)?;
        let report = runtime::run(&circuit, shots, seed)?;
        let samples: Vec<String> = report
            .samples
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .map(|bit| if *bit { '1' } else { '0' })
                    .collect()
            })
            .collect();
        Ok(json!({
            "backend": report.backend,
            "seed": report.seed,
            "counts": report.counts,
            "samples": samples,
            "ancilla_excitation": report.ancilla_excitation,
        })
        .to_string())
    }

    /// The `2n` signed Pauli images, as `[name, labels, phase]` entries.
    #[pyfunction]
    fn equivalence_images(circuit_json: &str) -> PyResult<String> {
        let circuit = parse_circuit(circuit_json)?;
        let images = equivalence::images(&circuit.ops, circuit.num_qubits)?;
        let entries: Vec<serde_json::Value> = images
            .iter()
            .enumerate()
            .map(|(index, (labels, phase))| {
                if labels.len() != circuit.num_qubits {
                    return json!([]);
                }
                json!([
                    equivalence::image_name(index),
                    labels
                        .iter()
                        .map(|label| LABEL_CHARS[*label as usize])
                        .collect::<String>(),
                    phase
                ])
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

    /// The exact Z-basis distribution of the state a circuit prepares.
    #[pyfunction]
    fn distribution(circuit_json: &str) -> PyResult<String> {
        let circuit = parse_circuit(circuit_json)?;
        let mut state = Stabilizer::new(circuit.num_qubits)?;
        let mut rng = crate::stabilizer::Rng::new(circuit.num_qubits as u64);
        for op in &circuit.ops {
            match op {
                ir::Op::Gate {
                    gate,
                    name,
                    targets,
                    params,
                    condition,
                } => {
                    if condition.is_empty() {
                        let gate = ir::executable(gate, name, params)?;
                        state.apply(gate, targets)?;
                    }
                }
                ir::Op::Reset { qubit, condition } => {
                    if condition.is_empty() {
                        state.reset(*qubit, &mut rng)?;
                    }
                }
                // A measured circuit is a sampler, not a state; ask for the state
                // with a circuit that does not measure.
                ir::Op::Measure { .. } => {
                    return Err(VqcError::Unsupported(
                        "the circuit measures, so it prepares a distribution over outcomes \
                         rather than a state; use run_report"
                            .to_string(),
                    )
                    .into())
                }
                ir::Op::Noise { kind, .. } => return Err(ir::refuse_noise(kind).into()),
            }
        }
        let outcomes: Vec<serde_json::Value> = state
            .distribution()?
            .iter()
            .map(|(bits, probability)| {
                json!([
                    bits.iter()
                        .map(|bit| if *bit { '1' } else { '0' })
                        .collect::<String>(),
                    probability
                ])
            })
            .collect();
        Ok(serde_json::Value::Array(outcomes).to_string())
    }

    /// The generators of the state a (measurement-free) circuit prepares.
    #[pyfunction]
    fn generators(circuit_json: &str) -> PyResult<String> {
        let circuit = parse_circuit(circuit_json)?;
        let mut state = Stabilizer::new(circuit.num_qubits)?;
        let mut rng = crate::stabilizer::Rng::new(0);
        for op in &circuit.ops {
            match op {
                ir::Op::Gate {
                    gate,
                    name,
                    targets,
                    params,
                    condition,
                } => {
                    if condition.is_empty() {
                        state.apply(ir::executable(gate, name, params)?, targets)?;
                    }
                }
                ir::Op::Reset { qubit, condition } => {
                    if condition.is_empty() {
                        state.reset(*qubit, &mut rng)?;
                    }
                }
                ir::Op::Measure { .. } => {
                    return Err(VqcError::Unsupported(
                        "the circuit measures; a measured stabilizer state is a sampler, not a \
                         set of generators"
                            .to_string(),
                    )
                    .into())
                }
                ir::Op::Noise { kind, .. } => return Err(ir::refuse_noise(kind).into()),
            }
        }
        Ok(json!(state.generator_strings()).to_string())
    }

    /// `<P>` for one signed Pauli string of the state a circuit prepares.
    #[pyfunction]
    fn expectation(circuit_json: &str, labels: &str) -> PyResult<f64> {
        let circuit = parse_circuit(circuit_json)?;
        let mut state = Stabilizer::new(circuit.num_qubits)?;
        let mut rng = crate::stabilizer::Rng::new(0);
        for op in &circuit.ops {
            match op {
                ir::Op::Gate {
                    gate,
                    name,
                    targets,
                    params,
                    condition,
                } => {
                    if condition.is_empty() {
                        state.apply(ir::executable(gate, name, params)?, targets)?;
                    }
                }
                ir::Op::Reset { qubit, condition } => {
                    if condition.is_empty() {
                        state.reset(*qubit, &mut rng)?;
                    }
                }
                ir::Op::Measure { .. } => {
                    return Err(VqcError::Unsupported(
                        "the circuit measures; there is no state left to take an expectation in"
                            .to_string(),
                    )
                    .into())
                }
                ir::Op::Noise { kind, .. } => return Err(ir::refuse_noise(kind).into()),
            }
        }
        let labels = parse_labels(labels, circuit.num_qubits)?;
        Ok(state.expectation(&labels)?)
    }

    /// The derived Clifford conjugation tables, so Python can pin them exactly.
    #[pyfunction]
    fn gate_tables() -> PyResult<String> {
        let mut single = serde_json::Map::new();
        for name in ["h", "x", "y", "z", "s", "sdg"] {
            let gate = pauli::Clifford::from_name(name).expect("known gate");
            let mut rows = Vec::new();
            for label in 0..4u8 {
                let rule = match gate {
                    pauli::Clifford::One(one) => pauli::conjugate_one(one, label),
                    _ => unreachable!("one-qubit name"),
                };
                rows.push(json!([rule.delta, rule.label]));
            }
            single.insert(name.to_string(), json!(rows));
        }
        let mut pair = serde_json::Map::new();
        for name in ["cx", "cz", "swap"] {
            let gate = pauli::Clifford::from_name(name).expect("known gate");
            let mut rows = Vec::new();
            for first in 0..4u8 {
                for second in 0..4u8 {
                    let rule = match gate {
                        pauli::Clifford::Two(two) => pauli::conjugate_pair(two, first, second),
                        _ => unreachable!("two-qubit name"),
                    };
                    rows.push(json!([rule.delta, rule.first, rule.second]));
                }
            }
            pair.insert(name.to_string(), json!(rows));
        }
        Ok(json!({ "single": single, "pair": pair }).to_string())
    }

    /// The module: one versioned surface, nine names.
    #[pymodule]
    fn apedp_vqc(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        module.add("IMPLEMENTATION", "rust")?;
        module.add(
            "CLIFFORD_GATES",
            vec!["cx", "cz", "h", "s", "sdg", "swap", "x", "y", "z"],
        )?;
        module.add_function(wrap_pyfunction!(run_report, module)?)?;
        module.add_function(wrap_pyfunction!(equivalence_images, module)?)?;
        module.add_function(wrap_pyfunction!(distribution, module)?)?;
        module.add_function(wrap_pyfunction!(generators, module)?)?;
        module.add_function(wrap_pyfunction!(expectation, module)?)?;
        module.add_function(wrap_pyfunction!(gate_tables, module)?)?;
        Ok(())
    }
}
