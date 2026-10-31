//! The shot loop: one stabilizer state per shot, classical bits accumulated,
//! outcomes counted.
//!
//! This mirrors `apedp_dyn/vqc/stabilizer.py::StabilizerBackend.run` operation
//! for operation, including the two details that are easy to get subtly wrong:
//!
//! * A measured bit that was never written reads as `false` in the returned
//!   bitstring, exactly as `bits.get(index, False)` does in Python.
//! * The ancilla figures are the *post-run* excitation probability of each
//!   ancilla, averaged over shots, not the count of ancilla measurements.

use crate::ir::{condition_holds, executable, refuse_noise, CircuitIr, Op};
use crate::stabilizer::{Rng, Stabilizer, VqcError};

/// What a run produced: the counters, the per-shot samples, the ancilla figures.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub backend: &'static str,
    pub seed: u64,
    pub counts: Vec<(String, usize)>,
    pub samples: Vec<Vec<bool>>,
    pub ancilla_excitation: Vec<(usize, f64)>,
}

fn bitstring(bits: &[bool]) -> String {
    bits.iter().map(|bit| if *bit { '1' } else { '0' }).collect()
}

/// Execute `circuit` for `shots` shots, seeded deterministically by `seed`.
pub fn run(circuit: &CircuitIr, shots: usize, seed: u64) -> Result<RunReport, VqcError> {
    if shots < 1 {
        return Err(VqcError::Value(format!("shots must be positive, got {shots}")));
    }
    let mut rng = Rng::new(seed);
    let mut counts: Vec<(String, usize)> = Vec::new();
    let mut samples: Vec<Vec<bool>> = Vec::with_capacity(shots);
    let mut excitation: Vec<f64> = vec![0.0; circuit.num_qubits];

    for _shot in 0..shots {
        let mut state = Stabilizer::new(circuit.num_qubits)?;
        let mut bits: Vec<Option<bool>> = vec![None; circuit.num_bits];
        for op in &circuit.ops {
            let collected: Vec<bool> = bits.iter().map(|bit| bit.unwrap_or(false)).collect();
            if !condition_holds(op.condition(), &collected) {
                continue;
            }
            match op {
                Op::Gate { gate, name, targets, params, .. } => {
                    let gate = executable(gate, name, params)?;
                    state.apply(gate, targets)?;
                }
                Op::Measure { qubit, bit, .. } => {
                    let outcome = state.measure(*qubit, None, &mut rng)?;
                    if *bit < bits.len() {
                        bits[*bit] = Some(outcome);
                    }
                }
                Op::Reset { qubit, .. } => state.reset(*qubit, &mut rng)?,
                Op::Noise { kind, .. } => return Err(refuse_noise(kind)),
            }
        }
        let outcome: Vec<bool> = (0..circuit.num_bits)
            .map(|index| bits[index].unwrap_or(false))
            .collect();
        let key = bitstring(&outcome);
        match counts.iter_mut().find(|(existing, _)| *existing == key) {
            Some((_, count)) => *count += 1,
            None => counts.push((key, 1)),
        }
        for ancilla in &circuit.ancillas {
            excitation[*ancilla] += state.outcome_of(*ancilla)?;
        }
        samples.push(outcome);
    }

    Ok(RunReport {
        backend: "stabilizer",
        seed,
        counts,
        samples,
        ancilla_excitation: circuit
            .ancillas
            .iter()
            .map(|ancilla| (*ancilla, excitation[*ancilla] / shots as f64))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;

    fn bell() -> CircuitIr {
        let text = r#"{"num_qubits":2,"num_bits":2,"ancillas":[],"ops":[
            {"condition":null,"name":"h","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"cx","op":"gate","params":[],"targets":[0,1]},
            {"bit":0,"condition":null,"op":"measure","qubit":0},
            {"bit":1,"condition":null,"op":"measure","qubit":1}]}"#;
        ir::parse(text).unwrap()
    }

    #[test]
    fn a_bell_pair_only_ever_reads_the_two_agreeing_strings() {
        let report = run(&bell(), 200, 5).unwrap();
        assert_eq!(report.samples.len(), 200);
        assert_eq!(report.counts.len(), 2);
        let keys: Vec<&String> = report.counts.iter().map(|(key, _)| key).collect();
        assert!(keys.contains(&&"00".to_string()));
        assert!(keys.contains(&&"11".to_string()));
        let total: usize = report.counts.iter().map(|(_, count)| count).sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn the_same_seed_replays_the_same_run_and_the_counters_still_add_up() {
        let first = run(&bell(), 64, 99).unwrap();
        let second = run(&bell(), 64, 99).unwrap();
        assert_eq!(first.samples, second.samples);
        assert_eq!(first.counts, second.counts);
        let other = run(&bell(), 64, 100).unwrap();
        assert_ne!(first.samples, other.samples);
    }

    #[test]
    fn an_unwritten_bit_reads_as_zero_in_the_bitstring() {
        let text = r#"{"num_qubits":1,"num_bits":3,"ancillas":[],"ops":[
            {"bit":2,"condition":null,"op":"measure","qubit":0}]}"#;
        let circuit = ir::parse(text).unwrap();
        let report = run(&circuit, 4, 1).unwrap();
        assert!(report.samples.iter().all(|sample| sample == &vec![false, false, false]));
        assert_eq!(report.counts, vec![("000".to_string(), 4)]);
    }

    #[test]
    fn a_noise_operation_is_refused_with_the_statevector_recommendation() {
        let text = r#"{"num_qubits":1,"num_bits":0,"ops":[
            {"condition":null,"kind":"bit_flip","op":"noise","probability":0.1,"qubit":0}]}"#;
        let circuit = ir::parse(text).unwrap();
        let error = run(&circuit, 1, 0).unwrap_err();
        assert!(error.message().contains("statevector tier"), "{error:?}");
    }

    #[test]
    fn zero_shots_is_a_value_error_not_an_empty_report() {
        let error = run(&bell(), 0, 0).unwrap_err();
        assert!(error.message().contains("shots must be positive"));
    }
}
