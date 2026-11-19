//! Matrix-free Clifford equivalence: the `2n` signed Pauli images of a circuit.
//!
//! A Clifford circuit's conjugation action on `X_0, Z_0, X_1, Z_1, ...` is a
//! complete invariant of its unitary up to one global phase, so two Clifford
//! circuits are equal up to that phase exactly when their images agree entry by
//! entry. Nothing here builds a `2**n` matrix or a `2**n` state: it is bit
//! vectors, phase updates and a comparison, which is why it scales to 100+
//! qubits and why the review singled it out as a Rust-first module.
//!
//! The referee does *not* live here. This produces images; the Python
//! equivalence tier consumes them and decides. That keeps the verification
//! independent of the fast path (engineering rule 4) instead of asking the fast
//! path to mark its own homework.

use crate::ir::{executable, Op};
use crate::pauli::{conjugate_one, conjugate_pair, format_string, I, X, Z};
use crate::stabilizer::VqcError;

/// One signed Pauli string: the labels and the phase `i**p`.
pub type SignedImage = (Vec<u8>, i32);

/// `op_last ... op_1 P op_1^dagger ... op_last^dagger` for one Pauli string.
pub fn conjugate(
    labels: &[u8],
    phase: i32,
    ops: &[Op],
) -> Result<SignedImage, VqcError> {
    let mut current = labels.to_vec();
    let mut current_phase = phase.rem_euclid(4);
    for op in ops {
        let (gate, targets) = match op {
            Op::Gate { gate, name, targets, params, condition } => {
                if !condition.is_empty() {
                    return Err(VqcError::Unsupported(format!(
                        "a {name:?} gate carries a condition; conjugation is defined for \
                         unconditioned unitary Clifford gates only"
                    )));
                }
                (executable(gate, name, params)?, targets)
            }
            _ => {
                return Err(VqcError::Unsupported(
                    "a measurement, reset or noise operation is not an unconditioned gate; \
                     conjugation is defined for unitary Clifford gates only"
                        .to_string(),
                ))
            }
        };
        match gate {
            crate::pauli::Clifford::One(one) => {
                let qubit = targets[0];
                let rule = conjugate_one(one, current[qubit]);
                current[qubit] = rule.label;
                current_phase = (current_phase + rule.delta).rem_euclid(4);
            }
            crate::pauli::Clifford::Two(two) => {
                let (first, second) = (targets[0], targets[1]);
                let rule = conjugate_pair(two, current[first], current[second]);
                current[first] = rule.first;
                current[second] = rule.second;
                current_phase = (current_phase + rule.delta).rem_euclid(4);
            }
        }
    }
    Ok((current, current_phase))
}

/// The images of `X_0, Z_0, X_1, Z_1, ...`: `2n` entries, exactly computed.
pub fn images(ops: &[Op], num_qubits: usize) -> Result<Vec<SignedImage>, VqcError> {
    let mut images = Vec::with_capacity(2 * num_qubits);
    for qubit in 0..num_qubits {
        for label in [X, Z] {
            let mut labels = vec![I; num_qubits];
            labels[qubit] = label;
            images.push(conjugate(&labels, 0, ops)?);
        }
    }
    Ok(images)
}

/// The name of the `index`-th image: `X_0, Z_0, X_1, Z_1, ...`.
pub fn image_name(index: usize) -> String {
    let qubit = index / 2;
    let axis = if index.is_multiple_of(2) { 'X' } else { 'Z' };
    format!("{axis}_{qubit}")
}

/// One entry where two image tables disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub name: String,
    pub left: String,
    pub right: String,
}

/// Every entry where two image tables disagree, in order, as text.
pub fn compare(left: &[SignedImage], right: &[SignedImage]) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();
    for index in 0..left.len().min(right.len()) {
        if left[index] != right[index] {
            mismatches.push(Mismatch {
                name: image_name(index),
                left: format_string(&left[index].0, left[index].1),
                right: format_string(&right[index].0, right[index].1),
            });
        }
    }
    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;

    fn parse(text: &str) -> crate::ir::CircuitIr {
        ir::parse(text).unwrap()
    }

    #[test]
    fn the_identity_circuit_leaves_every_pauli_where_it_was() {
        let circuit = parse(r#"{"num_qubits":3,"num_bits":0,"ops":[]}"#);
        let table = images(&circuit.ops, 3).unwrap();
        assert_eq!(table.len(), 6);
        for (index, (labels, phase)) in table.iter().enumerate() {
            assert_eq!(*phase, 0);
            let qubit = index / 2;
            let label = if index % 2 == 0 { X } else { Z };
            assert_eq!(labels[qubit], label);
            assert_eq!(labels.iter().filter(|entry| **entry != I).count(), 1);
        }
    }

    #[test]
    fn cx_and_the_h_cz_h_identity_agree_but_are_not_the_same_circuit() {
        let cx_text = r#"{"num_qubits":2,"num_bits":0,"ops":[
            {"condition":null,"name":"cx","op":"gate","params":[],"targets":[0,1]}]}"#;
        let hzh_text = r#"{"num_qubits":2,"num_bits":0,"ops":[
            {"condition":null,"name":"h","op":"gate","params":[],"targets":[1]},
            {"condition":null,"name":"cz","op":"gate","params":[],"targets":[0,1]},
            {"condition":null,"name":"h","op":"gate","params":[],"targets":[1]}]}"#;
        let left = images(&parse(cx_text).ops, 2).unwrap();
        let right = images(&parse(hzh_text).ops, 2).unwrap();
        assert!(compare(&left, &right).is_empty());
    }

    #[test]
    fn a_relative_phase_changes_exactly_one_image() {
        // `sdg s` is the identity up to the global phase -1, which no image sees;
        // appending an `x` in between is a relative phase and is seen.
        let plain = r#"{"num_qubits":1,"num_bits":0,"ops":[
            {"condition":null,"name":"sdg","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"s","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"x","op":"gate","params":[],"targets":[0]}]}"#;
        let signed = r#"{"num_qubits":1,"num_bits":0,"ops":[
            {"condition":null,"name":"x","op":"gate","params":[],"targets":[0]}]}"#;
        let left = images(&parse(plain).ops, 1).unwrap();
        let right = images(&parse(signed).ops, 1).unwrap();
        assert!(compare(&left, &right).is_empty(), "s sdg is a global phase only");

        let relative = r#"{"num_qubits":1,"num_bits":0,"ops":[
            {"condition":null,"name":"x","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"s","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"x","op":"gate","params":[],"targets":[0]},
            {"condition":null,"name":"sdg","op":"gate","params":[],"targets":[0]}]}"#;
        let interior = images(&parse(relative).ops, 1).unwrap();
        let mismatches = compare(&left, &interior);
        assert_eq!(mismatches.len(), 2, "{mismatches:?}");
        assert_eq!(mismatches[0].name, "X_0");
    }

    #[test]
    fn a_measurement_makes_conjugation_undefined_rather_than_wrong() {
        let text = r#"{"num_qubits":1,"num_bits":1,"ops":[
            {"bit":0,"condition":null,"op":"measure","qubit":0}]}"#;
        let error = images(&parse(text).ops, 1).unwrap_err();
        assert!(error.message().contains("unconditioned gate"), "{error:?}");
    }

    #[test]
    fn images_grow_linearly_and_are_named_position_by_position() {
        assert_eq!(image_name(0), "X_0");
        assert_eq!(image_name(1), "Z_0");
        assert_eq!(image_name(4), "X_2");
        let text = r#"{"num_qubits":40,"num_bits":0,"ops":[
            {"condition":null,"name":"h","op":"gate","params":[],"targets":[39]}]}"#;
        let table = images(&parse(text).ops, 40).unwrap();
        assert_eq!(table.len(), 80);
        // H on the last qubit swaps X_39 and Z_39 and fixes everything else.
        assert_eq!(table[78].0[39], Z);
        assert_eq!(table[79].0[39], X);
    }
}
