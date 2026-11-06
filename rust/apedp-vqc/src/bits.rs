//! A fixed-width bit vector, one `u64` word at a time.
//!
//! The stabilizer tableau is a set of GF(2) rows over `2n` columns (an `x` half
//! and a `z` half). Python stores each row as one arbitrary-precision integer;
//! Rust has no such type in the standard library, so the same row is an explicit
//! word vector here. The operations the tableau needs are exactly the ones
//! below: XOR, "lowest set bit", zero test, and a parity-of-AND for the
//! symplectic product. Nothing else is exposed, so there is no place for a
//! second, divergent notion of a row to grow.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// `len` zero bits.
    pub fn zeros(len: usize) -> Self {
        Bitset { words: vec![0u64; len.div_ceil(64)], len }
    }

    /// Build from big-endian-bit order, i.e. `bits[i]` is bit `i`.
    pub fn from_bits(bits: &[bool]) -> Self {
        let mut set = Bitset::zeros(bits.len());
        for (index, bit) in bits.iter().enumerate() {
            if *bit {
                set.set(index);
            }
        }
        set
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set(&mut self, index: usize) {
        assert!(index < self.len, "bit {index} outside a {}-bit vector", self.len);
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    pub fn get(&self, index: usize) -> bool {
        index < self.len && (self.words[index / 64] >> (index % 64)) & 1 == 1
    }

    pub fn xor_assign(&mut self, other: &Bitset) {
        assert_eq!(self.len, other.len, "bit vectors of different widths");
        for (left, right) in self.words.iter_mut().zip(other.words.iter()) {
            *left ^= right;
        }
    }

    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Index of the lowest set bit, or `None` for the zero vector.
    ///
    /// This is Python's `(mask & -mask).bit_length() - 1`: the pivot column of
    /// the row in the column-echelon basis.
    pub fn lowest(&self) -> Option<usize> {
        self.words.iter().enumerate().find_map(|(word_index, word)| {
            if *word == 0 {
                None
            } else {
                Some(word_index * 64 + word.trailing_zeros() as usize)
            }
        })
    }

    /// `((self & a) ^ (b & c)).count_ones() & 1`, the symplectic parity.
    pub fn parity_of(&self, a: &Bitset, b: &Bitset, c: &Bitset) -> bool {
        let mut parity = 0u32;
        for index in 0..self.words.len() {
            let left = self.words[index] & a.words[index];
            let right = b.words[index] & c.words[index];
            parity ^= (left ^ right).count_ones() & 1;
        }
        parity == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowest_bit_is_the_pivot_the_tableau_expects() {
        let mut bits = Bitset::zeros(200);
        bits.set(0);
        assert_eq!(bits.lowest(), Some(0));
        bits.set(129);
        assert_eq!(bits.lowest(), Some(0));
        bits.xor_assign(&{
            let mut other = Bitset::zeros(200);
            other.set(0);
            other
        });
        assert_eq!(bits.lowest(), Some(129));
    }

    #[test]
    fn a_row_wider_than_one_word_xors_exactly() {
        let mut left = Bitset::zeros(300);
        let mut right = Bitset::zeros(300);
        left.set(299);
        right.set(299);
        right.set(64);
        left.xor_assign(&right);
        assert_eq!(left.lowest(), Some(64));
        assert!(left.get(64));
        assert!(!left.get(299));
    }

    #[test]
    fn the_zero_vector_has_no_pivot() {
        assert_eq!(Bitset::zeros(70).lowest(), None);
        assert!(Bitset::zeros(70).is_zero());
    }
}
