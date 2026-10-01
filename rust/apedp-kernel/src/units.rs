//! Seven-dimensional SI unit algebra.
//!
//! A unit is the integer exponent vector of the seven SI base dimensions, in
//! the order (kg, m, s, A, K, mol, cd). Integer exponents keep the algebra
//! exact: dimensional homogeneity is a statement about integer powers, and
//! `m * v**2 == J` has to hold exactly rather than to within a tolerance.

use std::fmt;

/// The base dimensions, in exponent-vector order.
///
/// Length comes first, then mass: `tests/test_cross_implementation_conformance.py::test_unit_vector_algebra`
/// pins this order, and its expectations are only mutually consistent under
/// it (N*m is (2,1,-2) because the metre exponents add first, and
/// kg*(m/s)**2 lands on the same vector as a joule). SI's own ordering puts
/// mass first; switching to it here would silently break that contract, so the
/// order is stated rather than assumed.
pub const BASE_DIMENSIONS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// A dimension: one `i32` exponent per base dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitVector {
    exponents: [i32; 7],
}

/// Raised when a unit string cannot be resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitError(pub String);

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for UnitError {}

impl UnitVector {
    pub const fn dimensionless() -> UnitVector {
        UnitVector {
            exponents: [0; 7],
        }
    }

    pub fn as_tuple(&self) -> (i32, i32, i32, i32, i32, i32, i32) {
        (
            self.exponents[0],
            self.exponents[1],
            self.exponents[2],
            self.exponents[3],
            self.exponents[4],
            self.exponents[5],
            self.exponents[6],
        )
    }

    pub fn is_dimensionless(&self) -> bool {
        self.exponents.iter().all(|e| *e == 0)
    }

    pub fn mul(&self, other: &UnitVector) -> UnitVector {
        let mut out = [0i32; 7];
        for i in 0..7 {
            out[i] = self.exponents[i] + other.exponents[i];
        }
        UnitVector { exponents: out }
    }

    pub fn div(&self, other: &UnitVector) -> UnitVector {
        let mut out = [0i32; 7];
        for i in 0..7 {
            out[i] = self.exponents[i] - other.exponents[i];
        }
        UnitVector { exponents: out }
    }

    pub fn pow(&self, n: i32) -> UnitVector {
        let mut out = [0i32; 7];
        for i in 0..7 {
            out[i] = self.exponents[i] * n;
        }
        UnitVector { exponents: out }
    }

    /// Resolve a unit name, or a product/quotient/power of names such as
    /// `"kg*m/s**2"` or `"N*m"`.
    pub fn parse(text: &str) -> Result<UnitVector, UnitError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(UnitError("empty unit string".to_string()));
        }
        if text == "1" || text.eq_ignore_ascii_case("dimensionless") {
            return Ok(UnitVector::dimensionless());
        }

        // Split on '/' for division, respecting that `**` may follow a name.
        let (numerator_text, denominator_text) = match find_top_level(text, '/') {
            Some(idx) => (&text[..idx], &text[idx + 1..]),
            None => (text, ""),
        };

        let numerator = Self::parse_product(numerator_text)?;
        if denominator_text.is_empty() {
            return Ok(numerator);
        }
        let denominator = Self::parse_product(denominator_text)?;
        Ok(numerator.div(&denominator))
    }

    fn parse_product(text: &str) -> Result<UnitVector, UnitError> {
        let mut acc = UnitVector::dimensionless();
        for factor in split_top_level(text, '*') {
            let factor = factor.trim();
            if factor.is_empty() {
                continue;
            }
            acc = acc.mul(&Self::parse_factor(factor)?);
        }
        Ok(acc)
    }

    fn parse_factor(text: &str) -> Result<UnitVector, UnitError> {
        // A trailing `**n` or `^n` makes the whole factor a power.
        for marker in ["**", "^"] {
            if let Some(idx) = find_top_level_from_end(text, marker) {
                let (base, exp_text) = text.split_at(idx);
                let exp_text = exp_text[marker.len()..].trim();
                let exponent: i32 = exp_text.parse().map_err(|_| {
                    UnitError(format!("{exp_text:?} is not an integer exponent"))
                })?;
                return Ok(Self::parse_factor(base.trim())?.pow(exponent));
            }
        }
        Self::lookup(text)
    }

    fn lookup(name: &str) -> Result<UnitVector, UnitError> {
        let key = name.trim().trim_matches('(').trim_matches(')');
        let lower = key.to_ascii_lowercase();
        let unit = match lower.as_str() {
            // Dimensionless
            "" | "1" | "dimensionless" => UnitVector::dimensionless(),
            // Mass
            "g" | "gram" | "grams" => UnitVector::dimensionless().mul(&Self::kg(1)).mul(&Self::kg(-3)),
            "kg" | "kilogram" | "kilograms" => Self::kg(1),
            // Length
            "m" | "meter" | "meters" | "metre" | "metres" => Self::metre(1),
            "cm" | "centimeter" | "centimetre" => Self::metre(-2),
            "mm" | "millimeter" | "millimetre" => Self::metre(-3),
            "km" | "kilometer" | "kilometre" => Self::metre(3),
            // Time
            "s" | "sec" | "second" | "seconds" => Self::dim(2, 1),
            "ms" | "millisecond" | "milliseconds" => Self::dim(2, -3),
            "min" | "minute" | "minutes" => Self::dim(2, 1),
            "h" | "hr" | "hour" | "hours" => Self::dim(2, 1),
            // Current, temperature, amount, luminous intensity
            "a" | "ampere" | "amperes" | "amp" => Self::dim(3, 1),
            "k" | "kelvin" => Self::dim(4, 1),
            "mol" | "mole" | "moles" => Self::dim(5, 1),
            "cd" | "candela" => Self::dim(6, 1),
            // Derived SI units with special names
            "hz" | "hertz" => Self::dim(2, -1),
            // N = m * kg * s^-2
            "n" | "newton" | "newtons" => Self::metre(1).mul(&Self::kg(1)).mul(&Self::dim(2, -2)),
            "pa" | "pascal" | "pascals" => Self::metre(-1).mul(&Self::kg(1)).mul(&Self::dim(2, -2)),
            // J = m^2 * kg * s^-2
            "j" | "joule" | "joules" => Self::metre(2).mul(&Self::kg(1)).mul(&Self::dim(2, -2)),
            // W = m^2 * kg * s^-3
            "w" | "watt" | "watts" => Self::metre(2).mul(&Self::kg(1)).mul(&Self::dim(2, -3)),
            // V = m^2 * kg * s^-3 * A^-1
            "v" | "volt" | "volts" => {
                Self::metre(2).mul(&Self::kg(1)).mul(&Self::dim(2, -3)).mul(&Self::dim(3, -1))
            }
            "c" | "coulomb" | "coulombs" => {
                UnitVector::dimensionless().mul(&Self::dim(2, 1)).mul(&Self::dim(3, 1))
            }
            "f" | "farad" | "farads" => {
                UnitVector::dimensionless()
                    .mul(&Self::metre(-2))
                    .mul(&Self::kg(-1))
                    .mul(&Self::dim(2, 4))
                    .mul(&Self::dim(3, 2))
            }
            // ohm = m^2 * kg * s^-3 * A^-2
            "ohm" | "ohms" => Self::metre(2)
                .mul(&Self::kg(1))
                .mul(&Self::dim(2, -3))
                .mul(&Self::dim(3, -2)),
            _ => {
                return Err(UnitError(format!(
                    "unknown unit {key:?}; expected one of the SI base or derived names, \
                     or a product like \"kg*m/s**2\""
                )))
            }
        };
        Ok(unit)
    }

    /// Metre exponent. Index 0 -- see `BASE_DIMENSIONS`.
    fn metre(n: i32) -> UnitVector {
        Self::dim(0, n)
    }

    /// Kilogram exponent. Index 1 -- see `BASE_DIMENSIONS`.
    fn kg(n: i32) -> UnitVector {
        Self::dim(1, n)
    }

    fn dim(index: usize, n: i32) -> UnitVector {
        let mut e = [0i32; 7];
        e[index] = n;
        UnitVector { exponents: e }
    }
}

impl fmt::Display for UnitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }
        let mut parts: Vec<String> = Vec::new();
        for (i, exp) in self.exponents.iter().enumerate() {
            if *exp == 0 {
                continue;
            }
            if *exp == 1 {
                parts.push(BASE_DIMENSIONS[i].to_string());
            } else {
                parts.push(format!("{}**{}", BASE_DIMENSIONS[i], exp));
            }
        }
        write!(f, "{}", parts.join("*"))
    }
}

/// Index of the first `needle` at bracket depth zero, or None.
fn find_top_level(haystack: &str, needle: char) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in haystack.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            c if c == needle && depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Index of the last `needle` at bracket depth zero, or None.
fn find_top_level_from_end(haystack: &str, needle: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let nlen = needle.len();
    let mut depth = 0i32;
    let mut found = None;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            _ => {
                if depth == 0
                    && i + nlen <= bytes.len()
                    && &haystack[i..i + nlen] == needle
                {
                    found = Some(i);
                }
            }
        }
        i += 1;
    }
    found
}

/// Split on `separator` at bracket depth zero.
///
/// Skips the `**` operator: splitting `s**2` on `*` naively yields
/// ["s", "", "2"], and the trailing "2" then gets looked up as a unit name.
fn split_top_level(haystack: &str, separator: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = haystack.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Every separator and bracket byte here is ASCII, so stepping by byte
        // cannot land mid-character.
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b if b as char == separator && depth == 0 => {
                if separator == '*' && haystack[i + 1..].starts_with('*') {
                    i += 2;
                    continue;
                }
                out.push(&haystack[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out.push(&haystack[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seven_base_dimensions_are_dimensionless() {
        let u = UnitVector::dimensionless();
        assert!(u.is_dimensionless());
        assert_eq!(u.as_tuple(), (0, 0, 0, 0, 0, 0, 0));
    }

    #[test]
    fn newton_is_kg_m_per_s_squared() {
        assert_eq!(UnitVector::parse("newton").unwrap().as_tuple(), (1, 1, -2, 0, 0, 0, 0));
    }

    #[test]
    fn joule_is_newton_meter() {
        let n = UnitVector::parse("newton").unwrap();
        let m = UnitVector::parse("meter").unwrap();
        assert_eq!(n.mul(&m).as_tuple(), (2, 1, -2, 0, 0, 0, 0));
        assert_eq!(UnitVector::parse("joule").unwrap().as_tuple(), (2, 1, -2, 0, 0, 0, 0));
    }

    #[test]
    fn watt_is_joule_per_second() {
        let j = UnitVector::parse("joule").unwrap();
        let s = UnitVector::parse("second").unwrap();
        assert_eq!(j.div(&s).as_tuple(), (2, 1, -3, 0, 0, 0, 0));
        assert_eq!(UnitVector::parse("watt").unwrap().as_tuple(), (2, 1, -3, 0, 0, 0, 0));
    }

    #[test]
    fn kinetic_energy_is_dimensionally_homogeneous() {
        let kg = UnitVector::parse("kilogram").unwrap();
        let m = UnitVector::parse("meter").unwrap();
        let s = UnitVector::parse("second").unwrap();
        let v = m.div(&s);
        let v2 = v.pow(2);
        let ke = kg.mul(&v2);
        let j = UnitVector::parse("joule").unwrap();
        assert_eq!(ke.as_tuple(), j.as_tuple());
    }

    #[test]
    fn compound_strings_resolve() {
        assert_eq!(
            UnitVector::parse("kg*m/s**2").unwrap().as_tuple(),
            (1, 1, -2, 0, 0, 0, 0)
        );
        assert_eq!(
            UnitVector::parse("N*m").unwrap().as_tuple(),
            (2, 1, -2, 0, 0, 0, 0)
        );
    }

    #[test]
    fn an_unknown_unit_is_an_error_not_a_zero_vector() {
        // A silent zero would make every dimension look dimensionless, which
        // is the failure this whole algebra exists to prevent.
        let e = UnitVector::parse("furlong").unwrap_err();
        assert!(e.0.contains("unknown unit"), "{e}");
    }

    #[test]
    fn dimensionful_units_are_reported_as_such() {
        assert!(!UnitVector::parse("newton").unwrap().is_dimensionless());
    }

    /// The exact sequence asserted by
    /// `tests/test_cross_implementation_conformance.py::test_unit_vector_algebra`.
    /// Duplicated here so the two engines' unit orderings cannot drift apart
    /// without one of them failing.
    #[test]
    fn matches_the_conformance_suites_unit_algebra() {
        assert_eq!(UnitVector::dimensionless().as_tuple(), (0, 0, 0, 0, 0, 0, 0));
        assert!(UnitVector::dimensionless().is_dimensionless());
        assert_eq!(UnitVector::parse("newton").unwrap().as_tuple(), (1, 1, -2, 0, 0, 0, 0));

        let n = UnitVector::parse("newton").unwrap();
        let m = UnitVector::parse("meter").unwrap();
        assert_eq!(n.mul(&m).as_tuple(), (2, 1, -2, 0, 0, 0, 0));

        let s = UnitVector::parse("second").unwrap();
        assert_eq!(n.mul(&m).div(&s).as_tuple(), (2, 1, -3, 0, 0, 0, 0));

        let kg = UnitVector::parse("kilogram").unwrap();
        let v = m.div(&s);
        assert_eq!(kg.mul(&v.pow(2)).as_tuple(), (2, 1, -2, 0, 0, 0, 0));
    }
}