//! 7-dimensional SI unit vector algebra engine.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitVector {
    pub length: i8,      // meter [m]
    pub mass: i8,        // kilogram [kg]
    pub time: i8,        // second [s]
    pub current: i8,     // ampere [A]
    pub temp: i8,        // kelvin [K]
    pub amount: i8,      // mole [mol]
    pub luminosity: i8,  // candela [cd]
}

impl UnitVector {
    pub const DIMENSIONLESS: Self = Self {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temp: 0,
        amount: 0,
        luminosity: 0,
    };

    pub fn is_dimensionless(&self) -> bool {
        *self == Self::DIMENSIONLESS
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            length: self.length + rhs.length,
            mass: self.mass + rhs.mass,
            time: self.time + rhs.time,
            current: self.current + rhs.current,
            temp: self.temp + rhs.temp,
            amount: self.amount + rhs.amount,
            luminosity: self.luminosity + rhs.luminosity,
        }
    }

    #[inline(always)]
    pub fn div(self, rhs: Self) -> Self {
        Self {
            length: self.length - rhs.length,
            mass: self.mass - rhs.mass,
            time: self.time - rhs.time,
            current: self.current - rhs.current,
            temp: self.temp - rhs.temp,
            amount: self.amount - rhs.amount,
            luminosity: self.luminosity - rhs.luminosity,
        }
    }

    #[inline(always)]
    pub fn pow(self, power: i8) -> Self {
        Self {
            length: self.length * power,
            mass: self.mass * power,
            time: self.time * power,
            current: self.current * power,
            temp: self.temp * power,
            amount: self.amount * power,
            luminosity: self.luminosity * power,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "dimensionless" | "1" | "none" | "" => Ok(Self::DIMENSIONLESS),
            "meter" | "m" => Ok(Self { length: 1, ..Self::DIMENSIONLESS }),
            "meter**2" | "m**2" => Ok(Self { length: 2, ..Self::DIMENSIONLESS }),
            "meter**3" | "m**3" => Ok(Self { length: 3, ..Self::DIMENSIONLESS }),
            "per_meter" | "1/m" => Ok(Self { length: -1, ..Self::DIMENSIONLESS }),
            "kilogram" | "kg" => Ok(Self { mass: 1, ..Self::DIMENSIONLESS }),
            "second" | "s" => Ok(Self { time: 1, ..Self::DIMENSIONLESS }),
            "per_second" | "1/s" | "hz" => Ok(Self { time: -1, ..Self::DIMENSIONLESS }),
            "ampere" | "a" => Ok(Self { current: 1, ..Self::DIMENSIONLESS }),
            "kelvin" | "k" => Ok(Self { temp: 1, ..Self::DIMENSIONLESS }),
            "mole" | "mol" => Ok(Self { amount: 1, ..Self::DIMENSIONLESS }),
            "candela" | "cd" => Ok(Self { luminosity: 1, ..Self::DIMENSIONLESS }),
            "meter_per_second" | "m/s" => Ok(Self { length: 1, time: -1, ..Self::DIMENSIONLESS }),
            "meter_per_second**2" | "m/s**2" => Ok(Self { length: 1, time: -2, ..Self::DIMENSIONLESS }),
            "meter**2_per_second**2" => Ok(Self { length: 2, time: -2, ..Self::DIMENSIONLESS }),
            "meter**2_per_second**3" => Ok(Self { length: 2, time: -3, ..Self::DIMENSIONLESS }),
            "newton" | "n" => Ok(Self { length: 1, mass: 1, time: -2, ..Self::DIMENSIONLESS }),
            "pascal" | "pa" => Ok(Self { length: -1, mass: 1, time: -2, ..Self::DIMENSIONLESS }),
            "pascal_second" | "pa*s" => Ok(Self { length: -1, mass: 1, time: -1, ..Self::DIMENSIONLESS }),
            "joule" | "j" => Ok(Self { length: 2, mass: 1, time: -2, ..Self::DIMENSIONLESS }),
            "joule_second_per_meter" => Ok(Self { length: 1, mass: 1, time: -1, ..Self::DIMENSIONLESS }),
            "joule_second_per_meter**3" => Ok(Self { length: -1, mass: 1, time: -1, ..Self::DIMENSIONLESS }),
            "watt" | "w" => Ok(Self { length: 2, mass: 1, time: -3, ..Self::DIMENSIONLESS }),
            "astronomical_unit" | "au" => Ok(Self { length: 1, ..Self::DIMENSIONLESS }),
            "year" | "yr" => Ok(Self { time: 1, ..Self::DIMENSIONLESS }),
            _ => Err(format!("Unrecognized unit string: '{}'", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let u = UnitVector::DIMENSIONLESS;
        assert!(u.is_dimensionless());
    }

    #[test]
    fn test_mul_div() {
        let m = UnitVector::parse("meter").unwrap();
        let s = UnitVector::parse("second").unwrap();
        let v = m.div(s);
        assert_eq!(v.length, 1);
        assert_eq!(v.time, -1);
    }

    #[test]
    fn test_pow() {
        let m = UnitVector::parse("meter").unwrap();
        let a = m.pow(2);
        assert_eq!(a.length, 2);
    }
}
