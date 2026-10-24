//! Expression parser for the APEDP numerical kernel.
//!
//! This is deliberately an *independent* implementation rather than a binding
//! to sympy. The whole point of `tests/test_cross_implementation_conformance.py`
//! is to compare two implementations that were written separately; a Rust
//! parser that called into sympy would make the differential suite a tautology.
//!
//! The grammar covers what the search pipeline emits after
//! `apedp.safety.safe_parse_expr` has allowlisted it: arithmetic plus a small
//! transcendental set. Implicit multiplication (`2x`, `3(x+1)`) is accepted
//! because sympy accepts it, so the reference side would evaluate it and this
//! side has to agree.

use std::collections::HashMap;
use std::fmt;

/// A node in the parsed expression tree.
///
/// `Ident` is the parser's output; `compile` rewrites it into `Var` or
/// `Const` once the caller's variable table is known.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Const(f64),
    Ident(String),
    /// Index into the caller's variable table, resolved at compile time.
    Var(usize),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
    Min(Box<Node>, Box<Node>),
    Max(Box<Node>, Box<Node>),
    /// `**`. The exponent stays a full expression so `2**(x+1)` parses the way
    /// Python and sympy parse it.
    Pow(Box<Node>, Box<Node>),
    Neg(Box<Node>),
    Call(Func, Box<Node>),
}

/// The supported unary functions. `log` and `ln` are both natural logarithms,
/// matching sympy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Log10,
    Log2,
    Sqrt,
    Abs,
    Sign,
    Floor,
    Erf,
    BesselJ0,
    Cbrt,
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        let lower = name.to_ascii_lowercase();
        Some(match lower.as_str() {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "asin" | "arcsin" => Func::Asin,
            "acos" | "arccos" => Func::Acos,
            "atan" | "arctan" => Func::Atan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "tanh" => Func::Tanh,
            "exp" => Func::Exp,
            "log" | "ln" => Func::Ln,
            "log10" => Func::Log10,
            "log2" => Func::Log2,
            "sqrt" => Func::Sqrt,
            "abs" => Func::Abs,
            "sign" => Func::Sign,
            "floor" => Func::Floor,
            "erf" => Func::Erf,
            "besselj" | "besselj0" => Func::BesselJ0,
            "cbrt" => Func::Cbrt,
            _ => return None,
        })
    }

    /// Domain behaviour follows the C library, which is what numpy also wraps,
    /// so NaN/inf land in the same places on both sides of the differential.
    #[inline]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Asin => x.asin(),
            Func::Acos => x.acos(),
            Func::Atan => x.atan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Tanh => x.tanh(),
            Func::Exp => x.exp(),
            Func::Ln => x.ln(),
            Func::Log10 => x.log10(),
            Func::Log2 => x.log2(),
            Func::Sqrt => x.sqrt(),
            Func::Abs => x.abs(),
            Func::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    x // keeps -0.0 and NaN, as numpy does
                }
            }
            Func::Floor => x.floor(),
            Func::Erf => crate::numeric::erf_approx(x),
            Func::BesselJ0 => crate::numeric::bessel_j0(x),
            Func::Cbrt => x.cbrt(),
        }
    }
}

/// Raised when an expression cannot be compiled.
///
/// The bridge treats this as "this candidate is not evaluable by the fast
/// path", never as a verdict about the candidate's quality.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at position {}", self.message, self.position)
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    /// `**` or `^`.
    Pow,
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone, PartialEq)]
struct Spanned {
    token: Token,
    position: usize,
}

fn tokenize(src: &str) -> Result<Vec<Spanned>, ParseError> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        let start = i;
        let token = match c {
            '+' => {
                i += 1;
                Token::Plus
            }
            '-' => {
                i += 1;
                Token::Minus
            }
            '*' => {
                // `**` is power; a lone `*` is multiplication.
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    i += 2;
                    Token::Pow
                } else {
                    i += 1;
                    Token::Star
                }
            }
            '/' => {
                i += 1;
                Token::Slash
            }
            '^' => {
                i += 1;
                Token::Pow
            }
            '(' => {
                i += 1;
                Token::LParen
            }
            ')' => {
                i += 1;
                Token::RParen
            }
            ',' => {
                i += 1;
                Token::Comma
            }
            c if c.is_ascii_digit()
                || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) =>
            {
                let mut seen_dot = false;
                let mut seen_exp = false;
                while i < chars.len() {
                    let d = chars[i];
                    if d.is_ascii_digit() {
                        i += 1;
                    } else if d == '.' && !seen_dot && !seen_exp {
                        seen_dot = true;
                        i += 1;
                    } else if (d == 'e' || d == 'E') && !seen_exp {
                        // Consume the exponent only if it is well formed, so
                        // `2e` stays a parse error instead of silently
                        // becoming `2 * e`.
                        let next = i + 1;
                        if next < chars.len()
                            && (chars[next].is_ascii_digit()
                                || ((chars[next] == '+' || chars[next] == '-')
                                    && next + 1 < chars.len()
                                    && chars[next + 1].is_ascii_digit()))
                        {
                            seen_exp = true;
                            i += 1;
                            if chars[i] == '+' || chars[i] == '-' {
                                i += 1;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                let value = text.parse::<f64>().map_err(|_| ParseError {
                    message: format!("malformed number {text:?}"),
                    position: start,
                })?;
                Token::Number(value)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                Token::Ident(chars[start..i].iter().collect())
            }
            other => {
                return Err(ParseError {
                    message: format!("unexpected character {other:?}"),
                    position: start,
                })
            }
        };
        out.push(Spanned {
            token,
            position: start,
        });
    }

    Ok(out)
}

/// A parsed, variable-resolved expression ready to evaluate.
#[derive(Debug, Clone)]
pub struct Compiled {
    root: Node,
    num_vars: usize,
    /// Variable names in table order, so the evaluator's indexing is checkable
    /// against the caller's `data` ordering at the boundary.
    pub variables: Vec<String>,
    pub source: String,
}

impl Compiled {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Evaluate over `data`, one column per declared variable.
    ///
    /// Columns may be owned `Vec<f64>` or borrowed `ArrayView1<f64>` views
    /// into NumPy buffers — anything implementing [`crate::data::DataCol`].
    ///
    /// Returns a vector of length `data[0].row_len()`. Arithmetic that numpy
    /// would render as inf or NaN renders the same way here, because Rust's
    /// f64 does not trap.
    pub fn evaluate<V: crate::data::DataCol>(&self, data: &[V]) -> Result<Vec<f64>, String> {
        if data.len() != self.num_vars {
            return Err(format!(
                "expression '{}' declares {} variable(s) but {} array(s) were supplied",
                self.source,
                self.num_vars,
                data.len()
            ));
        }
        if data.is_empty() {
            // Nothing to infer an output length from. Returning an empty array
            // would look like a successful evaluation of zero points, which is
            // a different claim.
            return Err(format!(
                "expression '{}' has no variables, so there is no data length to \
                 evaluate over; a constant expression carries no row count",
                self.source
            ));
        }
        let expected = data[0].row_len();
        for (i, col) in data.iter().enumerate() {
            if col.row_len() != expected {
                return Err(format!(
                    "variable '{}' has length {} but '{}' has length {}; \
                     all data arrays must have the same length",
                    self.variables[i],
                    col.row_len(),
                    self.variables[0],
                    expected
                ));
            }
        }
        let n = expected;

        let mut out = vec![0.0f64; n];
        for (idx, slot) in out.iter_mut().enumerate() {
            *slot = eval_node(&self.root, data, idx);
        }
        Ok(out)
    }

    /// Evaluate over a single point, where `point[i]` is the value of `self.variables[i]`.
    ///
    /// Highly optimized for tight ODE inner integration loops: avoids all vector allocations.
    #[inline]
    pub fn eval_point(&self, point: &[f64]) -> f64 {
        eval_node_point(&self.root, point)
    }
}

#[inline]
fn eval_node_point(node: &Node, point: &[f64]) -> f64 {
    match node {
        Node::Const(c) => *c,
        Node::Var(v) => point[*v],
        Node::Add(a, b) => eval_node_point(a, point) + eval_node_point(b, point),
        Node::Sub(a, b) => eval_node_point(a, point) - eval_node_point(b, point),
        Node::Mul(a, b) => eval_node_point(a, point) * eval_node_point(b, point),
        Node::Div(a, b) => eval_node_point(a, point) / eval_node_point(b, point),
        Node::Min(a, b) => eval_node_point(a, point).min(eval_node_point(b, point)),
        Node::Max(a, b) => eval_node_point(a, point).max(eval_node_point(b, point)),
        Node::Pow(a, b) => eval_node_point(a, point).powf(eval_node_point(b, point)),
        Node::Neg(a) => -eval_node_point(a, point),
        Node::Call(f, a) => f.apply(eval_node_point(a, point)),
        Node::Ident(name) => unreachable!("unresolved identifier {name:?} reached evaluation"),
    }
}

#[inline]
fn eval_node<V: crate::data::DataCol>(node: &Node, data: &[V], i: usize) -> f64 {
    match node {
        Node::Const(c) => *c,
        Node::Var(v) => data[*v].row(i),
        Node::Add(a, b) => eval_node(a, data, i) + eval_node(b, data, i),
        Node::Sub(a, b) => eval_node(a, data, i) - eval_node(b, data, i),
        Node::Mul(a, b) => eval_node(a, data, i) * eval_node(b, data, i),
        Node::Div(a, b) => eval_node(a, data, i) / eval_node(b, data, i),
        Node::Min(a, b) => eval_node(a, data, i).min(eval_node(b, data, i)),
        Node::Max(a, b) => eval_node(a, data, i).max(eval_node(b, data, i)),
        Node::Pow(a, b) => eval_node(a, data, i).powf(eval_node(b, data, i)),
        Node::Neg(a) => -eval_node(a, data, i),
        Node::Call(f, a) => f.apply(eval_node(a, data, i)),
        // `compile` rejects unresolved identifiers, so an `Ident` surviving
        // into evaluation would be a compiler bug rather than user input.
        Node::Ident(name) => unreachable!("unresolved identifier {name:?} reached evaluation"),
    }
}

const PI: f64 = std::f64::consts::PI;
const E: f64 = std::f64::consts::E;

/// Compile an expression string against a declared variable table.
///
/// `variables` fixes the order of the `data` arrays. An identifier that is
/// neither in the table nor a known constant is an error, so a typo becomes a
/// visible compile failure rather than a silently-ignored term.
pub fn compile(expr: &str, variables: &[String]) -> Result<Compiled, ParseError> {
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        return Err(ParseError {
            message: "empty expression".to_string(),
            position: 0,
        });
    }
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
    };
    let parsed = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        return Err(ParseError {
            message: "trailing tokens after the expression".to_string(),
            position: parser.tokens[parser.pos].position,
        });
    }

    let mut table: HashMap<String, usize> = HashMap::new();
    for (i, name) in variables.iter().enumerate() {
        table.insert(name.clone(), i);
    }
    let root = resolve(&parsed, &table)?;

    Ok(Compiled {
        root,
        num_vars: variables.len(),
        variables: variables.to_vec(),
        source: expr.to_string(),
    })
}

fn resolve(node: &Node, table: &HashMap<String, usize>) -> Result<Node, ParseError> {
    Ok(match node {
        Node::Const(c) => Node::Const(*c),
        Node::Ident(name) => match name.as_str() {
            "pi" | "Pi" => Node::Const(PI),
            "E" => Node::Const(E),
            _ => match table.get(name) {
                Some(&slot) => Node::Var(slot),
                None => {
                    return Err(ParseError {
                        message: format!("unknown identifier {name:?}; it is not a declared variable, a known constant, or a supported function"),
                        position: 0,
                    })
                }
            },
        },
        Node::Var(_) => {
            return Err(ParseError {
                message: "internal: already-resolved node reached resolution".to_string(),
                position: 0,
            })
        }
        Node::Add(a, b) => Node::Add(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Sub(a, b) => Node::Sub(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Mul(a, b) => Node::Mul(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Div(a, b) => Node::Div(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Pow(a, b) => Node::Pow(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Neg(a) => Node::Neg(Box::new(resolve(a, table)?)),
        Node::Min(a, b) => Node::Min(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Max(a, b) => Node::Max(
            Box::new(resolve(a, table)?),
            Box::new(resolve(b, table)?),
        ),
        Node::Call(f, a) => Node::Call(*f, Box::new(resolve(a, table)?)),
    })
}

struct Parser<'a> {
    tokens: &'a [Spanned],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|s| &s.token)
    }

    fn position(&self) -> usize {
        self.tokens
            .get(self.pos)
            .or_else(|| self.tokens.last())
            .map(|s| s.position)
            .unwrap_or(0)
    }

    fn bump(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn error<T>(&self, message: &str) -> Result<T, ParseError> {
        Err(ParseError {
            message: message.to_string(),
            position: self.position(),
        })
    }

    /// expr := term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.bump();
                    let rhs = self.parse_term()?;
                    lhs = Node::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(Token::Minus) => {
                    self.bump();
                    let rhs = self.parse_term()?;
                    lhs = Node::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => return Ok(lhs),
            }
        }
    }

    /// term := unary (('*' | '/') unary | implicit-mul unary)*
    fn parse_term(&mut self) -> Result<Node, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.bump();
                    let rhs = self.parse_unary()?;
                    lhs = Node::Mul(Box::new(lhs), Box::new(rhs));
                }
                Some(Token::Slash) => {
                    self.bump();
                    let rhs = self.parse_unary()?;
                    lhs = Node::Div(Box::new(lhs), Box::new(rhs));
                }
                // Implicit multiplication. Without this the reference side
                // would accept `2x` and return a number while this side raised,
                // and the differential suite would report a divergence that is
                // really only a grammar gap.
                Some(Token::Number(_)) | Some(Token::Ident(_)) | Some(Token::LParen) => {
                    let rhs = self.parse_unary()?;
                    lhs = Node::Mul(Box::new(lhs), Box::new(rhs));
                }
                _ => return Ok(lhs),
            }
        }
    }

    /// unary := ('+' | '-') unary | power
    fn parse_unary(&mut self) -> Result<Node, ParseError> {
        match self.peek() {
            Some(Token::Minus) => {
                self.bump();
                Ok(Node::Neg(Box::new(self.parse_unary()?)))
            }
            Some(Token::Plus) => {
                self.bump();
                self.parse_unary()
            }
            _ => self.parse_power(),
        }
    }

    /// power := atom ('**' unary)? -- right associative, so `2**-1` parses and
    /// `2**3**2` is `2**(3**2)` as in Python and sympy.
    fn parse_power(&mut self) -> Result<Node, ParseError> {
        let base = self.parse_atom()?;
        if matches!(self.peek(), Some(Token::Pow)) {
            self.bump();
            let exp = self.parse_unary()?;
            return Ok(Node::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_atom(&mut self) -> Result<Node, ParseError> {
        match self.peek().cloned() {
            Some(Token::Number(v)) => {
                self.bump();
                Ok(Node::Const(v))
            }
            Some(Token::LParen) => {
                self.bump();
                let inner = self.parse_expr()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.bump();
                        Ok(inner)
                    }
                    _ => self.error("expected ')'"),
                }
            }
            Some(Token::Ident(name)) => {
                self.bump();
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.bump();
                    // min/max are the only two-argument forms: min(a, b).
                    let lname = name.to_ascii_lowercase();
                    if lname == "min" || lname == "max" {
                        let first = self.parse_expr()?;
                        if !matches!(self.peek(), Some(Token::Comma)) {
                            return self.error("min/max take exactly two arguments");
                        }
                        self.bump();
                        let second = self.parse_expr()?;
                        match self.peek() {
                            Some(Token::RParen) => {
                                self.bump();
                                let (a, b) = (Box::new(first), Box::new(second));
                                if lname == "min" {
                                    return Ok(Node::Min(a, b));
                                }
                                return Ok(Node::Max(a, b));
                            }
                            _ => return self.error("expected ')' after min/max arguments"),
                        }
                    }
                    let func = match Func::from_name(&name) {
                        Some(f) => f,
                        None => {
                            return Err(ParseError {
                                message: format!("unknown function {name:?}"),
                                position: self.position(),
                            })
                        }
                    };
                    let arg = self.parse_expr()?;
                    if let Some(Token::Comma) = self.peek() {
                        return self.error(
                            "functions in this kernel take exactly one argument; \
                             multi-argument maths must be written as a single expression",
                        );
                    }
                    match self.peek() {
                        Some(Token::RParen) => {
                            self.bump();
                            Ok(Node::Call(func, Box::new(arg)))
                        }
                        _ => self.error("expected ')' after function argument"),
                    }
                } else {
                    Ok(Node::Ident(name))
                }
            }
            Some(other) => Err(ParseError {
                message: format!("unexpected token {other:?}"),
                position: self.position(),
            }),
            None => self.error("expression ended unexpectedly"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn eval(expr: &str, names: &[&str], data: &[&[f64]]) -> Vec<f64> {
        let v = vars(names);
        let owned: Vec<Vec<f64>> = data.iter().map(|d| d.to_vec()).collect();
        compile(expr, &v).unwrap().evaluate(&owned).unwrap()
    }

    /// Evaluate over a two-point dummy variable, so precedence is checked
    /// without depending on how a constant-only expression sizes its output.
    fn eval1(expr: &str) -> f64 {
        eval(expr, &["x"], &[&[7.0, 7.0]])[0]
    }

    #[test]
    fn arithmetic_follows_precedence() {
        assert_eq!(eval1("1 + 2 * 3"), 7.0);
        assert_eq!(eval1("(1 + 2) * 3"), 9.0);
        assert_eq!(eval1("2 ** 3 ** 2"), 512.0, "power is right associative");
        assert_eq!(
            eval("x - 1 - 1", &["x"], &[&[3.0]]),
            vec![1.0],
            "subtraction is left associative"
        );
        assert_eq!(eval("-x + 1", &["x"], &[&[1.0, 2.0]]), vec![0.0, -1.0]);
        assert_eq!(eval1("2 ** -1"), 0.5);
    }

    #[test]
    fn implicit_multiplication_matches_sympy() {
        assert_eq!(eval("2x", &["x"], &[&[3.0]]), vec![6.0]);
        assert_eq!(eval("3(x+1)", &["x"], &[&[1.0]]), vec![6.0]);
        assert_eq!(eval("2x**2", &["x"], &[&[3.0]]), vec![18.0]);
    }

    #[test]
    fn functions_and_constants() {
        assert!(eval("sin(x)", &["x"], &[&[0.0]])[0].abs() < 1e-15);
        assert!((eval("log(x)", &["x"], &[&[std::f64::consts::E]])[0] - 1.0).abs() < 1e-15);
        assert!((eval("ln(x)", &["x"], &[&[std::f64::consts::E]])[0] - 1.0).abs() < 1e-15);
        assert!((eval("sqrt(x)", &["x"], &[&[4.0]])[0] - 2.0).abs() < 1e-15);
        assert!(eval("tanh(x)", &["x"], &[&[0.0]])[0].abs() < 1e-15);
        assert!(
            (eval("log(E)", &["x"], &[&[1.0]])[0] - 1.0).abs() < 1e-15,
            "`E` is a named constant"
        );
        assert!((eval("pi", &["x"], &[&[1.0]])[0] - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn a_variable_free_expression_cannot_infer_a_row_count() {
        // Silently returning an empty array would read as "evaluated zero
        // points successfully", which is a different claim from "no length".
        let c = compile("1 + 2*3", &vars(&[])).unwrap();
        let e = c.evaluate::<Vec<f64>>(&[]).unwrap_err();
        assert!(e.contains("no variables"), "{e}");
    }

    #[test]
    fn domain_edges_are_inf_and_nan_not_panics() {
        let r = eval("1.0/x", &["x"], &[&[0.0, 2.0]]);
        assert!(r[0].is_infinite(), "1/0 must be inf, not a panic");
        assert_eq!(r[1], 0.5);
        assert!(eval("sqrt(x)", &["x"], &[&[-1.0]])[0].is_nan());
    }

    #[test]
    fn unknown_identifiers_are_errors_not_silent_constants() {
        let e = compile("y + 1", &vars(&["x"])).unwrap_err();
        assert!(e.message.contains("unknown identifier"), "{e}");
    }

    #[test]
    fn a_wrong_number_of_arrays_is_an_error_not_a_partial_read() {
        let one = compile("x + 1", &vars(&["x"])).unwrap();
        assert!(one.evaluate(&[vec![1.0, 2.0], vec![1.0, 2.0]]).is_err());

        let c = compile(
            "a0 + a1 * x + a2 * Abs(v) * v",
            &vars(&["x", "v", "a0", "a1", "a2"]),
        )
        .unwrap();
        let pt = [2.0, -3.0, 1.0, 4.0, 0.5];
        let res = c.eval_point(&pt);
        assert!((res - 4.5).abs() < 1e-12, "got {res}, expected 4.5");

        let two = compile("x + y", &vars(&["x", "y"])).unwrap();
        assert!(two
            .evaluate(&[vec![1.0, 2.0], vec![1.0, 2.0, 3.0]])
            .unwrap_err()
            .contains("same length"));
    }
}
