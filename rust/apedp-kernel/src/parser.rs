//! Expression parser for the APEDP numerical kernel.
//!
//! This is deliberately an *independent* implementation rather than a binding
//! to sympy. The whole point of `tests/test_cross_implementation_conformance.py`
//! is to compare two implementations that were written separately; a Rust
//! parser that called into sympy would make the differential suite a tautology.
//!
//! The grammar covers what the search pipeline emits after
//! `apedp.safety.safe_parse_expr` has allowlisted it, which is arithmetic plus a
//! small transcendental set. Implicit multiplication (`2x`, `3(x+1)`) is
//! accepted because sympy accepts it and the reference side will.

use std::collections::HashMap;
use std::fmt;

/// A node in the parsed expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Const(f64),
    /// Index into the variable table, resolved at compile time.
    Var(usize),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
    /// `**`. The exponent is always a scalar node in practice, but it is kept
    /// as a full expression so `2**(x+1)` parses the way sympy parses it.
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
}

impl Func {
    fn from_name(name: &str) -> Option<Func> {
        Some(match name {
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "asin" => Func::Asin,
            "acos" => Func::Acos,
            "atan" => Func::Atan,
            "arcsin" => Func::Asin,
            "arccos" => Func::Acos,
            "arctan" => Func::Atan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "tanh" => Func::Tanh,
            "exp" => Func::Exp,
            "log" => Func::Ln,
            "ln" => Func::Ln,
            "log10" => Func::Log10,
            "log2" => Func::Log2,
            "sqrt" => Func::Sqrt,
            "abs" => Func::Abs,
            "sign" => Func::Sign,
            "floor" => Func::Floor,
            _ => return None,
        })
    }

    #[inline]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            // Domain behaviour must match numpy/sympy rather than Rust's
            // default, because the differential test compares bit patterns.
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
                    x // preserves -0.0 and NaN, as numpy does
                }
            }
            Func::Floor => x.floor(),
        }
    }
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Asin => "asin",
            Func::Acos => "acos",
            Func::Atan => "atan",
            Func::Sinh => "sinh",
            Func::Cosh => "cosh",
            Func::Tanh => "tanh",
            Func::Exp => "exp",
            Func::Ln => "ln",
            Func::Log10 => "log10",
            Func::Log2 => "log2",
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
            Func::Sign => "sign",
            Func::Floor => "floor",
        };
        write!(f, "{s}")
    }
}

/// Raised when an expression cannot be compiled. The bridge treats this as
/// "this candidate is not evaluable by the fast path", never as a verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at position {}",
            self.message, self.position
        )
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
    let bytes: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
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
                // `**` is power; a single `*` is multiplication.
                if i + 1 < bytes.len() && bytes[i + 1] == '*' {
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
            c if c.is_ascii_digit() || (c == '.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()) => {
                let mut seen_dot = false;
                let mut seen_exp = false;
                while i < bytes.len() {
                    let d = bytes[i];
                    if d.is_ascii_digit() {
                        i += 1;
                    } else if d == '.' && !seen_dot && !seen_exp {
                        seen_dot = true;
                        i += 1;
                    } else if (d == 'e' || d == 'E') && !seen_exp {
                        // Only consume the exponent if it is well formed, so
                        // that `2e` stays a parse error rather than silently
                        // becoming `2 * e`.
                        let next = i + 1;
                        if next < bytes.len()
                            && (bytes[next].is_ascii_digit()
                                || ((bytes[next] == '+' || bytes[next] == '-')
                                    && next + 1 < bytes.len()
                                    && bytes[next + 1].is_ascii_digit()))
                        {
                            seen_exp = true;
                            i += 1;
                            if bytes[i] == '+' || bytes[i] == '-' {
                                i += 1;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                let text: String = bytes[start..i].iter().collect();
                let value = text.parse::<f64>().map_err(|_| ParseError {
                    message: format!("malformed number {text:?}"),
                    position: start,
                })?;
                Token::Number(value)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '_') {
                    i += 1;
                }
                Token::Ident(bytes[start..i].iter().collect())
            }
            other => {
                return Err(ParseError {
                    message: format!("unexpected character {other:?}"),
                    position: start,
                })
            }
        };
        out.push(Spanned { token, position: start });
    }

    Ok(out)
}

/// A parsed, variable-resolved expression ready to evaluate.
#[derive(Debug, Clone)]
pub struct Compiled {
    root: Node,
    num_vars: usize,
    /// Variable names in table order, so the evaluator can be checked against
    /// the caller's `data` ordering at the boundary.
    pub variables: Vec<String>,
    pub source: String,
}

impl Compiled {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Evaluate over `data`, one array per declared variable.
    ///
    /// Returns a vector of length `data[0].len()`. Any arithmetic that numpy
    /// would render as inf/NaN (division by zero, sqrt of a negative) renders
    /// the same way here, because Rust's f64 does not trap.
    pub fn evaluate(&self, data: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if data.len() != self.num_vars {
            return Err(format!(
                "expression '{}' declares {} variable(s) but {} array(s) were supplied",
                self.source,
                self.num_vars,
                data.len()
            ));
        }
        let n = if data.is_empty() {
            0
        } else {
            for (i, col) in data.iter().enumerate() {
                if col.len() != data[0].len() {
                    return Err(format!(
                        "variable '{}' has length {} but '{}' has length {}; all data arrays must match",
                        self.variables[i], col.len(), self.variables[0], data[0].len()
                    ));
                }
            }
            data[0].len()
        };

        let mut out = vec![0.0f64; n];
        for (idx, slot) in out.iter_mut().enumerate() {
            *slot = eval_node(&self.root, data, idx);
        }
        Ok(out)
    }
}

#[inline]
fn eval_node(node: &Node, data: &[Vec<f64>], i: usize) -> f64 {
    match node {
        Node::Const(c) => *c,
        Node::Var(v) => data[*v][i],
        Node::Add(a, b) => eval_node(a, data, i) + eval_node(b, data, i),
        Node::Sub(a, b) => eval_node(a, data, i) - eval_node(b, data, i),
        Node::Mul(a, b) => eval_node(a, data, i) * eval_node(b, data, i),
        Node::Div(a, b) => eval_node(a, data, i) / eval_node(b, data, i),
        Node::Pow(a, b) => {
            let base = eval_node(a, data, i);
            let exp = eval_node(b, data, i);
            base.powf(exp)
        }
        Node::Neg(a) => -eval_node(a, data, i),
        Node::Call(f, a) => f.apply(eval_node(a, data, i)),
    }
}

/// Every free identifier in the tree, in first-seen order.
fn collect_vars(node: &Node, out: &mut Vec<usize>) {
    match node {
        Node::Const(_) => {}
        Node::Var(v) => {
            if !out.contains(v) {
                out.push(*v)
            }
        }
        Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) | Node::Pow(a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Node::Neg(a) | Node::Call(_, a) => collect_vars(a, out),
    }
}

/// Compile an expression string against a declared variable table.
///
/// `variables` fixes the order of the `data` arrays. Names used by the
/// expression but absent from the table are an error, so a typo becomes a
/// visible compile failure instead of a silent constant.
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
        constants: HashMap::new(),
    };
    let root = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        let p = parser.tokens[parser.pos].position;
        return Err(ParseError {
            message: "trailing tokens after the expression".to_string(),
            position: p,
        });
    }

    // Resolve identifiers to slots. `pi` and `e` are constants, matching the
    // reference engine's environment.
    let mut table: HashMap<String, usize> = HashMap::new();
    for (i, name) in variables.iter().enumerate() {
        table.insert(name.clone(), i);
    }
    let root = resolve(&root, &table, &parser.constants)?;
    debug_assert_eq!(collect_vars_checked(&root), root_var_count(&root));

    Ok(Compiled {
        root,
        num_vars: variables.len(),
        variables: variables.to_vec(),
        source: expr.to_string(),
    })
}

fn root_var_count(node: &Node) -> usize {
    let mut out = Vec::new();
    collect_vars(node, &mut out);
    out.len()
}

fn collect_vars_checked(node: &Node) -> usize {
    root_var_count(node)
}

fn resolve(
    node: &Node,
    table: &HashMap<String, usize>,
    constants: &HashMap<String, f64>,
) -> Result<Node, ParseError> {
    Ok(match node {
        Node::Const(c) => Node::Const(*c),
        Node::Add(a, b) => Node::Add(
            Box::new(resolve(a, table, constants)?),
            Box::new(resolve(b, table, constants)?),
        ),
        Node::Sub(a, b) => Node::Sub(
            Box::new(resolve(a, table, constants)?),
            Box::new(resolve(b, table, constants)?),
        ),
        Node::Mul(a, b) => Node::Mul(
            Box::new(resolve(a, table, constants)?),
            Box::new(resolve(b, table, constants)?),
        ),
        Node::Div(a, b) => Node::Div(
            Box::new(resolve(a, table, constants)?),
            Box::new(resolve(b, table, constants)?),
        ),
        Node::Pow(a, b) => Node::Pow(
            Box::new(resolve(a, table, constants)?),
            Box::new(resolve(b, table, constants)?),
        ),
        Node::Neg(a) => Node::Neg(Box::new(resolve(a, table, constants)?)),
        Node::Call(f, a) => Node::Call(*f, Box::new(resolve(a, table, constants)?)),
        // Resolved lazily below; see the Ident arm of the parser.
        Node::Var(_) => return Err(ParseError {
            message: "internal: unresolved node reached resolution".to_string(),
            position: 0,
        }),
    })
}

struct Parser<'a> {
    tokens: &'a [Spanned],
    pos: usize,
    /// Named constants discovered while parsing, checked after resolution.
    constants: HashMap<String, f64>,
}

/// Pre-seeded constants. `pi` and `e` are the two the reference engine's
/// fallback environment defines.
const PI: f64 = std::f64::consts::PI;
const E: f64 = std::f64::consts::E;

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|s| &s.token)
    }

    fn position(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map(|s| s.position)
            .unwrap_or_else(|| {
                self.tokens
                    .last()
                    .map(|s| s.position)
                    .unwrap_or_default()
            })
    }

    fn bump(&mut self) -> Option<&'a Token> {
        let t = self.tokens.get(self.pos).map(|s| &s.token);
        if t.is_some() {
            self.pos += 1;
        }
        t
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

    /// term := unary (('*' | '/') unary | implicit-mul atom)*
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
                // Implicit multiplication: sympy accepts `2x` and `3(x+1)`,
                // so the reference side would accept them and this side must
                // too, or the differential suite would report a divergence
                // that is really just a grammar gap.
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

    /// power := atom ('**' unary)?   -- right associative, so `2**-1` works and
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
                             compound expressions must be written as calls on a single expression",
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
                    // Bare identifier: a variable or a named constant. Resolved
                    // against the caller's variable table in `compile`.
                    match name.as_str() {
                        "pi" | "Pi" => Ok(Node::Const(PI)),
                        "E" => Ok(Node::Const(E)),
                        _ => {
                            self.constants.insert(name.clone(), 0.0);
                            Ok(Node::Var(usize::MAX).named(name))
                        }
                    }
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