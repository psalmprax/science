//! Symbolic AST tokenization, parsing, and bytecode compilation.

use crate::bytecode::{CompiledExpression, OpCode};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

pub struct Tokenizer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        while let Some(&(i, ch)) = self.chars.peek() {
            if ch.is_whitespace() {
                self.chars.next();
                continue;
            }

            match ch {
                '+' => {
                    self.chars.next();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    self.chars.next();
                    tokens.push(Token::Minus);
                }
                '*' => {
                    self.chars.next();
                    if let Some(&(_, '*')) = self.chars.peek() {
                        self.chars.next();
                        tokens.push(Token::Caret); // '**' treated as power
                    } else {
                        tokens.push(Token::Star);
                    }
                }
                '/' => {
                    self.chars.next();
                    tokens.push(Token::Slash);
                }
                '^' => {
                    self.chars.next();
                    tokens.push(Token::Caret);
                }
                '(' => {
                    self.chars.next();
                    tokens.push(Token::LParen);
                }
                ')' => {
                    self.chars.next();
                    tokens.push(Token::RParen);
                }
                ',' => {
                    self.chars.next();
                    tokens.push(Token::Comma);
                }
                '0'..='9' | '.' => {
                    let num = self.read_number(i)?;
                    tokens.push(Token::Number(num));
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_ident(i);
                    tokens.push(Token::Ident(ident));
                }
                _ => return Err(format!("Unexpected character '{}' at offset {}", ch, i)),
            }
        }

        Ok(tokens)
    }

    fn read_number(&mut self, start_idx: usize) -> Result<f64, String> {
        let mut end_idx = start_idx;
        let mut has_e = false;

        while let Some(&(i, ch)) = self.chars.peek() {
            if ch.is_ascii_digit() || ch == '.' {
                self.chars.next();
                end_idx = i + ch.len_utf8();
            } else if (ch == 'e' || ch == 'E') && !has_e {
                has_e = true;
                self.chars.next();
                end_idx = i + ch.len_utf8();
                if let Some(&(_, next_ch)) = self.chars.peek() {
                    if next_ch == '+' || next_ch == '-' {
                        let (sign_idx, _) = self.chars.next().unwrap();
                        end_idx = sign_idx + 1;
                    }
                }
            } else {
                break;
            }
        }

        let slice = &self.input[start_idx..end_idx];
        slice
            .parse::<f64>()
            .map_err(|e| format!("Failed to parse number '{}': {}", slice, e))
    }

    fn read_ident(&mut self, start_idx: usize) -> String {
        let mut end_idx = start_idx;
        while let Some(&(i, ch)) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                self.chars.next();
                end_idx = i + ch.len_utf8();
            } else {
                break;
            }
        }
        self.input[start_idx..end_idx].to_string()
    }
}

pub struct ExpressionCompiler;

impl ExpressionCompiler {
    /// Compiles an infix expression string into a linear bytecode sequence using Shunting-Yard.
    pub fn compile(expr_str: &str, variable_names: &[String]) -> Result<CompiledExpression, String> {
        let tokenizer = Tokenizer::new(expr_str);
        let tokens = tokenizer.tokenize()?;

        let mut output_queue: Vec<OpCode> = Vec::new();
        let mut op_stack: Vec<(Token, bool)> = Vec::new(); // (Token, is_unary)

        let mut prev_token: Option<Token> = None;

        for token in tokens {
            match token {
                Token::Number(num) => {
                    output_queue.push(OpCode::LoadConst(num));
                }
                Token::Ident(ref name) => {
                    match name.as_str() {
                        "sqrt" | "exp" | "log" | "ln" | "sin" | "cos" | "tan" | "tanh" => {
                            op_stack.push((token.clone(), false));
                        }
                        _ => {
                            // Find variable index
                            if let Some(pos) = variable_names.iter().position(|v| v == name) {
                                output_queue.push(OpCode::LoadVar(pos as u16));
                            } else if name == "pi" || name == "PI" {
                                output_queue.push(OpCode::LoadConst(std::f64::consts::PI));
                            } else if name == "e" || name == "E" {
                                output_queue.push(OpCode::LoadConst(std::f64::consts::E));
                            } else {
                                return Err(format!("Unknown variable or function: '{}'", name));
                            }
                        }
                    }
                }
                Token::LParen => {
                    op_stack.push((Token::LParen, false));
                }
                Token::RParen => {
                    let mut found_lparen = false;
                    while let Some((top_op, is_u)) = op_stack.pop() {
                        if top_op == Token::LParen {
                            found_lparen = true;
                            // Check if function was before LParen
                            if let Some(&(Token::Ident(_), _)) = op_stack.last() {
                                let (fn_token, _) = op_stack.pop().unwrap();
                                if let Token::Ident(fn_name) = fn_token {
                                    match fn_name.as_str() {
                                        "sqrt" => output_queue.push(OpCode::Sqrt),
                                        "exp" => output_queue.push(OpCode::Exp),
                                        "log" | "ln" => output_queue.push(OpCode::Ln),
                                        "sin" => output_queue.push(OpCode::Sin),
                                        "cos" => output_queue.push(OpCode::Cos),
                                        "tan" => output_queue.push(OpCode::Tan),
                                        "tanh" => output_queue.push(OpCode::Tanh),
                                        _ => return Err(format!("Unknown function: '{}'", fn_name)),
                                    }
                                }
                            }
                            break;
                        } else {
                            if is_u && top_op == Token::Minus {
                                output_queue.push(OpCode::Neg);
                            } else {
                                output_queue.push(Self::token_to_opcode(&top_op)?);
                            }
                        }
                    }
                    if !found_lparen {
                        return Err("Mismatched parentheses".to_string());
                    }
                }
                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Caret => {
                    // Check if unary minus
                    let is_unary = match prev_token {
                        None | Some(Token::LParen) | Some(Token::Plus) | Some(Token::Minus)
                        | Some(Token::Star) | Some(Token::Slash) | Some(Token::Caret) => true,
                        _ => false,
                    };

                    if is_unary && token == Token::Minus {
                        op_stack.push((Token::Minus, true));
                    } else if is_unary && token == Token::Plus {
                        // Unary plus is a no-op
                    } else {
                        while let Some(&(ref top_op, top_unary)) = op_stack.last() {
                            if *top_op == Token::LParen {
                                break;
                            }
                            let p1 = Self::precedence(&token, false);
                            let p2 = Self::precedence(top_op, top_unary);
                            let left_assoc = token != Token::Caret;

                            if (left_assoc && p1 <= p2) || (!left_assoc && p1 < p2) {
                                let (popped_op, is_u) = op_stack.pop().unwrap();
                                if is_u && popped_op == Token::Minus {
                                    output_queue.push(OpCode::Neg);
                                } else {
                                    output_queue.push(Self::token_to_opcode(&popped_op)?);
                                }
                            } else {
                                break;
                            }
                        }
                        op_stack.push((token.clone(), false));
                    }
                }
                Token::Comma => {
                    while let Some(&(ref top_op, _)) = op_stack.last() {
                        if *top_op == Token::LParen {
                            break;
                        }
                        let (popped_op, is_u) = op_stack.pop().unwrap();
                        if is_u && popped_op == Token::Minus {
                            output_queue.push(OpCode::Neg);
                        } else {
                            output_queue.push(Self::token_to_opcode(&popped_op)?);
                        }
                    }
                }
            }
            prev_token = Some(token);
        }

        while let Some((top_op, is_u)) = op_stack.pop() {
            if top_op == Token::LParen {
                return Err("Mismatched parentheses at end of expression".to_string());
            }
            if is_u && top_op == Token::Minus {
                output_queue.push(OpCode::Neg);
            } else {
                output_queue.push(Self::token_to_opcode(&top_op)?);
            }
        }

        output_queue.push(OpCode::Halt);

        Ok(CompiledExpression::new(output_queue, variable_names.to_vec()))
    }

    fn precedence(token: &Token, is_unary: bool) -> u8 {
        if is_unary {
            return 4; // Unary minus has high precedence
        }
        match token {
            Token::Plus | Token::Minus => 1,
            Token::Star | Token::Slash => 2,
            Token::Caret => 3,
            _ => 0,
        }
    }

    fn token_to_opcode(token: &Token) -> Result<OpCode, String> {
        match token {
            Token::Plus => Ok(OpCode::Add),
            Token::Minus => Ok(OpCode::Sub),
            Token::Star => Ok(OpCode::Mul),
            Token::Slash => Ok(OpCode::Div),
            Token::Caret => Ok(OpCode::Pow),
            _ => Err(format!("Cannot convert token {:?} to binary opcode", token)),
        }
    }
}
