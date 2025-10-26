//! Token generator module for defining custom token lists for lexing.
//!
//! This module provides a flexible system for developers to define their own
//! token patterns and generate efficient tokenizers using the trie-based matching system.
//!
//! # Example
//!
//! ```rust
//! use plurize::token::{TokenDefinition, TokenGenerator, TokenCategory};
//! use plurize::trie::Trie;
//!
//! // Define some tokens
//! let tokens = vec![
//!     TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
//!     TokenDefinition::new("fn", "KEYWORD_FN", TokenCategory::Keyword),
//!     TokenDefinition::new("==", "OP_EQ", TokenCategory::Operator),
//!     TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
//!     TokenDefinition::new("+", "OP_PLUS", TokenCategory::Operator),
//! ];
//!
//! // Generate a tokenizer
//! let generator = TokenGenerator::new();
//! let tokenizer = generator.generate_tokenizer(tokens);
//!
//! // Use the tokenizer
//! let code = "let x = 5";
//! let mut pos = 0;
//! while pos < code.len() {
//!     if let Some((end, token_type)) = tokenizer.match_longest_from(code, pos) {
//!         println!("Token: {} -> {}", &code[pos..end], token_type);
//!         pos = end;
//!     } else {
//!         // Handle non-matching characters
//!         if let Some(ch) = code.chars().nth(pos) {
//!             pos += ch.len_utf8();
//!         } else {
//!             break;
//!         }
//!     }
//! }
//! ```

pub mod generator;
pub mod types;

pub use generator::{TokenGenerator, Tokenizer, TokenizerConfig};
pub use types::{TokenDefinition, TokenCategory, TokenType};
