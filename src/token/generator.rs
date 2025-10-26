//! Token generator for creating lexers from token definitions.

use crate::trie::Trie;
use super::types::{TokenDefinition, TokenCategory, TokenType};
use std::collections::HashMap;

/// Generator for creating lexers from token definitions.
///
/// This struct provides a builder pattern for configuring and generating
/// efficient lexers using the trie-based matching system.
#[derive(Debug, Default)]
pub struct TokenGenerator {
    /// Whether to enable longest-match precedence (default: true)
    longest_match: bool,
    /// Whether to prefer operators over keywords when they share prefixes
    prefer_operators: bool,
}

impl TokenGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables or disables longest-match precedence.
    /// When enabled, longer matches are preferred over shorter ones.
    /// Default: true
    pub fn with_longest_match(mut self, enabled: bool) -> Self {
        self.longest_match = enabled;
        self
    }

    /// Sets whether operators should be preferred over keywords when they share prefixes.
    /// This affects matching precedence for tokens like "=" (assignment) vs "==" (equality).
    /// Default: false (keywords have higher precedence)
    pub fn with_operator_preference(mut self, enabled: bool) -> Self {
        self.prefer_operators = enabled;
        self
    }

    /// Generates a tokenizer from a collection of token definitions.
    ///
    /// # Arguments
    /// * `tokens` - Vector of token definitions to include in the tokenizer
    ///
    /// # Returns
    /// A `Tokenizer` instance configured with the provided tokens
    pub fn generate_tokenizer(&self, tokens: Vec<TokenDefinition>) -> Tokenizer {
        let mut tokenizer = Tokenizer {
            trie: Trie::new(),
            category_tries: HashMap::new(),
            longest_match: self.longest_match,
            prefer_operators: self.prefer_operators,
        };

        for token in tokens {
            tokenizer.trie.insert(&token.pattern, token.token_type.clone());
            tokenizer.category_tries
                .entry(token.category)
                .or_insert_with(|| Trie::new())
                .insert(&token.pattern, token.token_type);
        }

        tokenizer
    }
}

/// A tokenizer generated from token definitions.
///
/// This struct contains the compiled trie and matching logic for efficient
/// token recognition in source code.
#[derive(Debug)]
pub struct Tokenizer {
    trie: Trie<TokenType>,
    category_tries: HashMap<TokenCategory, Trie<TokenType>>,
    longest_match: bool,
    prefer_operators: bool,
}

impl Tokenizer {
    /// Creates a new empty tokenizer.
    pub fn new() -> Self {
        Self {
            trie: Trie::new(),
            category_tries: HashMap::new(),
            longest_match: true,
            prefer_operators: false,
        }
    }

    /// Matches the longest token starting at the given position in the text.
    ///
    /// # Arguments
    /// * `text` - The source text to scan
    /// * `start` - The starting position in the text
    ///
    /// # Returns
    /// * `Some((end_pos, token_type))` if a match is found
    /// * `None` if no token matches at the given position
    pub fn match_longest_from(&self, text: &str, start: usize) -> Option<(usize, &str)> {
        self.trie.match_longest_from(text, start)
            .map(|(end, token_type)| (end, token_type.as_str()))
    }

    /// Matches any token starting at the given position, considering precedence rules.
    ///
    /// This method applies category-based precedence when multiple tokens could match.
    ///
    /// # Arguments
    /// * `text` - The source text to scan
    /// * `start` - The starting position in the text
    ///
    /// # Returns
    /// * `Some((end_pos, token_type))` if a match is found
    /// * `None` if no token matches at the given position
    pub fn match_with_precedence(&self, text: &str, start: usize) -> Option<(usize, &str)> {
        if let Some((end, token_type)) = self.match_longest_from(text, start) {
            return Some((end, token_type));
        }

        for trie in self.category_tries.values() {
            if let Some((end, token_type)) = trie.match_longest_from(text, start) {
                return Some((end, token_type.as_str()));
            }
        }

        None
    }

    /// Returns all token types that match at the given position.
    /// Useful for debugging and understanding matching behavior.
    ///
    /// # Arguments
    /// * `text` - The source text to scan
    /// * `start` - The starting position in the text
    ///
    /// # Returns
    /// Vector of (end_position, token_type) pairs that match
    pub fn match_all_at(&self, text: &str, start: usize) -> Vec<(usize, String)> {
        let mut matches = Vec::new();

        // Check main trie
        if let Some((end, token_type)) = self.trie.match_longest_from(text, start) {
            matches.push((end, token_type.clone()));
        }

        // Check category tries
        for trie in self.category_tries.values() {
            if let Some((end, token_type)) = trie.match_longest_from(text, start) {
                matches.push((end, token_type.clone()));
            }
        }

        matches
    }

    /// Returns the underlying trie for advanced usage.
    pub fn trie(&self) -> &Trie<TokenType> {
        &self.trie
    }

    /// Returns the category-specific tries.
    pub fn category_tries(&self) -> &HashMap<TokenCategory, Trie<TokenType>> {
        &self.category_tries
    }

    /// Returns configuration information about this tokenizer.
    pub fn config(&self) -> TokenizerConfig {
        TokenizerConfig {
            longest_match: self.longest_match,
            prefer_operators: self.prefer_operators,
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration information for a tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerConfig {
    /// Whether longest-match precedence is enabled
    pub longest_match: bool,
    /// Whether operators have precedence over keywords
    pub prefer_operators: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            longest_match: true,
            prefer_operators: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::TokenCategory;

    #[test]
    fn simple_lexer_generation() {
        let generator = TokenGenerator::new();
        let tokens = vec![
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("fn", "KEYWORD_FN", TokenCategory::Keyword),
            TokenDefinition::new("==", "OP_EQ", TokenCategory::Operator),
            TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
        ];

        let tokenizer = generator.generate_tokenizer(tokens);

        let test_cases = [
            ("let x = 5", (0, Some((3, "KEYWORD_LET")))),
            ("fn foo()", (0, Some((2, "KEYWORD_FN")))),
            ("x == y", (2, Some((4, "OP_EQ")))),  // "==" starts at position 2, ends at position 4
            ("x = y", (2, Some((3, "OP_ASSIGN")))), // "=" starts at position 2, ends at position 3
        ];

        for (text, (start_pos, expected)) in test_cases {
            if let Some((expected_end, expected_token)) = expected {
                if let Some((end, token)) = tokenizer.match_longest_from(text, start_pos) {
                    assert_eq!(end, expected_end, "Failed for text: {}", text);
                    assert_eq!(token, expected_token, "Failed for text: {}", text);
                } else {
                    panic!("Expected match for text: {} at position {}", text, start_pos);
                }
            }
        }
    }

    #[test]
    fn longest_match_precedence() {
        let generator = TokenGenerator::new().with_longest_match(true);
        let tokens = vec![
            TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
            TokenDefinition::new("==", "OP_EQ", TokenCategory::Operator),
            TokenDefinition::new("===", "OP_STRICT_EQ", TokenCategory::Operator),
        ];

        let tokenizer = generator.generate_tokenizer(tokens);

        let text = "x === y";
        if let Some((end, token)) = tokenizer.match_longest_from(text, 2) {
            assert_eq!(end, 5, "Should match '===' not '=='");
            assert_eq!(token, "OP_STRICT_EQ");
        } else {
            panic!("Expected match for '==='");
        }
    }

    #[test]
    fn common_lexer_has_expected_tokens() {
        let generator = TokenGenerator::new();
        let tokens = vec![
            // Keywords
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("fn", "KEYWORD_FN", TokenCategory::Keyword),
            TokenDefinition::new("if", "KEYWORD_IF", TokenCategory::Keyword),
            TokenDefinition::new("else", "KEYWORD_ELSE", TokenCategory::Keyword),
            TokenDefinition::new("while", "KEYWORD_WHILE", TokenCategory::Keyword),
            TokenDefinition::new("for", "KEYWORD_FOR", TokenCategory::Keyword),
            TokenDefinition::new("return", "KEYWORD_RETURN", TokenCategory::Keyword),

            // Operators
            TokenDefinition::new("==", "OP_EQ", TokenCategory::Operator),
            TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
            TokenDefinition::new("!=", "OP_NEQ", TokenCategory::Operator),
            TokenDefinition::new("+", "OP_PLUS", TokenCategory::Operator),
            TokenDefinition::new("-", "OP_MINUS", TokenCategory::Operator),
            TokenDefinition::new("*", "OP_MULTIPLY", TokenCategory::Operator),
            TokenDefinition::new("/", "OP_DIVIDE", TokenCategory::Operator),
            TokenDefinition::new("<", "OP_LT", TokenCategory::Operator),
            TokenDefinition::new(">", "OP_GT", TokenCategory::Operator),
            TokenDefinition::new("<=", "OP_LE", TokenCategory::Operator),
            TokenDefinition::new(">=", "OP_GE", TokenCategory::Operator),
            TokenDefinition::new("&&", "OP_AND", TokenCategory::Operator),
            TokenDefinition::new("||", "OP_OR", TokenCategory::Operator),
            TokenDefinition::new("!", "OP_NOT", TokenCategory::Operator),

            // Literals
            TokenDefinition::new("true", "LITERAL_BOOL", TokenCategory::Literal),
            TokenDefinition::new("false", "LITERAL_BOOL", TokenCategory::Literal),
            TokenDefinition::new("null", "LITERAL_NULL", TokenCategory::Literal),

            // Symbols
            TokenDefinition::new("{", "SYMBOL_LBRACE", TokenCategory::Symbol),
            TokenDefinition::new("}", "SYMBOL_RBRACE", TokenCategory::Symbol),
            TokenDefinition::new("(", "SYMBOL_LPAREN", TokenCategory::Symbol),
            TokenDefinition::new(")", "SYMBOL_RPAREN", TokenCategory::Symbol),
            TokenDefinition::new("[", "SYMBOL_LBRACKET", TokenCategory::Symbol),
            TokenDefinition::new("]", "SYMBOL_RBRACKET", TokenCategory::Symbol),
            TokenDefinition::new(";", "SYMBOL_SEMICOLON", TokenCategory::Symbol),
            TokenDefinition::new(",", "SYMBOL_COMMA", TokenCategory::Symbol),
            TokenDefinition::new(".", "SYMBOL_DOT", TokenCategory::Symbol),
        ];

        let tokenizer = generator.generate_tokenizer(tokens);

        // Test keywords
        assert!(tokenizer.trie.get("let").is_some());
        assert!(tokenizer.trie.get("fn").is_some());
        assert!(tokenizer.trie.get("if").is_some());

        // Test operators
        assert!(tokenizer.trie.get("==").is_some());
        assert!(tokenizer.trie.get("=").is_some());
        assert!(tokenizer.trie.get("+").is_some());

        // Test symbols
        assert!(tokenizer.trie.get("{").is_some());
        assert!(tokenizer.trie.get("}").is_some());
        assert!(tokenizer.trie.get("(").is_some());
        assert!(tokenizer.trie.get(")").is_some());
    }

    #[test]
    fn category_separation() {
        let tokens = vec![
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("==", "OP_EQ", TokenCategory::Operator),
        ];

        let generator = TokenGenerator::new();
        let tokenizer = generator.generate_tokenizer(tokens);

        // Should have separate category tries
        assert!(tokenizer.category_tries.contains_key(&TokenCategory::Keyword));
        assert!(tokenizer.category_tries.contains_key(&TokenCategory::Operator));

        // Main trie should contain both
        assert!(tokenizer.trie.get("let").is_some());
        assert!(tokenizer.trie.get("==").is_some());
    }

    #[test]
    fn match_all_at_debugging() {
        let tokens = vec![
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("letter", "IDENTIFIER", TokenCategory::Custom("identifier")),
        ];

        let generator = TokenGenerator::new();
        let tokenizer = generator.generate_tokenizer(tokens);

        let matches = tokenizer.match_all_at("letter", 0);
        assert!(!matches.is_empty());

        // Should find both "let" (prefix) and "letter" (full match)
        let match_types: Vec<&str> = matches.iter().map(|(_, t)| t.as_str()).collect();
        assert!(match_types.contains(&"KEYWORD_LET"));
        assert!(match_types.contains(&"IDENTIFIER"));
    }
}
