//! Token generator for creating lexers from token definitions.

use crate::trie::Trie;
use super::types::{TokenDefinition, TokenCategory, TokenType};
use std::collections::HashMap;

/// Options that control literal recognition behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiteralOptions {
    /// Enable recognition of integer literals (e.g., 42, -10)
    pub enable_integer: bool,
    /// Enable recognition of floating-point literals (e.g., 3.14, -0.5)
    pub enable_float: bool,
    /// Allow scientific notation for float literals (e.g., 1e5, 2.5e-3)
    pub allow_scientific: bool,
    /// Enable recognition of string literals
    pub enable_string: bool,
    /// Delimiter to use for strings (default: '"')
    pub string_delimiter: char,
    /// Whether escape sequences like \" and \\ are recognized in strings
    pub allow_string_escape: bool,
    /// Enable recognition of character literals (e.g., 'a', '\n')
    pub enable_character: bool,
    /// Whether character escape sequences are recognized
    pub char_allow_escape: bool,
}

impl Default for LiteralOptions {
    fn default() -> Self {
        Self {
            enable_integer: true,
            enable_float: true,
            allow_scientific: true,
            enable_string: true,
            string_delimiter: '"',
            allow_string_escape: true,
            enable_character: false,
            char_allow_escape: true,
        }
    }
}

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
    /// Literal options for generated tokenizers
    literal_options: LiteralOptions,
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

    /// Sets literal recognition options.
    pub fn with_literal_options(mut self, options: LiteralOptions) -> Self {
        self.literal_options = options;
        self
    }

    pub fn enable_integer_literals(mut self, enabled: bool) -> Self {
        self.literal_options.enable_integer = enabled;
        self
    }

    pub fn enable_float_literals(mut self, enabled: bool) -> Self {
        self.literal_options.enable_float = enabled;
        self
    }

    pub fn allow_scientific_notation(mut self, enabled: bool) -> Self {
        self.literal_options.allow_scientific = enabled;
        self
    }

    pub fn enable_string_literals(mut self, enabled: bool) -> Self {
        self.literal_options.enable_string = enabled;
        self
    }

    pub fn with_string_delimiter(mut self, delimiter: char) -> Self {
        self.literal_options.string_delimiter = delimiter;
        self
    }

    pub fn allow_string_escape_sequences(mut self, enabled: bool) -> Self {
        self.literal_options.allow_string_escape = enabled;
        self
    }

    pub fn enable_character_literals(mut self, enabled: bool) -> Self {
        self.literal_options.enable_character = enabled;
        self
    }

    pub fn allow_char_escape_sequences(mut self, enabled: bool) -> Self {
        self.literal_options.char_allow_escape = enabled;
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
            literal_options: self.literal_options,
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

    /// Generates a tokenizer with common literal support (numbers, strings, etc.).
    ///
    /// # Arguments
    /// * `base_tokens` - Vector of base token definitions to include
    ///
    /// # Returns
    /// A `Tokenizer` instance configured with base tokens and literal recognition
    pub fn generate_tokenizer_with_literals(&self, base_tokens: Vec<TokenDefinition>) -> Tokenizer {
        let mut all_tokens = base_tokens;
        all_tokens.extend(self.get_common_literals());
        self.generate_tokenizer(all_tokens)
    }

    /// Returns common literal token definitions for numbers, strings, and other literals.
    fn get_common_literals(&self) -> Vec<TokenDefinition> {
        vec![
            // Integer patterns (common ones that might be used as keywords or special values)
            TokenDefinition::new("0", "LITERAL_INTEGER", TokenCategory::Integer),
            TokenDefinition::new("1", "LITERAL_INTEGER", TokenCategory::Integer),

            // String delimiters
            TokenDefinition::new("\"", "STRING_DELIMITER", TokenCategory::Symbol),

            // Character delimiters
            TokenDefinition::new("'", "CHAR_DELIMITER", TokenCategory::Symbol),
        ]
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
    literal_options: LiteralOptions,
}

impl Tokenizer {
    /// Creates a new empty tokenizer.
    pub fn new() -> Self {
        Self {
            trie: Trie::new(),
            category_tries: HashMap::new(),
            longest_match: true,
            prefer_operators: false,
            literal_options: LiteralOptions::default(),
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
    /// This method applies category-based precedence when multiple tokens could match,
    /// and also attempts to recognize literals like numbers and strings.
    ///
    /// # Arguments
    /// * `text` - The source text to scan
    /// * `start` - The starting position in the text
    ///
    /// # Returns
    /// * `Some((end_pos, token_type))` if a match is found
    /// * `None` if no token matches at the given position
    pub fn match_with_precedence(&self, text: &str, start: usize) -> Option<(usize, &str)> {
        // First try exact matches from the trie
        if let Some((end, token_type)) = self.match_longest_from(text, start) {
            return Some((end, token_type));
        }

        // Then try category-based matches
        for trie in self.category_tries.values() {
            if let Some((end, token_type)) = trie.match_longest_from(text, start) {
                return Some((end, token_type.as_str()));
            }
        }

        // Finally try literal recognition
        self.match_literal(text, start)
    }

    /// Attempts to match literals (numbers, strings, etc.) at the given position.
    ///
    /// # Arguments
    /// * `text` - The source text to scan
    /// * `start` - The starting position in the text
    ///
    /// # Returns
    /// * `Some((end_pos, token_type))` if a literal is found
    /// * `None` if no literal matches at the given position
    fn match_literal(&self, text: &str, start: usize) -> Option<(usize, &'static str)> {
        if text.len() <= start {
            return None;
        }

        let remaining = &text[start..];

        // Try to match numbers first if enabled
        if (self.literal_options.enable_integer || self.literal_options.enable_float) {
            if let Some((end, literal_type)) = self.match_number_literal(remaining) {
                return Some((start + end, literal_type));
            }
        }

        // Try to match string literals if enabled
        if self.literal_options.enable_string {
            if let Some((end, literal_type)) = self.match_string_literal(remaining) {
                return Some((start + end, literal_type));
            }
        }

        // Try to match character literals if enabled
        if self.literal_options.enable_character {
            if let Some((end, literal_type)) = self.match_char_literal(remaining) {
                return Some((start + end, literal_type));
            }
        }

        None
    }

    /// Attempts to match number literals (integers and floats).
    fn match_number_literal(&self, text: &str) -> Option<(usize, &'static str)> {
        let mut end = 0;
        let chars: Vec<char> = text.chars().collect();
        let mut has_dot = false;
        let mut has_exp = false;

        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                '0'..='9' => {
                    let mut num_end = i + 1;
                    // Consume remaining consecutive digits
                    while num_end < chars.len() && chars[num_end].is_ascii_digit() {
                        num_end += 1;
                    }
                    end = num_end;
                }
                '-' if i == 0 => {
                    end = i + 1;
                }
                '.' if self.literal_options.enable_float && !has_dot && i > 0 && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                    has_dot = true;
                    let mut decimal_end = i + 2;
                    // Consume remaining digits after decimal
                    while decimal_end < chars.len() && chars[decimal_end].is_ascii_digit() {
                        decimal_end += 1;
                    }
                    end = decimal_end;
                }
                'e' | 'E' if self.literal_options.enable_float && self.literal_options.allow_scientific && !has_exp && i > 0 && i + 1 < chars.len() => {
                    has_exp = true;
                    // Handle optional sign and digits
                    let mut exp_pos = i + 1;
                    if chars[i + 1] == '+' || chars[i + 1] == '-' {
                        if i + 2 >= chars.len() || !chars[i + 2].is_ascii_digit() {
                            break; // Invalid: sign without digit
                        }
                        exp_pos = i + 2;
                    } else if !chars[i + 1].is_ascii_digit() {
                        break; // Invalid: no digit after e
                    }

                    // Consume remaining digits
                    while exp_pos < chars.len() && chars[exp_pos].is_ascii_digit() {
                        exp_pos += 1;
                    }

                    end = exp_pos;
                }
                _ => break,
            }
        }

        if end == 0 {
            return None;
        }

        // Determine if it's a float or integer, respecting enabled options
        let is_float = has_dot || has_exp;
        if is_float {
            if self.literal_options.enable_float {
                Some((end, "LITERAL_FLOAT"))
            } else if self.literal_options.enable_integer {
                // Fallback to the longest integer prefix
                let mut int_end = 0usize;
                for (i, ch) in text.char_indices() {
                    match ch {
                        '-' if i == 0 => { int_end = i + 1; }
                        '0'..='9' => { int_end = i + 1; }
                        _ => break,
                    }
                }
                if int_end > 0 { Some((int_end, "LITERAL_INTEGER")) } else { None }
            } else {
                None
            }
        } else {
            if self.literal_options.enable_integer { Some((end, "LITERAL_INTEGER")) } else { None }
        }
    }

    /// Attempts to match string literals using configured delimiter and escape handling.
    fn match_string_literal(&self, text: &str) -> Option<(usize, &'static str)> {
        let chars: Vec<char> = text.chars().collect();
        let delimiter = self.literal_options.string_delimiter;

        if chars.is_empty() || chars[0] != delimiter {
            return None;
        }

        #[derive(Debug, Clone, Copy)]
        enum StringState { InString { escaped: bool } }

        let mut end = 1; // Start after the opening quote
        let mut state = StringState::InString { escaped: false };

        for &ch in &chars[1..] {
            end += 1;
            match state {
                StringState::InString { escaped: true } => {
                    // Previous was backslash; current is escaped if allowed
                    state = StringState::InString { escaped: false };
                }
                StringState::InString { escaped: false } => {
                    if self.literal_options.allow_string_escape && ch == '\\' {
                        state = StringState::InString { escaped: true };
                    } else if ch == delimiter {
                        return Some((end, "LITERAL_STRING"));
                    }
                }
            }
        }

        None
    }

    /// Attempts to match character literals (e.g., 'a', '\n').
    fn match_char_literal(&self, text: &str) -> Option<(usize, &'static str)> {
        let mut it = text.chars();
        let Some(first) = it.next() else { return None };
        if first != '\'' { return None }

        let Some(next) = it.next() else { return None };
        let mut consumed = 2usize;
        if next == '\\' {
            if !self.literal_options.char_allow_escape { return None }
            let Some(_) = it.next() else { return None };
            consumed += 1;
        }
        let Some(close) = it.next() else { return None };
        if close != '\'' { return None }
        consumed += 1;
        Some((consumed, "LITERAL_CHAR"))
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

        // Check literal matches
        if let Some((end, token_type)) = self.match_literal(text, start) {
            matches.push((end, token_type.to_string()));
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

    #[test]
    fn literal_number_recognition() {
        let generator = TokenGenerator::new();
        let tokens = vec![
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
        ];

        let tokenizer = generator.generate_tokenizer(tokens);

        // Test integer recognition
        let test_cases = [
            ("42", (0, "LITERAL_INTEGER")),
            ("-10", (0, "LITERAL_INTEGER")),
            ("0", (0, "LITERAL_INTEGER")),
            ("3.14", (0, "LITERAL_FLOAT")),
            ("-0.5", (0, "LITERAL_FLOAT")),
            ("1e10", (0, "LITERAL_FLOAT")),
            ("2.5e-3", (0, "LITERAL_FLOAT")),
        ];

        for (text, (start_pos, expected_token)) in test_cases {
            if let Some((end, token)) = tokenizer.match_with_precedence(text, start_pos) {
                assert_eq!(token, expected_token, "Failed for text: {}", text);
                assert_eq!(end, text.len(), "Should consume entire number: {}", text);
            } else {
                panic!("Expected literal match for text: {} at position {}", text, start_pos);
            }
        }
    }

    #[test]
    fn literal_string_recognition() {
        let generator = TokenGenerator::new();
        let tokens = vec![
            TokenDefinition::new("print", "KEYWORD_PRINT", TokenCategory::Keyword),
        ];

        let tokenizer = generator.generate_tokenizer(tokens);

        // Test string recognition
        let test_cases = [
            ("\"hello\"", (0, "LITERAL_STRING")),
            ("\"world\"", (0, "LITERAL_STRING")),
            ("\"hello world\"", (0, "LITERAL_STRING")),
            ("\"with\\\"quotes\"", (0, "LITERAL_STRING")),
        ];

        for (text, (start_pos, expected_token)) in test_cases {
            if let Some((end, token)) = tokenizer.match_with_precedence(text, start_pos) {
                assert_eq!(token, expected_token, "Failed for text: {}", text);
                assert_eq!(end, text.len(), "Should consume entire string: {}", text);
            } else {
                panic!("Expected string literal match for text: {} at position {}", text, start_pos);
            }
        }
    }

    #[test]
    fn tokenizer_with_literals() {
        let generator = TokenGenerator::new();
        let base_tokens = vec![
            TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword),
            TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator),
        ];

        let mut all_tokens = base_tokens.clone();
        all_tokens.push(TokenDefinition::new("42", "LITERAL_INTEGER", TokenCategory::Integer));
        let tokenizer = generator.generate_tokenizer(all_tokens);

        // Should have both base tokens and literal support
        let code = "let x = 42";
        let mut pos = 0;
        let mut tokens_found = Vec::new();

        while pos < code.len() {
            if let Some((end, token_type)) = tokenizer.match_with_precedence(code, pos) {
                tokens_found.push((pos, end, token_type));
                pos = end;
            } else {
                // Skip whitespace or single characters
                if let Some(ch) = code.chars().nth(pos) {
                    if ch.is_whitespace() {
                        pos += ch.len_utf8();
                    } else {
                        // Skip single characters that don't match
                        pos += ch.len_utf8();
                    }
                } else {
                    break;
                }
            }
        }

        // Should find: let, x (skipped), =, 42 (as literal)
        assert!(!tokens_found.is_empty());
        assert!(tokens_found.iter().any(|(_, _, t)| *t == "KEYWORD_LET"));
        assert!(tokens_found.iter().any(|(_, _, t)| *t == "OP_ASSIGN"));
        assert!(tokens_found.iter().any(|(_, _, t)| *t == "LITERAL_INTEGER"));
    }

    #[test]
    fn character_literal_recognition() {
        let generator = TokenGenerator::new().enable_character_literals(true);
        let tokens = vec![TokenDefinition::new("=", "OP_ASSIGN", TokenCategory::Operator)];
        let tokenizer = generator.generate_tokenizer(tokens);

        let cases = ["'a'", "'\\n'", "'\\''"];
        for s in cases {
            if let Some((end, token)) = tokenizer.match_with_precedence(s, 0) {
                assert_eq!(token, "LITERAL_CHAR");
                assert_eq!(end, s.len());
            } else {
                panic!("expected char literal match for: {}", s);
            }
        }
    }

    #[test]
    fn string_delimiter_customization() {
        let generator = TokenGenerator::new()
            .with_string_delimiter('\'')
            .enable_character_literals(false);
        let tokenizer = generator.generate_tokenizer(vec![]);

        let s = "'hello world'";
        if let Some((end, token)) = tokenizer.match_with_precedence(s, 0) {
            assert_eq!(token, "LITERAL_STRING");
            assert_eq!(end, s.len());
        } else {
            panic!("expected single-quoted string literal");
        }
    }

    #[test]
    fn disable_scientific_notation() {
        let generator = TokenGenerator::new().allow_scientific_notation(false);
        let tokenizer = generator.generate_tokenizer(vec![]);

        let s = "1e5";
        if let Some((end, token)) = tokenizer.match_with_precedence(s, 0) {
            assert_eq!(token, "LITERAL_INTEGER");
            assert_eq!(end, 1);
        } else {
            panic!("expected integer prefix match when scientific disabled");
        }
    }
}
