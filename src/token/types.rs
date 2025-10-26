//! Token type definitions for the token generator system.

/// Categories of tokens for organization and precedence rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenCategory {
    /// Keywords like "let", "fn", "if", "else"
    Keyword,
    /// Operators like "+", "-", "==", "!=", etc.
    Operator,
    /// Symbols like "{", "}", "(", ")", etc.
    Symbol,
    /// Literals like "true", "false", "null"
    Literal,
    /// Custom category for user-defined tokens
    Custom(&'static str),
}

impl TokenCategory {
    pub fn precedence(&self) -> u8 {
        match self {
            TokenCategory::Keyword => 0,
            TokenCategory::Operator => 1,
            TokenCategory::Symbol => 2,
            TokenCategory::Literal => 3,
            TokenCategory::Custom(_) => 4,
        }
    }

    pub fn has_precedence_over(&self, other: &TokenCategory) -> bool {
        self.precedence() < other.precedence()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDefinition {
    pub pattern: String,
    pub token_type: String,
    pub category: TokenCategory,
    pub description: Option<String>,
}

impl TokenDefinition {
    pub fn new(
        pattern: &str,
        token_type: &str,
        category: TokenCategory,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            token_type: token_type.into(),
            category,
            description: None,
        }
    }

    pub fn with_description(
        pattern: &str,
        token_type: &str,
        category: TokenCategory,
        description: &str,
    ) -> Self {
        Self {
            pattern: pattern.into(),
            token_type: token_type.into(),
            category,
            description: Some(description.into()),
        }
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    pub fn token_type(&self) -> &str {
        &self.token_type
    }

    pub fn category(&self) -> TokenCategory {
        self.category
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Type alias for token type strings.
pub type TokenType = String;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_definition_creation() {
        let token = TokenDefinition::new("let", "KEYWORD_LET", TokenCategory::Keyword);

        assert_eq!(token.pattern(), "let");
        assert_eq!(token.token_type(), "KEYWORD_LET");
        assert_eq!(token.category(), TokenCategory::Keyword);
        assert!(token.description().is_none());
    }

    #[test]
    fn token_definition_with_description() {
        let token = TokenDefinition::with_description(
            "==",
            "OP_EQ",
            TokenCategory::Operator,
            "Equality operator",
        );

        assert_eq!(token.pattern(), "==");
        assert_eq!(token.token_type(), "OP_EQ");
        assert_eq!(token.category(), TokenCategory::Operator);
        assert_eq!(token.description(), Some("Equality operator"));
    }

    #[test]
    fn token_category_precedence() {
        assert!(TokenCategory::Keyword.has_precedence_over(&TokenCategory::Operator));
        assert!(TokenCategory::Operator.has_precedence_over(&TokenCategory::Symbol));
        assert!(TokenCategory::Symbol.has_precedence_over(&TokenCategory::Literal));

        // Same category should not have precedence over itself
        assert!(!TokenCategory::Keyword.has_precedence_over(&TokenCategory::Keyword));
    }

    #[test]
    fn custom_token_category() {
        let custom_category = TokenCategory::Custom("MY_CUSTOM");
        assert_eq!(custom_category.precedence(), 4);

        // Custom categories with the same precedence don't have precedence over each other
        assert!(!custom_category.has_precedence_over(&TokenCategory::Custom("OTHER")));

        // Custom categories have lower precedence than literals (higher number = lower precedence)
        assert!(!custom_category.has_precedence_over(&TokenCategory::Literal));

        // But literals have higher precedence than custom categories
        assert!(TokenCategory::Literal.has_precedence_over(&custom_category));
    }
}
