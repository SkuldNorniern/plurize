use std::collections::HashMap;

/// A simple trie for fast string matching.
/// - ASCII characters (0-127) use fast array lookup
/// - Unicode characters use a HashMap
/// - Stores values at terminal nodes
#[derive(Debug, Default)]
pub struct Trie<T> {
    nodes: Vec<TrieNode<T>>,
}

#[derive(Debug)]
struct TrieNode<T> {
    // ASCII fast path: fixed-size array of child indices (0-127)
    ascii_children: [Option<usize>; 128],
    // Unicode fallback: allocated only when needed
    unicode_children: Option<HashMap<char, usize>>,
    // Value at this terminal node, if any
    value: Option<T>,
}

impl<T> Default for TrieNode<T> {
    fn default() -> Self {
        TrieNode {
            ascii_children: [None; 128],
            unicode_children: None,
            value: None,
        }
    }
}

impl<T> Trie<T> {
    pub fn new() -> Self {
        let mut nodes = Vec::new();
        nodes.push(TrieNode::default()); 
        Trie { nodes }
    }

    pub fn insert(&mut self, key: &str, value: T) {
        let mut node_idx = 0;
        for ch in key.chars() {
            let next_idx = if (ch as u32) < 128 {
                match self.nodes[node_idx].ascii_children[ch as usize] {
                    Some(existing_idx) => existing_idx,
                    None => {
                        let new_idx = self.nodes.len();
                        self.nodes.push(TrieNode::default());
                        self.nodes[node_idx].ascii_children[ch as usize] = Some(new_idx);
                        new_idx
                    }
                }
            } else {
                match self.nodes[node_idx].unicode_children.as_ref().and_then(|m| m.get(&ch)) {
                    Some(existing_idx) => *existing_idx,
                    None => {
                        let new_idx = self.nodes.len();
                        self.nodes.push(TrieNode::default());
                        self.nodes[node_idx].unicode_children
                            .get_or_insert_with(HashMap::new)
                            .insert(ch, new_idx);
                        new_idx
                    }
                }
            };
            node_idx = next_idx;
        }
        self.nodes[node_idx].value = Some(value);
    }

    pub fn get(&self, key: &str) -> Option<&T> {
        let mut node_idx = 0;
        for ch in key.chars() {
            node_idx = if (ch as u32) < 128 {
                match self.nodes[node_idx].ascii_children[ch as usize] {
                    Some(child_idx) => child_idx,
                    None => return None,
                }
            } else {
                match self.nodes[node_idx].unicode_children.as_ref().and_then(|m| m.get(&ch)) {
                    Some(child_idx) => *child_idx,
                    None => return None,
                }
            };
        }
        self.nodes[node_idx].value.as_ref()
    }

    pub fn match_longest_from(&self, text: &str, start: usize) -> Option<(usize, &T)> {
        // Ensure start is a character boundary
        let tail = match text.get(start..) {
            Some(s) => s,
            None => return None,
        };

        let mut node_idx = 0; // start at root
        let mut best_match: Option<(usize, &T)> = self.nodes[0].value.as_ref().map(|v| (start, v));

        for (off, ch) in tail.char_indices() {
            let next_idx = if (ch as u32) < 128 {
                match self.nodes[node_idx].ascii_children[ch as usize] {
                    Some(child_idx) => child_idx,
                    None => break,
                }
            } else {
                match self.nodes[node_idx].unicode_children.as_ref().and_then(|m| m.get(&ch)) {
                    Some(child_idx) => *child_idx,
                    None => break,
                }
            };

            node_idx = next_idx;
            if let Some(v) = &self.nodes[node_idx].value {
                let ch_len = ch.len_utf8();
                let end = start + off + ch_len;
                best_match = Some((end, v));
            }
        }

        best_match
    }

    pub fn match_longest(&self, text: &str) -> Option<(usize, &T)> {
        self.match_longest_from(text, 0)
    }

    pub fn walk_prefix_len(&self, text: &str, start: usize) -> usize {
        let Some(tail) = text.get(start..) else { return 0 };
        let mut node_idx = 0; 
        let mut consumed = 0usize;
        for (off, ch) in tail.char_indices() {
            let next_idx = if (ch as u32) < 128 {
                match self.nodes[node_idx].ascii_children[ch as usize] {
                    Some(child_idx) => child_idx,
                    None => break,
                }
            } else {
                match self.nodes[node_idx].unicode_children.as_ref().and_then(|m| m.get(&ch)) {
                    Some(child_idx) => *child_idx,
                    None => break,
                }
            };
            node_idx = next_idx;
            consumed = off + ch.len_utf8();
        }
        consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_basic() {
        let mut trie = Trie::new();
        trie.insert("let", 1u8);
        trie.insert("println", 2u8);
        trie.insert("->", 3u8);

        assert_eq!(trie.get("let"), Some(&1));
        assert_eq!(trie.get("println"), Some(&2));
        assert_eq!(trie.get("->"), Some(&3));
        assert_eq!(trie.get("print"), None);
    }

    #[test]
    fn longest_match_prefers_longer() {
        let mut trie = Trie::new();
        trie.insert("=", "assign");
        trie.insert("==", "eq");
        trie.insert("!=", "neq");

        let s = "a == b";
        // Find position of '='; the test string is known to contain '='
        let eq_pos = s.find('=').unwrap_or(0);
        let m = trie.match_longest_from(s, eq_pos);
        let (end, val) = match m {
            Some(pair) => pair,
            None => panic!("expected longest match for '==' to be present"),
        };
        assert_eq!(val, &"eq");
        assert_eq!(&s[eq_pos..end], "==");
    }

    #[test]
    fn supports_unicode_tokens() {
        let mut trie = Trie::new();
        trie.insert("π", 3.14f32);
        trie.insert("≠", -1.0f32);

        let text = "π ≠ x";
        let pi_pos = text.find('π').unwrap_or(0);
        let not_eq_pos = text.find('≠').unwrap_or(0);

        let m1 = trie.match_longest_from(text, pi_pos).map(|(e, v)| (&text[pi_pos..e], *v));
        let m2 = trie.match_longest_from(text, not_eq_pos).map(|(e, v)| (&text[not_eq_pos..e], *v));

        assert_eq!(m1, Some(("π", 3.14f32)));
        assert_eq!(m2, Some(("≠", -1.0f32)));
    }

    #[test]
    fn walk_prefix_len_handles_partials() {
        let mut trie = Trie::new();
        trie.insert("println", 1u8);
        let consumed = trie.walk_prefix_len("printx", 0);
        // "print" prefix is shared, then diverges at 'x'
        // Expected consumed bytes should be length of common prefix actually present in trie path
        // Here, common path covers "print" (5 bytes), then 'l' not found causes a stop.
        // But since we never created sub-paths for 'p','r','i','n','t' alone without values,
        // the structural path still exists due to insertion of "println".
        assert!(consumed >= 5);
        assert!(consumed <= core::cmp::min("printx".len(), "println".len()));
    }
}


