"""
Preprocessing Module
======================
Handles all text preprocessing of Java source code:
  1. Comment removal (single-line and block)
  2. Identifier normalization
  3. Literal normalization
  4. Whitespace normalization
  5. Method boundary extraction
  6. Token sequence preparation for CodeBERT input
"""

import re
import logging
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Java keyword sets
# ---------------------------------------------------------------------------

JAVA_KEYWORDS = frozenset({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int", "interface",
    "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "transient", "try",
    "void", "volatile", "while", "true", "false", "null",
    "var", "record", "sealed", "permits", "yield", "module", "open",
    "requires", "exports", "opens", "uses", "provides", "with", "to",
})

JAVA_PRIMITIVE_TYPES = frozenset({
    "int", "long", "short", "byte", "char", "float", "double",
    "boolean", "void",
})

JAVA_COMMON_TYPES = frozenset({
    "String", "Integer", "Long", "Double", "Float", "Boolean",
    "Object", "List", "ArrayList", "LinkedList", "Map", "HashMap",
    "Set", "HashSet", "Iterator", "Exception", "RuntimeException",
    "StringBuilder", "StringBuffer", "Array", "Arrays", "Math",
    "System", "Thread", "Runnable", "Callable", "Optional",
})

JAVA_OPERATORS = frozenset({
    "+", "-", "*", "/", "%", "++", "--",
    "==", "!=", "<", ">", "<=", ">=",
    "&&", "||", "!",
    "&", "|", "^", "~", "<<", ">>", ">>>",
    "=", "+=", "-=", "*=", "/=", "%=",
    "?", ":", ".", "->", "::",
})


# ---------------------------------------------------------------------------
# Comment Remover
# ---------------------------------------------------------------------------

class CommentRemover:
    """Removes Java comments while preserving line structure."""

    # Regex patterns
    _SINGLE_LINE = re.compile(r"//[^\n]*")
    _BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
    _JAVADOC = re.compile(r"/\*\*.*?\*/", re.DOTALL)
    _STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
    _CHAR_LITERAL = re.compile(r"'(?:[^'\\]|\\.)'")

    @classmethod
    def remove(cls, source_code: str) -> str:
        """
        Remove all comments from Java source code.
        Handles edge cases: comments inside strings, nested-like comments.
        """
        result = cls._remove_comments_safely(source_code)
        # Clean up extra blank lines
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    @classmethod
    def _remove_comments_safely(cls, code: str) -> str:
        """
        State-machine based comment removal that correctly handles
        string literals containing comment-like patterns.
        """
        result = []
        i = 0
        n = len(code)

        while i < n:
            # String literal
            if code[i] == '"':
                j = i + 1
                while j < n:
                    if code[j] == '\\':
                        j += 2
                        continue
                    if code[j] == '"':
                        j += 1
                        break
                    j += 1
                result.append(code[i:j])
                i = j

            # Char literal
            elif code[i] == "'":
                j = i + 1
                while j < n and code[j] != "'":
                    if code[j] == "\\":
                        j += 1
                    j += 1
                j += 1
                result.append(code[i:j])
                i = j

            # Block comment
            elif code[i:i+2] == "/*":
                end = code.find("*/", i + 2)
                if end == -1:
                    break
                # Preserve newlines for line number tracking
                newlines = code[i:end+2].count("\n")
                result.append("\n" * newlines)
                i = end + 2

            # Single-line comment
            elif code[i:i+2] == "//":
                end = code.find("\n", i)
                if end == -1:
                    break
                result.append("\n")
                i = end + 1

            else:
                result.append(code[i])
                i += 1

        return "".join(result)


# ---------------------------------------------------------------------------
# Identifier Normalizer
# ---------------------------------------------------------------------------

class IdentifierNormalizer:
    """
    Normalizes Java identifiers and literals to reduce vocabulary size
    while preserving structural information.
    """

    # Replacement tokens
    IDENTIFIER_TOKEN = "<ID>"
    STRING_TOKEN = "<STR>"
    NUMBER_TOKEN = "<NUM>"
    TYPE_TOKEN = "<TYPE>"
    ANNOTATION_TOKEN = "<ANN>"

    # Patterns
    _CAMEL_CASE = re.compile(r"(?<=[a-z])(?=[A-Z])")
    _SNAKE_CASE = re.compile(r"_([a-z])")
    _NUMBER = re.compile(r"\b\d+(\.\d+)?([eE][+-]?\d+)?[fFdDlL]?\b")
    _HEX_NUMBER = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
    _STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
    _CHAR = re.compile(r"'(?:[^'\\]|\\.)'")
    _ANNOTATION = re.compile(r"@\w+")

    def __init__(
        self,
        normalize_identifiers: bool = True,
        normalize_literals: bool = True,
        normalize_types: bool = False,
        keep_keywords: bool = True,
    ):
        self.normalize_identifiers = normalize_identifiers
        self.normalize_literals = normalize_literals
        self.normalize_types = normalize_types
        self.keep_keywords = keep_keywords

    def normalize_token(self, token: str) -> str:
        """Normalize a single token."""
        # Keep Java keywords as-is
        if token.lower() in JAVA_KEYWORDS:
            return token.lower() if self.keep_keywords else token

        # Keep operators
        if token in JAVA_OPERATORS:
            return token

        # Normalize string literals
        if self.normalize_literals:
            if self._STRING.fullmatch(token):
                return self.STRING_TOKEN
            if self._CHAR.fullmatch(token):
                return self.STRING_TOKEN

        # Normalize numbers
        if self.normalize_literals:
            if self._NUMBER.fullmatch(token) or self._HEX_NUMBER.fullmatch(token):
                return self.NUMBER_TOKEN

        # Normalize annotations
        if token.startswith("@"):
            return self.ANNOTATION_TOKEN

        # Normalize common type names
        if self.normalize_types and token in JAVA_COMMON_TYPES:
            return self.TYPE_TOKEN

        # Normalize identifiers (keep if it's a known type)
        if self.normalize_identifiers:
            if token[0].isupper() and token not in JAVA_COMMON_TYPES:
                return self.TYPE_TOKEN
            if token[0].islower() and token not in JAVA_KEYWORDS:
                return self.IDENTIFIER_TOKEN

        return token

    def normalize_sequence(self, tokens: List[str]) -> List[str]:
        """Normalize a sequence of tokens."""
        return [self.normalize_token(t) for t in tokens]

    def split_camel_case(self, identifier: str) -> List[str]:
        """Split camelCase into sub-tokens: 'getFieldName' -> ['get', 'field', 'name']"""
        # Split on camelCase boundaries
        parts = self._CAMEL_CASE.sub(" ", identifier).split()
        return [p.lower() for p in parts if p]

    def split_snake_case(self, identifier: str) -> List[str]:
        """Split snake_case into sub-tokens."""
        return [p.lower() for p in identifier.split("_") if p]


# ---------------------------------------------------------------------------
# Code Preprocessor (main pipeline)
# ---------------------------------------------------------------------------

class CodePreprocessor:
    """
    Full preprocessing pipeline for Java source code.

    Pipeline:
    1. Remove comments
    2. Extract method body (if needed)
    3. Normalize whitespace
    4. Tokenize
    5. Normalize identifiers/literals
    6. Truncate/pad to max length
    """

    # Java tokenizer pattern (handles strings, chars, numbers, identifiers, operators)
    _TOKENIZER = re.compile(
        r'"(?:[^"\\]|\\.)*"'      # String literal
        r"|'(?:[^'\\]|\\.)*'"     # Char literal
        r"|//[^\n]*"              # Single-line comment
        r"|/\*.*?\*/"            # Block comment
        r"|0[xX][0-9a-fA-F]+"   # Hex number
        r"|\d+\.?\d*(?:[eE][+-]?\d+)?[fFdDlL]?"  # Number
        r"|[a-zA-Z_$][a-zA-Z0-9_$]*"  # Identifier
        r"|>>>|>>=|<<=|>>|<<|>=|<=|==|!=|\+\+|--|&&|\|\||->|::|[+\-*/%&|^~<>=!?:;.,(){}[\]@]",
        re.DOTALL
    )

    def __init__(self, config: Dict):
        pp_config = config.get("preprocessing", {})

        self.remove_comments = pp_config.get("remove_comments", True)
        self.normalize_ids = pp_config.get("normalize_identifiers", True)
        self.normalize_lits = pp_config.get("normalize_literals", True)
        self.lowercase = pp_config.get("lowercase", False)
        self.min_tokens = pp_config.get("min_tokens", 10)
        self.max_tokens = pp_config.get("max_tokens", 512)

        self.comment_remover = CommentRemover()
        self.normalizer = IdentifierNormalizer(
            normalize_identifiers=self.normalize_ids,
            normalize_literals=self.normalize_lits,
        )

    def preprocess(self, source_code: str) -> str:
        """
        Full preprocessing pipeline.

        Returns:
            Preprocessed source code as a string.
        """
        # Step 1: Remove comments
        if self.remove_comments:
            source_code = self.comment_remover.remove(source_code)

        # Step 2: Normalize whitespace
        source_code = self._normalize_whitespace(source_code)

        return source_code

    def tokenize(self, source_code: str) -> List[str]:
        """Tokenize preprocessed source code."""
        tokens = self._TOKENIZER.findall(source_code)

        # Remove remaining comment tokens
        tokens = [t for t in tokens if not t.startswith("//") and not t.startswith("/*")]

        return tokens

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply identifier/literal normalization to token list."""
        return self.normalizer.normalize_sequence(tokens)

    def preprocess_for_model(self, source_code: str) -> str:
        """
        Full pipeline: preprocess → tokenize → normalize → rejoin.

        Returns a normalized token string suitable for model input.
        """
        # Remove comments
        code = self.preprocess(source_code)

        # Tokenize
        tokens = self.tokenize(code)

        # Normalize
        tokens = self.normalize_tokens(tokens)

        # Truncate
        tokens = tokens[:self.max_tokens]

        # Rejoin
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        return " ".join(tokens)

    def preprocess_batch(self, codes: List[str]) -> List[str]:
        """Process a batch of code strings."""
        return [self.preprocess_for_model(code) for code in codes]

    @staticmethod
    def _normalize_whitespace(code: str) -> str:
        """Normalize whitespace: collapse spaces, normalize newlines."""
        # Normalize line endings
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple blank lines
        code = re.sub(r"\n\s*\n\s*\n", "\n\n", code)
        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in code.split("\n")]
        return "\n".join(lines).strip()

    def is_valid(self, tokens: List[str]) -> bool:
        """Check if a token sequence meets minimum requirements."""
        return len(tokens) >= self.min_tokens

    def get_stats(self, source_code: str) -> Dict:
        """Compute preprocessing statistics for a code snippet."""
        original_tokens = self.tokenize(source_code)

        preprocessed = self.preprocess(source_code)
        processed_tokens = self.tokenize(preprocessed)
        normalized = self.normalize_tokens(processed_tokens)

        return {
            "original_chars": len(source_code),
            "original_token_count": len(original_tokens),
            "processed_token_count": len(processed_tokens),
            "normalized_token_count": len(normalized),
            "compression_ratio": len(normalized) / max(len(original_tokens), 1),
        }


# ---------------------------------------------------------------------------
# Batch Preprocessor with caching
# ---------------------------------------------------------------------------

class BatchPreprocessor:
    """
    Efficient batch preprocessor with disk caching for large datasets.
    """

    def __init__(self, config: Dict):
        self.preprocessor = CodePreprocessor(config)
        self.cache_dir = config.get("cache", {}).get("ast_cache_dir", "data/processed/ast_cache")
        self.use_cache = config.get("cache", {}).get("use_cache", True)

        import os
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_dataset(
        self,
        pairs: List[Dict],
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Process all code pairs in a dataset.

        Adds 'code1_processed' and 'code2_processed' keys to each pair.
        """
        from tqdm import tqdm

        processed = []
        for pair in tqdm(pairs, desc="Preprocessing", disable=not show_progress):
            pair = dict(pair)  # copy
            pair["code1_processed"] = self.preprocessor.preprocess_for_model(
                pair["code1"]
            )
            pair["code2_processed"] = self.preprocessor.preprocess_for_model(
                pair["code2"]
            )
            processed.append(pair)

        # Filter out pairs with too few tokens
        valid = [
            p for p in processed
            if (len(p["code1_processed"].split()) >= self.preprocessor.min_tokens
                and len(p["code2_processed"].split()) >= self.preprocessor.min_tokens)
        ]

        n_filtered = len(processed) - len(valid)
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} pairs with insufficient tokens")

        return valid
