"""
AST Processing Module
======================
Generates Abstract Syntax Trees for Java source code using Tree-Sitter
or javalang, then linearizes them into token sequences for embedding.

Supports two backends:
  - tree-sitter: Fast, robust, handles partial/incomplete code
  - javalang:    Pure Python, full Java grammar, rich node types
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST Node representation
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """Unified AST node representation."""
    node_type: str
    value: Optional[str] = None
    children: List["ASTNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)


# ---------------------------------------------------------------------------
# Abstract base parser
# ---------------------------------------------------------------------------

class BaseASTParser(ABC):
    """Abstract base class for AST parsers."""

    @abstractmethod
    def parse(self, source_code: str) -> Optional[ASTNode]:
        """Parse source code into an ASTNode tree."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this parser backend is available."""
        ...


# ---------------------------------------------------------------------------
# Tree-Sitter Parser
# ---------------------------------------------------------------------------

class TreeSitterParser(BaseASTParser):
    """
    Parser backend using Tree-Sitter.

    Tree-Sitter produces concrete syntax trees; we filter to
    AST-relevant nodes by skipping punctuation/whitespace.
    """

    # Node types to include (AST-level nodes, not tokens)
    STRUCTURAL_NODE_TYPES = {
        "method_declaration", "constructor_declaration", "class_declaration",
        "interface_declaration", "block", "expression_statement",
        "local_variable_declaration", "return_statement", "if_statement",
        "while_statement", "for_statement", "enhanced_for_statement",
        "try_statement", "catch_clause", "finally_clause",
        "method_invocation", "object_creation_expression",
        "assignment_expression", "binary_expression", "unary_expression",
        "conditional_expression", "array_access", "field_access",
        "identifier", "decimal_integer_literal", "hex_integer_literal",
        "floating_point_literal", "character_literal", "string_literal",
        "boolean_type", "void_type", "integral_type", "floating_point_type",
        "type_identifier", "array_type", "generic_type",
        "formal_parameter", "argument_list", "variable_declarator",
        "cast_expression", "instanceof_expression", "lambda_expression",
    }

    # Token leaf node types
    LEAF_NODE_TYPES = {
        "identifier", "decimal_integer_literal", "hex_integer_literal",
        "floating_point_literal", "character_literal", "string_literal",
        "true", "false", "null_literal",
        "this", "super",
        "boolean_type", "void_type", "integral_type", "floating_point_type",
        "type_identifier",
    }

    def __init__(self, grammar_path: Optional[str] = None):
        self.grammar_path = grammar_path
        self._parser = None
        self._language = None
        self._init_parser()

    def _init_parser(self):
        """Initialize Tree-Sitter parser with Java grammar."""
        try:
            import tree_sitter
            from tree_sitter import Language, Parser

            if self.grammar_path:
                # Build from local grammar
                Language.build_library(
                    "build/tree-sitter-java.so",
                    [self.grammar_path]
                )
                self._language = Language("build/tree-sitter-java.so", "java")
            else:
                # Try tree-sitter-languages package
                try:
                    from tree_sitter_languages import get_language, get_parser
                    self._language = get_language("java")
                    self._parser = get_parser("java")
                    logger.info("Tree-Sitter Java parser initialized via tree_sitter_languages")
                    return
                except ImportError:
                    logger.warning("tree_sitter_languages not available")
                    return

            self._parser = Parser()
            self._parser.set_language(self._language)
            logger.info("Tree-Sitter Java parser initialized")

        except ImportError:
            logger.warning("tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-languages")
        except Exception as e:
            logger.warning(f"Failed to initialize Tree-Sitter: {e}")

    def is_available(self) -> bool:
        return self._parser is not None

    def parse(self, source_code: str) -> Optional[ASTNode]:
        """Parse Java source code using Tree-Sitter."""
        if not self.is_available():
            return None

        try:
            tree = self._parser.parse(bytes(source_code, "utf-8"))
            return self._convert_node(tree.root_node, source_code)
        except Exception as e:
            logger.debug(f"Tree-Sitter parse error: {e}")
            return None

    def _convert_node(self, ts_node, source_code: str) -> ASTNode:
        """Recursively convert Tree-Sitter node to ASTNode."""
        node_type = ts_node.type

        # Extract value for leaf nodes
        value = None
        if ts_node.child_count == 0 or node_type in self.LEAF_NODE_TYPES:
            start, end = ts_node.start_byte, ts_node.end_byte
            value = source_code[start:end]

        ast_node = ASTNode(
            node_type=node_type,
            value=value,
            children=[]
        )

        # Recurse into children (skip punctuation/whitespace)
        for child in ts_node.children:
            if child.type not in {";", ",", "(", ")", "{", "}", "[", "]", ".", ":", "//", "/*", "*/"}:
                ast_node.children.append(
                    self._convert_node(child, source_code)
                )

        return ast_node


# ---------------------------------------------------------------------------
# Javalang Parser
# ---------------------------------------------------------------------------

class JavalangParser(BaseASTParser):
    """
    Parser backend using javalang (pure Python Java parser).

    Provides richer semantic node types compared to Tree-Sitter,
    but may fail on some edge cases.
    """

    # Mapping from javalang node class names to standardized types
    NODE_TYPE_MAP = {
        "MethodDeclaration": "METHOD",
        "ConstructorDeclaration": "CONSTRUCTOR",
        "ClassDeclaration": "CLASS",
        "InterfaceDeclaration": "INTERFACE",
        "BlockStatement": "BLOCK",
        "StatementExpression": "EXPR_STMT",
        "LocalVariableDeclaration": "VAR_DECL",
        "ReturnStatement": "RETURN",
        "IfStatement": "IF",
        "WhileStatement": "WHILE",
        "ForStatement": "FOR",
        "EnhancedForStatement": "FOR_EACH",
        "TryStatement": "TRY",
        "CatchClause": "CATCH",
        "MethodInvocation": "METHOD_CALL",
        "ClassCreator": "NEW",
        "Assignment": "ASSIGN",
        "BinaryOperation": "BINARY_OP",
        "MemberReference": "MEMBER_REF",
        "Literal": "LITERAL",
        "ReferenceType": "REF_TYPE",
        "BasicType": "BASIC_TYPE",
        "FormalParameter": "PARAM",
        "VariableDeclarator": "VAR_DECL",
        "Cast": "CAST",
        "ArraySelector": "ARRAY_ACCESS",
        "TernaryExpression": "TERNARY",
        "LambdaExpression": "LAMBDA",
    }

    def __init__(self):
        self._javalang = None
        self._init()

    def _init(self):
        try:
            import javalang
            self._javalang = javalang
            logger.info("javalang parser initialized")
        except ImportError:
            logger.warning("javalang not installed. Install with: pip install javalang")

    def is_available(self) -> bool:
        return self._javalang is not None

    def parse(self, source_code: str) -> Optional[ASTNode]:
        """
        Parse Java source code using javalang.

        Wraps methods in a class if needed for valid parsing.
        """
        if not self.is_available():
            return None

        # Try parsing as-is first, then wrapped in class
        for code in [source_code, self._wrap_in_class(source_code)]:
            try:
                tree = self._javalang.parse.parse(code)
                return self._convert_compilation_unit(tree)
            except Exception:
                continue

        logger.debug("javalang failed to parse code")
        return None

    @staticmethod
    def _wrap_in_class(code: str) -> str:
        return f"public class Wrapper {{\n{code}\n}}"

    def _convert_compilation_unit(self, tree) -> ASTNode:
        """Convert javalang CompilationUnit to ASTNode."""
        root = ASTNode(node_type="COMPILATION_UNIT")
        for _, node in tree:
            if node is not None:
                child = self._convert_node(node)
                if child:
                    root.children.append(child)
        return root

    def _convert_node(self, jl_node) -> Optional[ASTNode]:
        """Recursively convert javalang AST node."""
        if jl_node is None:
            return None

        node_class = type(jl_node).__name__
        node_type = self.NODE_TYPE_MAP.get(node_class, node_class.upper())

        # Extract value for terminal nodes
        value = None
        if hasattr(jl_node, "value") and isinstance(jl_node.value, str):
            value = jl_node.value
        elif hasattr(jl_node, "member") and isinstance(jl_node.member, str):
            value = jl_node.member
        elif hasattr(jl_node, "name") and isinstance(jl_node.name, str):
            value = jl_node.name

        ast_node = ASTNode(node_type=node_type, value=value)

        # Recurse into child nodes
        for attr_name in dir(jl_node):
            if attr_name.startswith("_"):
                continue
            try:
                attr_val = getattr(jl_node, attr_name)
            except Exception:
                continue

            if hasattr(attr_val, "attrs"):  # javalang AST node
                child = self._convert_node(attr_val)
                if child:
                    ast_node.children.append(child)
            elif isinstance(attr_val, (list, set, frozenset)):
                for item in attr_val:
                    if hasattr(item, "attrs"):
                        child = self._convert_node(item)
                        if child:
                            ast_node.children.append(child)

        return ast_node


# ---------------------------------------------------------------------------
# AST Linearizer
# ---------------------------------------------------------------------------

class ASTLinearizer:
    """
    Linearizes an AST into a flat sequence of tokens for model input.

    Supports multiple traversal strategies and token formatting.
    """

    def __init__(
        self,
        traversal: str = "preorder",
        include_node_types: bool = True,
        include_values: bool = True,
        max_tokens: int = 512,
    ):
        assert traversal in {"preorder", "postorder", "bfs"}
        self.traversal = traversal
        self.include_node_types = include_node_types
        self.include_values = include_values
        self.max_tokens = max_tokens

    def linearize(self, root: ASTNode) -> List[str]:
        """Convert AST tree to token sequence."""
        if self.traversal == "preorder":
            tokens = self._preorder(root)
        elif self.traversal == "postorder":
            tokens = self._postorder(root)
        else:  # bfs
            tokens = self._bfs(root)

        return tokens[:self.max_tokens]

    def _preorder(self, node: ASTNode) -> List[str]:
        tokens = []
        stack = [node]
        while stack:
            current = stack.pop()
            tokens.extend(self._node_tokens(current))
            # Push children in reverse to process left-to-right
            for child in reversed(current.children):
                stack.append(child)
        return tokens

    def _postorder(self, node: ASTNode) -> List[str]:
        tokens = []

        def _traverse(n: ASTNode):
            for child in n.children:
                _traverse(child)
            tokens.extend(self._node_tokens(n))

        _traverse(node)
        return tokens

    def _bfs(self, root: ASTNode) -> List[str]:
        from collections import deque
        tokens = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            tokens.extend(self._node_tokens(node))
            queue.extend(node.children)
        return tokens

    def _node_tokens(self, node: ASTNode) -> List[str]:
        """Generate token(s) for a single AST node."""
        tokens = []

        if self.include_node_types:
            tokens.append(f"[{node.node_type}]")

        if self.include_values and node.value is not None:
            # Clean the value
            val = node.value.strip()
            if val:
                tokens.append(val)

        return tokens

    def linearize_to_string(self, root: ASTNode) -> str:
        """Convert AST to space-separated token string."""
        return " ".join(self.linearize(root))


# ---------------------------------------------------------------------------
# AST Processor (orchestrator)
# ---------------------------------------------------------------------------

class ASTProcessor:
    """
    High-level orchestrator for AST generation and linearization.

    Tries Tree-Sitter first, falls back to javalang.
    """

    def __init__(self, config: Dict):
        ast_config = config.get("ast", {})
        parser_type = ast_config.get("parser", "tree-sitter")
        grammar_path = ast_config.get("grammar_path")

        self.traversal = ast_config.get("traversal", "preorder")
        self.include_node_types = ast_config.get("include_node_types", True)
        self.include_values = ast_config.get("include_values", True)
        self.max_tokens = ast_config.get("max_token_length", 512)

        # Initialize parsers with fallback
        self.primary_parser: Optional[BaseASTParser] = None
        self.fallback_parser: Optional[BaseASTParser] = None

        ts_parser = TreeSitterParser(grammar_path)
        jl_parser = JavalangParser()

        if parser_type == "tree-sitter":
            self.primary_parser = ts_parser if ts_parser.is_available() else jl_parser
            self.fallback_parser = jl_parser if ts_parser.is_available() else None
        else:
            self.primary_parser = jl_parser if jl_parser.is_available() else ts_parser
            self.fallback_parser = ts_parser if jl_parser.is_available() else None

        self.linearizer = ASTLinearizer(
            traversal=self.traversal,
            include_node_types=self.include_node_types,
            include_values=self.include_values,
            max_tokens=self.max_tokens,
        )

        logger.info(
            f"AST Processor initialized: primary={type(self.primary_parser).__name__}, "
            f"fallback={type(self.fallback_parser).__name__ if self.fallback_parser else 'None'}"
        )

    def process(self, source_code: str) -> Tuple[Optional[ASTNode], List[str]]:
        """
        Parse source code and return (AST, token_sequence).

        Returns:
            (ast_root, token_list) - ast_root may be None if parsing failed
        """
        # Try primary parser
        ast_root = None
        if self.primary_parser:
            ast_root = self.primary_parser.parse(source_code)

        # Fall back to secondary parser
        if ast_root is None and self.fallback_parser:
            ast_root = self.fallback_parser.parse(source_code)

        # If both fail, tokenize raw source
        if ast_root is None:
            logger.debug("Both parsers failed; falling back to simple tokenization")
            tokens = self._simple_tokenize(source_code)
            return None, tokens

        tokens = self.linearizer.linearize(ast_root)
        return ast_root, tokens

    def get_token_string(self, source_code: str) -> str:
        """Convenience: get linearized AST as a string."""
        _, tokens = self.process(source_code)
        return " ".join(tokens)

    @staticmethod
    def _simple_tokenize(source_code: str) -> List[str]:
        """Simple regex-based tokenization as last resort."""
        # Remove comments
        code = re.sub(r"//[^\n]*", "", source_code)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Split on whitespace and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", code)
        return [t for t in tokens if t.strip()]

    def get_statistics(self, source_code: str) -> Dict:
        """Return AST statistics for a method."""
        ast_root, tokens = self.process(source_code)
        stats = {
            "token_count": len(tokens),
            "parse_success": ast_root is not None,
        }
        if ast_root:
            stats["ast_depth"] = ast_root.depth()
            stats["ast_node_count"] = ast_root.node_count()
        return stats
