"""
PDE Tokenizer - Lample & Charton Style
Modular tokenization system for PDEs supporting:
1. Infix → Prefix conversion (for transformers)
2. Vocabulary building
3. Token → ID mapping
4. One-hot encoding (for Grammar VAE)
5. Batch processing
"""

import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict

from .normalize import normalize_pde_string


# Special tokens (following Lample & Charton convention)
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class PDEVocabulary:
    """
    Vocabulary builder for PDE expressions
    Handles all possible tokens that can appear in PDEs
    """

    def __init__(self):
        # Special tokens
        self.special_tokens = SPECIAL_TOKENS

        # PDE operators (derivatives)
        self.derivatives = [
            # Temporal
            "dt",
            "dtt",
            # Spatial 1st order
            "dx",
            "dy",
            "dz",
            # Spatial 2nd order
            "dxx",
            "dxy",
            "dxz",
            "dyy",
            "dyz",
            "dzz",
            # Spatial 3rd order
            "dxxx",
            "dxxy",
            "dxxz",
            "dxyy",
            "dxyz",
            "dxzz",
            "dyyy",
            "dyyz",
            "dyzz",
            "dzzz",
            # Spatial 4th order
            "dxxxx",
            "dxxxy",
            "dxxxz",
            "dxxyy",
            "dxxyz",
            "dxxzz",
            "dxyyy",
            "dxyyz",
            "dxyzz",
            "dxzzz",
            "dyyyy",
            "dyyyz",
            "dyyzz",
            "dyzzz",
            "dzzzz",
        ]

        # Variables
        self.variables = ["u", "x", "y", "z", "t"]

        # Operators (prefix notation)
        self.operators = [
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "neg",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "abs",
        ]

        # Parentheses and special symbols
        self.symbols = ["(", ")", "=", "0", "1", "2", "3", "^"]

        # Numeric tokens (for coefficients)
        # Use INT+ / INT- like Lample & Charton for sign
        # Then digits 0-9 and decimal point
        self.numeric = [
            "INT+",
            "INT-",
            "FLOAT",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
            "-",
            "e",
        ]

        # Build complete vocabulary
        self.build_vocabulary()

    def build_vocabulary(self):
        """Build the complete vocabulary with all tokens"""
        # Order matters: special tokens first for standard indices
        self.words = (
            self.special_tokens
            + self.derivatives
            + self.variables
            + self.operators
            + self.symbols
            + self.numeric
        )

        # Create bidirectional mappings
        self.word2id = {word: i for i, word in enumerate(self.words)}
        self.id2word = {i: word for i, word in enumerate(self.words)}

        # Store sizes
        self.vocab_size = len(self.words)
        self.pad_id = self.word2id["<PAD>"]
        self.sos_id = self.word2id["<SOS>"]
        self.eos_id = self.word2id["<EOS>"]
        self.unk_id = self.word2id["<UNK>"]

    def __len__(self):
        return self.vocab_size

    def get_token_id(self, token: str) -> int:
        """Get ID for a token, return UNK if not found"""
        return self.word2id.get(token, self.unk_id)

    def get_id_token(self, idx: int) -> str:
        """Get token for an ID"""
        return self.id2word.get(idx, "<UNK>")

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of IDs"""
        return [self.get_token_id(t) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of IDs to list of tokens"""
        return [self.get_id_token(i) for i in ids]

    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, "w") as f:
            for word in self.words:
                f.write(f"{word}\n")

    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file"""
        vocab = cls()
        with open(path, "r") as f:
            words = [line.strip() for line in f]
        vocab.words = words
        vocab.word2id = {word: i for i, word in enumerate(words)}
        vocab.id2word = {i: word for i, word in enumerate(words)}
        vocab.vocab_size = len(words)
        vocab.pad_id = vocab.word2id.get("<PAD>", 0)
        vocab.sos_id = vocab.word2id.get("<SOS>", 1)
        vocab.eos_id = vocab.word2id.get("<EOS>", 2)
        vocab.unk_id = vocab.word2id.get("<UNK>", 3)
        return vocab


class PDETokenizer:
    """
    PDE Tokenizer following Lample & Charton approach
    Converts between:
    - Infix notation (human-readable): "dt(u) - 2.345*dxx(u) = 0"
    - Prefix notation (for transformers): ["sub", "dt", "u", "mul", "2.345", "dxx", "u"]
    - Token IDs (for models)
    - One-hot encoding (for Grammar VAE)
    """

    def __init__(self, vocab: Optional[PDEVocabulary] = None):
        self.vocab = vocab if vocab is not None else PDEVocabulary()

    @staticmethod
    def _strip_eq0(pde_string: str) -> str:
        """Strip a trailing '= 0' (any whitespace) if present.

        We treat datasets with/without '= 0' as equivalent.
        """
        # Keep the name for backward compatibility, but delegate to the shared normalizer
        # so grammar/token pipelines cannot drift again.
        return normalize_pde_string(pde_string)

    def tokenize_infix(self, pde_string: str) -> List[str]:
        """
        Tokenize infix PDE string into tokens

        Input: "dt(u) - 2.345*dxx(u) = 0"
        Output: ["dt", "(", "u", ")", "-", "INT+", "2", ".", "3", "4", "5", "*", "dxx", "(", "u", ")", "=", "0"]

        Following Lample & Charton: numbers are tokenized as:
        - INT+ or INT- for sign
        - Then individual digits
        - Then . for decimal point
        - Then more digits
        """
        # Remove trailing '= 0' if present (robust to spacing)
        pde_string = self._strip_eq0(pde_string)

        # First, extract all number tokens
        # Pattern: optional sign, digits, optional decimal, optional more digits
        number_pattern = r"(-?[0-9]+\.[0-9]+|-?[0-9]+)"

        tokens = []
        pos = 0

        while pos < len(pde_string):
            # Skip whitespace
            if pde_string[pos].isspace():
                pos += 1
                continue

            # Try to match a number
            match = re.match(number_pattern, pde_string[pos:])
            if match:
                num_str = match.group(0)
                # Tokenize the number
                num_tokens = self._tokenize_number(num_str)
                tokens.extend(num_tokens)
                pos += len(num_str)
                continue

            # Try to match an identifier (derivative or variable)
            if pde_string[pos].isalpha():
                end = pos
                while end < len(pde_string) and pde_string[end].isalnum():
                    end += 1
                tokens.append(pde_string[pos:end])
                pos = end
                continue

            # Single character token (operators, parentheses)
            tokens.append(pde_string[pos])
            pos += 1

        return tokens

    def _tokenize_number(self, num_str: str) -> List[str]:
        """
        Tokenize a number following Lample & Charton convention

        "2.345" -> ["INT+", "2", ".", "3", "4", "5"]
        "-1.5" -> ["INT-", "1", ".", "5"]
        "3" -> ["INT+", "3"]
        """
        tokens = []

        # Handle sign
        if num_str.startswith("-"):
            tokens.append("INT-")
            num_str = num_str[1:]
        else:
            tokens.append("INT+")

        # Tokenize each character (digit or decimal point)
        for char in num_str:
            tokens.append(char)

        return tokens

    def infix_to_prefix(self, infix_tokens: List[str]) -> List[str]:
        """
        Convert infix tokens to prefix notation

        Input: ["dt", "(", "u", ")", "-", "2.345", "*", "dxx", "(", "u", ")"]
        Output: ["sub", "dt", "u", "mul", "2.345", "dxx", "u"]

        This uses Shunting Yard algorithm + reverse Polish notation
        """
        # Define operator precedence
        precedence = {
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
            "^": 3,
            "neg": 4,  # Unary negation
        }

        # Convert operators to prefix names
        infix_to_prefix_op = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "^": "pow",
        }

        output = []
        operator_stack = []

        i = 0
        while i < len(infix_tokens):
            token = infix_tokens[i]

            # Number (coefficient)
            if re.match(r"^[0-9]+(\.[0-9]+)?$", token):
                output.append(token)

            # Variable or derivative function
            elif token in self.vocab.variables or token in self.vocab.derivatives:
                # Check if followed by parentheses (function call)
                if i + 1 < len(infix_tokens) and infix_tokens[i + 1] == "(":
                    # Find matching closing parenthesis
                    paren_count = 1
                    j = i + 2
                    while j < len(infix_tokens) and paren_count > 0:
                        if infix_tokens[j] == "(":
                            paren_count += 1
                        elif infix_tokens[j] == ")":
                            paren_count -= 1
                        j += 1

                    # Recursively convert argument
                    arg_tokens = infix_tokens[i + 2 : j - 1]
                    arg_prefix = self.infix_to_prefix(arg_tokens)

                    # Add function and its argument
                    output.append(token)
                    output.extend(arg_prefix)

                    i = j - 1  # Skip processed tokens
                else:
                    output.append(token)

            # Left parenthesis
            elif token == "(":
                operator_stack.append(token)

            # Right parenthesis
            elif token == ")":
                while operator_stack and operator_stack[-1] != "(":
                    op = operator_stack.pop()
                    if op in infix_to_prefix_op:
                        output.insert(0, infix_to_prefix_op[op])
                if operator_stack:
                    operator_stack.pop()  # Remove '('

            # Operator
            elif token in ["+", "-", "*", "/", "^"]:
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and operator_stack[-1] in precedence
                    and precedence.get(operator_stack[-1], 0)
                    >= precedence.get(token, 0)
                ):
                    op = operator_stack.pop()
                    if op in infix_to_prefix_op:
                        output.insert(0, infix_to_prefix_op[op])
                operator_stack.append(token)

            i += 1

        # Pop remaining operators
        while operator_stack:
            op = operator_stack.pop()
            if op in infix_to_prefix_op:
                output.insert(0, infix_to_prefix_op[op])

        return output

    def infix_to_prefix_simple(self, pde_string: str) -> List[str]:
        """
        Simplified infix to prefix conversion
        For canonical PDEs: "dt(u) - 2.345*dxx(u) + u^2 = 0"

        Strategy: Parse as AST, then output in prefix
        """
        # Remove trailing '= 0' if present (robust to spacing)
        pde_string = self._strip_eq0(pde_string)

        # Simple recursive descent parser
        tokens = self.tokenize_infix(pde_string)
        prefix, _ = self._parse_expression(tokens, 0)
        return prefix

    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[List[str], int]:
        """
        Recursive descent parser for PDE expressions
        Returns (prefix_tokens, new_position)
        """
        # Parse first term
        prefix, pos = self._parse_term(tokens, pos)

        # Parse additional terms (with + or -)
        while pos < len(tokens) and tokens[pos] in ["+", "-"]:
            op = "add" if tokens[pos] == "+" else "sub"
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            prefix = [op] + prefix + right

        return prefix, pos

    def _parse_term(self, tokens: List[str], pos: int) -> Tuple[List[str], int]:
        """Parse a term (products/divisions)"""
        # Parse first factor
        prefix, pos = self._parse_factor(tokens, pos)

        # Parse additional factors (with * or /)
        while pos < len(tokens) and tokens[pos] in ["*", "/"]:
            op = "mul" if tokens[pos] == "*" else "div"
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            prefix = [op] + prefix + right

        return prefix, pos

    def _parse_factor(self, tokens: List[str], pos: int) -> Tuple[List[str], int]:
        """Parse a factor (numbers, variables, derivatives, powers, parentheses)"""
        if pos >= len(tokens):
            return [], pos

        token = tokens[pos]

        # Number (tokenized as INT+/INT- followed by digits)
        # Reconstruct as single token for prefix notation
        if token in ["INT+", "INT-"]:
            # Collect all number tokens
            num_parts = [token]
            pos += 1
            while pos < len(tokens) and (tokens[pos].isdigit() or tokens[pos] == "."):
                num_parts.append(tokens[pos])
                pos += 1
            # Return as single flattened list (keep individual tokens)
            return num_parts, pos

        # Single digit (shouldn't happen alone, but handle it)
        if token.isdigit() or token == ".":
            return [token], pos + 1

        # Derivative or variable with function call: dt(u), dxx(u), etc.
        if token in self.vocab.derivatives or token in self.vocab.variables:
            if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
                # Find matching )
                depth = 1
                end = pos + 2
                while end < len(tokens) and depth > 0:
                    if tokens[end] == "(":
                        depth += 1
                    elif tokens[end] == ")":
                        depth -= 1
                    end += 1

                # Parse argument
                arg_prefix, _ = self._parse_expression(tokens, pos + 2)
                result = [token] + arg_prefix

                # Check for power: dxx(u)^2
                if end < len(tokens) and tokens[end] == "^":
                    exp_prefix, end = self._parse_factor(tokens, end + 1)
                    result = ["pow"] + result + exp_prefix

                return result, end
            else:
                # Just a variable
                result = [token]

                # Check for power: u^2
                if pos + 1 < len(tokens) and tokens[pos + 1] == "^":
                    exp_prefix, pos = self._parse_factor(tokens, pos + 2)
                    result = ["pow"] + result + exp_prefix
                    return result, pos

                return result, pos + 1

        # Composite terms: u*dx(u), (dx(u))^2
        if token == "(" and pos + 1 < len(tokens):
            # Find matching )
            depth = 1
            end = pos + 1
            while end < len(tokens) and depth > 0:
                if tokens[end] == "(":
                    depth += 1
                elif tokens[end] == ")":
                    depth -= 1
                end += 1

            # Parse inside parentheses
            inner_prefix, _ = self._parse_expression(tokens, pos + 1)

            # Check for power: (...)^2
            if end < len(tokens) and tokens[end] == "^":
                exp_prefix, end = self._parse_factor(tokens, end + 1)
                result = ["pow"] + inner_prefix + exp_prefix
                return result, end

            return inner_prefix, end

        # Unknown token
        return [token], pos + 1

    def encode(self, pde_string: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode PDE string to token IDs

        Args:
            pde_string: Infix PDE string
            add_special_tokens: Add <SOS> and <EOS>

        Returns:
            List of token IDs
        """
        # Convert to prefix
        prefix_tokens = self.infix_to_prefix_simple(pde_string)

        # Convert to IDs
        ids = self.vocab.tokens_to_ids(prefix_tokens)

        # Add special tokens
        if add_special_tokens:
            ids = [self.vocab.sos_id] + ids + [self.vocab.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to prefix notation string

        Args:
            ids: List of token IDs
            skip_special_tokens: Skip <PAD>, <SOS>, <EOS>

        Returns:
            Prefix notation string
        """
        tokens = self.vocab.ids_to_tokens(ids)

        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]]

        return " ".join(tokens)

    def prefix_to_infix(self, prefix_tokens: List[str]) -> str:
        """
        Convert prefix notation back to human-readable infix notation

        Args:
            prefix_tokens: List of tokens in prefix notation

        Returns:
            Infix notation string (human-readable)

        Example:
            Input:  ['sub', 'dt', 'u', 'mul', 'INT+', '2', '.', '3', '4', '5', 'dxx', 'u']
            Output: 'dt(u) - 2.345*dxx(u)'
        """

        def reconstruct_number(tokens: List[str], pos: int) -> Tuple[str, int]:
            """Reconstruct a number from tokenized parts"""
            if pos >= len(tokens):
                return "", pos

            if tokens[pos] not in ["INT+", "INT-"]:
                return "", pos

            sign = "-" if tokens[pos] == "INT-" else ""
            pos += 1

            num_str = ""
            while pos < len(tokens) and (tokens[pos].isdigit() or tokens[pos] == "."):
                num_str += tokens[pos]
                pos += 1

            return sign + num_str, pos

        def parse_prefix(tokens: List[str], pos: int) -> Tuple[str, int]:
            """Recursively parse prefix notation and build infix expression"""
            if pos >= len(tokens):
                return "", pos

            token = tokens[pos]

            # Binary operators
            if token in ["add", "sub", "mul", "div", "pow"]:
                op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}
                op = op_map[token]

                # Parse left operand
                left, pos = parse_prefix(tokens, pos + 1)
                # Parse right operand
                right, pos = parse_prefix(tokens, pos)

                # Add parentheses based on precedence
                if token in ["mul", "div"] and ("+" in left or "-" in left):
                    left = f"({left})"
                if token in ["mul", "div"] and ("+" in right or "-" in right):
                    right = f"({right})"
                if token == "pow":
                    if any(op in left for op in ["+", "-", "*", "/"]):
                        left = f"({left})"

                return f"{left} {op} {right}", pos

            # Unary operators
            elif token == "neg":
                operand, pos = parse_prefix(tokens, pos + 1)
                return f"-{operand}", pos

            # Functions (sin, cos, etc.)
            elif token in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs"]:
                arg, pos = parse_prefix(tokens, pos + 1)
                return f"{token}({arg})", pos

            # Derivatives
            elif token in self.vocab.derivatives:
                arg, pos = parse_prefix(tokens, pos + 1)
                return f"{token}({arg})", pos

            # Variables
            elif token in self.vocab.variables:
                return token, pos + 1

            # Numbers
            elif token in ["INT+", "INT-"]:
                num, pos = reconstruct_number(tokens, pos)
                return num, pos

            # Direct number tokens (shouldn't happen in proper prefix, but handle it)
            elif token.isdigit() or token == ".":
                return token, pos + 1

            # Unknown token
            else:
                return token, pos + 1

        infix, _ = parse_prefix(prefix_tokens, 0)
        return infix

    def decode_to_infix(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs directly to human-readable infix notation

        Args:
            ids: List of token IDs
            skip_special_tokens: Skip <PAD>, <SOS>, <EOS>

        Returns:
            Infix notation string (human-readable)
        """
        tokens = self.vocab.ids_to_tokens(ids)

        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]]

        return self.prefix_to_infix(tokens)

    def encode_batch(
        self,
        pde_strings: List[str],
        add_special_tokens: bool = True,
        pad: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch of PDE strings to padded tensor

        Args:
            pde_strings: List of PDE strings
            add_special_tokens: Add <SOS> and <EOS>
            pad: Pad to same length
            max_length: Maximum length (truncate if needed)

        Returns:
            Dictionary with:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - lengths: (batch_size,)
        """
        # Encode all strings
        all_ids = [self.encode(pde, add_special_tokens) for pde in pde_strings]

        # Get lengths
        lengths = [len(ids) for ids in all_ids]

        if pad:
            # Determine max length
            if max_length is None:
                max_len = max(lengths)
            else:
                max_len = max_length
                # Truncate if needed
                all_ids = [ids[:max_len] for ids in all_ids]
                lengths = [min(l, max_len) for l in lengths]

            # Pad sequences
            padded_ids = []
            attention_mask = []
            for ids, length in zip(all_ids, lengths):
                # Pad
                padding_length = max_len - length
                padded = ids + [self.vocab.pad_id] * padding_length
                mask = [1] * length + [0] * padding_length

                padded_ids.append(padded)
                attention_mask.append(mask)

            return {
                "input_ids": torch.LongTensor(padded_ids),
                "attention_mask": torch.LongTensor(attention_mask),
                "lengths": torch.LongTensor(lengths),
            }
        else:
            return {
                "input_ids": [torch.LongTensor(ids) for ids in all_ids],
                "lengths": torch.LongTensor(lengths),
            }

    def to_one_hot(
        self, ids: List[int], vocab_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert token IDs to one-hot encoding (for Grammar VAE)

        Args:
            ids: List of token IDs
            vocab_size: Vocabulary size (default: self.vocab.vocab_size)

        Returns:
            One-hot matrix of shape (seq_len, vocab_size)
        """
        if vocab_size is None:
            vocab_size = self.vocab.vocab_size

        one_hot = np.zeros((len(ids), vocab_size), dtype=np.float32)
        for i, idx in enumerate(ids):
            if 0 <= idx < vocab_size:
                one_hot[i, idx] = 1.0

        return one_hot

    def batch_to_one_hot(
        self, pde_strings: List[str], max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert batch of PDEs to one-hot encoding

        Args:
            pde_strings: List of PDE strings
            max_length: Maximum sequence length (pad/truncate)

        Returns:
            One-hot tensor of shape (batch_size, max_seq_len, vocab_size)
        """
        # Encode all
        all_ids = [self.encode(pde, add_special_tokens=True) for pde in pde_strings]

        # Determine max length
        if max_length is None:
            max_len = max(len(ids) for ids in all_ids)
        else:
            max_len = max_length

        # Create batch one-hot
        batch_size = len(pde_strings)
        batch_one_hot = np.zeros(
            (batch_size, max_len, self.vocab.vocab_size), dtype=np.float32
        )

        for i, ids in enumerate(all_ids):
            # Truncate if needed
            ids = ids[:max_len]
            # Pad if needed
            ids = ids + [self.vocab.pad_id] * (max_len - len(ids))
            # Convert to one-hot
            batch_one_hot[i] = self.to_one_hot(ids)

        return batch_one_hot


def main():
    """Example usage"""
    # Create tokenizer
    tokenizer = PDETokenizer()

    print("PDE Tokenizer - Lample & Charton Style")
    print("=" * 80)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Special tokens: {tokenizer.vocab.special_tokens}")
    print(f"Sample derivatives: {tokenizer.vocab.derivatives[:10]}")
    print()

    # Example PDEs
    examples = [
        "dt(u) - 2.345*dxx(u) = 0",
        "dtt(u) - 1.5*dxx(u) - 1.5*dyy(u) = 0",
        "dt(u) + u*dx(u) - 0.5*dxx(u) = 0",
        "dt(u) - dxx(u) + u^3 - u = 0",
    ]

    print("Examples:")
    print("-" * 80)
    for pde in examples:
        print(f"\nOriginal (infix): {pde}")

        # Tokenize
        tokens = tokenizer.tokenize_infix(pde)
        print(f"Tokens: {tokens[:20]}...")

        # Convert to prefix
        prefix = tokenizer.infix_to_prefix_simple(pde)
        print(f"Prefix: {prefix}")

        # Encode to IDs
        ids = tokenizer.encode(pde, add_special_tokens=True)
        print(f"IDs: {ids}")

        # Decode back
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"Decoded: {decoded}")

    # Batch encoding
    print("\n" + "=" * 80)
    print("Batch Encoding:")
    batch = tokenizer.encode_batch(examples, pad=True)
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Lengths: {batch['lengths']}")

    # One-hot encoding (for Grammar VAE)
    print("\n" + "=" * 80)
    print("One-Hot Encoding (for Grammar VAE):")
    one_hot = tokenizer.batch_to_one_hot(examples[:2], max_length=50)
    print(f"One-hot shape: {one_hot.shape}")
    print(f"  (batch_size, max_seq_len, vocab_size)")

    # Save vocabulary
    tokenizer.vocab.save("pde_vocab.txt")
    print(f"\nVocabulary saved to pde_vocab.txt")


if __name__ == "__main__":
    main()
