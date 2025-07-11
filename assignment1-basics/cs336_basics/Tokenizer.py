import logging
from collections.abc import Iterable
from collections.abc import Iterator

import regex as re

logger = logging.getLogger(__name__)
# change the logging level to DEBUG to see debug messages
logging.basicConfig(level=logging.DEBUG)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Initializes the Tokenizer with vocabulary, merges, and special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []

    def pre_tokenize(self, text: str) -> list[tuple[bytes]]:
        """
        Pre-tokenizes the input text by splitting it into segments based on special tokens and applying regex patterns.
        If the text contains special tokens, it splits the text on those tokens while preserving them.
        Args:
            text (str): The input text to be pre-tokenized.
        Returns:
            list[tuple[bytes]]: A list of tuples, where each tuple contains bytes representing the pre-tokenized segments.

        Example: text = "the cat ate" output -> [(b't', b'h', 'e'), (b' ', 'c', b'a', b't'), (b' ', b'a', b't', b'e')]
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokenized = []
        # Split on all special tokens while preserving them
        segments = [text]
        for special_token in sorted(self.special_tokens, key=len, reverse=True):
            # sort on length to ensure longer tokens are split first
            new_segments = []
            for segment in segments:
                # Only split string segments, not already identified special tokens
                if isinstance(segment, str) and segment not in self.special_tokens and special_token in segment:
                    parts = segment.split(special_token)
                    for i, part in enumerate(parts):
                        if i > 0:  # Add the special token before each part except the first
                            new_segments.append(special_token)
                        if part:  # Only add non-empty parts
                            new_segments.append(part)
                else:
                    new_segments.append(segment)
            segments = new_segments

        logger.debug(f"Pre-tokenizing text: {text}")
        logger.debug(f"Segments after splitting on special tokens: {segments}")

        for segment in segments:
            if segment in self.special_tokens:
                # Special tokens are handled as single units
                pre_tokenized.append((segment.encode("utf-8"),))
            else:
                for m in re.finditer(PAT, segment):
                    token = m.group(0).encode("utf-8")
                    chunk = [token[i : i + 1] for i in range(len(token))]
                    pre_tokenized.append(tuple(chunk))
        logger.debug(f"Pre-tokenized segments: {pre_tokenized}")
        return pre_tokenized

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method to create a tokenizer from a serialized vocabulary and merges files.
        Args:
            vocab_filepath (str): Path to the vocabulary file.
            merges_filepath (str): Path to the merges file.
            special_tokens (list[str] | None): List of special tokens to be used in the tokenizer.
        """
        import json

        # Load vocabulary from JSON file
        with open(vocab_filepath) as f:
            vocab_json = json.load(f)

        # Convert vocabulary back to proper format (int keys, bytes values)
        vocab = {}
        for k, v in vocab_json.items():
            vocab[int(k)] = v.encode("utf-8")

        # Load merges from JSON file
        with open(merges_filepath) as f:
            merges_json = json.load(f)

        # Convert merges back to proper format (tuple of bytes)
        merges = []
        for pair in merges_json:
            merges.append((pair[0].encode("utf-8"), pair[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_merges(self, tokens: tuple[bytes]) -> tuple[bytes]:
        """
        Applies merges to the tokens in the order we obtained them
        """
        for merge in self.merges:
            # Keep applying this merge until no more instances are found
            while True:
                logger.debug(f"trying to merge: {merge}")
                # Recompute the pairs after each merge
                pairs = list(zip(tokens, tokens[1:]))
                logger.debug(f"pairs: {pairs}")
                if merge not in pairs:
                    break
                logger.debug(f"Found merge pair: {merge}")

                # Find the first occurrence of the merge pair
                for i in range(len(tokens) - 1):
                    if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                        # Perform the merge at position i
                        tokens = tokens[:i] + (merge[0] + merge[1],) + tokens[i + 2 :]
                        logger.debug(f"Merged {merge} into {tokens}")
                        logger.debug(f"Tokens after merge: {tokens}")
                        break
        return tokens

    def encode(self, text: str) -> list[int]:
        encoded_tokens = []
        for chunk in self.pre_tokenize(text):
            logger.debug(f"Processing chunk: {chunk}\n\n")

            # Check if this chunk is a special token
            if len(chunk) == 1 and chunk[0].decode("utf-8") in self.special_tokens:
                special_token = chunk[0]
                special_token_id = self.inv_vocab.get(special_token, -1)
                encoded_tokens.append(special_token_id)
                logger.debug(f"Added special token {special_token} with ID {special_token_id}")
            else:
                tokens_after_merge = self._apply_merges(chunk)
                logger.debug(f"Tokens after applying merges: {tokens_after_merge}")
                for token in tokens_after_merge:
                    if token in self.inv_vocab:
                        encoded_tokens.append(self.inv_vocab[token])
                    else:
                        logger.warning(f"Token {token} not found in vocabulary.")
                        encoded_tokens.append(self.inv_vocab.get(b"<unk>", -1))
            logger.debug("finished!!\n\n")

        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (example a python file handle), return a
        generator that lazy yields the encoded tokens for each string.
        This is required for memory efficient tokenization of large files
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        encoded_tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                encoded_tokens.append(self.vocab[token_id])
            else:
                logger.warning(f"Token ID {token_id} not found in vocabulary.")
                encoded_tokens.append(b"<unk>")
        return b"".join(encoded_tokens).decode("utf-8", errors="replace")


if __name__ == "__main__":
    text = "the cat ate"
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens={})
    encoded = tokenizer.encode(text)

    print(encoded)
