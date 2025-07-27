import logging
from collections.abc import Iterable
from collections.abc import Iterator

import regex as re

logger = logging.getLogger(__name__)
# change the logging level to DEBUG to see debug messages
logging.basicConfig(level=logging.INFO)


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
        Class method to create a tokenizer from serialized vocabulary and merges files.
        Args:
            vocab_filepath (str): Path to the vocabulary file (.pkl or .json).
            merges_filepath (str): Path to the merges file (.pkl or .json).
            special_tokens (list[str] | None): List of special tokens to be used in the tokenizer.
        """
        import pickle

        # Load from pickle files
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_merges(self, tokens: tuple[bytes]) -> tuple[bytes]:
        """
        Applies merges to the tokens in the order we obtained them - optimized version
        """
        current_tokens = tokens

        for merge in self.merges:
            if len(current_tokens) < 2:
                break
                
            merge_pair = (merge[0], merge[1])
            merged_token = merge[0] + merge[1]
            
            # Early exit if merge pair not present
            if merge_pair[0] not in current_tokens or merge_pair[1] not in current_tokens:
                continue

            new_tokens = []
            i = 0
            
            while i < len(current_tokens):
                if (i < len(current_tokens) - 1 and 
                    current_tokens[i] == merge_pair[0] and 
                    current_tokens[i + 1] == merge_pair[1]):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            
            current_tokens = tuple(new_tokens)

        return current_tokens

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
                        # This should not happen with properly serialized vocabulary
                        raise ValueError(
                            f"Token {token} not found in vocabulary. This indicates a vocabulary corruption."
                        )
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
    # text = "the cat ate"
    # vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    # merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    # tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens={})
    # encoded = tokenizer.encode(text)
    import time

    start_time = time.time()
    tokenizer = Tokenizer.from_files(
        vocab_filepath="./data/bpe_vocab_TinyStoriesV2-GPT4-train.txt.pkl",
        merges_filepath="./data/bpe_merges_TinyStoriesV2-GPT4-train.txt.pkl",
        special_tokens=["<|endoftext|>"],
    )

    import numpy as np
    import os
    import tempfile

    # Dynamic memory-mapped array approach
    initial_capacity = 10_000_000  # Start with 10M tokens (20MB)
    capacity = initial_capacity
    token_count = 0

    # Create temporary memmap file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
    temp_path = temp_file.name
    temp_file.close()

    tokens_array = np.memmap(temp_path, dtype=np.uint16, mode='w+', shape=(capacity,))
    
    logger.info(f"Starting tokenization with initial capacity: {initial_capacity:,}")

    with open("./data/TinyStoriesV2-GPT4-train.txt") as input_file:
        for token_id in tokenizer.encode_iterable(input_file):
            # Check if we need to grow the array
            if token_count >= capacity:
                new_capacity = capacity * 2
                logger.info(f"Growing memmap from {capacity:,} to {new_capacity:,}")
                
                # Create new larger memmap
                new_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                new_temp_path = new_temp_file.name
                new_temp_file.close()
                
                new_tokens_array = np.memmap(new_temp_path, dtype=np.uint16, mode='w+', shape=(new_capacity,))
                
                # Copy existing data
                new_tokens_array[:capacity] = tokens_array[:]
                
                # Clean up old memmap
                del tokens_array
                os.unlink(temp_path)
                
                # Switch to new memmap
                tokens_array = new_tokens_array
                temp_path = new_temp_path
                capacity = new_capacity

            tokens_array[token_count] = token_id
            token_count += 1

            # Log progress
            if token_count % 100_000 == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Tokens written: {token_count:,} | Rate: {tokens_per_second:,.0f} tokens/sec")

    # Create final memmap with exact size  
    output_path = "./data/tinystories_encoded.dat"
    final_tokens = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(token_count,))
    final_tokens[:] = tokens_array[:token_count]
    final_tokens.flush()
    
    # Save metadata for loading later
    metadata_path = "./data/tinystories_encoded_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"tokens: {token_count}\n")
        f.write(f"dtype: uint16\n")
        f.write(f"shape: ({token_count},)\n")
    
    # Clean up temporary file
    del tokens_array
    del final_tokens
    os.unlink(temp_path)

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = token_count / total_time if total_time > 0 else 0

    logger.info(f"Tokenization complete. Encoded tokens written to {output_path} in {total_time:.2f} seconds.")
    logger.info(f"Total tokens: {token_count:,}")
    logger.info(f"Tokenization rate: {tokens_per_second:,.0f} tokens/second")
    
    # Print file size info
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")
    logger.info("Storage per token: 2 bytes (uint16, memory-mapped)")
    logger.info("Output saved as memory-mapped array with metadata file")
