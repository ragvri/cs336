"""
Byte-Pair Encoding (BPE) Training Algorithm Implementation

BPE is a subword tokenization algorithm that learns to merge the most frequent pairs of bytes/characters
to build a vocabulary. This implementation is optimized for large-scale training with parallel processing.

=== CORE BPE ALGORITHM ===

1. **Pre-tokenization**: Split input text into initial tokens using regex patterns
   - Uses regex pattern to split text into words, punctuation, whitespace
   - Handles special tokens (like <|endoftext|>) as document boundaries
   - Each character becomes a single byte in the initial vocabulary (0-255)

2. **Initial Frequency Counting**: Convert tokens to byte sequences and count frequencies
   Example: "low below" becomes:
   - Token "low": (b'l', b'o', b'w') with frequency 1
   - Token "below": (b'b', b'e', b'l', b'o', b'w') with frequency 1

3. **Pair Frequency Counting**: Count all adjacent byte pairs across all tokens
   From the example above:
   - (b'l', b'o'): appears 2 times (once in "low", once in "below")
   - (b'o', b'w'): appears 2 times
   - (b'b', b'e'): appears 1 time
   - (b'e', b'l'): appears 1 time

4. **Iterative Merging**: Repeatedly merge the most frequent pair until vocabulary size reached
   - Find most frequent pair (e.g., (b'l', b'o') with count 2)
   - Merge this pair into single token: b'lo'
   - Update all affected tokens: "low" → (b'lo', b'w'), "below" → (b'b', b'e', b'lo', b'w')
   - Recalculate pair frequencies and repeat

5. **Vocabulary Building**: Each merge creates a new vocabulary entry
   - Initial vocab: 256 single bytes (0-255) + special tokens
   - Each merge adds one new token: vocab[256] = b'lo', vocab[257] = b'ow', etc.
   - Final vocabulary size determined by max_vocab_size parameter

=== KEY DATA STRUCTURES ===

- `freq_table`: Maps token tuples to their frequencies
  Example: {(b'l', b'o', b'w'): 1, (b'b', b'e', b'l', b'o', b'w'): 1}

- `pair_counts`: Maps byte pairs to total frequency across all tokens
  Example: {(b'l', b'o'): 2, (b'o', b'w'): 2, (b'b', b'e'): 1}

- `vocab`: Maps vocabulary IDs to byte sequences
  Example: {0: b'\\x00', 1: b'\\x01', ..., 256: b'<|endoftext|>', 257: b'lo'}

=== OPTIMIZATIONS IMPLEMENTED ===

**1. Parallel Pre-tokenization**:
- Splits large files into chunks at special token boundaries
- Processes chunks in parallel using multiprocessing
- Combines frequency tables from all chunks (MapReduce pattern)

**2. Incremental Pair Counting**:
- Maintains `pair_locations` to track which tokens contain each pair
- After merging pair X, only updates tokens that contained X (not all tokens)
- Transforms O(vocab_size × num_merges × avg_token_length) to O(affected_tokens × num_merges)

**3. Optimized Merging**:
- Uses pair_locations to identify exactly which tokens need merging
- Avoids iterating through entire frequency table for each merge
- Significant speedup for large vocabularies (10-100x faster)

=== USAGE ===

```python
trainer = OptimizedBPETrainer(
    input_path="data/text.txt",
    vocab_size=50000,
    special_tokens=["<|endoftext|>"]
)
vocab, merges = trainer.train()
trainer.serialize(vocab, merges)
```

The algorithm produces:
- `vocab`: Dictionary mapping token IDs to byte sequences
- `merges`: List of merge operations applied during training
- Serialized JSON files for later use in tokenization

"""

from collections import defaultdict
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
import logging
import collections


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(text: str, special_tokens: list[str] = ["<|endoftext|>"]) -> dict[tuple[bytes], int]:
    """
    Given a sting, pre-tokenize using the regex pattern and return a frequency table
    """

    freq_table = defaultdict(int)

    # Split on all special tokens
    segments = [text]
    for special_token in special_tokens:
        new_segments = []
        for segment in segments:
            new_segments.extend(segment.split(special_token))
        segments = new_segments

    for segment in segments:
        for m in re.finditer(PAT, segment):
            token = m.group(0).encode("utf-8")
            freq_table[tuple(token[i : i + 1] for i in range(len(token)))] += 1

    return freq_table


def process_chunk_for_pretokenization(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes], int]:
    """
    Process a single chunk for pre-tokenization.
    Args: tuple of (filepath, start, end, special_tokens)
    Returns: frequency dictionary from pre_tokenize
    """
    filepath, start, end, special_tokens = args

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        return pre_tokenize(chunk, special_tokens=special_tokens)


class OptimizedBPETrainer:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> None:
        """
        Args:
            input_path (str | os.PathLike): Path to the input file.
            vocab_size (int): Maximum vocabulary size.
            special_tokens (list[str]): List of special tokens to include in the vocabulary.
        """

        self.pair_counts = defaultdict(int)  # counts of byte pairs
        self.pair_locations = defaultdict(list)  # which intial tokens contain the pair along with their frequencies

        self.max_vocab_size = vocab_size  # max vocabulary size
        self.special_tokens = special_tokens
        self.filepath = input_path

        self.vocab = {i: bytes([i]) for i in range(256)}

        input_filename = os.path.basename(input_path)

        self.output_vocab_path = (
            f"/home/rjindal/local_code/cs336-stanford/assignment1-basics/data/bpe_vocab_{input_filename}.json"
        )
        self.output_merges_path = (
            f"/home/rjindal/local_code/cs336-stanford/assignment1-basics/data/bpe_merges_{input_filename}.json"
        )

    def find_chunk_boundaries(self, file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def get_frequency_table_parallel(self) -> dict[tuple[bytes], int]:
        """
        Get the frequency table for the entire file.
        """
        logging.info(f"Starting frequency table generation for file: {self.filepath}")
        desired_chunks = 100  # Number of chunks to create
        split_special_token = self.special_tokens[0].encode("utf-8")
        logging.info(f"Special token for splitting: {split_special_token.decode('utf-8')}")
        boundaries = self.find_chunk_boundaries(open(self.filepath, "rb"), desired_chunks, split_special_token)

        logging.info(f"Chunk boundaries found: {boundaries}")
        final_freq = defaultdict(int)

        # Prepare chunk arguments for parallel processing (MAP phase)
        chunk_args = [
            (self.filepath, start, end, self.special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        # Use optimal number of processes: min of (CPU cores, number of chunks)
        num_processes = min(cpu_count(), len(chunk_args))
        # MAP: Process each chunk in parallel using multiprocessing
        with Pool(processes=num_processes) as pool:
            chunk_frequencies = pool.map(process_chunk_for_pretokenization, chunk_args)

        # REDUCE: Combine all frequency dictionaries
        for freq in chunk_frequencies:
            for k, v in freq.items():
                final_freq[k] += v

        return final_freq

    def initialize_pair_counts(self, freq_table: dict[tuple[bytes], int]) -> None:
        """
        Initialize pair counts and locations from the frequency table.
        """
        for bytes_tuple, freq in freq_table.items():
            if len(bytes_tuple) < 2:
                continue
            for i in range(len(bytes_tuple) - 1):
                pair = (bytes_tuple[i], bytes_tuple[i + 1])
                self.pair_counts[pair] += freq
                self.pair_locations[pair].append((bytes_tuple, freq))

    def update_pair_counts_after_merge(
        self,
        old_freq_table: dict[tuple[bytes], int],
        new_freq_table: dict[tuple[bytes], int],
        merged_pair: tuple[bytes, bytes],
    ) -> None:
        """
        Update pair counts and locations after merging a pair.
        """
        # Step 1: Remove the merged pair entirely first
        if merged_pair in self.pair_counts:
            del self.pair_counts[merged_pair]
        if merged_pair in self.pair_locations:
            del self.pair_locations[merged_pair]

        # Step 2: Find all tokens that changed (were affected by the merge)
        changed_tokens = set()

        # Tokens that disappeared from old_freq_table
        for token in old_freq_table:
            if token not in new_freq_table:
                changed_tokens.add(token)

        # Tokens that appeared in new_freq_table
        for token in new_freq_table:
            if token not in old_freq_table:
                changed_tokens.add(token)

        # Tokens that changed frequency
        for token in old_freq_table:
            if token in new_freq_table and old_freq_table[token] != new_freq_table[token]:
                changed_tokens.add(token)

        # Step 3: Remove pair counts from old versions of changed tokens
        for token in changed_tokens:
            if token in old_freq_table:
                old_freq = old_freq_table[token]
                if len(token) >= 2:
                    for i in range(len(token) - 1):
                        pair = (token[i], token[i + 1])
                        if pair in self.pair_counts:
                            self.pair_counts[pair] -= old_freq
                            if self.pair_counts[pair] <= 0:
                                del self.pair_counts[pair]

                        # Remove from pair_locations
                        if pair in self.pair_locations:
                            # Find and remove this specific token entry
                            self.pair_locations[pair] = [(t, f) for (t, f) in self.pair_locations[pair] if t != token]
                            if not self.pair_locations[pair]:
                                del self.pair_locations[pair]

        # Step 4: Add pair counts from new versions of changed tokens
        for token in changed_tokens:
            if token in new_freq_table:
                new_freq = new_freq_table[token]
                if len(token) >= 2:
                    for i in range(len(token) - 1):
                        pair = (token[i], token[i + 1])
                        self.pair_counts[pair] += new_freq
                        self.pair_locations[pair].append((token, new_freq))

    def apply_merge_to_token(self, token_tuple: tuple[bytes], pair_to_merge: tuple[bytes, bytes]) -> tuple[bytes]:
        """Apply a merge to a token tuple and return the new token tuple."""
        new_tuple = []
        i = 0
        while i < len(token_tuple):
            if (
                i + 1 < len(token_tuple)
                and token_tuple[i] == pair_to_merge[0]
                and token_tuple[i + 1] == pair_to_merge[1]
            ):
                new_tuple.append(pair_to_merge[0] + pair_to_merge[1])
                i += 2
            else:
                new_tuple.append(token_tuple[i])
                i += 1
        return tuple(new_tuple)

    def merge(self, freq_table: dict[tuple[bytes], int], pair_to_merge: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
        """
        Merge a pair in the frequency table and return the updated frequency table.
        """
        new_freq_table = freq_table.copy()

        affected_tokens = self.pair_locations[pair_to_merge].copy()

        for old_token_tuple, old_freq in affected_tokens:
            # Check if token still exists in current freq_table (might have been merged already)
            if old_token_tuple in new_freq_table:
                # Use the current frequency from freq_table, not the stored one
                current_freq = new_freq_table[old_token_tuple]

                # remove the old token
                del new_freq_table[old_token_tuple]

                # add the merged token
                new_token_tuple = self.apply_merge_to_token(old_token_tuple, pair_to_merge)
                if new_token_tuple in new_freq_table:
                    new_freq_table[new_token_tuple] += current_freq
                else:
                    new_freq_table[new_token_tuple] = current_freq

        return new_freq_table

    def repeated_merges(
        self, freq_table: dict[tuple[bytes], int]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Perform repeated merges on the frequency table for a given number of merges.
        """
        self.initialize_pair_counts(freq_table)

        current_freq_table = freq_table.copy()
        merges = []

        while len(self.vocab) < self.max_vocab_size:
            if len(self.vocab) % 100 == 0:
                logging.info(f"Current vocabulary size: {len(self.vocab)}")

            # Check if we have any pairs left to merge
            if not self.pair_counts:
                logging.info("No more pairs to merge - stopping early")
                break

            # Get the most frequent pair (GPT-2 style tie-breaking)
            most_frequent_pair = max(self.pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            merges.append(most_frequent_pair)

            old_freq_table = current_freq_table.copy()
            current_freq_table = self.merge(current_freq_table, most_frequent_pair)

            self.vocab[len(self.vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

            self.update_pair_counts_after_merge(old_freq_table, current_freq_table, most_frequent_pair)
        return self.vocab, merges

    def merge_slow(
        self,
        freq_table: dict[tuple[bytes], int],
        pair_to_merge: tuple[bytes, bytes],
    ) -> dict[tuple[bytes], int]:
        """
        Merge `pair_to_merge` everywhere in `freq_table` (slow but simple).
        """
        a, b = pair_to_merge
        new_freq = defaultdict(int)

        for token, freq in freq_table.items():
            merged = []
            i = 0
            while i < len(token):
                if i + 1 < len(token) and token[i] == a and token[i + 1] == b:
                    merged.append(a + b)  # concatenate the bytes objects
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            new_freq[tuple(merged)] += freq

        return new_freq

    def repeated_merges_slow(
        self,
        freq_table: dict[tuple[bytes], int],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        current_freq_table = freq_table
        merges: list[tuple[bytes, bytes]] = []

        while len(self.vocab) < self.max_vocab_size:
            # 1. recompute pair counts from scratch
            pair_counts = self.recompute_pair_counts(current_freq_table)

            if not pair_counts:  # no pairs left ⇒ done early
                break

            # 2. pick the most-frequent pair, breaking ties with concatenated bytes
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]

            merges.append(best_pair)

            # 3. merge that pair everywhere
            current_freq_table = self.merge_slow(current_freq_table, best_pair)

            # 4. add the new token to the vocab
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]

        return self.vocab, merges

    def recompute_pair_counts(self, freq_table: dict[tuple[bytes], int]) -> collections.Counter:
        pair_counts = collections.Counter()
        for token, freq in freq_table.items():
            for i in range(len(token) - 1):
                pair_counts[(token[i], token[i + 1])] += freq
        return pair_counts

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the BPE model and return the vocabulary and merges.
        """
        logging.info(f"Training BPE model on file: {self.filepath}")
        freq_table = self.get_frequency_table_parallel()
        logging.info(f"Frequency table size: {len(freq_table)}")

        for special_token in self.special_tokens:
            if isinstance(special_token, str):
                special_token = special_token.encode("utf-8")
            self.vocab[len(self.vocab)] = special_token

        vocab, merges = self.repeated_merges(freq_table)

        logging.info(f"Vocabulary size after training: {len(vocab)}")
        return vocab, merges

    def serialize(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        """
        Serialize the vocabulary and merges to a format suitable for saving.
        """
        import json

        vocab_json = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}
        merges_json = [
            (pair[0].decode("utf-8", errors="replace"), pair[1].decode("utf-8", errors="replace")) for pair in merges
        ]

        with open(self.output_vocab_path, "w") as f:
            json.dump(vocab_json, f, indent=2)

        with open(self.output_merges_path, "w") as f:
            json.dump(merges_json, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    text = "a quick brown fox jumps over the lazy dog. <|endoftext|> Another sentence. low lower lowest"
    # put to a temp file for testing
    with open("data/text.txt", "w") as f:
        f.write(text)

    trainer = OptimizedBPETrainer(input_path="data/text.txt", vocab_size=500, special_tokens=["<|endoftext|>"])
    vocab, merges = trainer.train()
    trainer.serialize(vocab, merges)
    logging.info("BPE training complete and results serialized.")
