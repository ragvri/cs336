import os
from typing import BinaryIO
from cs336_basics.bpe_scratch import pre_tokenize
from collections import defaultdict
from multiprocessing import Pool, cpu_count

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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




def get_frequency_table(filepath: str, special_tokens: list[str] = ["<|endoftext|>"]) -> dict[tuple[bytes], int]:
    """
    Get the frequency table for the entire file.
    """
    with open(filepath, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    return pre_tokenize(text, special_tokens)   # ← plain, serial



final_freq = defaultdict(int)

if __name__ == "__main__":
    final_freq = get_frequency_table("/home/rjindal/local_code/cs336-stanford/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    print(f"Final frequency table size: {len(final_freq)}")
    assert "<|endoftext|>" not in final_freq, "Special token found in frequency table"

    # peak some items
    for k, v in list(final_freq.items())[:10]:
        print(f"{k}: {v}")