import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

initial_vocab = {i:bytes([i]) for i in range(256)}
initial_vocab[256] = bytes('<|endoftext|>'.encode('utf-8'))  # Special token for end of text


def pre_tokenize(text:str, special_tokens:list[str] = ["<|endoftext|>"]) -> dict[tuple[bytes],int]:
    """
    Given a sting, pre-tokenize using the regex pattern and return a frequency table
    """

    freq_table = defaultdict(int)

    split_tokens = special_tokens[0]

    for segment in text.split(split_tokens):
        for m in re.finditer(PAT, segment):
            token = m.group(0).encode('utf-8')
            freq_table[tuple(token[i:i+1] for i in range(len(token)))] += 1

    return freq_table

def get_byte_pair_frequency(bytes_freq_table: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    byte_pair_freq = defaultdict(int)

    for bytes_tuple, freq in bytes_freq_table.items():
        if len(bytes_tuple) < 2:
            continue
        for byte1, byte2 in zip(bytes_tuple, bytes_tuple[1:]):
            byte_pair_freq[(byte1, byte2)] += freq
    return byte_pair_freq

def get_most_frequent_pair(byte_pair_freq: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    Given a byte pair frequency table, return the most frequent pair
    """
    most_frequent_pair = max(byte_pair_freq.items(), key=lambda item: (item[1], item[0]))
    return most_frequent_pair[0]  # Return the pair, not the frequency

def merge(bytes_freq_table: dict[tuple[bytes], int], pair_to_merge: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:

    # iterate over the bytes freq table pair and if the pair is same as pair to merge, then merge it
    merged_freq_table = defaultdict(int)
    for bytes_tuple, freq in bytes_freq_table.items():
        if len(bytes_tuple) < 2:
            merged_freq_table[(bytes_tuple)] += freq
        else:
            tup = []
            i = 0
            while i < len(bytes_tuple):
                if i+1 < len(bytes_tuple) and bytes_tuple[i] == pair_to_merge[0] and bytes_tuple[i+1] == pair_to_merge[1]:
                    new_bytes_tuple = pair_to_merge[0] + pair_to_merge[1]
                    tup.append(new_bytes_tuple)
                    i+= 1  # Skip the next byte as it is already merged
                else:
                    tup.append(bytes_tuple[i])
                i += 1

            merged_freq_table[tuple(tup)] += freq
        
    return merged_freq_table

def repeated_merges(freq_table: dict[tuple[bytes], int], num_merges: int) -> dict[tuple[bytes], int]:
    """
    Perform repeated merges on the frequency table for a given number of merges
    """
    for _ in range(num_merges):
        byte_pair_freq = get_byte_pair_frequency(freq_table)
        most_frequent_pair = get_most_frequent_pair(byte_pair_freq)
        freq_table = merge(freq_table, most_frequent_pair)
        initial_vocab[len(initial_vocab)] = most_frequent_pair[0] + most_frequent_pair[1]
    
    return freq_table

            
        
    
if __name__ == "__main__":
    print(f"************initial_vocab:\n\n\n\n {initial_vocab}************\n\n\n")        


    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    freq_table = pre_tokenize(text)
    print("Pre-tokenized frequency table:")
    print(freq_table)

    num_merges = 6
    final_freq_table = repeated_merges(freq_table, num_merges)
    print(f"Final vocab after {num_merges} merges:\n\n\n\n {initial_vocab}************\n\n\n")
