from cs336_basics.BPETrainer import pre_tokenize, OptimizedBPETrainer
from collections import defaultdict

# Test the stylized example
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

print("=== DEBUGGING BPE ===")
print(f"Input text: {text}")

# Manual pre-tokenization for debugging
tokens = text.split()
freq_table = defaultdict(int)
for token in tokens:
    token_bytes = token.encode('utf-8')
    byte_tuple = tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))
    freq_table[byte_tuple] += 1

print(f"\nInitial frequency table:")
for token, freq in freq_table.items():
    token_str = b''.join(token).decode('utf-8')
    print(f"  {token_str}: {freq}")

# Initialize pair counts manually to debug
pair_counts = defaultdict(int)
for token_tuple, freq in freq_table.items():
    if len(token_tuple) < 2:
        continue
    for i in range(len(token_tuple) - 1):
        pair = (token_tuple[i], token_tuple[i + 1])
        pair_counts[pair] += freq

print(f"\nInitial pair counts:")
for pair, count in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0])):
    pair_str = b''.join(pair).decode('utf-8')
    print(f"  '{pair_str}': {count}")

# Check the most frequent pair
most_frequent = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
print(f"\nMost frequent pair: {most_frequent[0]} with count {most_frequent[1]}")
pair_str = b''.join(most_frequent[0]).decode('utf-8')
print(f"  As string: '{pair_str}'")