import sys
sys.path.append('.')

from cs336_basics.BPETrainer import OptimizedBPETrainer
from collections import defaultdict

# Create a simple test case
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

# Write to a test file
with open("test_input.txt", "w") as f:
    f.write(text)

print("=== Testing BPE with Stylized Example ===")
print(f"Input: {text}")

# Create trainer
trainer = OptimizedBPETrainer(
    input_path="test_input.txt",
    vocab_size=300,  # Small vocab to test
    special_tokens=["<|endoftext|>"]
)

# Get frequency table manually to debug
print("\n1. Getting frequency table...")
try:
    freq_table = trainer.get_frequency_table_parallel()
    print(f"   Frequency table has {len(freq_table)} unique tokens")
    
    # Show first few tokens
    for i, (token, freq) in enumerate(freq_table.items()):
        if i >= 5: break
        token_str = b''.join(token).decode('utf-8')
        print(f"   '{token_str}': {freq}")
        
    print("\n2. Initializing pair counts...")
    trainer.initialize_pair_counts(freq_table)
    print(f"   Found {len(trainer.pair_counts)} unique pairs")
    
    # Show top pairs
    sorted_pairs = sorted(trainer.pair_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
    for pair, count in sorted_pairs:
        pair_str = b''.join(pair).decode('utf-8')
        print(f"   '{pair_str}': {count}")
        
    print("\n3. Testing first merge...")
    most_frequent_pair = max(trainer.pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
    pair_str = b''.join(most_frequent_pair).decode('utf-8')
    print(f"   Most frequent pair: '{pair_str}' with count {trainer.pair_counts[most_frequent_pair]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()