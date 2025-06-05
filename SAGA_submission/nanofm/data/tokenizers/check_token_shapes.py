import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_vocab_size(token_dir: Path, num_files: int = 50):
    """
    Analyze the vocabulary size used in audio tokens.
    Args:
        token_dir: Directory containing .npy files
        num_files: Number of files to analyze
    """
    print(f"🔍 Analyzing vocabulary of audio tokens")
    print(f"📁 Directory: {token_dir}")
    print(f"📊 Analyzing {num_files} files...\n")

    all_tokens = []
    shapes = []
    token_files = list(token_dir.glob("*.npy"))[:num_files]

    if not token_files:
        print(f"❌ No .npy files found in {token_dir}")
        return

    for i, token_file in enumerate(token_files):
        try:
            tokens = np.load(token_file)
            all_tokens.append(tokens)
            shapes.append(tokens.shape)
            if i % 10 == 0:
                print(f"  Processing: {i+1}/{len(token_files)} files...")
        except Exception as e:
            print(f"❌ Error with {token_file.name}: {e}")

    if not all_tokens:
        print("❌ No valid tokens found!")
        return

    print(f"\n{'='*70}")
    print("📊 ANALYSIS BY CODEBOOK")
    print(f"{'='*70}")

    num_codebooks = all_tokens[0].shape[1]
    print(f"🔢 Number of codebooks: {num_codebooks}")
    vocab_sizes_per_codebook = []

    for codebook_idx in range(num_codebooks):
        codebook_tokens = []
        for tokens in all_tokens:
            codebook_tokens.extend(tokens[:, codebook_idx].flatten())
        unique_tokens = np.unique(codebook_tokens)
        vocab_size = len(unique_tokens)
        min_token = min(codebook_tokens)
        max_token = max(codebook_tokens)
        vocab_sizes_per_codebook.append(vocab_size)
        print(f"  Codebook {codebook_idx:2d}: Vocab={vocab_size:4d}, Range=[{min_token:4d}, {max_token:4d}]")
        if codebook_idx < 3:
            counter = Counter(codebook_tokens)
            most_common = counter.most_common(5)
            print(f"    Top 5 tokens: {most_common}")

    print(f"\n{'='*70}")
    print("📈 GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"🔢 Vocab size per codebook:")
    print(f"  Min: {min(vocab_sizes_per_codebook)}")
    print(f"  Max: {max(vocab_sizes_per_codebook)}")
    print(f"  Mean: {np.mean(vocab_sizes_per_codebook):.1f}")
    all_vocab_same = len(set(vocab_sizes_per_codebook)) == 1
    print(f"🎯 Uniform vocabulary: {'✅ Yes' if all_vocab_same else '❌ No'}")
    if all_vocab_same:
        vocab_size = vocab_sizes_per_codebook[0]
        print(f"🎉 Vocabulary size: {vocab_size}")
        import math
        if vocab_size > 0 and (vocab_size & (vocab_size - 1)) == 0:
            bits = int(math.log2(vocab_size))
            print(f"💡 It's 2^{bits} (power of 2)")
    print(f"\n📋 ADDITIONAL INFORMATION:")
    theoretical_max = 2048  # Theoretical vocab size for Mimi
    if all_vocab_same and vocab_sizes_per_codebook[0] <= theoretical_max:
        usage_density = vocab_sizes_per_codebook[0] / theoretical_max * 100
        print(f"📊 Usage density of theoretical vocab: {usage_density:.1f}%")
    unique_shapes = list(set(shapes))
    print(f"🔳 Found shapes: {unique_shapes}")
    total_tokens = sum(tokens.size for tokens in all_tokens)
    total_files = len(all_tokens)
    avg_tokens_per_file = total_tokens / total_files
    print(f"📏 Average tokens per file: {avg_tokens_per_file:.0f}")

def check_specific_vocab_properties(token_path: Path):
    """
    Specific checks for Mimi tokens.
    """
    print(f"🔬 Checking Mimi properties for: {token_path.name}")
    tokens = np.load(token_path)
    expected_codebooks = 32
    expected_vocab_size = 2048
    expected_range = (0, 2047)
    print(f"📐 Shape: {tokens.shape}")
    print(f"🔍 Mimi checks:")
    actual_codebooks = tokens.shape[1] if tokens.ndim == 2 else 1
    print(f"  Codebooks: {actual_codebooks} {'✅' if actual_codebooks == expected_codebooks else '❌'} (expected: {expected_codebooks})")
    min_val, max_val = tokens.min(), tokens.max()
    range_ok = min_val >= expected_range[0] and max_val <= expected_range[1]
    print(f"  Range: [{min_val}, {max_val}] {'✅' if range_ok else '❌'} (expected: {expected_range})")
    unique_tokens = len(np.unique(tokens))
    print(f"  Unique tokens: {unique_tokens} {'✅' if unique_tokens <= expected_vocab_size else '❌'} (max: {expected_vocab_size})")

def main():
    parser = argparse.ArgumentParser("Analyze the vocab size of audio tokens")
    parser.add_argument("token_path", type=Path,
                       help="Path to the token directory or file")
    parser.add_argument("--num_files", type=int, default=50,
                       help="Number of files to analyze")
    parser.add_argument("--quick_check", action="store_true",
                       help="Quick check of a single file")
    args = parser.parse_args()
    if args.token_path.is_file():
        check_specific_vocab_properties(args.token_path)
    elif args.token_path.is_dir():
        analyze_vocab_size(args.token_path, args.num_files)
        sample_file = next(args.token_path.glob("*.npy"), None)
        if sample_file:
            print(f"\n{'='*70}")
            check_specific_vocab_properties(sample_file)
    else:
        print(f"❌ Invalid path: {args.token_path}")

if __name__ == "__main__":
    main()