#!/usr/bin/env python3
"""
Quick test script for suffix_cache C++ extension.
Run this after building to verify the extension works correctly.
"""

import sys


def test_import():
    """Test that the extension can be imported."""
    print("Test 1: Importing extension...")
    try:
        from sglang.srt.speculative.suffix_cache._C import SuffixTree, Candidate
        print("✓ Import successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of SuffixTree."""
    print("\nTest 2: Basic functionality...")
    try:
        from sglang.srt.speculative.suffix_cache._C import SuffixTree

        # Create tree
        tree = SuffixTree(64)
        print("✓ Created SuffixTree with max_depth=64")

        # Add tokens
        tree.extend(0, [1, 2, 3, 4, 5])
        print("✓ Extended tree with tokens [1, 2, 3, 4, 5]")

        # Speculate
        candidate = tree.speculate(
            pattern=[3, 4],
            max_spec_tokens=8,
            max_spec_factor=1.0,
            max_spec_offset=0.0,
            min_token_prob=0.1,
            use_tree_spec=False,
        )
        print(f"✓ Speculation completed: {len(candidate.token_ids)} tokens")
        print(f"  - Token IDs: {candidate.token_ids}")
        print(f"  - Parents: {candidate.parents}")
        print(f"  - Match length: {candidate.match_len}")
        print(f"  - Score: {candidate.score:.4f}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_sequence():
    """Test multiple sequences."""
    print("\nTest 3: Multiple sequences...")
    try:
        from sglang.srt.speculative.suffix_cache._C import SuffixTree

        tree = SuffixTree(64)

        # Add tokens to multiple sequences
        tree.extend(0, [10, 20, 30, 40, 50])
        tree.extend(1, [10, 20, 30, 60, 70])
        tree.extend(2, [10, 20, 30, 40, 50])  # Same as seq 0

        print(f"✓ Created tree with {tree.num_seqs()} sequences")

        # Speculate on pattern that appears in multiple sequences
        candidate = tree.speculate(
            pattern=[10, 20, 30],
            max_spec_tokens=5,
            max_spec_factor=1.0,
            max_spec_offset=0.0,
            min_token_prob=0.1,
            use_tree_spec=False,
        )

        print(f"✓ Speculation on common pattern: {len(candidate.token_ids)} tokens")
        print(f"  - Token IDs: {candidate.token_ids}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_append():
    """Test single token append."""
    print("\nTest 4: Single token append...")
    try:
        from sglang.srt.speculative.suffix_cache._C import SuffixTree

        tree = SuffixTree(64)

        # Test append (single token)
        for token in [100, 200, 300, 400, 500]:
            tree.append(0, token)

        print("✓ Appended 5 tokens one by one")

        candidate = tree.speculate(
            pattern=[300, 400],
            max_spec_tokens=3,
            max_spec_factor=1.0,
            max_spec_offset=0.0,
            min_token_prob=0.1,
            use_tree_spec=False,
        )

        print(f"✓ Speculation after append: {len(candidate.token_ids)} tokens")
        print(f"  - Token IDs: {candidate.token_ids}")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("suffix_cache C++ Extension Test Suite")
    print("=" * 60)

    results = []
    results.append(("Import", test_import()))

    if results[0][1]:  # Only continue if import succeeded
        results.append(("Basic functionality", test_basic_functionality()))
        results.append(("Multiple sequences", test_multi_sequence()))
        results.append(("Append operation", test_append()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
