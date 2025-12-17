#!/usr/bin/env python3
"""
Test the prep_test_cases workflow to ensure backward compatibility.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prep_test_cases import prep_test_cases
from inference.openrouter import load_cases


def test_backward_compatibility():
    """Test that old format (plain list) still works."""
    print("Testing backward compatibility...")
    
    # Test with old format (plain list)
    old_format_path = "results/artifacts/claude-sonnet-4-test.json"
    if not Path(old_format_path).exists():
        print(f"  Skipping: {old_format_path} not found")
        return True
    
    with open(old_format_path) as f:
        data = json.load(f)
    
    # Should be a plain list
    assert isinstance(data, list), "Old format should be a list"
    print(f"  ✓ Old format is a list with {len(data)} items")
    
    # Test load_cases handles it
    cases, metadata = load_cases(old_format_path)
    assert isinstance(cases, list), "load_cases should return list"
    assert metadata is None, "Old format should have no metadata"
    print(f"  ✓ load_cases handles old format correctly")
    
    return True


def test_new_format():
    """Test that new format (with metadata) works."""
    print("\nTesting new format...")
    
    # Check if we created a test set
    test_set_path = "data/test_sets/quick-test-10.json"
    if not Path(test_set_path).exists():
        print(f"  Skipping: {test_set_path} not found")
        return True
    
    # Test load_cases handles new format
    cases, metadata = load_cases(test_set_path)
    
    assert isinstance(cases, list), "Cases should be a list"
    assert len(cases) == 10, f"Should have 10 cases, got {len(cases)}"
    assert metadata is not None, "New format should have metadata"
    assert metadata["seed"] == 42, "Seed should be 42"
    assert metadata["sampled_cases"] == 10, "Should record 10 sampled cases"
    
    print(f"  ✓ New format loaded correctly")
    print(f"  ✓ Metadata: seed={metadata['seed']}, cases={metadata['sampled_cases']}")
    
    return True


def test_reproducibility():
    """Test that same seed produces same results."""
    print("\nTesting reproducibility...")
    
    import tempfile
    import os
    
    # Create two test sets with same seed
    with tempfile.TemporaryDirectory() as tmpdir:
        output1 = os.path.join(tmpdir, "test1.json")
        output2 = os.path.join(tmpdir, "test2.json")
        
        prep_test_cases(
            "data/ddxplus_v0/cases.json",
            output1,
            num_cases=5,
            seed=123,
        )
        
        prep_test_cases(
            "data/ddxplus_v0/cases.json",
            output2,
            num_cases=5,
            seed=123,
        )
        
        # Load both
        cases1, _ = load_cases(output1)
        cases2, _ = load_cases(output2)
        
        # Should be identical
        assert len(cases1) == len(cases2), "Same number of cases"
        
        case_ids1 = [c["case_id"] for c in cases1]
        case_ids2 = [c["case_id"] for c in cases2]
        
        assert case_ids1 == case_ids2, "Same cases in same order"
        
        print(f"  ✓ Same seed produces identical results")
        print(f"  ✓ Cases: {case_ids1[:3]}...")
    
    return True


def test_different_seeds():
    """Test that different seeds produce different results."""
    print("\nTesting different seeds...")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output1 = os.path.join(tmpdir, "seed1.json")
        output2 = os.path.join(tmpdir, "seed2.json")
        
        prep_test_cases(
            "data/ddxplus_v0/cases.json",
            output1,
            num_cases=10,
            seed=111,
        )
        
        prep_test_cases(
            "data/ddxplus_v0/cases.json",
            output2,
            num_cases=10,
            seed=222,
        )
        
        cases1, _ = load_cases(output1)
        cases2, _ = load_cases(output2)
        
        case_ids1 = [c["case_id"] for c in cases1]
        case_ids2 = [c["case_id"] for c in cases2]
        
        # Should be different (very unlikely to be same with different seeds)
        assert case_ids1 != case_ids2, "Different seeds should produce different samples"
        
        print(f"  ✓ Different seeds produce different results")
    
    return True


def main():
    print("=" * 60)
    print("Testing prep_test_cases workflow")
    print("=" * 60)
    
    tests = [
        test_backward_compatibility,
        test_new_format,
        test_reproducibility,
        test_different_seeds,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            results.append((test.__name__, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

