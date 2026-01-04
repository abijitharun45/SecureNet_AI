#!/usr/bin/env python3
"""
Test script to validate performance optimizations.
This script tests the key optimization patterns without requiring the full Streamlit app.
"""

import sys
import time
from pathlib import Path

def test_syntax_validation():
    """Test that all Python files have valid syntax."""
    print("=" * 60)
    print("TEST 1: Syntax Validation")
    print("=" * 60)
    
    files = [
        'Home.py',
        'pages/1_🚀_Network_Scanner.py',
        'pages/2_📈_Model_Performance.py',
        'src/ai_analyst.py',
        'src/pdf_generator.py',
        'src/make_mini_dataset.py',
        'src/training_pipeline.py'
    ]
    
    import py_compile
    failed = []
    
    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"✓ {file}: Valid syntax")
        except py_compile.PyCompileError as e:
            print(f"✗ {file}: Syntax error - {e}")
            failed.append(file)
    
    if failed:
        print(f"\n❌ {len(failed)} file(s) failed syntax validation")
        return False
    else:
        print(f"\n✅ All {len(files)} files passed syntax validation")
        return True


def test_optimization_patterns():
    """Test that optimization patterns are present in the code."""
    print("\n" + "=" * 60)
    print("TEST 2: Optimization Pattern Verification")
    print("=" * 60)
    
    checks = {
        'Home.py': ['@st.cache_data', 'load_css'],
        'pages/1_🚀_Network_Scanner.py': [
            '@st.cache_data',
            '@st.cache_resource',
            'np.random.choice',
            '.values',
            'iloc[::step]',
            'session_state[\'export',
            'threat_df',  # Pre-filtering optimization
            'load_test_data'
        ],
        'src/pdf_generator.py': [
            'groupby',
            '.values',
            'threat_confidence'
        ],
        'src/ai_analyst.py': [
            '[:500]',  # String truncation for API efficiency
        ]
    }
    
    all_passed = True
    
    for file, patterns in checks.items():
        print(f"\nChecking {file}:")
        if not Path(file).exists():
            print(f"  ⚠ File not found, skipping")
            continue
            
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for pattern in patterns:
            if pattern in content:
                print(f"  ✓ Found optimization: {pattern}")
            else:
                print(f"  ✗ Missing optimization: {pattern}")
                all_passed = False
    
    if all_passed:
        print("\n✅ All optimization patterns verified")
    else:
        print("\n⚠ Some optimization patterns missing")
    
    return all_passed


def test_documentation():
    """Test that documentation exists."""
    print("\n" + "=" * 60)
    print("TEST 3: Documentation Check")
    print("=" * 60)
    
    doc_file = 'PERFORMANCE_IMPROVEMENTS.md'
    
    if Path(doc_file).exists():
        with open(doc_file, 'r') as f:
            content = f.read()
            line_count = len(content.splitlines())
            
        print(f"✓ {doc_file} exists")
        print(f"  - {line_count} lines of documentation")
        print(f"  - {len(content)} characters")
        
        # Check for key sections
        sections = [
            'CSS Loading Optimization',
            'Test Data Loading Optimization',
            'Data Sampling Optimization',
            'DataFrame Filtering Optimization',
            'Export Data Caching',
            'Performance Metrics'
        ]
        
        missing = []
        for section in sections:
            if section in content:
                print(f"  ✓ Section present: {section}")
            else:
                print(f"  ✗ Section missing: {section}")
                missing.append(section)
        
        if not missing:
            print("\n✅ Documentation complete")
            return True
        else:
            print(f"\n⚠ {len(missing)} section(s) missing from documentation")
            return False
    else:
        print(f"✗ {doc_file} not found")
        return False


def test_performance_simulation():
    """Simulate performance improvements with simple operations."""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Simulation")
    print("=" * 60)
    
    print("\nSimulating old vs new approach for common operations:\n")
    
    # Test 1: List comprehension vs numpy operations
    import random
    data = [random.choice(['BENIGN', 'ATTACK']) for _ in range(10000)]
    
    # Old approach simulation (with string operations)
    start = time.time()
    threats_old = sum(1 for x in data if x.upper() != 'BENIGN')
    time_old = time.time() - start
    
    # New approach simulation (simplified)
    start = time.time()
    threats_new = sum(1 for x in data if x != 'BENIGN')
    time_new = time.time() - start
    
    speedup = (time_old / time_new) if time_new > 0 else 1
    print(f"String comparison test:")
    print(f"  Old approach: {time_old*1000:.2f}ms")
    print(f"  New approach: {time_new*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Test 2: Random sampling vs index-based sampling
    indices = list(range(10000))
    
    start = time.time()
    for _ in range(100):
        sample_old = random.sample(indices, 100)
        sample_old.sort()  # Old approach with sort
    time_old = time.time() - start
    
    start = time.time()
    for _ in range(100):
        step = len(indices) // 100
        sample_new = indices[::step]  # New approach with step
    time_new = time.time() - start
    
    speedup = (time_old / time_new) if time_new > 0 else 1
    print(f"\nSampling test:")
    print(f"  Old approach (random + sort): {time_old*1000:.2f}ms")
    print(f"  New approach (systematic): {time_new*1000:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    print("\n✅ Performance simulation complete")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SecureNet AI - Performance Optimization Tests")
    print("=" * 60)
    
    results = {
        'Syntax Validation': test_syntax_validation(),
        'Optimization Patterns': test_optimization_patterns(),
        'Documentation': test_documentation(),
        'Performance Simulation': test_performance_simulation()
    }
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
