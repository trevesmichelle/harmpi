import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Test suite for animation methods in magnetized_analysis.py
Tests the three new physics-focused animations with minimal dump files
"""

import os
import sys
from magnetized_analysis import MagnetizedAnalysis, get_dump_files

def test_animation_suite(full_test=False):
    """Test all three animation methods"""
    
    # Test configuration
    test_dir = "./animation_tests"
    os.makedirs(test_dir, exist_ok=True)
    
    # Get dump files
    all_dumps = get_dump_files()
    if not all_dumps:
        print("❌ ERROR: No dump files found!")
        return False
    
    # Use subset for quick test, or all for full test
    if full_test:
        test_dumps = all_dumps
        print(f"FULL TEST MODE: Using all {len(all_dumps)} dump files")
    else:
        test_dumps = all_dumps[:10]  # First 10 for quick test
        print(f"QUICK TEST MODE: Using first 10 dump files")
    
    print("=" * 70)
    print("ANIMATION SUITE TEST")
    print("=" * 70)
    print(f"Testing with {len(test_dumps)} dump files")
    print(f"Dump range: {test_dumps[0]} to {test_dumps[-1]}")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = MagnetizedAnalysis(output_dir=test_dir)
    
    # Test definitions - UPDATED for new method names
    tests = [
        {
            'name': 'Velocity & Stagnation',
            'desc': 'Tests velocity magnitude + flow direction panels',
            'method': analyzer.create_velocity_and_stagnation_animation,
            'output': os.path.join(test_dir, 'test_velocity_stagnation.mp4'),
            'params': {
                'fps': 10,
                'sample_every': 1 if not full_test else 3
            }
        },
        {
            'name': 'Energy Zones',
            'desc': 'Tests energy extraction/dissipation zones',
            'method': analyzer.create_energy_zones_animation,
            'output': os.path.join(test_dir, 'test_energy_zones.mp4'),
            'params': {
                'fps': 10,
                'sample_every': 1 if not full_test else 3
            }
        },
        {
            'name': 'Magnetic Topology',
            'desc': 'Tests B_r(θ) profile + hemisphere flux evolution',
            'method': analyzer.create_magnetic_topology_animation,
            'output': os.path.join(test_dir, 'test_magnetic_topology.mp4'),
            'params': {
                'fps': 25 if full_test else 10,
                'sample_every': 1
            }
        }
    ]
    
    # Run tests
    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(tests)}: {test['name']}")
        print(f"Description: {test['desc']}")
        print(f"Output: {test['output']}")
        print("=" * 70)
        
        try:
            # Run animation method
            test['method'](test_dumps,
                          output_file=os.path.basename(test['output']),
                          **test['params'])
            
            # Check if file was created
            if os.path.exists(test['output']):
                file_size = os.path.getsize(test['output']) / (1024 * 1024)  # MB
                print(f"✓ SUCCESS: {test['name']}")
                print(f"  File size: {file_size:.2f} MB")
                results.append(('PASS', test['name'], file_size))
            else:
                print(f"✗ FAILED: {test['name']} - File not created")
                results.append(('FAIL', test['name'], f"File not created"))
                
        except Exception as e:
            print(f"✗ FAILED: {test['name']}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append(('FAIL', test['name'], str(e)))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r[0] == 'PASS')
    failed = sum(1 for r in results if r[0] == 'FAIL')
    
    for status, name, info in results:
        if status == 'PASS':
            print(f"✓ {name}: {status} ({info:.2f} MB)")
        else:
            print(f"✗ {name}: {status} - {info}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    if failed > 0:
        print("⚠️  {} test(s) failed. Check errors above.".format(failed))
        return False
    else:
        print("✅ All tests passed!")
        return True


if __name__ == "__main__":
    # Check command line arguments
    full_test = '--full' in sys.argv
    
    success = test_animation_suite(full_test=full_test)
    sys.exit(0 if success else 1)