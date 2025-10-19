#!/bin/bash
# ============================================================================
# CONSERVATIVE PROJECT CLEANUP
# Minimal reorganization that preserves functionality
# ============================================================================
#
# PHILOSOPHY: 
# - Move only what's safe
# - Keep dump folders at root (symlink strategy)
# - Keep harm_script.py at root (avoid import issues)
# - Only organize clutter (tests, backups, old results)
#
# Run from: ~/harmpi
# ============================================================================

set -e  # Exit on error

echo "==================================================================="
echo "HARMPI Project Cleanup - Conservative Approach"
echo "==================================================================="

# Safety check
if [ ! -f "makefile" ]; then
    echo "ERROR: Not in harmpi directory!"
    exit 1
fi

echo ""
echo "Step 1: Create minimal directory structure..."
mkdir -p tests
mkdir -p archive/{backups,old_results,old_tests}
mkdir -p results/{plots,animations}
mkdir -p docs

echo "✓ Directories created"

# ============================================================================
# Step 2: Move TEST SCRIPTS (safe - they're self-contained)
# ============================================================================
echo ""
echo "Step 2: Organizing test scripts..."

# Move current test scripts
if [ -f "test_animation_suite.py" ]; then
    mv test_animation_suite.py tests/
    echo "  ✓ Moved test_animation_suite.py"
fi

# Move old/obsolete test scripts to archive
for file in test_*.py session*_test*.py; do
    if [ -f "$file" ]; then
        mv "$file" archive/old_tests/
        echo "  ✓ Archived $file"
    fi
done

echo "✓ Test scripts organized"

# ============================================================================
# Step 3: Move BACKUP FILES (very safe - no dependencies)
# ============================================================================
echo ""
echo "Step 3: Archiving backup files..."

for file in *.backup*; do
    if [ -f "$file" ]; then
        mv "$file" archive/backups/
        echo "  ✓ Archived $file"
    fi
done

echo "✓ Backup files archived"

# ============================================================================
# Step 4: Move OLD RESULT FOLDERS (safe - just outputs)
# ============================================================================
echo ""
echo "Step 4: Archiving old result folders..."

OLD_RESULT_DIRS=(
    "session2_test_plots"
    "session3_test_plots"
    "session3_complete_test"
    "dipole_full_analysis"
    "dipole_hair_loss"
    "stagnation_test"
    "test_animations"
    "test_energy_flux"
)

for dir in "${OLD_RESULT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        mv "$dir" archive/old_results/
        echo "  ✓ Archived $dir/"
    fi
done

echo "✓ Old results archived"

# ============================================================================
# Step 5: Move CURRENT RESULT FOLDERS (safe)
# ============================================================================
echo ""
echo "Step 5: Organizing current results..."

if [ -d "magnetized_plots" ]; then
    mv magnetized_plots results/plots/
    echo "  ✓ Moved magnetized_plots/ → results/plots/"
fi

if [ -d "animation_tests" ]; then
    mv animation_tests results/animations/test_output
    echo "  ✓ Moved animation_tests/ → results/animations/test_output"
fi

echo "✓ Current results organized"

# ============================================================================
# Step 6: Keep ANALYSIS SCRIPTS at ROOT (avoids import issues)
# ============================================================================
echo ""
echo "Step 6: Analysis scripts status..."
echo "  → magnetized_analysis.py (KEEPING at root)"
echo "  → bondi_analysis.py (KEEPING at root)"
echo "  → monopole_2d_extended_analysis.py (KEEPING at root)"
echo "  → harm_script.py (KEEPING at root)"
echo ""
echo "  REASON: Avoiding import path issues with harm_script.py"
echo "  These can be moved later with proper import fixes"

# ============================================================================
# Step 7: Keep DUMP FOLDERS at ROOT (they're data, accessed by symlink)
# ============================================================================
echo ""
echo "Step 7: Dump folders status..."
echo "  → dumps/ (KEEPING at root)"
echo "  → dumps_monopole_2d/ (KEEPING at root)"
echo "  → dumps_dipole_bz/ (KEEPING at root)"
echo "  → dumps_session2_backup/ (KEEPING at root)"
echo ""
echo "  REASON: Analysis scripts expect dumps at root"
echo "  Use symlinks to switch: ln -sf dumps_monopole_2d dumps"

# ============================================================================
# Step 8: Move DOCUMENTATION (safe)
# ============================================================================
echo ""
echo "Step 8: Organizing documentation..."

DOC_FILES=("README.md" "tutorial.md" "exercises.md" "*.md")
for pattern in "${DOC_FILES[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ] && [ "$file" != "README.md" ]; then
            cp "$file" docs/  # Copy, don't move (keep README at root)
            echo "  ✓ Copied $file → docs/"
        fi
    done
done

echo "✓ Documentation organized"

# ============================================================================
# Step 9: Update .gitignore (ADDITIVE - keeps existing rules)
# ============================================================================
echo ""
echo "Step 9: Updating .gitignore..."

cat >> .gitignore << 'EOF'

# ====== Added by cleanup script ======

# Python
__pycache__/
*.pyc
*.pyo
venv/

# Simulation data
dumps/dump[0-9]*
gdump
dumps_*/

# Results (generated files)
results/
*.mp4
*.png
*.jpg

# Archives
archive/

# Compiled binaries
harm
gr
image_interp
*.o

# Logs
*.log
nohup.out

# Backups
*.backup
*.backup.*

# XCode (if not needed)
*.xcodeproj/
*.xcworkspace/
*.dSYM/

EOF

echo "✓ .gitignore updated (your *.o rule preserved)"

# ============================================================================
# Step 10: Create helper symlinks
# ============================================================================
echo ""
echo "Step 10: Creating helper symlinks..."

# Symlink to switch between dump folders easily
if [ ! -L "dumps" ]; then
    echo "  Note: 'dumps' is a real directory"
    echo "  To use different data: ln -sf dumps_monopole_2d dumps"
fi

echo "✓ Setup complete"

# ============================================================================
# Step 11: Update test script imports
# ============================================================================
echo ""
echo "Step 11: Fixing test script imports..."

if [ -f "tests/test_animation_suite.py" ]; then
    # Add parent directory to path
    sed -i '1i import sys\nimport os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n' tests/test_animation_suite.py
    echo "  ✓ Updated tests/test_animation_suite.py imports"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "==================================================================="
echo "CLEANUP COMPLETE!"
echo "==================================================================="
echo ""
echo "NEW STRUCTURE:"
echo "  tests/                    ← Test scripts"
echo "  archive/                  ← Old stuff (safe to delete later)"
echo "  results/                  ← Generated outputs (gitignored)"
echo "  docs/                     ← Documentation"
echo ""
echo "UNCHANGED (at root):"
echo "  *.py analysis scripts     ← Avoids import issues"
echo "  dumps*/                   ← Data folders"
echo "  *.c, *.h, makefile        ← Source code"
echo ""
echo "NEXT STEPS:"
echo "  1. Test that everything still works:"
echo "     python tests/test_animation_suite.py --full"
echo ""
echo "  2. If tests pass, commit:"
echo "     git add ."
echo "     git commit -m 'Refactor: Clean up project structure'"
echo ""
echo "  3. Later (optional): Move analysis scripts to analysis/"
echo "     (requires import path fixes in each script)"
echo ""
echo "  4. When confident, delete archive/:"
echo "     rm -rf archive/"
echo ""
echo "==================================================================="