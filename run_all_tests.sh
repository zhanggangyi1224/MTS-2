#!/bin/bash

# Automated test runner for MTS model
# Runs all tests and demonstrations automatically

echo "======================================================================"
echo "                  MTS Model - Automated Test Suite"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Check if training is complete"
echo "  2. Run 30-second generation tests"
echo "  3. Run long-form (3-4 min) generation tests"
echo "  4. Run full demonstration"
echo ""
echo "======================================================================"
echo ""

# Check if checkpoints exist
if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints/*.pt 2>/dev/null)" ]; then
    echo "❌ No model checkpoints found!"
    echo "   Please train the model first:"
    echo "   python3 train_mts_local.py"
    echo ""
    exit 1
fi

echo "✅ Found trained model checkpoints"
echo ""

# Check if training is still running
if pgrep -f "train_mts_local.py" > /dev/null; then
    echo "⚠️  Training is still running (PID: $(pgrep -f train_mts_local.py))"
    echo "   You can:"
    echo "   1. Wait for training to complete (recommended)"
    echo "   2. Stop training and use current checkpoint (Ctrl+C on training process)"
    echo "   3. Continue anyway with the current checkpoint"
    echo ""
    read -p "Continue with current checkpoint? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Run this script again after training completes."
        exit 0
    fi
fi

# Create output directory
mkdir -p test_results

# ============================================================================
# TEST 1: 30-Second Generation
# ============================================================================
echo "======================================================================"
echo "TEST 1: 30-Second Generation"
echo "======================================================================"
echo ""

python3 test_generation.py 2>&1 | tee test_results/test1_30s_generation.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 30-second generation test PASSED"
else
    echo ""
    echo "❌ 30-second generation test FAILED"
    echo "   Check log: test_results/test1_30s_generation.log"
fi

echo ""
echo "Press Enter to continue to next test..."
read

# ============================================================================
# TEST 2: Long-Form Generation
# ============================================================================
echo "======================================================================"
echo "TEST 2: Long-Form Generation (3-4 minutes)"
echo "======================================================================"
echo ""
echo "⚠️  This test will take several minutes..."
echo ""

python3 generate_long_form.py 2>&1 | tee test_results/test2_longform_generation.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Long-form generation test PASSED"
else
    echo ""
    echo "❌ Long-form generation test FAILED"
    echo "   Check log: test_results/test2_longform_generation.log"
fi

echo ""
echo "Press Enter to continue to full demo..."
read

# ============================================================================
# TEST 3: Full Demonstration
# ============================================================================
echo "======================================================================"
echo "TEST 3: Full MTS Demonstration"
echo "======================================================================"
echo ""
echo "⚠️  This will demonstrate all features (takes several minutes)..."
echo ""

python3 demo_full_mts.py 2>&1 | tee test_results/test3_full_demo.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Full demonstration PASSED"
else
    echo ""
    echo "❌ Full demonstration FAILED"
    echo "   Check log: test_results/test3_full_demo.log"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "======================================================================"
echo "                       TEST SUMMARY"
echo "======================================================================"
echo ""

if [ -d "generated_samples" ]; then
    echo "📁 Generated Samples:"
    ls -lh generated_samples/*.wav 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    echo ""
fi

if [ -d "demo_output" ]; then
    echo "📁 Demo Output:"
    ls -lh demo_output/*.wav 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    echo ""
fi

echo "📊 Test Logs:"
ls -lh test_results/*.log 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""

echo "======================================================================"
echo "✅ All tests complete!"
echo ""
echo "🎵 Play generated samples:"
echo "   afplay generated_samples/generated_01.wav"
echo "   afplay demo_output/demo4_longform_210s.wav"
echo "======================================================================"
