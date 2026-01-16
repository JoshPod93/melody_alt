#!/usr/bin/env python
"""Quick script to review motor imagery classification results."""
import json
import numpy as np
from pathlib import Path

session_dir = Path("motor_imagery_sessions/20260116_153022")

# Load classifications
with open(session_dir / "classifications.json", 'r') as f:
    classifications = json.load(f)

# Extract data
targets = [c['target'] for c in classifications]
confidences = [c['confidence'] for c in classifications]
left_scores = [c['left_score'] for c in classifications]
right_scores = [c['right_score'] for c in classifications]
raw_scores = [c['raw_score'] for c in classifications]
times = [c['time'] for c in classifications]

print("=" * 80)
print("MOTOR IMAGERY CLASSIFICATION RESULTS REVIEW")
print("=" * 80)
print()

# Classification distribution
print("CLASSIFICATION DISTRIBUTION:")
print("-" * 80)
print(f"  Total classifications: {len(classifications)}")
print(f"  UP (left hand): {targets.count('UP')} ({100*targets.count('UP')/len(targets):.1f}%)")
print(f"  DOWN (right hand): {targets.count('DOWN')} ({100*targets.count('DOWN')/len(targets):.1f}%)")
print(f"  NONE: {targets.count('NONE')} ({100*targets.count('NONE')/len(targets):.1f}%)")
print()

# Confidence statistics
print("CONFIDENCE STATISTICS:")
print("-" * 80)
print(f"  Mean: {np.mean(confidences):.3f}")
print(f"  Std:  {np.std(confidences):.3f}")
print(f"  Min:  {np.min(confidences):.3f}")
print(f"  Max:  {np.max(confidences):.3f}")
print(f"  Median: {np.median(confidences):.3f}")
print()

# Score statistics
print("SCORE STATISTICS:")
print("-" * 80)
print(f"  Left score:  mean={np.mean(left_scores):.3f}, std={np.std(left_scores):.3f}")
print(f"  Right score: mean={np.mean(right_scores):.3f}, std={np.std(right_scores):.3f}")
print(f"  Score difference (left-right): mean={np.mean(np.array(left_scores) - np.array(right_scores)):.3f}")
print()

# Raw score analysis
print("RAW SCORE ANALYSIS:")
print("-" * 80)
print(f"  Mean: {np.mean(raw_scores):.6f}")
print(f"  Std:  {np.std(raw_scores):.6f}")
print(f"  Min:  {np.min(raw_scores):.6f}")
print(f"  Max:  {np.max(raw_scores):.6f}")
print(f"  Negative (UP/left): {sum(1 for s in raw_scores if s < 0)} ({100*sum(1 for s in raw_scores if s < 0)/len(raw_scores):.1f}%)")
print(f"  Positive (DOWN/right): {sum(1 for s in raw_scores if s > 0)} ({100*sum(1 for s in raw_scores if s > 0)/len(raw_scores):.1f}%)")
print()

# Temporal analysis
print("TEMPORAL ANALYSIS:")
print("-" * 80)
duration = max(times) - min(times)
print(f"  Session duration: {duration:.2f} seconds")
print(f"  Classification rate: {len(classifications)/duration:.2f} Hz")
print(f"  Time between classifications: {duration/len(classifications):.3f} seconds")
print()

# Check for issues
print("POTENTIAL ISSUES:")
print("-" * 80)
issues = []

# Check for class imbalance
if targets.count('UP') / len(targets) > 0.8 or targets.count('DOWN') / len(targets) > 0.8:
    issues.append(f"[WARNING] Severe class imbalance: {100*targets.count('UP')/len(targets):.1f}% UP vs {100*targets.count('DOWN')/len(targets):.1f}% DOWN")

# Check for low confidence
low_conf_count = sum(1 for c in confidences if c < 0.5)
if low_conf_count > len(confidences) * 0.3:
    issues.append(f"[WARNING] Many low-confidence predictions: {low_conf_count} ({100*low_conf_count/len(confidences):.1f}%) below 0.5")

# Check for very small raw scores (weak signal)
small_raw_count = sum(1 for s in raw_scores if abs(s) < 0.01)
if small_raw_count > len(raw_scores) * 0.5:
    issues.append(f"[WARNING] Weak signal: {small_raw_count} ({100*small_raw_count/len(raw_scores):.1f}%) raw scores < 0.01")

# Check for no NONE predictions (might indicate threshold too low)
if targets.count('NONE') == 0:
    issues.append("[WARNING] No NONE predictions - threshold might be too low, allowing all predictions")

if not issues:
    print("  [OK] No major issues detected")
else:
    for issue in issues:
        print(f"  {issue}")

print()
print("=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print()
print("1. CLASS IMBALANCE:")
print("   The 74.4% UP vs 25.6% DOWN suggests either:")
print("   - User was primarily imagining left hand movements")
print("   - System bias toward UP predictions (needs investigation)")
print("   - Weak right-hand imagery signal")
print()
print("2. CONFIDENCE:")
print(f"   Mean confidence of {np.mean(confidences):.3f} is {'good' if np.mean(confidences) > 0.6 else 'moderate'}")
print("   This indicates the classifier is making decisions, but may need calibration")
print()
print("3. RAW SCORES:")
print(f"   Mean raw score of {np.mean(raw_scores):.6f} suggests {'weak' if abs(np.mean(raw_scores)) < 0.01 else 'moderate'} signal strength")
print("   Small raw scores indicate subtle differences between left/right imagery")
print()
print("4. RECOMMENDATIONS:")
print("   - Consider implementing CSP calibration to improve discrimination")
print("   - Check if baseline normalization is working correctly")
print("   - Verify C3/C4 power differences are being captured")
print("   - Consider adjusting confidence threshold if needed")
print()
