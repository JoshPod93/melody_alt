#!/usr/bin/env python
"""Quick script to review motor imagery classification results."""
import json
import numpy as np
import argparse
from pathlib import Path


def find_most_recent_session() -> Path:
    """Find the most recent motor imagery session with classifications."""
    base_dir = Path("motor_imagery_sessions")
    if not base_dir.exists():
        return None
    
    sessions = []
    for session_dir in sorted(base_dir.iterdir(), reverse=True):
        if session_dir.is_dir():
            classifications_file = session_dir / "classifications.json"
            if classifications_file.exists():
                sessions.append(session_dir)
    
    return sessions[0] if sessions else None


def main():
    parser = argparse.ArgumentParser(
        description="Review motor imagery classification results"
    )
    parser.add_argument(
        'session_dir',
        type=str,
        nargs='?',
        default=None,
        help='Path to session directory (e.g., motor_imagery_sessions/20260116_153022). If not provided, uses most recent session.'
    )
    
    args = parser.parse_args()
    
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = find_most_recent_session()
        if session_dir is None:
            print("ERROR: No session directory provided and no recent sessions found.")
            print("Usage: python review_results.py [session_dir]")
            return
        print(f"Using most recent session: {session_dir}")
    
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        return
    
    classifications_file = session_dir / "classifications.json"
    if not classifications_file.exists():
        print(f"ERROR: classifications.json not found in {session_dir}")
        return

    # Load classifications
    with open(classifications_file, 'r', encoding='utf-8') as f:
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
    print(f"Session: {session_dir.name}")
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
    
    up_pct = 100 * targets.count('UP') / len(targets)
    down_pct = 100 * targets.count('DOWN') / len(targets)
    
    print("1. CLASS IMBALANCE:")
    if up_pct > 70 or down_pct > 70:
        print(f"   The {up_pct:.1f}% UP vs {down_pct:.1f}% DOWN suggests either:")
        print("   - User was primarily imagining one hand movement")
        print("   - System bias toward one direction (needs investigation)")
        print("   - Weak signal for one direction")
    else:
        print(f"   Balanced distribution: {up_pct:.1f}% UP vs {down_pct:.1f}% DOWN")
    print()
    
    print("2. CONFIDENCE:")
    mean_conf = np.mean(confidences)
    print(f"   Mean confidence of {mean_conf:.3f} is {'good' if mean_conf > 0.6 else 'moderate' if mean_conf > 0.4 else 'low'}")
    print("   This indicates the classifier is making decisions, but may need calibration")
    print()
    
    print("3. RAW SCORES:")
    mean_raw = np.mean(raw_scores)
    print(f"   Mean raw score of {mean_raw:.6f} suggests {'weak' if abs(mean_raw) < 0.01 else 'moderate'} signal strength")
    print("   Small raw scores indicate subtle differences between left/right imagery")
    print()
    
    print("4. RECOMMENDATIONS:")
    print("   - Consider implementing CSP calibration to improve discrimination")
    print("   - Check if baseline normalization is working correctly")
    print("   - Verify C3/C4 power differences are being captured")
    print("   - Consider adjusting confidence threshold if needed")
    print()


if __name__ == "__main__":
    main()
