"""Check performance metrics from last composition."""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.bci.score import BCIScore
    
    # Look for saved scores
    score_files = list(Path(".").glob("*.json"))
    score_files = [f for f in score_files if f.name not in ["calibration_data.json", "screen_calibration.json"]]
    
    if score_files:
        # Get most recent
        latest = max(score_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading score from: {latest}")
        
        with open(latest, 'r') as f:
            data = json.load(f)
        
        if 'metadata' in data and 'performance_metrics' in data['metadata']:
            metrics = data['metadata']['performance_metrics']
            
            print("\n" + "=" * 60)
            print("PERFORMANCE METRICS - From Saved Score")
            print("=" * 60)
            
            if 'top_flicker' in metrics:
                tf = metrics['top_flicker']
                print(f"\nTop Flicker Rate:")
                print(f"  Target: {tf['target_hz']:.3f} Hz")
                print(f"  Actual: {tf['actual_hz']:.3f} Hz")
                print(f"  Error:  {tf['error_hz']:.3f} Hz ({tf['error_pct']:.1f}%)")
            
            if 'bottom_flicker' in metrics:
                bf = metrics['bottom_flicker']
                print(f"\nBottom Flicker Rate:")
                print(f"  Target: {bf['target_hz']:.3f} Hz")
                print(f"  Actual: {bf['actual_hz']:.3f} Hz")
                print(f"  Error:  {bf['error_hz']:.3f} Hz ({bf['error_pct']:.1f}%)")
            
            if 'classification' in metrics:
                cf = metrics['classification']
                print(f"\nClassification:")
                print(f"  Mean confidence: {cf['mean_confidence']:.3f}")
                print(f"  Classifications: {cf['n_classifications']}")
            
            print("=" * 60)
        else:
            print("No performance metrics found in saved score.")
    else:
        print("No saved scores found.")
        print("\nThe performance report should have printed to the console when")
        print("the composition completed. Check the console output for:")
        print("  [FLICKER RATE] messages during composition")
        print("  PERFORMANCE REPORT at the end")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
