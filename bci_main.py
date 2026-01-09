#!/usr/bin/env python
"""
BCI-UPIC - Brain-Computer Interface Controlled Music Synthesizer

A BCI-controlled version of the UPIC (Unité Polyagogique Informatique du CEMAMu)
system. Instead of mouse control, users control the synthesizer using SSVEP
(Steady-State Visual Evoked Potentials) from their brain signals.

Features:
- Two flickering SSVEP targets (15Hz top = UP, 10Hz bottom = DOWN)
- Automatic horizontal playhead movement over composition duration
- Real-time BCI-controlled vertical cursor movement
- Score generation and playback through additive synthesis

Run this file to start the BCI composition application.
"""

import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="BCI-UPIC: Brain-Computer Interface Music Synthesizer"
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['gui', 'validate', 'test', 'demo'],
        default='gui',
        help='Mode to run: gui (default), validate, test, or demo'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Composition duration in seconds (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for demo mode'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        # Run the full GUI application
        from src.bci.interface import run_bci_app
        run_bci_app()
    
    elif args.mode == 'validate':
        # Run validation tests
        from src.bci.randomizer import run_validation_cli
        run_validation_cli()
    
    elif args.mode == 'test':
        # Run quick test
        from src.bci.randomizer import QuickTest
        print("Running quick BCI system test...")
        test = QuickTest(duration=3.0)
        results = test.run()
        
        print("\nTest Results:")
        print("-" * 40)
        for target, result in results.items():
            status = "[OK]" if result['detected'] != 'NONE' else "[--]"
            print(f"{status} {target}:")
            print(f"    Detected: {result['detected']}")
            print(f"    Confidence: {result['confidence']:.2f}")
        
        # Check if system is working
        all_detected = all(r['confidence'] > 0.3 for r in results.values())
        print("\n" + "=" * 40)
        if all_detected:
            print("[OK] System is working correctly!")
        else:
            print("[!] Some targets may not be detected reliably.")
            print("    This is expected with simulated EEG data.")
    
    elif args.mode == 'demo':
        # Run a demo composition with random input
        from src.bci.controller import RandomController
        from src.bci.score import BCIScore, synthesize_score
        from pathlib import Path
        
        print(f"Running demo composition ({args.duration}s)...")
        
        # Create random controller
        controller = RandomController(duration=args.duration)
        controller.start()
        
        # Run until complete
        import time
        while controller.is_running:
            controller.update()
            time.sleep(0.016)
        
        # Create score
        trail = [p.to_tuple() for p in controller.trail]
        score = BCIScore(
            trail=trail,
            duration=args.duration,
            waveform_name="Sine",
            metadata={'demo': True}
        )
        
        # Stats
        stats = score.get_statistics()
        print(f"\nComposition Statistics:")
        print(f"  Points: {stats['num_points']}")
        print(f"  Pitch range: {stats['pitch_range']:.2f}")
        print(f"  Total movement: {stats['total_movement']:.2f}")
        
        # Output
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.wav':
                print(f"\nExporting to {output_path}...")
                synthesize_score(score, output_path)
                print("Done!")
            elif output_path.suffix == '.json':
                print(f"\nSaving score to {output_path}...")
                score.save(output_path)
                print("Done!")
            else:
                print(f"Unknown output format: {output_path.suffix}")
        else:
            # Play the score
            print("\nPlaying composition...")
            from src.bci.score import play_score
            play_score(score)


if __name__ == "__main__":
    main()
