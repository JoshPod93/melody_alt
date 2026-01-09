"""
Randomizer module for BCI-UPIC testing and validation.

Provides systematic testing of the BCI system with:
- Random target sequences for SSVEP validation
- Randomized trials for classifier testing
- Statistical analysis of classification accuracy
"""

from __future__ import annotations

import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from .classifier import AttentionTarget, ClassificationResult, SSVEPClassifier
from .preprocessing import EEGPreprocessor, SimulatedEEGSource
from .stimulus import SSVEPStimulus


class TrialType(Enum):
    """Type of validation trial."""
    ATTEND_UP = "attend_up"      # Attend to 15Hz (top)
    ATTEND_DOWN = "attend_down"  # Attend to 10Hz (bottom)
    REST = "rest"                # No specific attention


@dataclass
class Trial:
    """A single validation trial."""
    trial_id: int
    trial_type: TrialType
    duration: float
    target_frequency: Optional[float]
    
    # Results (filled after trial)
    classifications: List[ClassificationResult] = field(default_factory=list)
    accuracy: float = 0.0
    mean_confidence: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate accuracy and confidence metrics."""
        if not self.classifications:
            return
        
        # Determine expected target
        if self.trial_type == TrialType.ATTEND_UP:
            expected = AttentionTarget.UP
        elif self.trial_type == TrialType.ATTEND_DOWN:
            expected = AttentionTarget.DOWN
        else:
            expected = AttentionTarget.NONE
        
        # Calculate accuracy
        correct = sum(1 for c in self.classifications if c.target == expected)
        self.accuracy = correct / len(self.classifications)
        
        # Calculate mean confidence
        self.mean_confidence = np.mean([c.confidence for c in self.classifications])
    
    def to_dict(self) -> dict:
        """Serialize trial to dictionary."""
        return {
            'trial_id': self.trial_id,
            'trial_type': self.trial_type.value,
            'duration': self.duration,
            'target_frequency': self.target_frequency,
            'accuracy': self.accuracy,
            'mean_confidence': self.mean_confidence,
            'num_classifications': len(self.classifications)
        }


@dataclass
class ValidationSession:
    """
    A complete validation session with multiple trials.
    
    Used to test and validate the SSVEP classification system.
    
    Attributes:
        num_trials_per_type: Number of trials for each type
        trial_duration: Duration of each trial in seconds
        rest_duration: Rest period between trials
        randomize: Whether to randomize trial order
    """
    num_trials_per_type: int = 5
    trial_duration: float = 5.0
    rest_duration: float = 2.0
    randomize: bool = True
    
    # Session data
    trials: List[Trial] = field(default_factory=list)
    _current_trial_idx: int = field(default=0, repr=False)
    _session_start: datetime = field(default=None, repr=False)
    _session_end: datetime = field(default=None, repr=False)
    
    # Components
    _classifier: SSVEPClassifier = field(default=None, repr=False)
    _preprocessor: EEGPreprocessor = field(default=None, repr=False)
    _eeg_source: SimulatedEEGSource = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize session."""
        self._generate_trials()
        self._classifier = SSVEPClassifier()
        self._preprocessor = EEGPreprocessor()
        self._eeg_source = SimulatedEEGSource()
    
    def _generate_trials(self) -> None:
        """Generate the trial sequence."""
        self.trials = []
        trial_id = 0
        
        # Create trials for each type
        trial_types = [
            (TrialType.ATTEND_UP, 15.0),
            (TrialType.ATTEND_DOWN, 10.0),
            (TrialType.REST, None)
        ]
        
        for trial_type, freq in trial_types:
            for _ in range(self.num_trials_per_type):
                self.trials.append(Trial(
                    trial_id=trial_id,
                    trial_type=trial_type,
                    duration=self.trial_duration,
                    target_frequency=freq
                ))
                trial_id += 1
        
        # Randomize if requested
        if self.randomize:
            np.random.shuffle(self.trials)
    
    @property
    def total_trials(self) -> int:
        """Total number of trials."""
        return len(self.trials)
    
    @property
    def current_trial(self) -> Optional[Trial]:
        """Get current trial."""
        if 0 <= self._current_trial_idx < len(self.trials):
            return self.trials[self._current_trial_idx]
        return None
    
    @property
    def progress(self) -> float:
        """Session progress (0-1)."""
        return self._current_trial_idx / len(self.trials)
    
    def run_trial(self, trial: Trial) -> None:
        """
        Run a single trial with simulated EEG.
        
        Args:
            trial: Trial to run
        """
        # Set simulated attention
        self._eeg_source.set_target(trial.target_frequency)
        
        # Reset classifier
        self._classifier.reset()
        self._preprocessor.reset()
        
        # Run trial
        start_time = time.perf_counter()
        classification_interval = 0.25  # Classify every 250ms
        last_classification = 0
        
        while time.perf_counter() - start_time < trial.duration:
            # Generate and process EEG
            eeg_chunk = self._eeg_source.generate_chunk(16)
            self._preprocessor.process_chunk(eeg_chunk)
            
            # Classify periodically
            elapsed = time.perf_counter() - start_time
            if elapsed - last_classification >= classification_interval:
                eeg_buffer = self._preprocessor.get_recent_data(1.0)
                result = self._classifier.classify(eeg_buffer, method="fft")
                trial.classifications.append(result)
                last_classification = elapsed
            
            time.sleep(0.01)  # 10ms loop
        
        # Calculate metrics
        trial.calculate_metrics()
    
    def run_session(self, callback=None) -> Dict[str, Any]:
        """
        Run the complete validation session.
        
        Args:
            callback: Optional callback(trial_idx, trial, is_rest) for progress updates
            
        Returns:
            Session results dictionary
        """
        self._session_start = datetime.now()
        
        for idx, trial in enumerate(self.trials):
            self._current_trial_idx = idx
            
            # Notify callback
            if callback:
                callback(idx, trial, False)
            
            # Run trial
            self.run_trial(trial)
            
            # Rest period
            if idx < len(self.trials) - 1:
                if callback:
                    callback(idx, trial, True)
                time.sleep(self.rest_duration)
        
        self._session_end = datetime.now()
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Calculate and return session results.
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Group by trial type
        results_by_type = {
            TrialType.ATTEND_UP: [],
            TrialType.ATTEND_DOWN: [],
            TrialType.REST: []
        }
        
        for trial in self.trials:
            results_by_type[trial.trial_type].append(trial)
        
        # Calculate metrics per type
        metrics = {}
        for trial_type, trials in results_by_type.items():
            if trials:
                accuracies = [t.accuracy for t in trials]
                confidences = [t.mean_confidence for t in trials]
                metrics[trial_type.value] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_confidence': np.mean(confidences),
                    'num_trials': len(trials)
                }
        
        # Overall accuracy
        all_accuracies = [t.accuracy for t in self.trials]
        
        return {
            'overall_accuracy': np.mean(all_accuracies),
            'overall_std': np.std(all_accuracies),
            'by_type': metrics,
            'num_trials': len(self.trials),
            'duration': (self._session_end - self._session_start).total_seconds() if self._session_end else 0,
            'trials': [t.to_dict() for t in self.trials]
        }
    
    def save_results(self, filepath: Path | str) -> None:
        """Save session results to JSON file."""
        filepath = Path(filepath)
        results = self.get_results()
        results['session_start'] = self._session_start.isoformat() if self._session_start else None
        results['session_end'] = self._session_end.isoformat() if self._session_end else None
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


@dataclass
class QuickTest:
    """
    Quick test for immediate validation of classifier.
    
    Runs a short test to verify the system is working.
    """
    duration: float = 3.0
    
    def run(self) -> Dict[str, Any]:
        """
        Run quick validation test.
        
        Returns:
            Test results
        """
        classifier = SSVEPClassifier()
        preprocessor = EEGPreprocessor()
        eeg_source = SimulatedEEGSource()
        
        results = {}
        
        for target_name, freq in [("15Hz (UP)", 15.0), ("10Hz (DOWN)", 10.0), ("REST", None)]:
            eeg_source.set_target(freq)
            preprocessor.reset()
            classifier.reset()
            
            # Generate data
            start = time.perf_counter()
            while time.perf_counter() - start < self.duration:
                eeg_chunk = eeg_source.generate_chunk(16)
                preprocessor.process_chunk(eeg_chunk)
                time.sleep(0.01)
            
            # Classify
            eeg_buffer = preprocessor.get_recent_data(1.0)
            result = classifier.classify(eeg_buffer, method="fft")
            
            results[target_name] = {
                'detected': result.target.name,
                'confidence': result.confidence,
                'power_15hz': result.power_15hz,
                'power_10hz': result.power_10hz
            }
        
        return results


def run_validation_cli():
    """Run validation from command line."""
    print("=" * 60)
    print("BCI-UPIC Validation System")
    print("=" * 60)
    
    # Quick test first
    print("\nRunning quick test...")
    quick = QuickTest(duration=2.0)
    quick_results = quick.run()
    
    print("\nQuick Test Results:")
    for target, result in quick_results.items():
        print(f"  {target}: Detected={result['detected']}, "
              f"Confidence={result['confidence']:.2f}")
    
    # Full validation
    print("\n" + "-" * 60)
    print("Running full validation session...")
    print("-" * 60)
    
    session = ValidationSession(
        num_trials_per_type=3,
        trial_duration=3.0,
        rest_duration=1.0
    )
    
    def progress_callback(idx, trial, is_rest):
        if is_rest:
            print(f"  Rest period...")
        else:
            print(f"  Trial {idx + 1}/{session.total_trials}: {trial.trial_type.value}")
    
    results = session.run_session(callback=progress_callback)
    
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']*100:.1f}% "
          f"(±{results['overall_std']*100:.1f}%)")
    
    print("\nBy Target Type:")
    for type_name, metrics in results['by_type'].items():
        print(f"  {type_name}: {metrics['mean_accuracy']*100:.1f}% "
              f"(confidence: {metrics['mean_confidence']:.2f})")
    
    # Save results
    output_path = Path("validation_results.json")
    session.save_results(output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_validation_cli()
