"""Surplus and Distinction Dynamics"""
"""
Manages surplus stabilization, adaptation, and recursive feedback in the quantum system.
Regulates cognitive and ontological surplus levels based on phase coherence and entropy.
"""
import numpy as np
import time
import traceback
from collections import deque, defaultdict
from typing import Dict, Tuple, Any, Optional, Union, List
import logging_setup
from logging_setup import setup_logging
from data_classes import SurplusState

# Import only necessary constants and functions, not the ones we're making dynamic
from utilities import (
    MAX_SURPLUS,
    MOMENTUM_DECAY,
    ensure_real,
    MINIMUM_COHERENCE_FLOOR,
    HIDDEN_DIM,
    update_momentum,
    compute_phase_coherence,
)
from analysis import CausalityAnalysis

# ---------------------------
# EnhancedSurplusDynamics
# ---------------------------
class EnhancedSurplusDynamics:
    """
    Enhanced surplus dynamics with improved quantum coupling and state management.
    Implements advanced surplus accumulation, adaptation, and expulsion mechanisms.
    """
    def __init__(self):
        # Initialize dynamic constants - moved from utilities.py to instance variables
        # These constants can now be adjusted dynamically during runtime
        self.surplus_threshold = 1.5  # Default from SURPLUS_THRESHOLD
        self.expulsion_recovery_rate = 0.02  # Default from EXPULSION_RECOVERY_RATE
        self.surplus_adjustment_rate = 0.05  # Default from SURPLUS_ADJUSTMENT_RATE
        self.phase_scaling_factor = 0.3  # Default from PHASE_SCALING_FACTOR
        self.surplus_recycle_fraction = 0.7  # Default from SURPLUS_RECYCLE_FRACTION
        self.core_distinction_update_rate = 0.01  # Default from CORE_DISTINCTION_UPDATE_RATE
        self.distinction_anchor_weight = 0.2  # Default from DISTINCTION_ANCHOR_WEIGHT
        self.target_distinction = 0.7  # Default from TARGET_DISTINCTION
        self.collapse_dissipation_threshold = 0.35  # Default from COLLAPSE_DISSIPATION_THRESHOLD
        self.collapse_dissipation_rate = 0.02  # Default from COLLAPSE_DISSIPATION_RATE
        self.instability_grace_period = 3  # Default from INSTABILITY_GRACE_PERIOD

        # Learning rates for dynamic constant adaptation
        self.constant_adaptation_rate = 0.01  # Base learning rate for updating constants
        self.threshold_momentum = 0.0  # Momentum for surplus_threshold updates
        self.distinction_momentum = 0.0  # Momentum for target_distinction updates
        self.recovery_momentum = 0.0  # Momentum for expulsion_recovery_rate updates

        # History tracking for dynamic constant adaptation
        self.constants_history = deque(maxlen=1000)  # Track changes to constants
        self.adaptation_feedback = deque(maxlen=100)  # Track feedback from adaptation

        # Initialize counters and tracking first
        self.stability_threshold = 0.3
        self.initialization_attempts = 0
        self.max_initialization_attempts = 3
        self.steps_since_expulsion = 0
        self.expulsion_cooldown = 50
        self.stability_decay = 0.99
        self.coupling_momentum = 0.0
        self.expulsion_threshold = 0.95  # Adjust as needed
        self.instability_counter = 0  # Track system instability
        self.min_steps_before_recovery = 10  # Minimum steps before resetting
        self.steps_taken = 0  # Track steps taken
        self.recovery_state = None
        self.minimum_stability = 0.1

        # Initialize surplus state first! (BEFORE calling any methods using it)
        self.surplus_state = self._initialize_surplus_state()

        # Ensure surplus_state is valid before proceeding
        if self.surplus_state is None or not isinstance(self.surplus_state, SurplusState):
            raise ValueError("❌ Critical Error: SurplusState initialization failed completely.")

        # Initialize excess stability tracking
        self.excess_stability_potential = 0.0

        # Initialize history tracking
        self.expulsion_history = deque(maxlen=100)
        self.accumulation_history = deque(maxlen=1000)
        self.stability_history = deque(maxlen=1000)
        self.initialization_history = deque(maxlen=100)

        # Initialize surplus tracking attributes
        self.basal_surplus = self.surplus_state.values['basal']
        self.cognitive_surplus = self.surplus_state.values['cognitive']
        self.predictive_surplus = self.surplus_state.values['predictive']
        self.ontological_surplus = self.surplus_state.values['ontological']
        self.stability = self.surplus_state.stability
        self.quantum_coupling = self.surplus_state.quantum_coupling
        self.accumulation_momentum = {k: 0.0 for k in self.surplus_state.values.keys()}
        self.stability_momentum = 0.0

        # Prevent instant expulsion; only trigger if surplus exceeds a defined threshold
        if self.surplus_state.total_surplus() > self.surplus_threshold:
            self.gradual_surplus_expulsion()

        # Emergence tracking
        self.emergence_patterns = deque(maxlen=1000)
        self.pattern_momentum = 0.0
        self.emergence_threshold = 0.3
        self.emergence_counter = 0
        self.emergence_history = []
        self.novelty_score = 0.0
        self.complexity_score = 0.0
        self.emergence_sensitivity = 0.5

        # Record initial constants
        self._record_constants_state("initialization")

    def _record_constants_state(self, event_type: str) -> None:
        """Record the current state of all dynamic constants for tracking changes over time."""
        try:
            constants_state = {
                'surplus_threshold': self.surplus_threshold,
                'expulsion_recovery_rate': self.expulsion_recovery_rate,
                'surplus_adjustment_rate': self.surplus_adjustment_rate,
                'phase_scaling_factor': self.phase_scaling_factor,
                'surplus_recycle_fraction': self.surplus_recycle_fraction,
                'core_distinction_update_rate': self.core_distinction_update_rate,
                'distinction_anchor_weight': self.distinction_anchor_weight,
                'target_distinction': self.target_distinction,
                'collapse_dissipation_threshold': self.collapse_dissipation_threshold,
                'collapse_dissipation_rate': self.collapse_dissipation_rate,
                'instability_grace_period': self.instability_grace_period,
                'event_type': event_type,
                'timestamp': time.time()
            }
            self.constants_history.append(constants_state)
        except Exception as e:
            print(f"Error recording constants state: {e}")

    def update_dynamic_constants(self,
                               phase_coherence: float,
                               distinction_level: float,
                               stability: float,
                               emergence_detected: bool = False) -> None:
        """
        Update dynamic constants based on system state and performance feedback.
        This implements a rule-based and momentum-based approach to constant adaptation.

        Args:
            phase_coherence: Current phase coherence
            distinction_level: Current distinction level
            stability: Current system stability
            emergence_detected: Whether emergence was detected in the current step
        """
        try:
            # Ensure inputs are valid floats
            phase_coherence = ensure_real(phase_coherence, 0.5)
            distinction_level = ensure_real(distinction_level, 0.5)
            stability = ensure_real(stability, 0.5)

            # Store previous values to track changes
            prev_threshold = self.surplus_threshold
            prev_distinction = self.target_distinction

            # 1. Update surplus_threshold based on average surplus levels
            avg_surplus = self.surplus_state.total_surplus() / len(self.surplus_state.values)

            # If average surplus is consistently high, increase the threshold
            if avg_surplus > self.surplus_threshold * 1.2 and stability > 0.6:
                # Calculate threshold adjustment
                threshold_adjustment = 0.05 * self.constant_adaptation_rate * (avg_surplus / self.surplus_threshold - 1.0)

                # Update momentum
                self.threshold_momentum = update_momentum(
                    self.threshold_momentum,
                    threshold_adjustment,
                    decay=MOMENTUM_DECAY
                )

                # Apply adjustment with momentum
                self.surplus_threshold += threshold_adjustment + 0.1 * self.threshold_momentum
                self.surplus_threshold = min(self.surplus_threshold, MAX_SURPLUS * 0.5)  # Cap at 50% of MAX_SURPLUS

                print(f"Increased surplus_threshold to {self.surplus_threshold:.4f} based on high surplus")

            # If system is unstable and surplus is low, decrease threshold
            elif self.instability_counter > 1 and avg_surplus < self.surplus_threshold * 0.8:
                # Calculate threshold adjustment
                threshold_adjustment = -0.05 * self.constant_adaptation_rate * (1.0 - avg_surplus / self.surplus_threshold)

                # Update momentum
                self.threshold_momentum = update_momentum(
                    self.threshold_momentum,
                    threshold_adjustment,
                    decay=MOMENTUM_DECAY
                )

                # Apply adjustment with momentum
                self.surplus_threshold += threshold_adjustment + 0.1 * self.threshold_momentum
                self.surplus_threshold = max(self.surplus_threshold, 0.5)  # Ensure threshold doesn't go too low

                print(f"Decreased surplus_threshold to {self.surplus_threshold:.4f} based on instability")

            # 2. Update target_distinction based on current distinction and stability
            # If the system is stable, move target distinction toward current distinction
            distinction_delta = distinction_level - self.target_distinction

            # Only update if there's a significant difference and system is stable
            if abs(distinction_delta) > 0.1 and stability > 0.7:
                # Calculate distinction adjustment
                distinction_adjustment = 0.02 * self.constant_adaptation_rate * distinction_delta

                # Update momentum
                self.distinction_momentum = update_momentum(
                    self.distinction_momentum,
                    distinction_adjustment,
                    decay=MOMENTUM_DECAY
                )

                # Apply adjustment with momentum
                self.target_distinction += distinction_adjustment + 0.1 * self.distinction_momentum
                self.target_distinction = np.clip(self.target_distinction, 0.3, 0.9)  # Keep in reasonable range

                print(f"Updated target_distinction to {self.target_distinction:.4f} based on current distinction")

            # 3. Update expulsion_recovery_rate based on system performance
            if emergence_detected:
                # Increase recovery rate when emergence is detected
                self.expulsion_recovery_rate = min(self.expulsion_recovery_rate * 1.1, 0.05)
                print(f"Increased expulsion_recovery_rate to {self.expulsion_recovery_rate:.4f} based on emergence")

            # Adjust instability_grace_period based on system stability
            if self.instability_counter > self.instability_grace_period - 1:
                # System is approaching instability threshold, increase grace period
                self.instability_grace_period = min(self.instability_grace_period + 1, 10)
                print(f"Increased instability_grace_period to {self.instability_grace_period}")
            elif stability > 0.8 and self.instability_counter == 0 and self.instability_grace_period > 3:
                # System is very stable, gradually decrease grace period
                self.instability_grace_period = max(self.instability_grace_period - 1, 3)
                print(f"Decreased instability_grace_period to {self.instability_grace_period}")

            # Record significant constant changes
            if (abs(self.surplus_threshold - prev_threshold) > 0.05 or
                abs(self.target_distinction - prev_distinction) > 0.05):
                self._record_constants_state("significant_update")

            # Record adaptation feedback
            self.adaptation_feedback.append({
                'stability': stability,
                'distinction_level': distinction_level,
                'surplus': avg_surplus,
                'emergence_detected': emergence_detected,
                'constants_adjusted': {
                    'surplus_threshold': self.surplus_threshold != prev_threshold,
                    'target_distinction': self.target_distinction != prev_distinction
                },
                'timestamp': time.time()
            })

        except Exception as e:
            print(f"Error updating dynamic constants: {e}")
            traceback.print_exc()

    def track_emergence(self, current_state: Dict[str, float]):
        """Track and analyze emergent patterns while maintaining stability."""
        try:
            # -- FIX: Ensure novelty has a default value, even if emergence_patterns is empty
            novelty = 0.0

            # Calculate novelty compared to recent history
            if self.emergence_patterns:
                recent_states = list(self.emergence_patterns)[-10:]
                novelty = np.mean([
                    self._compute_state_difference(current_state, past_state)
                    for past_state in recent_states
                ])
                self.novelty_score = novelty

            # Calculate complexity
            complexity = self._calculate_complexity(current_state)
            self.complexity_score = complexity

            # Track the pattern
            self.emergence_patterns.append(current_state)

            # Factor in excess stability potential to emergence detection
            emergence_threshold_modifier = 1.0
            # Initialize excess_stability_potential if not exists
            if not hasattr(self, 'excess_stability_potential'):
                self.excess_stability_potential = 0.0

            if self.excess_stability_potential > 0:
                # Decrease threshold (making emergence more likely) when excess stability exists
                emergence_threshold_modifier = max(0.7, 1.0 - (self.excess_stability_potential * 0.5))
                print(f"DEBUG: Excess stability {self.excess_stability_potential:.4f} modifying emergence threshold by factor {emergence_threshold_modifier:.4f}")

            # Apply threshold modifier
            effective_threshold = self.emergence_threshold * emergence_threshold_modifier

            # Update emergence counter if threshold exceeded
            if novelty > effective_threshold and complexity > effective_threshold:
                # Calculate additional increments based on excess stability
                extra_increment = 0
                if self.excess_stability_potential > 0:
                    extra_increment = int(self.excess_stability_potential * 3)

                # Increment counter with potential bonus
                base_increment = 1
                total_increment = base_increment + extra_increment

                # Apply increment
                self.emergence_counter += total_increment

                if extra_increment > 0:
                    print(f"DEBUG: Emergence counter increased by {total_increment} (including {extra_increment} from excess stability)")

                self.emergence_history.append({
                    'timestamp': time.time(),
                    'novelty': novelty,
                    'complexity': complexity,
                    'state': current_state.copy(),
                    'excess_stability': self.excess_stability_potential,
                    'emergence_bonus': extra_increment
                })

                # Gradually increase sensitivity to emergence
                self.emergence_sensitivity = min(1.0, self.emergence_sensitivity * 1.05)

                # Update dynamic constants when emergence is detected
                self.update_dynamic_constants(
                    phase_coherence=0.5,  # Default value if not available
                    distinction_level=0.5,  # Default value if not available
                    stability=self.surplus_state.stability,
                    emergence_detected=True
                )

            else:
                # Gradually decrease sensitivity if no emergence detected
                self.emergence_sensitivity = max(0.1, self.emergence_sensitivity * 0.95)

        except Exception as e:
            print(f"Error tracking emergence: {e}")


    def _calculate_complexity(self, state: Dict[str, float]) -> float:
        """Calculate complexity of current state with dynamic scaling."""
        try:
            # Calculate interaction terms between different surplus types
            values = np.array(list(state.values()))
            interactions = np.outer(values, values)

            # Calculate entropy of interactions
            flat_interactions = interactions.flatten()
            normalized = flat_interactions / (np.sum(flat_interactions) + 1e-10)
            entropy = -np.sum(normalized * np.log2(normalized + 1e-10))

            # Calculate rate of change if history exists
            if self.emergence_patterns:
                prev_state = self.emergence_patterns[-1]
                prev_values = np.array(list(prev_state.values()))
                rate_of_change = np.mean(np.abs(values - prev_values))
            else:
                rate_of_change = 0.0

            # Combine metrics with dynamic weighting
            complexity = (entropy * 0.6 + rate_of_change * 0.4)
            return float(np.clip(complexity, 0, 1))

        except Exception as e:
            print(f"Error calculating complexity: {e}")
            return 0.0

    def _compute_state_difference(self, state1: Dict[str, float], state2: Dict[str, float]) -> float:
        """Compute difference between two states with dynamic weighting."""
        try:
            keys = set(state1.keys()) & set(state2.keys())
            if not keys:
                return 1.0  # Maximum difference if no common keys

            differences = []
            for key in keys:
                val1 = float(state1[key])
                val2 = float(state2[key])
                # Calculate normalized difference
                max_val = max(abs(val1), abs(val2), 1.0)  # Avoid division by zero
                diff = abs(val1 - val2) / max_val
                differences.append(diff)

            return float(np.mean(differences))

        except Exception as e:
            print(f"Error computing state difference: {e}")
            return 0.0

    def _initialize_surplus_state(self) -> SurplusState:
        """Ensure surplus state is always correctly initialized."""
        try:
            print("🔄 Initializing new surplus state...")

            # Create new surplus state with different accumulation rates
            new_state = SurplusState()

            # Set different accumulation rates for different surplus types
            new_state.accumulation_rate = {
                'basal': 0.008,       # Slower rate
                'cognitive': 0.012,   # Standard rate
                'predictive': 0.015,  # Faster rate
                'ontological': 0.010  # Medium rate
            }

            # Validate new state
            if not new_state.validate():
                print("⚠️ Initial surplus state validation failed, retrying...")
                new_state = SurplusState()  # Attempt a second initialization

                if not new_state.validate():
                    print("❌ Second validation attempt failed. Reverting to default.")
                    new_state = SurplusState()  # Assign a default, even if imperfect

            return new_state

        except Exception as e:
            print(f"❌ Error initializing surplus state: {e}")
            traceback.print_exc()
            return SurplusState()  # Ensure a valid fallback state

    def validate_current_state(self) -> bool:
        """Validate surplus state and reset if needed."""
        try:
            if not isinstance(self.surplus_state, SurplusState):
                print("⚠️ Invalid surplus state type detected, reinitializing...")
                self.surplus_state = self._initialize_surplus_state()

            elif not self.surplus_state.validate():
                print("⚠️ Surplus state validation failed. Resetting...")
                self.surplus_state = self._initialize_surplus_state()

            return self.surplus_state.validate()

        except Exception as e:
            print(f"❌ Error validating surplus state: {e}")
            return False

    def check_system_stability(self):
        """Monitors stability, avoiding immediate recovery unless multiple instability conditions persist."""
        if not hasattr(self, 'min_steps_before_recovery'):
            self.min_steps_before_recovery = 10  # Minimum steps before resetting
            self.steps_taken = 0  # Track steps

        self.steps_taken += 1  # Increment step count

        if self.surplus_state.stability < self.stability_threshold:
            self.instability_counter += 1  # Track consecutive instability events

            # Prevent premature recovery if we haven't hit the minimum step count
            if self.steps_taken < self.min_steps_before_recovery:
                return  # Allow system to evolve before triggering recovery

            if self.instability_counter >= self.instability_grace_period:
                print("⚠️ System instability detected! Entering recovery mode...")
                self.enter_recovery_mode()
                self.instability_counter = 0  # Reset counter
                self.steps_taken = 0  # Reset step count
        else:
            self.instability_counter = max(0, self.instability_counter - 1)  # Reduce counter if stable

    def enter_recovery_mode(self):
        """Enter recovery mode to stabilize the system."""
        try:
            self.recovery_state = {
                'steps_remaining': 30,
                'initial_stability': self.surplus_state.stability
            }

            # Reduce surplus values to prevent cascading instability
            for key in self.surplus_state.values:
                self.surplus_state.values[key] *= 0.8

            # Reset momentum values
            self.accumulation_momentum = {k: 0.0 for k in self.surplus_state.values.keys()}
            self.stability_momentum = 0.0

            # Set a moderate stability to allow recovery
            self.surplus_state.stability = max(self.surplus_state.stability, 0.3)

            print("⚠️ Entering recovery mode for 30 steps")

        except Exception as e:
            print(f"❌ Error entering recovery mode: {e}")

    def reset_surplus_state(self) -> bool:
        """Reset surplus state to initial values."""
        self.surplus_state = self._initialize_surplus_state()
        return self.surplus_state is not None  # Ensures we always return a valid state

    def gradual_surplus_expulsion(self):
        """Reduces surplus values gradually instead of instant expulsion."""
        if not hasattr(self, 'surplus_state') or self.surplus_state is None:
            print("⚠️ Warning: Cannot perform surplus expulsion—surplus_state not initialized.")
            return

        DECAY_FACTOR = 0.1  # Adjust based on tuning

        try:
            for key in self.surplus_state.values:
                self.surplus_state.values[key] *= (1 - DECAY_FACTOR)  # Slow decrease
            print("⚠️ Gradual surplus expulsion applied.")
        except Exception as e:
            print(f"❌ Error in gradual_surplus_expulsion: {e}")

    def _create_initial_surplus_state(self) -> SurplusState:
        """Create a properly initialized surplus state with retry mechanism."""
        self.initialization_attempts = 0  # Reset counter

        while self.initialization_attempts < self.max_initialization_attempts:
            try:
                print(f"Attempting surplus state initialization (attempt {self.initialization_attempts + 1})")
                new_state = SurplusState()

                if new_state.validate():
                    # Record successful initialization
                    self.initialization_history.append({
                        'timestamp': time.time(),
                        'attempt': self.initialization_attempts,
                        'success': True
                    })
                    print("✅ Surplus state initialized successfully")
                    return new_state
                else:
                    self.initialization_attempts += 1
                    print(f"❌ Surplus state validation failed, attempt {self.initialization_attempts}")

                    # Record failed attempt
                    self.initialization_history.append({
                        'timestamp': time.time(),
                        'attempt': self.initialization_attempts,
                        'success': False
                    })

            except Exception as e:
                self.initialization_attempts += 1
                print(f"❌ Error creating surplus state: {e}")

                # Record error
                self.initialization_history.append({
                    'timestamp': time.time(),
                    'attempt': self.initialization_attempts,
                    'success': False,
                    'error': str(e)
                })

        print("⚠️ Maximum initialization attempts reached, using default state")
        default_state = SurplusState()

        # Ensure default state is properly initialized
        default_state.__post_init__()

        return default_state

    def reset_state(self) -> bool:
        """Reset surplus state to initial values."""
        try:
            self.initialization_attempts = 0
            self.surplus_state = self._create_initial_surplus_state()
            self.accumulation_momentum = {k: 0.0 for k in self.surplus_state.values.keys()}
            self.coupling_momentum = 0.0
            self.steps_since_expulsion = 0
            return self.surplus_state.validate()
        except Exception as e:
            print(f"Error resetting surplus state: {e}")
            return False

    def check_expulsion_needed(self, distinction_level: float) -> bool:
        """Check if surplus expulsion is needed based on current state"""
        try:
            if self.steps_since_expulsion < self.expulsion_cooldown:
                self.steps_since_expulsion += 1
                return False

            total_surplus = self.surplus_state.total_surplus()
            stability = self.surplus_state.stability

            # Check expulsion conditions using dynamic surplus_threshold
            conditions = [
                total_surplus > self.surplus_threshold,  # Using dynamic threshold instead of constant
                stability < self.stability_threshold,
                abs(distinction_level - 0.5) > 0.3  # High distinction deviation
            ]

            return any(conditions)

        except Exception as e:
            print(f"Error checking expulsion need: {e}")
            return False

    def get_dynamic_constants(self) -> Dict[str, float]:
        """
        Return the current values of all dynamic constants.

        Returns:
            Dictionary of dynamic constants and their current values
        """
        try:
            return {
                'surplus_threshold': self.surplus_threshold,
                'expulsion_recovery_rate': self.expulsion_recovery_rate,
                'surplus_adjustment_rate': self.surplus_adjustment_rate,
                'phase_scaling_factor': self.phase_scaling_factor,
                'surplus_recycle_fraction': self.surplus_recycle_fraction,
                'core_distinction_update_rate': self.core_distinction_update_rate,
                'distinction_anchor_weight': self.distinction_anchor_weight,
                'target_distinction': self.target_distinction,
                'collapse_dissipation_threshold': self.collapse_dissipation_threshold,
                'collapse_dissipation_rate': self.collapse_dissipation_rate,
                'instability_grace_period': self.instability_grace_period,
                'constant_adaptation_rate': self.constant_adaptation_rate
            }
        except Exception as e:
            print(f"Error getting dynamic constants: {e}")
            return {}

    def get_constants_history(self) -> List[Dict]:
        """
        Get history of dynamic constant changes.

        Returns:
            List of constant state snapshots with timestamps
        """
        try:
            return list(self.constants_history)
        except Exception as e:
            print(f"Error getting constants history: {e}")
            return []

    def update_surplus(self, phase_coherence: float, normalized_entropy: float):
        """Update surplus values with enhanced validation and error handling."""
        try:
            # Validate current state before update
            if not self.validate_current_state():
                print("⚠️ State validation failed, resetting surplus state")
                self.reset_state()
                return

            # Track emergence with properly referenced self state
            if hasattr(self, 'track_emergence') and hasattr(self, 'surplus_state'):
                self.track_emergence(self.surplus_state.values)

            # Ensure inputs are proper floats
            phase_coherence = float(phase_coherence)
            normalized_entropy = float(normalized_entropy)

            # Compute base adjustment factors
            coherence_factor = phase_coherence
            entropy_factor = 1.0 - normalized_entropy

            # Store previous total surplus for stability calculation
            previous_total_surplus = self.surplus_state.total_surplus()

            # Update each surplus type with different factors for each category
            for key in self.surplus_state.values:
                try:
                    # Get base rate with safety check
                    base_rate = float(self.surplus_state.accumulation_rate.get(key, 0.01))

                    # Apply different adjustment calculations for each surplus type
                    if key == 'basal':
                        # Basal responds more to entropy
                        adjustment = base_rate * (0.3 * coherence_factor - 0.7 * entropy_factor)
                    elif key == 'cognitive':
                        # Cognitive responds more to coherence
                        adjustment = base_rate * (0.7 * coherence_factor - 0.3 * entropy_factor)
                    elif key == 'predictive':
                        # Predictive has its own unique pattern
                        adjustment = base_rate * (coherence_factor * entropy_factor)
                    elif key == 'ontological':
                        # Ontological responds to the balance between coherence and entropy
                        adjustment = base_rate * (coherence_factor - entropy_factor) * self.emergence_sensitivity
                    else:
                        # Default calculation for any other keys
                        adjustment = base_rate * (coherence_factor - entropy_factor)

                    # Add randomness factor to create more variability (small random fluctuations)
                    adjustment *= (1.0 + 0.1 * (np.random.random() - 0.5))

                    # Initialize momentum if needed
                    if not hasattr(self, 'accumulation_momentum'):
                        self.accumulation_momentum = {}

                    # Update momentum
                    if key not in self.accumulation_momentum:
                        self.accumulation_momentum[key] = 0.0

                    self.accumulation_momentum[key] = (
                        MOMENTUM_DECAY * self.accumulation_momentum[key] +
                        (1 - MOMENTUM_DECAY) * adjustment
                    )

                    # Adjust surplus value with momentum
                    current_value = float(self.surplus_state.values[key])
                    new_value = current_value * (
                        1.0 + adjustment + 0.1 * self.accumulation_momentum[key]
                    )

                    # Ensure value stays within bounds
                    self.surplus_state.values[key] = np.clip(new_value, 0.1, MAX_SURPLUS)

                except Exception as e:
                    print(f"❌ Error updating surplus for {key}: {e}")
                    self.surplus_state.values[key] = 1.0  # Reset to safe default

            # Calculate current total surplus
            current_total_surplus = self.surplus_state.total_surplus()

            # Calculate surplus change for stability update
            surplus_change = abs(current_total_surplus - previous_total_surplus)

            # Update stability based on surplus change
            stability_decrease = surplus_change * 0.05  # Adjust factor as needed
            stability_increase = 0.02  # Small increase if surplus is stable

            # Calculate raw stability before clipping
            raw_stability = self.surplus_state.stability - stability_decrease + stability_increase

            # Initialize excess stability potential attribute if it doesn't exist
            if not hasattr(self, 'excess_stability_potential'):
                self.excess_stability_potential = 0.0

            # Store excess stability potential
            self.excess_stability_potential = max(0.0, raw_stability - 1.0)

            # Apply clipping after storing the excess
            new_stability = np.clip(raw_stability, 0.1, 1.0)

            print(f"DEBUG: Before stability update - current stability: {self.surplus_state.stability}")
            print(f"DEBUG: surplus change: {surplus_change}, raw stability: {raw_stability}, excess: {self.excess_stability_potential}")
            self.surplus_state.stability = new_stability
            print(f"DEBUG: After stability update - new stability: {self.surplus_state.stability}")

            # Apply excess stability effects to surplus accumulation rates
            if hasattr(self, 'excess_stability_potential') and self.excess_stability_potential > 0:
                # Scale accumulation rates based on excess stability
                stability_multiplier = 1.0 + (self.excess_stability_potential * 0.5)
                for key in self.surplus_state.accumulation_rate:
                    original_rate = self.surplus_state.accumulation_rate[key]
                    self.surplus_state.accumulation_rate[key] = original_rate * stability_multiplier
                    print(f"DEBUG: Increased {key} accumulation rate from {original_rate:.6f} to {self.surplus_state.accumulation_rate[key]:.6f}")

            # Track history
            self.accumulation_history.append({
                'timestamp': time.time(),
                'values': self.surplus_state.values.copy(),
                'stability': self.surplus_state.stability,
                'momentum': self.accumulation_momentum.copy(),
                'excess_stability': getattr(self, 'excess_stability_potential', 0.0)
            })

            # Update dynamic constants based on current system state
            self.update_dynamic_constants(
                phase_coherence=phase_coherence,
                distinction_level=0.5,  # Default if not available
                stability=self.surplus_state.stability
            )

        except Exception as e:
            print(f"❌ Error in surplus update: {e}")
            self.reset_state()

    def get_emergence_metrics(self) -> Dict[str, float]:
        """Get metrics about emergence patterns."""
        return {
            'novelty_score': self.novelty_score,
            'complexity_score': self.complexity_score,
            'emergence_sensitivity': self.emergence_sensitivity,
            'emergence_count': self.emergence_counter,
            'recent_emergence_rate': len(self.emergence_history) / max(1, len(self.emergence_patterns))
        }

    def validate_state(self) -> bool:
        """Validate current surplus state."""
        try:
            if not isinstance(self.surplus_state, SurplusState):
                print("Invalid surplus state type")
                return False

            if not isinstance(self.surplus_state.values, dict):
                print("Invalid surplus values type")
                return False

            required_keys = {'basal', 'cognitive', 'predictive', 'ontological'}
            if not all(key in self.surplus_state.values for key in required_keys):
                print("Missing required surplus keys")
                return False

            if not all(isinstance(v, (int, float)) for v in self.surplus_state.values.values()):
                print("Invalid surplus value types")
                return False

            return True

        except Exception as e:
            print(f"Error validating surplus state: {e}")
            return False

    def adjust_surplus(self, distinction_level: float,
                   quantum_metrics: Dict[str, float],
                   field_resistance: float) -> None:
        """
        Adjust surplus values based on current distinction and quantum metrics.
        """
        try:
            # Compute a distinction factor to modulate adjustments.
            distinction_factor = 1.0 - abs(distinction_level - 0.5)
            base_rate = SURPLUS_ADJUSTMENT_RATE * distinction_factor
            self._update_coupling_and_stability(quantum_metrics)

            for key in self.surplus_state.accumulation_rate:
                # Differentiate adjustment calculations by surplus type
                if key == 'basal':
                    adjustment = base_rate * self.surplus_state.accumulation_rate[key] * 0.9
                    adjustment *= (1.0 + 0.4 * quantum_metrics.get('phase_coherence', 0.5))
                elif key == 'cognitive':
                    adjustment = base_rate * self.surplus_state.accumulation_rate[key] * 1.1
                    adjustment *= (1.0 + 0.7 * (1.0 - quantum_metrics.get('normalized_entropy', 0.5)))
                elif key == 'predictive':
                    adjustment = base_rate * self.surplus_state.accumulation_rate[key] * 1.0
                    adjustment *= (0.8 + 0.4 * (1.0 - field_resistance))
                elif key == 'ontological':
                    # Fixed the syntax error here - removed line continuation
                    adjustment = base_rate * self.surplus_state.accumulation_rate[key] * 1.2
                    adjustment *= (1.0 + 0.3 * quantum_metrics.get('phase_coherence', 0.5))
                    adjustment *= (1.0 - 0.3 * quantum_metrics.get('normalized_entropy', 0.5))

                # Add small random factor for natural variation
                adjustment *= (1.0 + 0.05 * (np.random.random() - 0.5))

                self.accumulation_momentum[key] = update_momentum(self.accumulation_momentum[key], adjustment)
                self.surplus_state.values[key] *= (1.0 + adjustment + 0.1 * self.accumulation_momentum[key])
                self.surplus_state.values[key] = np.clip(self.surplus_state.values[key], 0.1, MAX_SURPLUS)

            # Append a snapshot of the current accumulation history.
            self.accumulation_history.append({
                'values': self.surplus_state.values.copy(),
                'rates': self.surplus_state.accumulation_rate.copy(),
                'stability': self.surplus_state.stability,
                'quantum_coupling': self.surplus_state.quantum_coupling
            })
        except Exception as e:
            print(f"Error adjusting surplus: {e}")

    def _update_coupling_and_stability(self, quantum_metrics: Dict[str, float]) -> None:
        """
        Update quantum coupling and surplus stability based on quantum metrics.
        """
        try:
            target_coupling = quantum_metrics.get('phase_coherence', 0.5) * (1.0 - quantum_metrics.get('normalized_entropy', 0.5))
            self.coupling_momentum = update_momentum(self.coupling_momentum, target_coupling - self.surplus_state.quantum_coupling)
            self.surplus_state.quantum_coupling = np.clip(
                self.surplus_state.quantum_coupling + 0.1 * self.coupling_momentum, 0.1, 1.0
            )
            self.surplus_state.stability = np.clip(
                self.stability_decay * self.surplus_state.stability + 0.1 * quantum_metrics.get('phase_coherence', 0.5),
                0.1, 1.0
            )
            self.stability_history.append(self.surplus_state.stability)
        except Exception as e:
            print(f"Error updating coupling and stability: {e}")

    def get_surplus_state(self) -> SurplusState:
        """Return the current surplus state."""
        return self.surplus_state

    def perform_expulsion(self, quantum_state: Any) -> Tuple[Dict[str, float], float]:
        """Perform surplus expulsion with quantum feedback and proper type handling"""
        try:
            # Validate and store current surplus state
            if not isinstance(self.surplus_state, SurplusState):
                print("Warning: Invalid surplus state type, creating new instance")
                self.surplus_state = SurplusState()

            # Create a copy of current values before expulsion
            expelled = self.surplus_state.values.copy()

            # Calculate total surplus with proper type handling
            total_expelled = float(sum(self.surplus_state.values.values()))

            # Reset surplus to baseline with proper type handling
            for key in self.surplus_state.values:
                self.surplus_state.values[key] = 1.0

            # Apply quantum corrections
            try:
                quantum_state.apply_gate('x', [0])
                phase_shift = float(np.pi * (1.0 - self.surplus_state.stability))
                quantum_state.apply_phase_shift(phase_shift)
            except Exception as qe:
                print(f"Error applying quantum corrections: {qe}")

            # Update expulsion tracking
            self.surplus_state.last_expulsion = float(total_expelled)
            self.steps_since_expulsion = 0

            # Track expulsion event
            self.expulsion_history.append({
                'magnitude': float(total_expelled),
                'stability': float(self.surplus_state.stability),
                'timestamp': time.time()
            })

            # Set up recovery state with proper type handling
            self.recovery_state = {
                'steps_remaining': int(total_expelled * 10),
                'initial_magnitude': float(total_expelled)
            }

            return expelled, float(total_expelled)

        except Exception as e:
            print(f"Error performing expulsion: {e}")
            default_values = {'basal': 1.0, 'cognitive': 1.0, 'predictive': 1.0, 'ontological': 1.0}
            return default_values, 0.0

    def process_recovery(self, quantum_state: Any, distinction_level: float) -> None:
        """Process recovery after surplus expulsion"""
        if self.recovery_state is None:
            return

        try:
            self.recovery_state['steps_remaining'] -= 1
            if self.recovery_state['steps_remaining'] <= 0:
                self.recovery_state = None
                print("Recovery complete")
                return

            # Calculate recovery rate based on distinction
            recovery_rate = self.expulsion_recovery_rate * (1.0 + distinction_level)

            # Apply recovery to surplus values
            for key in self.surplus_state.values:
                self.surplus_state.values[key] += recovery_rate

            # Apply quantum corrections during early recovery
            phase_shift = np.pi * (0.8 - 0.5 * self.surplus_state.stability)
            quantum_state.apply_phase_shift(phase_shift)

        except Exception as e:
            print(f"Error processing recovery: {e}")
            self.recovery_state = None

    def get_surplus_metrics(self) -> Dict[str, float]:
        """Get comprehensive surplus metrics"""
        try:
            if not hasattr(self, 'surplus_state') or not isinstance(self.surplus_state, SurplusState):
                return {'error': 'Invalid surplus state'}

            metrics = {
                'total_surplus': self.surplus_state.total_surplus(),
                'stability': self.surplus_state.stability,
                'stability_momentum': self.stability_momentum,
                'coupling_momentum': self.coupling_momentum,
                'steps_since_expulsion': self.steps_since_expulsion,
                'in_recovery': self.recovery_state is not None,
                'excess_stability': getattr(self, 'excess_stability_potential', 0.0)  # Add this line
            }

            # Add per-type metrics
            for key in self.surplus_state.values:
                metrics[f'{key}_surplus'] = self.surplus_state.values[key]
                metrics[f'{key}_momentum'] = self.accumulation_momentum.get(key, 0.0)
                metrics[f'{key}_rate'] = self.surplus_state.accumulation_rate.get(key, 0.01)

            # Add recovery metrics if active
            if self.recovery_state is not None:
                metrics['recovery_progress'] = 1.0 - (
                    self.recovery_state['steps_remaining'] /
                    (self.recovery_state['initial_magnitude'] * 10)
                )

            return metrics

        except Exception as e:
            print(f"Error getting surplus metrics: {e}")
            return {
                'total_surplus': 0.0,
                'stability': self.minimum_stability,
                'error': str(e)
            }

# ---------------------------
# EnhancedDistinctionDynamics
# ---------------------------
class EnhancedDistinctionDynamics:
    """
    Enhanced distinction dynamics with momentum-based updates and core distinction anchoring.
    """
    def __init__(self, memory_size: int = 100, phase_history_maxlen: int = 100) -> None:
        self.cognitive_metrics = {}
        self.distinction_history = deque(maxlen=memory_size)
        self.phase_history = deque(maxlen=phase_history_maxlen)
        self.adjustment_momentum = 0.0
        self.distinction_level = 0.5
        self.stability_factor = 1.0
        self.quantum_influence = 0.0
        self.core_distinction = 1.0  # Core anchoring mechanism

        # Convert hard-coded constants to dynamic instance variables
        self.core_distinction_update_rate = 0.01  # Default from CORE_DISTINCTION_UPDATE_RATE
        self.distinction_threshold = 0.7  # Default from TARGET_DISTINCTION
        self.stability_decay = 0.7  # Default from MOMENTUM_DECAY
        self.distinction_anchor_weight = 0.2  # Default from DISTINCTION_ANCHOR_WEIGHT

        # Additional adaptive parameters for dynamic operation
        self.dynamic_anchor_weight = 0.2  # Initial value, can be adjusted
        self.anchor_weight_momentum = 0.0
        self.update_rate_momentum = 0.0
        self.constants_adaptation_rate = 0.01  # Base rate for constant adjustment

        self.coherence_momentum = 0.0
        self.distinction_momentum = 0.0
        self.adaptation_history = deque(maxlen=1000)
        self.stability_threshold = 0.1
        self.recovery_mode = False
        self.recovery_steps = 0
        self.minimum_distinction = 0.1
        self.learning_rate = 0.005  # Added learning rate attribute

        # History tracking for dynamic constants
        self.constants_history = deque(maxlen=100)
        self._record_constants_state("initialization")

    def _record_constants_state(self, event_type: str) -> None:
        """Record the current state of all dynamic constants for tracking changes over time."""
        try:
            constants_state = {
                'core_distinction_update_rate': self.core_distinction_update_rate,
                'distinction_threshold': self.distinction_threshold,
                'stability_decay': self.stability_decay,
                'distinction_anchor_weight': self.distinction_anchor_weight,
                'dynamic_anchor_weight': self.dynamic_anchor_weight,
                'event_type': event_type,
                'timestamp': time.time()
            }
            self.constants_history.append(constants_state)
        except Exception as e:
            print(f"Error recording constants state: {e}")

    def update_dynamic_constants(self,
                               distinction_level: float,
                               stability_factor: float,
                               distinction_variance: float,
                               coherence: float) -> None:
        """
        Update dynamic constants based on system behavior and performance.

        Args:
            distinction_level: Current distinction level
            stability_factor: Current stability factor
            distinction_variance: Variance in distinction level
            coherence: Current coherence level
        """
        try:
            # Store previous values to track changes
            prev_anchor_weight = self.distinction_anchor_weight
            prev_update_rate = self.core_distinction_update_rate

            # 1. Adjust anchor weight based on distinction variance and stability
            if distinction_variance > 0.1:
                # High variance - increase anchor weight to stabilize
                anchor_adjustment = 0.02 * self.constants_adaptation_rate * distinction_variance

                # Update momentum
                self.anchor_weight_momentum = update_momentum(
                    self.anchor_weight_momentum,
                    anchor_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.distinction_anchor_weight += anchor_adjustment + 0.1 * self.anchor_weight_momentum
                self.distinction_anchor_weight = np.clip(self.distinction_anchor_weight, 0.1, 0.5)

                print(f"Increased distinction_anchor_weight to {self.distinction_anchor_weight:.4f} based on high variance")

            elif stability_factor > 0.8 and distinction_variance < 0.05:
                # System is stable with low variance - can reduce anchor weight
                anchor_adjustment = -0.01 * self.constants_adaptation_rate

                # Update momentum
                self.anchor_weight_momentum = update_momentum(
                    self.anchor_weight_momentum,
                    anchor_adjustment,
                    decay=self.stability_decay
                )
                # Apply adjustment with momentum
                self.distinction_anchor_weight += anchor_adjustment + 0.1 * self.anchor_weight_momentum
                self.distinction_anchor_weight = np.clip(self.distinction_anchor_weight, 0.1, 0.5)

                print(f"Decreased distinction_anchor_weight to {self.distinction_anchor_weight:.4f} based on stability")

            # 2. Adjust update rate based on coherence and stability
            if coherence > 0.7 and stability_factor > 0.7:
                # High coherence and stability - can use faster update rate
                update_rate_adjustment = 0.005 * self.constants_adaptation_rate * coherence

                # Update momentum
                self.update_rate_momentum = update_momentum(
                    self.update_rate_momentum,
                    update_rate_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.core_distinction_update_rate += update_rate_adjustment + 0.05 * self.update_rate_momentum
                self.core_distinction_update_rate = np.clip(self.core_distinction_update_rate, 0.005, 0.05)

                print(f"Increased core_distinction_update_rate to {self.core_distinction_update_rate:.4f} based on coherence")

            elif stability_factor < 0.4 or coherence < 0.3:
                # Low stability or coherence - use slower update rate
                update_rate_adjustment = -0.005 * self.constants_adaptation_rate

                # Update momentum
                self.update_rate_momentum = update_momentum(
                    self.update_rate_momentum,
                    update_rate_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.core_distinction_update_rate += update_rate_adjustment + 0.05 * self.update_rate_momentum
                self.core_distinction_update_rate = np.clip(self.core_distinction_update_rate, 0.005, 0.05)

                print(f"Decreased core_distinction_update_rate to {self.core_distinction_update_rate:.4f} based on low stability")

            # 3. Adjust distinction threshold based on actual distinction level
            distinction_delta = distinction_level - self.distinction_threshold
            if abs(distinction_delta) > 0.2 and stability_factor > 0.6:
                # Move threshold toward actual distinction if stable
                threshold_adjustment = 0.02 * self.constants_adaptation_rate * distinction_delta
                self.distinction_threshold += threshold_adjustment
                self.distinction_threshold = np.clip(self.distinction_threshold, 0.3, 0.9)

                print(f"Adjusted distinction_threshold to {self.distinction_threshold:.4f} based on actual distinction")

            # Record significant constant changes
            if (abs(self.distinction_anchor_weight - prev_anchor_weight) > 0.02 or
                abs(self.core_distinction_update_rate - prev_update_rate) > 0.005):
                self._record_constants_state("significant_update")

        except Exception as e:
            print(f"Error updating distinction dynamic constants: {e}")
            traceback.print_exc()


    def compute_distinction(self, quantum_metrics: Dict[str, float],
                       field_resistance: float,
                       surplus_state: SurplusState,
                       excess_stability: float = 0.0) -> float:
       """
       Compute distinction with enhanced stability through momentum and core anchoring.

       Args:
           quantum_metrics: Dictionary of quantum system metrics
           field_resistance: Field resistance value
           surplus_state: Surplus state object
           excess_stability: Amount of excess stability potential

       Returns:
           Distinction level value in range [0.1, 1.0]
       """
       try:
           # Validate input metrics
           if not isinstance(quantum_metrics, dict):
               raise ValueError("Invalid metrics input")

           # Define default metrics with safe values
           default_metrics = {
               'normalized_entropy': 0.5,
               'phase_distinction': 0.5,
               'coherence_distinction': 0.5,
               'quantum_surplus_coupling': 0.5,
               'stability': 1.0,
               'quantum_coupling': 1.0
           }

           # Update metrics with defaults for missing values
           metrics = {}
           for key, default_value in default_metrics.items():
               metrics[key] = quantum_metrics.get(key, default_value)

           entropy_component = 1.0 - metrics['normalized_entropy']
           phase_component = metrics['phase_distinction']
           coherence_component = max(metrics['coherence_distinction'], MINIMUM_COHERENCE_FLOOR)

           base_distinction = (
               0.3 * phase_component +
               0.3 * entropy_component +
               0.2 * coherence_component +
               0.2 * (1.0 - field_resistance)
           )

           # Ensure surplus_state is valid
           if not isinstance(surplus_state, SurplusState):
               print("Warning: Invalid surplus state, creating new instance")
               surplus_state = SurplusState()

           # Get quantum coupling
           coupling = metrics['quantum_surplus_coupling']

           # Scale distinction
           scaled_distinction = base_distinction * (0.5 + 0.5 * coupling)

           # Update momentum
           delta = scaled_distinction - self.distinction_level
           self.adjustment_momentum = update_momentum(self.adjustment_momentum, delta)

           # Calculate momentum contribution
           momentum_contribution = self.adjustment_momentum * (1.0 - 0.8 * abs(scaled_distinction - 0.5))

           # Modify distinction calculation with excess stability
           if excess_stability > 0:
               # Amplify distinction changes when excess stability is present
               scaled_distinction *= (1.0 + excess_stability * 0.5)

               # Reduce anchor weight to allow more variability
               effective_anchor_weight = self.distinction_anchor_weight * (1.0 - min(0.4, excess_stability))

               # Update core distinction
               if scaled_distinction > self.core_distinction:
                   self.core_distinction = 0.95 * self.core_distinction + 0.05 * scaled_distinction
               else:
                   self.core_distinction = (
                       (1 - self.core_distinction_update_rate) * self.core_distinction +
                       self.core_distinction_update_rate * scaled_distinction
                   )

               # Compute final distinction
               final_distinction = (
                   (1 - effective_anchor_weight) * (scaled_distinction + momentum_contribution) +
                   effective_anchor_weight * self.core_distinction
               )
           else:
               # Update core distinction
               if scaled_distinction > self.core_distinction:
                   self.core_distinction = 0.95 * self.core_distinction + 0.05 * scaled_distinction
               else:
                   self.core_distinction = (
                       (1 - self.core_distinction_update_rate) * self.core_distinction +
                       self.core_distinction_update_rate * scaled_distinction
                   )

               # Apply momentum with stability
               momentum_contribution = self.adjustment_momentum * (1.0 - 0.8 * abs(scaled_distinction - 0.5))

               # Compute final distinction
               final_distinction = (
                   (1 - self.distinction_anchor_weight) * (scaled_distinction + momentum_contribution) +
                   self.distinction_anchor_weight * self.core_distinction
               )

           # Apply stability-based adjustment
           if self.stability_factor < 0.5:
               final_distinction = 0.7 * final_distinction + 0.3 * self.core_distinction

           # Store history
           self.distinction_history.append(final_distinction)

           # Calculate distinction variance for dynamic constant adjustment
           distinction_variance = np.var(list(self.distinction_history)[-10:]) if len(self.distinction_history) >= 10 else 0.0

           # Update dynamic constants based on performance
           self.update_dynamic_constants(
               distinction_level=final_distinction,
               stability_factor=self.stability_factor,
               distinction_variance=distinction_variance,
               coherence=metrics['coherence_distinction']
           )

           return float(np.clip(final_distinction, self.minimum_distinction, 1.0))

       except Exception as e:
           print(f"Error computing distinction: {e}")
           return self.distinction_history[-1] if self.distinction_history else 0.5


    def update_distinction_from_phase(self, estimated_phase: float) -> float:
        """
        Update the distinction level based on a new phase estimate.
        This uses an exponential moving average to update momentum and blends the current
        distinction with the historical average.
        """
        try:
            avg_phase = estimated_phase if not self.phase_history else np.mean(self.phase_history)
            phase_diff = abs(estimated_phase - avg_phase)
            self.distinction_momentum = update_momentum(self.distinction_momentum, phase_diff)
            new_distinction = ((1 - DISTINCTION_ANCHOR_WEIGHT) *
                               (self.distinction_level +
                                self.distinction_momentum +
                                0.1 * (self.core_distinction - self.distinction_level)) +
                               DISTINCTION_ANCHOR_WEIGHT * avg_phase)
            self.distinction_level = float(np.clip(new_distinction, self.minimum_distinction, 1.0))
            self.phase_history.append(estimated_phase)
            self._update_stability(new_distinction)
            return self.distinction_level
        except Exception as e:
            print(f"Error updating distinction from phase: {e}")
            return self.distinction_level

    def _update_stability(self, new_distinction: float) -> None:
        """
        Update stability metrics based on recent changes in distinction.
        """
        try:
            if len(self.distinction_history) > 1:
                recent_variance = np.var(list(self.distinction_history)[-10:])
                self.stability_factor = 1.0 / (1.0 + recent_variance)
                self.adaptation_history.append({
                    'distinction': new_distinction,
                    'stability': self.stability_factor,
                    'momentum': self.distinction_momentum
                })
                if self.stability_factor < self.stability_threshold and not self.recovery_mode:
                    if len(self.distinction_history) > 10 and np.var(list(self.distinction_history)[-10:]) > 0.01:
                        self.enter_recovery_mode()

        except Exception as e:
            print(f"Error updating stability: {e}")

    def enter_recovery_mode(self) -> None:
        """Enter recovery mode to reduce rapid fluctuations in distinction."""
        try:
            self.recovery_mode = True
            self.recovery_steps = 50
            self.distinction_momentum = 0.0
            print("Distinction dynamics entering recovery mode")
        except Exception as e:
            print(f"Error entering recovery mode: {e}")

    def get_distinction_metrics(self) -> Dict[str, float]:
       """
       Return a dictionary of distinction metrics including current distinction,
       core distinction, stability, momentum values, and quantum influence.
       """
       try:
           metrics = {
               'current_distinction': self.distinction_level,
               'core_distinction': self.core_distinction,
               'stability_factor': self.stability_factor,
               'distinction_momentum': self.distinction_momentum,
               'adjustment_momentum': self.adjustment_momentum,
               'quantum_influence': self.quantum_influence,
               'recovery_mode': self.recovery_mode,
               'recovery_steps': self.recovery_steps if self.recovery_mode else 0,
               # Add dynamic constants to metrics
               'distinction_anchor_weight': self.distinction_anchor_weight,
               'core_distinction_update_rate': self.core_distinction_update_rate,
               'distinction_threshold': self.distinction_threshold,
               'anchor_weight_momentum': self.anchor_weight_momentum,
               'update_rate_momentum': self.update_rate_momentum
           }
           if self.distinction_history:
               metrics.update({
                   'mean_distinction': float(np.mean(self.distinction_history)),
                   'distinction_variance': float(np.var(self.distinction_history))
               })
           return metrics
       except Exception as e:
           print(f"Error getting distinction metrics: {e}")
           return {
               'current_distinction': self.distinction_level,
               'core_distinction': self.core_distinction,
               'stability_factor': self.stability_factor
           }

    def get_constants_history(self) -> List[Dict]:
       """
       Get history of dynamic constant changes.

       Returns:
           List of constant state snapshots with timestamps
       """
       try:
           return list(self.constants_history)
       except Exception as e:
           print(f"Error getting constants history: {e}")
           return []

    def track_history(self):
        """Append the current distinction level to history."""
        self.distinction_history.append(self.distinction_level)

