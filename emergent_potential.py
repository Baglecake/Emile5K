"""
Emergent Potential Field for Émile-3K Simulation
------------------------------------------------
Tracks and manages excess stability potentials across system components to
create emergent properties and phase transitions.
"""
import time
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import traceback

class EmergentPotentialField:
    """
    Manages excess stability potentials across system components to create
    emergent properties and phase transitions.
    """

    def __init__(self, base_threshold: float = 0.5, history_size: int = 100):
        """
        Initialize the emergent potential field with enhanced adaptivity.

        Args:
            base_threshold: Baseline threshold for triggering emergence
            history_size: Size of history tracking
        """
        self.component_potentials = {}  # Maps components to their excess potentials
        self.component_weights = {}     # Importance weights for different components
        self.total_potential = 0.0      # Accumulated potential across all components
        self.base_threshold = base_threshold  # Threshold baseline - never changes
        self.threshold = base_threshold  # Current adaptive threshold

        # Advanced threshold adaptation parameters
        self.min_threshold = 0.1        # Lower bound for threshold
        self.max_threshold = 2.0        # Upper bound for threshold
        self.threshold_adaptation_rate = 0.05  # How quickly threshold adapts
        self.system_stability = 1.0     # Overall system stability measure

        # Enhanced emergence probability calculation
        self.min_probability = 0.001    # Minimum emergence probability
        self.probability_modifiers = {}  # Environmental factors affecting probability

        # Learning parameters
        self.learning_rate = 0.01       # Rate for learning from emergence events
        self.learning_enabled = True    # Whether to enable learning

        # History tracking
        self.potential_history = deque(maxlen=history_size)
        self.emergence_history = deque(maxlen=history_size)
        self.threshold_history = deque(maxlen=history_size)  # Track threshold changes

        # Emergence state
        self.last_emergence = None
        self.emergence_active = False
        self.emergence_cooldown = 0
        self.emergence_counter = 0

        # Adaptation parameters
        self.adaptation_rate = 0.02
        self.threshold_momentum = 0.0
        self.stability_factor = 1.0
        self.field_intensity = 1.0

        # Performance metrics
        self.avg_time_between_emergences = 120.0  # Default assumption (seconds)
        self.successful_emergences = 0
        self.failed_emergences = 0
        self.emergence_duration_history = deque(maxlen=20)

        # Initialize default component weights
        self._initialize_component_weights()

    def _initialize_component_weights(self):
        """Initialize default importance weights for different components."""
        self.component_weights = {
            'surplus': 1.0,       # Surplus dynamics is a primary contributor
            'distinction': 0.8,   # Distinction dynamics
            'quantum': 0.9,       # Quantum state
            'cognitive': 0.7,     # Cognitive structures
            'field': 0.5          # Ontological field
        }

        # Initialize probability modifiers
        self.probability_modifiers = {
            'system_complexity': 1.0,    # Higher complexity can increase probability
            'environmental_noise': 1.0,  # More noise can increase probability
            'recent_success_rate': 1.0,  # Success rate affects probability
            'time_factor': 1.0,          # Time since last emergence
        }

    def register_potential(self, component_id: str, potential: float,
                      component_type: str = 'surplus', state_metrics: Optional[Dict] = None):
        """
        Register excess stability potential from a component with enhanced adaptivity.

        Args:
            component_id: Unique identifier for the component
            potential: Excess stability potential value
            component_type: Type of component for weighting
            state_metrics: Optional associated state metrics

        Returns:
            True if an emergence event was triggered, False otherwise
        """
        try:
            # Apply component weight to potential
            weight = self.component_weights.get(component_type, 0.5)
            weighted_potential = potential * weight

            # Store component potential
            self.component_potentials[component_id] = {
                'raw_potential': potential,
                'weighted_potential': weighted_potential,
                'component_type': component_type,
                'timestamp': time.time(),
                'metrics': state_metrics or {}
            }

            # Calculate new total potential
            self._calculate_total_potential()

            # Adapt threshold based on system state
            self._adapt_threshold_dynamic(state_metrics)
            self.threshold_history.append(self.threshold)

            # Track history
            self.potential_history.append({
                'timestamp': time.time(),
                'total_potential': self.total_potential,
                'components': self.component_potentials.copy(),
                'threshold': self.threshold
            })

            # Calculate emergence probability with enhanced formula
            emergence_probability = self._calculate_emergence_probability(state_metrics)

            # Update system stability based on new metrics
            if state_metrics and 'stability' in state_metrics:
                self.system_stability = 0.9 * self.system_stability + 0.1 * state_metrics['stability']

            # Check for emergence threshold
            triggered = False
            if self.emergence_cooldown <= 0:
                # Use both threshold and random probability check
                if self.total_potential > self.threshold:
                    triggered = self.trigger_emergence()
                elif random.random() < emergence_probability:
                    print(f"RANDOM EMERGENCE: Probability triggered emergence with p={emergence_probability:.4f}")
                    triggered = self.trigger_emergence()

                    # This was a spontaneous emergence
                    if triggered and hasattr(self, 'last_emergence'):
                        self.last_emergence['spontaneous'] = True

            # Update state even if not triggered
            self._update_field_state()

            return triggered

        except Exception as e:
            print(f"Error registering potential: {e}")
            return False

    def _calculate_total_potential(self):
        """Calculate the total potential across all components with enhanced time decay."""
        try:
            # Apply time decay to older potentials
            current_time = time.time()
            decay_factor = 0.9  # Decay rate for older potentials

            # Calculate weighted sum with time decay
            total = 0.0
            component_count = 0

            for component_id, data in list(self.component_potentials.items()):
                time_elapsed = current_time - data['timestamp']
                if time_elapsed > 60:  # Remove entries older than 60 seconds
                    self.component_potentials.pop(component_id, None)
                    continue

                # Apply exponential time decay
                time_decay = decay_factor ** (time_elapsed / 10.0)  # Decay over time

                # Extract component metrics if available
                metrics = data.get('metrics', {})

                # Apply stability modifier if available in the metrics
                stability_modifier = 1.0
                if metrics and 'stability' in metrics:
                    stability = metrics['stability']
                    # Higher stability values lead to higher potentials
                    stability_modifier = 0.5 + min(1.5, stability)

                # Apply modified decay
                decayed_potential = data['weighted_potential'] * time_decay * stability_modifier

                # Add to total and increment counter
                total += decayed_potential
                component_count += 1

            # Apply component diversity factor - more components = more potential
            if component_count > 1:
                diversity_factor = min(1.5, 1.0 + (component_count - 1) * 0.1)
                total *= diversity_factor

            self.total_potential = total

        except Exception as e:
            print(f"Error calculating total potential: {e}")
            self.total_potential = sum(data.get('weighted_potential', 0.0)
                                     for data in self.component_potentials.values())

    def _calculate_emergence_probability(self, state_metrics: Optional[Dict] = None) -> float:
        """
        Calculate enhanced probability of emergence based on multiple factors.

        Args:
            state_metrics: Optional system state metrics

        Returns:
            Probability of emergence (0.0 to 1.0)
        """
        try:
            # Base probability is ratio of potential to threshold
            base_probability = self.total_potential / (self.threshold * 1.2)

            # Apply probability modifiers
            modifiers = 1.0

            # Time factor - more time since last emergence increases probability
            if self.last_emergence:
                time_since_last = time.time() - self.last_emergence['timestamp']
                time_factor = min(2.0, 1.0 + time_since_last / self.avg_time_between_emergences)
                self.probability_modifiers['time_factor'] = time_factor
                modifiers *= time_factor

            # Recent success rate affects probability
            if self.successful_emergences + self.failed_emergences > 0:
                success_rate = self.successful_emergences / (self.successful_emergences + self.failed_emergences)
                success_factor = 0.5 + success_rate
                self.probability_modifiers['recent_success_rate'] = success_factor
                modifiers *= success_factor

            # System state metrics can affect probability
            if state_metrics:
                # System complexity
                if 'complexity' in state_metrics:
                    complexity = state_metrics['complexity']
                    complexity_factor = min(1.5, 0.8 + complexity)
                    self.probability_modifiers['system_complexity'] = complexity_factor
                    modifiers *= complexity_factor

                # Environmental noise (entropy)
                if 'entropy' in state_metrics:
                    entropy = state_metrics['entropy']
                    noise_factor = min(1.5, 0.8 + entropy)
                    self.probability_modifiers['environmental_noise'] = noise_factor
                    modifiers *= noise_factor

            # Calculate final probability with all modifiers
            final_probability = base_probability * modifiers

            # Apply cooldown reduction if in cooldown
            if self.emergence_cooldown > 0:
                final_probability *= 0.1

            # Ensure probability is in valid range
            return max(self.min_probability, min(0.95, final_probability))

        except Exception as e:
            print(f"Error calculating emergence probability: {e}")
            return self.min_probability

    def _adapt_threshold_dynamic(self, state_metrics: Optional[Dict] = None):
        """
        Adapt the emergence threshold dynamically based on system state, history, and performance.

        Args:
            state_metrics: Optional system state metrics
        """
        try:
            # Start with current threshold
            new_threshold = self.threshold

            # Adjust based on emergence history and frequency
            if len(self.emergence_history) >= 2:
                # Calculate average time between emergences
                timestamps = [e['timestamp'] for e in self.emergence_history]
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

                if intervals:
                    self.avg_time_between_emergences = sum(intervals) / len(intervals)

                    # Target frequency adjustment
                    target_interval = 60.0  # Target 1 minute between emergences

                    if self.avg_time_between_emergences < target_interval * 0.5:
                        # Too frequent - increase threshold
                        adjustment = +0.05
                    elif self.avg_time_between_emergences > target_interval * 2.0:
                        # Too infrequent - decrease threshold
                        adjustment = -0.05
                    else:
                        # Within acceptable range - small adjustments for fine-tuning
                        interval_ratio = target_interval / self.avg_time_between_emergences
                        adjustment = 0.02 * (1.0 - interval_ratio)

                    # Apply adjustment with learning rate
                    new_threshold += adjustment * self.threshold_adaptation_rate

            # Adjust based on system state metrics
            if state_metrics:
                metrics_adjustment = 0.0

                # Adjust based on stability
                if 'stability' in state_metrics:
                    stability = state_metrics['stability']
                    # Lower thresholds for highly stable systems to encourage emergence
                    # Higher thresholds for unstable systems to prevent too much emergence
                    metrics_adjustment -= (stability - 0.5) * 0.1

                # Adjust based on complexity
                if 'complexity' in state_metrics:
                    complexity = state_metrics['complexity']
                    # More complex systems have lower thresholds (more likely to emerge)
                    metrics_adjustment -= (complexity - 0.5) * 0.1

                # Apply metrics-based adjustment
                new_threshold += metrics_adjustment * self.threshold_adaptation_rate

            # Apply threshold momentum for smoother transitions
            self.threshold_momentum = 0.9 * self.threshold_momentum + 0.1 * (new_threshold - self.threshold)
            new_threshold = self.threshold + self.threshold_momentum

            # Ensure threshold stays within bounds
            self.threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        except Exception as e:
            print(f"Error adapting threshold: {e}")
            # Keep current threshold on error

    def trigger_emergence(self):
        """
        Trigger system-wide emergence based on accumulated potential with enhanced tracking.

        Returns:
            True if emergence was triggered, False otherwise
        """
        try:
            # Safety check for cooldown
            if self.emergence_cooldown > 0:
                return False

            # Prepare emergence data
            self.last_emergence = {
                'timestamp': time.time(),
                'potential': self.total_potential,
                'threshold': self.threshold,
                'probability_modifiers': self.probability_modifiers.copy(),
                'contributors': {k: v.copy() for k, v in self.component_potentials.items()},
                'intensity': self._calculate_emergence_intensity(),
                'sequence_number': self.emergence_counter,
                'spontaneous': False  # Default - changed if random emergence
            }

            # Store in history
            self.emergence_history.append(self.last_emergence)

            # Update counters and state
            self.emergence_counter += 1
            self.emergence_active = True

            # Calculate adaptive cooldown based on emergence intensity
            intensity = self.last_emergence['intensity']
            base_cooldown = 10  # Minimum cooldown period
            intensity_factor = intensity * 3  # Higher intensity = longer cooldown

            # Apply randomness for variability
            random_factor = 0.8 + 0.4 * random.random()  # 0.8 to 1.2

            # Calculate final cooldown
            self.emergence_cooldown = int((base_cooldown + intensity_factor) * random_factor)

            # Reset potentials after emergence (but not completely)
            reset_factor = max(0.1, min(0.5, 0.3 / intensity))  # Stronger emergence = more complete reset
            self.component_potentials = {k: {**v, 'weighted_potential': v['weighted_potential'] * reset_factor}
                                       for k, v in self.component_potentials.items()}
            self._calculate_total_potential()

            # Adapt threshold based on emergence history and this event
            self._adapt_threshold()

            # Track successful emergence
            self.successful_emergences += 1

            print(f"EMERGENCE TRIGGERED: Potential {self.total_potential:.4f}, Intensity {self.last_emergence['intensity']:.4f}, Threshold {self.threshold:.4f}")

            return True

        except Exception as e:
            print(f"Error triggering emergence: {e}")
            self.failed_emergences += 1
            return False

    def _calculate_emergence_intensity(self):
        """
        Calculate the intensity of the emergence event with enhanced dynamics.

        Returns:
            Intensity value (1.0 to 5.0)
        """
        try:
            # Base intensity is how much the potential exceeds the threshold
            threshold_ratio = self.total_potential / max(self.threshold, 0.001)

            # Component diversity factor - more different components = more intense emergence
            component_types = set(data['component_type'] for data in self.component_potentials.values())
            diversity_factor = min(2.0, 1.0 + len(component_types) * 0.2)

            # System stability factor - less stable systems have more intense emergence
            stability_factor = max(1.0, 2.0 - self.system_stability)

            # Calculate base intensity with these factors
            base_intensity = threshold_ratio * diversity_factor * stability_factor

            # Scale to a reasonable range [1.0 - 5.0]
            scaled_intensity = 1.0 + min(4.0, base_intensity - 1.0)

            # Add some randomness for variation (±10%)
            randomness = 0.9 + 0.2 * random.random()
            final_intensity = scaled_intensity * randomness

            # Track this intensity in history for learning
            if hasattr(self, 'intensity_history'):
                self.intensity_history.append(final_intensity)
            else:
                self.intensity_history = deque(maxlen=20)
                self.intensity_history.append(final_intensity)

            return float(final_intensity)

        except Exception as e:
            print(f"Error calculating emergence intensity: {e}")
            return 1.0

    def _adapt_threshold(self):
        """
        Adapt the emergence threshold based on emergence history with learning capabilities.
        """
        try:
            # Get emergence timestamps and intensities
            if len(self.emergence_history) < 2:
                return

            recent_emergences = list(self.emergence_history)[-10:]
            timestamps = [e['timestamp'] for e in recent_emergences]
            intensities = [e.get('intensity', 1.0) for e in recent_emergences]

            # Skip adaptation if learning is disabled
            if not self.learning_enabled:
                return

            # Calculate time intervals
            intervals = []
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                intervals.append(interval)

            if not intervals:
                return

            # Calculate mean interval and intensity
            mean_interval = np.mean(intervals)
            mean_intensity = np.mean(intensities)

            # Learning parameters
            target_interval = 60.0  # Target 1 minute between emergences
            target_intensity = 2.5  # Target medium intensity

            # Adjustments based on targets
            interval_adjustment = 0.0
            intensity_adjustment = 0.0

            # Interval adjustment
            if mean_interval < target_interval * 0.5:
                # Too frequent - increase threshold
                interval_adjustment = 0.05 * (1.0 - (mean_interval / target_interval))
            elif mean_interval > target_interval * 2.0:
                # Too infrequent - decrease threshold
                interval_adjustment = -0.05 * ((mean_interval / target_interval) - 1.0)

            # Intensity adjustment
            if mean_intensity < target_intensity * 0.7:
                # Too weak - decrease threshold
                intensity_adjustment = -0.03 * (1.0 - (mean_intensity / target_intensity))
            elif mean_intensity > target_intensity * 1.3:
                # Too strong - increase threshold
                intensity_adjustment = 0.03 * ((mean_intensity / target_intensity) - 1.0)

            # Calculate combined adjustment
            combined_adjustment = (interval_adjustment * 0.7) + (intensity_adjustment * 0.3)

            # Apply momentum to changes
            self.threshold_momentum = 0.9 * self.threshold_momentum + 0.1 * combined_adjustment

            # Update threshold with momentum and learning rate
            threshold_change = self.threshold_momentum * self.learning_rate
            self.threshold = max(self.min_threshold, min(self.max_threshold, self.threshold + threshold_change))

            # Add some small random walk for exploration
            random_adjust = 0.002 * (random.random() - 0.5)
            self.threshold += random_adjust

            # Record this threshold change
            if hasattr(self, 'threshold_changes'):
                self.threshold_changes.append({
                    'timestamp': time.time(),
                    'new_threshold': self.threshold,
                    'adjustment': combined_adjustment,
                    'momentum': self.threshold_momentum
                })

        except Exception as e:
            print(f"Error adapting threshold: {e}")

    def _update_field_state(self):
        """
        Update internal field state based on potentials and recent history with enhanced dynamics.
        """
        try:
            # Decrease cooldown if active
            if self.emergence_cooldown > 0:
                self.emergence_cooldown -= 1

                # Deactivate emergence when cooldown reaches 0
                if self.emergence_cooldown == 0 and self.emergence_active:
                    self.emergence_active = False

                    # Calculate duration of this emergence
                    if self.last_emergence:
                        duration = time.time() - self.last_emergence['timestamp']
                        self.emergence_duration_history.append(duration)

                        # Update performance metrics
                        if hasattr(self, 'performance'):
                            self.performance = {
                                'avg_duration': np.mean(self.emergence_duration_history),
                                'max_duration': max(self.emergence_duration_history),
                                'min_duration': min(self.emergence_duration_history)
                            }

                    print(f"EMERGENCE DEACTIVATED: Field now stable. Threshold at {self.threshold:.4f}")

            # Update field intensity based on potential with more dynamic range
            target_intensity = 1.0 + min(4.0, self.total_potential * 2.0)
            self.field_intensity = 0.9 * self.field_intensity + 0.1 * target_intensity

            # Update stability factor - higher potentials decrease stability
            target_stability = 1.0

            if self.total_potential > (self.threshold * 0.7):
                # Approaching threshold - decrease stability
                target_stability = max(0.3, 1.0 - (self.total_potential / self.threshold))
            else:
                # Below threshold - maintain high stability
                potential_ratio = self.total_potential / self.threshold
                target_stability = min(1.0, 0.8 + (1.0 - potential_ratio) * 0.2)

            # Apply adjustment with smoothing
            self.stability_factor = 0.95 * self.stability_factor + 0.05 * target_stability

            # Update learning rate based on system stability
            # More stable systems can learn faster
            target_learning_rate = min(0.05, max(0.001, self.stability_factor * 0.05))
            self.learning_rate = 0.95 * self.learning_rate + 0.05 * target_learning_rate

        except Exception as e:
            print(f"Error updating field state: {e}")

    def update_component_weights(self, component_type: str, weight_change: float):
        """
        Dynamically update the importance weights for different components.

        Args:
            component_type: Type of component to update
            weight_change: Change in weight (positive or negative)
        """
        if component_type in self.component_weights:
            current_weight = self.component_weights[component_type]
            new_weight = max(0.1, min(2.0, current_weight + weight_change))
            self.component_weights[component_type] = new_weight
            print(f"Updated component weight for {component_type}: {current_weight:.2f} -> {new_weight:.2f}")

    def set_system_state(self, state_metrics: Dict[str, float]):
        """
        Update system state metrics to influence emergence behavior.

        Args:
            state_metrics: Dictionary of system state metrics
        """
        try:
            if 'stability' in state_metrics:
                self.system_stability = state_metrics['stability']

            # Update probability modifiers
            if 'complexity' in state_metrics:
                self.probability_modifiers['system_complexity'] = 0.5 + state_metrics['complexity']

            if 'entropy' in state_metrics:
                self.probability_modifiers['environmental_noise'] = 0.5 + state_metrics['entropy']

            # Adapt threshold based on new state
            self._adapt_threshold_dynamic(state_metrics)

        except Exception as e:
            print(f"Error setting system state: {e}")

    def get_field_state(self):
        """
        Get the current state of the emergent potential field with enhanced metrics.

        Returns:
            Dictionary with field state metrics
        """
        try:
            current_time = time.time()

            # Calculate time since last emergence
            time_since_emergence = 0
            if self.last_emergence:
                time_since_emergence = current_time - self.last_emergence['timestamp']

            # Calculate average emergence duration
            avg_duration = 0
            if self.emergence_duration_history:
                avg_duration = sum(self.emergence_duration_history) / len(self.emergence_duration_history)

            # Calculate emergence probability with current system state
            emergence_probability = self._calculate_emergence_probability()

            # Calculate threshold trend
            threshold_trend = 0
            if len(self.threshold_history) > 5:
                recent_thresholds = list(self.threshold_history)[-5:]
                first_avg = sum(recent_thresholds[:2]) / 2
                last_avg = sum(recent_thresholds[-2:]) / 2
                threshold_trend = last_avg - first_avg

            # Prepare state summary
            state = {
                'total_potential': self.total_potential,
                'threshold': self.threshold,
                'threshold_trend': threshold_trend,
                'threshold_momentum': self.threshold_momentum,
                'emergence_active': self.emergence_active,
                'emergence_cooldown': self.emergence_cooldown,
                'stability_factor': self.stability_factor,
                'field_intensity': self.field_intensity,
                'emergence_count': self.emergence_counter,
                'time_since_emergence': time_since_emergence,
                'avg_emergence_duration': avg_duration,
                'component_count': len(self.component_potentials),
                'component_types': list(set(data['component_type'] for data in self.component_potentials.values())),
                'strongest_component': self._get_strongest_component(),
                'emergence_probability': emergence_probability,
                'probability_modifiers': self.probability_modifiers,
                'learning_rate': self.learning_rate,
                'system_stability': self.system_stability
            }

            return state

        except Exception as e:
            print(f"Error getting field state: {e}")
            return {
                'error': str(e),
                'total_potential': self.total_potential,
                'emergence_active': self.emergence_active,
                'stability_factor': self.stability_factor,
                'threshold': self.threshold
            }

    def _get_strongest_component(self):
        """Get the strongest contributing component to the potential field."""
        try:
            if not self.component_potentials:
                return None

            # Find component with highest weighted potential
            strongest = max(self.component_potentials.items(),
                          key=lambda x: x[1]['weighted_potential'])

            return {
                'id': strongest[0],
                'type': strongest[1]['component_type'],
                'potential': strongest[1]['weighted_potential'],
                'raw_potential': strongest[1]['raw_potential']
            }

        except Exception as e:
            print(f"Error finding strongest component: {e}")
            return None
