"""
Cognitive Structures Module for Émile-2 Simulation
--------------------------------------------------
Implements recursive cognitive structures that enable emergence through
adaptive layering and ontological modeling.
"""
import logging
import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.cognitive_structures")

# Import constants
from utilities import (
    TARGET_DISTINCTION,
    CORE_DISTINCTION_UPDATE_RATE,
    MOMENTUM_DECAY,
    DISTINCTION_ANCHOR_WEIGHT,
    COLLAPSE_DISSIPATION_THRESHOLD,
    COLLAPSE_DISSIPATION_RATE,
    ensure_real
)

# Constants for cognitive structure dynamics
STABILITY_DECAY_RATE = 0.99
COLLAPSE_RESISTANCE_BASE = 0.8
COLLAPSE_RESISTANCE_DECAY = 0.95
COLLAPSE_RESISTANCE_UPDATE_RATE = 0.05
MIN_COLLAPSE_RESISTANCE = 0.1
MAX_COLLAPSE_RESISTANCE = 2.0
MIN_STABILITY = 0.1
MAX_STABILITY = 1.0
MIN_ADAPTATION_RATE = 0.001
MAX_ADAPTATION_RATE = 0.1
SIGMOID_STEEPNESS = 5.0
SIGMOID_MIDPOINT = 0.5
STRENGTH_MOMENTUM_INFLUENCE = 0.1
MAX_HISTORY_LENGTH = 1000
STABILITY_WEIGHT = 0.6  # For weighted average in feedback calculation
COHERENCE_WEIGHT = 0.4  # For weighted average in feedback calculation


class RecursiveCognitiveStructuring:
    """
    Multi-layer cognitive structure with recursive updating and collapse prevention.

    Manages a hierarchical cognitive structure where each layer tracks attributes
    like strength, stability, quantum coupling, and core identity.
    Enhanced with adaptive feedback strength based on stability and coherence.
    """
    def __init__(self, num_layers: int = 3):
        """
        Initialize recursive cognitive structure with improved layer differentiation.

        Args:
            num_layers: Number of cognitive layers
        """
        try:
            # Move constants from utilities.py to instance variables for dynamic adjustment
            self.target_distinction = 0.7  # Default from TARGET_DISTINCTION
            self.core_distinction_update_rate = 0.01  # Default from CORE_DISTINCTION_UPDATE_RATE
            self.stability_decay = 0.99  # Default from STABILITY_DECAY_RATE
            self.distinction_anchor_weight = 0.2  # Default from DISTINCTION_ANCHOR_WEIGHT
            self.collapse_dissipation_threshold = 0.35  # Default from COLLAPSE_DISSIPATION_THRESHOLD
            self.collapse_dissipation_rate = 0.02  # Default from COLLAPSE_DISSIPATION_RATE

            # Additional constants moved from the class code to instance variables
            self.collapse_resistance_base = 0.8  # Default from COLLAPSE_RESISTANCE_BASE
            self.collapse_resistance_decay = 0.95  # Default from COLLAPSE_RESISTANCE_DECAY
            self.collapse_resistance_update_rate = 0.05  # Default from COLLAPSE_RESISTANCE_UPDATE_RATE
            self.min_collapse_resistance = 0.1  # Default from MIN_COLLAPSE_RESISTANCE
            self.max_collapse_resistance = 2.0  # Default from MAX_COLLAPSE_RESISTANCE
            self.stability_weight = 0.6  # Default from STABILITY_WEIGHT
            self.coherence_weight = 0.4  # Default from COHERENCE_WEIGHT

            # Add momentum values for dynamic constant adjustment
            self.threshold_momentum = 0.0
            self.rate_momentum = 0.0
            self.constants_adaptation_rate = 0.01

            # Track constant history
            self.constants_history = deque(maxlen=100)

            # Initialize layers with more differentiated characteristics
            self.layers = []
            for i in range(num_layers):
                # More varied initialization values to break symmetry and promote dynamics
                layer = {
                    'strength': 1.0 + 0.2 * i + 0.1 * np.random.random(),  # Increasing by depth with randomness
                    'stability': 0.3 + 0.1 * i + 0.05 * np.random.random(),  # Start with lower stability
                    'quantum_coupling': 0.3 + 0.2 * i + 0.05 * np.random.random(),  # More variance by layer
                    'adaptation_rate': 0.01 * (0.8 ** i) * (1.0 + 0.2 * np.random.random()),  # Decreasing with randomness
                    'base_feedback_strength': 0.3 + 0.2 * i + 0.1 * np.random.random(),  # Higher variability
                    'feedback_strength': 0.2 ** (num_layers - i - 1) * (1.0 + 0.1 * np.random.random()),
                    'collapse_resistance': 0.3 + 0.2 * i + 0.1 * np.random.random(),
                    'core_identity': 1.2 ** (num_layers - i - 1) * (1.0 + 0.05 * np.random.random())
                }
                self.layers.append(layer)

            # History tracking with limited length to avoid memory issues
            self.history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.adaptation_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.collapse_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.identity_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.distinction_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.feedback_history = deque(maxlen=MAX_HISTORY_LENGTH)

            # Initialize feedback matrix for asymmetric inter-layer feedback
            self.feedback_matrix = np.zeros((num_layers, num_layers))
            for i in range(num_layers):
                for j in range(num_layers):
                    if i != j:  # No self-feedback
                        # More varied initial feedback strengths
                        self.feedback_matrix[i, j] = 0.2 ** abs(i - j) * (1.0 + 0.1 * np.random.random())

            # Dynamic parameters
            self.coupling_momentum = 0.0
            self.feedback_momentum = np.zeros(num_layers)
            self.identity_momentum = 0.0
            self.identity_preservation_rate = self.core_distinction_update_rate
            self.adaptation_momentum = np.zeros(num_layers)

            # State tracking
            self.metrics = {
                'last_update_time': time.time(),
                'updates_count': 0,
                'mean_strength': 1.0,
                'mean_stability': 0.3,  # Start with lower stability to encourage dynamics
                'collapse_probability': 0.0,
                'mean_feedback': 0.0
            }

            # Record initial constants state
            self._record_constants_state("initialization")

            logger.info(f"Initialized RecursiveCognitiveStructuring with {num_layers} layers")

        except Exception as e:
            logger.error(f"Error initializing RecursiveCognitiveStructuring: {e}")
            # Create minimal fallback structure
            self.layers = [{
                'strength': 1.0,
                'stability': 0.3,
                'quantum_coupling': 0.3,
                'adaptation_rate': 0.01,
                'base_feedback_strength': 0.5,
                'feedback_strength': 0.2,
                'collapse_resistance': 0.3,
                'core_identity': 1.2
            }]
            self.history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.adaptation_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.collapse_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.identity_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.distinction_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.feedback_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.feedback_matrix = np.zeros((1, 1))
            self.collapse_threshold = 0.35
            self.coupling_momentum = 0.0
            self.feedback_momentum = np.zeros(1)
            self.adaptation_momentum = np.zeros(1)

            # Initialize dynamic constants with default values
            self.target_distinction = 0.7
            self.core_distinction_update_rate = 0.01
            self.stability_decay = 0.99
            self.distinction_anchor_weight = 0.2
            self.collapse_dissipation_threshold = 0.35
            self.collapse_dissipation_rate = 0.02
            self.collapse_resistance_base = 0.8
            self.collapse_resistance_decay = 0.95
            self.collapse_resistance_update_rate = 0.05
            self.min_collapse_resistance = 0.1
            self.max_collapse_resistance = 2.0
            self.stability_weight = 0.6
            self.coherence_weight = 0.4

    def _record_constants_state(self, event_type: str) -> None:
        """Record the current state of all dynamic constants for tracking changes over time."""
        try:
            constants_state = {
                'target_distinction': self.target_distinction,
                'core_distinction_update_rate': self.core_distinction_update_rate,
                'stability_decay': self.stability_decay,
                'distinction_anchor_weight': self.distinction_anchor_weight,
                'collapse_dissipation_threshold': self.collapse_dissipation_threshold,
                'collapse_dissipation_rate': self.collapse_dissipation_rate,
                'collapse_resistance_base': self.collapse_resistance_base,
                'collapse_resistance_decay': self.collapse_resistance_decay,
                'collapse_resistance_update_rate': self.collapse_resistance_update_rate,
                'stability_weight': self.stability_weight,
                'coherence_weight': self.coherence_weight,
                'event_type': event_type,
                'timestamp': time.time()
            }
            self.constants_history.append(constants_state)
        except Exception as e:
            logger.error(f"Error recording constants state: {e}")

    def update_dynamic_constants(self,
                              phase_coherence: float,
                              distinction_level: float,
                              prediction_error: float,
                              mean_stability: float) -> None:
        """
        Update dynamic constants based on system performance and behavior.

        Args:
            phase_coherence: Current phase coherence
            distinction_level: Current distinction level
            prediction_error: Current prediction error
            mean_stability: Mean stability across layers
        """
        try:
            # Store previous values for tracking changes
            prev_threshold = self.collapse_dissipation_threshold
            prev_update_rate = self.core_distinction_update_rate

            # 1. Update collapse dissipation threshold based on mean stability
            # If system is very stable, can raise the threshold (less likely to trigger dissipation)
            if mean_stability > 0.7 and self.metrics['collapse_probability'] < 0.2:
                threshold_adjustment = 0.01 * self.constants_adaptation_rate * (mean_stability - 0.5)

                # Update momentum
                self.threshold_momentum = update_momentum(
                    self.threshold_momentum,
                    threshold_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.collapse_dissipation_threshold += threshold_adjustment + 0.1 * self.threshold_momentum
                self.collapse_dissipation_threshold = np.clip(self.collapse_dissipation_threshold, 0.25, 0.5)

                logger.info(f"Increased collapse_dissipation_threshold to {self.collapse_dissipation_threshold:.4f}")

            # If system is unstable, lower the threshold (more likely to trigger dissipation)
            elif mean_stability < 0.4 or self.metrics['collapse_probability'] > 0.6:
                threshold_adjustment = -0.01 * self.constants_adaptation_rate

                # Update momentum
                self.threshold_momentum = update_momentum(
                    self.threshold_momentum,
                    threshold_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.collapse_dissipation_threshold += threshold_adjustment + 0.1 * self.threshold_momentum
                self.collapse_dissipation_threshold = np.clip(self.collapse_dissipation_threshold, 0.25, 0.5)

                logger.info(f"Decreased collapse_dissipation_threshold to {self.collapse_dissipation_threshold:.4f}")

            # 2. Update core distinction update rate based on distinction level and prediction error
            # When prediction error is high, faster updates may be needed
            if prediction_error > 0.5 and phase_coherence > 0.6:
                rate_adjustment = 0.001 * self.constants_adaptation_rate * prediction_error

                # Update momentum
                self.rate_momentum = update_momentum(
                    self.rate_momentum,
                    rate_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.core_distinction_update_rate += rate_adjustment + 0.05 * self.rate_momentum
                self.core_distinction_update_rate = np.clip(self.core_distinction_update_rate, 0.005, 0.05)

                logger.info(f"Increased core_distinction_update_rate to {self.core_distinction_update_rate:.4f}")

            # When system is working well, can use slower updates
            elif prediction_error < 0.2 and mean_stability > 0.7:
                rate_adjustment = -0.001 * self.constants_adaptation_rate

                # Update momentum
                self.rate_momentum = update_momentum(
                    self.rate_momentum,
                    rate_adjustment,
                    decay=self.stability_decay
                )

                # Apply adjustment with momentum
                self.core_distinction_update_rate += rate_adjustment + 0.05 * self.rate_momentum
                self.core_distinction_update_rate = np.clip(self.core_distinction_update_rate, 0.005, 0.05)

                logger.info(f"Decreased core_distinction_update_rate to {self.core_distinction_update_rate:.4f}")

            # 3. Update target distinction based on current distinction level
            distinction_delta = distinction_level - self.target_distinction
            if abs(distinction_delta) > 0.2 and mean_stability > 0.6:
                # Move target distinction toward actual distinction if stable
                self.target_distinction += 0.01 * self.constants_adaptation_rate * distinction_delta
                self.target_distinction = np.clip(self.target_distinction, 0.3, 0.9)

                logger.info(f"Adjusted target_distinction to {self.target_distinction:.4f}")

            # 4. Update stability and coherence weights based on which is more effective
            # Compare recent stability-driven vs. coherence-driven adaptation success
            # This would require more detailed tracking, simplified version here:
            if phase_coherence > mean_stability and prediction_error < 0.3:
                # Coherence seems more reliable
                self.coherence_weight = min(0.6, self.coherence_weight + 0.01)
                self.stability_weight = 1.0 - self.coherence_weight
                logger.info(f"Adjusted weights: coherence={self.coherence_weight:.2f}, stability={self.stability_weight:.2f}")
            elif mean_stability > phase_coherence and prediction_error < 0.3:
                # Stability seems more reliable
                self.stability_weight = min(0.8, self.stability_weight + 0.01)
                self.coherence_weight = 1.0 - self.stability_weight
                logger.info(f"Adjusted weights: coherence={self.coherence_weight:.2f}, stability={self.stability_weight:.2f}")

            # Record significant constant changes
            if (abs(self.collapse_dissipation_threshold - prev_threshold) > 0.02 or
                abs(self.core_distinction_update_rate - prev_update_rate) > 0.002):
                self._record_constants_state("significant_update")

            # Make sure identity_preservation_rate is kept in sync with core_distinction_update_rate
            self.identity_preservation_rate = self.core_distinction_update_rate

        except Exception as e:
            logger.error(f"Error updating cognitive structure dynamic constants: {e}")


    def _sigmoid(self, x: float) -> float:
        """
        Helper function to calculate sigmoid for smooth transitions.

        Args:
            x: Input value

        Returns:
            Sigmoid of input value
        """
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except OverflowError:
            # Handle extreme values
            return 0.0 if x < 0 else 1.0
        except Exception as e:
            logger.error(f"Error in sigmoid calculation: {e}")
            return 0.5  # Safe default

    def update(self, phase_coherence: float, distinction_level: float,
           surplus: Dict[str, float], prediction_error: float,
           quantum_metrics: Dict[str, float]) -> bool:
        """
        Update the recursive cognitive structure based on current metrics.
        Includes adaptive feedback strength based on stability and coherence.

        Args:
            phase_coherence: Quantum phase coherence
            distinction_level: Current distinction level
            surplus: Dictionary of surplus values
            prediction_error: Error in prediction
            quantum_metrics: Dictionary of quantum metrics

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate inputs
            if not self._validate_cognitive_update(phase_coherence, distinction_level, surplus, prediction_error):
                logger.warning("Invalid inputs for cognitive update")
                return False

            # Apply safe casting to float values
            phase_coherence = ensure_real(phase_coherence, 0.5)
            distinction_level = ensure_real(distinction_level, 0.5)
            prediction_error = ensure_real(prediction_error, 0.0)

            # Extract additional metrics for enhanced behavior
            normalized_entropy = quantum_metrics.get('normalized_entropy', 0.5)
            phase_distinction = quantum_metrics.get('phase_distinction', 0.0)
            stability = quantum_metrics.get('stability', 1.0)
            quantum_coupling = quantum_metrics.get('quantum_coupling', 1.0)
            quantum_surplus_coupling = quantum_metrics.get('quantum_surplus_coupling', 1.0)

            # Compute base adaptation strength with enhanced metrics
            base_strength = phase_coherence * distinction_level * (1.0 - normalized_entropy)

            # Get quantum influence factor - enhanced with multiple metrics
            quantum_factor = (1.0 - normalized_entropy) * (0.5 + 0.5 * quantum_coupling)

            # Update coupling momentum
            self.coupling_momentum = MOMENTUM_DECAY * self.coupling_momentum + (1 - MOMENTUM_DECAY) * (quantum_factor - 0.5)

            # Calculate adaptation strength with quantum influence
            adaptation_strength = (
                base_strength *
                (1.0 + quantum_factor) *
                (1.0 + 0.5 * phase_coherence) *
                (1.0 + self.coupling_momentum)
            )

            # Calculate per-layer stability and update the feedback matrix
            self._update_feedback_matrix(phase_coherence, quantum_metrics)

            # Update each layer
            for i, layer in enumerate(self.layers):
                # Calculate depth factor (deeper layers update more slowly)
                depth_factor = 0.7 ** i

                # Enhanced strength update using more metrics
                strength_update = (
                    prediction_error *
                    (1 - distinction_level) *
                    phase_coherence *
                    depth_factor *
                    layer.get('collapse_resistance', 0.3) *
                    (1.0 + 0.1 * phase_distinction)  # Add phase distinction influence
                )

                # Update adaptation momentum
                self.adaptation_momentum[i] = (
                    MOMENTUM_DECAY * self.adaptation_momentum[i] +
                    (1 - MOMENTUM_DECAY) * strength_update
                )

                # Update strength with momentum influence
                layer['strength'] += (
                    strength_update * adaptation_strength +
                    STRENGTH_MOMENTUM_INFLUENCE * self.adaptation_momentum[i]
                )

                # Enhanced stability update with quantum metrics influence
                stability_update = (
                    phase_coherence * 0.01 +
                    (1 - prediction_error) * 0.01 +
                    stability * 0.005  # Add overall stability influence
                )

                layer['stability'] = np.clip(
                    layer['stability'] * self.stability_decay +
                    stability_update * (phase_coherence * depth_factor),
                    MIN_STABILITY, MAX_STABILITY
                )

                # Enhanced quantum coupling update
                coupling_update = quantum_metrics.get('phase_coherence', 0.5) * (1 - quantum_metrics.get('normalized_entropy', 0.5))
                layer['quantum_coupling'] = 0.9 * layer['quantum_coupling'] + 0.1 * coupling_update

                # Update adaptation rate with quantum surplus coupling
                layer['adaptation_rate'] = np.clip(
                    layer['adaptation_rate'] * (1.0 + 0.1 * (quantum_surplus_coupling - 0.5)),
                    MIN_ADAPTATION_RATE, MAX_ADAPTATION_RATE
                )

                # === Enhanced Adaptive Feedback Strength (now per-layer) ===
                # Use weighted average for combined metric with more inputs
                combined_metric = (
                    self.stability_weight * layer['stability'] +
                    self.coherence_weight * phase_coherence +
                    0.1 * (1.0 - normalized_entropy)  # Add entropy influence
                )

                # Scale and shift the sigmoid for smooth adaptation
                adaptive_feedback = layer['base_feedback_strength'] * self._sigmoid(
                    SIGMOID_STEEPNESS * (combined_metric - SIGMOID_MIDPOINT)
                )

                layer['feedback_strength'] = np.clip(adaptive_feedback, 0.1, 1.0)

                # Update feedback momentum - used for tracking, not for computation
                self.feedback_momentum[i] = (
                    MOMENTUM_DECAY * self.feedback_momentum[i] +
                    (1 - MOMENTUM_DECAY) * strength_update
                )

                # Enhanced collapse resistance update
                collapse_update = (
                    self.collapse_resistance_base *
                    (1.0 + layer['stability']) *
                    (1.0 + layer['quantum_coupling']) *
                    (1.0 + 0.1 * quantum_surplus_coupling)  # Add quantum surplus coupling influence
                )

                layer['collapse_resistance'] = np.clip(
                   layer['collapse_resistance'] * self.collapse_resistance_decay +
                   self.collapse_resistance_update_rate * collapse_update,
                   self.min_collapse_resistance, self.max_collapse_resistance
                )

                # Enhanced core identity update with inter-layer feedback
                identity_update = (
                    distinction_level *
                    layer['stability'] *
                    layer['quantum_coupling'] *
                    (1.0 + 0.1 * phase_distinction)  # Add phase distinction influence
                )

                # Enhanced core identity update with inter-layer feedback
                core_identity_update = identity_update
                for j, other_layer in enumerate(self.layers):
                    if i != j:  # Don't provide feedback to itself
                        # Get feedback strength from matrix (asymmetric)
                        feedback_strength = self.feedback_matrix[j, i]
                        # Add feedback contribution
                        core_identity_update += (
                            other_layer['strength'] *
                            other_layer['stability'] *
                            feedback_strength *
                            0.01  # Small weight factor to prevent overinfluence
                        )

                # Apply the core identity update - using dynamic update rate
                layer['core_identity'] = (
                    (1 - self.identity_preservation_rate) * layer['core_identity'] +
                    self.identity_preservation_rate * core_identity_update
                )

            # Track layer history
            self.history.append([layer.copy() for layer in self.layers])

            # Track identity history
            self.identity_history.append([layer['core_identity'] for layer in self.layers])

            # Track feedback history
            self.feedback_history.append({
                'mean_feedback': np.mean([layer['feedback_strength'] for layer in self.layers]),
                'feedback_matrix': self.feedback_matrix.copy(),
                'timestamp': time.time()
            })

            # Update metrics
            self.metrics.update({
                'last_update_time': time.time(),
                'updates_count': self.metrics['updates_count'] + 1,
                'mean_strength': np.mean([layer['strength'] for layer in self.layers]),
                'mean_stability': np.mean([layer['stability'] for layer in self.layers]),
                'mean_feedback': np.mean([layer['feedback_strength'] for layer in self.layers]),
                'quantum_influence': quantum_factor
            })

            # Track distinction level
            self.track_history(distinction_level)

            # Update dynamic constants
            self.update_dynamic_constants(
                phase_coherence=phase_coherence,
                distinction_level=distinction_level,
                prediction_error=prediction_error,
                mean_stability=self.metrics['mean_stability']
            )

            return True

        except Exception as e:
            logger.error(f"Error in cognitive update: {e}")
            return False

    def get_constants_history(self) -> List[Dict]:
       """
       Get history of dynamic constant changes.

       Returns:
           List of constant state snapshots with timestamps
       """
       try:
           return list(self.constants_history)
       except Exception as e:
           logger.error(f"Error getting constants history: {e}")
           return []

    def get_dynamic_constants(self) -> Dict[str, float]:
       """
       Return current values of all dynamic constants.

       Returns:
           Dictionary with current constant values
       """
       try:
           return {
               'target_distinction': self.target_distinction,
               'core_distinction_update_rate': self.core_distinction_update_rate,
               'stability_decay': self.stability_decay,
               'distinction_anchor_weight': self.distinction_anchor_weight,
               'collapse_dissipation_threshold': self.collapse_dissipation_threshold,
               'collapse_dissipation_rate': self.collapse_dissipation_rate,
               'collapse_resistance_base': self.collapse_resistance_base,
               'collapse_resistance_decay': self.collapse_resistance_decay,
               'collapse_resistance_update_rate': self.collapse_resistance_update_rate,
               'identity_preservation_rate': self.identity_preservation_rate,
               'stability_weight': self.stability_weight,
               'coherence_weight': self.coherence_weight
           }
       except Exception as e:
           logger.error(f"Error getting dynamic constants: {e}")
           return {}

    def _update_feedback_matrix(self, phase_coherence: float, quantum_metrics: Dict[str, float]) -> None:
        """
        Update the feedback matrix for asymmetric inter-layer feedback.

        Args:
            phase_coherence: Current phase coherence
            quantum_metrics: Dictionary of quantum metrics
        """
        try:
            num_layers = len(self.layers)

            # Calculate a global coherence factor
            coherence_factor = phase_coherence * (1.0 - quantum_metrics.get('normalized_entropy', 0.5))

            # Update each pair's feedback strength
            for i in range(num_layers):
                for j in range(num_layers):
                    if i != j:  # No self-feedback
                        # Get source and target layer stability
                        source_stability = self.layers[i]['stability']
                        target_stability = self.layers[j]['stability']

                        # Calculate combined metric with weighted stability and distance factor
                        distance_factor = 1.0 / (1.0 + abs(i - j))
                        combined_metric = (
                            STABILITY_WEIGHT * (source_stability * target_stability) +
                            COHERENCE_WEIGHT * coherence_factor
                        ) * distance_factor

                        # Update matrix value with sigmoid
                        self.feedback_matrix[i, j] = self.layers[i]['base_feedback_strength'] * self._sigmoid(
                            SIGMOID_STEEPNESS * (combined_metric - SIGMOID_MIDPOINT)
                        )

        except Exception as e:
            logger.error(f"Error updating feedback matrix: {e}")

    def predict_cognitive_collapse(self) -> float:
        """
        Predict the probability of cognitive collapse.

        Returns:
            Collapse probability (0.0 to 1.0)
        """
        try:
            if not self.layers:
                return 0.0

            # Calculate collapse factors from all layers
            avg_strength = np.mean([layer['strength'] * layer.get('collapse_resistance', 0.3) for layer in self.layers])
            avg_stability = np.mean([layer['stability'] for layer in self.layers])
            avg_coupling = np.mean([layer['quantum_coupling'] for layer in self.layers])

            # Calculate individual collapse factors
            stability_factor = 1.0 - avg_stability
            strength_factor = 1.0 - avg_strength
            coupling_factor = 1.0 - avg_coupling

            # Weighted combination of factors
            collapse_prob = (
                0.3 * stability_factor +
                0.2 * strength_factor +
                0.2 * coupling_factor
            )

            # Scale by collapse resistance
            collapse_prob *= (1.0 - self.collapse_resistance_base)

            # Ensure probability is in valid range
            collapse_prob = float(np.clip(collapse_prob, 0, 1))

            # Track history
            self.collapse_history.append(collapse_prob)

            # Update metrics
            self.metrics['collapse_probability'] = collapse_prob

            return collapse_prob

        except Exception as e:
            logger.error(f"Error predicting collapse: {e}")
            return 0.0

    def get_cognitive_state(self) -> Dict[str, float]:
        """
        Get the current cognitive state summary.

        Returns:
            Dictionary of cognitive state metrics
        """
        try:
            # Calculate mean values across layers
            avg_strength = np.mean([layer['strength'] for layer in self.layers])
            avg_stability = np.mean([layer['stability'] for layer in self.layers])
            avg_coupling = np.mean([layer['quantum_coupling'] for layer in self.layers])
            avg_identity = np.mean([layer['core_identity'] for layer in self.layers])
            avg_resistance = np.mean([layer.get('collapse_resistance', 0.3) for layer in self.layers])
            avg_feedback = np.mean([layer['feedback_strength'] for layer in self.layers])

            # Calculate feedback matrix metrics
            feedback_matrix_avg = np.mean(self.feedback_matrix)
            feedback_matrix_max = np.max(self.feedback_matrix)
            feedback_matrix_min = np.min(self.feedback_matrix[self.feedback_matrix > 0])  # Min of non-zero elements

            # Generate comprehensive state summary
            state = {
                'mean_strength': float(avg_strength),
                'mean_stability': float(avg_stability),
                'mean_coupling': float(avg_coupling),
                'mean_identity': float(avg_identity),
                'mean_collapse_resistance': float(avg_resistance),
                'mean_feedback_strength': float(avg_feedback),
                'feedback_matrix_avg': float(feedback_matrix_avg),
                'feedback_matrix_max': float(feedback_matrix_max),
                'feedback_matrix_min': float(feedback_matrix_min),
                'collapse_probability': float(self.predict_cognitive_collapse()),
                'quantum_influence': float(avg_coupling * avg_stability),
                'adaptation_momentum': float(np.mean(self.feedback_momentum)),
                'coupling_momentum': float(self.coupling_momentum),
                'identity_coherence': float(np.std([layer['core_identity'] for layer in self.layers]))
            }

            # Add per-layer information
            for i, layer in enumerate(self.layers):
                state[f'layer_{i}_strength'] = float(layer['strength'])
                state[f'layer_{i}_stability'] = float(layer['stability'])
                state[f'layer_{i}_coupling'] = float(layer['quantum_coupling'])
                state[f'layer_{i}_identity'] = float(layer['core_identity'])
                state[f'layer_{i}_resistance'] = float(layer.get('collapse_resistance', 0.3))
                state[f'layer_{i}_feedback'] = float(layer['feedback_strength'])

            return state

        except Exception as e:
            logger.error(f"Error getting cognitive state: {e}")
            # Return minimal state info
            return {
                'mean_strength': 1.0,
                'mean_stability': 0.1,
                'mean_coupling': 0.3,
                'collapse_probability': 0.0,
                'quantum_influence': 0.3
            }

    def dissipate_collapse(self, surplus_values: Dict[str, float]) -> Dict[str, float]:
        """
        Dissipate cognitive collapse by recycling surplus.

        Args:
            surplus_values: Dictionary of surplus values

        Returns:
            Dictionary of recycled surplus values
        """
        try:
            # Calculate collapse probability
            collapse_prob = self.predict_cognitive_collapse()

            # Check if collapse prevention is needed
            if collapse_prob < self.collapse_threshold:
                return {}

            logger.info(f"Dissipating collapse: probability={collapse_prob:.4f}")

            # Calculate recycling fractions based on collapse probability
            recycle_fraction = COLLAPSE_DISSIPATION_RATE * (collapse_prob / self.collapse_threshold)

            # Initialize recycled surplus
            recycled = {}

            # Recycle surplus values
            for key, value in surplus_values.items():
                recycle_amount = value * recycle_fraction
                recycled[key] = recycle_amount

            # Update feedback for each layer
            for layer in self.layers:
                layer['collapse_resistance'] = min(
                    layer['collapse_resistance'] * (1.0 + 0.1 * recycle_fraction),
                    MAX_COLLAPSE_RESISTANCE
                )

            logger.info(f"Recycled surplus: {recycled}")
            return recycled

        except Exception as e:
            logger.error(f"Error dissipating collapse: {e}")
            return {}

    def _validate_cognitive_update(self, phase_coherence: float,
                                   distinction_level: float,
                                   surplus: Dict[str, float],
                                   prediction_error: float) -> bool:
        """
        Validate that the inputs for cognitive update are within expected ranges.

        Args:
            phase_coherence: Quantum phase coherence
            distinction_level: Current distinction level
            surplus: Dictionary of surplus values
            prediction_error: Error in prediction

        Returns:
            True if inputs are valid, False otherwise
        """
        try:
            # Validate value ranges
            if not 0 <= phase_coherence <= 1:
                logger.warning(f"Invalid phase coherence: {phase_coherence}")
                return False

            if not 0 <= distinction_level <= 1:
                logger.warning(f"Invalid distinction level: {distinction_level}")
                return False

            # Validate surplus keys
            required_keys = ['basal', 'cognitive', 'predictive', 'ontological']
            if not all(key in surplus for key in required_keys):
                logger.warning(f"Missing surplus keys: {surplus.keys()}")
                return False

            # Validate prediction error
            if not isinstance(prediction_error, (int, float)) or prediction_error < 0:
                logger.warning(f"Invalid prediction error: {prediction_error}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating cognitive update: {e}")
            return False

    def _update_stability_metrics(self) -> None:
        """Update cognitive stability metrics from recent adaptation history."""
        try:
            if not hasattr(self, 'stability_metrics'):
                self.stability_metrics = {
                    'collapse_probability': 0.0,
                    'mean_stability': 0.1,
                    'adaptation_efficiency': 1.0
                }

            # Update stability metrics
            stabilities = [layer['stability'] for layer in self.layers]
            self.stability_metrics['mean_stability'] = float(np.mean(stabilities))

            # Update adaptation efficiency
            recent_changes = list(self.adaptation_history)[-10:] if self.adaptation_history else []
            if recent_changes:
                efficiency = np.mean([change.get('success', 0.0) for change in recent_changes])
                self.stability_metrics['adaptation_efficiency'] = float(efficiency)

        except Exception as e:
            logger.error(f"Error updating stability metrics: {e}")

    def track_history(self, distinction_level: float) -> None:
        """
        Track historical distinction levels.

        Args:
            distinction_level: Current distinction level to track
        """
        try:
            if not self.layers:
                logger.warning("No layers available to track distinction history")
                return

            # Validate input
            if not 0 <= distinction_level <= 1:
                logger.warning(f"Invalid distinction level for tracking: {distinction_level}")
                distinction_level = min(max(distinction_level, 0.0), 1.0)

            # Append to history
            self.distinction_history.append(float(distinction_level))

        except Exception as e:
            logger.error(f"Error tracking distinction history: {e}")

    def get_layer(self, index: int) -> Dict[str, float]:
        """
        Get layer information by index.

        Args:
            index: Layer index (0 to num_layers-1)

        Returns:
            Layer dictionary or empty dict if index invalid
        """
        try:
            if 0 <= index < len(self.layers):
                return self.layers[index].copy()
            else:
                logger.warning(f"Invalid layer index: {index}")
                return {}

        except Exception as e:
            logger.error(f"Error getting layer: {e}")
            return {}

    def get_historical_metrics(self) -> Dict[str, List]:
        """
        Get historical metrics for analysis.

        Returns:
            Dictionary of historical metric lists
        """
        try:
            # Extract relevant history for analysis
            history_metrics = {}

            # Get strength history
            if self.history:
                strength_history = [
                    np.mean([layer['strength'] for layer in snapshot])
                    for snapshot in self.history
                ]
                history_metrics['strength'] = list(strength_history)

            # Get stability history
            if self.history:
                stability_history = [
                    np.mean([layer['stability'] for layer in snapshot])
                    for snapshot in self.history
                ]
                history_metrics['stability'] = list(stability_history)

            # Get collapse history
            if self.collapse_history:
                history_metrics['collapse_probability'] = list(self.collapse_history)

            # Get distinction history
            if self.distinction_history:
                history_metrics['distinction'] = list(self.distinction_history)

            # Get feedback strength history
            if self.feedback_history:
                history_metrics['feedback_strength'] = [entry['mean_feedback'] for entry in self.feedback_history]

            # Get inter-layer feedback metrics
            if self.feedback_history:
                # Calculate average feedback matrix similarity (as a measure of asymmetry)
                matrix_symmetry = []
                for entry in self.feedback_history:
                    if 'feedback_matrix' in entry:
                        matrix = entry['feedback_matrix']
                        # Calculate symmetry as similarity between matrix and its transpose
                        diff = np.abs(matrix - matrix.T)
                        symmetry = 1.0 - (np.sum(diff) / (np.sum(matrix) + 1e-10))
                        matrix_symmetry.append(symmetry)

                if matrix_symmetry:
                    history_metrics['feedback_matrix_symmetry'] = matrix_symmetry

            return history_metrics

        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return {}

    def reset_to_baseline(self) -> bool:
        """
        Reset cognitive structure to baseline state.

        Returns:
            True if reset successful, False otherwise
        """
        try:
            logger.info("Resetting cognitive structure to baseline")

            # Reset each layer with original initialization pattern
            num_layers = len(self.layers)
            for i, layer in enumerate(self.layers):
                layer['strength'] = 1.0 + 0.2*i  # Increasing by depth
                layer['stability'] = 0.1 + 0.05 * i  # Increasing by depth
                layer['quantum_coupling'] = 0.3 + 0.1 * i  # Increasing by depth
                layer['adaptation_rate'] = 0.01 * (0.8 ** i)  # Decreasing by depth
                layer['base_feedback_strength'] = 0.5  # Base value
                layer['feedback_strength'] = 0.2 ** (num_layers - i - 1)  # Initial value
                layer['collapse_resistance'] = 0.3 + 0.2 * i
                layer['core_identity'] = 1.2 ** (num_layers - i - 1)  # Increasing by depth

            # Reset momentum
            self.coupling_momentum = 0.0
            self.feedback_momentum = np.zeros(len(self.layers))
            self.identity_momentum = 0.0
            self.adaptation_momentum = np.zeros(len(self.layers))

            # Reset feedback matrix
            for i in range(num_layers):
                for j in range(num_layers):
                    if i != j:  # No self-feedback
                        self.feedback_matrix[i, j] = 0.2 ** abs(i - j)  # Decaying with distance

            # Track reset event
            self.adaptation_history.append({
                'event': 'reset',
                'timestamp': time.time()
            })

            logger.info("Reset complete")
            return True

        except Exception as e:
            logger.error(f"Error resetting cognitive structure: {e}")
            return False

    def get_feedback_matrix_visualization(self) -> Dict[str, Any]:
        """
        Prepare feedback matrix data for visualization.

        Returns:
            Dictionary with matrix data and metadata
        """
        try:
            # Get current matrix
            matrix_data = self.feedback_matrix.tolist()

            # Get layer properties for context
            layer_stability = [layer['stability'] for layer in self.layers]
            layer_strength = [layer['strength'] for layer in self.layers]

            # Calculate properties
            symmetry = 1.0 - (np.sum(np.abs(self.feedback_matrix - self.feedback_matrix.T)) /
                             (np.sum(self.feedback_matrix) + 1e-10))

            feedback_influence = []
            for i in range(len(self.layers)):
                # How much this layer influences others
                outgoing = np.sum(self.feedback_matrix[i, :])
                # How much this layer is influenced by others
                incoming = np.sum(self.feedback_matrix[:, i])
                feedback_influence.append({
                    'layer': i,
                    'outgoing': float(outgoing),
                    'incoming': float(incoming),
                    'net': float(outgoing - incoming)
                })

            return {
                'matrix': matrix_data,
                'symmetry': float(symmetry),
                'layer_stability': layer_stability,
                'layer_strength': layer_strength,
                'feedback_influence': feedback_influence,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error preparing feedback matrix visualization: {e}")
            return {
                'error': str(e),
                'matrix': [[0.0]]
            }

    def _update_metrics(self) -> None:
        """Update directionality and strength metrics based on the causality matrix."""
        try:
            self.directionality = float(np.mean(np.abs(np.diff(self.causality_matrix, axis=0))))
            self.strength = float(np.mean(np.abs(self.causality_matrix)))
        except Exception as e:
            print(f"Error updating metrics: {e}")
            self.directionality = 0.0
            self.strength = 0.0

# For testing the module
if __name__ == "__main__":
    import random

    # Create recursive cognitive structure
    cognitive = RecursiveCognitiveStructuring(num_layers=3)

    # Run a series of updates with random values
    print("Running cognitive updates...")
    for i in range(10):
        # Generate random test values
        phase_coherence = random.uniform(0.3, 0.9)
        distinction_level = random.uniform(0.4, 0.8)
        surplus = {
            'basal': random.uniform(0.5, 1.5),
            'cognitive': random.uniform(0.5, 1.5),
            'predictive': random.uniform(0.5, 1.5),
            'ontological': random.uniform(0.5, 1.5)
        }
        prediction_error = random.uniform(0.0, 0.3)
        quantum_metrics = {
            'phase_coherence': phase_coherence,
            'normalized_entropy': random.uniform(0.1, 0.5)
        }

        # Update cognitive structure
        cognitive.update(phase_coherence, distinction_level, surplus, prediction_error, quantum_metrics)

        # Get and print cognitive state
        state = cognitive.get_cognitive_state()
        print(f"\nUpdate {i+1}:")
        print(f"  Strength: {state['mean_strength']:.4f}")
        print(f"  Stability: {state['mean_stability']:.4f}")
        print(f"  Collapse Probability: {state['collapse_probability']:.4f}")

        # Check for collapse prevention
        if state['collapse_probability'] > 0.5:
            print("  Dissipating collapse...")
            recycled = cognitive.dissipate_collapse(surplus)
            print(f"  Recycled surplus: {recycled}")


