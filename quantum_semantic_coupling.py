"""
Quantum-Semantic Coupling Module for Émile-4 Simulation
------------------------------------------------------
This module handles the integration between quantum phase processing and
semantic expression generation, creating a bidirectional feedback loop.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.quantum_semantic_coupling")

class QuantumSemanticCoupling:
    """
    Manages bidirectional coupling between quantum phase processing and
    semantic expression, allowing the quantum state to influence expression
    and expression coherence to feed back into quantum processing.
    """
    def __init__(self):
        """Initialize quantum-semantic coupling system."""
        self.coupling_strength = 0.3  # How strongly quantum state influences semantics
        self.feedback_strength = 0.2  # How strongly semantics influence quantum state

        # Tracking parameters
        self.coherence_coupling = 0.0
        self.phase_coupling = 0.0
        self.distinction_coupling = 0.0

        # Momentum tracking
        self.coherence_momentum = 0.0
        self.phase_momentum = 0.0
        self.distinction_momentum = 0.0

        # History tracking
        self.coupling_history = deque(maxlen=100)
        self.feedback_history = deque(maxlen=100)

        # Adaptation parameters
        self.adaptation_rate = 0.05
        self.min_coupling = 0.1
        self.max_coupling = 0.5

        logger.info("Quantum-Semantic coupling initialized")

    def calculate_semantic_coherence(self,
                                    phase_coherence: float,
                                    quantum_metrics: Dict[str, float]) -> float:
        """
        Calculate semantic coherence based on quantum metrics

        Args:
            phase_coherence: Current phase coherence
            quantum_metrics: Dictionary of quantum state metrics

        Returns:
            Semantic coherence value
        """
        try:
            # Extract relevant metrics with defaults
            normalized_entropy = quantum_metrics.get('normalized_entropy', 0.5)
            phase = quantum_metrics.get('phase', 0.0)
            phase_distinction = quantum_metrics.get('phase_distinction', 0.5)

            # Calculate base semantic coherence
            # High coherence, low entropy leads to higher semantic coherence
            base_coherence = phase_coherence * (1.0 - normalized_entropy)

            # Apply phase influence
            phase_factor = 0.5 + 0.5 * np.sin(phase)  # Oscillates between 0-1

            # Apply distinction coupling
            distinction_factor = 0.5 + 0.5 * phase_distinction

            # Calculate final coherence with coupling strengths
            semantic_coherence = (
                0.5 * base_coherence +
                0.25 * phase_factor * self.phase_coupling +
                0.25 * distinction_factor * self.distinction_coupling
            )

            # Ensure valid range
            semantic_coherence = np.clip(semantic_coherence, 0.0, 1.0)

            return float(semantic_coherence)

        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.5  # Default middle value

    def calculate_quantum_feedback(self,
                                 semantic_coherence: float,
                                 expression_components: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quantum feedback parameters based on semantic expression

        Args:
            semantic_coherence: Current semantic coherence value
            expression_components: Components of the semantic expression

        Returns:
            Dictionary of feedback parameters
        """
        try:
            # Initialize feedback parameters
            feedback = {
                'phase_adjustment': 0.0,
                'coherence_adjustment': 0.0,
                'distinction_adjustment': 0.0
            }

            # Skip feedback if expression components are missing
            if not expression_components:
                return feedback

            # Extract expression components
            descriptor = expression_components.get('descriptor', '')
            relation = expression_components.get('relation', '')
            concept = expression_components.get('concept', '')

            # Calculate descriptor-based phase adjustment
            # Different descriptors influence the quantum phase differently
            phase_map = {
                'Flux': 0.1,
                'Equilibrium': -0.1,
                'Distinction': 0.2,
                'Recursion': 0.15,
                'Convergence': -0.15,
                'Divergence': 0.25,
                'Resonance': -0.2,
                'Coherence': -0.1,
                'Entanglement': 0.3,
                'Superposition': 0.25,
                'Bifurcation': 0.2,
                'Integration': -0.15
            }

            # Calculate feedback based on expression components
            phase_adj = phase_map.get(descriptor, 0.0) * self.feedback_strength

            # Higher semantic coherence reinforces quantum coherence
            coherence_adj = (semantic_coherence - 0.5) * self.feedback_strength

            # Certain concepts influence distinction level
            distinction_map = {
                'distinction': 0.2,
                'emergence': 0.15,
                'complexity': 0.1,
                'recursion': 0.1,
                'entropy': -0.1,
                'stability': -0.15
            }

            # Check if any distinction-affecting concept is in the expression
            distinction_adj = 0.0
            for key, value in distinction_map.items():
                if key in concept.lower():
                    distinction_adj += value * self.feedback_strength

            # Limit adjustment ranges
            phase_adj = np.clip(phase_adj, -0.1, 0.1)
            coherence_adj = np.clip(coherence_adj, -0.05, 0.05)
            distinction_adj = np.clip(distinction_adj, -0.05, 0.05)

            # Update feedback parameters
            feedback['phase_adjustment'] = phase_adj
            feedback['coherence_adjustment'] = coherence_adj
            feedback['distinction_adjustment'] = distinction_adj

            # Record feedback
            self.feedback_history.append({
                'semantic_coherence': semantic_coherence,
                'phase_adjustment': phase_adj,
                'coherence_adjustment': coherence_adj,
                'distinction_adjustment': distinction_adj
            })

            return feedback

        except Exception as e:
            logger.error(f"Error calculating quantum feedback: {e}")
            return {
                'phase_adjustment': 0.0,
                'coherence_adjustment': 0.0,
                'distinction_adjustment': 0.0
            }

    def update_coupling_parameters(self, phase_coherence: float, distinction_level: float):
        """
        Update coupling parameters based on system state

        Args:
            phase_coherence: Current phase coherence
            distinction_level: Current distinction level
        """
        try:
            # Update coherence coupling
            target_coherence_coupling = np.clip(phase_coherence,
                                              self.min_coupling,
                                              self.max_coupling)
            self.coherence_coupling = (
                0.9 * self.coherence_coupling +
                0.1 * target_coherence_coupling
            )

            # Update distinction coupling
            target_distinction_coupling = np.clip(distinction_level,
                                                self.min_coupling,
                                                self.max_coupling)
            self.distinction_coupling = (
                0.9 * self.distinction_coupling +
                0.1 * target_distinction_coupling
            )

            # Update phase coupling (tends toward middle value)
            # This creates a balanced influence of phase on semantics
            target_phase_coupling = 0.3
            self.phase_coupling = (
                0.9 * self.phase_coupling +
                0.1 * target_phase_coupling
            )

            # Update momenta
            self.coherence_momentum = 0.9 * self.coherence_momentum + 0.1 * (target_coherence_coupling - self.coherence_coupling)
            self.distinction_momentum = 0.9 * self.distinction_momentum + 0.1 * (target_distinction_coupling - self.distinction_coupling)

            # Record coupling state
            self.coupling_history.append({
                'coherence_coupling': self.coherence_coupling,
                'distinction_coupling': self.distinction_coupling,
                'phase_coupling': self.phase_coupling,
                'coherence_momentum': self.coherence_momentum,
                'distinction_momentum': self.distinction_momentum
            })

        except Exception as e:
            logger.error(f"Error updating coupling parameters: {e}")

    def get_coupling_metrics(self) -> Dict[str, float]:
        """
        Get current coupling metrics.

        Returns:
            Dictionary of coupling metrics
        """
        try:
            metrics = {
                'coherence_coupling': float(self.coherence_coupling),
                'distinction_coupling': float(self.distinction_coupling),
                'phase_coupling': float(self.phase_coupling),
                'coherence_momentum': float(self.coherence_momentum),
                'distinction_momentum': float(self.distinction_momentum),
                'coupling_strength': float(self.coupling_strength),
                'feedback_strength': float(self.feedback_strength)
            }

            # Add trend metrics if history exists
            if len(self.coupling_history) > 10:
                recent = list(self.coupling_history)[-10:]

                # Calculate coupling stability (1.0 = stable, 0.0 = unstable)
                coherence_variance = np.var([r['coherence_coupling'] for r in recent])
                distinction_variance = np.var([r['distinction_coupling'] for r in recent])

                stability = 1.0 - np.clip(coherence_variance + distinction_variance, 0.0, 1.0)
                metrics['coupling_stability'] = float(stability)

                # Calculate feedback effectiveness if feedback history exists
                if len(self.feedback_history) > 10:
                    recent_feedback = list(self.feedback_history)[-10:]

                    # Effective feedback has consistent direction and appropriate magnitude
                    coherence_adj_consistency = np.mean([r['coherence_adjustment'] for r in recent_feedback])
                    coherence_adj_variance = np.var([r['coherence_adjustment'] for r in recent_feedback])

                    # Higher consistency magnitude and lower variance indicates more effective feedback
                    effectiveness = np.abs(coherence_adj_consistency) / (coherence_adj_variance + 0.01)
                    metrics['feedback_effectiveness'] = float(np.clip(effectiveness, 0.0, 1.0))

            return metrics

        except Exception as e:
            logger.error(f"Error getting coupling metrics: {e}")
            return {
                'coherence_coupling': self.coherence_coupling,
                'distinction_coupling': self.distinction_coupling,
                'phase_coupling': self.phase_coupling,
                'error': str(e)
            }
