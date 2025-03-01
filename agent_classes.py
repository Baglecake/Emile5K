"""
Enhanced Agent Module
----------------------
This module defines classes for an enhanced quantum‐aware agent. It includes a
base agent class, an agent class with prediction capabilities, and a final evolution
agent that integrates quantum state management, recursive memory, transformer-based
decision making, surplus regulation, and advanced adaptation mechanisms.
"""
import math
import time
import random
import numpy as np
import traceback
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qiskit.quantum_info import Statevector
from qiskit_aer.library import SaveStatevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from emergent_potential import EmergentPotentialField
from logging_setup import setup_logging
from emergence_monitor import EmergenceTracker, DimensionMonitor, EmergenceEvent
from transformer_modules import FourDimTransformerAdapter, RecursiveDistinctionTransformer
from core_quantum import EnhancedQuantumState
from surplusdynamics import EnhancedDistinctionDynamics, EnhancedSurplusDynamics
from memory_field import RecursiveDistinctionMemory, OntologicalField
from training_pipeline import (
    QuantumAwareOptimizer,
    EnhancedTrainingPipeline,
    EnhancedErrorRecovery,
    OptimizationCoordinator,
    QuantumStateValidator,
    StateSynchronizationManager,
    EnhancedQuantumSelfOptimization
)
from data_classes import TransformerOutput, SurplusState
from utilities import (
    _initialize_circuit,
    MOMENTUM_DECAY,
    MINIMUM_COHERENCE_FLOOR,
    INSTABILITY_GRACE_PERIOD,
    adapt_tensor_shape,
    NUM_QUBITS_PER_AGENT,
    COLLAPSE_DISSIPATION_THRESHOLD,
    EXPULSION_RECOVERY_RATE,
    CORE_DISTINCTION_UPDATE_RATE,
    TARGET_DISTINCTION,
    DISTINCTION_ANCHOR_WEIGHT,
    NUM_TRANSFORMER_LAYERS,
    NUM_TRANSFORMER_HEADS,
    HIDDEN_DIM
)
from training_pipeline import MetricValidator, QuantumStateValidator
from cognitive_structures import RecursiveCognitiveStructuring
from analysis import QuantumAnalyzer, CausalityAnalysis, BayesianAnalysis
from symbolic_output import SymbolicOutput



# Global constants
NUM_QUBITS_PER_AGENT = 4
LEARNING_RATE = 1e-4
LEARNING_RATE_MIN = 1e-5
LEARNING_RATE_MAX = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VALUE = 1.0
REWARD_SCALING = 1.0
EVOLUTION_TIME = 0.1

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Agent-Level Classes
# -----------------------------------------------------------------------------
class EnhancedSingleAgentBase:
    """
    Base class for an enhanced quantum-aware agent.
    Integrates an enhanced quantum state, recursive memory, and dynamics for computing distinction and regulating surplus.
    """
    def __init__(self, num_qubits: int = NUM_QUBITS_PER_AGENT):
        self.num_qubits = num_qubits
        self.qc, self.simulator = _initialize_circuit(self.num_qubits)
        self.minimum_coherence = MINIMUM_COHERENCE_FLOOR

        # Initialize quantum state first
        self.quantum_state = EnhancedQuantumState(agent=self, num_qubits=num_qubits)

        # Initialize distinction and surplus dynamics
        self.distinction_dynamics = EnhancedDistinctionDynamics()
        self.surplus_dynamics = EnhancedSurplusDynamics()
        print(f"DEBUG: Surplus Dynamics Initialized - Surplus State: {self.surplus_dynamics.surplus_state}")

        # Initialize memory system
        self.memory = RecursiveDistinctionMemory(max_size=10000, hierarchy_levels=4)

        # Initialize optimizer (with None model for now - will be set later)
        self.quantum_optimizer = QuantumAwareOptimizer(model=None)

        # History tracking
        self.distinction_history = deque(maxlen=1000)
        self.quantum_metric_history = deque(maxlen=1000)

        # State initialization
        self.distinction_level = 0.5  # Initialize with moderate distinction
        self.learning_rate = 0.005
        self.phase = 0.0
        self.stability_factor = 1.0

        # Initialize field
        self.ontological_field = OntologicalField()

        # Initialize statevector
        self.statevector = Statevector.from_label('0' * self.num_qubits)

        if self.qc is None or self.simulator is None:
            raise RuntimeError("❌ Failed to initialize quantum circuit and simulator.")

    def compute_distinction(self) -> float:
        """
        Compute the agent's distinction level by combining:
          - Phase-based distinction
          - Entropy-based distinction (1 - normalized entropy)
          - Coherence-based distinction
          - The effect of field resistance
        The result is scaled by the quantum-surplus coupling.
        """
        try:
            # Make sure stability_factor and other attributes are set
            self.stability_factor = getattr(self.distinction_dynamics, 'stability_factor', 1.0)
            self.learning_rate = getattr(self.distinction_dynamics, 'learning_rate', 0.005)
            self.minimum_coherence = getattr(self.distinction_dynamics, 'minimum_distinction', MINIMUM_COHERENCE_FLOOR)


            # Use the existing surplus state
            self.surplus_state = self.surplus_dynamics.surplus_state

            # Update quantum state parameters
            self.quantum_state.phase_coherence = getattr(self.distinction_dynamics, 'quantum_influence', MINIMUM_COHERENCE_FLOOR)
            self.quantum_state.minimum_coherence = self.minimum_coherence

            # Get quantum metrics and calculate distinction
            metrics = self.quantum_state.get_quantum_metrics()
            coupling = self.quantum_state.compute_quantum_surplus_coupling(self.surplus_state.values)
            field_resistance = self.ontological_field.resistance(self.distinction_level)

            # Calculate the total distinction from various components
            total_distinction = (
                0.3 * metrics.get('phase_distinction', 0.5) +
                0.3 * (1 - metrics.get('normalized_entropy', 0.5)) +
                0.2 * metrics.get('coherence_distinction', 0.5) +
                0.2 * (1 - field_resistance)
            )

            # Scale by quantum coupling
            total_distinction *= (0.5 + 0.5 * coupling)

            return float(np.clip(total_distinction, 0, 1))

        except Exception as e:
            print(f"Error in computing distinction: {e}")
            traceback.print_exc()
            return self.distinction_level  # Return current value if error occurs

    def _update_quantum_metrics(self):
        """Update and store quantum metrics from the current quantum state."""
        try:
            metrics = self.quantum_state.get_quantum_metrics() or {}  # Ensure it's a dict
            self.quantum_metric_history.append(metrics)

            self.phase = metrics.get('phase', 0.0)  # Avoid KeyError
            self.quantum_state.phase_coherence = metrics.get('phase_coherence', self.minimum_coherence)

        except Exception as e:
            print(f"Error updating quantum metrics: {e}")
            traceback.print_exc()

    def update_distinction_and_surplus(self) -> None:
        """Update distinction level and adjust surplus values accordingly."""
        try:
            # Get current metrics
            metrics = self.quantum_state.get_quantum_metrics()
            field_resistance = self.ontological_field.resistance(self.distinction_level)

            # Calculate new distinction level
            self.distinction_level = self.distinction_dynamics.compute_distinction(
                metrics, field_resistance, self.surplus_dynamics.surplus_state
            )

            # Validate surplus state before modifying
            if not isinstance(self.surplus_dynamics.surplus_state, SurplusState) or self.surplus_dynamics.surplus_state is None:
                print(f"Warning: Reinitializing surplus state as SurplusState()")
                self.surplus_dynamics.surplus_state = SurplusState()

            # Update surplus based on quantum metrics
            self.surplus_dynamics.update_surplus(
                metrics.get('phase_coherence', self.quantum_state.minimum_coherence),
                metrics.get('normalized_entropy', 0.0)
            )

            # Check if expulsion is needed
            if self.surplus_dynamics.check_expulsion_needed(self.distinction_level):
                expelled, magnitude = self.surplus_dynamics.perform_expulsion(self.quantum_state)
                print(f"Performed surplus expulsion with magnitude {magnitude:.4f}")

            # Process recovery if active
            self.surplus_dynamics.process_recovery(self.quantum_state, self.distinction_level)

        except Exception as e:
            print(f"Error updating distinction and surplus: {e}")
            traceback.print_exc()

    def update_state(self):
        """Update the quantum state and store the current distinction."""
        try:
            self.quantum_state.update_state()
            self.distinction_level = self.compute_distinction()
            self._update_quantum_metrics()

            # Store the current state in memory
            self.memory.store(
                self.quantum_state.phase_coherence,
                self.distinction_level,
                self.surplus_dynamics.surplus_state.copy()
            )

            # Track distinction history
            self.distinction_history.append(self.distinction_level)

        except Exception as e:
            print(f"Error updating state: {e}")
            traceback.print_exc()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive state summary with proper integration of all components.

        Returns:
            Dictionary containing current state metrics
        """
        try:
            # Get basic metrics
            metrics = self.quantum_state.get_quantum_metrics()
            distinction_mean = np.mean(list(self.distinction_history)) if self.distinction_history else self.distinction_level

            # Create core state summary
            summary = {
                'distinction_level': self.distinction_level,
                'distinction_mean': distinction_mean,
                'stability_factor': self.stability_factor,
                'adaptation_momentum': self.adaptation_momentum,
                'collapse_prevention_active': self.collapse_prevention_active,
                'recovery_mode': self.recovery_mode,
                'quantum_metrics': metrics,
                'phase': self.phase,
                'coherence': metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR)
            }

            # Add surplus state if available
            if hasattr(self, 'surplus_dynamics') and hasattr(self.surplus_dynamics, 'surplus_state'):
                summary['surplus'] = self.surplus_dynamics.surplus_state.copy()

            # Add cognitive state if available
            if hasattr(self, 'recursive_cognition'):
                summary['cognitive_state'] = self.recursive_cognition.get_cognitive_state()

            # Add training summary if available
            if hasattr(self, 'training_pipeline'):
                summary['training_summary'] = self.training_pipeline.get_training_summary()

            # Add learning rate if available
            if hasattr(self, 'learning_rate'):
                summary['learning_rate'] = self.learning_rate

            # Add emergent potential field data if available
            if hasattr(self, 'emergent_potential_field'):
                field_state = self.emergent_potential_field.get_field_state()
                summary['emergent_potential'] = {
                    'total_potential': field_state.get('total_potential', 0.0),
                    'emergence_probability': field_state.get('emergence_probability', 0.0),
                    'emergence_active': field_state.get('emergence_active', False),
                    'field_intensity': field_state.get('field_intensity', 1.0)
                }

            return summary
        except Exception as e:
            print(f"Error getting state summary: {e}")
            # Return minimal state information
            return {
                'distinction_level': getattr(self, 'distinction_level', 0.5),
                'error': str(e)
            }

    def ensure_minimum_mutation(self):
        """Ensure the quantum state undergoes at least some mutation to prevent stagnation."""
        try:
            # Apply a very small random rotation to all qubits
            for qubit in range(self.num_qubits):
                # Vary the rotation angle for each qubit
                random_angle = (0.01 + 0.05 * random.random()) * np.pi
                # Randomly choose rotation axis (rx, ry, or rz)
                rotation_choice = random.choice(['rx', 'ry', 'rz'])

                if rotation_choice == 'rx':
                    self.quantum_state.apply_gate('rx', [qubit], {'theta': random_angle})
                elif rotation_choice == 'ry':
                    # Use rz with equivalent params if ry not available
                    self.quantum_state.apply_gate('rz', [qubit], {'phi': random_angle})
                else:
                    self.quantum_state.apply_gate('rz', [qubit], {'phi': random_angle})

            # Add a small amount of phase shift that varies each time
            phase_shift = 0.02 * np.pi * (random.random() - 0.5)
            self.quantum_state.apply_phase_shift(phase_shift)

            # Update state
            self.quantum_state.update_state()

            # Log the mutation
            if hasattr(self, 'logger'):
                self.logger.debug(f"Applied minimum quantum mutation with phase shift: {phase_shift:.4f}")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error ensuring minimum mutation: {e}")
            else:
                print(f"Error ensuring minimum mutation: {e}")

class EnhancedSingleAgentWithPrediction(EnhancedSingleAgentBase):
    """
    Extends the base agent with a transformer-based decision network.
    The transformer predicts future distinction levels based on current quantum and surplus metrics.
    """
    def __init__(self, num_qubits: int = NUM_QUBITS_PER_AGENT):
        super().__init__(num_qubits)

        # Initialize memory with hierarchical levels
        self.memory = RecursiveDistinctionMemory(max_size=10000, hierarchy_levels=4)

        # Initialize transformer - ensure d_model matches input_size
        self.transformer = RecursiveDistinctionTransformer(
            input_size=20,
            d_model=20,  # Match input_size
            nhead=NUM_TRANSFORMER_HEADS,
            num_layers=NUM_TRANSFORMER_LAYERS
        ).to(DEVICE)

        # Initialize optimizer and loss criterion
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=LEARNING_RATE)
        self.loss_criterion = nn.MSELoss()

        # Experience buffer for training
        self.experience_buffer = deque(maxlen=1000)

        # Initialize training pipeline
        self.training_pipeline = EnhancedTrainingPipeline(self.transformer)

        # Set minimum coherence
        self.minimum_coherence = MINIMUM_COHERENCE_FLOOR

    def _initialize_transformer(self) -> nn.Module:
        """Initialize transformer with 4D input adapter."""
        print("Initializing transformer with 4D input adapter...")

        # Create base transformer
        base_transformer = RecursiveDistinctionTransformer(
            input_size=20,
            d_model=20,
            nhead=NUM_TRANSFORMER_HEADS,
            num_layers=NUM_TRANSFORMER_LAYERS,
            output_size=1
        ).to(DEVICE)

        # Initialize weights
        for p in base_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

        # Wrap with adapter
        transformer = FourDimTransformerAdapter(base_transformer, merge_strategy="merge")
        print("Transformer initialized with adaptive 4D handling.")
        return transformer

    def prepare_transformer_input(self) -> torch.Tensor:
        """
        Prepare transformer input with improved emergence detection.
        """
        try:
            # Get current quantum metrics
            metrics = self.quantum_state.get_quantum_metrics()
            if not isinstance(metrics, dict):
                self.logger.warning("Invalid metrics returned")
                return torch.zeros((1, 1, 20), dtype=torch.float32, device=DEVICE)

            # Get distinction momentum safely
            distinction_momentum = float(getattr(self.distinction_dynamics, 'adjustment_momentum', 0.0))

            # Get stability factor safely
            distinction_stability = float(getattr(self.distinction_dynamics, 'stability_factor', 1.0))

            # Validate surplus state
            if not hasattr(self.surplus_dynamics, 'surplus_state') or not isinstance(self.surplus_dynamics.surplus_state, SurplusState):
                self.logger.warning("Warning: Invalid surplus state detected; reinitializing.")
                self.surplus_dynamics.surplus_state = SurplusState()

            # Ensure distinction_level exists
            if not hasattr(self, 'distinction_level') or self.distinction_level is None:
                self.distinction_level = 0.5

            # Build the feature vector - exactly 20 features
            features = [
                float(metrics.get('phase_coherence', 0.0)),
                float(metrics.get('normalized_entropy', 0.0)),
                float(metrics.get('phase_stability', 0.0)),
                float(metrics.get('phase_distinction', 0.0)),
                float(self.distinction_level),
                distinction_momentum,
                distinction_stability,
                float(getattr(self.distinction_dynamics, 'quantum_influence', 0.0)),
                float(self.surplus_dynamics.surplus_state.values.get('basal', 0.0)),
                float(self.surplus_dynamics.surplus_state.values.get('cognitive', 0.0)),
                float(self.surplus_dynamics.surplus_state.values.get('predictive', 0.0)),
                float(self.surplus_dynamics.surplus_state.values.get('ontological', 0.0)),
                float(self.surplus_dynamics.surplus_state.stability),
                float(self.surplus_dynamics.surplus_state.quantum_coupling),
                float(self.surplus_dynamics.surplus_state.accumulation_rate.get('basal', 0.0)),
                float(self.surplus_dynamics.surplus_state.accumulation_rate.get('cognitive', 0.0)),
                float(self.surplus_dynamics.surplus_state.accumulation_rate.get('predictive', 0.0)),
                float(self.surplus_dynamics.surplus_state.accumulation_rate.get('ontological', 0.0)),
                float(getattr(self, 'learning_rate', 0.0)),
                float(self.ontological_field.resistance(self.distinction_level))
            ]

            # Create tensor with shape [1, 1, 20]
            input_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
            input_tensor = input_tensor.view(1, 1, -1)

            # Ensure the last dimension is exactly 20
            if input_tensor.shape[-1] < 20:
                padding = (0, 20 - input_tensor.shape[-1])
                input_tensor = F.pad(input_tensor, padding)
            elif input_tensor.shape[-1] > 20:
                input_tensor = input_tensor[:, :, :20]

            # Create emergence if conditions are right
            emergence_probability = 0.0

            # Increase emergence probability based on metrics
            if self.distinction_level > 0.8:
                emergence_probability += 0.2
            if metrics.get('phase_coherence', 0.0) > 0.8:
                emergence_probability += 0.2
            if metrics.get('normalized_entropy', 0.0) < 0.2:
                emergence_probability += 0.1
            if self.surplus_dynamics.surplus_state.total_surplus() > 5.0:
                emergence_probability += 0.2

            # Check for cognitive complexity
            if hasattr(self, 'recursive_cognition'):
                cognitive_state = self.recursive_cognition.get_cognitive_state()
                if cognitive_state.get('mean_strength', 0.0) > 1.5:
                    emergence_probability += 0.1
                if cognitive_state.get('mean_stability', 0.0) > 0.8:
                    emergence_probability += 0.1

            # Check for periodic emergence patterns at certain steps
            step_counter = getattr(self, 'step_counter', 0)
            if step_counter > 0 and step_counter % 100 == 0:
                emergence_probability += 0.1

            # Apply emergence if probability threshold met
            if random.random() < emergence_probability and not getattr(self, 'dimension_increase_detected', False):
                # Create a 4D tensor by adding an extra dimension
                # This simulates the emergence of a new dimension in the system
                expanded_tensor = input_tensor.unsqueeze(1)  # Shape becomes [1, 1, 1, 20]

                # Duplicate along the emergent dimension to create meaningful structure
                expanded_tensor = expanded_tensor.repeat(1, 4, 1, 1)  # Shape becomes [1, 4, 1, 20]

                # Handle the emergent dimension
                self.handle_emergent_dimension(
                    tensor_shape=expanded_tensor.shape,
                    source="prepare_transformer_input"
                )

                # Return the emergent tensor
                return expanded_tensor

            # Adapt the tensor shape if needed (standard case)
            return adapt_tensor_shape(input_tensor, expected_dim=3, expected_last_dim=20)

        except Exception as e:
            self.logger.error(f"Error preparing transformer input: {e}")
            traceback.print_exc()
            return torch.zeros((1, 1, 20), dtype=torch.float32, device=DEVICE)

    def optimize_decision_network(self):
        """
        Optimizes the decision network using recent experiences.
        Prepares input-target pairs, computes loss, backpropagates, and updates the transformer.
        Also processes a separate training step from the experience buffer.
        """
        if len(self.memory.memory[0]) < 15:
            return

        try:
            input_data = []
            targets = []

            # Get recent states from memory
            past_states = self.memory.retrieve_recent(10, level=2)

            for i in range(len(past_states) - 1):
                # Prepare input
                state_input = self.prepare_transformer_input()
                input_data.append(state_input.squeeze(0).cpu().numpy())

                # Prepare target
                metrics = self.quantum_state.get_quantum_metrics()
                field_resistance = self.ontological_field.resistance(self.distinction_level)
                target_dist = self.distinction_dynamics.compute_distinction(
                    metrics, field_resistance, self.surplus_dynamics.surplus_state
                )
                targets.append([target_dist])

            if not input_data:
                return

            # Convert to tensors
            input_tensor = torch.from_numpy(np.array(input_data)).float().to(DEVICE)
            target_tensor = torch.from_numpy(np.array(targets)).float().to(DEVICE)

            # Ensure correct shapes
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(1)  # Add sequence length dimension
            if target_tensor.dim() == 1:
                target_tensor = target_tensor.unsqueeze(1)

            # Forward pass
            predictions = self.transformer(input_tensor)
            if hasattr(predictions, "prediction"):
                predictions = predictions.prediction
            else:
                print("Warning: Transformer output does not have 'prediction' attribute. Using zeros.")
                predictions = torch.zeros_like(target_tensor).to(DEVICE)

            predictions = predictions.view_as(target_tensor)

            # Compute loss
            distinction_loss = self.loss_criterion(predictions, target_tensor)
            stability_penalty = -0.01 * self.surplus_dynamics.surplus_state.stability
            quantum_reg = -0.01 * self.distinction_dynamics.quantum_influence
            total_loss = distinction_loss + stability_penalty + quantum_reg

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
            self.optimizer.step()

            # Adjust learning rate based on prediction error
            avg_pred = predictions.mean().item()
            avg_target = target_tensor.mean().item()
            prediction_error = abs(avg_pred - avg_target)
            if prediction_error > 0.2:
                self.learning_rate *= 1.05
            else:
                self.learning_rate *= 0.95
            self.learning_rate = np.clip(self.learning_rate, 1e-5, 0.01)

        except Exception as e:
            print(f"Error in optimize_decision_network: {e}")
            traceback.print_exc()

        # Process additional training step if enough experiences
        if len(self.experience_buffer) < self.training_pipeline.batch_size:
            return

        try:
            # Get metrics for training
            metrics = self.quantum_state.get_quantum_metrics()
            metrics.update({
                'stability': self.surplus_dynamics.surplus_state.stability,
                'quantum_coupling': self.surplus_dynamics.surplus_state.quantum_coupling
            })

            # Sample batch and train
            batch = random.sample(self.experience_buffer, self.training_pipeline.batch_size)
            loss_components = self.training_pipeline.train_step(batch, metrics)

            # Log training metrics
            if loss_components:
                print("\nTraining metrics:")
                for key, value in loss_components.items():
                    print(f"{key}: {value:.4f}")

        except Exception as e:
            print(f"Error in training step: {e}")
            traceback.print_exc()

class EnhancedSingleAgentFinalEvolution(EnhancedSingleAgentWithPrediction):
    """
    Final enhanced agent class that integrates quantum state management,
    recursive memory, transformer-based decision making, surplus regulation,
    and advanced adaptation mechanisms.
    """
    def __init__(self, agent=None, num_qubits=NUM_QUBITS_PER_AGENT, **kwargs):
        """Initialize the final evolution agent with enhanced error handling and integration.

        Args:
            agent: Optional reference to parent agent
            num_qubits: Number of qubits to use
            **kwargs: Additional configuration parameters including:
                - expression_threshold: Threshold for significant distinction change (default: 0.1)
                - expression_cooldown_period: Number of steps to wait between expressions (default: 10)
                - expression_periodic_interval: Steps between periodic expressions (default: 100)
                - symbolic_history_size: Size of symbolic expression history (default: 100)
        """
        # Import required modules
        import random
        import time
        import traceback
        import logging
        import numpy as np
        import torch

        # Initialize logging first to capture all initialization messages
        self.logger = setup_logging()
        self.num_qubits = NUM_QUBITS_PER_AGENT
        try:
            # Store and validate configuration parameters
            self._initialize_config_parameters(kwargs)

            # Initialize state tracking variables early
            self.step_counter = 0
            self.stability_factor = 1.0
            self.distinction_level = 0.5
            self.phase = 0.0
            self.dimension_increase_detected = False
            self.recovery_mode = False
            self.recovery_steps = 0
            self.collapse_prevention_active = False
            self.consecutive_failures = 0
            self.max_failures = 3
            self.adaptation_momentum = 0.0
            self.learning_rate = LEARNING_RATE

            # Pre-initialization check to ensure basic components can be created
            self.logger.info("🔹 Running pre-initialization checks...")
            if not self._pre_init_check():
                self.logger.error("❌ Pre-initialization checks failed")
                raise RuntimeError("Pre-initialization checks failed")

            # Initialize base class (sets up basic components)
            self.logger.info("🔹 Initializing base components...")
            super().__init__(num_qubits)
            self.agent = agent

            # Clear redundant initializations from parent class
            self.transformer = None  # We'll initialize a custom one below

            # Initialize core quantum components
            self.logger.info("🔹 Initializing quantum components...")
            self._initialize_quantum_components()

            # Initialize cognitive components
            self.logger.info("🔹 Initializing cognitive components...")
            self._initialize_cognitive_components()

            # Initialize memory and field components
            self.logger.info("🔹 Initializing memory and field components...")
            self._initialize_memory_components()
            # Initialize emergent potential field
            try:
                self.emergent_potential_field = EmergentPotentialField()
                self.logger.info("✅ Emergent Potential Field initialized")
            except Exception as e:
                self.logger.error(f"❌ Error initializing Emergent Potential Field: {e}")
                traceback.print_exc()

            # Initialize transformer with 4D support
            self.logger.info("🔹 Initializing transformer...")
            self.transformer = self._initialize_transformer()

            # Initialize training components
            self.logger.info("🔹 Initializing training pipeline...")
            self._initialize_training_components()

            # Initialize analysis components
            self.logger.info("🔹 Initializing analysis components...")
            self._initialize_analysis_components()

            # Initialize error recovery and validation
            self.logger.info("🔹 Initializing error recovery...")
            self._initialize_error_recovery()

            # Initialize history tracking
            self.logger.info("🔹 Setting up history tracking...")
            self._initialize_history_tracking()

            # Initialize symbolic output system
            self.logger.info("🔹 Initializing symbolic output system...")
            self._initialize_symbolic_system()

            # Perform initial state synchronization with improved retry logic
            self.logger.info("🔹 Performing initial state synchronization...")
            if not self._perform_initial_synchronization():
                self.logger.error("❌ Failed to achieve initial state synchronization after recovery")
                raise RuntimeError("Failed to achieve initial state synchronization after recovery")

            # Important: Integrate training with quantum state as final initialization step
            self.logger.info("🔹 Integrating training pipeline with quantum components...")
            if not self._integrate_training_with_quantum():
                self.logger.warning("⚠️ Training integration incomplete - may affect performance")
            else:
                self.logger.info("✅ Training pipeline successfully integrated with quantum state")

            # Verify the initial system state
            self.logger.info("🔹 Verifying initial system state...")
            if self.verify_system_state():
                self.logger.info("✅ Initial system state verification successful")
            else:
                self.logger.warning("⚠️ Initial system state verification failed - some components may be misconfigured")

        except Exception as e:
            self.logger.error(f"❌ Error initializing agent: {e}")
            traceback.print_exc()
            raise

    def _initialize_config_parameters(self, kwargs):
        """Initialize and validate configuration parameters."""
        # Set default values first
        self.expression_threshold = 0.1
        self.expression_cooldown = 0
        self.expression_cooldown_period = 10
        self.expression_periodic_interval = 100
        self.symbolic_history_maxlen = 100

        # Override with provided values
        if 'expression_threshold' in kwargs:
            threshold = kwargs['expression_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
                self.logger.warning(f"Invalid expression_threshold: {threshold}, using default: 0.1")
            else:
                self.expression_threshold = threshold

        if 'expression_cooldown_period' in kwargs:
            period = kwargs['expression_cooldown_period']
            if not isinstance(period, int) or period < 0:
                self.logger.warning(f"Invalid expression_cooldown_period: {period}, using default: 10")
            else:
                self.expression_cooldown_period = period

        if 'expression_periodic_interval' in kwargs:
            interval = kwargs['expression_periodic_interval']
            if not isinstance(interval, int) or interval <= 0:
                self.logger.warning(f"Invalid expression_periodic_interval: {interval}, using default: 100")
            else:
                self.expression_periodic_interval = interval

        if 'symbolic_history_size' in kwargs:
            size = kwargs['symbolic_history_size']
            if not isinstance(size, int) or size <= 0:
                self.logger.warning(f"Invalid symbolic_history_size: {size}, using default: 100")
            else:
                self.symbolic_history_maxlen = size

    def _initialize_quantum_components(self):
        """Initialize quantum state and related components."""
        # Set up simulator if not already done
        if not hasattr(self, 'simulator') or self.simulator is None:
            self.simulator = AerSimulator()

        # Initialize quantum state parameters
        if not hasattr(self.quantum_state, 'phase_coherence'):
            self.quantum_state.phase_coherence = MINIMUM_COHERENCE_FLOOR

        # Initialize the quantum circuit
        self.qc, returned_simulator = _initialize_circuit(self.num_qubits)

        # Use existing simulator if possible, otherwise use the returned one
        if not self.simulator:
            self.simulator = returned_simulator

        # Initialize statevector
        self.statevector = Statevector.from_label('0' * self.num_qubits)

        # Initialize quantum optimizer
        self.quantum_optimizer = EnhancedQuantumSelfOptimization(self.num_qubits)

        # Set minimum coherence across components
        self.minimum_coherence = MINIMUM_COHERENCE_FLOOR

    def _initialize_cognitive_components(self):
        """Initialize cognitive structures and dynamics."""
        # Initialize recursive cognitive structure
        self.recursive_cognition = RecursiveCognitiveStructuring()

        # Verify or initialize distinction dynamics
        if not isinstance(self.distinction_dynamics, EnhancedDistinctionDynamics):
            self.logger.warning("Reinitializing distinction dynamics")
            self.distinction_dynamics = EnhancedDistinctionDynamics()

        # Verify or initialize surplus dynamics and state
        if not hasattr(self.surplus_dynamics, 'surplus_state') or \
          not isinstance(self.surplus_dynamics.surplus_state, SurplusState):
            self.logger.warning("Reinitializing surplus state as SurplusState()")
            self.surplus_dynamics.surplus_state = SurplusState()

    def _initialize_memory_components(self):
        """Initialize memory structures and ontological field."""
        # Initialize recursive memory with hierarchical levels
        self.memory = RecursiveDistinctionMemory(max_size=10000, hierarchy_levels=4)

        # Initialize ontological field
        self.ontological_field = OntologicalField()

    def _initialize_training_components(self):
        """Initialize training pipeline and related components."""
        # Create training pipeline with the transformer
        self.training_pipeline = EnhancedTrainingPipeline(self.transformer)

        # Initialize experience buffer if not present
        if not hasattr(self, 'experience_buffer'):
            self.experience_buffer = deque(maxlen=1000)

    def _initialize_analysis_components(self):
        """Initialize analysis components with proper initialization of trackers."""
        try:
            # Initialize quantum analyzer with properly initialized oscillation detection
            self.analyzer = QuantumAnalyzer(self.num_qubits)

            # Manually initialize oscillation detection if missing
            if not hasattr(self.analyzer, 'oscillation_detection'):
                self.analyzer.oscillation_detection = {
                    'coherence_history': deque(maxlen=100),
                    'entropy_history': deque(maxlen=100),
                    'distinction_history': deque(maxlen=100)
                }

            # Initialize phase transition tracking if missing
            if not hasattr(self.analyzer, 'phase_transitions'):
                self.analyzer.phase_transitions = 0
                self.analyzer.transition_magnitudes = []
                self.analyzer.last_state = None

            # Initialize optimization coordinator
            self.optimization_coordinator = OptimizationCoordinator(self)

            # Initialize emergence tracking
            self.emergence_tracker = EmergenceTracker()
            self.dimension_monitor = DimensionMonitor()

            self.logger.info("Analysis components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing analysis components: {e}")
            traceback.print_exc()

            # Create minimal functional components even if initialization fails
            self.analyzer = QuantumAnalyzer(self.num_qubits)
            self.emergence_tracker = EmergenceTracker()
            self.dimension_monitor = DimensionMonitor()

    def _initialize_error_recovery(self):
        """Initialize error recovery and validation components."""
        # Initialize error recovery
        self.error_recovery = EnhancedErrorRecovery(self)

        # Initialize validation components
        self.state_validator = QuantumStateValidator()

        # Initialize synchronization manager
        self.sync_manager = StateSynchronizationManager(
            self.quantum_state,
            self.surplus_dynamics,
            self.distinction_dynamics
        )

    def _initialize_history_tracking(self):
        """Initialize history tracking structures."""
        # Initialize main history trackers
        self.distinction_history = deque(maxlen=1000)
        self.quantum_metric_history = deque(maxlen=1000)

        # Initialize analysis history
        self.analysis_history = {
            'coherence': deque(maxlen=1000),
            'distinction': deque(maxlen=1000),
            'entropy': deque(maxlen=1000),
            'analysis_results': deque(maxlen=1000)
        }

        # Initialize adaptation history
        self.adaptation_history = deque(maxlen=1000)

        # Initialize tensor shape history for emergence tracking
        self.tensor_shape_history = []

    def _initialize_symbolic_system(self):
        """Initialize symbolic output system and related components."""
        # Initialize symbolic output generator
        self.symbolic_system = SymbolicOutput()

        # Initialize symbolic history tracker
        self.symbolic_history = deque(maxlen=self.symbolic_history_maxlen)

        # Initialize expression-related state
        self.previous_distinction = self.distinction_level
        self.last_symbolic_expression = None

    def _perform_initial_synchronization(self):
        """Perform initial state synchronization with improved retry logic."""
        sync_success = False
        sync_attempts = 0
        max_sync_attempts = 3

        while not sync_success and sync_attempts < max_sync_attempts:
            try:
                if self.sync_manager.synchronize_states():
                    self.logger.info("✅ Initial state synchronization successful")
                    sync_success = True
                    break

                sync_attempts += 1
                self.logger.warning(f"⚠️ Synchronization attempt {sync_attempts} failed, retrying...")

                # Try basic recovery between attempts with exponential backoff
                self._basic_state_recovery()
                time.sleep(0.1 * (2 ** sync_attempts))  # Exponential backoff

            except Exception as sync_err:
                self.logger.error(f"Error during synchronization attempt {sync_attempts}: {sync_err}")
                sync_attempts += 1

        if not sync_success:
            self.logger.error("Failed to achieve initial state synchronization after multiple attempts")
            self.logger.info("🔄 Attempting full recovery as last resort...")
            if not self._attempt_state_recovery():
                self.logger.error("Failed to achieve initial state synchronization after recovery")
                return False

            self.logger.info("✅ Full recovery successful, attempting final synchronization")

            # Final synchronization attempt after recovery
            try:
                if self.sync_manager.synchronize_states():
                    self.logger.info("✅ Post-recovery synchronization successful")
                    return True
                else:
                    self.logger.error("Failed post-recovery synchronization")
                    return False
            except Exception as e:
                self.logger.error(f"Error in post-recovery synchronization: {e}")
                return False

        return sync_success

    def _pre_init_check(self) -> bool:
        """Perform pre-initialization validation and setup."""
        try:
            # Verify quantum components
            self.quantum_state = EnhancedQuantumState(num_qubits=self.num_qubits)
            if not hasattr(self.quantum_state, 'phase_coherence'):
                print("Setting initial phase coherence...")
                self.quantum_state.phase_coherence = MINIMUM_COHERENCE_FLOOR

            # Initialize and verify surplus state
            if not self._initialize_surplus_state():
                print("Failed to initialize surplus state")
                return False

            # Initialize base metrics with error handling
            try:
                metrics = self.quantum_state.get_quantum_metrics()
                if not metrics.get('phase_coherence'):
                    print("Setting initial metrics...")
                    self.quantum_state.update_phase_coherence()
            except Exception as metrics_error:
                print(f"Error getting initial metrics: {metrics_error}")
                # Don't fail on metrics error, continue initialization
                pass

            return True

        except Exception as e:
            print(f"Error in pre-initialization check: {e}")
            traceback.print_exc()
            return False

    def verify_system_state(self) -> bool:
        """
        Verify that all system components are in a consistent state.
        Returns True if all checks pass, False otherwise.
        """
        try:
            verification_results = {}

            # 1. Check quantum state
            verification_results['quantum_state'] = self.state_validator.validate_quantum_state(self.quantum_state)

            # 2. Check surplus state
            verification_results['surplus_state'] = (
                isinstance(self.surplus_dynamics.surplus_state, SurplusState) and
                self.surplus_dynamics.surplus_state.validate()
            )

            # 3. Check distinction dynamics
            verification_results['distinction'] = (
                hasattr(self.distinction_dynamics, 'distinction_level') and
                0 <= self.distinction_level <= 1.0
            )

            # 4. Check transformer
            verification_results['transformer'] = (
                self.transformer is not None and
                isinstance(self.transformer, nn.Module)
            )

            # 5. Check training pipeline
            verification_results['training_pipeline'] = (
                hasattr(self, 'training_pipeline') and
                hasattr(self.training_pipeline, 'optimizer')
            )

            # Check overall verification results
            all_verified = all(verification_results.values())
            if not all_verified:
                self.logger.warning("System state verification failed:")
                for component, status in verification_results.items():
                    if not status:
                        self.logger.warning(f"  - {component}: FAILED")

            return all_verified

        except Exception as e:
            self.logger.error(f"Error verifying system state: {e}")
            traceback.print_exc()
            return False

    def _initialize_surplus_state(self) -> bool:
        """Initialize surplus state with proper validation."""
        try:
            # Import random explicitly here to ensure it's available
            import random

            if not hasattr(self, 'surplus_dynamics'):
                self.surplus_dynamics = EnhancedSurplusDynamics()
                print(f"DEBUG: Surplus Dynamics Initialized - Surplus State: {self.surplus_dynamics.surplus_state}")

            # Ensure proper SurplusState initialization
            if not isinstance(self.surplus_dynamics.surplus_state, SurplusState) or self.surplus_dynamics.surplus_state is None:
                print(f"Warning: Reinitializing surplus state as SurplusState()")
                self.surplus_dynamics.surplus_state = SurplusState()

            # Initialize with slightly different starting values to break symmetry
            self.surplus_dynamics.surplus_state.values = {
                'basal': 1.0,
                'cognitive': 1.1,
                'predictive': 0.9,
                'ontological': 1.05
            }

            # Initialize accumulation rates with more variation
            self.surplus_dynamics.surplus_state.accumulation_rate = {
                'basal': 0.01,
                'cognitive': 0.02,
                'predictive': 0.015,
                'ontological': 0.005
            }

            # Explicitly initialize accumulation momentum with different values
            self.surplus_dynamics.accumulation_momentum = {
                'basal': 0.01,
                'cognitive': 0.0,
                'predictive': -0.01,
                'ontological': 0.005
            }

            # Set initial stability
            self.surplus_dynamics.surplus_state.stability = 1.0
            self.surplus_dynamics.surplus_state.quantum_coupling = 1.0

            # Ensure the surplus update method will use these values
            # by initializing key tracking variables
            if hasattr(self.surplus_dynamics, 'track_emergence'):
                self.surplus_dynamics.track_emergence(self.surplus_dynamics.surplus_state.values)

            return True

        except Exception as e:
            print(f"Error initializing surplus state: {e}")
            traceback.print_exc()
            return False

    def _basic_state_recovery(self):
        """Perform basic state recovery between synchronization attempts."""
        try:
            self.logger.info("Performing basic state recovery...")

            # Reset quantum state phase and coherence
            self.quantum_state.phase = 0.0
            self.quantum_state.phase_coherence = MINIMUM_COHERENCE_FLOOR

            # Apply a simple quantum gate to refresh state
            self.quantum_state.apply_gate('h', [0])

            # Ensure surplus state is valid
            if not isinstance(self.surplus_dynamics.surplus_state, SurplusState):
                self.surplus_dynamics.surplus_state = SurplusState()

            # Reset distinction level
            self.distinction_level = 0.5

            return True
        except Exception as e:
            self.logger.error(f"Error in basic state recovery: {e}")
            return False

    def _attempt_state_recovery(self) -> bool:
        """Enhanced state recovery with better integration."""
        try:
            self.logger.info("\n🔄 Starting comprehensive state recovery process...")
            recovery_successful = False

            # First, validate the current system state
            validation_results = self.state_validator.validate_system_state(self)

            # Log validation results
            self.logger.info("System validation results:")
            for component, status in validation_results.items():
                if component != 'overall':
                    self.logger.info(f"  - {component}: {'PASSED' if status else 'FAILED'}")

            # If overall validation passed, no need for recovery
            if validation_results.get('overall', False):
                self.logger.info("✅ System validation passed, no recovery needed")
                return True

            # Use error recovery to perform component or full recovery
            if hasattr(self, 'error_recovery'):
                # Get list of failed components
                failed_components = [comp for comp, status in validation_results.items()
                                    if not status and comp != 'overall']

                self.logger.info(f"Initiating recovery for failed components: {failed_components}")

                # Check if critical components have failed
                critical_failure = any(comp in ['quantum_state', 'surplus_state', 'distinction']
                                      for comp in failed_components)

                if critical_failure or len(failed_components) > 2:
                    self.logger.warning("Critical failure detected, performing full recovery")
                    recovery_successful = self.error_recovery.initiate_full_recovery()
                else:
                    # Recover each failed component
                    recovery_successful = True
                    for component in failed_components:
                        if not self.error_recovery.initiate_component_recovery(component):
                            recovery_successful = False
                            self.logger.error(f"Failed to recover component: {component}")
            else:
                # Fallback to basic recovery if error_recovery not available
                self.logger.warning("Error recovery not initialized, falling back to basic recovery")
                recovery_successful = self._basic_state_recovery()

            # Verify recovery was successful
            if recovery_successful:
                self.logger.info("✅ State recovery completed successfully")
                # Reset recovery-related counters
                self.consecutive_failures = 0
                if hasattr(self, 'recovery_mode'):
                    self.recovery_mode = False
            else:
                self.logger.error("❌ State recovery failed")

            return recovery_successful

        except Exception as e:
            self.logger.error(f"Error in state recovery attempt: {e}")
            traceback.print_exc()
            return False

    def handle_emergent_dimension(self, tensor_shape: Tuple, source: str) -> None:
        """
        Handle detection of emergent dimensions with improved integration.

        Args:
            tensor_shape: The tensor shape where emergence was detected
            source: Source of the emergence detection
        """
        try:
            self.logger.info(f"\n🌟 EMERGENT DIMENSION DETECTED: {len(tensor_shape)}D tensor")
            self.logger.info(f"Shape: {tensor_shape}, Source: {source}")

            # Store previous state for comparison
            previous_distinction = self.distinction_level
            previous_coherence = self.quantum_state.phase_coherence

            # Mark emergence detected for other systems to respond
            self.dimension_increase_detected = True
            self.emergence_tracker = getattr(self, 'emergence_tracker', EmergenceTracker())

            # Record emergence in tracker
            metrics = self.quantum_state.get_quantum_metrics()
            resource_usage = {
                'cpu_percent': 0,  # Would come from monitoring in real implementation
                'memory_percent': 0  # Would come from monitoring in real implementation
            }
            self.emergence_tracker.record_emergence(
                tensor_shape=tensor_shape,
                timestamp=time.time(),
                resource_usage=resource_usage,
                agent_metrics=metrics
            )

            # Connect with emergent potential field if available
            if hasattr(self, 'emergent_potential_field'):
                # Register emergence with the field for future correlations
                self.emergent_potential_field.register_potential(
                    component_id=f"emergence_{source}",
                    potential=0.5,  # Base potential for dimensional emergence
                    component_type='quantum',
                    state_metrics={
                        'tensor_shape': tensor_shape,
                        'dimensionality': len(tensor_shape),
                        'source': source,
                        'distinction': self.distinction_level,
                        'coherence': metrics.get('phase_coherence', 0.5)
                    }
                )

            # Generate symbolic expression for the emergence event
            if hasattr(self, 'symbolic_system'):
                surplus = self.surplus_dynamics.surplus_state.total_surplus()
                distinction = self.distinction_level
                coherence = metrics.get('phase_coherence', 0.5)
                entropy = metrics.get('normalized_entropy', 0.5)

                expression = self.symbolic_system.handle_post_emergence(
                    surplus=surplus,
                    distinction=distinction,
                    coherence=coherence,
                    dimensionality=len(tensor_shape),
                    entropy=entropy
                )

                self.logger.info(f"\n🔹 Emergent Dimension Symbolic Output: {expression}\n")

                # Store the expression with additional context
                if hasattr(self, 'symbolic_history'):
                    self.symbolic_history.append({
                        'expression': expression,
                        'type': 'emergence',
                        'step': getattr(self, 'step_counter', 0),
                        'distinction': distinction,
                        'distinction_delta': distinction - previous_distinction,
                        'coherence': coherence,
                        'coherence_delta': coherence - previous_coherence,
                        'entropy': entropy,
                        'dimensionality': len(tensor_shape),
                        'tensor_shape': tensor_shape,
                        'timestamp': time.time()
                    })

            # Enhance the quantum state in response to emergence
            self._enhance_quantum_state_for_emergence()

            # Notify cognitive structures of emergence
            if hasattr(self, 'recursive_cognition'):
                cognitive_state = self.recursive_cognition.get_cognitive_state()
                self.recursive_cognition.update(
                    phase_coherence=metrics['phase_coherence'],
                    distinction_level=self.distinction_level,
                    surplus=self.surplus_dynamics.surplus_state.values,
                    prediction_error=0.0,  # No prediction error for emergence
                    quantum_metrics=metrics
                )

            # Track emergent tensor shapes
            self.tensor_shape_history = getattr(self, 'tensor_shape_history', [])
            self.tensor_shape_history.append({
                'shape': tensor_shape,
                'dimensionality': len(tensor_shape),
                'timestamp': time.time(),
                'source': source
            })

        except Exception as e:
            self.logger.error(f"Error handling emergent dimension: {e}")
            traceback.print_exc()

    def dump_emergent_field_data(self, filepath: str = "emergent_field_data.json") -> bool:
        """
        Save emergent potential field data to a JSON file for external analysis.

        Args:
            filepath: Path to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self, 'emergent_potential_field'):
                self.logger.error("No emergent potential field to dump data from")
                return False

            import json

            # Prepare data structure
            field_data = {
                'field_state': self.emergent_potential_field.get_field_state(),
                'potential_history': [
                    {
                        'timestamp': entry['timestamp'],
                        'total_potential': entry['total_potential'],
                        'threshold': entry['threshold']
                    }
                    for entry in list(self.emergent_potential_field.potential_history)
                ],
                'emergence_history': [
                    {
                        'timestamp': event['timestamp'],
                        'potential': event['potential'],
                        'intensity': event['intensity'],
                        'threshold': event['threshold'],
                        'sequence': event['sequence_number']
                    }
                    for event in list(self.emergent_potential_field.emergence_history)
                ],
                'component_weights': self.emergent_potential_field.component_weights
            }

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(field_data, f, indent=2)

            self.logger.info(f"Emergent field data saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error dumping emergent field data: {e}")
            traceback.print_exc()
            return False

    def _enhance_quantum_state_for_emergence(self) -> None:
        """Enhance quantum state in response to emergence detection."""
        try:
            # Apply a phase shift to enhance coherence
            self.quantum_state.apply_phase_shift(0.2 * np.pi)

            # Apply Hadamard gates to key qubits to increase superposition
            self.quantum_state.apply_gate('h', [0, 1])

            # Reinforce coherence to stabilize the emerged state
            self.quantum_optimizer.reinforce_coherence(
                self.quantum_state.qc,
                0.1,  # Low distinction variance for stability
                self.quantum_state.phase_coherence
            )

            # Update quantum state
            self.quantum_state.update_state()
            self._update_quantum_metrics()

            self.logger.info("✅ Quantum state enhanced for emergence")

        except Exception as e:
            self.logger.error(f"Error enhancing quantum state for emergence: {e}")

    def adapt(self):
        """
        Main adaptation pipeline with enhanced stability and collapse prevention.
        """
        try:
            # Get current state metrics
            metrics = self.quantum_state.get_quantum_metrics()

            # Check for potential collapse
            collapse_prob = self.recursive_cognition.predict_cognitive_collapse()

            # Activate collapse prevention if needed
            if collapse_prob > COLLAPSE_DISSIPATION_THRESHOLD and not self.collapse_prevention_active:
                print(f"⚠️ High collapse probability detected: {collapse_prob:.4f}")
                self.activate_collapse_prevention()

            # Update recursive cognitive structure - FIXED: pass full metrics dictionary
            self.recursive_cognition.update(
                phase_coherence=metrics['phase_coherence'],
                distinction_level=self.distinction_level,
                surplus=self.surplus_dynamics.surplus_state.values,
                prediction_error=self.training_pipeline.get_training_summary().get(
                    'avg_distinction_loss', 0.0
                ),
                quantum_metrics=metrics  # Pass the complete metrics dictionary
            )

            # Get cognitive state for optimization decisions
            cognitive_state = self.recursive_cognition.get_cognitive_state()

            # Optimize quantum state if not in recovery
            if not self.recovery_mode:
                self.quantum_optimizer.optimize_quantum_state(
                    self.quantum_state,
                    self.distinction_level,
                    cognitive_state
                )

            # Update distinction and surplus
            self.update_distinction_and_surplus()

            # Apply quantum feedback refinement if not in recovery
            if not self.recovery_mode:
                self.quantum_feedback_refinement()
                self.reinforce_quantum_coherence()

            # Optimize decision network
            self.optimize_decision_network()

            # Apply meta-adaptation if stable
            if self.stability_factor > 0.7:
                self.recursive_meta_adaptation()

            # Update state and metrics
            self.update_state()

            # Track adaptation
            self._track_adaptation()

        except Exception as e:
            print(f"Error in adaptation pipeline: {e}")
            traceback.print_exc()
            if not self.recovery_mode:
                self.enter_recovery_mode()

    def activate_collapse_prevention(self):
        """Activate collapse prevention mechanisms."""
        try:
            self.collapse_prevention_active = True

            # Recycle surplus for stability
            recycled_surplus = self.recursive_cognition.dissipate_collapse(
                self.surplus_dynamics.surplus_state.values
            )

            # Use recycled surplus to boost stability
            if recycled_surplus:
                self.stability_factor *= (1.0 + sum(recycled_surplus.values()) * 0.1)

            # Apply corrective quantum operations
            self.quantum_state.apply_phase_shift(0.1 * np.pi)

            # Increase coherence preservation
            self.quantum_optimizer.reinforce_coherence(
                self.quantum_state.qc,
                self.distinction_dynamics.get_distinction_metrics()['stability_factor'],
                self.quantum_state.phase_coherence
            )

            print("Collapse prevention mechanisms activated.")

        except Exception as e:
            print(f"Error in collapse prevention: {e}")
            traceback.print_exc()

    def compute_distinction(self) -> float:
        """
        Compute the agent's distinction level by combining multiple factors.
        """
        self.logger.info("Calculating distinction level...")
        try:
            metrics = self.quantum_state.get_quantum_metrics()


            # Calculate coupling and field resistance
            coupling = self.quantum_state.compute_quantum_surplus_coupling(self.surplus_dynamics.surplus_state.values)
            field_resistance = self.ontological_field.resistance(self.distinction_level)

            # Calculate total distinction from multiple components
            total_distinction = (
                0.3 * metrics.get('phase_distinction', 0.5) +
                0.3 * (1 - metrics.get('normalized_entropy', 0.5)) +
                0.2 * metrics.get('phase_coherence', 0.5) +
                0.2 * (1 - field_resistance)
            )

            # Scale by quantum coupling
            total_distinction *= (0.5 + 0.5 * coupling)

            return float(np.clip(total_distinction, 0, 1))

        except Exception as e:
            print(f"Error in computing distinction: {e}")
            traceback.print_exc()
            return self.distinction_level

    def update_distinction_and_surplus(self) -> None:
        """Update distinction level and adjust surplus values."""
        try:
            # Get metrics and calculate field resistance
            metrics = self.quantum_state.get_quantum_metrics()
            field_resistance = self.ontological_field.resistance(self.distinction_level)

            # Update distinction level
            self.distinction_level = self.distinction_dynamics.compute_distinction(
                metrics, field_resistance, self.surplus_dynamics.surplus_state
            )

            # Ensure valid surplus state
            if not isinstance(self.surplus_dynamics.surplus_state, SurplusState):
                print("Warning: surplus_state not properly initialized, reinitializing as SurplusState()")
                self.surplus_dynamics.surplus_state = SurplusState()

            # Update surplus based on quantum metrics
            self.surplus_dynamics.update_surplus(
                metrics.get('phase_coherence', self.quantum_state.minimum_coherence),
                metrics.get('normalized_entropy', 0.0)
            )

            # Check for surplus expulsion
            if self.surplus_dynamics.check_expulsion_needed(self.distinction_level):
                expelled, magnitude = self.surplus_dynamics.perform_expulsion(self.quantum_state)
                print(f"Performed surplus expulsion with magnitude {magnitude:.4f}")

            # Process recovery if active
            self.surplus_dynamics.process_recovery(self.quantum_state, self.distinction_level)

        except Exception as e:
            print(f"Error updating distinction and surplus: {e}")
            traceback.print_exc()

    def adjust_surplus(self):
        """
        Adjust surplus values based on current metrics with enhanced dynamics
        to promote emergence.
        """
        try:
            # Get quantum metrics and field resistance
            metrics = self.quantum_state.get_quantum_metrics()
            field_resistance = self.ontological_field.resistance(self.distinction_level)

            # First, update surplus using basic method
            self.surplus_dynamics.update_surplus(
                phase_coherence=metrics.get('phase_coherence', 1.0),
                normalized_entropy=metrics.get('normalized_entropy', 0.0)
            )

            # Then also call adjust_surplus for additional dynamic behavior
            self.surplus_dynamics.adjust_surplus(
                distinction_level=self.distinction_level,
                quantum_metrics=metrics,
                field_resistance=field_resistance
            )

            # Add additional random variation every few steps to break synchronization
            if hasattr(self, 'step_counter') and self.step_counter % 5 == 0:
                self._add_surplus_variation()

        except Exception as e:
            print(f"Error adjusting surplus: {e}")
            traceback.print_exc()

    def _add_surplus_variation(self):
        """
        Add small random variations to surplus values to break synchronization
        and promote emergence.
        """
        try:
            if not hasattr(self.surplus_dynamics, 'surplus_state') or \
              not isinstance(self.surplus_dynamics.surplus_state, SurplusState):
                return

            # Calculate the mean surplus for scaling purposes
            mean_surplus = sum(self.surplus_dynamics.surplus_state.values.values()) / \
                          max(1, len(self.surplus_dynamics.surplus_state.values))

            # Add small random variations, scaled by the mean surplus
            variation_scale = 0.05 * mean_surplus  # 5% variation

            for key in self.surplus_dynamics.surplus_state.values:
                # Generate random variation, more likely to be positive than negative
                variation = variation_scale * (np.random.random() * 2 - 0.8)

                # Apply variation
                current_value = self.surplus_dynamics.surplus_state.values[key]
                new_value = current_value + variation

                # Ensure value stays within bounds
                self.surplus_dynamics.surplus_state.values[key] = np.clip(
                    new_value, 0.1, MAX_SURPLUS
                )

            # Log the variation if it's significant
            if variation_scale > 0.1:
                print(f"Added surplus variation of scale {variation_scale:.4f}")

        except Exception as e:
            print(f"Error adding surplus variation: {e}")

    def quantum_feedback_refinement(self):
        """Apply quantum feedback refinement based on metrics."""
        try:
            metrics = self.quantum_state.get_quantum_metrics()

            # Apply phase shift for high entropy
            if metrics.get('normalized_entropy', 0.0) > 0.6:
                phase_shift = float((1 - float(metrics.get('phase_coherence', 0.5))) * float(np.pi))
                phase_shift *= float(1.0 + 0.1 * float(self.adaptation_momentum))
                self.quantum_state.apply_phase_shift(phase_shift)

            # Apply corrective gates for low coherence
            if metrics.get('phase_coherence', 1.0) < 0.3:
                self.quantum_state.apply_gate('x', [0])
                self.quantum_state.apply_phase_shift(float(0.1 * np.pi))

            # Update state and metrics
            self.quantum_state.update_state()
            self._update_quantum_metrics()

        except Exception as e:
            print(f"Error in quantum feedback refinement: {e}")
            traceback.print_exc()

    def _integrate_training_with_quantum(self) -> bool:
        """
        Ensure training pipeline is correctly integrated with quantum state for better
        initialization and gradient flow between quantum and classical components.

        Returns:
            bool: True if integration successful, False otherwise
        """
        try:
            self.logger.info("Integrating training pipeline with quantum state...")

            # Get comprehensive initial metrics from quantum state
            initial_metrics = self.quantum_state.get_quantum_metrics()
            distinction_metrics = self.quantum_state.get_quantum_distinction_metrics()

            # Combine metrics for more comprehensive initialization
            combined_metrics = {**initial_metrics, **distinction_metrics}

            # Prepare initial input with proper error handling
            try:
                initial_input = self.prepare_transformer_input()
                if not isinstance(initial_input, torch.Tensor):
                    self.logger.warning("Initial input is not a tensor, creating default")
                    initial_input = torch.zeros((1, 1, 20), device=DEVICE)
            except Exception as input_err:
                self.logger.error(f"Error preparing initial input: {input_err}")
                initial_input = torch.zeros((1, 1, 20), device=DEVICE)

            # Create multiple initial experiences with slight variations to improve
            # initial training stability and prevent overfitting to a single point
            for variation in range(5):
                # Add small random variation to create diversity in initial training data
                distinction_variation = self.distinction_level * (1.0 + 0.05 * (random.random() - 0.5))
                coherence_variation = initial_metrics['phase_coherence'] * (1.0 + 0.05 * (random.random() - 0.5))

                # Create varied metrics for each experience
                varied_metrics = combined_metrics.copy()
                varied_metrics['phase_coherence'] = coherence_variation
                varied_metrics['coherence_distinction'] = coherence_variation

                # Create varied initial experience
                varied_experience = {
                    'state': initial_input.cpu().numpy() if isinstance(initial_input, torch.Tensor) else np.zeros((1, 1, 20)),
                    'prediction': float(distinction_variation),
                    'actual': float(self.distinction_level),
                    'quantum_metrics': varied_metrics,
                    'reward': 0.0,  # Initial reward neutral
                    'stability': float(self.stability_factor),
                    'adaptation_momentum': float(self.adaptation_momentum),
                    'next_distinction': float(self.distinction_level)  # For temporal learning
                }

                # Add initial experience to buffer with appropriate priority
                priority = 0.5 if variation == 0 else 0.3  # Higher priority for base experience
                self.training_pipeline.add_experience(varied_experience, priority)

            # Ensure the optimizer is aware of the model's parameters
            if hasattr(self.training_pipeline, 'optimizer'):
                if self.transformer is not None:
                    self.training_pipeline.optimizer = QuantumAwareOptimizer(self.transformer)

            # Validate metrics with proper error reporting
            validated_metrics = self.training_pipeline.metric_validator.validate_metrics(combined_metrics)
            if len(validated_metrics) < len(combined_metrics):
                missing = set(combined_metrics.keys()) - set(validated_metrics.keys())
                self.logger.warning(f"Some metrics were not validated: {missing}")

            # Perform an initial optimization step if we have enough experiences
            if len(self.training_pipeline.experience_buffer) >= self.training_pipeline.batch_size:
                self.logger.info("Performing initial training step to establish gradients")
                batch = random.sample(
                    self.training_pipeline.experience_buffer,
                    self.training_pipeline.batch_size
                )
                loss_components = self.training_pipeline.train_step(batch, validated_metrics)
                self.logger.info(f"Initial training loss: {loss_components}")

            # Set up the synchronization manager for training pipeline
            if hasattr(self, 'sync_manager'):
                self.sync_manager.training_pipeline = self.training_pipeline

            self.logger.info("✅ Quantum-training integration complete")
            return True

        except Exception as e:
            self.logger.error(f"Error integrating training with quantum: {e}")
            traceback.print_exc()

            # Try basic recovery to maintain functionality
            try:
                self.training_pipeline.reset_to_baseline = True  # Flag for future reset
                self.training_pipeline.add_experience({
                    'state': np.zeros((1, 1, 20)),
                    'prediction': 0.5,
                    'actual': 0.5,
                    'quantum_metrics': {'phase_coherence': MINIMUM_COHERENCE_FLOOR}
                })
                self.logger.warning("Applied minimal recovery to training pipeline")
                return False
            except:
                self.logger.error("Complete failure in recovery attempt")
                return False

    def optimize_decision_network(self):
        """Optimize decision network with enhanced stability and proper emergent dimension handling."""
        try:
            # Skip optimization if in recovery
            if self.recovery_mode:
                return

            # Prepare input data
            input_tensor = self.prepare_transformer_input()

            # Forward pass
            output = self.transformer(input_tensor)
            if not hasattr(output, "prediction") or output.prediction is None:
                self.logger.warning("Transformer output is missing prediction attribute")
                # Create a default prediction tensor
                prediction = torch.tensor([0.5], device=input_tensor.device)
            else:
                prediction = output.prediction  # keep as tensor for gradient flow

            # Handle multidimensional predictions properly
            if prediction.numel() == 1:
                prediction_value = prediction.item()  # simple scalar case
            else:
                # For emergent dimensions, take the mean to get a single value
                prediction_value = prediction.mean().item()
                self.logger.info(f"Handling emergent dimension prediction with shape {prediction.shape}")

            # Apply quantum feedback based on prediction
            if self.stability_factor > 0.5:
                if prediction_value > 0.5:
                    phase_shift = prediction_value * np.pi
                    self.quantum_state.apply_phase_shift(phase_shift)
                else:
                    self.quantum_state.apply_gate('x', [0])
                    self.quantum_state.apply_phase_shift(prediction_value * np.pi / 2)

            # Update quantum state
            self.quantum_state.update_state()
            self._update_quantum_metrics()

            # Store experience
            experience = self._prepare_experience(prediction_value)
            self.training_pipeline.add_experience(experience)

            # Get cognitive state to adjust training
            cognitive_state = self.recursive_cognition.get_cognitive_state()

            # Train if enough experiences
            if hasattr(self.training_pipeline, 'experience_buffer') and hasattr(self.training_pipeline, 'batch_size'):
                if len(self.training_pipeline.experience_buffer) >= self.training_pipeline.batch_size:
                    metrics = self.quantum_state.get_quantum_metrics()
                    batch = random.sample(
                        self.training_pipeline.experience_buffer,
                        self.training_pipeline.batch_size
                    )
                    self.training_pipeline.train_step(batch, metrics)

                    # Apply cognitive-based training adjustments
                    if hasattr(self.training_pipeline, 'adjust_training_with_cognitive_state'):
                        self.training_pipeline.adjust_training_with_cognitive_state(cognitive_state)

        except Exception as e:
            self.logger.error(f"Error optimizing decision network: {e}")
            traceback.print_exc()

    def _adjust_learning_rates(self):
        """
        Adjust learning rates based on current system state.
        This provides a unified point for all learning rate adjustments.
        """
        try:
            # Get current metrics for adjustment
            metrics = self.quantum_state.get_quantum_metrics()
            cognitive_state = self.recursive_cognition.get_cognitive_state()

            # Calculate adjustment factors
            stability = cognitive_state.get('mean_stability', 0.5)
            coherence = metrics.get('phase_coherence', 0.5)
            prediction_error = self.training_pipeline.get_training_summary().get('avg_distinction_loss', 0.0)

            # Combined factor - higher values indicate more stable conditions
            adjustment_factor = (0.4 * stability + 0.3 * coherence + 0.3 * (1.0 - prediction_error))

            # Adjust based on factor
            if adjustment_factor > 0.7:  # Very stable, can increase learning rate
                new_lr = min(self.learning_rate * 1.05, LEARNING_RATE_MAX)
            elif adjustment_factor < 0.3:  # Unstable, decrease learning rate
                new_lr = max(self.learning_rate * 0.8, LEARNING_RATE_MIN)
            else:  # Maintain with small adjustments
                new_lr = self.learning_rate * (0.95 + 0.1 * adjustment_factor)

            # Apply the adjustment
            self.learning_rate = new_lr

            # Apply to training pipeline if available
            if hasattr(self, 'training_pipeline') and hasattr(self.training_pipeline, 'optimizer'):
                for param_group in self.training_pipeline.optimizer.param_groups:
                    param_group['lr'] = new_lr

            # Log adjustment
            self.logger.debug(f"Learning rate adjusted: {self.learning_rate:.6f} (factor: {adjustment_factor:.3f})")

        except Exception as e:
            self.logger.error(f"Error adjusting learning rates: {e}")
            # Don't raise - keep default learning rate

    def get_cognitive_feedback_visualization(self) -> Dict[str, Any]:
        """
        Get visualization data for cognitive feedback matrix.

        Returns:
            Dictionary with feedback matrix data and metrics
        """
        try:
            # Get feedback visualization from cognitive structure
            feedback_data = self.recursive_cognition.get_feedback_matrix_visualization()

            # Add additional agent metrics for context
            metrics = self.quantum_state.get_quantum_metrics()
            feedback_data.update({
                'agent_distinction': float(self.distinction_level),
                'agent_coherence': float(metrics.get('phase_coherence', 0.5)),
                'agent_entropy': float(metrics.get('normalized_entropy', 0.5)),
                'agent_stability': float(self.stability_factor)
            })

            return feedback_data

        except Exception as e:
            print(f"Error getting cognitive feedback visualization: {e}")
            return {
                'error': str(e),
                'matrix': [[0.0]]
            }

    def recursive_meta_adaptation(self):
        """Apply meta-adaptation to transformer parameters based on distinction variance."""
        try:
            # Get recent distinction variance
            past_states = self.memory.retrieve_recent(10, level=1)
            if not past_states:
                print("Warning: No past states available for meta-adaptation")
                return

            # Calculate distinction variance
            distinction_values = [s[1] for s in past_states if isinstance(s, tuple) and len(s) > 1]
            if not distinction_values:
                print("Warning: No valid distinction values for meta-adaptation")
                return

            distinction_var = float(np.var(distinction_values))

            # Apply meta-adaptation if variance is high
            if distinction_var > 0.03 and self.stability_factor > 0.5:
                # Perturb transformer parameters
                with torch.no_grad():
                    for param in self.transformer.parameters():
                        if param.requires_grad:
                            noise_scale = 0.001 * (1.0 - self.stability_factor)
                            param.data += torch.randn_like(param) * noise_scale

                # Update adaptation momentum
                self.adaptation_momentum = (
                    MOMENTUM_DECAY * self.adaptation_momentum +
                    (1 - MOMENTUM_DECAY) * distinction_var
                )

                print("Meta-adaptation applied.")

        except Exception as e:
            print(f"Error in meta-adaptation: {e}")
            traceback.print_exc()

    def reinforce_quantum_coherence(self):
        """Reinforce quantum coherence based on metrics."""
        try:
            quantum_metrics = self.quantum_state.get_quantum_distinction_metrics()

            # Check if reinforcement is needed
            if quantum_metrics.get('phase_coherence', 1.0) < 0.5 or self.surplus_dynamics.surplus_state.stability < 0.4:
                # Calculate distinction variance from memory
                memory_entries = self.memory.retrieve_recent(10, level=1)
                distinction_var = np.var([s[1] for s in memory_entries]) if memory_entries else 0.0
                distinction_var = float(distinction_var)  # Ensure distinction variance is a float

                # Apply reinforcement
                self.quantum_optimizer.reinforce_coherence(
                    self.quantum_state.qc,
                    distinction_var,
                    quantum_metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR)
                )

                # Update state
                self.quantum_state.update_state()
                self._update_quantum_metrics()

        except Exception as e:
            print(f"Error in reinforce_quantum_coherence: {e}")
            traceback.print_exc()

    def _track_adaptation(self):
        """Track adaptation stability and update metrics."""
        try:
            # Initialize adaptation_momentum if needed
            if not hasattr(self, 'adaptation_momentum'):
                self.adaptation_momentum = 0.0

            # Calculate adaptation momentum
            if len(self.distinction_history) > 0:
                # Measure deviation from recent mean
                current_distinction = self.distinction_level
                recent_distinctions = list(self.distinction_history)
                mean_distinction = np.mean(recent_distinctions)

                # Update momentum based on deviation
                self.adaptation_momentum = (
                    0.9 * self.adaptation_momentum +
                    0.1 * abs(current_distinction - mean_distinction)
                )

                # Apply coherence reinforcement if needed
                if self.adaptation_momentum > 0.05:
                    print("Adaptation stability fluctuating, adjusting quantum coherence.")
                    self.quantum_optimizer.reinforce_coherence(
                        self.quantum_state.qc,
                        self.distinction_level,
                        self.quantum_state.phase_coherence
                    )

            # Update stability metrics
            stability_metrics = getattr(self, 'stability_metrics', {})
            stability_metrics = {
                'adaptation_stability': 1.0 / (1.0 + self.adaptation_momentum),
                'quantum_coupling': 1.0,
                'surplus_stability': 1.0
            }
            self.stability_metrics = stability_metrics

            # Track adaptation history
            adaptation_history = getattr(self, 'adaptation_history', deque(maxlen=1000))
            adaptation_history.append({
                'momentum': float(self.adaptation_momentum),
                'distinction_level': float(self.distinction_level),
                'stability': float(stability_metrics['adaptation_stability']),
                'timestamp': time.time()
            })
            self.adaptation_history = adaptation_history

        except Exception as e:
            print(f"Error in adaptation tracking: {e}")
            if not hasattr(self, 'adaptation_momentum'):
                self.adaptation_momentum = 0.0

    def enter_recovery_mode(self):
        """Enter recovery mode with reduced activity."""
        try:
            self.recovery_mode = True
            self.stability_factor *= 0.5
            print("Entering agent recovery mode.")

            # Apply recovery operations
            self.adaptation_momentum = 0.0  # Reset momentum

            # Apply stabilizing quantum operations
            self.quantum_state.apply_phase_shift(0.1 * np.pi)

            # Increase coherence preservation
            self.quantum_optimizer.reinforce_coherence(
                self.quantum_state.qc,
                0.5,  # Reduced distinction variance
                self.quantum_state.phase_coherence
            )

            # Schedule recovery exit
            self.recovery_steps = 50

        except Exception as e:
            print(f"Error applying recovery operations: {e}")
            traceback.print_exc()

    def update_analysis_history(self) -> None:
        """Update history with the current metrics."""
        try:
            # Initialize analysis_history if it doesn't exist
            if not hasattr(self, 'analysis_history'):
                self.analysis_history = {
                    'coherence': deque(maxlen=1000),
                    'distinction': deque(maxlen=1000),
                    'entropy': deque(maxlen=1000),
                    'phase': deque(maxlen=1000),  # Added phase key
                    'analysis_results': deque(maxlen=1000)
                }

            # Also check if specific key is missing and add it
            if 'phase' not in self.analysis_history:
                self.analysis_history['phase'] = deque(maxlen=1000)

            # Get current metrics
            metrics = self.quantum_state.get_quantum_metrics()

            # Ensure metrics has valid values
            coherence = metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR)
            entropy = metrics.get('normalized_entropy', 0.0)
            phase = metrics.get('phase', 0.0)

            # Update history with properly validated values
            self.analysis_history['coherence'].append(float(coherence))
            self.analysis_history['distinction'].append(float(self.distinction_level))
            self.analysis_history['entropy'].append(float(entropy))
            self.analysis_history['phase'].append(float(phase))

            # Log progress periodically
            if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
                self.logger.info(f"Analysis history updated: coherence={len(self.analysis_history['coherence'])}, "
                              f"distinction={len(self.analysis_history['distinction'])}, "
                              f"entropy={len(self.analysis_history['entropy'])}, "
                              f"phase={len(self.analysis_history['phase'])}")

        except Exception as e:
            self.logger.error(f"Error updating analysis history: {e}")
            traceback.print_exc()

    def perform_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis and store results with improved error handling."""
        try:
            # Ensure analyzer is properly initialized
            if not hasattr(self, 'analyzer') or self.analyzer is None:
                self.logger.warning("Analyzer not initialized. Creating a new one.")
                self.analyzer = QuantumAnalyzer(self.num_qubits)

            # Ensure analysis_history is properly initialized
            if not hasattr(self, 'analysis_history') or not isinstance(self.analysis_history, dict):
                self.logger.warning("Analysis history not properly initialized. Initializing now.")
                self.analysis_history = {
                    'coherence': deque(maxlen=1000),
                    'distinction': deque(maxlen=1000),
                    'entropy': deque(maxlen=1000),
                    'phase': deque(maxlen=1000),
                    'analysis_results': deque(maxlen=1000)
                }

            # Check if we have enough data for analysis
            required_keys = ['coherence', 'distinction', 'entropy']
            if not all(key in self.analysis_history for key in required_keys):
                self.logger.warning("Analysis history missing required keys. Cannot perform analysis.")
                return {"status": "Missing required history keys"}

            if not all(len(self.analysis_history[key]) > 5 for key in required_keys):
                self.logger.warning("Not enough data points in analysis history. Cannot perform analysis.")
                return {"status": "Insufficient data points for analysis"}

            # Log the history sizes for debugging
            self.logger.info(f"Analysis history sizes: coherence={len(self.analysis_history['coherence'])}, "
                          f"distinction={len(self.analysis_history['distinction'])}, "
                          f"entropy={len(self.analysis_history['entropy'])}")

            # Perform analysis
            results = self.analyzer.analyze_quantum_evolution(self.quantum_state, self.analysis_history)

            # Store results in history
            if 'analysis_results' in self.analysis_history:
                self.analysis_history['analysis_results'].append(results)

            # Log analysis completion
            analysis_status = results.get('status', 'complete')
            self.logger.info(f"Analysis completed with status: {analysis_status}")

            return results

        except Exception as e:
            self.logger.error(f"Error performing analysis: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }

    def step(self):
        """Execute a single simulation step with comprehensive integration and monitoring."""
        try:
            # Ensure we have a step counter
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0
            self.step_counter += 1

            # Initialize emergent potential field if not existing
            if not hasattr(self, 'emergent_potential_field'):
                from emergent_potential import EmergentPotentialField
                self.emergent_potential_field = EmergentPotentialField()
                self.logger.info("Initialized Emergent Potential Field")

            # Apply minimum mutation to prevent stasis
            self.ensure_minimum_mutation()

            # CRITICAL FIX - DIRECT QUANTUM STATE INTERVENTION
            # Force statevector modification to break out of static state
            if hasattr(self, 'quantum_state') and hasattr(self.quantum_state, 'statevector'):
                # Get current statevector
                if isinstance(self.quantum_state.statevector, np.ndarray):
                    current_state = self.quantum_state.statevector.copy()
                elif hasattr(self.quantum_state.statevector, 'data'):
                    try:
                        current_state = np.array(self.quantum_state.statevector.data)
                    except:
                        # Create a fresh statevector if all else fails
                        from qiskit.quantum_info import Statevector
                        self.quantum_state.statevector = Statevector.from_label('0' * self.quantum_state.num_qubits)
                        current_state = np.array(self.quantum_state.statevector.data)

                # Apply meaningful modification to break stasis
                # Add a small random phase to each amplitude
                random_phases = np.exp(1j * 0.1 * np.random.random(current_state.shape))
                modified_state = current_state * random_phases

                # Normalize the modified state
                norm = np.linalg.norm(modified_state)
                if norm > 0:
                    modified_state = modified_state / norm

                # Update the statevector directly
                try:
                    from qiskit.quantum_info import Statevector
                    self.quantum_state.statevector = Statevector(modified_state)
                except:
                    # Fallback to direct array assignment
                    self.quantum_state.statevector = modified_state

            # Verify system state periodically (every 20 steps)
            if self.step_counter % 20 == 0:
                if hasattr(self, 'verify_system_state'):
                    self.verify_system_state()

            # Apply quantum operations to ensure state evolution
            # Apply different gates based on step to create variation
            qubit_indices = list(range(self.quantum_state.num_qubits))
            random.shuffle(qubit_indices)  # Add randomization

            # Mix different quantum operations for better evolution dynamics
            if self.step_counter % 3 == 0:
                # Apply Hadamard gates periodically to increase entropy
                for q in qubit_indices[:2]:  # Apply to subset of qubits
                    self.quantum_state.apply_gate('h', [q])
            elif self.step_counter % 3 == 1:
                # Apply rotation gates with dynamic angles
                angle = 0.1 * np.pi * (0.5 + 0.5 * np.sin(self.step_counter * 0.1))
                for q in qubit_indices[:2]:
                    self.quantum_state.apply_gate('rx', [q], {'theta': angle})
            else:
                # Apply phase rotation with varying angle
                angle = 0.1 * np.pi * (0.5 + 0.5 * np.cos(self.step_counter * 0.07))
                for q in qubit_indices[:2]:
                    self.quantum_state.apply_gate('rz', [q], {'phi': angle})

            # Always apply a small phase shift with varying angle to ensure evolution
            phase_angle = 0.05 * np.pi * np.sin(self.step_counter * 0.05)
            self.quantum_state.apply_phase_shift(phase_angle)

            # Force quantum state update
            if hasattr(self.quantum_state, 'update_state'):
                self.quantum_state.update_state()

            # Force entropy and coherence recalculation
            if hasattr(self.quantum_state, 'update_phase_coherence'):
                self.quantum_state.update_phase_coherence()

            # Break out of extreme entropy states if needed
            try:
                metrics = self.quantum_state.get_quantum_metrics()
                if abs(metrics['normalized_entropy'] - 1.0) < 0.01 or metrics['normalized_entropy'] < 0.001:
                    # Apply a series of gates to shift entropy away from extremes
                    for i, q in enumerate(qubit_indices[:3]):
                        if i % 3 == 0:
                            self.quantum_state.apply_gate('h', [q])
                        elif i % 3 == 1:
                            self.quantum_state.apply_gate('x', [q])
                        else:
                            angle = 0.3 * np.pi * np.random.random()
                            self.quantum_state.apply_gate('rx', [q], {'theta': angle})

                    # Apply controlled operation between qubits
                    if len(qubit_indices) >= 2:
                        self.quantum_state.apply_gate('cx', [qubit_indices[0], qubit_indices[1]])

                    # Force update again
                    self.quantum_state.update_state()
                    # Re-check metrics
                    metrics = self.quantum_state.get_quantum_metrics()
            except Exception as entropy_error:
                self.logger.warning(f"Error adjusting entropy: {entropy_error}")

            # Now execute the adaptation pipeline
            if hasattr(self, 'adapt'):
                self.adapt()

            # Process recovery if active
            if hasattr(self, 'recovery_mode') and self.recovery_mode:
                if not hasattr(self, 'recovery_steps'):
                    self.recovery_steps = 10  # Default if not set
                self.recovery_steps -= 1
                if self.recovery_steps <= 0:
                    self.recovery_mode = False
                    if hasattr(self, 'stability_factor'):
                        self.stability_factor = min(self.stability_factor * 1.5, 1.0)
                    if hasattr(self, 'logger'):
                        self.logger.info("Exiting recovery mode.")
                    else:
                        print("Exiting recovery mode.")

            # Get current metrics - with error handling
            try:
                metrics = self.quantum_state.get_quantum_metrics()

                # Safety check for extreme entropy values
                if abs(metrics['normalized_entropy'] - 1.0) < 0.01:
                    metrics['normalized_entropy'] = 0.95  # Slightly away from 1.0
                elif metrics['normalized_entropy'] < 0.01:
                    metrics['normalized_entropy'] = 0.05  # Slightly above 0.0
            except Exception as metrics_error:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error getting quantum metrics: {metrics_error}")
                else:
                    print(f"Error getting quantum metrics: {metrics_error}")
                metrics = {
                    'phase_coherence': 0.5,
                    'normalized_entropy': 0.5,
                    'phase': 0.0,
                    'quantum_surplus_coupling': 0.5
                }

            # Process excess stability and emergent potential field
            surplus_metrics = self.surplus_dynamics.get_surplus_metrics()
            if hasattr(self.surplus_dynamics, 'excess_stability_potential'):
                excess_stability = self.surplus_dynamics.excess_stability_potential

                # Register with emergent potential field
                emergence_triggered = self.emergent_potential_field.register_potential(
                    component_id='surplus_dynamics',
                    potential=excess_stability,
                    component_type='surplus',
                    state_metrics=surplus_metrics
                )

                # If emergence was triggered by the potential field
                if emergence_triggered:
                    self.logger.info("\n🔆 EMERGENT POTENTIAL TRIGGERED EMERGENCE EVENT")

                    # Get field state for context
                    field_state = self.emergent_potential_field.get_field_state()

                    # Create emergence event with field information
                    self.handle_emergent_dimension(
                        tensor_shape=(4, 4, 4, field_state['field_intensity']),
                        source="emergent_potential_field"
                    )

                    # Generate special symbolic expression
                    if hasattr(self, 'symbolic_system'):
                        surplus = self.surplus_dynamics.surplus_state.total_surplus()
                        distinction = self.distinction_level
                        coherence = self.quantum_state.phase_coherence

                        expression = self.symbolic_system.handle_post_emergence(
                            surplus=surplus,
                            distinction=distinction,
                            coherence=coherence,
                            dimensionality=int(2 + field_state['field_intensity']),
                            entropy=metrics.get('normalized_entropy', 0.5)
                        )

                        self.logger.info(f"\n🌌 Emergent Potential Field Expression: {expression}")

                # Apply excess stability to ontological field
                if hasattr(self, 'ontological_field'):
                    field_resistance = self.ontological_field.resistance(self.distinction_level)
                    self.ontological_field.adapt_to_agent(
                        self.distinction_level,
                        quantum_coupling=metrics.get('quantum_surplus_coupling', 1.0),
                        field_threshold=0.1,
                        excess_stability=excess_stability
                    )

            # Update field state and print status regularly
            if hasattr(self, 'emergent_potential_field') and self.step_counter % 20 == 0:
                field_state = self.emergent_potential_field.get_field_state()
                self.logger.info(f"\nEmergent Potential Field State: Total={field_state['total_potential']:.4f}, "
                              f"Threshold={field_state['threshold']:.4f}, "
                              f"Emergence Prob={field_state.get('emergence_probability', 0.0):.2f}")

            # Handle symbolic expression logic
            if not hasattr(self, 'expression_cooldown'):
                self.expression_cooldown = 0
                self.expression_cooldown_period = 50  # Default if not set

            if self.expression_cooldown <= 0:
                try:
                    if hasattr(self, '_check_expression_triggers'):
                        should_express = self._check_expression_triggers(metrics)
                    else:
                        # Default expression trigger logic if method missing
                        random_factor = 0.1 if self.step_counter < 100 else 0.02
                        should_express = random.random() < random_factor

                    if should_express:
                        if hasattr(self, 'generate_advanced_symbolic_expression'):
                            expression = self.generate_advanced_symbolic_expression()
                        else:
                            # Fallback to basic symbolic generation
                            expression = self._generate_basic_symbolic_expression(metrics)

                        if hasattr(self, 'logger'):
                            self.logger.info(f"\n🔮 Symbolic Expression: {expression}")
                        else:
                            print(f"\n🔮 Symbolic Expression: {expression}")

                        # Get cognitive state with proper error handling
                        try:
                            cognitive_state = self.recursive_cognition.get_cognitive_state() if hasattr(self, 'recursive_cognition') else {}
                        except Exception:
                            cognitive_state = {}

                        # Check for significance and add analysis if appropriate
                        if (self.distinction_level > 0.8 or
                            metrics.get('phase_coherence', 0.5) < 0.3 or
                            cognitive_state.get('collapse_probability', 0.0) > 0.5):

                            # Generate analysis with patterns
                            if hasattr(self, 'symbolic_system') and hasattr(self.symbolic_system, 'analyze_emergence_patterns'):
                                try:
                                    patterns = self.symbolic_system.analyze_emergence_patterns()
                                    if patterns and 'dominant_patterns' in patterns:
                                        dominant = patterns.get('dominant_patterns', {})
                                        analysis = (f"\n📊 Pattern Analysis: {dominant.get('descriptor', 'Unknown')} "
                                                  f"pattern with {patterns.get('component_diversity', {}).get('overall', 0):.3f} diversity")
                                        if hasattr(self, 'logger'):
                                            self.logger.info(analysis)
                                        else:
                                            print(analysis)
                                except Exception as pattern_error:
                                    if hasattr(self, 'logger'):
                                        self.logger.warning(f"Error analyzing patterns: {pattern_error}")
                                    else:
                                        print(f"Error analyzing patterns: {pattern_error}")

                        self.expression_cooldown = self.expression_cooldown_period
                except Exception as expr_error:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Error in symbolic expression generation: {expr_error}")
                    else:
                        print(f"Error in symbolic expression generation: {expr_error}")
                    self.expression_cooldown = 10  # Shorter cooldown after error
            else:
                self.expression_cooldown -= 1

            # Update state with ontological field interaction - with error handling
            try:
                if hasattr(self, 'ontological_field'):
                    field_resistance = self.ontological_field.resistance(self.distinction_level)
                    self.ontological_field.adapt_to_agent(
                        self.distinction_level,
                        quantum_coupling=metrics.get('quantum_surplus_coupling', 1.0),
                        field_threshold=0.1
                    )
            except Exception as field_error:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Error in ontological field interaction: {field_error}")
                else:
                    print(f"Error in ontological field interaction: {field_error}")

            # Update analysis history
            try:
                if hasattr(self, 'update_analysis_history'):
                    self.update_analysis_history()
            except Exception as analysis_error:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Error updating analysis history: {analysis_error}")
                else:
                    print(f"Error updating analysis history: {analysis_error}")

            # Interact with emergence tracking
            try:
                if hasattr(self, 'dimension_increase_detected') and self.dimension_increase_detected and hasattr(self, 'emergence_tracker'):
                    # Handle emergence explicitly
                    sample_shape = (4, 4, 4, 4)  # Default shape
                    # Try to get actual shape if possible
                    if hasattr(self, 'current_tensor_shape'):
                        tensor_shape = self.current_tensor_shape
                    elif hasattr(self.quantum_state, 'statevector') and hasattr(self.quantum_state.statevector, 'shape'):
                        tensor_shape = self.quantum_state.statevector.shape
                    else:
                        tensor_shape = sample_shape

                    self.emergence_tracker.record_emergence(
                        tensor_shape=tensor_shape,
                        timestamp=time.time(),
                        resource_usage={'cpu_percent': 0, 'memory_percent': 0},  # Would come from monitoring
                        agent_metrics=metrics
                    )
            except Exception as emergence_error:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Error tracking emergence: {emergence_error}")
                else:
                    print(f"Error tracking emergence: {emergence_error}")

            # Perform periodic analysis every 10 steps
            if self.step_counter % 10 == 0:
                try:
                    if hasattr(self, 'perform_analysis'):
                        self.logger.info("\nPerforming analysis...")
                        results = self.perform_analysis()

                        if results and results.get('status') != 'error':
                            self.logger.info("\nAnalysis Report:")
                            if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'generate_analysis_report'):
                                report = self.analyzer.generate_analysis_report(results)
                                self.logger.info(report)
                            else:
                                # Fallback if generate_analysis_report isn't available
                                for key, value in results.items():
                                    if isinstance(value, (int, float, str, bool)):
                                        self.logger.info(f"{key}: {value}")
                        else:
                            self.logger.info("\nAnalysis results: " + (results.get('status', 'No results') if results else "No results"))
                            if results and 'error_message' in results:
                                self.logger.info(f"Error: {results['error_message']}")
                except Exception as analysis_error:
                    self.logger.warning(f"Error performing analysis: {analysis_error}")
                    traceback.print_exc()
                    self.logger.info("No analysis results available due to error.")

            # Print status periodically
            if self.step_counter % 50 == 0 or self.step_counter == 1:
                status_message = (f"\n[Step {self.step_counter}] "
                              f"Distinction: {self.distinction_level:.3f}, "
                              f"Coherence: {metrics['phase_coherence']:.3f}, "
                              f"Entropy: {metrics['normalized_entropy']:.3f}, "
                              f"Stability: {getattr(self, 'stability_factor', 1.0):.3f}, "
                              f"SurplusStab: {self.surplus_dynamics.surplus_state.stability:.3f}")

                if hasattr(self, 'logger'):
                    self.logger.info(status_message)
                else:
                    print(status_message)

            # Add a helper method for fallback symbolic expressions if needed
            if not hasattr(self, '_generate_basic_symbolic_expression'):
                def _generate_basic_symbolic_expression(self, metrics):
                    """Basic fallback for symbolic expression generation."""
                    descriptors = ["Flux", "Equilibrium", "Distinction", "Recursion", "Convergence", "Divergence"]
                    relations = ["aligns with", "dissolves across", "bends toward", "extends beyond", "contracts into"]
                    concepts = ["stability", "recursion", "entropy", "phase shift", "emergence", "ontology"]

                    descriptor = random.choice(descriptors)
                    relation = random.choice(relations)
                    concept = random.choice(concepts)

                    return f"{descriptor} {relation} {concept}"

                # Add the method to the instance
                setattr(self, '_generate_basic_symbolic_expression', _generate_basic_symbolic_expression.__get__(self, type(self)))

            return True  # Return success

        except Exception as e:
            error_message = f"Error in step execution: {e}"
            if hasattr(self, 'logger'):
                self.logger.error(error_message)
            else:
                print(error_message)
            traceback.print_exc()

            if hasattr(self, 'recovery_mode') and not self.recovery_mode:
                if hasattr(self, 'enter_recovery_mode'):
                    self.enter_recovery_mode()

            return False  # Return failure

    def get_emergent_potential_visualization(self) -> Dict[str, Any]:
        """
        Get visualization data for the emergent potential field.

        Returns:
            Dictionary with visualization data
        """
        try:
            if not hasattr(self, 'emergent_potential_field'):
                return {'error': 'Emergent potential field not initialized'}

            # Get field state
            field_state = self.emergent_potential_field.get_field_state()

            # Get component contributions
            components = []
            for component_id, data in self.emergent_potential_field.component_potentials.items():
                components.append({
                    'id': component_id,
                    'type': data['component_type'],
                    'potential': data['weighted_potential'],
                    'raw_potential': data['raw_potential'],
                    'timestamp': data['timestamp']
                })

            # Sort components by potential
            components.sort(key=lambda x: x['potential'], reverse=True)

            # Get emergence history
            emergence_events = []
            for event in list(self.emergent_potential_field.emergence_history)[-10:]:  # Last 10 events
                emergence_events.append({
                    'timestamp': event['timestamp'],
                    'intensity': event['intensity'],
                    'potential': event['potential'],
                    'threshold': event['threshold'],
                    'sequence': event['sequence_number']
                })

            # Prepare visualization data
            visualization_data = {
                'field_state': field_state,
                'components': components,
                'emergence_events': emergence_events,
                'threshold_history': [
                    {
                        'timestamp': entry['timestamp'],
                        'threshold': entry['threshold'],
                        'total_potential': entry['total_potential']
                    }
                    for entry in list(self.emergent_potential_field.potential_history)[-20:]  # Last 20 entries
                ]
            }

            # If we have cognitive state, add correlation data
            if hasattr(self, 'recursive_cognition'):
                cognitive_state = self.recursive_cognition.get_cognitive_state()
                visualization_data['cognitive_correlation'] = {
                    'collapse_probability': cognitive_state.get('collapse_probability', 0.0),
                    'mean_stability': cognitive_state.get('mean_stability', 0.0),
                    'emergence_probability': field_state.get('emergence_probability', 0.0),
                    'field_stability': field_state.get('stability_factor', 1.0)
                }

            return visualization_data

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error getting emergent potential visualization: {e}")
            else:
                print(f"Error getting emergent potential visualization: {e}")
            return {'error': str(e)}

    def _check_expression_triggers(self, metrics: Dict[str, float]) -> bool:
        """
        Check if the agent should generate a symbolic expression based on
        comprehensive state evaluation across multiple systems.

        Args:
            metrics: Dictionary of quantum metrics

        Returns:
            Boolean indicating whether an expression should be generated
        """
        try:
            # Track distinction change
            distinction_change = abs(self.distinction_level - self.previous_distinction)
            self.previous_distinction = self.distinction_level

            # Get cognitive state for enhanced detection
            cognitive_state = self.recursive_cognition.get_cognitive_state()
            collapse_probability = cognitive_state.get('collapse_probability', 0.0)
            cognitive_stability = cognitive_state.get('mean_stability', 0.5)

            # Extract key metrics
            coherence = metrics.get('phase_coherence', 0.5)
            entropy = metrics.get('normalized_entropy', 0.5)
            phase_stability = metrics.get('phase_stability', 1.0)

            # Emergence and dimension detection
            dimension_change = self.dimension_increase_detected
            emergence_active = hasattr(self, 'emergence_tracker') and getattr(self.emergence_tracker, 'is_emergence_active', False)

            # Calculate enriched triggers using cognitive state

            # Critical cognitive transitions
            cognitive_transition = False
            if hasattr(self, 'recursive_cognition') and hasattr(self.recursive_cognition, 'history'):
                if len(self.recursive_cognition.history) >= 2:
                    # Compare current and previous cognitive states
                    current = self.recursive_cognition.get_cognitive_state()
                    previous = self.recursive_cognition.history[-1]

                    # Check for significant changes in key metrics
                    strength_change = abs(current.get('mean_strength', 1.0) -
                                        getattr(previous[0], 'strength', 1.0))
                    identity_change = abs(current.get('mean_identity', 1.0) -
                                        getattr(previous[0], 'core_identity', 1.0))

                    cognitive_transition = (strength_change > 0.2 or identity_change > 0.2)

            # Enhanced detection criteria
            significant_distinction_change = distinction_change > self.expression_threshold
            high_coherence = coherence > 0.7 and entropy < 0.4
            low_coherence = coherence < 0.3 and entropy > 0.6
            high_entropy = entropy > 0.7 and coherence < 0.5
            instability_risk = phase_stability < 0.3 or cognitive_stability < 0.3
            collapse_risk = collapse_probability > COLLAPSE_DISSIPATION_THRESHOLD * 0.8

            # System state changes
            system_change = (
                self.recovery_mode or
                self.collapse_prevention_active or
                emergence_active
            )

            # Periodic expression with randomness for more natural timing
            base_interval = self.expression_periodic_interval
            random_factor = random.random() * 0.2 + 0.9  # 0.9-1.1 multiplier
            adjusted_interval = int(base_interval * random_factor)
            periodic_trigger = (self.step_counter % adjusted_interval == 0)

            # Decision to express with weighting for more interesting expressions
            should_express = any([
                significant_distinction_change,  # Basic trigger
                high_coherence and random.random() < 0.7,  # High coherence is interesting but not always
                low_coherence and random.random() < 0.9,  # Low coherence is usually interesting
                high_entropy and random.random() < 0.8,   # High entropy is often interesting
                instability_risk and random.random() < 0.8,  # Instability is often interesting
                collapse_risk,  # Always interesting
                cognitive_transition and random.random() < 0.9,  # Cognitive shifts are important
                dimension_change,  # Always express on dimension changes
                emergence_active and random.random() < 0.8,  # Often express during emergence
                system_change and random.random() < 0.7,  # Sometimes express on system changes
                periodic_trigger  # Regular expressions
            ])

            return should_express

        except Exception as e:
            self.logger.error(f"Error checking expression triggers: {e}")
            return False

    def generate_advanced_symbolic_expression(self) -> str:
        """
        Generate a sophisticated symbolic expression that integrates multiple
        components of the agent's cognitive state, quantum state, and surplus dynamics.

        Returns:
            A symbolic expression representing the agent's current ontological state
        """
        try:
            # Get comprehensive metrics from different systems
            quantum_metrics = self.quantum_state.get_quantum_metrics()
            cognitive_state = self.recursive_cognition.get_cognitive_state()
            surplus_metrics = self.surplus_dynamics.get_surplus_metrics()

            # Extract key values with proper error handling
            coherence = quantum_metrics.get('phase_coherence', 0.5)
            entropy = quantum_metrics.get('normalized_entropy', 0.5)
            phase_stability = quantum_metrics.get('phase_stability', 1.0)
            distinction = self.distinction_level

            # Cognitive components
            cognitive_strength = cognitive_state.get('mean_strength', 1.0)
            cognitive_stability = cognitive_state.get('mean_stability', 0.5)
            collapse_probability = cognitive_state.get('collapse_probability', 0.0)
            mean_identity = cognitive_state.get('mean_identity', 1.0)
            quantum_influence = cognitive_state.get('quantum_influence', 0.5)

            # Get dimensionality if available
            dimensionality = None
            if hasattr(self, 'dimension_monitor') and hasattr(self.dimension_monitor, 'last_dimensionality'):
                dimensionality = self.dimension_monitor.last_dimensionality

            # Get causality metrics if available
            directionality = 0.0
            causality_strength = 0.0
            if hasattr(self, 'analyzer') and hasattr(self.analyzer, 'causality_analysis'):
                directionality = getattr(self.analyzer.causality_analysis, 'directionality', 0.0)
                causality_strength = getattr(self.analyzer.causality_analysis, 'strength', 0.0)

            # Calculate enriched parameters for symbolic system
            # Surplus is weighted by cognitive components
            enriched_surplus = self.surplus_dynamics.surplus_state.total_surplus() * (
                0.6 + 0.4 * cognitive_strength + 0.2 * quantum_influence
            )

            # Distinction is modulated by cognitive factors
            enriched_distinction = distinction * (
                0.7 + 0.3 * mean_identity + 0.1 * (1.0 - collapse_probability)
            )

            # Coherence is enhanced by stability factors
            enriched_coherence = coherence * (
                0.8 + 0.2 * cognitive_stability + 0.1 * phase_stability
            )

            # Generate expression using enriched parameters
            expression = self.symbolic_system.generate_symbolic_expression(
                surplus=enriched_surplus,
                distinction=enriched_distinction,
                coherence=enriched_coherence,
                entropy=entropy,
                dimensionality=dimensionality
            )

            # Store comprehensive metrics with expression for later analysis
            self.symbolic_history.append({
                'expression': expression,
                'step': self.step_counter,
                'distinction': distinction,
                'enriched_distinction': enriched_distinction,
                'coherence': coherence,
                'enriched_coherence': enriched_coherence,
                'entropy': entropy,
                'dimensionality': dimensionality,
                'collapse_probability': collapse_probability,
                'mean_strength': cognitive_strength,
                'mean_stability': cognitive_stability,
                'mean_identity': mean_identity,
                'quantum_influence': quantum_influence,
                'directionality': directionality,
                'causality_strength': causality_strength,
                'timestamp': time.time()
            })

            return expression

        except Exception as e:
            self.logger.error(f"Error generating advanced symbolic expression: {e}")
            traceback.print_exc()
            return "Flux aligns with stability."  # Safe default

    def get_symbolic_insights(self, limit: int = 5) -> Dict[str, Any]:
        """
        Get insights from symbolic expressions history with pattern analysis
        for a deeper understanding of the agent's cognitive evolution.

        Args:
            limit: Maximum number of recent expressions to analyze

        Returns:
            Dictionary with symbolic insights and patterns
        """
        try:
            if not hasattr(self, 'symbolic_history') or not self.symbolic_history:
                return {"error": "No symbolic history available"}

            # Get pattern analysis
            patterns = self.symbolic_system.analyze_emergence_patterns()

            # Get recent expressions
            recent = list(self.symbolic_history)[-limit:]

            # Calculate expression trajectory
            trajectory = "stable"
            if len(recent) >= 3:
                # Analyze distinction and coherence trends
                distinction_trend = [entry.get('distinction', 0.5) for entry in recent]
                coherence_trend = [entry.get('coherence', 0.5) for entry in recent]

                # Simple trajectory classification
                d_trend = sum(np.diff(distinction_trend)) / (len(distinction_trend) - 1) if len(distinction_trend) > 1 else 0
                c_trend = sum(np.diff(coherence_trend)) / (len(coherence_trend) - 1) if len(coherence_trend) > 1 else 0

                if d_trend > 0.05 and c_trend > 0.05:
                    trajectory = "ascending"
                elif d_trend < -0.05 and c_trend < -0.05:
                    trajectory = "descending"
                elif abs(d_trend) > 0.1 or abs(c_trend) > 0.1:
                    trajectory = "oscillating"

            # Extract pattern information
            dominant_pattern = "none"
            pattern_stability = 0.0
            component_diversity = 0.0

            if patterns:
                dominant_pattern = patterns.get('dominant_patterns', {}).get('descriptor', 'none')
                pattern_stability = patterns.get('coherence_stability', 0.0)
                component_diversity = patterns.get('component_diversity', {}).get('overall', 0.0)

            # Construct insights
            insights = {
                "recent_expressions": [entry.get('expression') for entry in recent],
                "expression_count": len(self.symbolic_history),
                "trajectory": trajectory,
                "dominant_pattern": dominant_pattern,
                "pattern_stability": pattern_stability,
                "component_diversity": component_diversity,
                "expression_frequency": len(self.symbolic_history) / max(1, self.step_counter),
                "patterns": patterns
            }

            return insights

        except Exception as e:
            self.logger.error(f"Error getting symbolic insights: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def _log_step_summary(self, metrics: Dict[str, float],
                          surplus_state: SurplusState,
                          field_resistance: float):
        """Log comprehensive step summary."""
        try:
            self.logger.info(f"\nStep Summary:")
            self.logger.info(f"Coherence: {metrics.get('phase_coherence', 0.0):.4f}")
            self.logger.info(f"Distinction: {self.distinction_level:.4f}")
            self.logger.info(f"Stability: {self.stability_factor:.4f}")
            self.logger.info(f"Field Resistance: {field_resistance:.4f}")

            # Log special modes
            if self.collapse_prevention_active:
                self.logger.info("Collapse Prevention: Active")
            if self.recovery_mode:
                self.logger.info(f"Recovery Mode: {self.recovery_steps} steps remaining")

            # Log surplus state
            self.logger.info("\nSurplus State:")
            for key, value in surplus_state.values.items():
                self.logger.info(f"  {key}: {value:.4f}")
        except Exception as e:
            self.logger.error(f"Error logging step summary: {e}")

    def _generate_symbolic_expression(self) -> None:
        """Generate and log a symbolic expression based on internal state."""
        try:
            # Get current metrics
            metrics = self.quantum_state.get_quantum_metrics()

            # Get cognitive state for enhanced expressions
            cognitive_state = self.recursive_cognition.get_cognitive_state()

            # Generate expression using surplus, distinction, and coherence from the agent itself
            surplus = self.surplus_dynamics.surplus_state.total_surplus()
            distinction = self.distinction_level
            coherence = metrics.get('phase_coherence', 0.5)
            entropy = metrics.get('normalized_entropy', 0.5)

            # Get dimensionality if available
            dimensionality = None
            if hasattr(self, 'dimension_monitor') and hasattr(self.dimension_monitor, 'last_dimensionality'):
                dimensionality = self.dimension_monitor.last_dimensionality

            # Generate the expression
            expression = self.symbolic_system.generate_symbolic_expression(
                surplus, distinction, coherence, entropy, dimensionality
            )

            # Store expression with enhanced metrics
            self.last_symbolic_expression = expression
            self.symbolic_history.append({
                'expression': expression,
                'step': self.step_counter,
                'distinction': distinction,
                'coherence': coherence,
                'entropy': entropy,
                'dimensionality': dimensionality,
                'collapse_probability': cognitive_state.get('collapse_probability', 0.0),
                'mean_strength': cognitive_state.get('mean_strength', 1.0),
                'mean_stability': cognitive_state.get('mean_stability', 0.5),
                'mean_identity': cognitive_state.get('mean_identity', 1.0),
                'timestamp': time.time()
            })

            # Log the expression
            self.logger.info(f"\n🔮 Symbolic Expression: {expression}")

            # If expression seems particularly significant (high distinction or low stability),
            # generate an additional analysis
            if (distinction > 0.8 or
                cognitive_state.get('mean_stability', 1.0) < 0.3 or
                entropy > 0.7 or
                cognitive_state.get('collapse_probability', 0.0) > 0.5):
                patterns = self.symbolic_system.analyze_emergence_patterns()
                if patterns and 'dominant_pattern' in patterns:
                    analysis = f"\n📊 Pattern Analysis: {patterns.get('dominant_pattern', 'Unknown')} pattern emerging with {patterns.get('coherence_stability', 0.0):.3f} stability."
                    self.logger.info(analysis)

        except Exception as e:
            self.logger.error(f"Error generating symbolic expression: {e}")

    def get_symbolic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the recent symbolic expression history.

        Args:
            limit: Maximum number of expressions to return

        Returns:
            List of recent expressions with associated metrics
        """
        try:
            if not hasattr(self, 'symbolic_history'):
                return []

            recent = list(self.symbolic_history)[-limit:]
            return recent

        except Exception as e:
            self.logger.error(f"Error getting symbolic history: {e}")
            return []

    def _prepare_experience(self, prediction: float) -> Dict[str, Any]:
        """Prepare experience for training with enhanced data and dimension handling."""
        try:
            # Get input tensor
            input_tensor = self.prepare_transformer_input()

            # Handle dimensionality properly
            has_emergent_dim = False
            if hasattr(input_tensor, 'dim') and input_tensor.dim() > 3:
                # For emergent dimensions, store the full tensor shape
                # but convert to numpy in a safe way that preserves dimensionality
                state_np = input_tensor.cpu().detach().numpy()
                has_emergent_dim = True

                # Log the emergence handling
                self.logger.debug(f"Preparing experience with emergent dimensions: {input_tensor.shape}")
            else:
                # Standard case
                state_np = input_tensor.cpu().numpy()

            return {
                'state': state_np,
                'prediction': float(prediction),
                'actual': float(self.distinction_level),
                'quantum_metrics': self.quantum_state.get_quantum_metrics(),
                'stability': float(self.stability_factor),
                'adaptation_momentum': float(self.adaptation_momentum),
                'has_emergent_dim': has_emergent_dim
            }
        except Exception as e:
            self.logger.error(f"Error preparing experience: {e}")
            # Return safe default
            return {
                'state': np.zeros((1, 1, 20)),
                'prediction': 0.0,
                'actual': 0.5,
                'quantum_metrics': {},
                'stability': 1.0,
                'adaptation_momentum': 0.0,
                'has_emergent_dim': False
            }

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive state summary with proper integration of all components.

        Returns:
            Dictionary containing current state metrics
        """
        try:
            # Get basic metrics
            metrics = self.quantum_state.get_quantum_metrics()
            distinction_mean = np.mean(list(self.distinction_history)) if self.distinction_history else self.distinction_level

            # Create core state summary
            summary = {
                'distinction_level': self.distinction_level,
                'distinction_mean': distinction_mean,
                'stability_factor': self.stability_factor,
                'adaptation_momentum': self.adaptation_momentum,
                'collapse_prevention_active': self.collapse_prevention_active,
                'recovery_mode': self.recovery_mode,
                'quantum_metrics': metrics,
                'phase': self.phase,
                'coherence': metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR)
            }

            # Add surplus state if available
            if hasattr(self, 'surplus_dynamics') and hasattr(self.surplus_dynamics, 'surplus_state'):
                summary['surplus'] = self.surplus_dynamics.surplus_state.copy()

            # Add cognitive state if available
            if hasattr(self, 'recursive_cognition'):
                summary['cognitive_state'] = self.recursive_cognition.get_cognitive_state()

            # Add training summary if available
            if hasattr(self, 'training_pipeline'):
                summary['training_summary'] = self.training_pipeline.get_training_summary()

            # Add learning rate if available
            if hasattr(self, 'learning_rate'):
                summary['learning_rate'] = self.learning_rate

            # Add emergent potential field data if available
            if hasattr(self, 'emergent_potential_field'):
                field_state = self.emergent_potential_field.get_field_state()
                summary['emergent_potential'] = {
                    'total_potential': field_state.get('total_potential', 0.0),
                    'emergence_probability': field_state.get('emergence_probability', 0.0),
                    'emergence_active': field_state.get('emergence_active', False),
                    'field_intensity': field_state.get('field_intensity', 1.0)
                }

            return summary
        except Exception as e:
            print(f"Error getting state summary: {e}")
            # Return minimal state information
            return {
                'distinction_level': getattr(self, 'distinction_level', 0.5),
                'error': str(e)
            }

    def _initialize_transformer(self) -> nn.Module:
        """Initialize transformer with 4D input adapter."""
        print("Initializing transformer with 4D input adapter...")

        try:
            # Create base transformer
            base_transformer = RecursiveDistinctionTransformer(
                input_size=20,
                d_model=20,
                nhead=NUM_TRANSFORMER_HEADS,
                num_layers=NUM_TRANSFORMER_LAYERS,
                output_size=1
            ).to(DEVICE)

            # Initialize weights
            for p in base_transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.1)

            # Wrap with adapter
            transformer = FourDimTransformerAdapter(base_transformer, merge_strategy="merge")
            print("Transformer initialized with adaptive 4D handling.")
            return transformer

        except Exception as e:
            print(f"Error initializing transformer: {e}")
            traceback.print_exc()

            # Create fallback transformer
            fallback_transformer = nn.Sequential(
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 1),
                nn.Sigmoid()
            ).to(DEVICE)

            print("⚠️ Using fallback transformer due to initialization error")

            # Wrap with basic adapter to handle 4D inputs
            class BasicAdapter(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model

                def forward(self, x):
                    # Handle 4D inputs by flattening to 2D
                    if x.dim() > 2:
                        # Preserve batch dimension
                        batch_size = x.size(0)
                        # Flatten all other dimensions
                        x = x.view(batch_size, -1)
                        # Ensure last dimension is 20
                        if x.size(-1) != 20:
                            x = F.pad(x, (0, 20 - x.size(-1) % 20))[:, :20]
                    # Create output structure similar to TransformerOutput
                    output = self.base_model(x)
                    # Structure to mimic TransformerOutput
                    class SimpleOutput:
                        def __init__(self, pred):
                            self.prediction = pred
                            self.phase_prediction = pred * 0.5
                            self.value_estimate = pred * 0.8
                            self.entropy = torch.tensor(0.5, device=pred.device)
                            self.coherence_estimate = torch.tensor(0.5, device=pred.device)
                            self.attention_weights = {}

                    return SimpleOutput(output)

            return BasicAdapter(fallback_transformer)

# =============================================================================
# Final Agent Class
# =============================================================================
class EnhancedSingleAgentFinal(EnhancedSingleAgentBase):
    """
    Final enhanced agent class that integrates quantum state management,
    recursive memory, transformer-based decision making, surplus regulation,
    and advanced adaptation mechanisms.
    """
    def __init__(self, num_qubits: int = NUM_QUBITS_PER_AGENT):
        try:
            # Core components
            self.num_qubits = num_qubits
            self.quantum_state = EnhancedQuantumState(num_qubits)
            self.distinction_dynamics = EnhancedDistinctionDynamics()
            self.surplus_dynamics = EnhancedSurplusDynamics()
            print(f"DEBUG: Surplus Dynamics Initialized - Surplus State: {self.surplus_dynamics.surplus_state}")
            self.recursive_cognition = RecursiveCognitiveStructuring()
            self.quantum_optimizer = EnhancedQuantumSelfOptimization(num_qubits)
            self.memory = RecursiveDistinctionMemory(max_size=10000, hierarchy_levels=4)
            self.error_recovery = EnhancedErrorRecovery(agent=self)
            self.minimum_coherence = MINIMUM_COHERENCE_FLOOR

            # Validation and synchronization
            self.state_validator = QuantumStateValidator()
            self.sync_manager = StateSynchronizationManager(
                self.quantum_state,
                self.surplus_dynamics,
                self.distinction_dynamics
            )

            # Create a base transformer and assert input size matches d_model
            assert HIDDEN_DIM == 20, "ERROR: Transformer `d_model` must match input_size (20)"
            base_transformer = RecursiveDistinctionTransformer(
                input_size=20,
                d_model=20,
                nhead=NUM_TRANSFORMER_HEADS,
                num_layers=NUM_TRANSFORMER_LAYERS
            ).to(DEVICE)

            # Wrap with 4D adapter
            self.transformer = FourDimTransformerAdapter(
                base_transformer,
                merge_strategy="merge"  # or "separate" if you prefer
            )

            # Training components
            self.training_pipeline = EnhancedTrainingPipeline(self.transformer)

            # Analysis components
            self.analyzer = QuantumAnalyzer(num_qubits)
            self.optimization_coordinator = OptimizationCoordinator(self)

            # State tracking
            self.distinction_level = 0.5  # Start with median distinction
            self.phase = 0.0
            self.stability_factor = 1.0
            self.learning_rate = LEARNING_RATE

            # Recovery mode
            self.recovery_mode = False
            self.recovery_steps = 0

            # Ensure initial synchronization
            if not self.sync_manager.synchronize_states():
                print("Warning: Initial state synchronization failed, attempting recovery")
                self.error_recovery.initiate_full_recovery()
                if not self.sync_manager.synchronize_states():
                    raise RuntimeError("Failed to achieve initial state synchronization")

        except Exception as e:
            print(f"Error initializing agent: {e}")
            traceback.print_exc()
            raise
