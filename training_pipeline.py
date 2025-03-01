"""
Training Pipeline Module
-----------------------
This module provides classes for training the Émile-2 agent's neural networks,
including optimizers, loss functions, error recovery, and state validation.
"""

import os
import math
import time
import random
import traceback
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, Tuple, List, Any, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SaveStatevector

from data_classes import SurplusState, TransformerOutput
from utilities import (
    MOMENTUM_DECAY,
    NUM_QUBITS_PER_AGENT,
    MINIMUM_COHERENCE_FLOOR,
    DISTINCTION_ANCHOR_WEIGHT,
    update_momentum,
    compute_phase_coherence,
    LEARNING_RATE,
    LEARNING_RATE_MIN,
    LEARNING_RATE_MAX,
    WEIGHT_DECAY,
    GRADIENT_CLIP_VALUE,
    NUM_TRANSFORMER_HEADS,
    NUM_TRANSFORMER_LAYERS,
    HIDDEN_DIM
)

# Determine if CUDA (GPU) is available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Transformer Wrapper
# -----------------------------------------------------------------------------
class EnhancedTransformerWrapper(nn.Module):
    """
    Wrapper to ensure consistent transformer output by converting any output type
    to a standardized TransformerOutput object.
    """
    def __init__(self, base_transformer: nn.Module):
        super().__init__()
        self.base_transformer = base_transformer

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor] = None) -> TransformerOutput:
        """
        Forward pass that ensures output is always a TransformerOutput object.
        Handles various output types from the base transformer.
        """
        try:
            # Get base output
            base_output = self.base_transformer(x, phase)

            # If already TransformerOutput, return as is
            if isinstance(base_output, TransformerOutput):
                return base_output

            # If tensor, wrap in TransformerOutput
            if isinstance(base_output, torch.Tensor):
                return TransformerOutput(
                    prediction=base_output,
                    phase_prediction=phase,
                    value_estimate=torch.tensor(0.0, device=base_output.device),
                    attention_weights={},
                    entropy=torch.tensor(0.0, device=base_output.device),
                    coherence_estimate=torch.tensor(0.0, device=base_output.device)
                )

            # If dict, convert to TransformerOutput
            if isinstance(base_output, dict):
                device = self.get_device()
                return TransformerOutput(
                    prediction=base_output.get('prediction', torch.zeros(1, device=device)),
                    phase_prediction=base_output.get('phase_prediction', None),
                    value_estimate=base_output.get('value_estimate', None),
                    attention_weights=base_output.get('attention_weights', {}),
                    entropy=base_output.get('entropy', None),
                    coherence_estimate=base_output.get('coherence_estimate', None)
                )

            raise ValueError(f"Unexpected output type: {type(base_output)}")

        except Exception as e:
            print(f"Error in transformer forward pass: {e}")
            traceback.print_exc()
            return self._create_default_output()

    def _create_default_output(self) -> TransformerOutput:
        """Create default output to use as fallback on error."""
        device = self.get_device()
        return TransformerOutput(
            prediction=torch.zeros(1, device=device),
            phase_prediction=torch.tensor(0.0, device=device),
            value_estimate=torch.tensor(0.0, device=device),
            attention_weights={},
            entropy=torch.tensor(0.0, device=device),
            coherence_estimate=torch.tensor(0.0, device=device)
        )

    def get_device(self) -> torch.device:
        """Get device of base transformer."""
        try:
            return next(self.base_transformer.parameters()).device
        except Exception:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# Metric Validator
# -----------------------------------------------------------------------------
class MetricValidator:
    """
    Validates and cleans metrics dictionaries, ensuring all required metrics
    are present with valid values.
    """
    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and clean metrics dictionary with better error handling.
        Ensures all required metrics have valid values.
        """
        try:
            if not isinstance(metrics, dict):
                print("Warning: Metrics is not a dictionary, creating new one")
                metrics = {}

            # Define required metrics with defaults
            default_metrics = {
                'phase_coherence': 0.5,
                'normalized_entropy': 0.5,
                'stability': 1.0,
                'quantum_coupling': 1.0,
                'phase': 0.0,
                'phase_distinction': 0.5,
                'coherence_distinction': 0.5,
                'quantum_surplus_coupling': 1.0
            }

            validated_metrics = {}

            # Process each metric with proper validation
            for key, default in default_metrics.items():
                try:
                    if key in metrics:
                        value = float(metrics[key])
                        # Bound specific metrics
                        if key in ['phase_coherence', 'normalized_entropy', 'stability', 'quantum_coupling']:
                            value = np.clip(value, 0.0, 1.0)
                        validated_metrics[key] = value
                    else:
                        print(f"Warning: Missing required metric {key}, using default value")
                        validated_metrics[key] = default
                except (TypeError, ValueError) as e:
                    print(f"Warning: Invalid value for {key}, using default. Error: {e}")
                    validated_metrics[key] = default

            # Include any additional metrics not in defaults
            for key, value in metrics.items():
                if key not in validated_metrics:
                    try:
                        validated_metrics[key] = float(value)
                    except (TypeError, ValueError):
                        print(f"Warning: Could not convert {key} to float, skipping")

            return validated_metrics

        except Exception as e:
            print(f"Error in metric validation: {e}")
            traceback.print_exc()
            return default_metrics

# -----------------------------------------------------------------------------
# Quantum-Aware Optimizer
# -----------------------------------------------------------------------------
class QuantumAwareOptimizer(torch.optim.Optimizer):
    """
    Optimizer that wraps an AdamW optimizer and rescales gradients based on
    a moving average and variance of the gradient norms, with quantum-aware
    adjustment based on coherence and stability metrics.
    """
    def __init__(self, model=None, device=None, learning_rate=1e-4, momentum_decay=0.7, gradient_clip_value=1.0):
        """
        Initialize the optimizer with learning rate and gradient tracking.

        Args:
            model: The neural network model being optimized.
            device: The computational device (CPU/GPU).
            learning_rate: Learning rate for the optimizer (default: 1e-4)
            momentum_decay: Decay factor for momentum calculations (default: 0.7)
            gradient_clip_value: Maximum norm for gradient clipping (default: 1.0)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store constants as instance variables for dynamic adjustment
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.gradient_clip_value = gradient_clip_value

        # Initialize parameters list correctly
        if hasattr(self.model, 'parameters'):
            try:
                # Convert parameters to list explicitly
                params = [p for p in self.model.parameters()]
            except:
                params = []
        else:
            params = []

        if not params:
            print("Warning: No parameters found in model for optimization.")

        # Create inner optimizer with default parameters if needed
        self.optimizer = torch.optim.Adam(
            [{'params': params}] if params else [{'params': [torch.nn.Parameter(torch.zeros(1))]}],
            lr=self.learning_rate,
            betas=(0.9, 0.999)
        )

        # Gradient tracking dictionaries
        self.grad_moving_avg = {}
        self.grad_variance = {}
        self.grad_momentum = {}

        # Stability tracking
        self.stability_factor = 1.0
        self.update_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)

        # Recovery attributes
        self.recovery_mode = False
        self.recovery_steps = 0
        self.recovery_lr_scale = 0.1
        self.consecutive_failures = 0
        self.max_failures = 5

    def step(self, loss: torch.Tensor, metrics: Dict[str, float], closure: Optional[callable] = None) -> None:
        """
        Take an optimization step with enhanced error handling and gradient tracking.

        Args:
            loss: The loss tensor to optimize
            metrics: Dictionary of metrics used to adjust optimization
            closure: Optional closure for computing loss (used by some optimizers)
        """
        try:
            # Ensure the loss requires gradient
            if not loss.requires_grad:
                print("Warning: Loss does not require gradient. Creating differentiable copy.")
                loss = loss.detach().requires_grad_(True)

            # Compute gradients
            loss.backward(retain_graph=True)

            with torch.no_grad():
                # Track gradient statistics
                total_grad_norm = 0.0
                param_count = 0

                # Process each parameter's gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param_count += 1
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm ** 2

                            # Initialize tracking if needed
                            if name not in self.grad_moving_avg:
                                self.grad_moving_avg[name] = grad_norm
                                self.grad_variance[name] = 0.0
                                self.grad_momentum[name] = 0.0

                            # Update tracking metrics
                            self.grad_moving_avg[name] = (
                                self.momentum_decay * self.grad_moving_avg[name] +
                                (1 - self.momentum_decay) * grad_norm
                            )

                            grad_diff = grad_norm - self.grad_moving_avg[name]
                            self.grad_variance[name] = (
                                self.momentum_decay * self.grad_variance[name] +
                                (1 - self.momentum_decay) * (grad_diff ** 2)
                            )

                            self.grad_momentum[name] = (
                                self.momentum_decay * self.grad_momentum[name] +
                                (1 - self.momentum_decay) * grad_diff
                            )

                            # Scale gradients based on metrics
                            stability = metrics.get('stability', 1.0)
                            coherence = metrics.get('phase_coherence', 0.5)

                            base_scale = 1.0 / (1.0 + math.sqrt(self.grad_variance[name]))
                            momentum_scale = 1.0 + 0.1 * self.grad_momentum[name]
                            metric_scale = 1.0 + 0.1 * (stability * coherence - 0.5)

                            final_scale = base_scale * momentum_scale * metric_scale
                            param.grad.mul_(final_scale)

                if param_count == 0:
                    print("Warning: No parameters require gradients")
                    return

                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"DEBUG: Total gradient norm (before clipping): {total_norm:.4f}")

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip_value
                )

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store update history
                self.update_history.append({
                    'timestamp': time.time(),
                    'grad_norm': float(total_grad_norm),
                    'loss': float(loss.item()),
                    'stability': float(stability) if 'stability' in locals() else 1.0,
                    'coherence': float(coherence) if 'coherence' in locals() else 0.5
                })

                # Reset failure counter on successful update
                self.consecutive_failures = 0

        except Exception as e:
            print(f"Error in optimizer step: {e}")
            traceback.print_exc()

            # Track failures
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures and not self.recovery_mode:
                self.enter_recovery_mode()

    def enter_recovery_mode(self):
        """Enters recovery mode when training instability is detected."""
        self.recovery_mode = True
        self.recovery_steps = 50

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.recovery_lr_scale

        print("Entering optimizer recovery mode.")

        # Reset momentum
        for name in self.grad_momentum:
            self.grad_momentum[name] *= 0.1

    def __getattr__(self, attr):
        """Delegate attribute access to the underlying optimizer."""
        try:
            return getattr(self.optimizer, attr)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def get_optimizer_stats(self) -> Dict[str, float]:
        """Retrieve statistics about the optimizer's state."""
        try:
            recent_history = list(self.update_history)[-100:]
            stats = {
                'mean_grad_norm': np.mean([h['grad_norm'] for h in recent_history]) if recent_history else 0.0,
                'grad_norm_std': np.std([h['grad_norm'] for h in recent_history]) if recent_history else 0.0,
                'stability_factor': self.stability_factor,
                'recovery_mode': self.recovery_mode,
                'recovery_steps': self.recovery_steps if self.recovery_mode else 0,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'consecutive_failures': self.consecutive_failures
            }

            if self.grad_momentum:
                momentum_values = list(self.grad_momentum.values())
                stats.update({
                    'mean_momentum': float(np.mean(momentum_values)),
                    'momentum_std': float(np.std(momentum_values))
                })

            return stats
        except Exception as e:
            print(f"Error getting optimizer stats: {e}")
            traceback.print_exc()
            return {}

    def process_recovery(self):
        """Handle recovery mode processing and exit."""
        if not self.recovery_mode:
            return

        self.recovery_steps -= 1

        # Exit recovery mode if steps completed
        if self.recovery_steps <= 0:
            self.recovery_mode = False

            # Gradually restore learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(
                    param_group['lr'] * 2.0,
                    self.learning_rate
                )

            print("Exiting optimizer recovery mode")

# -----------------------------------------------------------------------------
# Enhanced Loss Function
# -----------------------------------------------------------------------------
class EnhancedLossFunction:
    """
    Enhanced loss function with proper TransformerOutput handling and component-wise losses
    that are weighted by quantum coherence and stability metrics.
    """
    def __init__(self):
        """Initialize the loss function with tracking history."""
        self.criterion = nn.MSELoss()
        self.loss_history = deque(maxlen=1000)

    def compute_loss(self, output: TransformerOutput,
                    targets: Dict[str, torch.Tensor],
                    metrics: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss with TransformerOutput handling.

        Args:
            output: The transformer output object
            targets: Dictionary of target tensors
            metrics: Dictionary of metrics for weighting

        Returns:
            Tuple of (total loss tensor, dictionary of loss components)
        """
        try:
            device = output.prediction.device

            # Get prediction from TransformerOutput
            prediction = output.prediction
            if prediction.dim() == 3:  # If (batch, seq_len, feature)
                prediction = prediction.squeeze(-1)

            # Get target tensor
            target = targets.get('distinction')
            if target is None:
                print("Warning: No distinction target provided")
                target = torch.zeros_like(prediction)

            # Ensure shapes match
            if target.shape != prediction.shape:
                target = target.view_as(prediction)

            # Compute main distinction loss
            distinction_loss = self.criterion(prediction, target)

            # Compute auxiliary losses if available
            phase_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if output.phase_prediction is not None and 'phase' in targets:
                phase_pred = output.phase_prediction.squeeze(-1)
                phase_target = targets['phase'].view_as(phase_pred)
                phase_loss = self.criterion(phase_pred, phase_target)

            value_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if output.value_estimate is not None and 'value' in targets:
                value_pred = output.value_estimate.squeeze(-1)
                value_target = targets['value'].view_as(value_pred)
                value_loss = self.criterion(value_pred, value_target)

            # Weight losses based on metrics
            coherence = metrics.get('phase_coherence', 0.5)
            stability = metrics.get('stability', 0.5)

            # Combine losses with weights
            total_loss = (
                distinction_loss +
                0.1 * phase_loss * coherence +
                0.1 * value_loss * stability
            )

            # Collect loss components for tracking
            loss_components = {
                'total_loss': float(total_loss.item()),
                'distinction_loss': float(distinction_loss.item()),
                'phase_loss': float(phase_loss.item()),
                'value_loss': float(value_loss.item())
            }

            # Store history
            self.loss_history.append({
                'components': loss_components,
                'metrics': metrics,
                'timestamp': time.time()
            })

            return total_loss, loss_components

        except Exception as e:
            print(f"Error computing loss: {e}")
            traceback.print_exc()

            # Return safe default
            default_loss = torch.tensor(1.0, device=output.prediction.device, requires_grad=True)
            return default_loss, {'total_loss': 1.0, 'error': str(e)}

    def get_loss_stats(self) -> Dict[str, float]:
        """Get statistics about recent losses."""
        try:
            if not self.loss_history:
                return {'no_history': True}

            recent = list(self.loss_history)[-100:]

            return {
                'mean_total_loss': np.mean([r['components']['total_loss'] for r in recent]),
                'mean_distinction_loss': np.mean([r['components']['distinction_loss'] for r in recent]),
                'mean_phase_loss': np.mean([r['components']['phase_loss'] for r in recent]),
                'mean_value_loss': np.mean([r['components']['value_loss'] for r in recent]),
                'loss_std': np.std([r['components']['total_loss'] for r in recent]),
                'recent_loss_trend': np.mean(np.diff([r['components']['total_loss'] for r in recent][-10:]))
                if len(recent) >= 10 else 0.0
            }
        except Exception as e:
            print(f"Error getting loss stats: {e}")
            return {'error': str(e)}

# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------
class EnhancedTrainingPipeline:
    """
    Handles experience buffering, batch preparation, and training steps.
    Tracks loss, gradient norms, and other training statistics.
    """
    def __init__(self, model: nn.Module, batch_size: int = 32, buffer_size: int = 10000):
        """
        Initialize the training pipeline with enhanced validation.

        Args:
            model: The model to train
            batch_size: Batch size for training
            buffer_size: Maximum size of experience buffer
        """
        # Validate model is a proper nn.Module
        if not isinstance(model, nn.Module):
            raise TypeError(f"Model must be an instance of nn.Module, got {type(model)}")

        # Validate batch and buffer sizes
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if buffer_size <= batch_size:
            raise ValueError(f"Buffer size must be greater than batch size, got {buffer_size} <= {batch_size}")

        self.model = model
        self.transformer = model  # Alias for clarity
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Initialize optimizer and loss function
        self.optimizer = QuantumAwareOptimizer(model)
        self.loss_function = EnhancedLossFunction()

        # Experience buffers
        self.experience_buffer = deque(maxlen=buffer_size)
        self.priority_buffer = []
        self.priority_threshold = 0.8

        # Validation and metrics
        self.metric_validator = MetricValidator()

        # Learning rate and training stats
        self.learning_rate = LEARNING_RATE
        self.training_stats = {
            'loss_history': [],
            'gradient_stats': [],
            'priority_stats': [],
            'stability_metrics': []
        }

        # Training state
        self.update_counter = 0
        self.stability_factor = 1.0
        self.training_momentum = 0.0
        self.recovery_mode = False
        self.recovery_steps = 0
        self.min_experiences_for_training = batch_size * 2

        # Stability and recovery settings
        self.stability_threshold = 0.3
        self.consecutive_failures = 0
        self.max_failures = 5

        # Create learning rate scheduler
        self.lr_scheduler = EnhancedLearningRateScheduler(self.optimizer)

        # Log initialization
        print(f"Training pipeline initialized with batch size {batch_size}, buffer size {buffer_size}")
        print(f"Model architecture: {model.__class__.__name__}")

    def add_experience(self, experience: Dict[str, Any], priority: Optional[float] = None) -> None:
        """
        Add an experience to the buffer, optionally with priority.

        Args:
            experience: Dictionary containing experience data
            priority: Optional priority value (higher = more important)
        """
        try:
            if not experience:
                return

            # Calculate priority if not provided
            if priority is None:
                priority = self._compute_experience_priority(experience)

            # Add to priority buffer if above threshold
            if priority > self.priority_threshold:
                self.priority_buffer.append({
                    'experience': experience,
                    'priority': priority,
                    'timestamp': time.time()
                })
                # Sort by priority (highest first) and limit size
                self.priority_buffer.sort(key=lambda x: x['priority'], reverse=True)
                self.priority_buffer = self.priority_buffer[:self.buffer_size // 10]
            else:
                # Otherwise add to regular buffer
                self.experience_buffer.append(experience)

            # Track priority stats
            self.training_stats['priority_stats'].append({
                'priority': priority,
                'buffer_type': 'priority' if priority > self.priority_threshold else 'normal'
            })

        except Exception as e:
            print(f"Error adding experience: {e}")
            traceback.print_exc()

    def adjust_training_with_cognitive_state(self, cognitive_state: Dict[str, float]) -> None:
        """
        Adjust training parameters based on cognitive state metrics.

        Args:
            cognitive_state: Dictionary with cognitive state metrics
        """
        try:
            # Extract relevant metrics
            stability = cognitive_state.get('mean_stability', 0.5)
            collapse_probability = cognitive_state.get('collapse_probability', 0.0)
            feedback_strength = cognitive_state.get('mean_feedback_strength', 0.5)
            quantum_influence = cognitive_state.get('quantum_influence', 0.5)

            # Adjust learning rate based on stability and collapse probability
            current_lr = self.optimizer.param_groups[0]['lr']

            if collapse_probability > 0.3:
                # Reduce learning rate when collapse is likely
                new_lr = current_lr * 0.8
                print(f"Reducing learning rate due to high collapse probability: {current_lr:.6f} -> {new_lr:.6f}")
            elif stability > 0.7 and feedback_strength > 0.6:
                # Slightly increase learning rate when very stable
                new_lr = min(current_lr * 1.05, LEARNING_RATE_MAX)
                print(f"Increasing learning rate due to high stability: {current_lr:.6f} -> {new_lr:.6f}")
            else:
                # Maintain current learning rate with slight adjustments
                momentum_factor = 1.0 + 0.1 * (quantum_influence - 0.5)
                new_lr = current_lr * momentum_factor

            # Apply the new learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = np.clip(new_lr, LEARNING_RATE_MIN, LEARNING_RATE_MAX)

            # Adjust gradient clipping based on stability
            if stability < 0.3:
                # More aggressive clipping for unstable states
                self.gradient_clip_value = GRADIENT_CLIP_VALUE * 0.7
            elif stability > 0.8:
                # Less aggressive clipping for stable states
                self.gradient_clip_value = GRADIENT_CLIP_VALUE * 1.2
            else:
                # Default clipping
                self.gradient_clip_value = GRADIENT_CLIP_VALUE

        except Exception as e:
            print(f"Error adjusting training with cognitive state: {e}")

    def _compute_experience_priority(self, experience: Dict[str, Any]) -> float:
        """
        Compute priority value for an experience based on its characteristics.

        Args:
            experience: The experience dictionary

        Returns:
            Priority value between 0.0 and 1.0
        """
        try:
            priority_factors = []

            # Prediction error
            if 'prediction' in experience and 'actual' in experience:
                priority_factors.append(abs(experience['prediction'] - experience['actual']))

            # Distinction change
            if 'distinction_level' in experience and 'next_distinction' in experience:
                priority_factors.append(abs(experience['next_distinction'] - experience['distinction_level']))

            # Quantum metrics influence
            if 'quantum_metrics' in experience:
                metrics = experience['quantum_metrics']
                coherence = metrics.get('phase_coherence', 0.5)
                entropy = metrics.get('normalized_entropy', 0.5)
                priority_factors.append(coherence * (1 - entropy))

            return float(np.mean(priority_factors)) if priority_factors else 0.5

        except Exception as e:
            print(f"Error computing experience priority: {e}")
            traceback.print_exc()
            return 0.5

    def prepare_batch(self, experiences: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepare a training batch from a list of experiences with robust handling of emergent dimensions.

        Args:
            experiences: List of experience dictionaries

        Returns:
            Tuple of (input tensor, target tensors dictionary)
        """
        try:
            if not experiences:
                return self._create_empty_batch()

            states = []
            targets = {'distinction': [], 'phase': [], 'value': []}

            # Extract data from experiences
            for exp in experiences:
                state = exp.get('state')

                # Skip invalid states
                if not isinstance(state, (np.ndarray, list, torch.Tensor)):
                    print(f"Warning: Invalid state type: {type(state)}")
                    continue

                # Handle potential emergent dimensions
                has_emergent_dim = exp.get('has_emergent_dim', False)

                # Append state with metadata about its shape
                states.append((state, has_emergent_dim))

                # Extract targets with safe defaults
                targets['distinction'].append(float(exp.get('actual', 0.0)))
                # Extract phase from quantum_metrics or use default
                if 'quantum_metrics' in exp and 'phase' in exp['quantum_metrics']:
                    targets['phase'].append(float(exp['quantum_metrics']['phase']))
                else:
                    targets['phase'].append(0.0)
                targets['value'].append(float(exp.get('reward', 0.0)))

            if not states:
                return self._create_empty_batch()

            # Process states and handle inhomogeneous dimensions
            processed_states = []
            for state_data, has_emergent_dim in states:
                # Convert to numpy if it's a tensor
                if isinstance(state_data, torch.Tensor):
                    state_data = state_data.cpu().numpy()

                # Make state data have consistent shape by reshaping emergent dimensions
                if has_emergent_dim or len(np.array(state_data).shape) > 3:
                    # Handle 4D or higher states by flattening extra dimensions
                    state_array = np.array(state_data)
                    # Get original shape
                    original_shape = state_array.shape

                    if len(original_shape) > 3:
                        # Reshape to [batch, seq_len, features]
                        # Flatten all dimensions between batch and features
                        flat_seq_len = np.prod(original_shape[1:-1])
                        reshaped_state = state_array.reshape(original_shape[0], flat_seq_len, original_shape[-1])
                        processed_states.append(reshaped_state)
                    else:
                        processed_states.append(state_array)
                else:
                    # Regular 3D state - no reshaping needed
                    processed_states.append(np.array(state_data))

            # Now that all shapes are standardized, find the maximum sequence length
            max_seq_len = max(state.shape[1] if len(state.shape) > 1 else 1 for state in processed_states)
            feature_dim = processed_states[0].shape[-1] if processed_states and len(processed_states[0].shape) > 0 else 20

            # Pad all states to the same sequence length
            padded_states = []
            for state in processed_states:
                if len(state.shape) < 2:
                    # Handle 1D states
                    padded = np.zeros((1, max_seq_len, feature_dim))
                    padded[0, 0, :min(state.shape[0], feature_dim)] = state[:min(state.shape[0], feature_dim)]
                    padded_states.append(padded)
                elif len(state.shape) < 3:
                    # Handle 2D states
                    padded = np.zeros((1, max_seq_len, feature_dim))
                    padded[0, :min(state.shape[0], max_seq_len), :min(state.shape[1], feature_dim)] = \
                        state[:min(state.shape[0], max_seq_len), :min(state.shape[1], feature_dim)]
                    padded_states.append(padded)
                else:
                    # Handle 3D states
                    padded = np.zeros((state.shape[0], max_seq_len, feature_dim))
                    padded[:, :min(state.shape[1], max_seq_len), :min(state.shape[2], feature_dim)] = \
                        state[:, :min(state.shape[1], max_seq_len), :min(state.shape[2], feature_dim)]
                    padded_states.append(padded)

            # Stack all states together
            states_array = np.vstack(padded_states)

            # Convert to tensors
            states_tensor = torch.tensor(states_array, dtype=torch.float32, device=DEVICE)

            # Ensure feature dimension is exactly 20
            if states_tensor.size(-1) != 20:
                if states_tensor.size(-1) < 20:
                    # Pad features
                    padding = (0, 20 - states_tensor.size(-1), 0, 0, 0, 0)
                    states_tensor = F.pad(states_tensor, padding)
                else:
                    # Truncate features
                    states_tensor = states_tensor[..., :20]

            # Convert targets to tensors
            target_tensors = {
                key: torch.tensor(value, dtype=torch.float32, device=DEVICE).view(-1, 1)
                for key, value in targets.items()
            }

            return states_tensor, target_tensors

        except Exception as e:
            print(f"Error preparing batch: {e}")
            traceback.print_exc()
            return self._create_empty_batch()

    def _create_empty_batch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Create an empty batch as a fallback."""
        empty_state = torch.zeros((1, 1, 20), dtype=torch.float32, device=DEVICE)
        empty_targets = {
            'distinction': torch.zeros((1, 1), dtype=torch.float32, device=DEVICE),
            'phase': torch.zeros((1, 1), dtype=torch.float32, device=DEVICE),
            'value': torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)
        }
        return empty_state, empty_targets

    def _validate_batch(self, experiences: List[Dict]) -> bool:
        """
        Validate a batch of experiences to ensure they have required fields.

        Args:
            experiences: List of experience dictionaries

        Returns:
            True if valid, False otherwise
        """
        try:
            if not experiences:
                return False

            required_keys = ['state', 'prediction', 'actual', 'quantum_metrics']

            for exp in experiences:
                # Check all required keys exist
                if not all(key in exp for key in required_keys):
                    print(f"Missing required keys in experience: {exp.keys()}")
                    return False

                # Check state is correct type
                if not isinstance(exp['state'], (np.ndarray, torch.Tensor)):
                    print(f"Invalid state type: {type(exp['state'])}")
                    return False

            return True

        except Exception as e:
            print(f"Error validating batch: {e}")
            traceback.print_exc()
            return False

    def _handle_training_error(self, error: Exception):
        """Handle training errors with potential recovery."""
        print(f"Training error encountered: {error}")
        traceback.print_exc()

        # Track error count
        self.consecutive_failures = getattr(self, 'consecutive_failures', 0) + 1

        # Enter recovery if too many errors
        if self.consecutive_failures > self.max_failures and not self.recovery_mode:
            print("Multiple training errors detected. Entering recovery mode.")
            self.enter_recovery_mode()

    def _update_training_stats(self, loss_components: Dict[str, float]):
        """
        Update training statistics with new loss values.

        Args:
            loss_components: Dictionary of loss components
        """
        try:
            # Initialize stats if not present
            if not hasattr(self, 'training_stats'):
                self.training_stats = {
                    'loss_history': [],
                    'gradient_stats': [],
                    'stability_metrics': []
                }

            # Add loss to history
            self.training_stats['loss_history'].append(loss_components)

            # Update stability factor
            current_stability = 1.0 / (1.0 + loss_components.get('total_loss', 0))
            self.stability_factor = 0.95 * self.stability_factor + 0.05 * current_stability

            # Track stability
            self.training_stats['stability_metrics'].append({
                'timestamp': time.time(),
                'stability_factor': self.stability_factor,
                'loss': loss_components.get('total_loss', 0.0)
            })

        except Exception as e:
            print(f"Error updating training stats: {e}")
            traceback.print_exc()

    def _compute_gradient_norm(self) -> float:
        """
        Compute the total gradient norm across all parameters.

        Returns:
            Total gradient norm as a float
        """
        try:
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            return float(np.sqrt(total_norm))
        except Exception as e:
            print(f"Error computing gradient norm: {e}")
            traceback.print_exc()
            return 0.0

    def _check_loss_stability(self, current_loss: float) -> bool:
        """
        Check if loss value is stable compared to recent history.

        Args:
            current_loss: The current loss value

        Returns:
            True if stable, False if unstable
        """
        try:
            # Check for invalid loss
            if not isinstance(current_loss, (int, float)) or math.isnan(current_loss) or math.isinf(current_loss):
                return False

            # Need enough history for comparison
            if len(self.training_stats['loss_history']) < 10:
                return True

            # Get recent losses
            recent_losses = [h.get('total_loss', 0.0) for h in self.training_stats['loss_history'][-10:]]
            loss_mean = np.mean(recent_losses)
            loss_std = np.std(recent_losses)

            # Check multiple stability conditions
            return (
                current_loss <= loss_mean + 3 * loss_std and  # Not too high
                current_loss >= loss_mean - 3 * loss_std and  # Not too low
                current_loss < 10 * loss_mean  # Not diverging
            )

        except Exception as e:
            print(f"Error checking loss stability: {e}")
            traceback.print_exc()
            return False

    def enter_recovery_mode(self):
        """Enter recovery mode to stabilize training."""
        try:
            self.recovery_mode = True
            self.recovery_steps = 50
            self.consecutive_failures = 0

            # Reset momentum
            self.training_momentum = 0.0

            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

            print("Entering training recovery mode.")

            # Notify model component if it has a recovery mode
            if hasattr(self.model, 'enter_recovery_mode'):
                self.model.enter_recovery_mode()

        except Exception as e:
            print(f"Error entering recovery mode: {e}")
            traceback.print_exc()

    def train_step(self, experiences: List[Dict], metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Perform a single training step on a batch of experiences.

        Args:
            experiences: List of experience dictionaries
            metrics: Dictionary of metrics for the loss function

        Returns:
            Dictionary of loss components
        """
        try:
            # Validate metrics first
            metrics = self.metric_validator.validate_metrics(metrics)

            # Skip training if in recovery mode
            if self.recovery_mode:
                self.recovery_steps -= 1
                if self.recovery_steps <= 0:
                    self.recovery_mode = False
                    print("Exiting training recovery mode.")
                return {'skipped_update': True, 'recovery_steps': self.recovery_steps}

            # Prepare input batch
            states, targets = self.prepare_batch(experiences)
            if states.size(0) == 0:
                return {}

            # Set model to training mode
            self.model.train()
            self.update_counter += 1

            # Forward pass with error handling
            try:
                # Get model output
                output = self.model(states, phase=targets.get('phase'))

                # Validate output
                if not hasattr(output, 'prediction'):
                    print("Warning: Model output doesn't have prediction attribute")
                    self.enter_recovery_mode()
                    return {'invalid_output': True}

                # Validate prediction
                prediction = output.prediction
                if not isinstance(prediction, torch.Tensor):
                    print("Warning: Prediction is not a tensor")
                    self.enter_recovery_mode()
                    return {'invalid_prediction': True}

                # Check for stability before proceeding
                if self._check_loss_stability(prediction.mean().item()):
                    # Compute loss - FIXED: ensure loss requires gradient
                    loss, loss_components = self.loss_function.compute_loss(output, targets, metrics)

                    # Explicitly make loss require gradients if it doesn't already
                    if not loss.requires_grad:
                        loss = loss.detach().requires_grad_(True)

                    # Reset gradients and compute new ones
                    self.optimizer.zero_grad()

                    # Manually call backward
                    loss.backward()

                    # Compute gradient norm for tracking
                    grad_norm = self._compute_gradient_norm()
                    self.training_stats['gradient_stats'].append(grad_norm)

                    # Clip gradients to prevent explosions
                    torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), GRADIENT_CLIP_VALUE)

                    # Perform optimizer step
                    for param_group in self.optimizer.param_groups:
                        for param in param_group['params']:
                            if param.grad is not None:
                                param.data.add_(param.grad, alpha=-LEARNING_RATE)

                    # Update stability tracking
                    self.stability_factor = 0.95 * self.stability_factor + 0.05 * (1.0 / (1.0 + loss.item()))
                    self.consecutive_failures = 0

                else:
                    print("Loss instability detected")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self.enter_recovery_mode()
                    return {'loss_instability': True}

            except Exception as e:
                print(f"Error in forward/backward pass: {e}")
                traceback.print_exc()
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.enter_recovery_mode()
                return {}

            # Update training stats
            self.training_stats['loss_history'].append(loss_components)
            self.training_stats['stability_metrics'].append({
                'stability_factor': self.stability_factor,
                'training_momentum': self.training_momentum,
                'grad_norm': grad_norm if 'grad_norm' in locals() else 0.0
            })

            return loss_components

        except Exception as e:
            print(f"Error in training step: {e}")
            traceback.print_exc()
            self.enter_recovery_mode()
            return {}

    def get_training_summary(self) -> Dict[str, float]:
        """
        Get a summary of training statistics.

        Returns:
            Dictionary of training summary metrics
        """
        try:
            if not self.training_stats['loss_history']:
                return {}

            # Get recent stats for summary
            recent_losses = self.training_stats['loss_history'][-100:]
            recent_grads = self.training_stats['gradient_stats'][-100:]
            recent_stability = self.training_stats['stability_metrics'][-100:]

            # Compute summary metrics
            summary = {
                'avg_total_loss': np.mean([l['total_loss'] for l in recent_losses]),
                'avg_distinction_loss': np.mean([l['distinction_loss'] for l in recent_losses]),
                'avg_gradient_norm': np.mean(recent_grads) if recent_grads else 0.0,
                'stability_factor': np.mean([m['stability_factor'] for m in recent_stability]) if recent_stability else 1.0,
                'training_momentum': self.training_momentum,
                'recovery_mode': self.recovery_mode,
                'recovery_steps': self.recovery_steps if self.recovery_mode else 0,
                'update_counter': self.update_counter
            }

            # Add loss trend if enough data
            if len(recent_losses) >= 10:
                recent_total_losses = [l['total_loss'] for l in recent_losses[-10:]]
                summary['recent_loss_trend'] = np.mean(np.diff(recent_total_losses))

            return summary

        except Exception as e:
            print(f"Error getting training summary: {e}")
            traceback.print_exc()
            return {}

# -----------------------------------------------------------------------------
# Error Recovery
# -----------------------------------------------------------------------------
class EnhancedErrorRecovery:
    """Handles error recovery and state restoration for the agent."""
    def __init__(self, agent: Any):
        """
        Initialize error recovery system.

        Args:
            agent: The agent to manage recovery for
        """
        self.agent = agent
        self.backup_manager = StateBackupManager()
        self._error_counts = defaultdict(int)
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.min_stability_threshold = 0.3
        self.recovery_delay = 0.1
        self.last_recovery_time = time.time()
        self.recovery_history = deque(maxlen=1000)

    @property
    def error_counts(self) -> Dict[str, int]:
        """Property getter for error_counts."""
        return dict(self._error_counts)

    def handle_error(self, error: Exception, component: str) -> bool:
        """
        Handle component errors with recovery mechanisms.

        Args:
            error: The exception that occurred
            component: The component name where the error occurred

        Returns:
            True if recovery succeeded, False otherwise
        """
        try:
            current_time = time.time()
            recovery_cooldown = 1.0  # seconds

            # Track error count
            self._error_counts[component] += 1
            logger.error(f"Error in {component}: {str(error)}")

            # Check if enough time has passed since last recovery
            if current_time - self.last_recovery_time < recovery_cooldown:
                logger.warning("Recovery attempt too soon, waiting...")
                return False

            # Run system validation before attempting recovery
            validation_results = self.backup_manager.state_validator.validate_system_state(self.agent)

            # Log validation results
            logger.info("System validation results before recovery:")
            for comp, status in validation_results.items():
                if comp != 'overall':
                    logger.info(f"  - {comp}: {'PASSED' if status else 'FAILED'}")

            # Determine recovery strategy based on validation results
            failed_components = [comp for comp, status in validation_results.items()
                                if not status and comp != 'overall']

            # If critical components have failed, perform full recovery
            critical_components = ['quantum_state', 'surplus_state', 'distinction']
            if any(comp in failed_components for comp in critical_components):
                logger.warning(f"Critical component failure detected: {failed_components}")
                success = self.initiate_full_recovery()
            elif len(failed_components) > 2:
                # If multiple non-critical components have failed
                logger.warning(f"Multiple component failures detected: {failed_components}")
                success = self.initiate_full_recovery()
            else:
                # Recover just the failed components
                success = True
                for failed_comp in failed_components:
                    if not self.initiate_component_recovery(failed_comp):
                        success = False

            self.last_recovery_time = current_time
            return success

        except Exception as e:
            logger.error(f"Error in error handling: {e}")
            traceback.print_exc()
            return False

    def initiate_component_recovery(self, component: str) -> bool:
        """
        Initialize recovery for a specific component.

        Args:
            component: The component name to recover

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Initiating recovery for component: {component}")

            # Component-specific recovery
            if component == 'quantum_state':
                return self.recover_quantum_state()
            elif component == 'surplus_dynamics':
                return self.recover_surplus_state()
            elif component == 'transformer':
                return self.recover_training_state()
            else:
                # Default component recovery strategy
                return False

        except Exception as e:
            print(f"Error in component recovery: {e}")
            traceback.print_exc()
            return False

    def initiate_full_recovery(self) -> bool:
        """
        Initiate full system recovery with enhanced error handling.

        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n🔄 Initiating full recovery sequence...")

            if self.recovery_attempts >= self.max_recovery_attempts:
                print("Maximum recovery attempts exceeded, performing full reinitialization...")
                return self._perform_full_reinitialization()

            self.recovery_attempts += 1
            success = True

            # Step 1: Quantum State Recovery
            print("Step 1: Recovering quantum state...")
            if not self.recover_quantum_state():
                print("❌ Quantum state recovery failed")
                success = False

            # Step 2: Surplus State Recovery
            print("Step 2: Recovering surplus state...")
            if not self.recover_surplus_state():
                print("❌ Surplus state recovery failed")
                success = False

            # Step 3: Training State Recovery
            print("Step 3: Recovering training state...")
            if not self.recover_training_state():
                print("❌ Training state recovery failed")
                success = False

            # Record recovery attempt
            self._record_recovery_attempt(success)

            if success:
                print("✅ Full recovery successful")
                self._error_counts.clear()
                self.recovery_attempts = 0
                return True

            print("❌ Recovery failed")
            return False

        except Exception as e:
            print(f"❌ Error in full recovery: {e}")
            traceback.print_exc()
            return False

    def _perform_full_reinitialization(self) -> bool:
        """
        Perform complete agent reinitialization when other recovery attempts fail.

        Returns:
            True if successful, False otherwise
        """
        try:
            print("🔄 Performing full agent reinitialization...")

            # Store important parameters
            num_qubits = self.agent.num_qubits

            # Create new instance - this requires agent to handle reinitialization
            if hasattr(self.agent, '__init__'):
                self.agent.__init__(num_qubits=num_qubits)
            else:
                print("❌ Agent doesn't have init method, can't reinitialize")
                return False

            # Validate initialization if method exists
            if hasattr(self.agent, '_validate_initialization'):
                if not self.agent._validate_initialization():
                    print("❌ Reinitialization validation failed")
                    return False

            self.recovery_attempts = 0
            print("✅ Agent reinitialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error in agent reinitialization: {e}")
            traceback.print_exc()
            return False

    def recover_quantum_state(self) -> bool:
        """
        Recover quantum state with enhanced validation.

        Returns:
            True if successful, False otherwise
        """
        try:
            # First try to restore from backup
            backup = self.backup_manager.restore_state(
                self.agent.quantum_state,
                restore_point='most_stable'
            )

            if backup and isinstance(backup, dict):
                try:
                    # Apply backup state
                    self.agent.quantum_state.statevector = backup['statevector']
                    self.agent.quantum_state.phase_coherence = backup.get('phase_coherence', MINIMUM_COHERENCE_FLOOR)
                    self.agent.distinction_level = backup.get('distinction_level', 0.5)

                    # Validate restored state
                    metrics = self.agent.quantum_state.get_quantum_metrics()
                    if metrics.get('phase_coherence', 0) >= MINIMUM_COHERENCE_FLOOR:
                        print("✅ Successfully restored quantum state from backup")
                        return True
                except Exception as e:
                    print(f"Error applying backup state: {e}")
                    traceback.print_exc()

            # If backup restoration fails, reinitialize ground state
            print("No valid backup found. Reinitializing ground state...")
            if hasattr(self.agent.quantum_state, '_prepare_ground_state_with_coherence'):
                self.agent.quantum_state._prepare_ground_state_with_coherence()
            else:
                print("❌ Cannot reinitialize ground state - missing method")
                return False

            # Validate reinitialized state
            return self._validate_quantum_state()

        except Exception as e:
            print(f"❌ Error recovering quantum state: {e}")
            traceback.print_exc()
            return False

    def recover_surplus_state(self) -> bool:
        """
        Recover surplus state with validation.

        Returns:
            True if successful, False otherwise
        """
        try:
            print("Reinitializing surplus state...")

            # Create new surplus state
            from data_classes import SurplusState
            new_surplus_state = SurplusState()

            # Validate new state
            if not new_surplus_state.validate():
                print("❌ New surplus state validation failed")
                return False

            # Apply to agent
            self.agent.surplus_dynamics.surplus_state = new_surplus_state

            # Use reset method if available
            if hasattr(self.agent.surplus_dynamics, 'reset_state'):
                self.agent.surplus_dynamics.reset_state()

            # Validate result
            return self.agent.surplus_dynamics.surplus_state.validate()

        except Exception as e:
            print(f"❌ Error recovering surplus state: {e}")
            traceback.print_exc()
            return False

    def _validate_quantum_state(self) -> bool:
        """
        Validate quantum state after recovery.

        Returns:
            True if valid, False otherwise
        """
        try:
            if not hasattr(self.agent.quantum_state, 'phase_coherence'):
                return False

            metrics = self.agent.quantum_state.get_quantum_metrics()
            return metrics.get('phase_coherence', 0) >= MINIMUM_COHERENCE_FLOOR

        except Exception as e:
            print(f"❌ Error validating quantum state: {e}")
            traceback.print_exc()
            return False

    def _record_recovery_attempt(self, success: bool) -> None:
        """
        Record recovery attempt details for analysis.

        Args:
            success: Whether the recovery was successful
        """
        try:
            self.recovery_history.append({
                'timestamp': time.time(),
                'success': success,
                'attempt': self.recovery_attempts,
                'components_recovered': {
                    'quantum_state': self._validate_quantum_state(),
                    'surplus_state': (
                        hasattr(self.agent.surplus_dynamics, 'surplus_state') and
                        self.agent.surplus_dynamics.surplus_state.validate()
                    ),
                    'training': hasattr(self.agent, 'training_pipeline')
                }
            })
        except Exception as e:
            print(f"Error recording recovery attempt: {e}")
            traceback.print_exc()

    def recover_training_state(self) -> bool:
        """
        Recover training state and optimizer.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Reinitialize optimizer
            if hasattr(self.agent, 'transformer'):
                self.agent.training_pipeline.optimizer = QuantumAwareOptimizer(
                    self.agent.transformer
                )
                return True
            return False

        except Exception as e:
            print(f"Error recovering training state: {e}")
            traceback.print_exc()
            return False

    def validate_recovery(self) -> bool:
        """
        Validate system state after recovery.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Get current metrics
            metrics = self.agent.quantum_state.get_quantum_metrics()

            # Define stability checks
            stability_checks = [
                metrics.get('phase_coherence', 0.0) >= self.min_stability_threshold,
                isinstance(self.agent.surplus_dynamics.surplus_state, SurplusState),
                self.agent.surplus_dynamics.surplus_state.stability >= self.min_stability_threshold,
                isinstance(self.agent.distinction_level, (int, float)) and 0 <= self.agent.distinction_level <= 1
            ]

            return all(stability_checks)

        except Exception as e:
            print(f"Error validating recovery: {e}")
            traceback.print_exc()
            return False

    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about recovery operations.

        Returns:
            Dictionary of recovery statistics
        """
        try:
            stats = {
                'total_recoveries': len(self.recovery_history),
                'successful_recoveries': sum(1 for r in self.recovery_history if r['success']),
                'current_error_counts': dict(self._error_counts),
                'last_recovery_time': self.last_recovery_time,
                'recovery_attempts': self.recovery_attempts
            }

            if self.recovery_history:
                recent = list(self.recovery_history)[-10:]
                stats.update({
                    'recent_success_rate': sum(1 for r in recent if r['success']) / len(recent),
                    'components_recovered': recent[-1]['components_recovered']
                })

            return stats

        except Exception as e:
            print(f"Error getting recovery stats: {e}")
            traceback.print_exc()
            return {}

class StateBackupManager:
    """Manages state backups and restoration for the quantum system."""
    def __init__(self, backup_frequency: int = 10):
        self.backup_frequency = backup_frequency
        self.state_backups = deque(maxlen=100)
        self.step_counter = 0
        self.backup_attempts = 0
        self.max_backup_attempts = 3
        self.min_backups = 1

        # Create initial backup state
        self.last_successful_state = self._create_initial_backup()
        self.backup_history = deque(maxlen=1000)

    def has_valid_backup(self) -> bool:
        """Check if there is at least one valid backup available."""
        return (len(self.state_backups) >= self.min_backups and
                self.last_successful_state is not None)

    def _create_initial_backup(self) -> Dict[str, Any]:
        """Create initial backup state with retries."""
        for attempt in range(self.max_backup_attempts):
            try:
                print(f"Creating initial backup (attempt {attempt + 1})")

                # Import SurplusState within the method to avoid circular imports
                from data_classes import SurplusState

                # Create new surplus state
                surplus_state = SurplusState()
                if not surplus_state.validate():
                    raise ValueError("Initial surplus state validation failed")

                initial_state = {
                    'statevector': Statevector.from_label('0' * 4),
                    'phase_coherence': MINIMUM_COHERENCE_FLOOR,
                    'distinction_level': 0.5,
                    'surplus_state': surplus_state,
                    'stability': 1.0,
                    'timestamp': time.time()
                }

                if self._validate_backup(initial_state):
                    print("✅ Initial backup created successfully")
                    self.state_backups.append(initial_state)
                    return initial_state

            except Exception as e:
                print(f"Error in backup attempt {attempt + 1}: {e}")
                traceback.print_exc()

        print("⚠️ Failed to create valid initial backup, using safe defaults")
        return self._create_safe_default_backup()

    def _create_safe_default_backup(self) -> Dict[str, Any]:
        """Create a safe default backup state."""
        from data_classes import SurplusState
        return {
            'statevector': Statevector.from_label('0' * 4),
            'phase_coherence': MINIMUM_COHERENCE_FLOOR,
            'distinction_level': 0.5,
            'surplus_state': SurplusState(),
            'stability': 1.0,
            'timestamp': time.time()
        }

    def _validate_backup(self, backup: Dict[str, Any]) -> bool:
        """Validate backup state."""
        try:
            from data_classes import SurplusState

            required_keys = {'statevector', 'phase_coherence', 'distinction_level',
                           'surplus_state', 'stability', 'timestamp'}

            # Check required keys
            if not all(key in backup for key in required_keys):
                return False

            # Validate types and values
            if not isinstance(backup['statevector'], (Statevector, np.ndarray)):
                return False

            if not isinstance(backup['phase_coherence'], (int, float)):
                return False

            if not MINIMUM_COHERENCE_FLOOR <= backup['phase_coherence'] <= 1.0:
                return False

            if not isinstance(backup['distinction_level'], (int, float)):
                return False

            if not 0.0 <= backup['distinction_level'] <= 1.0:
                return False

            if not isinstance(backup['surplus_state'], SurplusState):
                return False

            if not backup['surplus_state'].validate():
                return False

            return True

        except Exception as e:
            print(f"Error validating backup: {e}")
            return False

    def store_state(self, quantum_state: 'EnhancedQuantumState',
                    distinction_level: float,
                    surplus_state: SurplusState) -> bool:
        """Store current state with validation."""
        try:
            self.step_counter += 1
            if self.step_counter % self.backup_frequency != 0:
                return True

            # Create state backup
            backup = {
                'statevector': quantum_state.statevector.copy() if isinstance(quantum_state.statevector, np.ndarray)
                             else Statevector(quantum_state.statevector.data),
                'phase_coherence': float(quantum_state.phase_coherence),
                'distinction_level': float(distinction_level),
                'surplus_state': surplus_state.copy(),
                'stability': float(surplus_state.stability),
                'timestamp': time.time()
            }

            # Validate backup before storing
            if not self._validate_backup(backup):
                print("Invalid backup state, not storing")
                return False

            self.state_backups.append(backup)
            self.last_successful_state = backup

            # Track backup history
            self.backup_history.append({
                'timestamp': time.time(),
                'success': True,
                'metrics': {
                    'phase_coherence': backup['phase_coherence'],
                    'distinction_level': backup['distinction_level'],
                    'stability': backup['stability']
                }
            })

            return True

        except Exception as e:
            print(f"Error storing state backup: {e}")
            traceback.print_exc()
            self.backup_history.append({
                'timestamp': time.time(),
                'success': False,
                'error': str(e)
            })
            return False

    def restore_state(self, quantum_state: 'EnhancedQuantumState',
                 restore_point: str = 'last') -> Optional[Dict[str, Any]]:
        """Restore state from backup with enhanced validation."""
        try:
            # Check if we can use a backup first
            if not self.has_valid_backup():
                logger.warning("No valid backup available for restoration")
                return None

            # Get appropriate backup with better selection logic
            if restore_point == 'last':
                backup = self.last_successful_state
            elif restore_point == 'most_stable':
                # Sort backups by stability and take the most stable one
                stable_backups = sorted(
                    self.state_backups,
                    key=lambda x: x.get('stability', 0.0),
                    reverse=True
                )
                backup = stable_backups[0] if stable_backups else self.last_successful_state
            elif restore_point == 'most_coherent':
                # Sort backups by coherence and take the most coherent one
                coherent_backups = sorted(
                    self.state_backups,
                    key=lambda x: x.get('phase_coherence', 0.0),
                    reverse=True
                )
                backup = coherent_backups[0] if coherent_backups else self.last_successful_state
            else:
                logger.warning(f"Invalid restore point: {restore_point}")
                backup = self.last_successful_state if self.last_successful_state else None

            # Validate backup before restoration
            if not backup or not self._validate_backup(backup):
                logger.warning("Invalid backup state for restoration")
                return None

            # Apply backup to quantum state with better error handling
            try:
                # Add timestamp for tracking
                restoration_time = time.time()

                # Clone the statevector to prevent reference issues
                if isinstance(backup['statevector'], Statevector):
                    quantum_state.statevector = backup['statevector'].copy()
                elif isinstance(backup['statevector'], np.ndarray):
                    quantum_state.statevector = Statevector(backup['statevector'])
                else:
                    logger.error(f"Unknown statevector type: {type(backup['statevector'])}")
                    return None

                # Restore other quantum properties
                quantum_state.phase_coherence = backup['phase_coherence']

                # Record restoration event
                self.restoration_history.append({
                    'timestamp': restoration_time,
                    'backup_time': backup.get('timestamp', 0),
                    'backup_age': restoration_time - backup.get('timestamp', 0),
                    'stability': backup.get('stability', 0.0)
                })

                logger.info("✅ State restored successfully from backup")
                return backup

            except Exception as e:
                logger.error(f"Error applying backup state: {e}")
                traceback.print_exc()
                return None

        except Exception as e:
            logger.error(f"Error restoring state: {e}")
            traceback.print_exc()
            return None

    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about backups."""
        try:
            if not self.backup_history:
                return {}

            recent = list(self.backup_history)[-100:]
            return {
                'total_backups': len(self.state_backups),
                'successful_backups': sum(1 for b in recent if b['success']),
                'last_backup_time': self.last_successful_state['timestamp'] if self.last_successful_state else 0,
                'backup_frequency': self.backup_frequency,
                'average_stability': np.mean([b['metrics']['stability']
                                           for b in recent if b['success'] and 'metrics' in b])
            }

        except Exception as e:
            print(f"Error getting backup stats: {e}")
            traceback.print_exc()
            return {}

class StateValidationManager:
    """Manages ongoing state validation and monitoring."""
    def __init__(self, agent: 'EnhancedSingleAgentFinalEvolution'):
        self.agent = agent
        self.validation_history = deque(maxlen=1000)
        self.last_validation_time = time.time()
        self.validation_interval = 0.1  # seconds

    def validate_current_state(self) -> bool:
        """Validate current agent state comprehensively."""
        try:
            current_time = time.time()

            # Check if validation is needed
            if current_time - self.last_validation_time < self.validation_interval:
                return True

            validation_results = {}

            # Validate quantum state
            quantum_valid = (
                hasattr(self.agent, 'quantum_state') and
                hasattr(self.agent.quantum_state, 'phase_coherence') and
                isinstance(self.agent.quantum_state.phase_coherence, (int, float)) and
                self.agent.quantum_state.phase_coherence >= MINIMUM_COHERENCE_FLOOR
            )
            validation_results['quantum_state'] = quantum_valid

            # Validate surplus state
            surplus_valid = (
                hasattr(self.agent, 'surplus_dynamics') and
                hasattr(self.agent.surplus_dynamics, 'surplus_state') and
                isinstance(self.agent.surplus_dynamics.surplus_state, SurplusState) and
                isinstance(self.agent.surplus_dynamics.surplus_state.values, dict) and
                len(self.agent.surplus_dynamics.surplus_state.values) >= 4
            )
            validation_results['surplus_state'] = surplus_valid

            # Validate distinction level
            distinction_valid = (
                hasattr(self.agent, 'distinction_level') and
                isinstance(self.agent.distinction_level, (int, float)) and
                0 <= self.agent.distinction_level <= 1
            )
            validation_results['distinction'] = distinction_valid

            # Store validation results
            self.validation_history.append({
                'timestamp': current_time,
                'results': validation_results,
                'all_valid': all(validation_results.values())
            })

            self.last_validation_time = current_time
            return all(validation_results.values())

        except Exception as e:
            print(f"Error in state validation: {e}")
            traceback.print_exc()
            return False

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        try:
            if not self.validation_history:
                return {}

            recent = list(self.validation_history)[-100:]
            return {
                'total_validations': len(recent),
                'success_rate': sum(1 for v in recent if v['all_valid']) / len(recent) if recent else 0,
                'component_success_rates': {
                    component: sum(1 for v in recent if v['results'].get(component, False)) / len(recent) if recent else 0
                    for component in ('quantum_state', 'surplus_state', 'distinction')
                },
                'last_validation_time': self.last_validation_time
            }

        except Exception as e:
            print(f"Error getting validation stats: {e}")
            traceback.print_exc()
            return {}

class StateSynchronizationManager:
    """Manages synchronization between quantum state and other components."""
    def __init__(self, quantum_state: 'EnhancedQuantumState',
                 surplus_dynamics: 'EnhancedSurplusDynamics',
                 distinction_dynamics: 'EnhancedDistinctionDynamics',
                 transformer: Optional[nn.Module] = None):
        self.quantum_state = quantum_state
        self.surplus_dynamics = surplus_dynamics
        self.distinction_dynamics = distinction_dynamics
        self.transformer = transformer  # Optional transformer model
        self.sync_history = deque(maxlen=1000)
        self.last_sync_time = time.time()
        self.sync_retries = 0
        self.max_retries = 3

    def synchronize_states(self) -> bool:
        """Synchronize states across components with enhanced error handling."""
        try:
            self.sync_retries = 0
            while self.sync_retries < self.max_retries:
                # Ensure quantum state has required attributes
                if not hasattr(self.quantum_state, 'phase_coherence'):
                    self.quantum_state.phase_coherence = MINIMUM_COHERENCE_FLOOR

                if not hasattr(self.quantum_state, 'statevector'):
                    self.quantum_state.statevector = Statevector.from_label('0' * self.quantum_state.num_qubits)

                # Get current metrics with safe defaults
                try:
                    metrics = self.quantum_state.get_quantum_metrics()
                except Exception as e:
                    print(f"Error getting metrics during sync: {e}")
                    metrics = {
                        'phase_coherence': self.quantum_state.phase_coherence,
                        'normalized_entropy': 0.0,
                        'phase': 0.0
                    }

                # Initialize surplus state if needed
                if not isinstance(self.surplus_dynamics.surplus_state, SurplusState) or self.surplus_dynamics.surplus_state is None:
                    print(f"Warning: Reinitializing surplus state as SurplusState()")
                    self.surplus_dynamics.surplus_state = SurplusState()

                # Update surplus dynamics
                try:
                    self.surplus_dynamics.update_surplus(
                        metrics['phase_coherence'],
                        metrics['normalized_entropy']
                    )
                except Exception as e:
                    print(f"Error updating surplus during sync: {e}")
                    traceback.print_exc()
                    self.surplus_dynamics.surplus_state = SurplusState()

                # Move transformer to correct device if needed
                if self.transformer is not None:
                    try:
                        device = next(self.quantum_state.parameters()).device
                        self.transformer.to(device)
                    except Exception as e:
                        print(f"Error moving transformer to device: {e}")

                # Track resource history
                if not hasattr(self, 'resource_history'):
                    self.resource_history = []

                # Store sync history
                self.sync_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics.copy(),
                    'surplus_stability': self.surplus_dynamics.surplus_state.stability
                })

                self.last_sync_time = time.time()
                return True

                self.sync_retries += 1
                print(f"Sync attempt {self.sync_retries} failed, retrying...")
                time.sleep(0.1)

            return False

        except Exception as e:
            print(f"Error in state synchronization: {e}")
            traceback.print_exc()
            return False

    def check_sync_needed(self) -> bool:
        """Check if synchronization is needed."""
        return time.time() - self.last_sync_time > 0.1  # Sync every 100ms

    def get_sync_stats(self) -> Dict[str, float]:
        """Get synchronization statistics."""
        try:
            if not self.sync_history:
                return {}

            recent_syncs = list(self.sync_history)[-100:]
            return {
                'sync_frequency': len(recent_syncs) / (time.time() - recent_syncs[0]['timestamp']) if recent_syncs else 0,
                'mean_coherence': np.mean([s['metrics'].get('phase_coherence', 0) for s in recent_syncs]) if recent_syncs else 0,
                'mean_surplus_stability': np.mean([s.get('surplus_stability', 0) for s in recent_syncs]) if recent_syncs else 0,
                'last_sync_time': self.last_sync_time
            }

        except Exception as e:
            print(f"Error getting sync stats: {e}")
            traceback.print_exc()
            return {}

class EnhancedQuantumSelfOptimization:
    """
    Enhanced quantum self-optimization with improved coherence management and adaptive optimization.
    """
    def __init__(self, num_qubits: int = NUM_QUBITS_PER_AGENT):
        self.num_qubits = num_qubits
        self.optimization_history = deque(maxlen=1000)
        self.coherence_history = deque(maxlen=100)
        self.phase_history = deque(maxlen=100)
        self.coherence_momentum = 0.5  # Initialize with moderate momentum
        self.phase_momentum = 0.5      # Initialize with moderate momentum
        self.adaptation_rates = {
            'coherence': 0.3,    # Increased from 0.1
            'entropy': 0.2,      # Increased from 0.05
            'phase': 0.25        # Increased from 0.08
        }
        self.stability_factor = 1.0
        self.optimization_momentum = 0.5  # Initialize with moderate momentum
        self.minimum_coherence = MINIMUM_COHERENCE_FLOOR
        self.target_coherence = 0.8
        self.consecutive_failures = 0
        self.max_failures = 3    # Reduced from 5 to be more responsive

        # Recovery parameters
        self.recovery_threshold = 0.3    # Increased from 0.2
        self.recovery_factor = 0.7       # Increased from 0.5 for gentler recovery
        self.in_recovery = False
        self.recovery_steps = 0
        self.max_recovery_steps = 10

    def reinforce_coherence(self, qc: QuantumCircuit, distinction_variance: float, phase_coherence: float) -> None:
        """Enhanced coherence reinforcement with type safety."""
        try:
            # Ensure inputs are real floats
            phase_coherence = float(np.real(phase_coherence))
            distinction_variance = float(np.real(distinction_variance))

            # Calculate correction parameters
            coherence_error = self.target_coherence - phase_coherence
            base_angle = float(np.real((np.pi / 8) * np.sign(coherence_error)))
            variance_factor = min(1.0, distinction_variance / 0.02)

            # Update momentum with type safety
            self.coherence_momentum = float(np.real(
                MOMENTUM_DECAY * self.coherence_momentum +
                (1 - MOMENTUM_DECAY) * coherence_error
            ))

            # Calculate final angle with momentum influence
            angle = float(np.real(base_angle * variance_factor * (1.0 + 0.1 * self.coherence_momentum)))

            # Apply quantum operations with explicit float conversion
            for q in range(self.num_qubits):
                qc.rz(float(0.1 * angle), q)
                qc.rx(float(angle), q)

            # Track optimization
            self.optimization_history.append({
                'type': 'coherence_reinforcement',
                'angle': angle,
                'coherence': phase_coherence,
                'momentum': self.coherence_momentum,
                'timestamp': time.time()
            })

            self.stability_factor = min(1.0, self.stability_factor + 0.1 * (phase_coherence - 0.5))
            self.consecutive_failures = 0

        except Exception as e:
            print(f"Error in coherence reinforcement: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self._enter_recovery_mode()

    def optimize_quantum_state(self, quantum_state: 'EnhancedQuantumState',
                         distinction_level: float,
                         cognitive_state: Dict[str, float]) -> None:
        """
        Optimize quantum state with enhanced stability and adaptation
        """
        try:
            # Validate inputs with improved defaults
            if not isinstance(cognitive_state, dict):
                cognitive_state = {}

            required_metrics = {
                'stability': 1.0,
                'quantum_coupling': 1.0,
                'mean_stability': 1.0,
                'quantum_influence': 1.0,
                'collapse_probability': 0.0,
                'coherence_distinction': 0.5,
                'quantum_surplus_coupling': 1.0
            }

            # Add missing metrics with defaults
            for key, default in required_metrics.items():
                if key not in cognitive_state:
                    cognitive_state[key] = default

            # Get current metrics with enhanced validation
            metrics = quantum_state.get_quantum_metrics()
            coherence_error = self.target_coherence - metrics['phase_coherence']
            entropy_factor = 1.0 - metrics.get('normalized_entropy', 0.5)
            cognitive_factor = cognitive_state.get('mean_stability', 1.0)

            # Define improvement thresholds
            COHERENCE_THRESHOLD = 0.1
            ENTROPY_THRESHOLD = 0.3
            COLLAPSE_THRESHOLD = 0.3
            MOMENTUM_THRESHOLD = 0.2

            # Check if optimization is needed with more granular conditions
            needs_optimization = (
                abs(coherence_error) > COHERENCE_THRESHOLD or
                entropy_factor < ENTROPY_THRESHOLD or
                cognitive_state.get('collapse_probability', 0.0) > COLLAPSE_THRESHOLD or
                self.optimization_momentum > MOMENTUM_THRESHOLD
            )

            if needs_optimization:
                # Save initial state for comparison
                initial_statevector = quantum_state.statevector.copy()
                initial_metrics = metrics.copy()

                # Calculate optimization strength with quantum influence
                base_strength = self.adaptation_rates['coherence']
                quantum_factor = cognitive_state.get('quantum_influence', 1.0)
                stability_factor = cognitive_state.get('stability', 1.0)

                opt_strength = base_strength * (
                    1.0 + quantum_factor
                ) * stability_factor

                # Update optimization momentum with decay
                self.optimization_momentum = (
                    MOMENTUM_DECAY * self.optimization_momentum +
                    (1 - MOMENTUM_DECAY) * opt_strength
                )

                # Apply optimizations based on state with enhanced momentum
                if coherence_error > 0:
                    self._apply_coherence_enhancement(
                        quantum_state,
                        opt_strength + 0.1 * self.optimization_momentum,
                        distinction_level
                    )
                else:
                    self._apply_coherence_reduction(
                        quantum_state,
                        opt_strength,
                        distinction_level
                    )

                if entropy_factor < ENTROPY_THRESHOLD:
                    self._apply_entropy_reduction(
                        quantum_state,
                        opt_strength
                    )

                # Verify improvement with multiple metrics
                final_metrics = quantum_state.get_quantum_metrics()

                # Calculate improvement scores
                coherence_improvement = final_metrics['phase_coherence'] - initial_metrics['phase_coherence']
                entropy_improvement = initial_metrics['normalized_entropy'] - final_metrics['normalized_entropy']
                stability_improvement = final_metrics.get('stability', 1.0) - initial_metrics.get('stability', 1.0)

                # Consider multiple factors for improvement
                improvement = (
                    coherence_improvement > -0.01 or  # Allow small degradation
                    entropy_improvement > -0.01 or
                    stability_improvement > 0
                )

                if not improvement:
                    print("Warning: Optimization did not improve state")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self._enter_recovery_mode()
                else:
                    self.consecutive_failures = max(0, self.consecutive_failures - 1)

                # Track optimization with enhanced metrics
                self.optimization_history.append({
                    'initial_metrics': initial_metrics,
                    'final_metrics': final_metrics,
                    'improvement': improvement,
                    'optimization_strength': opt_strength,
                    'momentum': self.optimization_momentum,
                    'coherence_change': coherence_improvement,
                    'entropy_change': entropy_improvement,
                    'stability_change': stability_improvement,
                    'consecutive_failures': self.consecutive_failures
                })

        except Exception as e:
            print(f"Error in quantum optimization: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self._enter_recovery_mode()

    def _enter_recovery_mode(self):
        """Enter recovery mode when optimization repeatedly fails"""
        try:
            print("Entering quantum optimization recovery mode")

            # Scale down adaptation rates but maintain some adaptation ability
            self.adaptation_rates = {
                k: v * self.recovery_factor for k, v in self.adaptation_rates.items()
            }

            # Reduce momentum values but don't zero them out
            self.optimization_momentum *= self.recovery_factor
            self.coherence_momentum *= self.recovery_factor
            self.phase_momentum *= self.recovery_factor

            # Maintain some stability while in recovery
            self.stability_factor = max(self.stability_factor * self.recovery_factor, 0.3)

            # Reset failure counter
            self.consecutive_failures = 0

            # Enter recovery mode
            self.in_recovery = True
            self.recovery_steps = self.max_recovery_steps

            print(f"Recovery mode entered with:")
            print(f"- Optimization momentum: {self.optimization_momentum:.3f}")
            print(f"- Coherence momentum: {self.coherence_momentum:.3f}")
            print(f"- Phase momentum: {self.phase_momentum:.3f}")
            print(f"- Stability factor: {self.stability_factor:.3f}")

        except Exception as e:
            print(f"Error entering recovery mode: {e}")
            traceback.print_exc()

    def _apply_coherence_enhancement(self, quantum_state: 'EnhancedQuantumState',
                                   strength: float,
                                   distinction_level: float) -> None:
        """Apply coherence enhancement with proper error handling"""
        try:
            # Calculate rotation angle with momentum
            angle = np.pi * strength * distinction_level

            # Apply rotations with stability
            for qubit in range(quantum_state.num_qubits):
                quantum_state.qc.rz(0.1 * angle, qubit)  # Phase-preserving operation
                quantum_state.qc.rx(angle * (1.0 + 0.2 * self.phase_momentum), qubit)

            # Apply global phase shift
            quantum_state.qc.rz(angle * 0.5, 0)

            # Update state
            quantum_state.update_state()

        except Exception as e:
            print(f"Error in coherence enhancement: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1

    def _apply_coherence_reduction(self, quantum_state: 'EnhancedQuantumState',
                                 strength: float,
                                 distinction_level: float) -> None:
        """Apply coherence reduction with minimum coherence preservation"""
        try:
            # Calculate damping strength
            gamma = min(strength * (1.0 - distinction_level),
                       1.0 - self.minimum_coherence)

            # Apply controlled amplitude damping
            for qubit in range(quantum_state.num_qubits):
                quantum_state.qc.id(qubit)  # Identity gate with noise
                quantum_state.qc.rz(0.1 * np.pi * gamma, qubit)  # Stabilizing operation

            # Update state
            quantum_state.update_state()

        except Exception as e:
            print(f"Error in coherence reduction: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1

    def _apply_entropy_reduction(self, quantum_state: 'EnhancedQuantumState',
                               strength: float) -> None:
        """Apply entropy reduction with momentum-based stability"""
        try:
            # Apply bit flips with stability
            for qubit in range(quantum_state.num_qubits):
                quantum_state.qc.rz(0.1 * np.pi * strength, qubit)  # Stabilizing rotation
                quantum_state.qc.x(qubit)  # Main operation

            # Apply phase shift with momentum
            phase_angle = np.pi * strength * (1.0 + 0.1 * self.phase_momentum)
            quantum_state.qc.rz(phase_angle, 0)

            # Update state
            quantum_state.update_state()

        except Exception as e:
            print(f"Error in entropy reduction: {e}")
            traceback.print_exc()
            self.consecutive_failures += 1

    def get_optimization_summary(self) -> Dict[str, float]:
        """Get summary of optimization parameters and history."""
        try:
            if not self.optimization_history:
                return {}

            recent = list(self.optimization_history)[-100:]
            summary = {
                'mean_opt_strength': np.mean([r.get('optimization_strength', 0.0) for r in recent if 'optimization_strength' in r]),
                'optimization_frequency': len(recent) / 100.0,
                'optimization_momentum': self.optimization_momentum,
                'stability_factor': self.stability_factor,
                'in_recovery': self.in_recovery,
                'recovery_steps': self.recovery_steps if self.in_recovery else 0,
                'consecutive_failures': self.consecutive_failures,
                'coherence_momentum': self.coherence_momentum,
                'phase_momentum': self.phase_momentum
            }

            for key, rate in self.adaptation_rates.items():
                summary[f'{key}_adaptation_rate'] = rate

            return summary

        except Exception as e:
            print(f"Error getting optimization summary: {e}")
            traceback.print_exc()
            return {}

class PerformanceMonitor:
    """Monitors system performance and provides optimization recommendations."""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = {
            'coherence': deque(maxlen=window_size),
            'distinction': deque(maxlen=window_size),
            'entropy': deque(maxlen=window_size),
            'surplus_stability': deque(maxlen=window_size),
            'training_loss': deque(maxlen=window_size),
            'prediction_accuracy': deque(maxlen=window_size)
        }
        self.alert_thresholds = {
            'coherence_min': 0.3,
            'distinction_variance_max': 0.1,
            'entropy_max': 0.8,
            'stability_min': 0.4
        }
        self.optimization_history = deque(maxlen=1000)
        self.last_metrics_update = time.time()
        self.update_counter = 0
        self.alert_counter = defaultdict(int)
        self.current_recommendations = []

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update stored metrics and track performance trends.

        Args:
            metrics: Dictionary of current metrics
        """
        try:
            current_time = time.time()
            self.update_counter += 1

            # Record metrics in history
            for key, value in metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)

            # Store complete snapshot
            self.optimization_history.append({
                'timestamp': current_time,
                'metrics': metrics.copy(),
                'update_counter': self.update_counter
            })

            # Check for alerts
            self._check_alerts(metrics)

            # Update recommendations every 10 updates
            if self.update_counter % 10 == 0:
                self.current_recommendations = self._generate_recommendations()

            self.last_metrics_update = current_time

        except Exception as e:
            print(f"Error updating performance metrics: {e}")
            traceback.print_exc()

    def _check_alerts(self, metrics: Dict[str, float]) -> None:
        """Check for alert conditions based on current metrics."""
        try:
            alerts = []

            # Check coherence
            if metrics.get('coherence', 1.0) < self.alert_thresholds['coherence_min']:
                alerts.append('low_coherence')
                self.alert_counter['low_coherence'] += 1
            else:
                self.alert_counter['low_coherence'] = 0

            # Check entropy
            if metrics.get('entropy', 0.0) > self.alert_thresholds['entropy_max']:
                alerts.append('high_entropy')
                self.alert_counter['high_entropy'] += 1
            else:
                self.alert_counter['high_entropy'] = 0

            # Check stability
            if metrics.get('stability', 1.0) < self.alert_thresholds['stability_min']:
                alerts.append('low_stability')
                self.alert_counter['low_stability'] += 1
            else:
                self.alert_counter['low_stability'] = 0

            # Check distinction variance if we have enough history
            distinction_values = list(self.metrics_history.get('distinction', []))
            if len(distinction_values) > 10:
                variance = np.var(distinction_values[-10:])
                if variance > self.alert_thresholds['distinction_variance_max']:
                    alerts.append('high_distinction_variance')
                    self.alert_counter['high_distinction_variance'] += 1
                else:
                    self.alert_counter['high_distinction_variance'] = 0

            # Store alerts with metrics
            if alerts and len(self.optimization_history) > 0:
                self.optimization_history[-1]['alerts'] = alerts

        except Exception as e:
            print(f"Error checking alerts: {e}")
            traceback.print_exc()

    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate correlation with proper error handling for division by zero issues.

        Args:
            x: First data array
            y: Second data array

        Returns:
            Correlation coefficient or 0.0 if calculation fails
        """
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0

            # First, handle NaN values
            x = np.nan_to_num(x, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            # Calculate standard deviations
            std_x = np.std(x)
            std_y = np.std(y)

            # Check for constant arrays which would cause division by zero
            if std_x < 1e-10 or std_y < 1e-10:
                # Add small random noise to prevent constant arrays
                if std_x < 1e-10:
                    x = x + np.random.normal(0, 1e-5, size=x.shape)
                    std_x = np.std(x)

                if std_y < 1e-10:
                    y = y + np.random.normal(0, 1e-5, size=y.shape)
                    std_y = np.std(y)

                # If still constant after adding noise, return 0
                if std_x < 1e-10 or std_y < 1e-10:
                    return 0.0

            # Manually calculate correlation to avoid NumPy warning
            x_normalized = (x - np.mean(x)) / std_x
            y_normalized = (y - np.mean(y)) / std_y
            correlation = np.mean(x_normalized * y_normalized)

            # Check for NaN results
            if np.isnan(correlation):
                return 0.0

            return float(correlation)
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0.0

    def get_performance_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance metrics and provide detailed insights.

        Returns:
            Dictionary with analysis results
        """
        try:
            analysis = {}

            # Calculate statistics for each metric
            for key, values in self.metrics_history.items():
                if values:
                    analysis[f'{key}_mean'] = float(np.mean(values))
                    analysis[f'{key}_std'] = float(np.std(values))
                    analysis[f'{key}_trend'] = float(np.mean(np.diff(list(values)))) if len(values) > 1 else 0.0
                    analysis[f'{key}_min'] = float(np.min(values))
                    analysis[f'{key}_max'] = float(np.max(values))

            # Determine active alerts
            alerts = []
            for alert_type, count in self.alert_counter.items():
                if count > 2:  # Alert only after multiple occurrences
                    alerts.append(alert_type)

            analysis['alerts'] = alerts
            analysis['optimization_needed'] = len(alerts) > 0
            analysis['update_counter'] = self.update_counter
            analysis['last_update_time'] = self.last_metrics_update

            # Add relationship analysis if we have enough data
            if all(len(self.metrics_history[k]) > 10 for k in ['coherence', 'distinction', 'entropy']):
                coherence = list(self.metrics_history['coherence'])[-10:]
                distinction = list(self.metrics_history['distinction'])[-10:]
                entropy = list(self.metrics_history['entropy'])[-10:]

                analysis['coherence_distinction_corr'] = self._safe_correlation(np.array(coherence), np.array(distinction))
                analysis['coherence_entropy_corr'] = self._safe_correlation(np.array(coherence), np.array(entropy))
                analysis['distinction_entropy_corr'] = self._safe_correlation(np.array(distinction), np.array(entropy))

            return analysis

        except Exception as e:
            print(f"Error in performance analysis: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on observed metrics.

        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            analysis = self.get_performance_analysis()

            # Handle low coherence
            if 'low_coherence' in analysis.get('alerts', []):
                recommendations.append({
                    'component': 'quantum_state',
                    'action': 'reinforce_coherence',
                    'priority': 'high',
                    'params': {'target_coherence': 0.7}
                })

            # Handle high distinction variance
            if 'high_distinction_variance' in analysis.get('alerts', []):
                recommendations.append({
                    'component': 'distinction_dynamics',
                    'action': 'stabilize_distinction',
                    'priority': 'medium',
                    'params': {'momentum_scale': 0.8}
                })

            # Handle high entropy
            if 'high_entropy' in analysis.get('alerts', []):
                recommendations.append({
                    'component': 'quantum_state',
                    'action': 'reduce_entropy',
                    'priority': 'high',
                    'params': {'target_entropy': 0.5}
                })

            # Handle low stability
            if 'low_stability' in analysis.get('alerts', []):
                recommendations.append({
                    'component': 'surplus_dynamics',
                    'action': 'reinforce_stability',
                    'priority': 'high',
                    'params': {'stability_boost': 0.3}
                })

            # Add general learning rate recommendation based on trend
            if self.update_counter > 20 and 'training_loss' in self.metrics_history:
                loss_trend = analysis.get('training_loss_trend', 0)
                if loss_trend > 0:  # Loss is increasing
                    recommendations.append({
                        'component': 'training_pipeline',
                        'action': 'reduce_learning_rate',
                        'priority': 'medium',
                        'params': {'scale_factor': 0.7}
                    })
                elif loss_trend < -0.01:  # Loss is decreasing nicely
                    recommendations.append({
                        'component': 'training_pipeline',
                        'action': 'maintain_learning_rate',
                        'priority': 'low',
                        'params': {}
                    })

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            traceback.print_exc()
            return []

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get the current optimization recommendations.

        Returns:
            List of recommendation dictionaries
        """
        return self.current_recommendations

class OptimizationCoordinator:
    """Coordinates optimization actions across system components."""
    def __init__(self, agent):
        # Here 'agent' is assumed to be your overall system agent (e.g., EnhancedSingleAgentFinalEvolution)
        self.agent = agent
        self.monitor = PerformanceMonitor()
        self.optimization_queue = deque()
        self.last_optimization_time = time.time()
        self.optimization_cooldown = 10  # seconds between optimizations
        self.current_step = 0
        self.optimization_history = deque(maxlen=1000)
        self.applied_optimizations = defaultdict(int)
        self.recovery_mode = False
        self.recovery_count = 0
        self.max_recovery_attempts = 3

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update performance monitoring and process optimization queue.

        Args:
            metrics: Current system metrics
        """
        try:
            self.current_step += 1

            # Update performance monitor
            self.monitor.update_metrics(metrics)

            # Check if optimization is needed
            if self._should_optimize():
                recommendations = self.monitor.get_optimization_recommendations()
                self._queue_optimizations(recommendations)

            # Process optimization queue
            self._process_optimization_queue()

        except Exception as e:
            print(f"Error in optimization update: {e}")
            traceback.print_exc()

    def _should_optimize(self) -> bool:
        """
        Determine if optimization should be performed.

        Returns:
            True if optimization should be performed, False otherwise
        """
        # Check time elapsed since last optimization
        time_elapsed = time.time() - self.last_optimization_time

        # Get performance analysis
        analysis = self.monitor.get_performance_analysis()

        # Check if optimization is needed and cooldown has passed
        return (time_elapsed > self.optimization_cooldown and
                analysis.get('optimization_needed', False))

    def _queue_optimizations(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Queue optimization recommendations based on priority.

        Args:
            recommendations: List of optimization recommendations
        """
        try:
            # Define priority mapping (lower number = higher priority)
            priority_map = {'high': 0, 'medium': 1, 'low': 2}

            # Sort recommendations by priority
            sorted_recs = sorted(
                recommendations,
                key=lambda x: priority_map.get(x.get('priority', 'low'), 999)
            )

            # Add recommendations to queue
            for rec in sorted_recs:
                # Check if this optimization has been applied too many times
                component = rec.get('component', '')
                action = rec.get('action', '')
                key = f"{component}_{action}"

                # Limit the number of times a specific optimization can be applied
                if self.applied_optimizations[key] < 3:
                    self.optimization_queue.append(rec)

        except Exception as e:
            print(f"Error queueing optimizations: {e}")
            traceback.print_exc()

    def _process_optimization_queue(self) -> None:
        """Process the optimization queue."""
        try:
            if not self.optimization_queue:
                return

            # Process up to 3 optimizations at once
            optimizations_applied = 0
            while self.optimization_queue and optimizations_applied < 3:
                # Get next optimization
                opt = self.optimization_queue.popleft()

                # Apply optimization
                success = self._apply_optimization(opt)

                if success:
                    optimizations_applied += 1
                    component = opt.get('component', '')
                    action = opt.get('action', '')
                    key = f"{component}_{action}"
                    self.applied_optimizations[key] += 1

                    # Record application
                    self.optimization_history.append({
                        'timestamp': time.time(),
                        'optimization': opt,
                        'success': True,
                        'step': self.current_step
                    })

            self.last_optimization_time = time.time()

        except Exception as e:
            print(f"Error processing optimization queue: {e}")
            traceback.print_exc()

    def _apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """
        Apply a specific optimization.

        Args:
            optimization: Optimization recommendation dictionary

        Returns:
            True if optimization was applied successfully, False otherwise
        """
        try:
            component = optimization.get('component', '')
            action = optimization.get('action', '')
            params = optimization.get('params', {})

            print(f"Applying optimization: {component}.{action} with params {params}")

            # Apply component-specific optimizations
            if component == 'quantum_state':
                if action == 'reinforce_coherence' and hasattr(self.agent, 'quantum_optimizer'):
                    self.agent.quantum_optimizer.reinforce_coherence(
                        self.agent.quantum_state.qc,
                        params.get('distinction_variance', 0.5),
                        self.agent.quantum_state.phase_coherence
                    )
                    return True

                elif action == 'reduce_entropy':
                    if hasattr(self.agent.quantum_state, 'apply_gate') and hasattr(self.agent.quantum_state, 'apply_phase_shift'):
                        # Apply operations to reduce entropy
                        self.agent.quantum_state.apply_gate('x', [0])
                        self.agent.quantum_state.apply_phase_shift(0.1 * np.pi)
                        return True

            elif component == 'distinction_dynamics':
                if action == 'stabilize_distinction' and hasattr(self.agent, 'distinction_dynamics'):
                    # Stabilize distinction by reducing momentum
                    if hasattr(self.agent.distinction_dynamics, 'adjustment_momentum'):
                        momentum_scale = params.get('momentum_scale', 0.8)
                        self.agent.distinction_dynamics.adjustment_momentum *= momentum_scale
                        return True

            elif component == 'surplus_dynamics':
                if action == 'reinforce_stability' and hasattr(self.agent, 'surplus_dynamics'):
                    # Boost stability
                    stability_boost = params.get('stability_boost', 0.3)
                    if hasattr(self.agent.surplus_dynamics.surplus_state, 'stability'):
                        current = self.agent.surplus_dynamics.surplus_state.stability
                        self.agent.surplus_dynamics.surplus_state.stability = min(1.0, current + stability_boost)
                        return True

            elif component == 'training_pipeline':
                if action == 'reduce_learning_rate' and hasattr(self.agent, 'training_pipeline'):
                    # Reduce learning rate
                    scale_factor = params.get('scale_factor', 0.7)
                    if hasattr(self.agent.training_pipeline, 'optimizer'):
                        for param_group in self.agent.training_pipeline.optimizer.param_groups:
                            param_group['lr'] *= scale_factor
                        return True

                elif action == 'maintain_learning_rate':
                    # No action needed, just acknowledge
                    return True

            print(f"Optimization {component}.{action} not implemented or components not found")
            return False

        except Exception as e:
            print(f"Error applying optimization {component}.{action}: {e}")
            traceback.print_exc()
            return False

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.

        Returns:
            Dictionary with optimization statistics
        """
        try:
            stats = {
                'total_optimizations': len(self.optimization_history),
                'current_step': self.current_step,
                'queue_length': len(self.optimization_queue),
                'recovery_mode': self.recovery_mode,
                'recovery_count': self.recovery_count,
                'optimization_types': dict(self.applied_optimizations)
            }

            if self.optimization_history:
                recent = list(self.optimization_history)[-10:]
                component_counts = defaultdict(int)
                action_counts = defaultdict(int)

                for opt in recent:
                    optimization = opt.get('optimization', {})
                    component = optimization.get('component', 'unknown')
                    action = optimization.get('action', 'unknown')
                    component_counts[component] += 1
                    action_counts[action] += 1

                stats['recent_components'] = dict(component_counts)
                stats['recent_actions'] = dict(action_counts)
                stats['success_rate'] = sum(1 for opt in recent if opt.get('success', False)) / len(recent)

            return stats

        except Exception as e:
            print(f"Error getting optimization stats: {e}")
            traceback.print_exc()
            return {}

class EnhancedLearningRateScheduler:
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    Also applies a stability-based modulation.
    """
    def __init__(self, optimizer: torch.optim.Optimizer,
                 min_lr: float = LEARNING_RATE_MIN,
                 max_lr: float = LEARNING_RATE_MAX,
                 warmup_steps: int = 1000,
                 cycle_steps: int = 10000):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.step_count = 0
        self.last_update_time = time.time()
        self.update_history = deque(maxlen=1000)
        self.current_lr = min_lr

    def step(self, loss: float, metrics: Dict[str, float]) -> None:
        """
        Update learning rate based on schedule and metrics.

        Args:
            loss: Current loss value
            metrics: Current system metrics
        """
        try:
            self.step_count += 1

            # Calculate base learning rate from schedule
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = self.min_lr + (self.max_lr - self.min_lr) * (self.step_count / self.warmup_steps)
            else:
                # Cosine annealing
                progress = (self.step_count - self.warmup_steps) / (self.cycle_steps - self.warmup_steps)
                progress = min(1.0, progress)
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(progress * np.pi))

            # Modulate based on stability metrics
            stability = metrics.get('stability', 1.0)
            coherence = metrics.get('phase_coherence', 0.5)

            # Apply stability-based scaling
            lr_mod = lr * stability * coherence

            # Ensure lr is within bounds
            lr_final = np.clip(lr_mod, self.min_lr, self.max_lr)

            # Apply to optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_final

            # Track history
            self.update_history.append({
                'step': self.step_count,
                'base_lr': float(lr),
                'final_lr': float(lr_final),
                'stability': float(stability),
                'coherence': float(coherence),
                'timestamp': time.time()
            })

            self.current_lr = lr_final
            self.last_update_time = time.time()
            print(f"DEBUG: Learning Rate Scheduler - Step: {self.step_count}, Base LR: {lr:.6f}, Final LR: {lr_final:.6f}, Stability: {stability:.4f}, Coherence: {coherence:.4f}")

        except Exception as e:
            print(f"Error in learning rate adjustment: {e}")
            traceback.print_exc()

    def get_lr_stats(self) -> Dict[str, float]:
        """
        Get learning rate statistics.

        Returns:
            Dictionary with learning rate statistics
        """
        try:
            stats = {'current_lr': self.current_lr}

            if self.update_history:
                recent = list(self.update_history)[-100:]
                stats.update({
                    'mean_lr': np.mean([h['final_lr'] for h in recent]),
                    'lr_std': np.std([h['final_lr'] for h in recent]),
                    'warmup_progress': min(1.0, self.step_count / self.warmup_steps),
                    'cycle_progress': (min(1.0, (self.step_count - self.warmup_steps) / self.cycle_steps)
                                   if self.step_count > self.warmup_steps else 0.0)
                })

            return stats

        except Exception as e:
            print(f"Error getting LR stats: {e}")
            traceback.print_exc()
            return {'current_lr': self.current_lr}

class QuantumStateValidator:
    """Helper class for validating quantum state across components."""
    def __init__(self, agent=None):
        self.agent = agent
        self.validation_history = deque(maxlen=1000)
        self.validation_stats = defaultdict(int)
        self.validation_results = {}



    @staticmethod
    def validate_quantum_state(quantum_state: Any) -> bool:
        """
        Validate quantum state with comprehensive error checking.

        Args:
            quantum_state: Quantum state to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic attribute checks
            required_attrs = ['statevector', 'phase_coherence', 'phase', 'num_qubits']
            for attr in required_attrs:
                if not hasattr(quantum_state, attr):
                    print(f"Missing required attribute: {attr}")
                    return False

            # Validate statevector
            if quantum_state.statevector is None:
                print("Statevector is None")
                return False

            # Get statevector data
            if isinstance(quantum_state.statevector, np.ndarray):
                state_array = quantum_state.statevector
            elif hasattr(quantum_state.statevector, 'data'):
                state_array = np.array(quantum_state.statevector.data)
            else:
                print("Invalid statevector type")
                return False

            # Check statevector dimensions
            expected_dim = 2 ** quantum_state.num_qubits
            if state_array.shape != (expected_dim,):
                print(f"Invalid statevector dimension: {state_array.shape}, expected ({expected_dim},)")
                return False

            # Check normalization
            norm = np.linalg.norm(state_array)
            if not np.isclose(norm, 1.0, atol=1e-6):
                print(f"Statevector not normalized: norm = {norm}")
                return False

            # Validate phase coherence
            try:
                phase_coherence = float(np.real(quantum_state.phase_coherence))
                if not MINIMUM_COHERENCE_FLOOR <= phase_coherence <= 1.0:
                    print(f"Phase coherence out of range: {phase_coherence}")
                    return False
            except (TypeError, ValueError) as e:
                print(f"Invalid phase coherence value: {e}")
                return False

            # Validate phase
            try:
                phase = float(np.real(quantum_state.phase))
                if not 0 <= phase <= 2 * np.pi:
                    quantum_state.phase = phase % (2 * np.pi)  # Normalize phase
            except (TypeError, ValueError) as e:
                print(f"Invalid phase value: {e}")
                return False

            return True

        except Exception as e:
            print(f"Error in quantum state validation: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> bool:
        """
        Validate quantum metrics with bounds checking.

        Args:
            metrics: Dictionary of quantum metrics

        Returns:
            True if valid, False otherwise
        """
        try:
            required_metrics = {
                'phase_coherence': (MINIMUM_COHERENCE_FLOOR, 1.0),
                'normalized_entropy': (0.0, 1.0),
                'phase': (0.0, 2 * np.pi)
            }

            for metric, (min_val, max_val) in required_metrics.items():
                if metric not in metrics:
                    print(f"Missing required metric: {metric}")
                    return False

                try:
                    value = float(np.real(metrics[metric]))
                    if not min_val <= value <= max_val:
                        print(f"Metric {metric} out of range: {value}")
                        return False
                except (TypeError, ValueError) as e:
                    print(f"Invalid value for metric {metric}: {e}")
                    return False

            return True

        except Exception as e:
            print(f"Error validating metrics: {e}")
            traceback.print_exc()
            return False

    def validate_metrics_dict(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and clean metrics dictionary.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Validated and cleaned metrics dictionary
        """
        try:
            required_metrics = {
                'phase_coherence': MINIMUM_COHERENCE_FLOOR,
                'normalized_entropy': 0.0,
                'phase': 0.0,
                'phase_distinction': 0.0,
                'mean_coherence': MINIMUM_COHERENCE_FLOOR,
                'phase_stability': 1.0
            }

            validated_metrics = {}

            for key, default_value in required_metrics.items():
                try:
                    value = metrics.get(key, default_value)
                    # Convert to float and handle complex numbers
                    if isinstance(value, complex):
                        value = float(value.real)
                    else:
                        value = float(value)

                    # Validate ranges for specific metrics
                    if key in ['phase_coherence', 'normalized_entropy', 'mean_coherence']:
                        value = np.clip(value, 0.0, 1.0)
                    elif key == 'phase':
                        value = value % (2 * np.pi)

                    validated_metrics[key] = value

                except (TypeError, ValueError) as e:
                    print(f"Error validating metric {key}: {e}")
                    validated_metrics[key] = default_value

            return validated_metrics

        except Exception as e:
            print(f"Error in metrics validation: {e}")
            traceback.print_exc()
            return {k: v for k, v in required_metrics.items()}

    def validate_system_state(self, agent) -> Dict[str, bool]:
        """
        Perform comprehensive system state validation before recovery.

        Args:
            agent: The agent instance to validate

        Returns:
            Dictionary of validation results by component
        """
        try:
            validation_results = {}

            # 1. Validate quantum state
            validation_results['quantum_state'] = self.validate_quantum_state(agent.quantum_state)

            # 2. Validate surplus state
            validation_results['surplus_state'] = (
                hasattr(agent, 'surplus_dynamics') and
                hasattr(agent.surplus_dynamics, 'surplus_state') and
                isinstance(agent.surplus_dynamics.surplus_state, SurplusState) and
                agent.surplus_dynamics.surplus_state.validate()
            )

            # 3. Validate distinction dynamics
            validation_results['distinction'] = (
                hasattr(agent, 'distinction_level') and
                isinstance(agent.distinction_level, (int, float)) and
                0 <= agent.distinction_level <= 1.0
            )

            # 4. Validate transformer
            validation_results['transformer'] = (
                hasattr(agent, 'transformer') and
                isinstance(agent.transformer, nn.Module)
            )

            # 5. Validate memory
            validation_results['memory'] = (
                hasattr(agent, 'memory') and
                hasattr(agent.memory, 'store')
            )

            # 6. Validate ontological field
            validation_results['ontological_field'] = (
                hasattr(agent, 'ontological_field') and
                hasattr(agent.ontological_field, 'resistance')
            )

            # 7. Validate cognitive structure
            validation_results['cognitive_structure'] = (
                hasattr(agent, 'recursive_cognition') and
                hasattr(agent.recursive_cognition, 'update')
            )

            # Calculate overall validation
            validation_results['overall'] = all([
                validation_results.get('quantum_state', False),
                validation_results.get('surplus_state', False),
                validation_results.get('distinction', False)
            ])

            return validation_results

        except Exception as e:
            logger.error(f"Error in system validation: {e}")
            traceback.print_exc()
            return {'overall': False, 'error': str(e)}


