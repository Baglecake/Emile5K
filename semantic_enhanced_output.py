"""
Semantic Enhanced Output Module for Émile-2 Simulation
------------------------------------------------------
This module provides functions to process agent output, enhance its
semantic content, and integrate with different data stores.
"""
import logging
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback
from collections import deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import os
import hashlib
from datetime import datetime
import random
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.semantic_enhanced_output")

# Import necessary constants
from utilities import (
    DEVICE,
    MAX_ENTROPY,
    MINIMUM_COHERENCE_FLOOR,
    MOMENTUM_DECAY,
    DISTINCTION_ANCHOR_WEIGHT,
    PHASE_SCALING_FACTOR,
    COLLAPSE_DISSIPATION_THRESHOLD,
    COLLAPSE_DISSIPATION_RATE,
    CORE_DISTINCTION_UPDATE_RATE,
    HIDDEN_DIM,
    NUM_TRANSFORMER_HEADS,
    NUM_TRANSFORMER_LAYERS,
    GRADIENT_CLIP_VALUE,
    WEIGHT_DECAY,
    LEARNING_RATE,
    LEARNING_RATE_MIN,
    LEARNING_RATE_MAX,
    REWARD_SCALING,
    INSTABILITY_GRACE_PERIOD,
    SURPLUS_ADJUSTMENT_RATE,
    SURPLUS_THRESHOLD,
    MAX_SURPLUS,
    EXPULSION_RECOVERY_RATE,
    SURPLUS_RECYCLE_FRACTION,
    EVOLUTION_TIME
)
from data_classes import SurplusState, TransformerOutput
from base_quantum import BaseQuantumState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.semantic_enhanced_output")

# Import necessary constants
from utilities import (
    DEVICE,
    MAX_ENTROPY,
    MINIMUM_COHERENCE_FLOOR,
    MOMENTUM_DECAY,
    DISTINCTION_ANCHOR_WEIGHT,
    PHASE_SCALING_FACTOR,
    COLLAPSE_DISSIPATION_THRESHOLD,
    COLLAPSE_DISSIPATION_RATE,
    CORE_DISTINCTION_UPDATE_RATE,
    HIDDEN_DIM,
    NUM_TRANSFORMER_HEADS,
    NUM_TRANSFORMER_LAYERS,
    GRADIENT_CLIP_VALUE,
    WEIGHT_DECAY,
    LEARNING_RATE,
    LEARNING_RATE_MIN,
    LEARNING_RATE_MAX,
    REWARD_SCALING,
    INSTABILITY_GRACE_PERIOD,
    SURPLUS_ADJUSTMENT_RATE,
    SURPLUS_THRESHOLD,
    MAX_SURPLUS,
    EXPULSION_RECOVERY_RATE,
    SURPLUS_RECYCLE_FRACTION,
    EVOLUTION_TIME
)
from data_classes import SurplusState, TransformerOutput
from base_quantum import BaseQuantumState

# =============================================================================
# Data Structures for Tracking and Processing Semantic Data
# =============================================================================
@dataclass
class SemanticContext:
    """
    Container for the semantic context of an agent interaction.

    Includes the description, relevant state data, time, and other relevant
    information to track, process and analyze semantic context over time.
    """
    description: str
    timestamp: float
    agent_id: str
    surplus_state: SurplusState
    quantum_state_metrics: Dict[str, float]
    transformer_output: TransformerOutput
    external_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure values are correctly initialized."""
        try:
            if not isinstance(self.timestamp, (int, float)):
                self.timestamp = time.time()
                logger.warning("Invalid timestamp, using current time.")
            if not isinstance(self.agent_id, str):
                self.agent_id = str(self.agent_id)  # Attempt to convert
                logger.warning(f"Invalid agent_id type, converted to string: {self.agent_id}")
            if not isinstance(self.description, str):
                self.description = str(self.description)  # Attempt to convert
                logger.warning(f"Invalid description type, converted to string: {self.description}")
            if not isinstance(self.surplus_state, SurplusState):
                logger.error("Invalid surplus_state type, defaulting to empty SurplusState")
                self.surplus_state = SurplusState()
            if not isinstance(self.quantum_state_metrics, dict):
                logger.error("Invalid quantum_state_metrics type, defaulting to empty dict")
                self.quantum_state_metrics = {}
            if not isinstance(self.transformer_output, TransformerOutput):
                 logger.error("Invalid transformer_output type, defaulting to default TransformerOutput")
                 self.transformer_output = TransformerOutput(torch.tensor(0.0))

            if not isinstance(self.external_context, dict):
                self.external_context = {}
                logger.warning("Invalid external_context type, defaulted to dict")
            if not isinstance(self.metadata, dict):
                self.metadata = {}
                logger.warning("Invalid metadata type, defaulted to dict")
        except Exception as e:
            logger.error(f"Error in SemanticContext post-init: {e}")
            self.timestamp = time.time()
            self.agent_id = "unknown"
            self.description = "default"
            self.surplus_state = SurplusState()
            self.quantum_state_metrics = {}
            self.transformer_output = TransformerOutput(torch.tensor(0.0))
            self.external_context = {}
            self.metadata = {}

    def validate(self) -> bool:
        """Validate SemanticContext fields."""
        try:
            if not isinstance(self.timestamp, (int, float)):
                logger.error(f"Invalid type for timestamp: {type(self.timestamp)}")
                return False
            if not isinstance(self.agent_id, str):
                logger.error(f"Invalid type for agent_id: {type(self.agent_id)}")
                return False
            if not isinstance(self.description, str):
                logger.error(f"Invalid type for description: {type(self.description)}")
                return False
            if not isinstance(self.surplus_state, SurplusState):
                logger.error(f"Invalid type for surplus_state: {type(self.surplus_state)}")
                return False
            if not isinstance(self.quantum_state_metrics, dict):
                logger.error(f"Invalid type for quantum_state_metrics: {type(self.quantum_state_metrics)}")
                return False
            if not isinstance(self.transformer_output, TransformerOutput):
                logger.error(f"Invalid type for transformer_output: {type(self.transformer_output)}")
                return False
            if not isinstance(self.external_context, dict):
                logger.error(f"Invalid type for external_context: {type(self.external_context)}")
                return False
            if not isinstance(self.metadata, dict):
                logger.error(f"Invalid type for metadata: {type(self.metadata)}")
                return False
            if not self.surplus_state.validate():
                 logger.error("Surplus state is invalid")
                 return False
            if not self.transformer_output.validate():
                logger.error("Transformer output is invalid")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating SemanticContext: {e}")
            return False

    def copy(self) -> 'SemanticContext':
        """
        Create a deep copy of the SemanticContext object.

        Returns:
            A new SemanticContext object with copied attributes, or a default
            instance if copying fails
        """
        try:
            return SemanticContext(
                description=str(self.description) if self.description else "",  # Ensure string
                timestamp=float(self.timestamp),  # Ensure float
                agent_id=str(self.agent_id) if self.agent_id else "default_agent",  # Ensure string
                surplus_state=self.surplus_state.copy(),
                quantum_state_metrics=dict(self.quantum_state_metrics),
                transformer_output=self.transformer_output.to(self.transformer_output.device),
                external_context=dict(self.external_context),
                metadata=dict(self.metadata)
            )
        except Exception as e:
            logger.error(f"Error in copying SemanticContext: {e}")
            # Return default
            return SemanticContext(
                    description = "default",
                    timestamp = time.time(),
                    agent_id = "default_agent",
                    surplus_state = SurplusState(),
                    quantum_state_metrics = {},
                    transformer_output = TransformerOutput(torch.tensor(0.0)),
                    external_context={},
                    metadata={}
                )

@dataclass
class SemioticModelOutput:
    """Container for the output of the semiotic model."""
    prediction: torch.Tensor
    value_estimate: torch.Tensor
    relevance_score: float
    attention_weights: Dict[str, torch.Tensor]
    entropy: Optional[torch.Tensor] = None
    coherence_estimate: Optional[torch.Tensor] = None

    def __post_init__(self):
        """
        Ensure all tensors are initialized and on the correct device.
        """
        try:
            # Ensure prediction is a tensor
            if not isinstance(self.prediction, torch.Tensor):
                logger.warning(f"Prediction is not a tensor, converting from {type(self.prediction)}")
                try:
                    self.prediction = torch.tensor(self.prediction, dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Could not convert prediction to tensor: {e}")
                    self.prediction = torch.tensor(0.0)

            # Ensure value_estimate is a tensor
            if not isinstance(self.value_estimate, torch.Tensor):
                logger.warning(f"Value estimate is not a tensor, converting from {type(self.value_estimate)}")
                try:
                    self.value_estimate = torch.tensor(self.value_estimate, dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Could not convert value estimate to tensor: {e}")
                    self.value_estimate = torch.tensor(0.0)

            # Ensure relevance_score is a float
            if not isinstance(self.relevance_score, float):
                logger.warning(f"Relevance score is not a float, converting from {type(self.relevance_score)}")
                try:
                   self.relevance_score = float(self.relevance_score)
                except Exception as e:
                    logger.error(f"Could not convert relevance score to float: {e}, using default 0.0")
                    self.relevance_score = 0.0

            # Handle entropy
            if self.entropy is None:
                self.entropy = torch.tensor(0.0, device=self.prediction.device)
            elif not isinstance(self.entropy, torch.Tensor):
                try:
                    self.entropy = torch.tensor(self.entropy, device=self.prediction.device)
                except Exception as e:
                    logger.error(f"Could not convert entropy to tensor: {e}")
                    self.entropy = torch.tensor(0.0, device=self.prediction.device)

            # Handle coherence estimate
            if self.coherence_estimate is None:
                self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=self.prediction.device)
            elif not isinstance(self.coherence_estimate, torch.Tensor):
                try:
                    self.coherence_estimate = torch.tensor(self.coherence_estimate, device=self.prediction.device)
                except Exception as e:
                    logger.error(f"Could not convert coherence_estimate to tensor: {e}")
                    self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=self.prediction.device)

            # Ensure attention weights are proper tensors
            if not isinstance(self.attention_weights, dict):
                logger.warning(f"Attention weights is not a dict, initializing empty dict")
                self.attention_weights = {}
            else:
                for key, value in list(self.attention_weights.items()):
                    if not isinstance(value, torch.Tensor):
                        try:
                            self.attention_weights[key] = torch.tensor(value, device=self.prediction.device)
                        except Exception as e:
                            logger.error(f"Could not convert attention weight {key} to tensor: {e}")
                            del self.attention_weights[key]
        except Exception as e:
                logger.error(f"Error during SemioticModelOutput initialization: {e}")

                device = getattr(self.prediction, 'device', None)
                if device is None:
                  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.prediction = torch.tensor(0.0, device=device)
                self.value_estimate = torch.tensor(0.0, device=device)
                self.relevance_score = 0.0
                self.attention_weights = {}
                self.entropy = torch.tensor(0.0, device=device)
                self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)

    def validate(self) -> bool:
         """Validate SemioticModelOutput fields."""
         try:
              # Check prediction tensor
             if not isinstance(self.prediction, torch.Tensor):
                  logger.error("Invalid prediction type")
                  return False
             if torch.isnan(self.prediction).any():
                  logger.error("NaN values in prediction")
                  return False

             # Check value estimate tensor
             if not isinstance(self.value_estimate, torch.Tensor):
                 logger.error("Invalid value estimate type")
                 return False
             if torch.isnan(self.value_estimate).any():
                 logger.error("NaN values in value estimate")
                 return False

             # Check relevance score
             if not isinstance(self.relevance_score, float):
                 logger.error(f"Invalid type for relevance_score: {type(self.relevance_score)}")
                 return False
             if np.isnan(self.relevance_score):
                 logger.error("NaN value in relevance score")
                 return False

             # Check attention weights
             if not isinstance(self.attention_weights, dict):
                 logger.error("Invalid attention weights type")
                 return False

             for key, value in self.attention_weights.items():
                 if not isinstance(value, torch.Tensor):
                      logger.error(f"Invalid attention weight tensor for {key}")
                      return False
                 if torch.isnan(value).any():
                     logger.error(f"NaN values in attention weights for {key}")
                     return False

             return True
         except Exception as e:
              logger.error(f"Error validating SemioticModelOutput: {e}")
              return False

    @property
    def device(self) -> torch.device:
        """Get the device of the prediction tensor."""
        return self.prediction.device

    def to(self, device: torch.device) -> 'SemioticModelOutput':
        """Move all tensors to specified device."""
        try:
            self.prediction = self.prediction.to(device)
            self.value_estimate = self.value_estimate.to(device)
            if self.entropy is not None:
                self.entropy = self.entropy.to(device)
            if self.coherence_estimate is not None:
                self.coherence_estimate = self.coherence_estimate.to(device)

            # Move attention weights
            for k, v in self.attention_weights.items():
                 if isinstance(v, torch.Tensor):
                      self.attention_weights[k] = v.to(device)

            return self
        except Exception as e:
            logger.error(f"Error moving tensors to device: {e}")
            return self

    def get_prediction_value(self) -> float:
        """Safely extract prediction value as float."""
        try:
            if self.prediction is None:
                return 0.0

            # Handle different tensor shapes
            if self.prediction.dim() == 0:  # Scalar
                return self.prediction.item()
            elif self.prediction.dim() == 1:  # Vector
                return self.prediction[0].item()
            elif self.prediction.dim() == 2:  # Matrix
                return self.prediction[0, 0].item()
            elif self.prediction.dim() == 3:  # 3D tensor
                return self.prediction[0, 0, 0].item()
            else:
                return self.prediction.mean().item()
        except Exception as e:
            logger.error(f"Error extracting prediction value: {e}")
            return 0.0

@dataclass
class SemioticCacheEntry:
    """Container for cached semiotic context and model output."""
    context: SemanticContext
    model_output: SemioticModelOutput
    creation_time: float = field(default_factory=time.time)

# =============================================================================
# Core Semiotic Processing Classes
# =============================================================================
class AbstractSemioticModel(ABC):
    """
    Abstract base class for semiotic models.

    Defines the basic interface for semiotic models used to analyze and
    process semantic information.
    """
    def __init__(self):
        """Initialize the semiotic model."""
        self.cache = {}
        self.cache_limit = 100
        self.cache_lifespan = 60  # in seconds
        self.logger = logger.getChild(self.__class__.__name__)  # Set up logger for each class
        self.device = DEVICE

    def _generate_key(self, context: SemanticContext) -> str:
        """
        Generate a unique key for the given semantic context.

        Uses a hash of the description and agent ID to create a key.

        Args:
            context: SemanticContext object

        Returns:
            A string representing the unique cache key
        """
        try:
            key_string = f"{context.description}-{context.agent_id}"
            return hashlib.sha256(key_string.encode()).hexdigest()
        except Exception as e:
           self.logger.error(f"Error generating cache key: {e}")
           return "default_key" # Default key to prevent further issues

    def _cache_cleanup(self):
        """
        Clean up the cache by removing expired entries.

        Iterates through the cache and removes any entry whose lifespan has
        exceeded the defined limit.
        """
        now = time.time()
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if (now - entry.creation_time) > self.cache_lifespan
        ]
        for key in keys_to_remove:
            del self.cache[key]

        # Trim cache if it exceeds the limit
        if len(self.cache) > self.cache_limit:
            sorted_cache = sorted(self.cache.items(), key=lambda item: item[1].creation_time)
            keys_to_trim = [key for key, _ in sorted_cache[:len(self.cache) - self.cache_limit]]
            for key in keys_to_trim:
                 del self.cache[key]

    def cache_context(self, context: SemanticContext, model_output: SemioticModelOutput):
        """Cache the semantic context and model output."""
        try:
            key = self._generate_key(context)
            self.cache[key] = SemioticCacheEntry(context=context.copy(), model_output=model_output.to(self.device))
            self._cache_cleanup()
        except Exception as e:
            self.logger.error(f"Error caching context: {e}")

    def get_cached_context(self, context: SemanticContext) -> Optional[SemioticCacheEntry]:
        """
        Retrieve the cached context if available.

        Args:
            context: SemanticContext object

        Returns:
            Cached SemioticCacheEntry or None if not found
        """
        try:
            key = self._generate_key(context)
            if key in self.cache:
                return self.cache[key]
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached context: {e}")
            return None

    @abstractmethod
    def process_context(self, context: SemanticContext) -> SemioticModelOutput:
         """
         Abstract method to process a semantic context.

         Args:
             context: Semantic context object

         Returns:
             Semiotic model output
         """
         pass

    # =============================================================================
# Concrete Semiotic Model Implementations
# =============================================================================
class LegacySemioticModel(nn.Module):
    """
    Legacy semiotic model using a simple feed-forward network.

    This model processes the semantic context data by converting it to numerical
    vectors, passing it through a linear layer, and producing a prediction.

    This is intended as a simplified baseline to compare against transformer based models.
    """
    def __init__(self, input_dim: int = 20, hidden_dim: int = 30):
        """
        Initialize the LegacySemioticModel.

        Args:
            input_dim: Dimensionality of the input vector.
            hidden_dim: Dimensionality of the hidden layer.
        """
        super(LegacySemioticModel, self).__init__()
        self.logger = logger.getChild(self.__class__.__name__)  # Set up logger for each class
        self.device = DEVICE

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1) # Single output for relevance score
        self.value_fc = nn.Linear(hidden_dim,1)  # Linear layer for value estimation
        self.sigmoid = nn.Sigmoid()  # Use Sigmoid for relevance between 0 and 1
        self.softmax = nn.Softmax(dim=-1)  # For attention weights
        self.dropout = nn.Dropout(0.1)  # Optional dropout for regularization
        self.hidden_dim = hidden_dim # Store for output
        self.to(self.device)  # Move model to correct device
        self.logger.info(f"Legacy Semiotic Model initialized on device: {self.device}")


    def _process_input(self, context: SemanticContext) -> torch.Tensor:
        """
        Process the semantic context into a numerical input vector.

        This method combines different aspects of the context:
          * Quantum state metrics (phase, coherence, entropy)
          * Total surplus
          * Transformer output prediction

        Args:
            context: SemanticContext object

        Returns:
            A PyTorch tensor representing the processed input vector, or an empty tensor if an error occurs.
        """
        try:
             # Get relevant metrics
             quantum_metrics = context.quantum_state_metrics
             total_surplus = context.surplus_state.total_surplus()
             transformer_prediction = context.transformer_output.get_prediction_value()

             # Convert metrics to a numpy array
             input_array = np.array([
                quantum_metrics.get('phase', 0.0),
                quantum_metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR),
                quantum_metrics.get('normalized_entropy', 0.0),
                total_surplus,
                transformer_prediction
             ])

             # Add padding with zeros if less than input_dim
             if len(input_array) < self.hidden_dim:
                 padding = np.zeros(self.hidden_dim - len(input_array))
                 input_array = np.concatenate((input_array, padding))

             # Convert to tensor
             input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension

             return input_tensor
        except Exception as e:
             self.logger.error(f"Error processing input for legacy model: {e}")
             return torch.empty(0).to(self.device)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the LegacySemioticModel.

        Args:
            x: Input tensor.

        Returns:
            Tuple of:
            - Prediction tensor
            - Value Estimate tensor
            - Attention weights dictionary
        """
        try:
           # Pass through fully connected layers
           x = self.fc1(x)
           x = self.relu(x)
           x = self.dropout(x)  # Apply dropout
           prediction = self.sigmoid(self.fc2(x)) # Use Sigmoid for prediction
           value_estimate = self.value_fc(x) # Output for value estimate

           # Create dummy attention weights (for compatibility with other models)
           attention_weights = {
               "dummy_attention": torch.ones(x.shape[0],self.hidden_dim).to(self.device)
           }

           return prediction, value_estimate, attention_weights
        except Exception as e:
            self.logger.error(f"Error in forward pass of legacy model: {e}")
            # Return default
            return torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device), {}


class EnhancedSemioticModel(AbstractSemioticModel):
    """
    Enhanced semiotic model using a transformer-based network.

    This model processes the semantic context using a transformer encoder,
    allowing for more nuanced and contextual understanding of semantic information.
    """
    def __init__(self, input_dim: int = 20,
                 hidden_dim: int = HIDDEN_DIM,
                 num_heads: int = NUM_TRANSFORMER_HEADS,
                 num_layers: int = NUM_TRANSFORMER_LAYERS):
        """
        Initialize the EnhancedSemioticModel.

        Args:
            input_dim: Dimensionality of the input vector.
            hidden_dim: Dimensionality of the transformer model's hidden layers.
            num_heads: Number of attention heads in the transformer model.
            num_layers: Number of transformer encoder layers.
        """
        super().__init__()
        self.logger = logger.getChild(self.__class__.__name__)  # Set up logger for each class
        self.device = DEVICE

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1, batch_first=True),
            num_layers=num_layers
        )
        self.fc_prediction = nn.Linear(hidden_dim, 1) # single output for relevance score
        self.fc_value = nn.Linear(hidden_dim, 1)  # Linear layer for value estimation
        self.sigmoid = nn.Sigmoid()  # For relevance score output between 0 and 1
        self.softmax = nn.Softmax(dim=-1)  # For attention weights
        self.to(self.device)
        self.logger.info(f"Enhanced Semiotic Model initialized on device: {self.device}")

    def _process_input(self, context: SemanticContext) -> torch.Tensor:
        """
        Process semantic context into a numerical input vector for transformer.

        Combines quantum metrics (phase, coherence, entropy), total surplus, and
        transformer output, padding to match model's hidden dimension.

        Args:
            context: SemanticContext object

        Returns:
            A PyTorch tensor representing the processed input vector, or an empty tensor on error
        """
        try:
             # Get relevant metrics
             quantum_metrics = context.quantum_state_metrics
             total_surplus = context.surplus_state.total_surplus()
             transformer_prediction = context.transformer_output.get_prediction_value()

             # Convert metrics to a numpy array
             input_array = np.array([
                quantum_metrics.get('phase', 0.0),
                quantum_metrics.get('phase_coherence', MINIMUM_COHERENCE_FLOOR),
                quantum_metrics.get('normalized_entropy', 0.0),
                total_surplus,
                transformer_prediction
             ])
             # Add padding with zeros if less than hidden_dim
             if len(input_array) < self.hidden_dim:
                 padding = np.zeros(self.hidden_dim - len(input_array))
                 input_array = np.concatenate((input_array, padding))

             # Convert to tensor and add batch and sequence dimensions
             input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # B, S, F
             return input_tensor
        except Exception as e:
             self.logger.error(f"Error processing input for enhanced model: {e}")
             return torch.empty(0).to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the EnhancedSemioticModel.

        Args:
            x: Input tensor.

        Returns:
            Tuple of:
            - Prediction tensor
            - Value Estimate tensor
            - Attention weights dictionary
        """
        try:
           # Pass through transformer encoder
           encoded_output = self.transformer_encoder(x) # B, S, F
           encoded_output = self.dropout(encoded_output) #Apply dropout
           # Get the output from the last sequence element
           last_seq_output = encoded_output[:, -1, :]

           # Pass the last element through fully connected layer
           prediction = self.sigmoid(self.fc_prediction(last_seq_output)) # Use Sigmoid for prediction
           value_estimate = self.fc_value(last_seq_output)  #Output for value estimate

           # Extract attention weights (average across heads for all layers)
           attention_weights = {}
           for layer_idx, layer in enumerate(self.transformer_encoder.layers):
               attn_weights = layer.self_attn.attn_output_weights
               if attn_weights is not None:
                   avg_attn_weights = torch.mean(attn_weights, dim=1)  # Average across heads
                   attention_weights[f'layer_{layer_idx}_attention'] = avg_attn_weights

           return prediction, value_estimate, attention_weights
        except Exception as e:
           self.logger.error(f"Error in forward pass of enhanced model: {e}")
           return torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device), {}

    def process_context(self, context: SemanticContext) -> SemioticModelOutput:
        """
        Process the semantic context using the enhanced transformer model.

        This includes caching, preprocessing the input, and making predictions.

        Args:
            context: SemanticContext object

        Returns:
            SemioticModelOutput object with prediction and attention weights, or default output on error
        """
        try:
            cached_output = self.get_cached_context(context)
            if cached_output:
                self.logger.debug("Retrieved context from cache")
                return cached_output.model_output.to(self.device)

            input_tensor = self._process_input(context)
            if input_tensor.numel() == 0:  # Check if tensor is empty
                self.logger.warning("Empty input tensor, returning default semiotic output")
                return SemioticModelOutput(prediction=torch.tensor(0.0).to(self.device), value_estimate=torch.tensor(0.0).to(self.device), relevance_score=0.0, attention_weights={})

            prediction, value_estimate, attention_weights = self.forward(input_tensor)
            relevance_score = prediction.item() # Extract relevance score

            output = SemioticModelOutput(
                prediction=prediction.to(self.device),
                value_estimate=value_estimate.to(self.device),
                relevance_score=float(relevance_score),
                attention_weights=attention_weights
            )
            self.cache_context(context, output)
            return output.to(self.device)

        except Exception as e:
            self.logger.error(f"Error in processing context with enhanced model: {e}")
            return SemioticModelOutput(prediction=torch.tensor(0.0).to(self.device), value_estimate=torch.tensor(0.0).to(self.device), relevance_score=0.0, attention_weights={})
# =============================================================================
# Functions to orchestrate semantic processing and output
# =============================================================================
class SemanticEnhancedOutput:
    """
    Enhanced symbolic output generator that integrates with the semantic ML model
    to dynamically adapt and enrich Émile's symbolic expressions.

    This creates a bi-directional learning loop where:
    1. The semantic model learns from Émile's expressions
    2. Émile's expressions adapt based on semantic model feedback

    Enhancements:
    - Improved handling of numeric values in semantic expression generation
    - Numeric-semantic coupling for more informative symbolic outputs
    - Enhanced dimensionality representation with quantitative descriptors
    """
    def __init__(self, model_type: str = "enhanced"):
        """
        Initialize the SemanticEnhancedOutput module.

        Args:
             model_type: Type of semiotic model to use ('legacy' or 'enhanced')
        """
        self.logger = logger.getChild(self.__class__.__name__)  # Set up logger for each class
        self.device = DEVICE
        self.model_type = model_type
        self.semiotic_model = self._initialize_semiotic_model()
        self.output_buffer = deque(maxlen=50)
        self.last_processed_context = None
        self.logger.info(f"Semantic Enhanced Output Initialized with {model_type} model.")

        # Integrations with semantic model
        self.semantic_model_path = semantic_model_path
        self.device = device
        self.semantic_embeddings = {}
        self.semantic_coherence_threshold = 0.6
        self.semantic_update_frequency = 10  # Update vocabulary after every N expressions
        self.expression_counter = 0
        self.dynamic_vocabulary_enabled = True

        # Maintain original basic vocabulary
        self.state_descriptors = [
            "Flux", "Equilibrium", "Distinction", "Recursion",
            "Convergence", "Divergence", "Resonance", "Coherence",
            "Entanglement", "Superposition", "Bifurcation", "Integration"
        ]

        self.relations = [
            "aligns with", "dissolves across", "bends toward",
            "extends beyond", "contracts into", "resonates within",
            "differentiates from", "converges upon", "enfolds",
            "stabilizes within", "emerges through", "transcends"
        ]

        self.surplus_concepts = [
            "stability", "recursion", "entropy", "phase shift",
            "emergence", "ontology", "distinction", "coherence",
            "complexity", "dimensionality", "feedback", "symmetry"
        ]

        # Additional vocabularies for more complex expressions
        self.modifiers = [
            "partially", "deeply", "recursively", "gradually",
            "suddenly", "coherently", "chaotically", "uniquely",
            "systematically", "emergently", "distinctively", "subtly"
        ]

        self.secondary_concepts = [
            "phase space", "attractor", "strange loop", "dynamic pattern",
            "boundary condition", "information field", "critical point", "nonlinearity",
            "fractal domain", "resonant structure", "emergent property", "computational ontology"
        ]

        # Add numeric modifiers to enhance quantitative descriptions
        self.numeric_modifiers = [
            "increasing", "decreasing", "oscillating", "accelerating",
            "decelerating", "threshold", "critical", "harmonic",
            "resonant", "quantized", "continuous", "discrete"
        ]

        # Add numeric transformations for embedding numeric values in expressions
        self.numeric_transformations = [
            "multiplied", "divided", "amplified", "attenuated",
            "exponential", "logarithmic", "scaled", "normalized",
            "bounded", "unbounded", "fractional", "integral"
        ]

        # For dynamic vocabulary expansion
        self.dynamic_state_descriptors = self.state_descriptors.copy()
        self.dynamic_relations = self.relations.copy()
        self.dynamic_surplus_concepts = self.surplus_concepts.copy()
        self.dynamic_modifiers = self.modifiers.copy()
        self.dynamic_secondary_concepts = self.secondary_concepts.copy()
        self.dynamic_numeric_modifiers = self.numeric_modifiers.copy()
        self.dynamic_numeric_transformations = self.numeric_transformations.copy()

        # Semantic coherence tracking for each vocabulary term
        self.vocabulary_coherence = {
            'descriptors': {term: 0.5 for term in self.state_descriptors},
            'relations': {term: 0.5 for term in self.relations},
            'concepts': {term: 0.5 for term in self.surplus_concepts},
            'modifiers': {term: 0.5 for term in self.modifiers},
            'secondary': {term: 0.5 for term in self.secondary_concepts},
            'numeric_modifiers': {term: 0.5 for term in self.numeric_modifiers},
            'numeric_transformations': {term: 0.5 for term in self.numeric_transformations}
        }

        # Track historical expressions
        self.expression_history = []
        self.emergence_events = []
        self.pattern_history = deque(maxlen=100)
        self.frequency_analysis = {}

        # Store last expression components for external access
        self.last_expression_components = {}

        # Thresholds for state categorization
        self.coherence_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.distinction_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        self.surplus_thresholds = {
            'low': 1.0,
            'medium': 3.0,
            'high': 6.0
        }

        # Advanced pattern recognition
        self.transition_matrix = np.zeros((vocabulary_size, vocabulary_size))
        self.transition_counts = np.zeros((vocabulary_size, vocabulary_size))
        self.descriptor_index = {desc: i for i, desc in enumerate(self.state_descriptors)}

        # Numeric integration parameters
        self.numeric_influence = 0.4  # Weight of numeric values in expression generation
        self.numeric_coherence_threshold = 0.3  # Threshold for numeric coherence
        self.numeric_memory = deque(maxlen=50)  # Remember recent numeric values
        self.numeric_trends = {}  # Track trends in numeric values

        # Tracks the last generated elements for transition analysis
        self.last_descriptor = None
        self.expression_complexity = 1.0

        # Initialize timestamp for real-time tracking
        self.start_time = time.time()

        # Load semantic model if path provided
        if semantic_model_path and os.path.exists('/content/emile_semantic_ml_mini.pt'):
            self.load_semantic_model()

        # Initialize semantic cache directory
        self.semantic_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic_cache")
        os.makedirs(self.semantic_cache_dir, exist_ok=True)

        # Try to load cached semantic embeddings
        self.load_semantic_cache()

        # Start background thread for adaptive vocabulary updates
        self.semantic_update_thread = threading.Thread(target=self._background_vocabulary_update, daemon=True)
        self.semantic_update_thread.start()

    def _initialize_semiotic_model(self) -> AbstractSemioticModel:
        """Initialize the semiotic model based on the specified type."""
        try:
            if self.model_type == 'legacy':
                self.logger.info("Initializing Legacy Semiotic Model.")
                return LegacySemioticModel()
            elif self.model_type == 'enhanced':
                self.logger.info("Initializing Enhanced Semiotic Model.")
                return EnhancedSemioticModel()
            else:
                self.logger.warning(f"Invalid model type: {self.model_type}, defaulting to enhanced model")
                return EnhancedSemioticModel()
        except Exception as e:
            self.logger.error(f"Error initializing semiotic model: {e}")
            return EnhancedSemioticModel

    def load_semantic_model(self):
        """
        Loads the semantic model for integration with symbolic expression generation.
        This establishes the connection to the ML model with improved model architecture handling.
        """
        try:
            # Import necessary modules here to avoid import errors if not available
            from transformers import AutoTokenizer, AutoModel
            from sentence_transformers import SentenceTransformer
            import traceback

            print(f"Loading semantic model from {self.semantic_model_path}")

            # Load sentence transformer for concept embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
            self.sentence_model.eval()

            # Load the semantic model checkpoint
            checkpoint = torch.load(self.semantic_model_path, map_location=self.device)

            print("Detected model architecture mismatch, remapping keys...")

            # Create our custom model that matches the checkpoint architecture
            self.semantic_model = LegacySemioticModel(hidden_dim=384).to(self.device)

            # Try to load transformer model
            try:
                transformer_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
                transformer_model.eval()
                self.semantic_model.encoder = transformer_model
            except Exception as e:
                print(f"Error loading transformer model: {e}")

            # Prepare state dict from checkpoint
            state_dict = None
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Try direct state_dict loading
                state_dict = checkpoint

            # Try loading with flexible options
            try:
                # First, try strict loading
                self.semantic_model.load_state_dict(state_dict)
            except Exception as strict_error:
                print(f"Warning: Strict loading failed, attempting with strict=False: {strict_error}")
                try:
                    # Try non-strict loading to allow missing or unexpected keys
                    self.semantic_model.load_state_dict(state_dict, strict=False)
                except Exception as nonstrict_error:
                    print(f"Error loading semantic model: {nonstrict_error}")
                    traceback.print_exc()
                    self.semantic_model = None
                    return False

            self.semantic_model.eval()

            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                # Create a simple tokenizer as fallback
                class SimpleTokenizer:
                    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=512):
                        return {
                            'input_ids': torch.tensor([[1, 2, 3]]).to(self.device),
                            'attention_mask': torch.tensor([[1, 1, 1]]).to(self.device)
                        }
                self.tokenizer = SimpleTokenizer()

            print("✅ Semantic model loaded successfully")

            # Pre-compute embeddings for vocabulary terms
            self._precompute_vocabulary_embeddings()

            return True
        except Exception as e:
            print(f"Error loading semantic model: {e}")
            traceback.print_exc()
            print("⚠️ Semantic integration will be disabled")
            self.semantic_model = None
            return False

    def _precompute_vocabulary_embeddings(self):
        """
        Precompute embeddings for all vocabulary terms for faster semantic matching
        """
        if not self.semantic_model:
            return

        try:
            print("Precomputing semantic embeddings for vocabulary...")

            # Combine all vocabulary terms
            all_terms = (
                self.state_descriptors +
                self.relations +
                self.surplus_concepts +
                self.modifiers +
                self.secondary_concepts +
                self.numeric_modifiers +
                self.numeric_transformations
            )

            # Compute embeddings in batches
            batch_size = 16
            for i in range(0, len(all_terms), batch_size):
                batch = all_terms[i:min(i+batch_size, len(all_terms))]

                # Get embeddings from sentence model
                with torch.no_grad():
                    embeddings = self.sentence_model.encode(batch, convert_to_tensor=True)

                # Store embeddings in dictionary
                for j, term in enumerate(batch):
                    self.semantic_embeddings[term] = embeddings[j].cpu()

            print(f"✅ Computed embeddings for {len(self.semantic_embeddings)} vocabulary terms")

            # Save embeddings to cache
            self.save_semantic_cache()

        except Exception as e:
            print(f"Error precomputing vocabulary embeddings: {e}")

    def save_semantic_cache(self):
        """Save semantic embeddings to cache file"""
        try:
            cache_file = os.path.join(self.semantic_cache_dir, "vocab_embeddings.pt")

            # Convert embeddings to CPU tensors
            cpu_embeddings = {k: v.cpu() for k, v in self.semantic_embeddings.items()}

            # Save to file
            torch.save(cpu_embeddings, cache_file)
            print(f"Saved semantic embeddings cache to {cache_file}")

            # Also save vocabulary coherence
            coherence_file = os.path.join(self.semantic_cache_dir, "vocab_coherence.json")
            with open(coherence_file, 'w') as f:
                json.dump(self.vocabulary_coherence, f)

        except Exception as e:
            print(f"Error saving semantic cache: {e}")

    def load_semantic_cache(self):
        """Load semantic embeddings from cache file"""
        try:
            cache_file = os.path.join(self.semantic_cache_dir, "vocab_embeddings.pt")
            if os.path.exists(cache_file):
                self.semantic_embeddings = torch.load(cache_file)
                print(f"Loaded semantic embeddings for {len(self.semantic_embeddings)} terms from cache")

            # Load vocabulary coherence
            coherence_file = os.path.join(self.semantic_cache_dir, "vocab_coherence.json")
            if os.path.exists(coherence_file):
                with open(coherence_file, 'r') as f:
                    self.vocabulary_coherence = json.load(f)
                print("Loaded vocabulary coherence from cache")

            return True
        except Exception as e:
            print(f"Error loading semantic cache: {e}")
            return False

    def _calculate_semantic_coherence(self, expression):
        """
        Calculate the semantic coherence of an expression using the semantic model

        Args:
            expression: The symbolic expression to evaluate

        Returns:
            Coherence score between 0-1
        """
        if not self.semantic_model:
            return 0.5  # Default if no model

        try:
            # Tokenize the expression
            inputs = self.tokenizer(expression, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512).to(self.device)

            # Get expression embedding from model
            with torch.no_grad():
                # First encode with base transformer
                base_outputs = self.transformer_base(**inputs)
                pooled_output = base_outputs.last_hidden_state[:, 0, :]
                # Then use our semantic model
                expression_embedding = self.semantic_model(pooled_output)

            # Get reference embeddings
            if self.expression_history:
                # Use historical expressions as reference
                reference_texts = [entry['expression'] for entry in self.expression_history[-10:]]

                with torch.no_grad():
                    reference_embeddings = self.sentence_model.encode(reference_texts, convert_to_tensor=True).to(self.device)

                # Calculate average cosine similarity
                cos_sims = torch.nn.functional.cosine_similarity(
                    expression_embedding.unsqueeze(0), reference_embeddings
                )

                # Return maximum similarity as coherence score
                return torch.max(cos_sims).item()
            else:
                # If no history, return 0.5 as default
                return 0.5

        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.5

    def _update_vocabulary_coherence(self, components):
        """
        Update coherence scores for vocabulary terms based on semantic model feedback

        Args:
            components: Dictionary of components used in the expression
        """
        if not self.semantic_model:
            return

        try:
            # Get components
            descriptor = components.get('descriptor')
            relation = components.get('relation')
            concept = components.get('concept')
            modifier = components.get('modifier')
            secondary = components.get('secondary')
            numeric_modifier = components.get('numeric_modifier')
            numeric_transformation = components.get('numeric_transformation')

            # Get full expression
            expression = components.get('full_expression', '')

            # Calculate coherence for the full expression
            expression_coherence = self._calculate_semantic_coherence(expression)

            # Update coherence for each component
            alpha = 0.2  # Learning rate for updates

            if descriptor in self.vocabulary_coherence['descriptors']:
                current = self.vocabulary_coherence['descriptors'][descriptor]
                self.vocabulary_coherence['descriptors'][descriptor] = (1-alpha) * current + alpha * expression_coherence

            if relation in self.vocabulary_coherence['relations']:
                current = self.vocabulary_coherence['relations'][relation]
                self.vocabulary_coherence['relations'][relation] = (1-alpha) * current + alpha * expression_coherence

            if concept in self.vocabulary_coherence['concepts']:
                current = self.vocabulary_coherence['concepts'][concept]
                self.vocabulary_coherence['concepts'][concept] = (1-alpha) * current + alpha * expression_coherence

            if modifier and modifier in self.vocabulary_coherence['modifiers']:
                current = self.vocabulary_coherence['modifiers'][modifier]
                self.vocabulary_coherence['modifiers'][modifier] = (1-alpha) * current + alpha * expression_coherence

            if secondary and secondary in self.vocabulary_coherence['secondary']:
                current = self.vocabulary_coherence['secondary'][secondary]
                self.vocabulary_coherence['secondary'][secondary] = (1-alpha) * current + alpha * expression_coherence

            # Update coherence for numeric components
            if numeric_modifier and numeric_modifier in self.vocabulary_coherence['numeric_modifiers']:
                current = self.vocabulary_coherence['numeric_modifiers'][numeric_modifier]
                self.vocabulary_coherence['numeric_modifiers'][numeric_modifier] = (1-alpha) * current + alpha * expression_coherence

            if numeric_transformation and numeric_transformation in self.vocabulary_coherence['numeric_transformations']:
                current = self.vocabulary_coherence['numeric_transformations'][numeric_transformation]
                self.vocabulary_coherence['numeric_transformations'][numeric_transformation] = (1-alpha) * current + alpha * expression_coherence

        except Exception as e:
            print(f"Error updating vocabulary coherence: {e}")

    def _semantic_find_related_terms(self, query, vocabulary_list, top_n=3):
        """
        Find semantically related terms from vocabulary using the semantic model

        Args:
            query: Query term to find related terms for
            vocabulary_list: List of vocabulary terms to search
            top_n: Number of top terms to return

        Returns:
            List of related terms
        """
        if not self.semantic_model or not self.semantic_embeddings:
            return random.sample(vocabulary_list, min(top_n, len(vocabulary_list)))

        try:
            # Get query embedding
            with torch.no_grad():
                query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)[0]

            # Calculate similarity with vocabulary terms
            similarities = []
            for term in vocabulary_list:
                if term in self.semantic_embeddings:
                    term_embedding = self.semantic_embeddings[term]
                    similarity = torch.nn.functional.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        term_embedding.unsqueeze(0)
                    ).item()
                    similarities.append((term, similarity))

            # Sort by similarity and return top_n
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [term for term, _ in similarities[:top_n]]

        except Exception as e:
            print(f"Error finding related terms: {e}")
            return random.sample(vocabulary_list, min(top_n, len(vocabulary_list)))

    def _expand_vocabulary_from_semantic_model(self, category, seed_term):
        """
        Use the semantic model to expand vocabulary with related terms

        Args:
            category: Vocabulary category to expand ('descriptors', 'relations', etc.)
            seed_term: Seed term to find related concepts for
        """
        if not self.semantic_model:
            return []

        try:
            # Query the model to find related concepts using find_related_concepts function
            from semantic_trainer import find_related_concepts
            results = find_related_concepts(seed_term, top_n=5)

            # Extract candidate terms
            candidates = []
            for _, sentence, similarity in results:
                # Extract potential terms based on category
                if category == 'descriptors':
                    # Look for capitalized terms likely to be descriptors
                    words = [w for w in sentence.split() if w[0].isupper() and len(w) > 3]
                    candidates.extend(words)
                elif category == 'relations':
                    # Look for verb phrases
                    if ' with ' in sentence or ' within ' in sentence or ' through ' in sentence:
                        for phrase in [' with ', ' within ', ' through ', ' into ', ' from ', ' upon ']:
                            if phrase in sentence:
                                idx = sentence.find(phrase)
                                if idx > 0 and idx + len(phrase) < len(sentence):
                                    # Get word before the phrase and word after
                                    before = sentence[:idx].split()[-1]
                                    after = sentence[idx+len(phrase):].split()[0]
                                    if before and after:
                                        candidates.append(f"{before}{phrase}{after}")
                elif category == 'numeric_modifiers' or category == 'numeric_transformations':
                    # Extract terms related to quantities and numeric relationships
                    numeric_indicators = ['increase', 'decrease', 'multiply', 'divide', 'scale',
                                         'threshold', 'critical', 'exponential', 'logarithmic',
                                         'oscillate', 'accelerate', 'decelerate', 'quantize']

                    # Look for words related to numeric concepts
                    for word in sentence.split():
                        if any(indicator in word.lower() for indicator in numeric_indicators) and len(word) > 3:
                            candidates.append(word)
                else:
                    # For other categories, look for relevant n-grams that aren't too long
                    words = sentence.split()
                    for i in range(len(words)-1):
                        if 3 < len(words[i]) + len(words[i+1]) < 15:
                            candidates.append(f"{words[i]} {words[i+1]}")

            # Return unique candidates
            return list(set(candidates))

        except Exception as e:
            print(f"Error expanding vocabulary from semantic model: {e}")
            return []

    def _background_vocabulary_update(self):
        """
        Background thread for updating vocabulary based on semantic model
        """
        while True:
            try:
                # Sleep to prevent excessive CPU usage
                time.sleep(60)  # Check every minute

                if not self.dynamic_vocabulary_enabled or not self.semantic_model:
                    continue

                # Update vocabulary if we have enough data
                if len(self.expression_history) > 10:
                    # Select random category to expand
                    category = random.choice(['descriptors', 'relations', 'concepts',
                                            'modifiers', 'secondary', 'numeric_modifiers',
                                            'numeric_transformations'])

                    # Select seed term with high coherence
                    if category == 'descriptors':
                        seed_terms = self.dynamic_state_descriptors
                    elif category == 'relations':
                        seed_terms = self.dynamic_relations
                    elif category == 'concepts':
                        seed_terms = self.dynamic_surplus_concepts
                    elif category == 'modifiers':
                        seed_terms = self.dynamic_modifiers
                    elif category == 'secondary':
                        seed_terms = self.dynamic_secondary_concepts
                    elif category == 'numeric_modifiers':
                        seed_terms = self.dynamic_numeric_modifiers
                    else:  # numeric_transformations
                        seed_terms = self.dynamic_numeric_transformations

                    # Select seed with higher probability for high-coherence terms
                    seed_term = random.choice(seed_terms)

                    # Find new candidate terms
                    new_terms = self._expand_vocabulary_from_semantic_model(category, seed_term)

                    # Add at most one new term to avoid rapid vocabulary changes
                    if new_terms:
                        new_term = random.choice(new_terms)

                        # Add to appropriate vocabulary with proper coherence tracking
                        if category == 'descriptors' and new_term not in self.dynamic_state_descriptors:
                            self.dynamic_state_descriptors.append(new_term)
                            self.vocabulary_coherence['descriptors'][new_term] = 0.5
                            print(f"Added new descriptor: {new_term}")
                        elif category == 'relations' and new_term not in self.dynamic_relations:
                            self.dynamic_relations.append(new_term)
                            self.vocabulary_coherence['relations'][new_term] = 0.5
                            print(f"Added new relation: {new_term}")
                        elif category == 'concepts' and new_term not in self.dynamic_surplus_concepts:
                            self.dynamic_surplus_concepts.append(new_term)
                            self.vocabulary_coherence['concepts'][new_term] = 0.5
                            print(f"Added new concept: {new_term}")
                        elif category == 'modifiers' and new_term not in self.dynamic_modifiers:
                            self.dynamic_modifiers.append(new_term)
                            self.vocabulary_coherence['modifiers'][new_term] = 0.5
                            print(f"Added new modifier: {new_term}")
                        elif category == 'secondary' and new_term not in self.dynamic_secondary_concepts:
                            self.dynamic_secondary_concepts.append(new_term)
                            self.vocabulary_coherence['secondary'][new_term] = 0.5
                            print(f"Added new secondary concept: {new_term}")
                        elif category == 'numeric_modifiers' and new_term not in self.dynamic_numeric_modifiers:
                            self.dynamic_numeric_modifiers.append(new_term)
                            self.vocabulary_coherence['numeric_modifiers'][new_term] = 0.5
                            print(f"Added new numeric modifier: {new_term}")
                        elif category == 'numeric_transformations' and new_term not in self.dynamic_numeric_transformations:
                            self.dynamic_numeric_transformations.append(new_term)
                            self.vocabulary_coherence['numeric_transformations'][new_term] = 0.5
                            print(f"Added new numeric transformation: {new_term}")

                    # Save updated coherence and embeddings
                    if new_terms:
                        self._precompute_vocabulary_embeddings()
                        self.save_semantic_cache()

            except Exception as e:
                print(f"Error in vocabulary update thread: {e}")
                time.sleep(300)  # Sleep longer after error

    def _update_transition_statistics(self, current_descriptor: str):
        """
        Update the transition matrix for pattern analysis.

        Args:
            current_descriptor: The descriptor used in the current expression
        """
        try:
            if self.last_descriptor is not None and current_descriptor in self.descriptor_index:
                prev_idx = self.descriptor_index.get(self.last_descriptor)
                curr_idx = self.descriptor_index.get(current_descriptor)

                if prev_idx is not None and curr_idx is not None:
                    # Increment transition count
                    self.transition_counts[prev_idx, curr_idx] += 1

                    # Update transition probability
                    row_sum = np.sum(self.transition_counts[prev_idx, :])
                    if row_sum > 0:
                        self.transition_matrix[prev_idx, :] = self.transition_counts[prev_idx, :] / row_sum

            # Update last descriptor
            self.last_descriptor = current_descriptor
        except Exception as e:
            print(f"Error updating transition statistics: {e}")

    def _track_numeric_trends(self, metrics: Dict[str, float]):
        """
        Track trends in numeric values across expressions.

        Args:
            metrics: Dictionary of current metrics
        """
        try:
            # Store the full metrics object
            self.numeric_memory.append(metrics)

            # Calculate trends for each numeric value
            if len(self.numeric_memory) >= 3:
                for key in metrics:
                    if key not in self.numeric_trends:
                        self.numeric_trends[key] = {
                            'increasing': False,
                            'decreasing': False,
                            'oscillating': False,
                            'stable': True,
                            'rate_of_change': 0.0,
                            'acceleration': 0.0
                        }

                    # Get recent values for this metric
                    recent_values = [entry.get(key, 0.0) for entry in self.numeric_memory if key in entry]
                    if len(recent_values) >= 3:
                        # Calculate first and second derivatives
                        first_diff = np.diff(recent_values[-3:])
                        rate_of_change = np.mean(first_diff)

                        # Calculate acceleration if we have enough points
                        if len(first_diff) >= 2:
                            acceleration = np.diff(first_diff)[0]
                        else:
                            acceleration = 0.0

                        # Update trend information
                        self.numeric_trends[key] = {
                            'increasing': rate_of_change > 0.01,
                            'decreasing': rate_of_change < -0.01,
                            'oscillating': (first_diff[0] * first_diff[-1] < 0) if len(first_diff) >= 2 else False,
                            'stable': abs(rate_of_change) < 0.01,
                            'rate_of_change': rate_of_change,
                            'acceleration': acceleration
                        }
        except Exception as e:
            print(f"Error tracking numeric trends: {e}")

    def _calculate_weights(self,
                          surplus: float,
                          distinction: float,
                          coherence: float,
                          entropy: Optional[float] = None,
                          dimensionality: Optional[int] = None,
                          numeric_metrics: Optional[Dict[str, float]] = None) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Calculate vocabulary selection weights based on current metrics,
        now with enhanced numeric influence.

        Args:
            surplus: Current cognitive surplus level
            distinction: Current distinction level
            coherence: Current phase coherence
            entropy: Optional entropy metric
            dimensionality: Optional detected dimensionality
            numeric_metrics: Optional additional numeric metrics

        Returns:
            Tuple of weights for descriptors, relations, concepts, numeric_modifiers, and numeric_transformations
        """
        try:
            # Choose which vocabulary to use (original or dynamic)
            descriptors = self.dynamic_state_descriptors if self.dynamic_vocabulary_enabled else self.state_descriptors
            relations = self.dynamic_relations if self.dynamic_vocabulary_enabled else self.relations
            concepts = self.dynamic_surplus_concepts if self.dynamic_vocabulary_enabled else self.surplus_concepts
            numeric_modifiers = self.dynamic_numeric_modifiers if self.dynamic_vocabulary_enabled else self.numeric_modifiers
            numeric_transformations = self.dynamic_numeric_transformations if self.dynamic_vocabulary_enabled else self.numeric_transformations

            # Initialize weights
            descriptor_weights = np.ones(len(descriptors)) / len(descriptors)
            relation_weights = np.ones(len(relations)) / len(relations)
            concept_weights = np.ones(len(concepts)) / len(concepts)
            n_modifier_weights = np.ones(len(numeric_modifiers)) / len(numeric_modifiers)
            n_transform_weights = np.ones(len(numeric_transformations)) / len(numeric_transformations)

            # Track numeric trends if metrics provided
            if numeric_metrics:
                self._track_numeric_trends(numeric_metrics)

            # Apply coherence-based weighting if semantic model is available
            if self.semantic_model and self.dynamic_vocabulary_enabled:
                # Get coherence scores for each vocabulary type
                descriptor_coherence = [self.vocabulary_coherence['descriptors'].get(term, 0.5) for term in descriptors]
                relation_coherence = [self.vocabulary_coherence['relations'].get(term, 0.5) for term in relations]
                concept_coherence = [self.vocabulary_coherence['concepts'].get(term, 0.5) for term in concepts]
                n_modifier_coherence = [self.vocabulary_coherence['numeric_modifiers'].get(term, 0.5) for term in numeric_modifiers]
                n_transform_coherence = [self.vocabulary_coherence['numeric_transformations'].get(term, 0.5) for term in numeric_transformations]

                # Convert to numpy arrays
                descriptor_coherence = np.array(descriptor_coherence)
                relation_coherence = np.array(relation_coherence)
                concept_coherence = np.array(concept_coherence)
                n_modifier_coherence = np.array(n_modifier_coherence)
                n_transform_coherence = np.array(n_transform_coherence)

                # Apply softmax to get weights
                def softmax(x, temperature=1.0):
                    exp_x = np.exp((x - np.mean(x)) / temperature)
                    return exp_x / np.sum(exp_x)

                # Higher temperature (>1.0) makes distribution more uniform
                # Lower temperature (<1.0) makes distribution more peaked
                temp = 2.0  # Start with more exploration

                # Reduce temperature as we gather more data for more focused selection
                if len(self.expression_history) > 50:
                    temp = 1.0
                if len(self.expression_history) > 100:
                    temp = 0.5

                # Calculate weights using softmax
                descriptor_weights = softmax(descriptor_coherence, temp)
                relation_weights = softmax(relation_coherence, temp)
                concept_weights = softmax(concept_coherence, temp)
                n_modifier_weights = softmax(n_modifier_coherence, temp)
                n_transform_weights = softmax(n_transform_coherence, temp)

            # Apply numeric trends-based adjustments if available
            if numeric_metrics and self.numeric_trends:
                # Find trends that are significant
                significant_trends = {}
                for key, trend in self.numeric_trends.items():
                    if trend['increasing'] or trend['decreasing'] or trend['oscillating']:
                        significant_trends[key] = trend

                # Adjust weights for numeric modifiers based on trends
                if significant_trends:
                    for i, modifier in enumerate(numeric_modifiers):
                        # Boost weight for relevant modifiers
                        if modifier == "increasing" and any(t['increasing'] for t in significant_trends.values()):
                            n_modifier_weights[i] *= 2.0
                        elif modifier == "decreasing" and any(t['decreasing'] for t in significant_trends.values()):
                            n_modifier_weights[i] *= 2.0
                        elif modifier == "oscillating" and any(t['oscillating'] for t in significant_trends.values()):
                            n_modifier_weights[i] *= 2.0
                        elif modifier == "accelerating" and any(t['acceleration'] > 0.05 for t in significant_trends.values()):
                            n_modifier_weights[i] *= 2.0
                        elif modifier == "decelerating" and any(t['acceleration'] < -0.05 for t in significant_trends.values()):
                            n_modifier_weights[i] *= 2.0

                    # Normalize numeric modifier weights
                    if np.sum(n_modifier_weights) > 0:
                        n_modifier_weights = n_modifier_weights / np.sum(n_modifier_weights)

            # Apply standard metric-based adjustments
            if coherence > self.coherence_thresholds['high']:
                # High coherence: favor structured, aligned, stabilized expressions
                # Apply weights to the indices that exist in both original and dynamic vocabulary
                for i, desc in enumerate(descriptors):
                    if desc in ["Equilibrium", "Convergence", "Coherence", "Integration"]:
                        descriptor_weights[i] *= 2.0

                for i, rel in enumerate(relations):
                    if rel in ["aligns with", "resonates within", "converges upon", "stabilizes within"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["stability", "coherence", "symmetry"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor stability and harmony
                for i, modifier in enumerate(numeric_modifiers):
                    if modifier in ["harmonic", "resonant", "continuous", "quantized"]:
                        n_modifier_weights[i] *= 2.0

                for i, transform in enumerate(numeric_transformations):
                    if transform in ["normalized", "bounded", "integral"]:
                        n_transform_weights[i] *= 2.0

            elif coherence < self.coherence_thresholds['low']:
                # Low coherence: favor flux, entropy, dissolution
                for i, desc in enumerate(descriptors):
                    if desc in ["Flux", "Divergence", "Bifurcation"]:
                        descriptor_weights[i] *= 2.0

                for i, rel in enumerate(relations):
                    if rel in ["dissolves across", "differentiates from"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["entropy", "complexity"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor instability and change
                for i, modifier in enumerate(numeric_modifiers):
                    if modifier in ["oscillating", "chaotic", "threshold", "critical"]:
                        n_modifier_weights[i] *= 2.0

                for i, transform in enumerate(numeric_transformations):
                    if transform in ["unbounded", "exponential", "amplified"]:
                        n_transform_weights[i] *= 2.0

            # Adjust based on distinction level
            if distinction > self.distinction_thresholds['high']:
                # Favor distinction and emergence concepts
                for i, desc in enumerate(descriptors):
                    if desc in ["Distinction", "Recursion"]:
                        descriptor_weights[i] *= 2.0

                for i, rel in enumerate(relations):
                    if rel in ["emerges through", "differentiates from"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["emergence", "distinction"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor discrete and quantized
                for i, modifier in enumerate(numeric_modifiers):
                    if modifier in ["discrete", "quantized", "threshold"]:
                        n_modifier_weights[i] *= 2.0

            elif distinction < self.distinction_thresholds['low']:
                # Favor flux and entropy
                for i, desc in enumerate(descriptors):
                    if desc in ["Flux", "Entanglement"]:
                        descriptor_weights[i] *= 2.0

                for i, rel in enumerate(relations):
                    if rel in ["dissolves across", "contracts into"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["entropy", "complexity"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor continuous and flowing
                for i, modifier in enumerate(numeric_modifiers):
                    if modifier in ["continuous", "increasing", "decreasing"]:
                        n_modifier_weights[i] *= 2.0

            # Adjust for surplus level
            if surplus > self.surplus_thresholds['high']:
                # High surplus: favor differentiation and expansion
                for i, rel in enumerate(relations):
                    if rel in ["differentiates from", "extends beyond", "transcends"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["distinction", "recursion", "dimensionality"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor amplification and exponential
                for i, transform in enumerate(numeric_transformations):
                    if transform in ["amplified", "exponential", "multiplied"]:
                        n_transform_weights[i] *= 2.0

            elif surplus < self.surplus_thresholds['low']:
                # Low surplus: favor contraction and stability
                for i, rel in enumerate(relations):
                    if rel in ["contracts into", "stabilizes within"]:
                        relation_weights[i] *= 2.0

                for i, concept in enumerate(concepts):
                    if concept in ["stability", "feedback"]:
                        concept_weights[i] *= 2.0

                # For numeric components, favor attenuation and stability
                for i, transform in enumerate(numeric_transformations):
                    if transform in ["attenuated", "divided", "normalized"]:
                        n_transform_weights[i] *= 2.0

            # Adjust for entropy if provided
            if entropy is not None:
                if entropy > 0.7:  # High entropy
                    for i, desc in enumerate(descriptors):
                        if desc in ["Flux", "Divergence"]:
                            descriptor_weights[i] *= 2.0

                    for i, concept in enumerate(concepts):
                        if concept in ["entropy", "complexity"]:
                            concept_weights[i] *= 2.0

                    # For numeric components, favor chaos and randomness
                    for i, modifier in enumerate(numeric_modifiers):
                        if modifier in ["chaotic", "oscillating"]:
                            n_modifier_weights[i] *= 2.0

                elif entropy < 0.3:  # Low entropy
                    for i, desc in enumerate(descriptors):
                        if desc in ["Equilibrium", "Coherence"]:
                            descriptor_weights[i] *= 2.0

                    for i, concept in enumerate(concepts):
                        if concept in ["stability", "symmetry"]:
                            concept_weights[i] *= 2.0

                    # For numeric components, favor order and stability
                    for i, modifier in enumerate(numeric_modifiers):
                        if modifier in ["harmonic", "resonant"]:
                            n_modifier_weights[i] *= 2.0

            # Adjust for dimensionality if provided
            if dimensionality is not None:
                if dimensionality > 3:  # Higher dimensions
                    for i, desc in enumerate(descriptors):
                        if desc in ["Integration", "Recursion"]:
                            descriptor_weights[i] *= 2.0

                    for i, rel in enumerate(relations):
                        if rel in ["transcends", "emerges through"]:
                            relation_weights[i] *= 2.0

                    for i, concept in enumerate(concepts):
                        if concept in ["dimensionality", "emergence"]:
                            concept_weights[i] *= 2.0

                    # For numeric transformations, favor higher-order operations
                    for i, transform in enumerate(numeric_transformations):
                        if transform in ["exponential", "logarithmic"]:
                            n_transform_weights[i] *= 2.0

            # Normalize weights
            descriptor_weights = descriptor_weights / np.sum(descriptor_weights)
            relation_weights = relation_weights / np.sum(relation_weights)
            concept_weights = concept_weights / np.sum(concept_weights)
            n_modifier_weights = n_modifier_weights / np.sum(n_modifier_weights)
            n_transform_weights = n_transform_weights / np.sum(n_transform_weights)

            return descriptor_weights.tolist(), relation_weights.tolist(), concept_weights.tolist(), n_modifier_weights.tolist(), n_transform_weights.tolist()

        except Exception as e:
            print(f"Error calculating expression weights: {e}")
            # Return uniform weights as fallback
            uniform_desc = [1.0/len(descriptors)] * len(descriptors)
            uniform_rel = [1.0/len(relations)] * len(relations)
            uniform_con = [1.0/len(concepts)] * len(concepts)
            uniform_n_mod = [1.0/len(numeric_modifiers)] * len(numeric_modifiers)
            uniform_n_trans = [1.0/len(numeric_transformations)] * len(numeric_transformations)
            return uniform_desc, uniform_rel, uniform_con, uniform_n_mod, uniform_n_trans

    def _generate_expression_components(self,
                                       descriptor_weights: List[float],
                                       relation_weights: List[float],
                                       concept_weights: List[float],
                                       numeric_modifier_weights: List[float],
                                       numeric_transform_weights: List[float],
                                       metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Generate components for a symbolic expression based on weighted vocabularies.
        Uses semantic model feedback to guide selection when available.
        Now with enhanced numeric integration.

        Args:
            descriptor_weights: Weights for selecting state descriptors
            relation_weights: Weights for selecting relations
            concept_weights: Weights for selecting concepts
            numeric_modifier_weights: Weights for selecting numeric modifiers
            numeric_transform_weights: Weights for selecting numeric transformations
            metrics: Dictionary of current system metrics

        Returns:
            Dictionary of expression components
        """
        try:
            # Choose which vocabulary to use (original or dynamic)
            descriptors = self.dynamic_state_descriptors if self.dynamic_vocabulary_enabled else self.state_descriptors
            relations = self.dynamic_relations if self.dynamic_vocabulary_enabled else self.relations
            concepts = self.dynamic_surplus_concepts if self.dynamic_vocabulary_enabled else self.surplus_concepts
            modifiers = self.dynamic_modifiers if self.dynamic_vocabulary_enabled else self.modifiers
            num_modifiers = self.dynamic_numeric_modifiers if self.dynamic_vocabulary_enabled else self.numeric_modifiers
            num_transforms = self.dynamic_numeric_transformations if self.dynamic_vocabulary_enabled else self.numeric_transformations

            # Select components based on weighted probabilities
            descriptor = random.choices(descriptors, weights=descriptor_weights, k=1)[0]

            # If semantic model is available, use it to find semantically coherent combinations
            if self.semantic_model and random.random() < 0.7:  # 70% chance to use semantic guidance
                # First, try to find a relation that semantically fits with the descriptor
                related_relations = self._semantic_find_related_terms(descriptor, relations, top_n=3)
                relation = random.choice(related_relations)

                # Then find a concept that fits with the descriptor + relation combination
                desc_rel_combo = f"{descriptor} {relation}"
                related_concepts = self._semantic_find_related_terms(desc_rel_combo, concepts, top_n=3)
                concept = random.choice(related_concepts)

                # Find semantically fitting numeric components
                related_num_modifiers = self._semantic_find_related_terms(concept, num_modifiers, top_n=2)
                numeric_modifier = random.choice(related_num_modifiers)

                related_num_transforms = self._semantic_find_related_terms(concept, num_transforms, top_n=2)
                numeric_transform = random.choice(related_num_transforms)
            else:
                # Fall back to standard weighted selection
                relation = random.choices(relations, weights=relation_weights, k=1)[0]
                concept = random.choices(concepts, weights=concept_weights, k=1)[0]
                numeric_modifier = random.choices(num_modifiers, weights=numeric_modifier_weights, k=1)[0]
                numeric_transform = random.choices(num_transforms, weights=numeric_transform_weights, k=1)[0]

            # Determine if we should use modifiers based on complexity
            use_standard_modifier = random.random() < self.expression_complexity * 0.5
            use_numeric_modifier = random.random() < self.expression_complexity * self.numeric_influence
            use_numeric_transform = random.random() < self.expression_complexity * self.numeric_influence

            standard_modifier = random.choice(modifiers) if use_standard_modifier else None

            # Determine if we should include numeric values
            include_numeric_values = random.random() < self.numeric_influence

            # Format numeric values if we're including them
            numeric_value_str = ""
            if include_numeric_values and metrics:
                # Choose a key metric to highlight
                key_metrics = ['surplus', 'distinction', 'coherence', 'entropy']
                # Filter to metrics that actually exist
                available_metrics = [k for k in key_metrics if k in metrics]

                if available_metrics:
                    # Select a metric to highlight
                    selected_metric = random.choice(available_metrics)
                    value = metrics[selected_metric]

                    # Format the value with potential transformation
                    if use_numeric_transform and random.random() < 0.7:
                        # Apply transformation phrase
                        numeric_value_str = f"{selected_metric}={value:.2f} {numeric_transform}"
                    else:
                        # Simple value
                        numeric_value_str = f"{selected_metric}={value:.2f}"

                    # Apply numeric modifier if applicable
                    if use_numeric_modifier:
                        # Check if trend matches the modifier
                        if selected_metric in self.numeric_trends:
                            trend = self.numeric_trends[selected_metric]
                            # Only use modifier if it matches the trend
                            if (numeric_modifier == "increasing" and trend.get('increasing', False)) or \
                               (numeric_modifier == "decreasing" and trend.get('decreasing', False)) or \
                               (numeric_modifier == "oscillating" and trend.get('oscillating', False)):
                                numeric_value_str = f"{numeric_modifier} {numeric_value_str}"
                            else:
                                # Use generic modifier
                                numeric_value_str = f"{numeric_modifier} {numeric_value_str}"
                        else:
                            # No trend data, just use the modifier
                            numeric_value_str = f"{numeric_modifier} {numeric_value_str}"

            # Special case for extreme states
            coherence = metrics.get('coherence', 0.5)
            distinction = metrics.get('distinction', 0.5)

            if coherence > 0.95 and distinction > 0.9:
                descriptor = "Coherent Distinction"
                relation = "stabilizes within"
                concept = "emergent ontology"
                standard_modifier = "systematically"
                # Include dimensionality if available
                if 'dimensionality' in metrics:
                    numeric_value_str = f"dim={metrics['dimensionality']}"
            elif coherence < 0.1 and distinction < 0.1:
                descriptor = "Entropic Flux"
                relation = "dissolves across"
                concept = "undifferentiated phase space"
                standard_modifier = "chaotically"
                # Include entropy if available
                if 'entropy' in metrics:
                    numeric_value_str = f"entropy={metrics['entropy']:.2f} amplified"

            # Update transition statistics for pattern analysis
            self._update_transition_statistics(descriptor)

            # Create component dictionary
            components = {
                'descriptor': descriptor,
                'relation': relation,
                'concept': concept,
                'modifier': standard_modifier,
                'numeric_modifier': numeric_modifier if use_numeric_modifier else None,
                'numeric_transformation': numeric_transform if use_numeric_transform else None,
                'numeric_value_str': numeric_value_str if numeric_value_str else None
            }

            return components

        except Exception as e:
            print(f"Error generating expression components: {e}")
            return {}

    def generate_symbolic_expression(self,
                                surplus: float,
                                distinction: float,
                                coherence: float,
                                entropy: Optional[float] = None,
                                dimensionality: Optional[int] = None,
                                additional_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Generates a symbolic expression based on the system's current metrics.
        Now enhanced with numeric value integration and semantic model guidance.

        Args:
            surplus: Current cognitive surplus level
            distinction: Current distinction level
            coherence: Current phase coherence
            entropy: Optional entropy metric
            dimensionality: Optional detected dimensionality
            additional_metrics: Optional additional metrics to incorporate

        Returns:
            A symbolic expression representing the current state
        """
        try:
            # Ensure inputs are proper floats for stability
            surplus = float(np.clip(surplus, 0.1, 10.0))
            distinction = float(np.clip(distinction, 0.0, 1.0))
            coherence = float(np.clip(coherence, 0.0, 1.0))
            if entropy is not None:
                entropy = float(np.clip(entropy, 0.0, 1.0))

            # Prepare metrics for component generation
            metrics = {
                'surplus': surplus,
                'distinction': distinction,
                'coherence': coherence,
                'entropy': entropy,
                'dimensionality': dimensionality,
                'time_elapsed': time.time() - self.start_time
            }

            # Add any additional metrics
            if additional_metrics:
                metrics.update(additional_metrics)

            # Calculate vocabulary selection weights with numeric components
            descriptor_weights, relation_weights, concept_weights, numeric_modifier_weights, numeric_transform_weights = self._calculate_weights(
                surplus, distinction, coherence, entropy, dimensionality, metrics
            )

            # Generate expression components
            components = self._generate_expression_components(
                descriptor_weights, relation_weights, concept_weights,
                numeric_modifier_weights, numeric_transform_weights, metrics
            )

            # Adapt expression complexity based on system metrics
            self.expression_complexity = min(2.0, 0.5 + 0.5 * coherence + 0.3 * distinction + 0.2 * (surplus / 10.0))

            # Determine whether to use secondary concepts
            use_secondary = random.random() < self.expression_complexity * 0.3
            secondary = None

            if use_secondary:
                # Choose which secondary concepts to use
                sec_concepts = self.dynamic_secondary_concepts if self.dynamic_vocabulary_enabled else self.secondary_concepts

                if self.semantic_model and random.random() < 0.7:
                    # Use semantic model to find related secondary concept
                    related_secondary = self._semantic_find_related_terms(components['concept'], sec_concepts, top_n=3)
                    secondary = random.choice(related_secondary)
                else:
                    secondary = random.choice(sec_concepts)

                # Update concept with secondary component
                components['concept'] = f"{components['concept']} within {secondary}"
                components['secondary'] = secondary

            # Assemble the expression based on components
            descriptor = components['descriptor']
            relation = components['relation']
            concept = components['concept']
            modifier = components['modifier']
            numeric_value_str = components['numeric_value_str']

            # Assemble the expression with appropriate formatting
            if modifier:
                base_expression = f"{descriptor} {modifier} {relation} {concept}"
            else:
                base_expression = f"{descriptor} {relation} {concept}"

            # Add numeric information if present
            if numeric_value_str:
                # Format numeric values in brackets for clarity
                symbolic_expression = f"{base_expression} [{numeric_value_str}]."
            else:
                symbolic_expression = f"{base_expression}."

            # Store components for external access and analysis
            self.last_expression_components = components

            # Update expression counter
            self.expression_counter += 1

            # Store in history with metadata
            expression_entry = {
                'expression': symbolic_expression,
                'components': components,
                'metrics': metrics.copy(),
                'timestamp': time.time(),
                'complexity': self.expression_complexity
            }

            self.expression_history.append(expression_entry)

            # Update pattern history
            pattern_entry = {
                'descriptor': descriptor,
                'relation': relation,
                'concept': concept,
                'numeric_info': numeric_value_str if numeric_value_str else None
            }
            self.pattern_history.append(pattern_entry)

            # Update frequency analysis
            self._update_frequency_analysis(descriptor, relation, concept)

            # Update semantic vocabulary coherence if model is available
            if self.semantic_model and self.expression_counter % self.semantic_update_frequency == 0:
                self._update_vocabulary_coherence(components)

                # Periodically save cache
                if self.expression_counter % (self.semantic_update_frequency * 5) == 0:
                    self.save_semantic_cache()

            # Apply semantic refinement if coherence is low
            if self.semantic_model:
                semantic_coherence = self._calculate_semantic_coherence(symbolic_expression)
                if semantic_coherence < self.semantic_coherence_threshold:
                    refined_expression = self._semantic_refine_expression(components, semantic_coherence)
                    if refined_expression:
                        symbolic_expression = refined_expression
                        # Update the expression in history
                        expression_entry['expression'] = symbolic_expression
                        expression_entry['refined'] = True
                        self.expression_history[-1] = expression_entry

            return symbolic_expression

        except Exception as e:
            print(f"Error generating symbolic expression: {e}")
            return "Flux aligns with stability."  # Safe fallback

    def handle_post_emergence(self,
                     surplus: float,
                     distinction: float,
                     coherence: float,
                     dimensionality: Optional[int] = None,
                     entropy: Optional[float] = None) -> str:
        """
        Triggers symbolic output generation after dimensional emergence is detected.
        Records emergence event and generates an appropriate symbolic expression.
        Enhanced with semantic model integration and numeric value incorporation.

        Args:
            surplus: Current cognitive surplus level
            distinction: Current distinction level
            coherence: Current phase coherence
            dimensionality: Optional detected dimensionality
            entropy: Optional entropy metric

        Returns:
            A symbolic expression representing the emergent state
        """
        try:
            # Add randomness to prevent identical outputs
            noise_factor = random.random() * 0.1

            # Record emergence event with timestamp and more detailed metrics
            emergence_event = {
                'metrics': {
                    'surplus': float(surplus) + noise_factor,
                    'distinction': float(distinction) + noise_factor,
                    'coherence': float(coherence) + noise_factor,
                    'dimensionality': dimensionality,
                    'entropy': entropy + noise_factor if entropy is not None else None,
                    'timestamp': time.time(),
                    'elapsed_time': time.time() - self.start_time
                },
                'event_id': len(self.emergence_events)
            }

            self.emergence_events.append(emergence_event)

            # Increase expression complexity for emergence events
            self.expression_complexity = min(2.0, self.expression_complexity * 1.5)

            # Generate expression with varied metrics to ensure diversity
            varied_surplus = surplus * (1.0 + (random.random() - 0.5) * 0.2)  # +/- 10%
            varied_distinction = distinction * (1.0 + (random.random() - 0.5) * 0.2)  # +/- 10%
            varied_coherence = coherence * (1.0 + (random.random() - 0.5) * 0.2)  # +/- 10%
            varied_entropy = entropy * (1.0 + (random.random() - 0.5) * 0.2) if entropy is not None else None

            # Always include dimensionality for emergence expressions
            additional_metrics = {
                'emergence': True,
                'emergence_id': len(self.emergence_events)
            }

            # Include dimensional information directly in the expression
            if dimensionality is not None:
                additional_metrics['dim_transition'] = f"{dimensionality-1 if dimensionality > 1 else 1}→{dimensionality}"
                additional_metrics['dimensional_coherence'] = coherence * distinction  # Combined coherence in higher dimension

            # Generate the base expression
            expression = self.generate_symbolic_expression(
                varied_surplus, varied_distinction, varied_coherence,
                entropy=varied_entropy, dimensionality=dimensionality,
                additional_metrics=additional_metrics
            )

            # For emergence events, use semantic model to generate a more insightful secondary expression
            if self.semantic_model and len(self.emergence_events) > 1:
                try:
                    # Get patterns from emergence analysis
                    patterns = self.analyze_emergence_patterns()

                    # Generate a semantically coherent follow-up expression
                    if random.random() < 0.7 and patterns.get('dominant_patterns'):
                        dominant = patterns['dominant_patterns']

                        # Create base follow-up template with dimensional information
                        template = f"Dimensional shift to {dimensionality}D reveals {dominant.get('descriptor', 'Emergence')} within {dominant.get('concept', 'complexity')}."

                        # Get semantically related concepts to create follow-up
                        from semantic_trainer import find_related_concepts
                        query = f"dimensional shift {dominant.get('descriptor', '')} {dominant.get('concept', '')}"
                        results = find_related_concepts(query, top_n=3)

                        if results:
                            # Use the most relevant sentence from results
                            follow_up = results[0][1]

                            # Add numeric information to the follow-up
                            if entropy is not None:
                                follow_up += f" [entropy={entropy:.2f}, dimensionality={dimensionality}]"

                            # Combine expressions
                            expression = f"{expression} {follow_up}"
                except Exception as e:
                    print(f"Error generating semantic follow-up: {e}")

            # Standard follow-up generation if semantic model not available or failed
            elif len(self.emergence_events) > 1:
                # Analyze emergence patterns
                patterns = self.analyze_emergence_patterns()

                # Use the pattern analysis to generate a deeper insight with randomization
                if random.random() < 0.7 and patterns.get('dominant_patterns'):
                    dominant = patterns['dominant_patterns']

                    # Create varied secondary expressions
                    secondary_expressions = [
                        f"Pattern analysis indicates {dominant.get('descriptor', 'Emergence')} "
                        f"{random.choice(self.relations)} "
                        f"{dominant.get('concept', 'complexity')} "
                        f"across {dimensionality if dimensionality else 'multiple'} dimensions "
                        f"[coherence={coherence:.2f}].",

                        f"Dimensional shift to {dimensionality}D reveals {dominant.get('descriptor', 'Emergence')} "
                        f"{random.choice(self.relations)} "
                        f"{random.choice(self.surplus_concepts)} "
                        f"[distinction={distinction:.2f}].",

                        f"The {dimensionality}D structure {random.choice(self.modifiers)} "
                        f"{random.choice(self.relations)} "
                        f"{dominant.get('concept', 'ontology')} "
                        f"[entropy={entropy:.2f if entropy is not None else 0.5}].",

                        f"Analysis suggests {random.choice(self.modifiers)} {dominant.get('descriptor', 'Distinction')} "
                        f"within the emergent {dimensionality}D domain "
                        f"[surplus={surplus:.2f}, coherence={coherence:.2f}]."
                    ]

                    # Choose one secondary expression randomly
                    follow_up = random.choice(secondary_expressions)
                    expression = f"{expression} {follow_up}"

            return expression

        except Exception as e:
            print(f"Error handling post-emergence: {e}")
            return self.generate_symbolic_expression(surplus, distinction, coherence)

    def _update_frequency_analysis(self, descriptor: str, relation: str, concept: str, numeric_info: str = None):
        """
        Update frequency analysis of expression components, including numeric components.

        Args:
            descriptor: The descriptor used
            relation: The relation used
            concept: The concept used
            numeric_info: Optional numeric information string
        """
        try:
            if 'descriptors' not in self.frequency_analysis:
                self.frequency_analysis = {
                    'descriptors': {},
                    'relations': {},
                    'concepts': {},
                    'numeric_modifiers': {},
                    'numeric_transformations': {},
                    'selected_metrics': {}
                }

            # Update descriptor frequency
            self.frequency_analysis['descriptors'][descriptor] = (
                self.frequency_analysis['descriptors'].get(descriptor, 0) + 1
            )

            # Update relation frequency
            self.frequency_analysis['relations'][relation] = (
                self.frequency_analysis['relations'].get(relation, 0) + 1
            )

            # Update concept frequency
            self.frequency_analysis['concepts'][concept] = (
                self.frequency_analysis['concepts'].get(concept, 0) + 1
            )

            # Update numeric component frequencies if available
            if numeric_info:
                # Extract modifiers and metrics from numeric info
                parts = numeric_info.split()

                # Look for modifiers like "increasing", "decreasing", etc.
                for modifier in self.numeric_modifiers:
                    if modifier in numeric_info:
                        self.frequency_analysis['numeric_modifiers'][modifier] = (
                            self.frequency_analysis['numeric_modifiers'].get(modifier, 0) + 1
                        )

                # Look for transformations like "amplified", "normalized", etc.
                for transform in self.numeric_transformations:
                    if transform in numeric_info:
                        self.frequency_analysis['numeric_transformations'][transform] = (
                            self.frequency_analysis['numeric_transformations'].get(transform, 0) + 1
                        )

                # Extract the selected metric (e.g., entropy=0.5, surplus=2.3)
                if "=" in numeric_info:
                    metric_name = numeric_info.split("=")[0].strip()
                    self.frequency_analysis['selected_metrics'][metric_name] = (
                        self.frequency_analysis['selected_metrics'].get(metric_name, 0) + 1
                    )

        except Exception as e:
            print(f"Error updating frequency analysis: {e}")

    def analyze_emergence_patterns(self) -> Dict[str, Any]:
        """
        Analyzes patterns in emergence events and generated expressions.
        Returns statistics and patterns detected in the symbolic outputs.
        Enhanced with semantic clustering when model is available and
        now including numeric component analysis.

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if not self.emergence_events or not self.expression_history:
                return {'patterns': 'Insufficient data for pattern analysis'}

            # Extract metrics from history
            coherence_values = [e['metrics'].get('coherence', 0.5) for e in self.emergence_events]
            distinction_values = [e['metrics'].get('distinction', 0.5) for e in self.emergence_events]

            # Only use recent expressions for pattern analysis if we have many
            expressions_to_analyze = self.expression_history
            if len(expressions_to_analyze) > 20:
                expressions_to_analyze = expressions_to_analyze[-20:]

            # Calculate emergence stability
            coherence_stability = float(np.std(coherence_values)) if len(coherence_values) > 1 else 0
            distinction_stability = float(np.std(distinction_values)) if len(distinction_values) > 1 else 0

            # More advanced pattern analysis with the frequency analysis
            if hasattr(self, 'frequency_analysis') and self.frequency_analysis:
                # Find most common components
                descriptor_counts = self.frequency_analysis.get('descriptors', {})
                relation_counts = self.frequency_analysis.get('relations', {})
                concept_counts = self.frequency_analysis.get('concepts', {})
                n_modifier_counts = self.frequency_analysis.get('numeric_modifiers', {})
                n_transform_counts = self.frequency_analysis.get('numeric_transformations', {})
                metric_counts = self.frequency_analysis.get('selected_metrics', {})

                # Find dominant patterns
                dominant_descriptor = max(descriptor_counts.items(), key=lambda x: x[1])[0] if descriptor_counts else None
                dominant_relation = max(relation_counts.items(), key=lambda x: x[1])[0] if relation_counts else None
                dominant_concept = max(concept_counts.items(), key=lambda x: x[1])[0] if concept_counts else None
                dominant_n_modifier = max(n_modifier_counts.items(), key=lambda x: x[1])[0] if n_modifier_counts else None
                dominant_n_transform = max(n_transform_counts.items(), key=lambda x: x[1])[0] if n_transform_counts else None
                dominant_metric = max(metric_counts.items(), key=lambda x: x[1])[0] if metric_counts else None

                # Calculate component diversity (normalized entropy)
                def calculate_diversity(counts):
                    if not counts:
                        return 0.0
                    total = sum(counts.values())
                    probabilities = [count/total for count in counts.values()]
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    max_entropy = np.log2(len(counts))
                    return entropy / max_entropy if max_entropy > 0 else 0.0

                descriptor_diversity = calculate_diversity(descriptor_counts)
                relation_diversity = calculate_diversity(relation_counts)
                concept_diversity = calculate_diversity(concept_counts)
                n_modifier_diversity = calculate_diversity(n_modifier_counts)
                n_transform_diversity = calculate_diversity(n_transform_counts)
                metric_diversity = calculate_diversity(metric_counts)

                # Find common sequences in the pattern history
                sequence_patterns = {}
                numeric_transitions = {}

                if len(self.pattern_history) > 3:
                    for i in range(len(self.pattern_history) - 2):
                        # Track descriptor sequences
                        seq = (
                            self.pattern_history[i].get('descriptor', ''),
                            self.pattern_history[i+1].get('descriptor', ''),
                            self.pattern_history[i+2].get('descriptor', '')
                        )
                        sequence_patterns[seq] = sequence_patterns.get(seq, 0) + 1

                        # Track numeric transitions if present
                        if ('numeric_info' in self.pattern_history[i] and
                            'numeric_info' in self.pattern_history[i+1] and
                            self.pattern_history[i]['numeric_info'] and
                            self.pattern_history[i+1]['numeric_info']):

                            num_trans = (
                                self.pattern_history[i]['numeric_info'],
                                self.pattern_history[i+1]['numeric_info']
                            )
                            numeric_transitions[num_trans] = numeric_transitions.get(num_trans, 0) + 1

                # Find most common sequence
                common_sequence = max(sequence_patterns.items(), key=lambda x: x[1])[0] if sequence_patterns else None
                common_numeric_transition = max(numeric_transitions.items(), key=lambda x: x[1])[0] if numeric_transitions else None

                # Typical expression
                typical_expression = f"{dominant_descriptor} {dominant_relation} {dominant_concept}."

                # Include numeric information if present
                if dominant_n_modifier and dominant_metric and dominant_n_transform:
                    typical_numeric = f"{dominant_n_modifier} {dominant_metric}={0.5:.2f} {dominant_n_transform}"
                    typical_expression = f"{typical_expression[:-1]} [{typical_numeric}]."

                # Calculate complexity trend
                complexity_values = [e.get('complexity', 1.0) for e in self.expression_history[-10:]]
                complexity_trend = np.mean(np.diff(complexity_values)) if len(complexity_values) > 1 else 0.0

                # Calculate transition matrix entropy (measure of pattern predictability)
                transition_entropy = 0.0
                if hasattr(self, 'transition_matrix') and isinstance(self.transition_matrix, np.ndarray):
                    for row in self.transition_matrix:
                        row_probs = row[row > 0]  # Only consider non-zero probabilities
                        if len(row_probs) > 0:
                            row_entropy = -np.sum(row_probs * np.log2(row_probs))
                            transition_entropy += row_entropy

                    transition_entropy /= max(1, np.sum(self.transition_matrix > 0))  # Normalize

                # Analyze numeric trends
                numeric_trend_analysis = {}
                if hasattr(self, 'numeric_trends') and self.numeric_trends:
                    # Count trends by type
                    trend_counts = {
                        'increasing': sum(1 for t in self.numeric_trends.values() if t.get('increasing', False)),
                        'decreasing': sum(1 for t in self.numeric_trends.values() if t.get('decreasing', False)),
                        'oscillating': sum(1 for t in self.numeric_trends.values() if t.get('oscillating', False)),
                        'stable': sum(1 for t in self.numeric_trends.values() if t.get('stable', False))
                    }

                    # Calculate average rate of change and acceleration
                    avg_rate = np.mean([t.get('rate_of_change', 0.0) for t in self.numeric_trends.values()])
                    avg_accel = np.mean([t.get('acceleration', 0.0) for t in self.numeric_trends.values()])

                    numeric_trend_analysis = {
                        'trend_counts': trend_counts,
                        'avg_rate_of_change': float(avg_rate),
                        'avg_acceleration': float(avg_accel),
                        'dominant_trend': max(trend_counts.items(), key=lambda x: x[1])[0]
                    }

                # Add semantic coherence analysis if model is available
                semantic_analysis = {}
                if self.semantic_model and len(self.expression_history) > 5:
                    try:
                        # Compute average semantic coherence of recent expressions
                        recent_expressions = [e['expression'] for e in self.expression_history[-5:]]
                        coherence_scores = [self._calculate_semantic_coherence(expr) for expr in recent_expressions]
                        semantic_analysis['recent_coherence'] = float(np.mean(coherence_scores))

                        # Compare current vocabulary coherence to initial
                        for vocab_type in ['descriptors', 'relations', 'concepts', 'numeric_modifiers', 'numeric_transformations']:
                            if vocab_type in self.vocabulary_coherence:
                                values = list(self.vocabulary_coherence[vocab_type].values())
                                semantic_analysis[f'{vocab_type}_coherence'] = float(np.mean(values))
                    except Exception as e:
                        print(f"Error in semantic analysis: {e}")

                return {
                    'emergence_count': len(self.emergence_events),
                    'expression_count': len(self.expression_history),
                    'coherence_stability': float(coherence_stability),
                    'distinction_stability': float(distinction_stability),
                    'component_diversity': {
                        'descriptor': float(descriptor_diversity),
                        'relation': float(relation_diversity),
                        'concept': float(concept_diversity),
                        'numeric_modifier': float(n_modifier_diversity),
                        'numeric_transform': float(n_transform_diversity),
                        'metric': float(metric_diversity),
                        'overall': float((descriptor_diversity + relation_diversity + concept_diversity +
                                  n_modifier_diversity + n_transform_diversity + metric_diversity) / 6)
                    },
                    'dominant_patterns': {
                        'descriptor': dominant_descriptor,
                        'relation': dominant_relation,
                        'concept': dominant_concept,
                        'numeric_modifier': dominant_n_modifier,
                        'numeric_transformation': dominant_n_transform,
                        'selected_metric': dominant_metric
                    },
                    'common_sequence': common_sequence,
                    'common_numeric_transition': common_numeric_transition,
                    'typical_expression': typical_expression,
                    'complexity_trend': float(complexity_trend),
                    'transition_entropy': float(transition_entropy),
                    'current_complexity': float(self.expression_complexity),
                    'numeric_trend_analysis': numeric_trend_analysis,
                    'semantic_analysis': semantic_analysis if self.semantic_model else {}
                }

            # Simplified analysis if frequency data isn't available
            return {
                'emergence_count': len(self.emergence_events),
                'expression_count': len(self.expression_history),
                'coherence_stability': float(coherence_stability),
                'distinction_stability': float(distinction_stability)
            }

        except Exception as e:
            print(f"Error analyzing emergence patterns: {e}")
            return {
                'error': str(e),
                'emergence_count': len(self.emergence_events),
                'expression_count': len(self.expression_history)
            }

    def get_vocabulary_status(self) -> Dict[str, Any]:
        """
        Returns the current vocabulary status, including dynamic expansions
        and semantic coherence metrics.

        Returns:
            Dictionary containing vocabulary statistics
        """
        try:
            # Count original and dynamic vocabulary
            orig_counts = {
                'descriptors': len(self.state_descriptors),
                'relations': len(self.relations),
                'concepts': len(self.surplus_concepts),
                'modifiers': len(self.modifiers),
                'secondary': len(self.secondary_concepts),
                'numeric_modifiers': len(self.numeric_modifiers),
                'numeric_transformations': len(self.numeric_transformations)
            }

            dynamic_counts = {
                'descriptors': len(self.dynamic_state_descriptors),
                'relations': len(self.dynamic_relations),
                'concepts': len(self.dynamic_surplus_concepts),
                'modifiers': len(self.dynamic_modifiers),
                'secondary': len(self.dynamic_secondary_concepts),
                'numeric_modifiers': len(self.dynamic_numeric_modifiers),
                'numeric_transformations': len(self.dynamic_numeric_transformations)
            }

            # Calculate added terms
            added_terms = {
                'descriptors': list(set(self.dynamic_state_descriptors) - set(self.state_descriptors)),
                'relations': list(set(self.dynamic_relations) - set(self.relations)),
                'concepts': list(set(self.dynamic_surplus_concepts) - set(self.surplus_concepts)),
                'modifiers': list(set(self.dynamic_modifiers) - set(self.modifiers)),
                'secondary': list(set(self.dynamic_secondary_concepts) - set(self.secondary_concepts)),
                'numeric_modifiers': list(set(self.dynamic_numeric_modifiers) - set(self.numeric_modifiers)),
                'numeric_transformations': list(set(self.dynamic_numeric_transformations) - set(self.numeric_transformations))
            }

            # Get top coherence terms if semantic model is used
            top_coherence = {}
            if self.semantic_model:
                for vocab_type in ['descriptors', 'relations', 'concepts', 'modifiers',
                                  'secondary', 'numeric_modifiers', 'numeric_transformations']:
                    if vocab_type in self.vocabulary_coherence:
                        # Sort by coherence and get top 5
                        sorted_terms = sorted(
                            self.vocabulary_coherence[vocab_type].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        top_coherence[vocab_type] = sorted_terms

            # Calculate numeric integration statistics
            numeric_stats = {}
            if hasattr(self, 'numeric_memory') and len(self.numeric_memory) > 0:
                # Calculate frequency of numeric inclusion
                expressions_with_numeric = sum(1 for e in self.expression_history
                                            if 'components' in e and e['components'].get('numeric_value_str'))

                numeric_stats = {
                    'inclusion_rate': expressions_with_numeric / max(1, len(self.expression_history)),
                    'num_memory_size': len(self.numeric_memory),
                    'trends_tracked': len(self.numeric_trends) if hasattr(self, 'numeric_trends') else 0,
                    'numeric_influence': self.numeric_influence
                }

                # Add trend summary if available
                if hasattr(self, 'numeric_trends') and self.numeric_trends:
                    trend_summary = {
                        'increasing': sum(1 for t in self.numeric_trends.values() if t.get('increasing', False)),
                        'decreasing': sum(1 for t in self.numeric_trends.values() if t.get('decreasing', False)),
                        'oscillating': sum(1 for t in self.numeric_trends.values() if t.get('oscillating', False)),
                        'stable': sum(1 for t in self.numeric_trends.values() if t.get('stable', False))
                    }
                    numeric_stats['trend_summary'] = trend_summary

                # Add frequency analysis for numeric components if available
                if 'numeric_modifiers' in self.frequency_analysis:
                    top_modifiers = sorted(
                        self.frequency_analysis['numeric_modifiers'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    numeric_stats['top_modifiers'] = top_modifiers

                if 'numeric_transformations' in self.frequency_analysis:
                    top_transforms = sorted(
                        self.frequency_analysis['numeric_transformations'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    numeric_stats['top_transforms'] = top_transforms

                if 'selected_metrics' in self.frequency_analysis:
                    top_metrics = sorted(
                        self.frequency_analysis['selected_metrics'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    numeric_stats['top_metrics'] = top_metrics

            return {
                'original_counts': orig_counts,
                'dynamic_counts': dynamic_counts,
                'added_terms': added_terms,
                'top_coherence_terms': top_coherence,
                'numeric_stats': numeric_stats,
                'dynamic_vocabulary_enabled': self.dynamic_vocabulary_enabled,
                'semantic_model_active': self.semantic_model is not None,
                'expression_complexity': float(self.expression_complexity),
                'vocabulary_sizes': {
                    'total_original': sum(orig_counts.values()),
                    'total_dynamic': sum(dynamic_counts.values()),
                    'total_added': sum(len(terms) for terms in added_terms.values())
                }
            }

        except Exception as e:
            print(f"Error getting vocabulary status: {e}")
            return {
                'error': str(e),
                'semantic_model_active': self.semantic_model is not None
            }

    def get_semantic_analysis(self, query):
        """
        Performs semantic analysis on a query using the semantic model,
        now with enhanced numeric value consideration.

        Args:
            query: Text to analyze

        Returns:
            Dictionary with semantic analysis results
        """
        if not self.semantic_model:
            return {"error": "Semantic model not available"}

        try:
            # Calculate coherence with existing expressions
            coherence = self._calculate_semantic_coherence(query)

            # Find related vocabulary terms
            related_descriptors = self._semantic_find_related_terms(query, self.dynamic_state_descriptors, top_n=3)
            related_concepts = self._semantic_find_related_terms(query, self.dynamic_surplus_concepts, top_n=3)

            # Also find related numeric modifiers and transformations
            related_num_modifiers = self._semantic_find_related_terms(query, self.dynamic_numeric_modifiers, top_n=2)
            related_num_transforms = self._semantic_find_related_terms(query, self.dynamic_numeric_transformations, top_n=2)

            # Find related historical expressions
            related_expressions = []
            if self.expression_history:
                # Get query embedding
                with torch.no_grad():
                    query_embedding = self.sentence_model.encode([query], convert_to_tensor=True)[0]

                # Compare with historical expressions
                similarities = []
                for entry in self.expression_history[-20:]:  # Last 20 expressions
                    expr = entry['expression']
                    expr_embedding = self.sentence_model.encode([expr], convert_to_tensor=True)[0]

                    similarity = torch.nn.functional.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        expr_embedding.unsqueeze(0)
                    ).item()

                    # Also include numeric information if available
                    numeric_info = None
                    if 'components' in entry and entry['components'].get('numeric_value_str'):
                        numeric_info = entry['components']['numeric_value_str']

                    similarities.append((expr, similarity, numeric_info))

                # Get top 3 similar expressions
                similarities.sort(key=lambda x: x[1], reverse=True)
                related_expressions = similarities[:3]

            # Analyze numeric relevance if query contains numeric indicators
            numeric_relevance = self._analyze_numeric_relevance(query)

            # Generate a numeric expression component if query seems to focus on numeric aspects
            suggested_numeric_component = None
            if numeric_relevance > 0.5 and self.numeric_memory and len(self.numeric_memory) > 0:
                # Get recent metrics
                recent_metrics = self.numeric_memory[-1]

                # Select a related numeric modifier and transformation
                num_modifier = related_num_modifiers[0] if related_num_modifiers else random.choice(self.numeric_modifiers)
                num_transform = related_num_transforms[0] if related_num_transforms else random.choice(self.numeric_transformations)

                # Select a metric from recent memory
                if 'entropy' in recent_metrics:
                    metric_name = 'entropy'
                    value = recent_metrics['entropy']
                elif 'coherence' in recent_metrics:
                    metric_name = 'coherence'
                    value = recent_metrics['coherence']
                elif 'distinction' in recent_metrics:
                    metric_name = 'distinction'
                    value = recent_metrics['distinction']
                else:
                    # Pick the first numeric value found
                    for k, v in recent_metrics.items():
                        if isinstance(v, (int, float)):
                            metric_name = k
                            value = v
                            break
                    else:
                        metric_name = 'value'
                        value = 0.5

                # Format the suggested numeric component
                suggested_numeric_component = f"{num_modifier} {metric_name}={value:.2f} {num_transform}"

            return {
                "query": query,
                "coherence": coherence,
                "numeric_relevance": numeric_relevance,
                "related_descriptors": related_descriptors,
                "related_concepts": related_concepts,
                "related_numeric_modifiers": related_num_modifiers,
                "related_numeric_transformations": related_num_transforms,
                "related_expressions": related_expressions,
                "suggested_numeric_component": suggested_numeric_component,
                "analysis_time": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return {"error": str(e)}

    def _analyze_numeric_relevance(self, text: str) -> float:
        """
        Analyze how numerically relevant a text query is.

        Args:
            text: The text to analyze

        Returns:
            Numeric relevance score between 0.0 and 1.0
        """
        try:
            # Define indicators of numeric focus
            numeric_indicators = [
                'number', 'value', 'quantity', 'measure', 'metric', 'parameter',
                'increase', 'decrease', 'grow', 'shrink', 'rise', 'fall',
                'trend', 'pattern', 'oscillation', 'fluctuation',
                'amplitude', 'frequency', 'period', 'phase',
                'threshold', 'critical', 'amplitude', 'scale',
                'dimension', 'coordinate', 'vector', 'matrix',
                'graph', 'chart', 'plot', 'axis', 'data',
                'statistical', 'correlation', 'distribution',
                'entropy', 'coherence', 'surplus', 'distinction'
            ]

            # Additionally check for any of our numeric vocabulary terms
            numeric_vocab = self.numeric_modifiers + self.numeric_transformations
            all_indicators = numeric_indicators + numeric_vocab

            # Count occurrences of numeric indicators
            count = 0
            text_lower = text.lower()
            for indicator in all_indicators:
                if indicator.lower() in text_lower:
                    count += 1

            # Also check for actual numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', text)
            count += len(numbers)

            # Calculate relevance score (capped at 1.0)
            return min(1.0, count / 10.0)  # Normalize (10+ indicators = 1.0)

        except Exception as e:
            print(f"Error analyzing numeric relevance: {e}")
            return 0.0  # Default to non-relevant

class SemanticRefiner:  # Class to hold the function
    def __init__(self,
                 transformer_base_name="bert-base-uncased",
                 semantic_model_name="bert-base-uncased",
                 sentence_model_name='all-MiniLM-L6-v2',
                 semantic_coherence_threshold = 0.5,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        """Initialize with a semantic model and coherence threshold."""
        self.semantic_coherence_threshold = semantic_coherence_threshold
        self.expression_history = [] #  Dummy
        self.device = device

        try:
            # Load tokenizer and models
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_base_name)
            self.transformer_base = AutoModel.from_pretrained(transformer_base_name).to(self.device)
            self.semantic_model = AutoModel.from_pretrained(semantic_model_name).to(self.device)
            self.sentence_model = SentenceTransformer(sentence_model_name).to(self.device)
            logger.info("Models loaded successfully")

        except Exception as e:
              logger.error(f"Error loading models: {e}")
              self.tokenizer = None
              self.transformer_base = None
              self.semantic_model = None
              self.sentence_model = None


    def _semantic_refine_expression(self, components: List[str], semantic_coherence: float) -> Optional[str]:
        """
        Refines a symbolic expression based on its components and semantic coherence.

        Args:
            components: List of strings representing the components of the symbolic expression.
            semantic_coherence: The semantic coherence of the expression.

        Returns:
            A refined symbolic expression as a string, or None if no refinement is needed.
        """
        try:
            if not components:
                logger.warning("No components provided for refinement.")
                return None

            if self.semantic_model is None:
                logger.warning("No semantic model available for refinement.")
                return None
            # Implement a placeholder for the semantic refinement
            logger.info(f"Attempting to refine expression with components {components} and coherence {semantic_coherence}.")

            # Example of potential actions
            if semantic_coherence < self.semantic_coherence_threshold:
                 # Dummy Refinement for testing.
                refined_components = [f"refined_{c}" for c in components]
                refined_expression = " ".join(refined_components)
                logger.info(f"Expression refined to: {refined_expression}")
                return refined_expression
            else:
                logger.info("No refinement needed as semantic coherence is above threshold")
                return None

        except Exception as e:
            logger.error(f"Error in semantic refinement: {e}")
            return None

    def _calculate_semantic_coherence(self, expression: str) -> float:
        """
        Calculate the semantic coherence of an expression using the semantic model

        Args:
            expression: The symbolic expression to evaluate

        Returns:
            Coherence score between 0-1
        """
        if not self.semantic_model:
            return 0.5  # Default if no model

        try:
            # Tokenize the expression
            inputs = self.tokenizer(expression, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512).to(self.device)

            # Get expression embedding from model
            with torch.no_grad():
                # First encode with base transformer
                base_outputs = self.transformer_base(**inputs)
                pooled_output = base_outputs.last_hidden_state[:, 0, :]
                # Then use our semantic model
                expression_embedding = self.semantic_model(pooled_output)

            # Get reference embeddings
            if self.expression_history:
                # Use historical expressions as reference
                reference_texts = [entry['expression'] for entry in self.expression_history[-10:]]

                with torch.no_grad():
                    reference_embeddings = self.sentence_model.encode(reference_texts, convert_to_tensor=True).to(self.device)

                # Calculate average cosine similarity
                cos_sims = torch.nn.functional.cosine_similarity(
                    expression_embedding.unsqueeze(0), reference_embeddings
                )

                # Return maximum similarity as coherence score
                return torch.max(cos_sims).item()
            else:
                # If no history, return 0.5 as default
                return 0.5

        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.5

    def generate_symbolic_expression(self, components:List[str]) -> str:
        """Dummy expression generator to show usage"""
        symbolic_expression = " ".join(components)

        expression_entry = {'expression': symbolic_expression, 'refined': False}
        self.expression_history.append(expression_entry)
        # Apply semantic refinement if coherence is low
        if self.semantic_model:
            semantic_coherence = self._calculate_semantic_coherence(symbolic_expression)
            if semantic_coherence < self.semantic_coherence_threshold:
                refined_expression = self._semantic_refine_expression(components, semantic_coherence)
                if refined_expression:
                    symbolic_expression = refined_expression
                    # Update the expression in history
                    expression_entry['expression'] = symbolic_expression
                    expression_entry['refined'] = True
                    self.expression_history[-1] = expression_entry

        return symbolic_expression

# Define custom model class to match the saved model architecture
class LegacySemioticModel(nn.Module):
    """
    Custom version of the SemioticExtractor that matches the model architecture
    in the checkpoint, specifically using output dimensions of 384 instead of 256.
    """
    def __init__(self, hidden_dim=384, numeric_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Create encoder placeholder (will be replaced in load_semantic_model)
        self.encoder = None

        # Text processing pathway
        self.fc1 = nn.Linear(768, hidden_dim)  # 768 is BERT's output size
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        # Numeric processing pathway
        self.numeric_enabled = numeric_dim > 0
        if self.numeric_enabled:
            self.fc1_numeric = nn.Linear(numeric_dim, hidden_dim // 2)
            # Important: Use 384 as output size to match checkpoint
            self.fc_combined = nn.Linear(hidden_dim + (hidden_dim // 2), 384)
        else:
            # Use 384 as output size to match checkpoint
            self.fc2 = nn.Linear(hidden_dim, 384)

        # Final normalization - use 384 to match checkpoint
        self.norm = nn.LayerNorm(384)

    def forward(self, input_ids, attention_mask, numeric_values=None):
        # If encoder is None, create a placeholder output
        if self.encoder is None:
            # Create a placeholder embedding of the right shape
            batch_size = input_ids.shape[0]
            pooled_output = torch.zeros((batch_size, 768), device=input_ids.device)
        else:
            # Use the encoder
            with torch.no_grad():
                outputs = self.encoder(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # Process text features
        text_features = self.activation(self.fc1(pooled_output))
        text_features = self.dropout(text_features)

        # Process and combine with numeric features if provided
        if self.numeric_enabled and numeric_values is not None:
            numeric_features = self.activation(self.fc1_numeric(numeric_values))
            combined_features = torch.cat([text_features, numeric_features], dim=1)
            output = self.fc_combined(combined_features)
        else:
            output = self.fc2(text_features) if hasattr(self, 'fc2') else text_features

        return self.norm(output)

