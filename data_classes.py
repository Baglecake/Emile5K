
"""
Data Classes for Émile-2 Simulation
-----------------------------------
Core data structures that encapsulate state and ensure type safety.
"""
import logging
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.data_classes")

# Import necessary constants
from utilities import MINIMUM_COHERENCE_FLOOR


@dataclass
class SurplusState:
    """Container for surplus state information with stability tracking."""
    values: Dict[str, float] = field(
        default_factory=lambda: {'basal': 1.0, 'cognitive': 1.0, 'predictive': 1.0, 'ontological': 1.0}
    )
    accumulation_rate: Dict[str, float] = field(
        default_factory=lambda: {'basal': 0.01, 'cognitive': 0.01, 'predictive': 0.01, 'ontological': 0.01}
    )
    stability: float = 1.0
    quantum_coupling: float = 1.0
    stability_momentum: float = 0.0
    last_expulsion: float = 0.0
    recycled_surplus: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize all values."""
        # Required keys for complete state
        required_keys = {'basal', 'cognitive', 'predictive', 'ontological'}

        # Ensure all required keys exist in values and accumulation_rate
        for key in required_keys:
            if key not in self.values:
                logger.warning(f"Missing required key '{key}' in values, initializing to 1.0")
                self.values[key] = 1.0
            if key not in self.accumulation_rate:
                logger.warning(f"Missing required key '{key}' in accumulation_rate, initializing to 0.01")
                self.accumulation_rate[key] = 0.01

        # Ensure all values are floats
        try:
            self.values = {k: float(v) for k, v in self.values.items()}
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting values to float: {e}")
            self.values = {k: 1.0 for k in required_keys}

        try:
            self.accumulation_rate = {k: float(v) for k, v in self.accumulation_rate.items()}
        except (TypeError, ValueError) as e:
            logger.error(f"Error converting accumulation_rate to float: {e}")
            self.accumulation_rate = {k: 0.01 for k in required_keys}

        # Ensure numeric values are floats
        try:
            self.stability = float(self.stability)
        except (TypeError, ValueError):
            logger.warning("Invalid stability value, defaulting to 1.0")
            self.stability = 1.0

        try:
            self.quantum_coupling = float(self.quantum_coupling)
        except (TypeError, ValueError):
            logger.warning("Invalid quantum_coupling value, defaulting to 1.0")
            self.quantum_coupling = 1.0

        try:
            self.stability_momentum = float(self.stability_momentum)
        except (TypeError, ValueError):
            logger.warning("Invalid stability_momentum value, defaulting to 0.0")
            self.stability_momentum = 0.0

        try:
            self.last_expulsion = float(self.last_expulsion)
        except (TypeError, ValueError):
            logger.warning("Invalid last_expulsion value, defaulting to 0.0")
            self.last_expulsion = 0.0

        # Ensure recycled_surplus is a dictionary with float values
        if not isinstance(self.recycled_surplus, dict):
            logger.warning("Invalid recycled_surplus type, defaulting to empty dict")
            self.recycled_surplus = {}
        else:
            try:
                self.recycled_surplus = {k: float(v) for k, v in self.recycled_surplus.items()}
            except (TypeError, ValueError):
                logger.warning("Error converting recycled_surplus values to float, using empty dict")
                self.recycled_surplus = {}

    def validate(self) -> bool:
        """Validate SurplusState fields."""
        try:
            # Validate types
            if not isinstance(self.values, dict):
                logger.error("SurplusState.values is not a dict")
                return False
            if not isinstance(self.accumulation_rate, dict):
                logger.error("SurplusState.accumulation_rate is not a dict")
                return False
            if not isinstance(self.stability, float):
                logger.error(f"SurplusState.stability is not a float: {type(self.stability)}")
                return False
            if not isinstance(self.quantum_coupling, float):
                logger.error(f"SurplusState.quantum_coupling is not a float: {type(self.quantum_coupling)}")
                return False

            # Validate required keys
            required_keys = {'basal', 'cognitive', 'predictive', 'ontological'}
            for key in required_keys:
                if key not in self.values:
                    logger.error(f"SurplusState missing key: {key}")
                    return False
                if not isinstance(self.values[key], (int, float)):
                    logger.error(f"Invalid type for {key}: {type(self.values[key])}")
                    return False

            # Validate value ranges (optional, based on domain knowledge)
            for key, value in self.values.items():
                if value < 0:
                    logger.warning(f"Negative value for {key}: {value}")

            if self.stability < 0 or self.stability > 1.0:
                logger.warning(f"Stability outside [0,1] range: {self.stability}")

            if self.quantum_coupling < 0 or self.quantum_coupling > 1.0:
                logger.warning(f"Quantum coupling outside [0,1] range: {self.quantum_coupling}")

            return True
        except Exception as e:
            logger.error(f"Error in SurplusState validation: {e}")
            return False

    def copy(self) -> 'SurplusState':
        """Create a deep copy of the surplus state with proper type handling."""
        try:
            # Handle values dictionary
            new_values = {}
            for k, v in self.values.items():
                if callable(v):  # Skip if it's a method
                    continue
                try:
                    new_values[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid value for {k}, using default")
                    new_values[k] = 1.0

            # Handle accumulation rates
            new_accumulation_rate = {}
            for k, v in self.accumulation_rate.items():
                if callable(v):  # Skip if it's a method
                    continue
                try:
                    new_accumulation_rate[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid accumulation rate for {k}, using default")
                    new_accumulation_rate[k] = 0.01

            # Handle recycled surplus
            new_recycled_surplus = {}
            for k, v in self.recycled_surplus.items():
                if callable(v):  # Skip if it's a method
                    continue
                try:
                    new_recycled_surplus[k] = float(v)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid recycled surplus for {k}, using default")
                    new_recycled_surplus[k] = 0.0

            # Create new instance with properly typed values
            return SurplusState(
                values=new_values,
                accumulation_rate=new_accumulation_rate,
                stability=float(self.stability) if not callable(self.stability) else 1.0,
                quantum_coupling=float(self.quantum_coupling) if not callable(self.quantum_coupling) else 1.0,
                stability_momentum=float(self.stability_momentum) if not callable(self.stability_momentum) else 0.0,
                last_expulsion=float(self.last_expulsion) if not callable(self.last_expulsion) else 0.0,
                recycled_surplus=new_recycled_surplus
            )

        except Exception as e:
            logger.error(f"Error copying SurplusState: {e}")
            # Return a new default instance if copying fails
            return SurplusState()

    def total_surplus(self) -> float:
        """Calculate total surplus with proper type handling."""
        try:
            if not isinstance(self.values, dict):
                logger.warning("Values is not a dictionary")
                return 0.0

            total = 0.0
            for key, value in self.values.items():
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    logger.warning(f"Could not convert value for {key} to float")

            return total
        except Exception as e:
            logger.error(f"Error calculating total surplus: {e}")
            return 0.0


@dataclass
class TransformerOutput:
    """Container for transformer outputs with quantum-aware processing."""
    prediction: torch.Tensor
    phase_prediction: Optional[torch.Tensor] = None
    value_estimate: Optional[torch.Tensor] = None
    attention_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    entropy: Optional[torch.Tensor] = None
    coherence_estimate: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Ensure all tensors are properly initialized and on correct device."""
        try:
            # Ensure prediction is a tensor
            if not isinstance(self.prediction, torch.Tensor):
                logger.warning(f"Prediction is not a tensor, converting from {type(self.prediction)}")
                try:
                    self.prediction = torch.tensor(self.prediction, dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Could not convert prediction to tensor: {e}")
                    self.prediction = torch.tensor(0.0)

            # Get device from prediction tensor
            device = self.prediction.device

            # Handle phase prediction
            if self.phase_prediction is None:
                self.phase_prediction = torch.zeros_like(self.prediction)
            elif not isinstance(self.phase_prediction, torch.Tensor):
                try:
                    self.phase_prediction = torch.tensor(self.phase_prediction, device=device)
                except Exception as e:
                    logger.error(f"Could not convert phase_prediction to tensor: {e}")
                    self.phase_prediction = torch.zeros_like(self.prediction)

            # Handle value estimate
            if self.value_estimate is None:
                self.value_estimate = torch.zeros_like(self.prediction)
            elif not isinstance(self.value_estimate, torch.Tensor):
                try:
                    self.value_estimate = torch.tensor(self.value_estimate, device=device)
                except Exception as e:
                    logger.error(f"Could not convert value_estimate to tensor: {e}")
                    self.value_estimate = torch.zeros_like(self.prediction)

            # Handle entropy
            if self.entropy is None:
                self.entropy = torch.tensor(0.0, device=device)
            elif not isinstance(self.entropy, torch.Tensor):
                try:
                    self.entropy = torch.tensor(self.entropy, device=device)
                except Exception as e:
                    logger.error(f"Could not convert entropy to tensor: {e}")
                    self.entropy = torch.tensor(0.0, device=device)

            # Handle coherence estimate
            if self.coherence_estimate is None:
                self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)
            elif not isinstance(self.coherence_estimate, torch.Tensor):
                try:
                    self.coherence_estimate = torch.tensor(self.coherence_estimate, device=device)
                except Exception as e:
                    logger.error(f"Could not convert coherence_estimate to tensor: {e}")
                    self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)

            # Ensure attention weights are proper tensors
            if not isinstance(self.attention_weights, dict):
                logger.warning(f"Attention weights is not a dict, initializing empty dict")
                self.attention_weights = {}
            else:
                for key, value in list(self.attention_weights.items()):
                    if not isinstance(value, torch.Tensor):
                        try:
                            self.attention_weights[key] = torch.tensor(value, device=device)
                        except Exception as e:
                            logger.error(f"Could not convert attention weight {key} to tensor: {e}")
                            del self.attention_weights[key]

        except Exception as e:
            logger.error(f"Error in TransformerOutput initialization: {e}")
            # Set safe default values, ensuring device is preserved
            device = getattr(self.prediction, 'device', None)
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.prediction = torch.tensor(0.0, device=device)
            self.phase_prediction = torch.tensor(0.0, device=device)
            self.value_estimate = torch.tensor(0.0, device=device)
            self.attention_weights = {}
            self.entropy = torch.tensor(0.0, device=device)
            self.coherence_estimate = torch.tensor(MINIMUM_COHERENCE_FLOOR, device=device)

    def validate(self) -> bool:
        """Validate transformer output state."""
        try:
            # Check prediction tensor
            if not isinstance(self.prediction, torch.Tensor):
                logger.error("Invalid prediction type")
                return False

            # Check tensor dimensions
            if self.prediction.dim() > 3:
                logger.error(f"Invalid prediction dimensions: {self.prediction.dim()}")
                return False

            # Check for NaN values
            if torch.isnan(self.prediction).any():
                logger.error("NaN values in prediction")
                return False

            # Check phase prediction
            if self.phase_prediction is not None:
                if not isinstance(self.phase_prediction, torch.Tensor):
                    logger.error("Invalid phase prediction type")
                    return False
                if torch.isnan(self.phase_prediction).any():
                    logger.error("NaN values in phase prediction")
                    return False

            # Check value estimate
            if self.value_estimate is not None:
                if not isinstance(self.value_estimate, torch.Tensor):
                    logger.error("Invalid value estimate type")
                    return False
                if torch.isnan(self.value_estimate).any():
                    logger.error("NaN values in value estimate")
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
            logger.error(f"Error validating transformer output: {e}")
            return False

    @property
    def device(self) -> torch.device:
        """Get the device of the prediction tensor."""
        return self.prediction.device

    def to(self, device: torch.device) -> 'TransformerOutput':
        """Move all tensors to specified device."""
        try:
            self.prediction = self.prediction.to(device)

            if self.phase_prediction is not None:
                self.phase_prediction = self.phase_prediction.to(device)

            if self.value_estimate is not None:
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
