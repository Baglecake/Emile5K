"""
Utilities and Helper Functions for Émile-2 Simulation
----------------------------------------------------
Core constants and utility functions used throughout the simulation.
"""
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import traceback
from typing import Union, List, Dict, Optional, Tuple, Any
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SaveStatevector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("emile4.utilities")

# ==========================
# Global Constants
# ==========================
# Core simulation parameters
NUM_QUBITS_PER_AGENT = 4
DECOHERENCE_RATE = 0.01
MINIMUM_COHERENCE_FLOOR = 0.0001
MOMENTUM_DECAY = 0.7
DISTINCTION_ANCHOR_WEIGHT = 0.2

# Surplus and distinction parameters
SURPLUS_ADJUSTMENT_RATE = 0.05
MAX_SURPLUS = 10.0
EXPULSION_RECOVERY_RATE = 0.02
SURPLUS_THRESHOLD = 1.5
TARGET_DISTINCTION = 0.7
PHASE_SCALING_FACTOR = 0.3
SURPLUS_RECYCLE_FRACTION = 0.7
COLLAPSE_DISSIPATION_THRESHOLD = 0.35
COLLAPSE_DISSIPATION_RATE = 0.02  # Added missing constant
CORE_DISTINCTION_UPDATE_RATE = 0.01
INSTABILITY_GRACE_PERIOD = 3

# Learning and training parameters
LEARNING_RATE = 1e-3
LEARNING_RATE_MIN = 1e-5
LEARNING_RATE_MAX = 1e-3
WEIGHT_DECAY = 0.005
GRADIENT_CLIP_VALUE = 1.0
REWARD_SCALING = 1.5
EVOLUTION_TIME = 0.1

# QPE (Quantum Phase Estimation) precision
QPE_PRECISION_QUBITS = 4

# Maximum entropy placeholder
MAX_ENTROPY = np.log(2 ** NUM_QUBITS_PER_AGENT)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer parameters
HIDDEN_DIM = 20  # Added missing constant
NUM_TRANSFORMER_HEADS = 4  # Added missing constant
NUM_TRANSFORMER_LAYERS = 2  # Added missing constant

# ==========================
# Custom Exceptions
# ==========================
class QuantumCircuitError(Exception):
    """Error during quantum circuit initialization or operation."""
    pass

class TensorShapeError(Exception):
    """Error when tensor shapes don't match expected dimensions."""
    pass

class ValueConversionError(Exception):
    """Error when converting between numeric types."""
    pass

# ==========================
# Utility Functions
# ==========================

def _initialize_circuit(num_qubits: int) -> Tuple[QuantumCircuit, Optional[AerSimulator]]:
    """Initialize quantum circuit and simulator."""
    try:
        logger.info(f"Initializing quantum circuit with {num_qubits} qubits")
        qc = QuantumCircuit(num_qubits, name="agent_circuit")

        try:
            simulator = AerSimulator(method='statevector')
        except Exception as sim_err:
            logger.warning(f"Error initializing AerSimulator: {sim_err}. Falling back to default.")
            simulator = None

        logger.info("Quantum circuit and simulator initialized successfully.")
        return qc, simulator
    except Exception as e:
        logger.error(f"Error in _initialize_circuit: {e}")
        traceback.print_exc()
        raise QuantumCircuitError(f"Failed to initialize quantum circuit: {str(e)}")

def ensure_real(value: Any, default: float = 0.0) -> float:
    """
    Convert value to real float, handling complex numbers and arrays.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Real float value
    """
    try:
        if value is None:
            return default
        if isinstance(value, (complex, np.complex64, np.complex128)):
            return float(np.real(value))
        elif isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                return default
            return float(np.real(value[0]))
        return float(value)
    except (TypeError, ValueError, IndexError) as e:
        logger.debug(f"Value conversion error: {e}, using default {default}")
        return default

def adapt_tensor_shape(x: torch.Tensor, expected_dim: int = 3,
                       expected_last_dim: int = 20) -> torch.Tensor:
    """
    Adapt tensor shape to expected dimensions.

    Removes extra singleton dimensions (except batch) and ensures the last
    dimension matches expected_last_dim by padding or trimming.

    Args:
        x: Input tensor
        expected_dim: Expected number of dimensions
        expected_last_dim: Expected size of last dimension

    Returns:
        Tensor with appropriate shape
    """
    try:
        # If x is a scalar, unsqueeze it
        if x.dim() == 0:
            x = x.unsqueeze(0)

        # Save original batch size
        batch_size = x.size(0)

        # Squeeze dimensions beyond the batch dimension
        new_shape = [batch_size]
        for i in range(1, x.dim()):
            if x.size(i) != 1:
                new_shape.append(x.size(i))

        # If nothing remains after batch dimension, add sequence dimension of size 1
        if len(new_shape) == 1:
            new_shape.append(1)

        # Try to reshape the tensor
        try:
            x = x.view(*new_shape)
        except RuntimeError as e:
            logger.warning(f"Error reshaping tensor from {x.shape} to {new_shape}: {e}")
            # If view fails, try reshape as a fallback
            try:
                x = x.reshape(batch_size, -1)
            except RuntimeError:
                pass

        # Ensure tensor is n-dimensional
        if x.dim() == 2:
            # If it is [B, feature], add a sequence dimension
            x = x.unsqueeze(1)
        elif x.dim() > expected_dim:
            # If more than expected dims, collapse dims 1 to n-1 into one
            shape = x.shape
            x = x.view(shape[0], -1, shape[-1])

        # Ensure last dimension equals expected_last_dim
        current_last_dim = x.size(-1)
        if current_last_dim < expected_last_dim:
            padding = (0, expected_last_dim - current_last_dim)
            x = F.pad(x, padding)
        elif current_last_dim > expected_last_dim:
            x = x[..., :expected_last_dim]

        return x
    except Exception as e:
        logger.error(f"Failed to adapt tensor shape: {e}")
        raise TensorShapeError(f"Failed to adapt tensor shape: {str(e)}")

def update_momentum(old_value: float, new_sample: float,
                   decay: float = MOMENTUM_DECAY) -> float:
    """
    Update value using exponential moving average.

    Args:
        old_value: Previous momentum value
        new_sample: New sample value
        decay: Momentum decay factor (default: MOMENTUM_DECAY)

    Returns:
        Updated momentum value
    """
    return float(decay * old_value + (1 - decay) * new_sample)

def to_float(x: Union[float, int, np.number, torch.Tensor],
            default: float = 0.0) -> float:
    """
    Convert various numeric types to float.

    Args:
        x: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value
    """
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)
    except (TypeError, ValueError) as e:
        logger.debug(f"Float conversion error: {e}, using default {default}")
        return default

def compute_phase_coherence(phases: Optional[List[float]] = None,
                           default: float = MINIMUM_COHERENCE_FLOOR) -> float:
    """
    Compute phase coherence from a list of phases.

    Args:
        phases: List of phase values
        default: Default value if computation fails

    Returns:
        Phase coherence value in range [MINIMUM_COHERENCE_FLOOR, 1.0]
    """
    if not isinstance(phases, list) or not phases:
        logger.debug("Invalid input to compute_phase_coherence, using default")
        return default

    try:
        # Filter and convert values to float
        cleaned = np.array([to_float(p) for p in phases if isinstance(p, (int, float))])
        if cleaned.size == 0:
            return default

        # Compute coherence using complex phase representation
        complex_phases = np.exp(1j * cleaned)
        coherence = float(np.abs(np.mean(complex_phases)))
        return max(coherence, MINIMUM_COHERENCE_FLOOR)
    except Exception as e:
        logger.warning(f"Error in compute_phase_coherence: {e}")
        return default

def compute_normalized_entropy(probabilities: Union[List[float], np.ndarray]) -> float:
    """
    Compute normalized entropy from probability distribution.

    Args:
        probabilities: Probability distribution

    Returns:
        Normalized entropy in range [0, 1]
    """
    try:
        from scipy.stats import entropy

        # Convert to numpy array and normalize
        probabilities = np.array(probabilities, dtype=np.float64)

        # Handle potential NaN or infinite values
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for empty or all-zero array
        if probabilities.size == 0 or np.sum(probabilities) <= 1e-10:
            return 1.0

        # Normalize probabilities safely
        probabilities = probabilities / max(np.sum(probabilities), 1e-10)

        # Calculate entropy and normalize
        entropy_value = float(entropy(probabilities, base=2))
        max_possible_entropy = float(np.log2(len(probabilities))) if len(probabilities) > 0 else 1.0

        # Avoid division by zero
        if max_possible_entropy < 1e-10:
            return 1.0

        return entropy_value / max_possible_entropy
    except Exception as e:
        logger.warning(f"Error computing normalized entropy: {e}")
        return 1.0  # Return maximum entropy on error

def compute_context_similarity(ctx1: Dict, ctx2: Dict) -> float:
    """
    Compute cosine similarity between context dictionaries.

    Args:
        ctx1: First context dictionary
        ctx2: Second context dictionary

    Returns:
        Similarity score between 0 and 1
    """
    try:
        common_keys = set(ctx1.keys()) & set(ctx2.keys())
        if not common_keys:
            return 0.0

        vec1, vec2 = [], []
        for key in common_keys:
            try:
                vec1.append(to_float(ctx1[key]))
                vec2.append(to_float(ctx2[key]))
            except Exception:
                continue

        if not vec1:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
        similarity = (cosine_sim + 1) / 2.0

        # Weight by proportion of common keys
        total_keys = max(len(ctx1), len(ctx2))
        key_ratio = len(common_keys) / total_keys if total_keys > 0 else 1.0

        return similarity * key_ratio
    except Exception as e:
        logger.warning(f"Error in compute_context_similarity: {e}")
        return 0.0

def local_operator(i: int, n: int, op: np.ndarray) -> np.ndarray:
    """
    Create local quantum operator on specified qubit.

    Args:
        i: Target qubit index
        n: Total number of qubits
        op: Single-qubit operator (2x2 matrix)

    Returns:
        Full n-qubit operator
    """
    if i < 0 or i >= n:
        raise ValueError(f"Qubit index {i} out of range [0, {n-1}]")

    ops = [np.eye(2) for _ in range(n)]
    ops[i] = op

    result = ops[0]
    for k in range(1, n):
        result = np.kron(result, ops[k])

    return result

def two_qubit_operator(i: int, j: int, n: int, op: np.ndarray) -> np.ndarray:
    """
    Create two-qubit operator acting on specified qubits.

    Args:
        i: First qubit index
        j: Second qubit index
        n: Total number of qubits
        op: Two-qubit operator (4x4 matrix)

    Returns:
        Full n-qubit operator
    """
    if i < 0 or i >= n or j < 0 or j >= n or i == j:
        raise ValueError(f"Invalid qubit indices: i={i}, j={j}, n={n}")
    if op.shape != (4, 4):
        raise ValueError(f"Operator must be 4x4, got {op.shape}")

    # Create n-qubit identity operator
    result = np.eye(2**n)

    # Calculate the indices for the 2-qubit subspace
    # This is a more general approach than the original function
    subspace_indices = []
    for k in range(2**n):
        binary = format(k, f'0{n}b')
        if (i < j and binary[i] + binary[j] in ['00', '01', '10', '11']) or \
           (i > j and binary[j] + binary[i] in ['00', '01', '10', '11']):
            subspace_indices.append(k)

    # For each pair of basis states in the 2-qubit subspace
    for ii, idx1 in enumerate(subspace_indices):
        for jj, idx2 in enumerate(subspace_indices):
            # Apply the corresponding element of the 2-qubit operator
            result[idx1, idx2] = op[ii, jj]

    return result

def reshape_input_tensor(tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Reshape tensor to expected shape, with graceful fallback.

    Args:
        tensor: Input tensor
        expected_shape: Expected tensor shape

    Returns:
        Reshaped tensor
    """
    try:
        # Try direct view operation first
        return tensor.view(expected_shape)
    except RuntimeError:
        logger.warning(f"Cannot view tensor from {tensor.shape} to {expected_shape}")

        # If element count matches, try reshape
        if tensor.numel() == np.prod(expected_shape):
            try:
                return tensor.reshape(expected_shape)
            except RuntimeError as e:
                logger.warning(f"Reshape failed: {e}")

        # Try to preserve batch dimension and final dimension
        try:
            return tensor.view(-1, expected_shape[-1])
        except RuntimeError:
            logger.warning(f"Failed to reshape tensor to {expected_shape}")

        # Return original tensor as fallback
        return tensor
