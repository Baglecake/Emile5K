"""
Base Quantum Module for Émile-2 Simulation
------------------------------------------
Foundational quantum state functionality for the simulation.
"""
import logging
import numpy as np
from typing import Dict, Optional
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SaveStatevector
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.base_quantum")

# Import necessary constants
from utilities import MINIMUM_COHERENCE_FLOOR

class BaseQuantumState:
    """
    Base class for quantum state handling.

    Provides basic initialization and metrics calculation for quantum states.
    """
    def __init__(self, num_qubits: int = 4):
        """
        Initialize quantum state with the specified number of qubits.

        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.minimum_coherence = MINIMUM_COHERENCE_FLOOR

        # Initialize quantum circuit
        self.qc = QuantumCircuit(num_qubits)
        logger.debug(f"Initialized quantum circuit with {num_qubits} qubits")

        # Initialize simulator
        try:
            self.simulator = AerSimulator(method='statevector')
            logger.debug("AerSimulator initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing simulator: {e}")
            self.simulator = None

        # Initialize statevector
        try:
            self.statevector = Statevector.from_label('0' * num_qubits)
            logger.debug(f"Statevector initialized to |{'0' * num_qubits}⟩")
        except Exception as e:
            logger.error(f"Error initializing statevector: {e}")
            self.statevector = None

        # Set initial phase and metrics
        self.phase = 0.0
        self.phase_coherence = MINIMUM_COHERENCE_FLOOR

    def get_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic quantum metrics from the current state.

        Returns:
            Dictionary containing phase, phase_coherence, and normalized_entropy
        """
        try:
            # Handle invalid statevector
            if self.statevector is None:
                logger.warning("Statevector is None, returning default metrics")
                return {
                    'phase': 0.0,
                    'phase_coherence': self.minimum_coherence,
                    'normalized_entropy': 0.0
                }

            # Extract probabilities from statevector
            if isinstance(self.statevector, np.ndarray):
                probs = np.abs(self.statevector) ** 2
            else:
                probs = np.abs(np.array(self.statevector.data)) ** 2

            # Calculate entropy
            entropy = self._calculate_entropy(probs)
            max_entropy = np.log2(len(probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Calculate coherence as complement of normalized entropy
            coherence = max(self.minimum_coherence, 1.0 - normalized_entropy)

            return {
                'phase': float(self.phase),
                'phase_coherence': float(coherence),
                'normalized_entropy': float(normalized_entropy),
                'probabilities': probs.tolist()  # Add probabilities for advanced analysis
            }

        except Exception as e:
            logger.error(f"Error getting basic metrics: {e}")
            return {
                'phase': 0.0,
                'phase_coherence': self.minimum_coherence,
                'normalized_entropy': 0.0
            }

    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate the von Neumann entropy from probability distribution.

        Args:
            probabilities: Probability distribution from statevector

        Returns:
            Entropy value
        """
        try:
            # Add small epsilon to avoid log(0)
            epsilon = np.finfo(float).eps
            probabilities = probabilities + epsilon
            probabilities = probabilities / np.sum(probabilities)  # Renormalize

            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return float(entropy)

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """
        Execute a quantum circuit and return measurement results.

        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots

        Returns:
            Dictionary of measurement results
        """
        try:
            if self.simulator is None:
                logger.error("Simulator is None, cannot execute circuit")
                return {}

            # Execute circuit
            result = self.simulator.run(circuit, shots=shots).result()

            # Return counts
            if result.success:
                return dict(result.get_counts())
            else:
                logger.error(f"Circuit execution failed: {result.status}")
                return {}

        except Exception as e:
            logger.error(f"Error executing circuit: {e}")
            return {}

    def reset_to_ground_state(self) -> bool:
        """
        Reset quantum state to the ground state.

        Returns:
            True if reset successful, False otherwise
        """
        try:
            # Create new circuit and statevector
            self.qc = QuantumCircuit(self.num_qubits)
            self.statevector = Statevector.from_label('0' * self.num_qubits)
            self.phase = 0.0
            self.phase_coherence = self.minimum_coherence

            logger.info("Reset to ground state successful")
            return True

        except Exception as e:
            logger.error(f"Error resetting to ground state: {e}")
            return False

    def is_valid(self) -> bool:
        """
        Check if quantum state is valid and ready for operations.

        Returns:
            True if state is valid, False otherwise
        """
        try:
            # Check quantum circuit
            if not isinstance(self.qc, QuantumCircuit):
                logger.error("Invalid quantum circuit")
                return False

            # Check statevector
            if self.statevector is None:
                logger.error("Statevector is None")
                return False

            # Check statevector normalization
            if isinstance(self.statevector, np.ndarray):
                state_data = self.statevector
            else:
                state_data = np.array(self.statevector.data)

            norm = np.linalg.norm(state_data)
            if not np.isclose(norm, 1.0, atol=1e-6):
                logger.error(f"Statevector not normalized: norm = {norm}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating quantum state: {e}")
            return False

    def duplicate(self) -> Optional['BaseQuantumState']:
        """
        Create a copy of the current quantum state.

        Returns:
            New BaseQuantumState instance with same values, or None if duplication fails
        """
        try:
            # Create new instance
            new_state = BaseQuantumState(self.num_qubits)

            # Copy statevector
            if isinstance(self.statevector, np.ndarray):
                new_state.statevector = np.copy(self.statevector)
            elif hasattr(self.statevector, 'copy'):
                new_state.statevector = self.statevector.copy()

            # Copy phase and coherence
            new_state.phase = self.phase
            new_state.phase_coherence = self.phase_coherence
            new_state.minimum_coherence = self.minimum_coherence

            return new_state

        except Exception as e:
            logger.error(f"Error duplicating quantum state: {e}")
            return None

