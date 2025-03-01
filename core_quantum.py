"""
Core Quantum Management and State
---------------------------------
Advanced quantum state management for Émile-4 simulation with enhanced
stability and error handling.
"""

import logging
import math
import traceback
import numpy as np
import random
from collections import deque
from typing import Optional, Dict, Tuple, List, Any, Union
import time
import scipy.linalg

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
from qiskit_aer.library import SaveStatevector

from base_quantum import BaseQuantumState
from data_classes import SurplusState
from utilities import (
    DECOHERENCE_RATE,
    MINIMUM_COHERENCE_FLOOR,
    MOMENTUM_DECAY,
    compute_phase_coherence,
    update_momentum,
    ensure_real
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.core_quantum")


def create_noise_model() -> NoiseModel:
    """
    Create noise model for quantum simulations.

    Returns:
        NoiseModel with amplitude and phase damping errors
    """
    try:
        noise_model = NoiseModel()
        error_amp = amplitude_damping_error(DECOHERENCE_RATE)
        error_phase = phase_damping_error(DECOHERENCE_RATE)

        # Add errors for common gates
        for gate in ['id', 'u1', 'u2', 'u3']:
            noise_model.add_all_qubit_quantum_error(error_amp, [gate])
            noise_model.add_all_qubit_quantum_error(error_phase, [gate])

        return noise_model
    except Exception as e:
        logger.error(f"Error creating noise model: {e}")
        return NoiseModel()


class SaveStatevectorWrapper:
    """
    Wrapper for SaveStatevector instruction with proper error handling.
    """
    def __init__(self, num_qubits: int):
        """
        Initialize SaveStatevector wrapper.

        Args:
            num_qubits: Number of qubits in the system
        """
        self.instruction = None
        try:
            self.instruction = SaveStatevector(num_qubits)
        except Exception as e:
            logger.warning(f"Could not initialize SaveStatevector: {e}")
            logger.warning("This may affect statevector operations")

    def apply(self, qc: QuantumCircuit) -> bool:
        """
        Apply SaveStatevector instruction to circuit.

        Args:
            qc: Quantum circuit to apply instruction to

        Returns:
            True if successful, False otherwise
        """
        if self.instruction is None:
            logger.warning("SaveStatevector instruction is missing. State will not be saved.")
            return False

        try:
            qc.append(self.instruction, qc.qubits)
            return True
        except Exception as e:
            logger.error(f"Error applying SaveStatevector: {e}")
            return False


class EnhancedQuantumState(BaseQuantumState):
    """
    Enhanced quantum state management with improved coherence and stability.
    """
    def __init__(self, agent=None, num_qubits: int = 4):
        """
        Initialize enhanced quantum state.

        Args:
            agent: Reference to parent agent (optional)
            num_qubits: Number of qubits in the system
        """
        # Call parent initializer
        super().__init__(num_qubits)

        # Store agent reference for callbacks
        self.agent = agent

        # Initialize quantum parameters
        self.phase = 0.0
        self.phase_coherence = MINIMUM_COHERENCE_FLOOR

        # Initialize tracking metrics
        self.coherence_momentum = 0.0
        self.coherence_history = deque(maxlen=100)
        self.evolution_history = deque(maxlen=100)
        self.phase_history = deque(maxlen=100)
        self.measurement_history = deque(maxlen=100)

        # Initialize recovery tracking
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

        # Initialize noise model
        try:
            self.noise_model = create_noise_model()
            logger.debug("Noise model initialized successfully")
        except Exception as e:
            logger.warning(f"Error initializing noise model: {e}")
            self.noise_model = None

        # Ensure initial state is prepared
        self._prepare_ground_state_with_coherence()

        # Store initial metrics
        try:
            initial_metrics = self.get_quantum_metrics()
            self.coherence_history.append(initial_metrics['phase_coherence'])
            self.phase_history.append(initial_metrics['phase'])
            logger.debug(f"Initial metrics: coherence={initial_metrics['phase_coherence']:.4f}, phase={initial_metrics['phase']:.4f}")
        except Exception as e:
            logger.error(f"Error storing initial metrics: {e}")
            # Set default values
            self.coherence_history.append(self.phase_coherence)
            self.phase_history.append(self.phase)

    def get_coherence_variance(self) -> float:
        """
        Calculate variance of recent coherence values.

        Returns:
            Float representing variance of coherence over recent history
        """
        try:
            # Ensure coherence_history exists
            if not hasattr(self, 'coherence_history'):
                self.coherence_history = deque(maxlen=100)
                # Populate with some slightly varied initial values if empty
                if len(self.coherence_history) == 0:
                    base_coherence = self.phase_coherence
                    for _ in range(5):
                        variation = base_coherence * (1.0 + 0.02 * (random.random() - 0.5))
                        self.coherence_history.append(variation)

            # Calculate variance with minimum sample size check
            if len(self.coherence_history) >= 2:
                coherence_variance = float(np.var(list(self.coherence_history)))

                # Add small noise to prevent exactly zero variance
                if coherence_variance < 1e-6:
                    coherence_variance = 1e-6 + np.random.random() * 1e-5

                return coherence_variance
            else:
                # Default non-zero value if not enough history
                return 0.001

        except Exception as e:
            logger.error(f"Error calculating coherence variance: {e}")
            return 0.001  # Safe default

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

    def _prepare_ground_state_with_coherence(self) -> bool:
        """
        Prepare initial quantum state with minimum coherence.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new circuit
            self.qc = QuantumCircuit(self.num_qubits)
            logger.debug(f"New quantum circuit created with {self.num_qubits} qubits")

            # Initialize statevector if invalid
            if self.statevector is None or not isinstance(self.statevector, (Statevector, np.ndarray)):
                logger.warning("Statevector is invalid, reinitializing to |0⟩ state")
                self.statevector = Statevector.from_label('0' * self.num_qubits)

            # Initialize quantum parameters
            self.phase_coherence = MINIMUM_COHERENCE_FLOOR
            self.phase = 0.0

            # Apply Hadamard gates to create superposition
            for q in range(self.num_qubits):
                self.qc.h(q)
            logger.debug(f"Applied Hadamard gates to all {self.num_qubits} qubits")

            # Add state saving instruction
            save_sv = SaveStatevectorWrapper(self.num_qubits)
            save_sv.apply(self.qc)

            # Execute circuit with error handling
            if self.simulator:
                try:
                    # Transpile circuit
                    transpiled = transpile(self.qc, self.simulator)

                    # Run simulation with noise model
                    result = self.simulator.run(
                        transpiled,
                        noise_model=self.noise_model if hasattr(self, 'noise_model') else None
                    ).result()

                    # Extract statevector
                    if 'statevector' in result.data():
                        self.statevector = result.get_statevector()
                        logger.debug("Statevector extracted successfully from simulation result")

                        # Validate and normalize statevector
                        if isinstance(self.statevector, Statevector):
                            state_data = np.array(self.statevector.data)
                        else:
                            state_data = np.array(self.statevector)

                        # Check normalization
                        norm = np.linalg.norm(state_data)
                        if not np.isclose(norm, 1.0, atol=1e-6):
                            logger.warning(f"Statevector not normalized (norm={norm}). Renormalizing.")
                            state_data /= norm
                            self.statevector = Statevector(state_data)
                    else:
                        logger.warning("No statevector found in simulation result. Using default.")
                        self.statevector = Statevector.from_label('0' * self.num_qubits)
                except Exception as e:
                    logger.error(f"Simulation error: {e}")
                    logger.error("Using default state initialization")
                    self.statevector = Statevector.from_label('0' * self.num_qubits)
            else:
                logger.warning("Simulator is not available. Using default initialization.")
                self.statevector = Statevector.from_label('0' * self.num_qubits)

            # Update quantum metrics
            self.update_phase_coherence()
            self._update_state_metrics()
            logger.debug(f"After initialization: coherence={self.phase_coherence:.4f}, phase={self.phase:.4f}")

            # Store initial state in history
            if hasattr(self, 'evolution_history'):
                self.evolution_history.append({
                    'time': 0,
                    'statevector': self.statevector.copy() if isinstance(self.statevector, Statevector) else Statevector(self.statevector),
                    'phase_coherence': self.phase_coherence,
                    'phase': self.phase
                })

            return True

        except Exception as e:
            logger.error(f"Error in ground state preparation: {e}")
            traceback.print_exc()
            return self._attempt_state_recovery()

    def _attempt_state_recovery(self) -> bool:
        """
        Attempt recovery from failed state operations.

        Returns:
            True if recovery successful, False otherwise
        """
        try:
            self.recovery_attempts += 1
            logger.warning(f"Attempting state recovery (attempt {self.recovery_attempts}/{self.max_recovery_attempts})")

            # Reset circuit
            self.qc = QuantumCircuit(self.num_qubits)

            # Reset to basic state
            self.statevector = Statevector.from_label('0' * self.num_qubits)

            # Apply basic Hadamard to all qubits
            for q in range(self.num_qubits):
                self.qc.h(q)

            # Update metrics with safe defaults
            self.phase_coherence = self.minimum_coherence
            self.phase = 0.0

            logger.info("Basic state recovery complete")

            # Reset recovery counter if successful
            if self.recovery_attempts >= self.max_recovery_attempts:
                logger.warning("Maximum recovery attempts reached")
                self.recovery_attempts = 0

            return True

        except Exception as e:
            logger.error(f"Error in state recovery: {e}")

            # Set absolute minimum working state as last resort
            self.statevector = Statevector.from_label('0' * self.num_qubits)
            self.phase_coherence = self.minimum_coherence
            self.phase = 0.0

            return False

    def update_state(self) -> bool:
        """
        Update quantum state with enhanced error handling and recovery.

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Create new circuit
            self.qc = QuantumCircuit(self.num_qubits)

            # Validate statevector
            if self.statevector is None or not isinstance(self.statevector, (Statevector, np.ndarray)):
                logger.warning("Invalid statevector detected. Resetting to |0⟩ state.")
                self.statevector = Statevector.from_label('0' * self.num_qubits)

            # Extract state data
            if isinstance(self.statevector, Statevector):
                state_data = self.statevector.data
            else:
                state_data = self.statevector

            # Initialize circuit to current state
            self.qc.initialize(list(state_data), range(self.num_qubits))
            logger.debug("Initialized quantum circuit to current state")

            # Add state saving instruction
            save_sv = SaveStatevectorWrapper(self.num_qubits)
            save_sv.apply(self.qc)

            # Execute circuit with simulator
            if self.simulator:
                try:
                    # Transpile circuit
                    transpiled = transpile(self.qc, self.simulator)

                    # Run simulation
                    result = self.simulator.run(
                        transpiled,
                        noise_model=self.noise_model if hasattr(self, 'noise_model') else None,
                        shots=1
                    ).result()

                    # Process result
                    if 'statevector' in result.data():
                        self.statevector = result.get_statevector()
                        logger.debug("Updated statevector from simulation result")

                        # Update quantum metrics
                        self.update_phase_coherence()
                        self._update_state_metrics()

                        # Store state in history
                        if hasattr(self, 'evolution_history'):
                            self.evolution_history.append({
                                'time': len(self.evolution_history),
                                'statevector': self.statevector.copy() if isinstance(self.statevector, Statevector) else Statevector(self.statevector),
                                'phase_coherence': self.phase_coherence,
                                'phase': self.phase
                            })

                        return True
                    else:
                        logger.warning("No statevector found. Falling back to recovery.")
                        return self._attempt_state_recovery()
                except Exception as e:
                    logger.error(f"Error in circuit execution: {e}")
                    return self._attempt_state_recovery()
            else:
                logger.warning("Simulator unavailable. Using fallback statevector.")
                self.statevector = Statevector.from_label('0' * self.num_qubits)
                self.update_phase_coherence()
                self._update_state_metrics()
                return True

        except Exception as e:
            logger.error(f"Error updating quantum state: {e}")
            traceback.print_exc()
            return self._attempt_state_recovery()

    def reinforce_coherence(self, qc: QuantumCircuit,
                           distinction_variance: float,
                           phase_coherence: float) -> bool:
        """
        Reinforce quantum coherence based on distinction and coherence metrics.

        Args:
            qc: Quantum circuit to modify
            distinction_variance: Variance in distinction level
            phase_coherence: Current phase coherence

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure inputs are real floats
            phase_coherence = ensure_real(phase_coherence, MINIMUM_COHERENCE_FLOOR)
            distinction_variance = ensure_real(distinction_variance, 0.0)

            # Target coherence for the system
            target_coherence = 0.8  # High coherence goal

            # Calculate correction parameters
            coherence_error = target_coherence - phase_coherence
            base_angle = float((np.pi / 8) * np.sign(coherence_error))
            variance_factor = min(1.0, distinction_variance / 0.02)

            # Update momentum with type safety
            self.coherence_momentum = update_momentum(
                self.coherence_momentum,
                coherence_error
            )

            # Calculate final angle with momentum influence
            angle = base_angle * variance_factor * (1.0 + 0.1 * self.coherence_momentum)
            angle = ensure_real(angle, 0.0)

            # Apply quantum operations
            for q in range(self.num_qubits):
                qc.rz(0.1 * angle, q)  # Small phase adjustment
                qc.rx(angle, q)        # Main rotation

            # Track optimization
            if hasattr(self, 'optimization_history'):
                self.optimization_history.append({
                    'type': 'coherence_reinforcement',
                    'angle': float(angle),
                    'coherence': float(phase_coherence),
                    'momentum': float(self.coherence_momentum),
                    'timestamp': time.time()
                })

            logger.debug(f"Applied coherence reinforcement: angle={angle:.4f}, momentum={self.coherence_momentum:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error in coherence reinforcement: {e}")
            return False

    def _update_state_metrics(self) -> None:
        """Update and store quantum state evolution metrics."""
        try:
            # Extract probabilities from statevector
            if isinstance(self.statevector, Statevector):
                probs = np.abs(np.array(self.statevector.data))**2
            else:
                probs = np.abs(self.statevector)**2

            # Calculate entropy
            entropy_val = self._calculate_entropy(probs)

            # Handle potential complex values in phase_coherence
            coherence_real = ensure_real(self.phase_coherence, MINIMUM_COHERENCE_FLOOR)
            coherence_imag = 0.0
            if isinstance(self.phase_coherence, complex):
                coherence_imag = float(self.phase_coherence.imag)

            # Store metrics in evolution history
            self.evolution_history.append({
                'time': len(self.evolution_history),
                'entropy': float(entropy_val),
                'coherence_real': coherence_real,
                'coherence_imag': coherence_imag,
                'coherence_magnitude': abs(self.phase_coherence),
                'phase': float(self.phase)
            })

            logger.debug(f"Updated state metrics: entropy={entropy_val:.4f}, coherence={coherence_real:.4f}, phase={self.phase:.4f}")

        except Exception as e:
            logger.error(f"Error updating state metrics: {e}")
            # Add safe default values
            self.evolution_history.append({
                'time': len(self.evolution_history),
                'entropy': 0.0,
                'coherence': float(self.phase_coherence),
                'phase': float(self.phase)
            })

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
            # Handle case where all probabilities are zero or contain NaN values
            probabilities = np.nan_to_num(probabilities, nan=0.0)

            # Check if sum is zero
            total_prob = np.sum(probabilities)
            if total_prob < epsilon:
                return 0.0  # Return zero entropy for invalid distribution

            # Normalize properly with safeguard against division by zero
            probabilities = probabilities / total_prob

            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
            return float(entropy)

        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0

    def apply_gate(self, gate: str, qubits: List[int], params: Optional[Dict] = None) -> bool:
        """
        Apply quantum gate to circuit with error handling.

        Args:
            gate: Gate name ('h', 'x', 'rz', 'rx', 'cx')
            qubits: List of qubit indices
            params: Additional parameters for parameterized gates

        Returns:
            True if successful, False otherwise
        """
        if params is None:
            params = {}

        try:
            # Validate qubits
            if not qubits or not all(isinstance(q, int) and 0 <= q < self.num_qubits for q in qubits):
                logger.warning(f"Invalid qubits {qubits} for gate {gate}. Skipping operation.")
                return False

            # Create new circuit for gate application
            gate_circuit = QuantumCircuit(self.num_qubits)

            # Validate statevector
            if self.statevector is None or not isinstance(self.statevector, Statevector):
                logger.warning("Statevector is None. Resetting to |0⟩ state.")
                self.statevector = Statevector.from_label('0' * self.num_qubits)

            # Initialize to current state
            gate_circuit.initialize(list(self.statevector.data), range(self.num_qubits))

            # Apply requested gate
            if gate == 'h':
                for q in qubits:
                    gate_circuit.h(q)
                logger.debug(f"Applied Hadamard gate to qubits {qubits}")
            elif gate == 'x':
                for q in qubits:
                    gate_circuit.x(q)
                logger.debug(f"Applied X gate to qubits {qubits}")
            elif gate == 'rz':
                phi = ensure_real(params.get('phi', 0.0), 0.0)
                for q in qubits:
                    gate_circuit.rz(phi, q)
                logger.debug(f"Applied RZ gate with phi={phi:.4f} to qubits {qubits}")
            elif gate == 'rx':
                theta = ensure_real(params.get('theta', 0.0), 0.0)
                for q in qubits:
                    gate_circuit.rx(theta, q)
                logger.debug(f"Applied RX gate with theta={theta:.4f} to qubits {qubits}")
            elif gate == 'cx' and len(qubits) >= 2:
                gate_circuit.cx(qubits[0], qubits[1])
                logger.debug(f"Applied CX gate from qubit {qubits[0]} to {qubits[1]}")
            else:
                logger.warning(f"Unsupported gate: {gate}")
                return False

            # Add state saving instruction
            save_sv = SaveStatevectorWrapper(self.num_qubits)
            if save_sv.instruction is not None:
                gate_circuit.append(save_sv.instruction, gate_circuit.qubits)

            # Execute circuit
            if self.simulator is not None:
                try:
                    transpiled = transpile(gate_circuit, self.simulator)
                    result = self.simulator.run(transpiled, noise_model=self.noise_model).result()

                    if 'statevector' in result.data():
                        self.statevector = result.get_statevector()
                        return True
                    else:
                        logger.warning("No statevector returned after gate application. Attempting recovery.")
                        return self._attempt_state_recovery()
                except Exception as e:
                    logger.error(f"Error applying {gate}: {e}")
                    return self._attempt_state_recovery()
            else:
                logger.warning(f"Simulator unavailable while applying {gate}. Using fallback state.")
                self.statevector = Statevector.from_label('0' * self.num_qubits)
                self.update_phase_coherence()
                self._update_state_metrics()
                return True

        except Exception as e:
            logger.error(f"Error applying gate '{gate}' on qubits {qubits}: {e}")
            return self._attempt_state_recovery()

    def apply_phase_shift(self, angle: float) -> bool:
        """
        Apply phase shift to quantum state.

        Args:
            angle: Phase shift angle

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure angle is a real float
            angle = ensure_real(angle, 0.0)

            # Create phase shift circuit
            phase_circuit = QuantumCircuit(self.num_qubits)

            # Initialize to current state
            if isinstance(self.statevector, np.ndarray):
                current_state = Statevector(self.statevector)
            else:
                current_state = self.statevector

            phase_circuit.initialize(list(current_state.data), range(self.num_qubits))

            # Apply phase shift
            phase_circuit.rz(angle, 0)
            logger.debug(f"Applied phase shift of {angle:.4f} to qubit 0")

            # Add state saving instruction
            save_sv = SaveStatevectorWrapper(self.num_qubits)
            if save_sv.instruction is not None:
                phase_circuit.append(save_sv.instruction, phase_circuit.qubits)

            # Execute circuit
            if self.simulator is not None:
                transpiled = transpile(phase_circuit, self.simulator)
                result = self.simulator.run(transpiled, noise_model=self.noise_model).result()

                if 'statevector' in result.data():
                    self.statevector = result.get_statevector()
                    self.phase = (self.phase + angle) % (2 * np.pi)

                    if hasattr(self, 'phase_history'):
                        self.phase_history.append(self.phase)

                    # Update coherence
                    self.update_phase_coherence()
                    self._update_state_metrics()
                    return True
                else:
                    logger.error("No statevector in phase shift result")
                    return self._attempt_state_recovery()
            else:
                logger.error("Simulator not available")
                return self._attempt_state_recovery()

        except Exception as e:
            logger.error(f"Error applying phase shift: {e}")
            return self._attempt_state_recovery()

    def get_quantum_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive quantum metrics with robust error handling.

        Returns:
            Dictionary containing phase, phase_coherence, and normalized_entropy
        """
        try:
            # Ensure phase coherence is initialized
            if not hasattr(self, 'phase_coherence') or self.phase_coherence is None:
                self.phase_coherence = MINIMUM_COHERENCE_FLOOR

            # Get statevector probabilities
            if isinstance(self.statevector, np.ndarray):
                probs = np.abs(self.statevector) ** 2
            elif hasattr(self.statevector, 'data'):
                probs = np.abs(np.array(self.statevector.data)) ** 2
            else:
                logger.warning("Invalid statevector, reinitializing...")
                from qiskit.quantum_info import Statevector
                self.statevector = Statevector.from_label('0' * self.num_qubits)
                probs = np.abs(np.array(self.statevector.data)) ** 2

            # Calculate entropy
            entropy_val = self._calculate_entropy(probs)
            max_entropy = np.log2(2**self.num_qubits)
            normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0.0

            # Get phase information
            if not hasattr(self, 'phase'):
                self.phase = 0.0

            # Calculate phase distinction
            if hasattr(self, 'phase_history') and len(self.phase_history) > 0:
                mean_phase = float(np.mean(list(self.phase_history)))
                phase_distinction = float(abs(self.phase - mean_phase))
            else:
                phase_distinction = 0.0

            # Calculate mean coherence
            if hasattr(self, 'coherence_history') and len(self.coherence_history) > 0:
                mean_coherence = float(np.mean(list(self.coherence_history)))
            else:
                mean_coherence = float(self.phase_coherence)

            # Calculate phase stability
            if hasattr(self, 'phase_history') and len(self.phase_history) > 0:
                phase_stability = float(np.std(list(self.phase_history)))
            else:
                phase_stability = 1.0

            # Calculate coherence distinction and quantum coupling
            coherence_distinction = float(np.clip(self.phase_coherence, MINIMUM_COHERENCE_FLOOR, 1.0))
            quantum_coupling = float(np.clip(self.phase_coherence * (1.0 - normalized_entropy), 0.0, 1.0))
            quantum_surplus_coupling = float(np.clip(mean_coherence * (1.0 - normalized_entropy), 0.0, 1.0))
            stability = float(np.clip((mean_coherence + (1.0 - normalized_entropy)) / 2, 0.0, 1.0))

            # Calculate coherence variance using the dedicated method
            coherence_variance = self.get_coherence_variance()

            # Prepare metrics dictionary with explicit type conversion
            metrics = {
                'phase': float(self.phase % (2*np.pi)),
                'phase_coherence': float(self.phase_coherence),
                'entropy': float(entropy_val),
                'normalized_entropy': float(normalized_entropy),
                'mean_coherence': float(mean_coherence),
                'phase_stability': float(phase_stability),
                'phase_distinction': float(phase_distinction),
                'coherence_distinction': coherence_distinction,
                'quantum_coupling': quantum_coupling,
                'quantum_surplus_coupling': quantum_surplus_coupling,
                'stability': stability,
                'coherence_variance': coherence_variance  # Add coherence variance to metrics
            }

            # Validate metrics
            for key, value in metrics.items():
                if not isinstance(value, float) or math.isnan(value) or math.isinf(value):
                    logger.warning(f"Invalid {key} value: {value}, using default")
                    if key in ['phase_coherence', 'mean_coherence', 'coherence_distinction']:
                        metrics[key] = float(MINIMUM_COHERENCE_FLOOR)
                    elif key in ['quantum_coupling', 'quantum_surplus_coupling', 'stability']:
                        metrics[key] = 1.0
                    elif key == 'coherence_variance':  # Special handling for coherence_variance
                        metrics[key] = 0.001  # Default small non-zero value
                    else:
                        metrics[key] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error computing quantum metrics: {e}")
            traceback.print_exc()

            # Import random here to ensure it's available
            import random

            # Return safe default metrics
            return {
                'phase': random.uniform(0.0, 2*np.pi),
                'phase_coherence': float(MINIMUM_COHERENCE_FLOOR),
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'mean_coherence': float(MINIMUM_COHERENCE_FLOOR),
                'phase_stability': 1.0,
                'phase_distinction': 0.0,
                'coherence_distinction': float(MINIMUM_COHERENCE_FLOOR),
                'quantum_coupling': 1.0,
                'quantum_surplus_coupling': 1.0,
                'stability': 1.0,
                'coherence_variance': 0.001  # Include default coherence_variance in fallback
            }

    def compute_quantum_surplus_coupling(self, surplus_state: Any) -> float:
        """
        Compute coupling strength between quantum state and surplus values.

        Args:
            surplus_state: Either a SurplusState object or a dictionary of surplus values

        Returns:
            Coupling strength in range [0.1, 1.0]
        """
        try:
            # Get quantum metrics
            metrics = self.get_quantum_metrics()

            # Get key components from quantum metrics
            coherence_factor = max(metrics['phase_coherence'], self.minimum_coherence)
            entropy_factor = 1.0 - metrics['normalized_entropy']
            phase_factor = metrics.get('phase_distinction', 0.5)

            # Process the surplus_state input
            if isinstance(surplus_state, SurplusState):
                surplus_values = surplus_state.values
            elif isinstance(surplus_state, dict):
                surplus_values = surplus_state
            else:
                logger.warning("Invalid surplus state type; using default values")
                surplus_values = {
                    'basal': 1.0,
                    'cognitive': 1.0,
                    'predictive': 1.0,
                    'ontological': 1.0
                }

            # Calculate the total surplus
            surplus_total = sum(surplus_values.values())
            if surplus_total <= 0:
                logger.warning("Total surplus is zero or negative; using fallback value")
                surplus_total = 0.1  # Prevent division by zero or negative totals

            # Compute the coupling
            coupling = (0.4 * coherence_factor + 0.3 * entropy_factor + 0.3 * phase_factor) / (1.0 + surplus_total)

            # Return the coupling clipped between 0.1 and 1.0
            return float(np.clip(coupling, 0.1, 1.0))

        except Exception as e:
            logger.error(f"Error computing quantum surplus coupling: {e}")
            return 0.5

    def get_quantum_distinction_metrics(self) -> Dict[str, float]:
        """
        Get distinction-related quantum metrics.

        Returns:
            Dictionary of distinction metrics
        """
        try:
            # Get full quantum metrics
            quantum_metrics = self.get_quantum_metrics()

            # Calculate distinction factor from phase history
            distinction_factor = 0.0
            if hasattr(self, 'phase_history') and self.phase_history:
                avg_phase = np.mean(list(self.phase_history))
                distinction_factor = abs(quantum_metrics['phase'] - avg_phase)

            # Calculate field resistance based on entropy
            field_resistance = np.exp(-quantum_metrics['normalized_entropy'])

            # Calculate ontological stability combining coherence and distinction
            ontological_stability = (0.7 * quantum_metrics['phase_coherence'] +
                                     0.3 * (1.0 - distinction_factor))

            # Return distinction metrics
            return {
                'phase_distinction': float(distinction_factor),
                'field_resistance': float(field_resistance),
                'ontological_stability': float(ontological_stability),
                'coherence_distinction': float(quantum_metrics.get('phase_coherence', 0.0))
            }

        except Exception as e:
            logger.error(f"Error computing quantum distinction metrics: {e}")
            return {
                'phase_distinction': 0.0,
                'field_resistance': 0.0,
                'ontological_stability': self.minimum_coherence,
                'coherence_distinction': self.minimum_coherence
            }

    def update_phase_coherence(self) -> float:
        """
        Update phase coherence measure from the current statevector.

        Returns:
            Updated phase coherence value
        """
        try:
            # Validate statevector
            if self.statevector is None:
                logger.warning("Statevector is None. Defaulting phase coherence.")
                self.phase_coherence = self.minimum_coherence
                return self.phase_coherence

            # Extract numerical array from statevector
            if isinstance(self.statevector, Statevector):
                state_array = np.array(self.statevector.data, dtype=np.complex128)
            else:
                state_array = np.array(self.statevector, dtype=np.complex128)

            # Handle potential NaN values
            state_array = np.nan_to_num(state_array, nan=0.0)

            # Form the density matrix
            rho = np.outer(state_array, np.conj(state_array))

            # Validate density matrix
            if rho.shape[0] == 0:
                logger.warning("Density matrix is empty. Defaulting phase coherence.")
                self.phase_coherence = self.minimum_coherence
                return self.phase_coherence

            # Calculate coherence from off-diagonal elements
            diag_mask = np.eye(rho.shape[0], dtype=bool)
            off_diag_mask = ~diag_mask

            # Sum the absolute values of off-diagonal elements
            off_diag_sum = np.sum(np.abs(rho[off_diag_mask]))

            # Count off-diagonal elements
            count = np.sum(off_diag_mask)

            # Handle edge cases
            if count == 0 or np.isnan(off_diag_sum):
                logger.warning("No valid coherence data. Using minimum coherence.")
                self.phase_coherence = self.minimum_coherence
            else:
                # Ensure real value and proper bounds
                coherence = float(np.real(off_diag_sum / count))

                # Add small noise to encourage variation
                noise = (np.random.random() - 0.5) * 0.01
                coherence += noise

                self.phase_coherence = max(min(coherence, 1.0), self.minimum_coherence)

            # Ensure coherence_history attribute exists
            if not hasattr(self, 'coherence_history'):
                self.coherence_history = deque(maxlen=100)

            # Store in history
            self.coherence_history.append(self.phase_coherence)

            # Print debug info for coherence tracking
            if len(self.coherence_history) > 1:
                recent_coherence = list(self.coherence_history)[-5:]
                coherence_variance = np.var(recent_coherence)
                logger.debug(f"Recent coherence values: {recent_coherence}")
                logger.debug(f"Coherence variance: {coherence_variance}")

            return self.phase_coherence

        except Exception as e:
            logger.error(f"Error in update_phase_coherence: {e}")
            self.phase_coherence = self.minimum_coherence
            return self.phase_coherence

    def measure_state(self, shots: int = 1024) -> Dict[str, int]:
        """
        Measure the quantum state and return measurement results.

        Args:
            shots: Number of measurement shots

        Returns:
            Dictionary of measurement outcomes and counts
        """
        try:
            # Create measurement circuit
            measure_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

            # Initialize to current state
            if isinstance(self.statevector, np.ndarray):
                current_state = Statevector(self.statevector)
            else:
                current_state = self.statevector

            measure_circuit.initialize(list(current_state.data), range(self.num_qubits))

            # Add measurement operations
            for q in range(self.num_qubits):
                measure_circuit.measure(q, q)

            # Execute circuit
            if self.simulator is not None:
                transpiled = transpile(measure_circuit, self.simulator)
                result = self.simulator.run(transpiled, shots=shots).result()

                if result.success:
                    counts = dict(result.get_counts())

                    # Store in measurement history
                    if hasattr(self, 'measurement_history'):
                        self.measurement_history.append({
                            'time': len(self.measurement_history),
                            'counts': counts,
                            'shots': shots
                        })

                    return counts
                else:
                    logger.error(f"Measurement failed: {result.status}")
                    return {}
            else:
                logger.error("Simulator not available for measurement")
                return {}

        except Exception as e:
            logger.error(f"Error in state measurement: {e}")
            return {}

    def compute_overlap(self, target_state: Union[Statevector, np.ndarray]) -> float:
        """
        Compute overlap between current state and target state.

        Args:
            target_state: Target quantum state

        Returns:
            Fidelity (overlap squared) between states
        """
        try:
            # Ensure current statevector is valid
            if self.statevector is None:
                logger.warning("Current statevector is None. Cannot compute overlap.")
                return 0.0

            # Convert current state to numpy array
            if isinstance(self.statevector, Statevector):
                current_array = np.array(self.statevector.data, dtype=np.complex128)
            else:
                current_array = np.array(self.statevector, dtype=np.complex128)

            # Convert target state to numpy array
            if isinstance(target_state, Statevector):
                target_array = np.array(target_state.data, dtype=np.complex128)
            else:
                target_array = np.array(target_state, dtype=np.complex128)

            # Ensure dimensions match
            if current_array.shape != target_array.shape:
                logger.error(f"State dimensions do not match: {current_array.shape} vs {target_array.shape}")
                return 0.0

            # Compute overlap
            overlap = np.abs(np.vdot(current_array, target_array)) ** 2
            return float(overlap)

        except Exception as e:
            logger.error(f"Error computing state overlap: {e}")
            return 0.0

    def apply_custom_unitary(self, unitary_matrix: np.ndarray, qubits: List[int]) -> bool:
        """
        Apply custom unitary operation to specified qubits.

        Args:
            unitary_matrix: Unitary matrix to apply
            qubits: Qubits to apply operation to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate qubits
            if not qubits or not all(0 <= q < self.num_qubits for q in qubits):
                logger.error(f"Invalid qubits: {qubits}")
                return False

            # Validate unitary matrix
            matrix_size = 2 ** len(qubits)
            if unitary_matrix.shape != (matrix_size, matrix_size):
                logger.error(f"Unitary matrix size mismatch: {unitary_matrix.shape} vs ({matrix_size}, {matrix_size})")
                return False

            # Create custom unitary circuit
            custom_circuit = QuantumCircuit(self.num_qubits)

            # Initialize to current state
            if isinstance(self.statevector, np.ndarray):
                current_state = Statevector(self.statevector)
            else:
                current_state = self.statevector

            custom_circuit.initialize(list(current_state.data), range(self.num_qubits))

            # Apply custom unitary
            # For single qubit unitary
            if len(qubits) == 1:
                custom_circuit.unitary(unitary_matrix, qubits[0], label='custom')
            # For multi-qubit unitary
            else:
                custom_circuit.unitary(unitary_matrix, qubits, label='custom')

            # Add state saving instruction
            save_sv = SaveStatevectorWrapper(self.num_qubits)
            save_sv.apply(custom_circuit)

            # Execute circuit
            if self.simulator is not None:
                transpiled = transpile(custom_circuit, self.simulator)
                result = self.simulator.run(transpiled, noise_model=self.noise_model).result()

                if 'statevector' in result.data():
                    self.statevector = result.get_statevector()
                    self.update_phase_coherence()
                    self._update_state_metrics()
                    return True
                else:
                    logger.error("No statevector returned after custom unitary")
                    return self._attempt_state_recovery()
            else:
                logger.error("Simulator not available for custom unitary")
                return False

        except Exception as e:
            logger.error(f"Error applying custom unitary: {e}")
            return False

    def get_evolution_history(self) -> List[Dict]:
        """
        Get the history of quantum state evolution.

        Returns:
            List of state evolution records
        """
        try:
            if not hasattr(self, 'evolution_history') or not self.evolution_history:
                return []

            # Create a simplified version for return (avoiding statevector copies)
            history = []
            for entry in self.evolution_history:
                history_entry = {k: v for k, v in entry.items() if k != 'statevector'}
                history.append(history_entry)

            return history

        except Exception as e:
            logger.error(f"Error retrieving evolution history: {e}")
            return []

    def is_stable(self) -> bool:
        """
        Check if the quantum state is stable based on recent history.

        Returns:
            True if state is stable, False otherwise
        """
        try:
            # Check if we have enough history
            if not hasattr(self, 'coherence_history') or len(self.coherence_history) < 10:
                return True  # Assume stable if not enough history

            # Calculate stability metrics
            recent_coherence = list(self.coherence_history)[-10:]
            coherence_variance = np.var(recent_coherence)

            # A state is stable if coherence variance is small
            stability_threshold = 0.01
            return coherence_variance < stability_threshold

        except Exception as e:
            logger.error(f"Error checking stability: {e}")
            return False

    def get_state_representation(self) -> Dict[str, Any]:
        """
        Get a comprehensive representation of the current quantum state.

        Returns:
            Dictionary with state details
        """
        try:
            # Get basic metrics
            metrics = self.get_quantum_metrics()

            # Calculate additional state representations
            if isinstance(self.statevector, Statevector):
                probabilities = np.abs(np.array(self.statevector.data))**2
            else:
                probabilities = np.abs(self.statevector)**2

            # Find most probable states
            top_states = []
            for i, prob in enumerate(probabilities):
                if prob > 0.01:  # Only include states with significant probability
                    state_label = format(i, f'0{self.num_qubits}b')
                    top_states.append({
                        'state': state_label,
                        'probability': float(prob),
                        'amplitude': float(np.abs(self.statevector[i])),
                        'phase': float(np.angle(self.statevector[i]))
                    })

            # Sort by probability
            top_states.sort(key=lambda x: x['probability'], reverse=True)

            # Return comprehensive state representation
            return {
                'metrics': metrics,
                'top_states': top_states[:8],  # Limit to top 8 states
                'num_qubits': self.num_qubits,
                'coherence': float(self.phase_coherence),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error getting state representation: {e}")
            return {
                'metrics': {},
                'top_states': [],
                'num_qubits': self.num_qubits,
                'coherence': float(self.minimum_coherence),
                'timestamp': time.time()
            }


# Main execution for testing
if __name__ == "__main__":
    # Create a quantum state
    print("Initializing quantum state...")
    quantum_state = EnhancedQuantumState(num_qubits=4)

    # Apply some gates
    print("\nApplying Hadamard gates...")
    for q in range(quantum_state.num_qubits):
        quantum_state.apply_gate('h', [q])

    # Apply a phase shift
    print("\nApplying phase shift...")
    quantum_state.apply_phase_shift(0.5)

    # Get and print metrics
    metrics = quantum_state.get_quantum_metrics()
    print("\nQuantum Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Measure the state
    print("\nMeasuring quantum state...")
    measurement = quantum_state.measure_state(shots=1024)
    print("\nMeasurement Results:")
    for state, count in measurement.items():
        print(f"  |{state}⟩: {count} ({count/1024:.4f})")
