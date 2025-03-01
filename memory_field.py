"""
Memory Field Module for Émile-2 Simulation
------------------------------------------
Implements hierarchical memory structures and ontological field dynamics
for emergent behavior.
"""
import logging
import time
import numpy as np
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.memory_field")

# Import from other modules
from utilities import (
    MOMENTUM_DECAY,
    update_momentum,
    compute_phase_coherence,
    compute_context_similarity
)
from data_classes import SurplusState

# =============================================================================
# RecursiveDistinctionMemory
# =============================================================================
class RecursiveDistinctionMemory:
    """
    Hierarchical memory system for storing and retrieving quantum state information.

    Implements multiple memory levels with different temporal scales and prioritization,
    along with memory consolidation and retrieval mechanisms.
    """
    def __init__(self, max_size: int = 10000, hierarchy_levels: int = 4):
        """
        Initialize the recursive memory structure.

        Args:
            max_size: Maximum number of entries in memory
            hierarchy_levels: Number of hierarchical levels in memory
        """
        # Create a list of deques, one per hierarchy level
        self.memory = [deque(maxlen=max(max_size // hierarchy_levels, 100)) for _ in range(hierarchy_levels)]

        # Memory configuration
        self.importance_thresholds = [0.3, 0.5, 0.7, 0.9][:hierarchy_levels]  # thresholds per level
        self.retrieval_weights = [1.0, 0.8, 0.6, 0.4][:hierarchy_levels]      # weights for retrieval
        self.consolidation_counters = [0] * hierarchy_levels
        self.consolidation_threshold = 100

        # Memory statistics and tracking
        self.access_counts = [0] * hierarchy_levels
        self.consolidation_history = []
        self.retrieval_stats = {'hits': 0, 'misses': 0}
        self.memory_stability = 1.0
        self.consolidation_momentum = np.zeros(hierarchy_levels)

        # Metrics
        self.metrics = {
            'total_entries': 0,
            'last_importance': 0.0,
            'last_retrieval_time': 0.0,
            'memory_utilization': [0.0] * hierarchy_levels
        }

        logger.info(f"Initialized RecursiveDistinctionMemory with {hierarchy_levels} levels")

    def store(self, phase_coherence: float, distinction_level: float,
              surplus_state: Any, importance: Optional[float] = None) -> bool:
        """
        Store information in memory with hierarchical organization.

        Args:
            phase_coherence: Quantum phase coherence
            distinction_level: Current distinction level
            surplus_state: SurplusState object or dict of surplus values
            importance: Optional importance score (calculated if None)

        Returns:
            True if storage successful, False otherwise
        """
        try:
            # Process surplus_state: convert SurplusState to dict if needed
            if isinstance(surplus_state, SurplusState):
                surplus_values = surplus_state.values
            elif isinstance(surplus_state, dict):
                surplus_values = surplus_state
            else:
                logger.warning(f"Invalid surplus_state type: {type(surplus_state)}, using default")
                surplus_values = {
                    'basal': 1.0,
                    'cognitive': 1.0,
                    'predictive': 1.0,
                    'ontological': 1.0
                }

            # Create a deep copy of surplus values to prevent shared references
            surplus_copy = {k: float(v) for k, v in surplus_values.items()}

            # Ensure values are of correct type
            try:
                phase_coherence = float(phase_coherence)
                distinction_level = float(distinction_level)
            except (TypeError, ValueError) as e:
                logger.warning(f"Type conversion error: {e}, using default values")
                phase_coherence = 0.5
                distinction_level = 0.5

            # Compute importance if not provided
            if importance is None:
                importance = self._calculate_importance(phase_coherence, distinction_level, surplus_copy)

            # Create memory entry
            entry = {
                'phase_coherence': phase_coherence,
                'distinction_level': distinction_level,
                'surplus_state': surplus_copy,
                'importance': float(importance),
                'timestamp': time.time(),
                'stability': float(self.memory_stability)
            }

            # Store entry in appropriate levels based on importance
            stored_levels = []
            for level in range(len(self.memory)):
                if importance > self.importance_thresholds[level]:
                    self.memory[level].append(entry)
                    self.access_counts[level] += 1
                    stored_levels.append(level)

            # Update metrics
            self.metrics['total_entries'] += 1
            self.metrics['last_importance'] = importance
            for level in range(len(self.memory)):
                self.metrics['memory_utilization'][level] = len(self.memory[level]) / self.memory[level].maxlen

            # Check for consolidation
            self._check_consolidation()

            logger.debug(f"Stored memory entry with importance {importance:.4f} in levels {stored_levels}")
            return True

        except Exception as e:
            logger.error(f"Error storing in memory: {e}")
            return False

    def retrieve_recent(self, steps: int = 10, level: int = 0) -> List[Tuple[float, float, Dict[str, float]]]:
        """
        Retrieve recent memory entries from a specified level.

        Adapts to distinction trends by using deeper memory levels when distinction is volatile.

        Args:
            steps: Number of recent entries to retrieve
            level: Memory hierarchy level (0 is most recent)

        Returns:
            List of (phase_coherence, distinction_level, surplus_state) tuples
        """
        try:
            # Record retrieval time
            self.metrics['last_retrieval_time'] = time.time()

            # Validate level index
            if level >= len(self.memory) or not self.memory[level]:
                logger.warning(f"Invalid level {level} or empty memory")
                self.retrieval_stats['misses'] += 1
                default_entry = (1.0, 1.0, {'basal': 1.0, 'cognitive': 1.0, 'predictive': 1.0, 'ontological': 1.0})
                return [default_entry] * steps

            # Convert deque to list before slicing for better safety
            memory_list = list(self.memory[level])

            # Analyze distinction volatility
            if len(memory_list) < steps:
                distinction_values = [entry['distinction_level'] for entry in memory_list]
            else:
                distinction_values = [entry['distinction_level'] for entry in memory_list[-steps:]]

            # Compute distinction variance to detect volatility
            distinction_variance = np.var(distinction_values) if len(distinction_values) > 1 else 0.0

            # If distinction variance is high, retrieve from deeper levels
            if distinction_variance > 0.15 and level < len(self.memory) - 1:
                logger.debug(f"High distinction variance {distinction_variance:.4f}, retrieving from level {level+1}")
                return self.retrieve_recent(steps, level + 1)

            # Retrieve most recent entries
            entries = memory_list[-steps:]

            # Pad with last entry if needed
            if len(entries) < steps and entries:
                entries.extend([entries[-1]] * (steps - len(entries)))

            # Update statistics
            self.access_counts[level] += 1
            self.retrieval_stats['hits'] += 1

            # Extract and return the required tuple format
            result = [(
                entry['phase_coherence'],
                entry['distinction_level'],
                entry['surplus_state']
            ) for entry in entries]

            return result

        except Exception as e:
            logger.error(f"Error retrieving from memory: {e}")
            # Return safe defaults
            default_entry = (1.0, 1.0, {'basal': 1.0, 'cognitive': 1.0, 'predictive': 1.0, 'ontological': 1.0})
            return [default_entry] * steps

    def retrieve_by_similarity(self, current_state: Dict[str, float],
                              threshold: float = 0.8) -> List[Dict]:
        """
        Retrieve memories similar to the current state.

        Args:
            current_state: Dictionary representing the current state
            threshold: Similarity threshold for retrieval

        Returns:
            List of similar memory entries with added similarity and level information
        """
        try:
            similar_memories = []

            # Validate current state
            if not isinstance(current_state, dict) or not current_state:
                logger.warning("Invalid current state")
                return []

            # Search all memory levels
            for level in range(len(self.memory)):
                for entry in self.memory[level]:
                    # Skip entries with missing surplus_state
                    if 'surplus_state' not in entry or not isinstance(entry['surplus_state'], dict):
                        continue

                    # Compute similarity
                    similarity = compute_context_similarity(current_state, entry['surplus_state'])

                    # Filter by threshold
                    if similarity > threshold:
                        # Add similarity and level info to the entry
                        similar_entry = entry.copy()
                        similar_entry['similarity'] = similarity
                        similar_entry['level'] = level
                        similar_memories.append(similar_entry)

            # Sort results by similarity and importance
            similar_memories.sort(key=lambda x: (x['similarity'], x.get('importance', 0.0)), reverse=True)

            logger.debug(f"Retrieved {len(similar_memories)} similar memories")
            return similar_memories

        except Exception as e:
            logger.error(f"Error in similarity retrieval: {e}")
            return []

    def _calculate_importance(self, phase_coherence: float,
                              distinction_level: float,
                              surplus_state: Dict[str, float]) -> float:
        """
        Calculate an importance score for a memory entry.

        Args:
            phase_coherence: Quantum phase coherence
            distinction_level: Distinction level
            surplus_state: Dictionary of surplus values

        Returns:
            Importance score (0.0 to 1.0)
        """
        try:
            # Define weights for different components
            coherence_weight = 0.3
            distinction_weight = 0.3
            surplus_weight = 0.4

            # Process surplus factor (lower surplus means higher importance)
            surplus_values = list(surplus_state.values())
            surplus_factor = 1.0 / (1.0 + np.mean(surplus_values))

            # Add stability bonus
            stability_bonus = 0.1 * self.memory_stability

            # Calculate combined importance score
            importance = (
                coherence_weight * phase_coherence +
                distinction_weight * distinction_level +
                surplus_weight * surplus_factor +
                stability_bonus
            )

            # Ensure result is in valid range
            return float(np.clip(importance, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.5

    def _compute_state_similarity(self, state1: Dict[str, float],
                                  state2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two state dictionaries.

        Args:
            state1: First state dictionary
            state2: Second state dictionary

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Find common keys
            common_keys = set(state1.keys()) & set(state2.keys())
            if not common_keys:
                return 0.0

            # Build vectors from common keys
            vec1 = []
            vec2 = []
            for key in common_keys:
                try:
                    vec1.append(float(state1[key]))
                    vec2.append(float(state2[key]))
                except (TypeError, ValueError):
                    # Skip keys with invalid values
                    continue

            # Handle empty vectors
            if not vec1:
                return 0.0

            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate norms
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            # Handle zero norms
            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            cosine_sim = dot_product / (norm1 * norm2)

            # Scale to [0, 1] range
            similarity = (cosine_sim + 1) / 2.0

            # Weight by proportion of common keys
            total_keys = max(len(state1), len(state2))
            key_ratio = len(common_keys) / total_keys if total_keys > 0 else 1.0

            return float(similarity * key_ratio)

        except Exception as e:
            logger.error(f"Error computing state similarity: {e}")
            return 0.0

    def _check_consolidation(self) -> None:
        """
        Check if memory consolidation is needed and perform if necessary.

        Adjusts consolidation frequency based on surplus fluctuations.
        """
        try:
            for level in range(len(self.memory) - 1):  # Skip last level
                self.consolidation_counters[level] += 1

                # Adjust threshold based on average surplus in this level
                if self.memory[level]:
                    avg_surplus = np.mean([
                        sum(entry['surplus_state'].values())
                        for entry in self.memory[level]
                    ])
                    adaptive_threshold = self.consolidation_threshold * (
                        1.2 if avg_surplus > 2.0 else 0.8
                    )
                else:
                    adaptive_threshold = self.consolidation_threshold

                # Consolidate if threshold reached
                if self.consolidation_counters[level] >= adaptive_threshold:
                    self._consolidate_memory(level)
                    self.consolidation_counters[level] = 0

        except Exception as e:
            logger.error(f"Error checking consolidation: {e}")

    def _consolidate_memory(self, level: int) -> None:
        """
        Consolidate memories from one level to the next.

        Args:
            level: Level to consolidate from (to level+1)
        """
        try:
            # Skip if this is already the highest level
            if level >= len(self.memory) - 1:
                return

            # Get memories at this level
            memories = list(self.memory[level])
            if not memories:
                return

            # Sort memories by importance and stability
            memories.sort(key=lambda x: (x['importance'], x.get('stability', 0.0)), reverse=True)

            # Track redundancy (how often the same distinction state appears)
            distinction_counts = {}
            for entry in memories:
                # Create key from coherence and distinction (rounded to reduce noise)
                distinction_state = (
                    round(entry['phase_coherence'], 2),
                    round(entry['distinction_level'], 2)
                )
                distinction_counts[distinction_state] = distinction_counts.get(distinction_state, 0) + 1

            # Remove redundant distinction states (keep only a few of each)
            filtered_memories = []
            for entry in memories:
                distinction_state = (
                    round(entry['phase_coherence'], 2),
                    round(entry['distinction_level'], 2)
                )

                if distinction_counts[distinction_state] > 5:
                    # Reduce count for next entry with this state
                    distinction_counts[distinction_state] -= 1
                else:
                    # Keep entries with unique or rare states
                    filtered_memories.append(entry)

            # Calculate how many memories to consolidate
            consolidation_count = max(
                int(len(filtered_memories) * 0.25 * (1.0 + 0.1 * self.consolidation_momentum[level])),
                1
            )

            # Select top memories for consolidation
            top_memories = filtered_memories[:consolidation_count]

            # Move selected memories to next level if important enough
            consolidated_count = 0
            for memory in top_memories:
                if memory['importance'] > self.importance_thresholds[level + 1]:
                    self.memory[level + 1].append(memory)
                    consolidated_count += 1

            # Record consolidation event
            self.consolidation_history.append({
                'level': level,
                'consolidated_count': consolidated_count,
                'timestamp': time.time(),
                'momentum': self.consolidation_momentum[level]
            })

            logger.debug(f"Consolidated {consolidated_count} memories from level {level} to {level+1}")

        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage and performance.

        Returns:
            Dictionary of memory statistics
        """
        try:
            # Calculate memory statistics
            stats = {
                'memory_usage': [len(m) for m in self.memory],
                'memory_capacity': [m.maxlen for m in self.memory],
                'usage_percentage': [len(m) / m.maxlen if m.maxlen else 0 for m in self.memory],
                'access_counts': self.access_counts.copy(),
                'retrieval_hits': self.retrieval_stats['hits'],
                'retrieval_misses': self.retrieval_stats['misses'],
                'hit_rate': (self.retrieval_stats['hits'] /
                             (self.retrieval_stats['hits'] + self.retrieval_stats['misses'])
                             if (self.retrieval_stats['hits'] + self.retrieval_stats['misses']) > 0 else 0),
                'consolidation_events': len(self.consolidation_history),
                'memory_stability': self.memory_stability,
                'consolidation_momentum': self.consolidation_momentum.tolist(),
                'avg_importance': []
            }

            # Calculate average importance per level
            for level in range(len(self.memory)):
                if self.memory[level]:
                    avg_imp = np.mean([m['importance'] for m in self.memory[level]])
                    stats['avg_importance'].append(float(avg_imp))
                else:
                    stats['avg_importance'].append(0.0)

            # Add recent consolidation events
            if self.consolidation_history:
                recent_events = self.consolidation_history[-5:]
                stats['recent_consolidations'] = [
                    {
                        'level': event['level'],
                        'count': event['consolidated_count'],
                        'time_ago': time.time() - event['timestamp']
                    }
                    for event in recent_events
                ]

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    def clear_memory(self) -> None:
        """Clear all memory levels."""
        try:
            for level in range(len(self.memory)):
                self.memory[level].clear()

            # Reset counters
            self.consolidation_counters = [0] * len(self.memory)
            self.access_counts = [0] * len(self.memory)
            self.retrieval_stats = {'hits': 0, 'misses': 0}

            logger.info(f"Cleared all memory levels")

        except Exception as e:
            logger.error(f"Error clearing memory: {e}")

    def prune_by_age(self, max_age_seconds: float) -> int:
        """
        Remove memories older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of memories pruned
        """
        try:
            current_time = time.time()
            pruned_count = 0

            for level in range(len(self.memory)):
                original_size = len(self.memory[level])

                # Filter memories by age
                self.memory[level] = deque(
                    entry for entry in self.memory[level]
                    if current_time - entry['timestamp'] <= max_age_seconds
                )

                # Count pruned entries
                pruned_count += original_size - len(self.memory[level])

            logger.info(f"Pruned {pruned_count} memories older than {max_age_seconds} seconds")
            return pruned_count

        except Exception as e:
            logger.error(f"Error pruning memory by age: {e}")
            return 0

# =============================================================================
# OntologicalField
# =============================================================================
class OntologicalField:
    """
    Ontological field with sophisticated resistance and adaptation mechanisms.

    Implements field dynamics that influence—and are influenced by—the agent's
    quantum distinctions, representing the environment that the agent interacts with.
    """
    def __init__(self, field_size: int = 500,
                 resistance_factor: float = 0.02,
                 adaptation_rate: float = 0.01):
        """
        Initialize the ontological field.

        Args:
            field_size: Size of the field array
            resistance_factor: Base factor for field resistance
            adaptation_rate: Rate of field adaptation to agent states
        """
        # Initialize field as uniform random array
        self.field = np.random.uniform(0, 1, size=field_size)

        # Convert constants to dynamic instance variables
        self.resistance_factor = resistance_factor
        self.adaptation_rate = adaptation_rate
        self.field_threshold = 0.1  # Default threshold for field updates
        self.momentum_decay = 0.7  # From MOMENTUM_DECAY

        # Additional parameters for dynamic adaptation
        self.min_adaptation_rate = 0.001
        self.max_adaptation_rate = 0.05
        self.min_resistance_factor = 0.005
        self.max_resistance_factor = 0.1
        self.constants_adaptation_rate = 0.01

        # Momentum for constant updates
        self.adaptation_rate_momentum = 0.0
        self.resistance_factor_momentum = 0.0
        self.field_threshold_momentum = 0.0

        # Track historical data
        self.field_history = deque(maxlen=1000)
        self.resistance_history = deque(maxlen=1000)
        self.adaptation_history = deque(maxlen=1000)
        self.constants_history = deque(maxlen=100)

        # Dynamic parameters
        self.field_momentum = np.zeros(field_size)
        self.field_gradient = np.zeros(field_size)
        self.stability_factor = 1.0
        self.coherence_coupling = 0.0

        # Momentum tracking
        self.resistance_momentum = 0.0
        self.adaptation_momentum = 0.0

        # Statistics tracking
        self.stats = {
            'mean_resistance': [],
            'field_stability': [],
            'adaptation_events': 0,
            'coherence_history': []
        }

        # Record initial constants
        self._record_constants_state("initialization")

        logger.info(f"Initialized OntologicalField with size {field_size}")

    def _record_constants_state(self, event_type: str) -> None:
        """Record the current state of all dynamic constants for tracking changes over time."""
        try:
            constants_state = {
                'resistance_factor': self.resistance_factor,
                'adaptation_rate': self.adaptation_rate,
                'field_threshold': self.field_threshold,
                'momentum_decay': self.momentum_decay,
                'event_type': event_type,
                'timestamp': time.time()
            }
            self.constants_history.append(constants_state)
        except Exception as e:
            logger.error(f"Error recording constants state: {e}")

    def update_dynamic_constants(self,
                              distinction_level: float,
                              distinction_change: float,
                              field_stability: float,
                              quantum_coupling: float = 0.5) -> None:
        """
        Update dynamic constants based on system performance and behavior.

        Args:
            distinction_level: Current distinction level
            distinction_change: Magnitude of recent distinction change
            field_stability: Current field stability
            quantum_coupling: Quantum coupling strength (default 0.5)
        """
        try:
            # Store previous values for tracking changes
            prev_adaptation_rate = self.adaptation_rate
            prev_resistance_factor = self.resistance_factor

            # 1. Update adaptation rate based on distinction change and field stability
            # If distinction is changing rapidly, we may need faster adaptation
            if distinction_change > 0.1 and field_stability > 0.5:
                # Calculate adaptation rate adjustment
                adaptation_adjustment = 0.002 * self.constants_adaptation_rate * distinction_change

                # Update momentum
                self.adaptation_rate_momentum = update_momentum(
                    self.adaptation_rate_momentum,
                    adaptation_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.adaptation_rate += adaptation_adjustment + 0.1 * self.adaptation_rate_momentum
                self.adaptation_rate = np.clip(self.adaptation_rate, self.min_adaptation_rate, self.max_adaptation_rate)

                logger.info(f"Increased adaptation_rate to {self.adaptation_rate:.4f} based on distinction change")

            # If system is stable with minimal changes, can reduce adaptation rate
            elif distinction_change < 0.05 and field_stability > 0.8:
                # Calculate adaptation rate adjustment
                adaptation_adjustment = -0.001 * self.constants_adaptation_rate

                # Update momentum
                self.adaptation_rate_momentum = update_momentum(
                    self.adaptation_rate_momentum,
                    adaptation_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.adaptation_rate += adaptation_adjustment + 0.1 * self.adaptation_rate_momentum
                self.adaptation_rate = np.clip(self.adaptation_rate, self.min_adaptation_rate, self.max_adaptation_rate)

                logger.info(f"Decreased adaptation_rate to {self.adaptation_rate:.4f} based on stability")

            # 2. Update resistance factor based on field stability and quantum coupling
            # If field is unstable, increase resistance to slow down changes
            if field_stability < 0.4:
                # Calculate resistance factor adjustment
                resistance_adjustment = 0.002 * self.constants_adaptation_rate * (1.0 - field_stability)

                # Update momentum
                self.resistance_factor_momentum = update_momentum(
                    self.resistance_factor_momentum,
                    resistance_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.resistance_factor += resistance_adjustment + 0.1 * self.resistance_factor_momentum
                self.resistance_factor = np.clip(self.resistance_factor, self.min_resistance_factor, self.max_resistance_factor)

                logger.info(f"Increased resistance_factor to {self.resistance_factor:.4f} based on instability")

            # If quantum coupling is high and field is stable, can reduce resistance
            elif quantum_coupling > 0.7 and field_stability > 0.7:
                # Calculate resistance factor adjustment
                resistance_adjustment = -0.001 * self.constants_adaptation_rate * quantum_coupling

                # Update momentum
                self.resistance_factor_momentum = update_momentum(
                    self.resistance_factor_momentum,
                    resistance_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.resistance_factor += resistance_adjustment + 0.1 * self.resistance_factor_momentum
                self.resistance_factor = np.clip(self.resistance_factor, self.min_resistance_factor, self.max_resistance_factor)

                logger.info(f"Decreased resistance_factor to {self.resistance_factor:.4f} based on quantum coupling")

            # 3. Update field threshold based on stability and distinction level
            # If field is stable, can use a higher threshold to prevent unnecessary updates
            if field_stability > 0.8:
                # Calculate field threshold adjustment
                threshold_adjustment = 0.005 * self.constants_adaptation_rate * field_stability

                # Update momentum
                self.field_threshold_momentum = update_momentum(
                    self.field_threshold_momentum,
                    threshold_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.field_threshold += threshold_adjustment + 0.05 * self.field_threshold_momentum
                self.field_threshold = np.clip(self.field_threshold, 0.05, 0.2)

                logger.info(f"Increased field_threshold to {self.field_threshold:.4f} based on stability")

            # If distinction is extreme, lower threshold to ensure field responds
            elif abs(distinction_level - 0.5) > 0.3:
                # Calculate field threshold adjustment
                threshold_adjustment = -0.005 * self.constants_adaptation_rate

                # Update momentum
                self.field_threshold_momentum = update_momentum(
                    self.field_threshold_momentum,
                    threshold_adjustment,
                    decay=self.momentum_decay
                )

                # Apply adjustment with momentum
                self.field_threshold += threshold_adjustment + 0.05 * self.field_threshold_momentum
                self.field_threshold = np.clip(self.field_threshold, 0.05, 0.2)

                logger.info(f"Decreased field_threshold to {self.field_threshold:.4f} based on extreme distinction")

            # Record significant constant changes
            if (abs(self.adaptation_rate - prev_adaptation_rate) > 0.002 or
                abs(self.resistance_factor - prev_resistance_factor) > 0.002):
                self._record_constants_state("significant_update")

        except Exception as e:
            logger.error(f"Error updating field dynamic constants: {e}")

    def resistance(self, agent_distinction: float) -> float:
        """
        Calculate the field resistance based on the agent's distinction.

        Uses a combination of mean field value, gradient, momentum, and
        coherence coupling to determine resistance.

        Args:
            agent_distinction: The agent's current distinction level

        Returns:
            Field resistance value (0.0 to 1.0)
        """
        try:
            # Ensure agent_distinction is within valid range
            agent_distinction = float(np.clip(agent_distinction, 0.0, 1.0))

            # Calculate base resistance from field-distinction difference
            base_resistance = abs(np.mean(self.field) - agent_distinction)

            # Calculate smooth resistance using historical values
            if len(self.resistance_history) >= 10:
                past_resistance_avg = np.mean(list(self.resistance_history)[-10:])
            else:
                past_resistance_avg = base_resistance

            # Update resistance momentum to retain history of past influence
            self.resistance_momentum = update_momentum(
                self.resistance_momentum,
                base_resistance,
                MOMENTUM_DECAY
            )

            # Calculate final resistance with smoothing and momentum
            resistance = (
                (0.6 * past_resistance_avg + 0.4 * base_resistance) *
                self.resistance_factor *
                (1.0 + 0.1 * self.resistance_momentum)
            )

            # Store in history
            self.resistance_history.append(resistance)

            # Ensure valid range
            resistance = float(np.clip(resistance, 0, 1))

            logger.debug(f"Field resistance: {resistance:.4f} (base: {base_resistance:.4f})")
            return resistance

        except Exception as e:
            logger.error(f"Error calculating field resistance: {e}")
            return self.resistance_factor  # Return default value on error

    def adapt_to_agent(self, agent_distinction: float, quantum_coupling: float = 1.0,
                  field_threshold: Optional[float] = None, excess_stability: float = 0.0) -> None:
        """
        Adapt the field to the agent's state with stability-aware dynamics.

        Args:
            agent_distinction: The agent's current distinction level
            quantum_coupling: The quantum coupling strength
            field_threshold: The threshold for field updates (uses instance value if None)
            excess_stability: Amount of excess stability potential to consider
        """
        try:
            # Validate inputs
            agent_distinction = float(np.clip(agent_distinction, 0.0, 1.0))
            quantum_coupling = float(np.clip(quantum_coupling, 0.0, 1.0))

            # Use instance field_threshold if none provided
            if field_threshold is None:
                field_threshold = self.field_threshold

            # Track distinction stability - prevent adaptation if distinction hasn't meaningfully changed
            if len(self.stats['coherence_history']) >= 5:
                recent_coherence = np.mean(self.stats['coherence_history'][-5:])
            else:
                recent_coherence = 0.5  # Default mid-value

            # Calculate distinction change magnitude
            distinction_change = abs(agent_distinction - recent_coherence)

            # Consider excess stability when determining whether to skip adaptation
            skip_threshold = 0.05
            if excess_stability > 0:
                # Lower the skip threshold when excess stability exists
                skip_threshold *= max(0.5, 1.0 - excess_stability)

            # Skip adaptation if distinction hasn't shifted enough
            if distinction_change < skip_threshold:
                logger.debug("Skipping field adaptation: minimal distinction change")
                return

            # Calculate adaptation strength - using dynamic adaptation_rate
            adaptation_strength = self.adaptation_rate * quantum_coupling

            # Enhance adaptation strength with excess stability
            if excess_stability > 0:
                adaptation_boost = 1.0 + (excess_stability * 2.0)
                adaptation_strength *= adaptation_boost
                print(f"DEBUG: Boosting field adaptation by factor {adaptation_boost:.4f} due to excess stability")

            # Get stability trend
            stability_trend = np.mean(self.stats['field_stability'][-10:]) if len(self.stats['field_stability']) > 10 else 1.0
            stability_factor = 1.0 - np.clip(stability_trend, 0.5, 1.0)

            # Update adaptation momentum for smoother changes
            self.adaptation_momentum = (
                self.momentum_decay * self.adaptation_momentum +
                (1 - self.momentum_decay) * adaptation_strength
            )

            # Calculate target state and update field
            target_state = agent_distinction
            current_mean = np.mean(self.field)

            # Calculate field updates with momentum
            field_update = (
                (target_state - self.field) * adaptation_strength * stability_factor +
                self.field_momentum * 0.1 +
                0.05 * self.adaptation_momentum
            )

            # Modify field update threshold based on excess stability
            effective_threshold = field_threshold
            if excess_stability > 0:
                # Lower threshold when excess stability exists for more responsive field
                effective_threshold *= max(0.3, 1.0 - excess_stability)
                print(f"DEBUG: Lowering field update threshold from {field_threshold} to {effective_threshold} due to excess stability")

            # Apply updates only where they exceed threshold
            update_mask = np.abs(field_update) > effective_threshold
            self.field[update_mask] += field_update[update_mask]

            # Store previous field and calculate gradient
            prev_field = self.field.copy()
            self.field_gradient = self.field - prev_field

            # Enforce field bounds
            self.field = np.clip(self.field, 0, 1)

            # Update stability factor
            self.stability_factor = self._calculate_stability()

            # Record adaptation
            self.adaptation_history.append({
                'target': target_state,
                'mean_update': float(np.mean(np.abs(field_update))),
                'stability': float(self.stability_factor),
                'adaptation_momentum': float(self.adaptation_momentum),
                'excess_stability': excess_stability,
                'timestamp': time.time()
            })

            # Update statistics
            self.stats['adaptation_events'] += 1
            self.field_history.append(self.field.copy())

            # Update dynamic constants
            self.update_dynamic_constants(
                distinction_level=agent_distinction,
                distinction_change=distinction_change,
                field_stability=self.stability_factor,
                quantum_coupling=quantum_coupling
            )

            logger.debug(
                f"Field adapted: target={target_state:.4f}, "
                f"mean update={np.mean(np.abs(field_update)):.4f}, "
                f"stability={self.stability_factor:.4f}, "
                f"excess_stability={excess_stability:.4f}"
            )

        except Exception as e:
            logger.error(f"Error adapting field: {e}")

    def get_dynamic_constants(self) -> Dict[str, float]:
        """
        Return the current values of all dynamic constants.

        Returns:
            Dictionary of dynamic constants and their current values
        """
        try:
            return {
                'resistance_factor': self.resistance_factor,
                'adaptation_rate': self.adaptation_rate,
                'field_threshold': self.field_threshold,
                'momentum_decay': self.momentum_decay,
                'constants_adaptation_rate': self.constants_adaptation_rate
            }
        except Exception as e:
            logger.error(f"Error getting dynamic constants: {e}")
            return {}

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

    def _calculate_stability(self) -> float:
        """
        Calculate field stability based on recent history and distinction momentum.

        Returns:
            Stability factor (0.0 to 1.0)
        """
        try:
            # Can't calculate stability without history
            if len(self.field_history) < 2:
                return 1.0

            # Track recent field changes and distinction momentum
            recent_changes = []
            distinction_momentum = []

            # Calculate changes for the last 10 history points (or fewer if not available)
            for i in range(1, min(10, len(self.field_history))):
                # Calculate mean absolute change between consecutive field states
                change = np.mean(np.abs(self.field_history[-i] - self.field_history[-i-1]))
                recent_changes.append(change)

                # Track distinction trend changes (mean field value is proxy for distinction)
                distinction_change = abs(
                    np.mean(self.field_history[-i]) -
                    np.mean(self.field_history[-i-1])
                )
                distinction_momentum.append(distinction_change)

            # Handle empty lists (shouldn't happen given the check above)
            if not recent_changes:
                return 1.0

            # Calculate stability metrics
            mean_change = np.mean(recent_changes)
            distinction_trend = np.mean(distinction_momentum) if distinction_momentum else 0.0

            # Calculate stability as inverse of change magnitudes
            stability = 1.0 / (1.0 + mean_change + 0.5 * distinction_trend)

            # Update stability history
            self.stats['field_stability'].append(float(stability))

            return float(np.clip(stability, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating stability: {e}")
            return 1.0  # Return maximum stability on error

    def apply_quantum_influence(self, quantum_metrics: Dict[str, float]) -> None:
        """
        Apply quantum influence to the field dynamics.

        Args:
            quantum_metrics: Dictionary of quantum state metrics
        """
        try:
            # Validate input
            if not isinstance(quantum_metrics, dict):
                logger.warning("Invalid quantum metrics provided")
                return

            # Extract key metrics with defaults
            coherence = quantum_metrics.get('phase_coherence', 0.5)
            entropy = quantum_metrics.get('normalized_entropy', 0.5)

            # Update coherence coupling
            self.coherence_coupling = 0.8 * self.coherence_coupling + 0.2 * coherence

            # Calculate quantum factor - higher coherence and lower entropy means stronger influence
            quantum_factor = coherence * (1.0 - entropy)

            # Create quantum-influenced field modulation
            # This uses a deterministic approach based on the field's existing values
            # rather than pure randomness
            field_indices = np.arange(len(self.field))
            field_modulation = quantum_factor * 0.1 * (
                np.sin(field_indices / len(self.field) * 2 * np.pi + self.coherence_coupling) - 0.5
            )

            # Apply modulation to field
            self.field += field_modulation

            # Update adaptation rate based on quantum influence
            self.adaptation_rate *= (1.0 + 0.1 * (quantum_factor - 0.5))
            self.adaptation_rate = np.clip(self.adaptation_rate, 0.001, 0.1)

            # Track coherence history
            self.stats['coherence_history'].append(coherence)

            # Enforce field bounds
            self.field = np.clip(self.field, 0, 1)

            logger.debug(
                f"Applied quantum influence: factor={quantum_factor:.4f}, "
                f"coherence={coherence:.4f}, entropy={entropy:.4f}"
            )

        except Exception as e:
            logger.error(f"Error applying quantum influence: {e}")

    def get_field_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive metrics about the field state.

        Returns:
            Dictionary containing field metrics
        """
        try:
            # Calculate basic metrics
            metrics = {
                'mean_field': float(np.mean(self.field)),
                'field_variance': float(np.var(self.field)),
                'stability': float(self.stability_factor),
                'coherence_coupling': float(self.coherence_coupling),
                'mean_momentum': float(np.mean(np.abs(self.field_momentum))),
                'mean_gradient': float(np.mean(np.abs(self.field_gradient))),
                'adaptation_rate': float(self.adaptation_rate),
                'resistance_momentum': float(self.resistance_momentum),
                'adaptation_momentum': float(self.adaptation_momentum),
                'field_size': len(self.field),
                'field_min': float(np.min(self.field)),
                'field_max': float(np.max(self.field))
            }

            # Add resistance history metrics if available
            if self.resistance_history:
                metrics['mean_resistance'] = float(np.mean(self.resistance_history))
                metrics['resistance_std'] = float(np.std(self.resistance_history))

            # Add adaptation metrics if available
            if self.adaptation_history:
                # Get most recent adaptations
                recent_adaptations = list(self.adaptation_history)[-100:]
                metrics['recent_adaptation_strength'] = float(
                    np.mean([a['mean_update'] for a in recent_adaptations])
                )
                metrics['mean_stability'] = float(
                    np.mean([a['stability'] for a in recent_adaptations])
                )
                metrics['adaptation_count'] = len(self.adaptation_history)

            # Calculate field entropy as an additional complexity metric
            # Use histogram to approximate probability distribution
            hist, _ = np.histogram(self.field, bins=20, density=True)
            hist = hist / np.sum(hist)  # Ensure normalization
            entropy_val = -np.sum(hist * np.log2(hist + 1e-10))
            metrics['field_entropy'] = float(entropy_val)

            return metrics

        except Exception as e:
            logger.error(f"Error getting field metrics: {e}")
            return {
                'mean_field': 0.5,
                'field_variance': 0.0,
                'stability': 1.0,
                'error': str(e)
            }

    def get_condensed_field(self, sections: int = 10) -> List[float]:
        """
        Get a condensed representation of the field for visualization or analysis.

        Args:
            sections: Number of sections to divide the field into

        Returns:
            List of mean values for each section
        """
        try:
            # Calculate section size
            section_size = max(len(self.field) // sections, 1)

            # Create condensed representation
            condensed = []
            for i in range(0, len(self.field), section_size):
                section = self.field[i:i+section_size]
                condensed.append(float(np.mean(section)))

            return condensed

        except Exception as e:
            logger.error(f"Error getting condensed field: {e}")
            return [0.5] * sections  # Return default values

    def reset(self) -> None:
        """Reset the field to initial random state."""
        try:
            # Re-initialize the field
            self.field = np.random.uniform(0, 1, size=len(self.field))

            # Reset momentum and gradient
            self.field_momentum = np.zeros_like(self.field)
            self.field_gradient = np.zeros_like(self.field)

            # Reset other parameters
            self.stability_factor = 1.0
            self.coherence_coupling = 0.0
            self.resistance_momentum = 0.0
            self.adaptation_momentum = 0.0
            self.adaptation_rate = 0.01

            # Clear histories but keep stats structure
            self.field_history.clear()
            self.resistance_history.clear()
            self.adaptation_history.clear()
            self.stats['coherence_history'] = []
            self.stats['field_stability'] = []

            logger.info("Reset ontological field to initial state")

        except Exception as e:
            logger.error(f"Error resetting field: {e}")

