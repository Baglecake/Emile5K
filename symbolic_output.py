
import random
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

class SymbolicOutput:
    """
    Enhanced symbolic output generator that integrates with quantum and cognitive metrics
    to produce meaningful symbolic expressions representing the system's emergent states.

    The system uses weighted vocabulary selection based on the agent's current
    quantum and cognitive state to generate expressions that reflect the underlying
    computational ontology.
    """
    def __init__(self, vocabulary_size: int = 12):
        # Basic vocabulary with hierarchical organization
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

        # Track historical expressions
        self.expression_history = []
        self.emergence_events = []
        self.pattern_history = deque(maxlen=100)
        self.frequency_analysis = {}

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

        # Tracks the last generated elements for transition analysis
        self.last_descriptor = None
        self.expression_complexity = 1.0

        # Initialize timestamp for real-time tracking
        self.start_time = time.time()

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

    def _calculate_weights(self,
                          surplus: float,
                          distinction: float,
                          coherence: float,
                          entropy: Optional[float] = None,
                          dimensionality: Optional[int] = None) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate vocabulary selection weights based on current metrics.

        Args:
            surplus: Current cognitive surplus level
            distinction: Current distinction level
            coherence: Current phase coherence
            entropy: Optional entropy metric
            dimensionality: Optional detected dimensionality

        Returns:
            Tuple of weights for descriptors, relations, and concepts
        """
        try:
            # Initialize weights
            descriptor_weights = np.ones(len(self.state_descriptors)) / len(self.state_descriptors)
            relation_weights = np.ones(len(self.relations)) / len(self.relations)
            concept_weights = np.ones(len(self.surplus_concepts)) / len(self.surplus_concepts)

            # Adjust based on coherence
            if coherence > self.coherence_thresholds['high']:
                # High coherence: favor structured, aligned, stabilized expressions
                descriptor_weights = np.array([0.1, 0.3, 0.2, 0.1, 0.2, 0.1, 0, 0, 0, 0, 0, 0])
                relation_weights = np.array([0.1, 0, 0.1, 0.1, 0, 0.2, 0.1, 0.2, 0.1, 0.1, 0, 0])
                concept_weights = np.array([0.3, 0.1, 0, 0, 0.2, 0.1, 0.1, 0.2, 0, 0, 0, 0])
            elif coherence > self.coherence_thresholds['medium']:
                # Medium coherence: balanced distribution
                descriptor_weights = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0])
                relation_weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
                concept_weights = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0])
            else:
                # Low coherence: favor flux, entropy, dissolution
                descriptor_weights = np.array([0.3, 0, 0.1, 0.1, 0, 0.3, 0.1, 0, 0.1, 0, 0, 0])
                relation_weights = np.array([0, 0.3, 0.2, 0, 0.2, 0, 0.1, 0, 0.1, 0, 0.1, 0])
                concept_weights = np.array([0, 0.1, 0.3, 0.2, 0, 0, 0.1, 0, 0.2, 0.1, 0, 0])

            # Adjust based on distinction level
            if distinction > self.distinction_thresholds['high']:
                # Favor distinction and emergence concepts
                descriptor_weights[2] *= 2.0  # Distinction
                relation_weights[10] = max(relation_weights[10], 0.2)  # "emerges through"
                concept_weights[4] = max(concept_weights[4], 0.3)  # "emergence"

                # Increase probability of integration for high distinction
                descriptor_weights[11] = max(descriptor_weights[11], 0.2)  # Integration

            elif distinction < self.distinction_thresholds['low']:
                # Favor flux and entropy
                descriptor_weights[0] *= 2.0  # Flux
                concept_weights[2] = max(concept_weights[2], 0.3)  # "entropy"

                # Increase probability of dissolution terms
                relation_weights[1] = max(relation_weights[1], 0.3)  # "dissolves across"

            # Adjust for surplus level
            if surplus > self.surplus_thresholds['high']:
                # High surplus: favor differentiation and expansion
                relation_weights[6] = max(relation_weights[6], 0.3)  # "differentiates from"
                concept_weights[6] = max(concept_weights[6], 0.3)  # "distinction"
                descriptor_weights[3] = max(descriptor_weights[3], 0.2)  # "Recursion"
            elif surplus < self.surplus_thresholds['low']:
                # Low surplus: favor contraction and stability
                relation_weights[4] = max(relation_weights[4], 0.3)  # "contracts into"
                concept_weights[0] = max(concept_weights[0], 0.3)  # "stability"

            # Adjust for entropy if provided
            if entropy is not None:
                if entropy > 0.7:  # High entropy
                    descriptor_weights[0] = max(descriptor_weights[0], 0.3)  # Flux
                    descriptor_weights[5] = max(descriptor_weights[5], 0.3)  # Divergence
                    concept_weights[2] = max(concept_weights[2], 0.3)  # "entropy"
                elif entropy < 0.3:  # Low entropy
                    descriptor_weights[1] = max(descriptor_weights[1], 0.3)  # Equilibrium
                    descriptor_weights[7] = max(descriptor_weights[7], 0.3)  # Coherence

            # Adjust for dimensionality if provided
            if dimensionality is not None:
                if dimensionality > 3:  # Higher dimensions
                    descriptor_weights[11] = max(descriptor_weights[11], 0.4)  # "Integration"
                    relation_weights[11] = max(relation_weights[11], 0.3)  # "transcends"
                    concept_weights[9] = max(concept_weights[9], 0.4)  # "dimensionality"

                    # Extra emphasis on emergence for higher dimensions
                    if dimensionality > 4:
                        concept_weights[4] = max(concept_weights[4], 0.5)  # "emergence"
                        relation_weights[10] = max(relation_weights[10], 0.4)  # "emerges through"

            # Normalize weights
            descriptor_weights = descriptor_weights / np.sum(descriptor_weights)
            relation_weights = relation_weights / np.sum(relation_weights)
            concept_weights = concept_weights / np.sum(concept_weights)

            return descriptor_weights.tolist(), relation_weights.tolist(), concept_weights.tolist()

        except Exception as e:
            print(f"Error calculating expression weights: {e}")
            # Return uniform weights as fallback
            uniform_desc = [1.0/len(self.state_descriptors)] * len(self.state_descriptors)
            uniform_rel = [1.0/len(self.relations)] * len(self.relations)
            uniform_con = [1.0/len(self.surplus_concepts)] * len(self.surplus_concepts)
            return uniform_desc, uniform_rel, uniform_con

    def _generate_expression_components(self,
                                       descriptor_weights: List[float],
                                       relation_weights: List[float],
                                       concept_weights: List[float],
                                       metrics: Dict[str, float]) -> Tuple[str, str, str, Optional[str]]:
        """
        Generate components for a symbolic expression based on weighted vocabularies.

        Args:
            descriptor_weights: Weights for selecting state descriptors
            relation_weights: Weights for selecting relations
            concept_weights: Weights for selecting concepts
            metrics: Dictionary of current system metrics

        Returns:
            Tuple of (descriptor, relation, concept, modifier)
        """
        try:
            # Select components based on weighted probabilities
            descriptor = random.choices(self.state_descriptors, weights=descriptor_weights, k=1)[0]
            relation = random.choices(self.relations, weights=relation_weights, k=1)[0]
            concept = random.choices(self.surplus_concepts, weights=concept_weights, k=1)[0]

            # Determine if we should use a modifier based on complexity
            use_modifier = random.random() < self.expression_complexity * 0.5
            modifier = random.choice(self.modifiers) if use_modifier else None

            # Special case for extreme states
            coherence = metrics.get('coherence', 0.5)
            distinction = metrics.get('distinction', 0.5)

            if coherence > 0.95 and distinction > 0.9:
                descriptor = "Coherent Distinction"
                relation = "stabilizes within"
                concept = "emergent ontology"
                modifier = "systematically"
            elif coherence < 0.1 and distinction < 0.1:
                descriptor = "Entropic Flux"
                relation = "dissolves across"
                concept = "undifferentiated phase space"
                modifier = "chaotically"

            # Update transition statistics for pattern analysis
            self._update_transition_statistics(descriptor)

            return descriptor, relation, concept, modifier

        except Exception as e:
            print(f"Error generating expression components: {e}")
            return "Flux", "aligns with", "stability", None

    def generate_symbolic_expression(self,
                                    surplus: float,
                                    distinction: float,
                                    coherence: float,
                                    entropy: Optional[float] = None,
                                    dimensionality: Optional[int] = None) -> str:
        """
        Generates a symbolic expression based on the system's current metrics.

        Args:
            surplus: Current cognitive surplus level
            distinction: Current distinction level
            coherence: Current phase coherence
            entropy: Optional entropy metric
            dimensionality: Optional detected dimensionality

        Returns:
            A symbolic expression representing the current state
        """
        try:
            # Ensure inputs are proper floats for stability
            surplus = float(np.clip(surplus, 0.1, 10.0))
            distinction = float(np.clip(distinction, 0.0, 1.0))
            coherence = float(np.clip(coherence, 0.0, 1.0))

            # Prepare metrics for component generation
            metrics = {
                'surplus': surplus,
                'distinction': distinction,
                'coherence': coherence,
                'entropy': entropy,
                'dimensionality': dimensionality,
                'time_elapsed': time.time() - self.start_time
            }

            # Calculate vocabulary selection weights
            descriptor_weights, relation_weights, concept_weights = self._calculate_weights(
                surplus, distinction, coherence, entropy, dimensionality
            )

            # Generate expression components
            descriptor, relation, concept, modifier = self._generate_expression_components(
                descriptor_weights, relation_weights, concept_weights, metrics
            )

            # Adapt expression complexity based on system metrics
            self.expression_complexity = min(2.0, 0.5 + 0.5 * coherence + 0.3 * distinction + 0.2 * (surplus / 10.0))

            # Use secondary concepts with a certain probability based on complexity
            use_secondary = random.random() < self.expression_complexity * 0.3

            if use_secondary:
                secondary = random.choice(self.secondary_concepts)
                concept = f"{concept} within {secondary}"

            # Assemble the expression with or without modifier
            if modifier:
                symbolic_expression = f"{descriptor} {modifier} {relation} {concept}."
            else:
                symbolic_expression = f"{descriptor} {relation} {concept}."

            # Store in history with metadata
            expression_entry = {
                'expression': symbolic_expression,
                'components': {
                    'descriptor': descriptor,
                    'relation': relation,
                    'concept': concept,
                    'modifier': modifier,
                    'secondary': secondary if use_secondary else None
                },
                'metrics': metrics.copy(),
                'timestamp': time.time(),
                'complexity': self.expression_complexity
            }

            self.expression_history.append(expression_entry)

            # Update pattern history
            self.pattern_history.append({
                'descriptor': descriptor,
                'relation': relation,
                'concept': concept
            })

            # Update frequency analysis
            self._update_frequency_analysis(descriptor, relation, concept)

            return symbolic_expression

        except Exception as e:
            print(f"Error generating symbolic expression: {e}")
            return "Flux aligns with stability."  # Safe fallback

    def _update_frequency_analysis(self, descriptor: str, relation: str, concept: str):
        """
        Update frequency analysis of expression components.

        Args:
            descriptor: The descriptor used
            relation: The relation used
            concept: The concept used
        """
        try:
            if 'descriptors' not in self.frequency_analysis:
                self.frequency_analysis = {
                    'descriptors': {},
                    'relations': {},
                    'concepts': {}
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

        except Exception as e:
            print(f"Error updating frequency analysis: {e}")

    def handle_post_emergence(self,
                     surplus: float,
                     distinction: float,
                     coherence: float,
                     dimensionality: Optional[int] = None,
                     entropy: Optional[float] = None) -> str:
        """
        Triggers symbolic output generation after dimensional emergence is detected.
        Records emergence event and generates an appropriate symbolic expression.

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

            expression = self.generate_symbolic_expression(
                varied_surplus, varied_distinction, varied_coherence,
                entropy=varied_entropy,
                dimensionality=dimensionality
            )

            # For emergence events, generate a more complex secondary expression
            if len(self.emergence_events) > 1:
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
                        f"across {dimensionality if dimensionality else 'multiple'} dimensions.",

                        f"Dimensional shift to {dimensionality}D reveals {dominant.get('descriptor', 'Emergence')} "
                        f"{random.choice(self.relations)} "
                        f"{random.choice(self.surplus_concepts)}.",

                        f"The {dimensionality}D structure {random.choice(self.modifiers)} "
                        f"{random.choice(self.relations)} "
                        f"{dominant.get('concept', 'ontology')}.",

                        f"Analysis suggests {random.choice(self.modifiers)} {dominant.get('descriptor', 'Distinction')} "
                        f"within the emergent {dimensionality}D domain."
                    ]

                    # Choose one secondary expression randomly
                    follow_up = random.choice(secondary_expressions)
                    expression = f"{expression} {follow_up}"

            return expression

        except Exception as e:
            print(f"Error handling post-emergence: {e}")
            return self.generate_symbolic_expression(surplus, distinction, coherence)

    def analyze_emergence_patterns(self) -> Dict[str, Any]:
        """
        Analyzes patterns in emergence events and generated expressions.
        Returns statistics and patterns detected in the symbolic outputs.

        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            if not self.emergence_events or not self.expression_history:
                return {'patterns': 'Insufficient data for pattern analysis'}

            # Extract metrics from history
            coherence_values = [e['metrics']['coherence'] for e in self.emergence_events]
            distinction_values = [e['metrics']['distinction'] for e in self.emergence_events]

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

                # Find dominant patterns
                dominant_descriptor = max(descriptor_counts.items(), key=lambda x: x[1])[0] if descriptor_counts else None
                dominant_relation = max(relation_counts.items(), key=lambda x: x[1])[0] if relation_counts else None
                dominant_concept = max(concept_counts.items(), key=lambda x: x[1])[0] if concept_counts else None

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

                # Find common sequences in the pattern history
                sequence_patterns = {}
                if len(self.pattern_history) > 3:
                    for i in range(len(self.pattern_history) - 2):
                        seq = (
                            self.pattern_history[i]['descriptor'],
                            self.pattern_history[i+1]['descriptor'],
                            self.pattern_history[i+2]['descriptor']
                        )
                        sequence_patterns[seq] = sequence_patterns.get(seq, 0) + 1

                # Find most common sequence
                common_sequence = max(sequence_patterns.items(), key=lambda x: x[1])[0] if sequence_patterns else None

                # Typical expression
                typical_expression = f"{dominant_descriptor} {dominant_relation} {dominant_concept}."

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

                return {
                    'emergence_count': len(self.emergence_events),
                    'expression_count': len(self.expression_history),
                    'coherence_stability': float(coherence_stability),
                    'distinction_stability': float(distinction_stability),
                    'component_diversity': {
                        'descriptor': float(descriptor_diversity),
                        'relation': float(relation_diversity),
                        'concept': float(concept_diversity),
                        'overall': float((descriptor_diversity + relation_diversity + concept_diversity) / 3)
                    },
                    'dominant_patterns': {
                        'descriptor': dominant_descriptor,
                        'relation': dominant_relation,
                        'concept': dominant_concept
                    },
                    'common_sequence': common_sequence,
                    'typical_expression': typical_expression,
                    'complexity_trend': float(complexity_trend),
                    'transition_entropy': float(transition_entropy),
                    'current_complexity': float(self.expression_complexity)
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

    def get_expression_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the most recent expression history.

        Args:
            limit: Maximum number of expressions to return

        Returns:
            List of recent expression entries
        """
        try:
            if not self.expression_history:
                return []

            recent = self.expression_history[-limit:]
            return recent

        except Exception as e:
            print(f"Error getting expression history: {e}")
            return []

# Example usage
if __name__ == "__main__":
    symbolic_system = SymbolicOutput()
    example_expression = symbolic_system.handle_post_emergence(surplus=1.5, distinction=0.4, coherence=0.8)
    print("Generated Symbolic Expression:", example_expression)
