
"""
Semantic Integration Module for Émile-4 Simulation
--------------------------------------------------
This module integrates the SemanticEnhancedOutput with the existing Émile architecture,
providing bi-directional learning between symbolic expression and quantum/surplus dynamics.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
import json

# Import the SemanticEnhancedOutput class
from semantic_enhanced_output import SemanticEnhancedOutput

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emile4.semantic_integration")

class SemanticIntegrationManager:
    """
    Manages the integration between the quantum/surplus system and the semantic model,
    creating a bi-directional learning loop where:
    1. The quantum/surplus system informs semantic expression generation
    2. Semantic expressions provide feedback to the quantum/surplus system
    """
    def __init__(self,
                 semantic_model_path=None,
                 enable_bidirectional=True,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the semantic integration manager.

        Args:
            semantic_model_path: Path to the semantic model checkpoint (optional)
            enable_bidirectional: Whether to enable bidirectional learning
            device: Device for model computation
        """
        logger.info("Initializing SemanticIntegrationManager")

        # Create semantic output generator
        self.symbolic_output = SemanticEnhancedOutput(
            semantic_model_path=semantic_model_path,
            device=device
        )

        # Integration settings
        self.enable_bidirectional = enable_bidirectional
        self.device = device
        self.semantic_model_loaded = self.symbolic_output.semantic_model is not None

        # Bidirectional learning parameters
        self.feedback_strength = 0.2
        self.semantic_coherence_threshold = 0.7
        self.adaptation_rate = 0.05

        # History tracking
        self.expression_history = []
        self.coherence_history = deque(maxlen=100)
        self.distinction_history = deque(maxlen=100)
        self.integration_metrics = {
            'semantic_coherence': [],
            'quantum_influence': [],
            'feedback_strength': [],
            'expression_complexity': []
        }

        # Statistics
        self.total_expressions = 0
        self.total_feedback_cycles = 0
        self.emergence_events = 0

        # Integration state
        self.last_expression_time = time.time()
        self.last_expression = None
        self.last_expression_components = {}
        self.last_expression_coherence = 0.0

        logger.info(f"Semantic integration initialized. Model loaded: {self.semantic_model_loaded}")

    def generate_symbolic_expression(self,
                                    surplus: float,
                                    distinction: float,
                                    coherence: float,
                                    entropy: Optional[float] = None,
                                    dimensionality: Optional[int] = None) -> str:
        """
        Generate a symbolic expression based on the current system state,
        using the semantic-enhanced output module.

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
            # Generate expression using semantic model
            expression = self.symbolic_output.generate_symbolic_expression(
                surplus=surplus,
                distinction=distinction,
                coherence=coherence,
                entropy=entropy,
                dimensionality=dimensionality
            )

            # Get expression components for feedback loop
            last_entry = self.symbolic_output.expression_history[-1] if self.symbolic_output.expression_history else None

            if last_entry:
                self.last_expression_components = last_entry.get('components', {})
                self.last_expression_coherence = self.symbolic_output._calculate_semantic_coherence(expression)

            # Update tracking
            self.last_expression = expression
            self.last_expression_time = time.time()
            self.total_expressions += 1

            # Store in history with metrics
            self.expression_history.append({
                'expression': expression,
                'metrics': {
                    'surplus': surplus,
                    'distinction': distinction,
                    'coherence': coherence,
                    'entropy': entropy,
                    'dimensionality': dimensionality
                },
                'components': self.last_expression_components,
                'semantic_coherence': self.last_expression_coherence,
                'timestamp': self.last_expression_time
            })

            # Update history collections
            self.coherence_history.append(coherence)
            self.distinction_history.append(distinction)

            # Update integration metrics
            self.integration_metrics['semantic_coherence'].append(self.last_expression_coherence)

            logger.debug(f"Generated expression: {expression}")
            return expression

        except Exception as e:
            logger.error(f"Error generating symbolic expression: {e}")
            return "System integration in flux."  # Safe fallback

    def handle_post_emergence(self,
                          surplus: float,
                          distinction: float,
                          coherence: float,
                          dimensionality: Optional[int] = None,
                          entropy: Optional[float] = None) -> str:
        """
        Handle post-emergence expression generation with semantic enhancement.

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
            # Generate emergent expression
            expression = self.symbolic_output.handle_post_emergence(
                surplus=surplus,
                distinction=distinction,
                coherence=coherence,
                dimensionality=dimensionality,
                entropy=entropy
            )

            # Track emergence event
            self.emergence_events += 1
            self.last_expression = expression
            self.last_expression_time = time.time()

            # Store in history
            self.expression_history.append({
                'expression': expression,
                'type': 'emergence',
                'metrics': {
                    'surplus': surplus,
                    'distinction': distinction,
                    'coherence': coherence,
                    'entropy': entropy,
                    'dimensionality': dimensionality
                },
                'timestamp': self.last_expression_time
            })

            logger.info(f"Generated emergence expression: {expression}")
            return expression

        except Exception as e:
            logger.error(f"Error handling post-emergence: {e}")
            return self.generate_symbolic_expression(surplus, distinction, coherence)

    def calculate_feedback_parameters(self) -> Dict[str, float]:
        """
        Calculate feedback parameters for the bidirectional learning loop.

        Returns:
            Dictionary of feedback parameters
        """
        try:
            # Default parameters
            feedback = {
                'coherence_adjustment': 0.0,
                'distinction_adjustment': 0.0,
                'semantic_coherence': self.last_expression_coherence,
                'feedback_strength': self.feedback_strength
            }

            # Skip if bidirectional learning is disabled
            if not self.enable_bidirectional:
                return feedback

            # Skip if no expression history
            if not self.expression_history:
                return feedback

            # Get semantic coherence from last expression
            semantic_coherence = self.last_expression_coherence

            # Calculate historical averages if available
            coherence_avg = np.mean(list(self.coherence_history)) if self.coherence_history else 0.5
            distinction_avg = np.mean(list(self.distinction_history)) if self.distinction_history else 0.5

            # Calculate adjustments based on semantic coherence
            # Higher coherence means the expression is semantically consistent
            if semantic_coherence > self.semantic_coherence_threshold:
                # Encourage stability in high-coherence states
                coherence_adjustment = self.adaptation_rate * (semantic_coherence - self.semantic_coherence_threshold)
                distinction_adjustment = 0.0  # Maintain distinction
            else:
                # Encourage exploration in low-coherence states
                coherence_adjustment = -self.adaptation_rate * (self.semantic_coherence_threshold - semantic_coherence)
                distinction_adjustment = self.adaptation_rate * 0.5  # Increase distinction slightly

            # Scale adjustments based on history stability
            if len(self.coherence_history) > 10:
                coherence_stability = 1.0 - np.std(list(self.coherence_history)[-10:])
                distinction_stability = 1.0 - np.std(list(self.distinction_history)[-10:])

                coherence_adjustment *= coherence_stability
                distinction_adjustment *= distinction_stability

            # Update feedback
            feedback['coherence_adjustment'] = coherence_adjustment
            feedback['distinction_adjustment'] = distinction_adjustment
            feedback['semantic_coherence'] = semantic_coherence

            # Log the feedback cycle
            self.total_feedback_cycles += 1

            return feedback

        except Exception as e:
            logger.error(f"Error calculating feedback parameters: {e}")
            return {
                'coherence_adjustment': 0.0,
                'distinction_adjustment': 0.0,
                'semantic_coherence': 0.5,
                'feedback_strength': 0.0
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

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the current status of semantic integration.

        Returns:
            Dictionary with integration status
        """
        try:
            # Calculate basic stats
            if self.integration_metrics['semantic_coherence']:
                avg_coherence = np.mean(self.integration_metrics['semantic_coherence'])
                coherence_trend = np.mean(np.diff(self.integration_metrics['semantic_coherence'][-10:])) if len(self.integration_metrics['semantic_coherence']) > 10 else 0.0
            else:
                avg_coherence = 0.0
                coherence_trend = 0.0

            # Get vocabulary status
            vocab_status = self.symbolic_output.get_vocabulary_status()

            # Prepare status report
            status = {
                'semantic_model_loaded': self.semantic_model_loaded,
                'bidirectional_enabled': self.enable_bidirectional,
                'total_expressions': self.total_expressions,
                'emergence_events': self.emergence_events,
                'total_feedback_cycles': self.total_feedback_cycles,
                'avg_semantic_coherence': avg_coherence,
                'coherence_trend': coherence_trend,
                'vocabulary_status': vocab_status,
                'last_expression_time': self.last_expression_time,
                'feedback_strength': self.feedback_strength,
                'adaptation_rate': self.adaptation_rate
            }

            # Add most recent expression
            if self.last_expression:
                status['last_expression'] = self.last_expression

            # Add emergence analysis if available
            if self.emergence_events > 0:
                emergence_analysis = self.symbolic_output.analyze_emergence_patterns()
                status['emergence_analysis'] = emergence_analysis

            return status

        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {
                'error': str(e),
                'semantic_model_loaded': self.semantic_model_loaded
            }

    def export_semantic_knowledge(self, filepath=None) -> Optional[str]:
        """
        Export semantic knowledge to a file.

        Args:
            filepath: Optional filepath, or None to use automatic name

        Returns:
            Path to exported file, or None if export failed
        """
        try:
            # Use symbolic_output's export function
            return self.semantic_enhanced_output.export_semantic_knowledge(filepath)
        except Exception as e:
            logger.error(f"Error exporting semantic knowledge: {e}")
            return None

    def import_semantic_knowledge(self, filepath: str) -> bool:
        """
        Import semantic knowledge from a file.

        Args:
            filepath: Path to the semantic knowledge file

        Returns:
            Boolean indicating success
        """
        try:
            # Use symbolic_output's import function
            return self.symbolic_output.import_semantic_knowledge(filepath)
        except Exception as e:
            logger.error(f"Error importing semantic knowledge: {e}")
            return False

# Adapter function for legacy code compatibility
def generate_symbolic_expression(
    surplus: float,
    distinction: float,
    coherence: float,
    entropy: Optional[float] = None,
    dimensionality: Optional[int] = None,
    semantic_manager: Optional[SemanticIntegrationManager] = None
) -> str:
    """
    Adapter function to generate symbolic expressions using either the
    semantic integration manager (if provided) or legacy fallback.

    Args:
        surplus: Current cognitive surplus level
        distinction: Current distinction level
        coherence: Current phase coherence
        entropy: Optional entropy metric
        dimensionality: Optional detected dimensionality
        semantic_manager: Optional semantic integration manager

    Returns:
        A symbolic expression representing the current state
    """
    try:
        if semantic_manager is not None:
            return semantic_manager.generate_symbolic_expression(
                surplus=surplus,
                distinction=distinction,
                coherence=coherence,
                entropy=entropy,
                dimensionality=dimensionality
            )
        else:
            # Legacy fallback if no semantic manager is provided
            from symbolic_output import SymbolicOutput
            symbolic_system = SymbolicOutput()
            return symbolic_system.generate_symbolic_expression(
                surplus=surplus,
                distinction=distinction,
                coherence=coherence,
                entropy=entropy,
                dimensionality=dimensionality
            )
    except Exception as e:
        logger.error(f"Error in adapter function: {e}")
        return "System in flux."  # Safe fallback
