# This will be written to simulation_runner_logs.py
import asyncio
import threading
from queue import Queue
import time
import os
import sys
import select
from typing import Optional, Dict, Any, List, Tuple
import psutil
import traceback
import numpy as np
import torch
import traceback
# Import the logging setup
from logging_setup import setup_logging

# Import required modules
from agent_classes import EnhancedSingleAgentFinalEvolution
from emergence_monitor import EmergenceTracker, DimensionMonitor, EmergenceEvent
from symbolic_output import SymbolicOutput
from emergent_potential import EmergentPotentialField


class InteractiveSimulation:
    def __init__(self, agent):
        # Set up logging
        self.logger = setup_logging()

        self.agent = agent
        self.running = True
        self.command_queue = Queue()
        self.response_queue = Queue()
        self.symbolic_system = SymbolicOutput()  # Initialize symbolic output system

        # Emergence tracking
        self.dimension_increase_detected = False
        self.running = True
        self.command_queue = Queue()
        self.response_queue = Queue()

        # Resource monitoring
        self.resource_history = []
        self.resource_limit_warnings = {
            'cpu': False,
            'memory': False
        }

        # Performance thresholds
        self.cpu_warning_threshold = 85.0
        self.memory_warning_threshold = 80.0
        self.cpu_emergency_threshold = 95.0
        self.memory_emergency_threshold = 90.0

        # Emergence support
        self.emergence_tracker = EmergenceTracker()
        self.dimension_monitor = DimensionMonitor()
        self.emergence_adaptation_active = False
        self.dimension_increase_detected = False
        self.tensor_shape_history = []
        self.adaptation_params = {
            'resource_allocation_factor': 1.0,
            'processing_interval': 0.01,
            'dimension_support_multiplier': 1.0,
            'adaptive_step_size': True
        }

        # Monitor for emergent dimensions
        self._patch_agent_for_emergence_detection()

        # Thread and task references
        self.sim_thread = None
        self.sim_task = None

        # Initialize emergence statistics
        self.emergence_stats = {
            'events': 0,
            'last_timestamp': None,
            'total_duration': 0,
            'dimensions_reached': set(),
            'peak_resource_usage': {
                'cpu': 0,
                'memory': 0
            }
        }

    def _patch_agent_for_emergence_detection(self):
        """Patch agent methods to detect emergent dimensions with improved integration."""
        try:
            if hasattr(self.agent, "prepare_transformer_input"):
                original_prepare = self.agent.prepare_transformer_input

                def patched_prepare(*p_args, **p_kwargs):
                    try:
                        result = original_prepare(*p_args, **p_kwargs)

                        # Check for emergence in the result
                        if isinstance(result, torch.Tensor) and result.dim() > 3:
                            # Only trigger if agent doesn't already know about it
                            if not getattr(self.agent, 'dimension_increase_detected', False):
                                if hasattr(self.agent, 'handle_emergent_dimension'):
                                    # Use the agent's own handler if available
                                    self.agent.handle_emergent_dimension(result.shape, "prepare_transformer_input")
                                else:
                                    # Fallback to simulation runner handling
                                    self._handle_emergent_dimension(result.shape, "prepare_transformer_input")

                                # Share the emergence information with the agent
                                self.agent.dimension_increase_detected = True

                        return result
                    except Exception as e:
                        self.logger.error(f"Error in patched prepare_transformer_input: {e}")
                        traceback.print_exc()
                        # Return the original tensor if possible, otherwise a safe default
                        try:
                            return original_prepare(*p_args, **p_kwargs)
                        except:
                            return torch.zeros((1, 1, 20), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                self.agent.prepare_transformer_input = patched_prepare

            # Monitor other tensor-producing methods
            if hasattr(self.agent, "predict"):
                original_predict = self.agent.predict

                def patched_predict(*p_args, **p_kwargs):
                    try:
                        result = original_predict(*p_args, **p_kwargs)
                        if isinstance(result, torch.Tensor) and result.dim() > 3:
                            if hasattr(self.agent, 'handle_emergent_dimension'):
                                self.agent.handle_emergent_dimension(result.shape, "predict")
                            else:
                                self._handle_emergent_dimension(result.shape, "predict")
                        return result
                    except Exception as e:
                        self.logger.error(f"Error in patched predict: {e}")
                        traceback.print_exc()
                        # Return the original result if possible
                        try:
                            return original_predict(*p_args, **p_kwargs)
                        except:
                            # Return a safe default
                            return torch.zeros((1, 1, 1), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                self.agent.predict = patched_predict

            self.logger.info("✅ Agent successfully patched for emergence detection")

        except Exception as e:
            self.logger.error(f"Error patching agent for emergence detection: {e}")
            traceback.print_exc()

    def _handle_emergent_dimension(self, shape, source="unknown"):
        """Handle detection of an emergent dimension with better agent integration."""
        try:
            if len(shape) > 3 and not self.dimension_increase_detected:
                self.dimension_increase_detected = True
                self.logger.info(f"\n🌟 EMERGENCE DETECTED: Dimensional expansion to {len(shape)}D")
                self.logger.info(f"    Tensor shape: {shape}")
                self.logger.info(f"    Source: {source}")
                self.logger.info(f"    Émile is evolving its computational ontology!")

                # Update emergence statistics
                self.emergence_stats['events'] += 1
                self.emergence_stats['last_timestamp'] = time.time()
                self.emergence_stats['dimensions_reached'].add(len(shape))

                # Update agent if agent doesn't have its own handler
                if hasattr(self.agent, 'dimension_increase_detected'):
                    self.agent.dimension_increase_detected = True

                # Register with dimension monitor
                self.dimension_monitor.register_shape(shape, source)

                # Record emergence event
                try:
                    # Get agent metrics if possible
                    if hasattr(self.agent, 'quantum_state') and hasattr(self.agent.quantum_state, 'get_quantum_metrics'):
                        metrics = self.agent.quantum_state.get_quantum_metrics()
                    else:
                        metrics = {}

                    # Get resource usage
                    resource_data = self.monitor_resources()

                    # Update peak resource usage during emergence
                    self.emergence_stats['peak_resource_usage']['cpu'] = max(
                        self.emergence_stats['peak_resource_usage']['cpu'],
                        resource_data['cpu_percent']
                    )
                    self.emergence_stats['peak_resource_usage']['memory'] = max(
                        self.emergence_stats['peak_resource_usage']['memory'],
                        resource_data['memory_percent']
                    )

                    # Record in emergence tracker
                    self.emergence_tracker.record_emergence(
                        tensor_shape=shape,
                        timestamp=time.time(),
                        resource_usage=resource_data,
                        agent_metrics=metrics
                    )

                    # Activate emergence support systems
                    self._activate_emergence_support()

                    # Connect with emergent potential field if agent has one
                    if hasattr(self.agent, 'emergent_potential_field'):
                        try:
                            # Register emergence with the field
                            self.agent.emergent_potential_field.register_potential(
                                component_id=f"dimensional_emergence_{source}",
                                potential=0.8,  # High potential for dimensional emergence
                                component_type='quantum',
                                state_metrics={
                                    'tensor_shape': shape,
                                    'dimensionality': len(shape),
                                    'source': source,
                                    'distinction': getattr(self.agent, 'distinction_level', 0.5),
                                    'coherence': metrics.get('phase_coherence', 0.5)
                                }
                            )
                            self.logger.info("✅ Emergence registered with potential field")
                        except Exception as epf_error:
                            self.logger.error(f"Error registering with emergent potential field: {epf_error}")

                except Exception as e:
                    self.logger.error(f"Error recording emergence event: {e}")
                    traceback.print_exc()

                # Trigger symbolic expression
                try:
                    # Extract parameters from agent if possible
                    if hasattr(self.agent, 'surplus_dynamics') and hasattr(self.agent.surplus_dynamics, 'surplus_state'):
                        surplus = self.agent.surplus_dynamics.surplus_state.total_surplus()
                    else:
                        surplus = 1.0

                    if hasattr(self.agent, 'distinction_level'):
                        distinction = self.agent.distinction_level
                    else:
                        distinction = 0.5

                    if hasattr(self.agent, 'quantum_state') and hasattr(self.agent.quantum_state, 'phase_coherence'):
                        coherence = self.agent.quantum_state.phase_coherence
                    else:
                        coherence = 0.5

                    # Generate symbolic expression
                    symbolic_expression = self.symbolic_system.handle_post_emergence(
                        surplus=surplus,
                        distinction=distinction,
                        coherence=coherence,
                        dimensionality=len(shape),
                        entropy=metrics.get('normalized_entropy', None) if 'metrics' in locals() else None
                    )

                    # Log symbolic expression to console
                    self.logger.info(f"\n🔹 Symbolic Output Post-Emergence: {symbolic_expression}\n")

                    # Add pattern analysis if we have multiple emergence events
                    if len(self.symbolic_system.emergence_events) > 1:
                        patterns = self.symbolic_system.analyze_emergence_patterns()
                        self.logger.info("\n📊 Symbolic Pattern Analysis:")
                        self.logger.info(f"  Emergence Events: {patterns['emergence_count']}")
                        self.logger.info(f"  Expression Stability: {patterns.get('coherence_stability', 0):.3f}")
                        if patterns.get('typical_expression'):
                            self.logger.info(f"  Typical Expression: {patterns['typical_expression']}")

                    # Return the expression to the agent if it has symbolic history
                    if hasattr(self.agent, 'symbolic_history'):
                        self.agent.last_symbolic_expression = symbolic_expression
                        self.agent.symbolic_history.append({
                            'expression': symbolic_expression,
                            'type': 'emergence',
                            'step': getattr(self.agent, 'step_counter', 0),
                            'dimensionality': len(shape),
                            'tensor_shape': shape,
                            'source': source,
                            'timestamp': time.time()
                        })

                except Exception as e:
                    self.logger.error(f"Error generating symbolic expression: {e}")
                    traceback.print_exc()

            elif len(shape) <= 3 and self.dimension_increase_detected:
                self.dimension_increase_detected = False
                self.logger.info("\n📉 Dimensional reduction detected: returning to standard dimensionality")

                # Calculate emergence duration
                if self.emergence_stats['last_timestamp'] is not None:
                    duration = time.time() - self.emergence_stats['last_timestamp']
                    self.emergence_stats['total_duration'] += duration
                    self.logger.info(f"Emergence duration: {duration:.2f} seconds")

                # End the emergence event
                try:
                    # Get final metrics
                    if hasattr(self.agent, 'quantum_state'):
                        metrics = self.agent.quantum_state.get_quantum_metrics()
                    else:
                        metrics = {}

                    self.emergence_tracker.end_emergence(
                        timestamp=time.time(),
                        metrics=metrics
                    )
                except Exception as end_error:
                    self.logger.error(f"Error ending emergence: {end_error}")

                # Deactivate emergence support
                self._deactivate_emergence_support()

                # Update agent if needed
                if hasattr(self.agent, 'dimension_increase_detected'):
                    self.agent.dimension_increase_detected = False

        except Exception as e:
            self.logger.error(f"Error handling emergent dimension: {e}")
            traceback.print_exc()


    def _get_symbolic_analysis(self) -> str:
        """Generate a comprehensive analysis of symbolic expressions."""
        try:
            if not hasattr(self, 'symbolic_system') or not self.symbolic_system.emergence_events:
                return "No symbolic expressions have been generated yet."

            patterns = self.symbolic_system.analyze_emergence_patterns()

            # Format the analysis as a multi-line string
            analysis = [
                "\n🔮 Symbolic Expression Analysis",
                "==============================",
                f"Total Emergence Events: {patterns['emergence_count']}",
                f"Total Expressions Generated: {patterns['expression_count']}",
                f"Coherence Stability: {patterns.get('coherence_stability', 0):.4f}",
                f"Distinction Stability: {patterns.get('distinction_stability', 0):.4f}",
                "\nDominant Patterns:",
            ]

            dominant = patterns.get('dominant_patterns', {})
            if dominant:
                analysis.extend([
                    f"  Primary Descriptor: {dominant.get('descriptor', 'None')}",
                    f"  Primary Relation: {dominant.get('relation', 'None')}",
                    f"  Primary Concept: {dominant.get('concept', 'None')}"
                ])

            if patterns.get('typical_expression'):
                analysis.extend([
                    "\nTypical Expression:",
                    f"  {patterns['typical_expression']}"
                ])

            # Add recent expressions
            recent_expressions = self.symbolic_system.expression_history[-3:] if self.symbolic_system.expression_history else []
            if recent_expressions:
                analysis.extend([
                    "\nRecent Expressions:",
                ])
                for i, expr in enumerate(recent_expressions):
                    analysis.append(f"  {i+1}. {expr['expression']}")

            return "\n".join(analysis)
        except Exception as e:
            self.logger.error(f"Error generating symbolic analysis: {e}")
            return f"Error generating symbolic analysis: {e}"

    def monitor_resources(self):
        """Monitors and logs CPU and memory usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()

            resource_data = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'memory_available': memory_info.available / (1024 * 1024)  # MB
            }

            # Store historical data
            self.resource_history.append(resource_data)
            if len(self.resource_history) > 1000:  # Keep last 1000 measurements
                self.resource_history.pop(0)

            # Check for resource warnings
            self._check_resource_warnings(resource_data)

            return resource_data
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available': 0.0
            }

    def _check_resource_warnings(self, resource_data):
        """Check if resource usage exceeds thresholds and issue warnings."""
        try:
            # CPU warning
            if resource_data['cpu_percent'] > self.cpu_warning_threshold and not self.resource_limit_warnings['cpu']:
                self.logger.info(f"\n⚠️ WARNING: High CPU usage detected ({resource_data['cpu_percent']:.1f}%)")
                if self.emergence_tracker.is_emergence_active:
                    self.logger.info("  This may be related to the ongoing emergence process")
                    self._adjust_for_high_resource_usage()
                self.resource_limit_warnings['cpu'] = True
            elif resource_data['cpu_percent'] < self.cpu_warning_threshold - 10 and self.resource_limit_warnings['cpu']:
                self.resource_limit_warnings['cpu'] = False
                if self.emergence_tracker.is_emergence_active:
                    self._restore_normal_processing()

            # Memory warning
            if resource_data['memory_percent'] > self.memory_warning_threshold and not self.resource_limit_warnings['memory']:
                self.logger.info(f"\n⚠️ WARNING: High memory usage detected ({resource_data['memory_percent']:.1f}%)")
                if self.emergence_tracker.is_emergence_active:
                    self.logger.info("  This may be related to ongoing emergence processing")
                    self._adjust_for_high_resource_usage()
                self.resource_limit_warnings['memory'] = True
            elif resource_data['memory_percent'] < self.memory_warning_threshold - 10 and self.resource_limit_warnings['memory']:
                self.resource_limit_warnings['memory'] = False
                if self.emergence_tracker.is_emergence_active:
                    self._restore_normal_processing()

            # Emergency thresholds
            if resource_data['cpu_percent'] > self.cpu_emergency_threshold:
                self.logger.warning(f"\n🚨 CRITICAL: CPU usage at {resource_data['cpu_percent']:.1f}%")
                self.logger.warning("  Applying emergency resource management")
                self._apply_emergency_resource_management()

            if resource_data['memory_percent'] > self.memory_emergency_threshold:
                self.logger.warning(f"\n🚨 CRITICAL: Memory usage at {resource_data['memory_percent']:.1f}%")
                self.logger.warning("  Applying emergency resource management")
                self._apply_emergency_resource_management()

        except Exception as e:
            self.logger.error(f"Error checking resource warnings: {e}")

    def _adjust_for_high_resource_usage(self):
        """Adapt to high resource usage during emergence."""
        try:
            # Increase processing interval for breathing room
            self.adaptation_params['processing_interval'] = min(
                0.05, self.adaptation_params['processing_interval'] * 1.5
            )
            # Reduce resource allocation factor
            self.adaptation_params['resource_allocation_factor'] = 0.8
            self.logger.info(f"⚙️ Adapting to high resource usage during emergence:")
            self.logger.info(f"  - Adjusted processing interval: {self.adaptation_params['processing_interval']:.3f}s")
            self.logger.info(f"  - Resource allocation factor: {self.adaptation_params['resource_allocation_factor']:.2f}")
        except Exception as e:
            self.logger.error(f"Error adjusting for high resource usage: {e}")

    def _restore_normal_processing(self):
        """Restore normal processing parameters."""
        try:
            self.adaptation_params['processing_interval'] = max(
                0.01, self.adaptation_params['processing_interval'] * 0.8
            )
            self.adaptation_params['resource_allocation_factor'] = min(
                1.0, self.adaptation_params['resource_allocation_factor'] * 1.1
            )
        except Exception as e:
            self.logger.error(f"Error restoring normal processing: {e}")

    def _apply_emergency_resource_management(self):
        """Apply emergency resource management to prevent crashes."""
        try:
            # Drastically increase sleep time
            self.adaptation_params['processing_interval'] = 0.2
            # Reduce resource allocation to minimum
            self.adaptation_params['resource_allocation_factor'] = 0.5

            # Force garbage collection
            import gc
            gc.collect()

            # Clear any unnecessary caches
            if hasattr(self.agent, 'analysis_history') and isinstance(self.agent.analysis_history, dict):
                for k in list(self.agent.analysis_history.keys()):
                    if hasattr(self.agent.analysis_history[k], 'clear'):
                        self.agent.analysis_history[k].clear()

            # Clear tensor memory if torch is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("🧹 Applied emergency resource management")
        except Exception as e:
            self.logger.error(f"Error applying emergency resource management: {e}")

    def _activate_emergence_support(self):
        """Activate support systems for emergence."""
        try:
            if self.emergence_adaptation_active:
                return
            self.emergence_adaptation_active = True
            self.logger.info("\n🔄 Activating emergence support systems")

            # Adjust parameters to support emergence
            self.adaptation_params['processing_interval'] = 0.02
            self.adaptation_params['dimension_support_multiplier'] = 1.2

            # Track stability during emergence
            if hasattr(self.agent, 'stability_factor'):
                self.emergence_tracker.update_stability(self.agent.stability_factor, time.time())
        except Exception as e:
            self.logger.error(f"Error activating emergence support: {e}")

    def _deactivate_emergence_support(self):
        """Deactivate emergence support systems."""
        try:
            if not self.emergence_adaptation_active:
                return
            self.emergence_adaptation_active = False
            self.logger.info("\n✓ Deactivating emergence support systems")

            # Reset parameters
            self.adaptation_params['processing_interval'] = 0.01
            self.adaptation_params['dimension_support_multiplier'] = 1.0

            # Generate emergence summary
            summary = self.emergence_tracker.get_emergence_summary()
            detailed = self.emergence_tracker.get_detailed_analysis()

            self.logger.info("\n📑 Emergence Event Summary:")
            self.logger.info(f"  Total Events: {summary.get('emergence_count', 0)}")
            self.logger.info(f"  Duration: {summary.get('total_duration', 0):.2f} seconds")
            self.logger.info(f"  Dimension Transitions: {summary.get('dimension_transitions', 0)}")
            self.logger.info(f"  Pattern Diversity: {summary.get('pattern_diversity', 0)}")
            self.logger.info(f"  Resource Impact: {summary.get('resource_impact', 0):.2f}% CPU")

            if detailed.get('dominant_pattern'):
                self.logger.info(f"  Dominant Pattern: {detailed['dominant_pattern']}")
        except Exception as e:
            self.logger.error(f"Error deactivating emergence support: {e}")

    async def run_simulation(self):
        """Run the simulation while handling interactive commands."""
        try:
            self.logger.info("\n🚀 Starting Interactive Émile-3K Simulation...")
            step = 0
            last_resource_log = time.time()
            resource_log_interval = 5  # Log every 5 seconds

            # Store task reference
            self.sim_task = asyncio.current_task()

            while self.running:
                step_start_time = time.time()

                # Process any pending commands
                while not self.command_queue.empty():
                    command = self.command_queue.get()
                    await self.handle_command(command)

                # Execute simulation step
                try:
                    step_success = self.agent.step()
                    if not step_success:
                        self.logger.warning(f"\n⚠️ Agent step returned unsuccessful status at step {step}")
                        # Apply small delay after unsuccessful step
                        await asyncio.sleep(0.1)
                except Exception as step_error:
                    self.logger.error(f"\n❌ Error in agent step: {step_error}")
                    traceback.print_exc()  # Goes to log file
                    await asyncio.sleep(0.1)
                    continue

                # Get metrics silently (don't log them)
                try:
                    metrics = {
                        'distinction_level': float(self.agent.distinction_level),
                        'coherence': float(self.agent.quantum_state.phase_coherence),
                        'entropy': float(self.agent.quantum_state.get_quantum_metrics().get('normalized_entropy', 0)),
                        'stability': float(self.agent.stability_factor),
                        'surplus_stability': float(self.agent.surplus_dynamics.surplus_state.stability)
                    }

                    # Check for dimensional changes without logging
                    if hasattr(self.agent, 'prepare_transformer_input'):
                        try:
                            input_tensor = self.agent.prepare_transformer_input()
                            if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
                                event = self.dimension_monitor.register_shape(input_tensor, "regular_progress_check")
                                if event:
                                    # Process emergence event
                                    self.logger.info(f"\n🔍 New tensor shape detected: {input_tensor.shape}")
                        except Exception:
                            pass  # silent

                    # Update stability tracking during emergence
                    if self.emergence_tracker.is_emergence_active:
                        self.emergence_tracker.update_stability(metrics['stability'], time.time())

                    # Check for emergent potential field events
                    if (hasattr(self.agent, 'emergent_potential_field') and
                        step % 10 == 0):  # Check every 10 steps to avoid overhead
                        try:
                            field_state = self.agent.emergent_potential_field.get_field_state()
                            # Log high emergence probability
                            if field_state.get('emergence_probability', 0) > 0.8:
                                self.logger.info(f"\n⚡ High emergence potential detected: {field_state['emergence_probability']:.2f}")
                            # Log active emergence
                            if field_state.get('emergence_active', False) and not getattr(self, 'logged_field_emergence', False):
                                self.logger.info(f"\n🌠 Emergent potential field has activated emergence")
                                setattr(self, 'logged_field_emergence', True)
                            # Reset field emergence log flag if no longer active
                            elif not field_state.get('emergence_active', False) and getattr(self, 'logged_field_emergence', False):
                                setattr(self, 'logged_field_emergence', False)
                        except Exception as field_error:
                            pass  # silent

                except Exception as metrics_error:
                    metrics = {
                        'distinction_level': 0.0,
                        'coherence': 0.0,
                        'entropy': 0.0,
                        'stability': 0.0,
                        'surplus_stability': 0.0
                    }

                # Log progress periodically
                if step % 50 == 0:
                    self._log_progress(step, metrics)

                # Monitor resources periodically
                current_time = time.time()
                if current_time - last_resource_log >= resource_log_interval:
                    resource_usage = self.monitor_resources()
                    self._log_resources(resource_usage)
                    last_resource_log = current_time

                step += 1

                # Adaptive sleep interval based on emergence
                sleep_time = self.adaptation_params['processing_interval']
                if self.emergence_adaptation_active:
                    # Dynamic adjustment if steps are taking a while
                    step_duration = time.time() - step_start_time
                    if step_duration > 0.1 and self.adaptation_params['adaptive_step_size']:
                        sleep_time = min(0.05, sleep_time * 1.2)

                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            self.logger.info("\n⚠️ Simulation task cancelled")
        except Exception as e:
            self.logger.error(f"\n❌ Error in simulation: {e}")
            traceback.print_exc()  # Goes to log file
        finally:
            self.logger.info("\n✨ Simulation ended")
            # Final resource report
            self.logger.info("\n📊 Final Resource Usage Report:")
            if self.resource_history:
                avg_cpu = sum(r['cpu_percent'] for r in self.resource_history) / len(self.resource_history)
                avg_memory = sum(r['memory_percent'] for r in self.resource_history) / len(self.resource_history)
                peak_cpu = max(r['cpu_percent'] for r in self.resource_history)
                peak_memory = max(r['memory_percent'] for r in self.resource_history)

                self.logger.info(f"  Average CPU: {avg_cpu:.1f}%")
                self.logger.info(f"  Average Memory: {avg_memory:.1f}%")
                self.logger.info(f"  Peak CPU: {peak_cpu:.1f}%")
                self.logger.info(f"  Peak Memory: {peak_memory:.1f}%")

            # Final emergence report
            if self.emergence_tracker.emergence_events:
                summary = self.emergence_tracker.get_emergence_summary()
                self.logger.info("\n🌟 Final Emergence Report:")
                self.logger.info(f"  Total Emergence Events: {summary.get('emergence_count', 0)}")
                self.logger.info(f"  Dimension Transitions: {summary.get('dimension_transitions', 0)}")
                self.logger.info(f"  Total Duration: {summary.get('total_duration', 0):.2f} seconds")
                self.logger.info(f"  Pattern Diversity: {summary.get('pattern_diversity', 0)}")

                detailed = self.emergence_tracker.get_detailed_analysis()
                if detailed.get('dominant_pattern'):
                    dominant = detailed['dominant_pattern']
                    pattern_details = detailed['pattern_details'].get(dominant, {})
                    self.logger.info(f"\n  Dominant Pattern: {dominant}")
                    self.logger.info(f"    Occurrences: {pattern_details.get('count', 0)}")
                    self.logger.info(f"    Avg CPU Impact: {pattern_details.get('avg_cpu_impact', 0):.2f}%")
                    self.logger.info(f"    Stability Mean: {pattern_details.get('stability_mean', 0):.4f}")
                    self.logger.info(f"    Coherence Mean: {pattern_details.get('coherence_mean', 0):.4f}")

    def _log_progress(self, step: int, metrics: Dict[str, float]):
        """Log simulation progress for each step interval."""
        self.logger.info(
            f"\n[Step {step}] "
            f"Distinction: {metrics.get('distinction_level', 0.0):.3f}, "
            f"Coherence: {metrics.get('coherence', 0.0):.3f}, "
            f"Entropy: {metrics.get('entropy', 0.0):.3f}, "
            f"Stability: {metrics.get('stability', 0.0):.3f}, "
            f"SurplusStab: {metrics.get('surplus_stability', 0.0):.3f}"
        )

        # Log emergent potential field data if available
        if hasattr(self.agent, 'emergent_potential_field') and step % 100 == 0:
            try:
                field_state = self.agent.emergent_potential_field.get_field_state()
                self.logger.info(
                    f"Emergent Potential: {field_state.get('total_potential', 0.0):.3f}, "
                    f"Threshold: {field_state.get('threshold', 0.0):.3f}, "
                    f"Prob: {field_state.get('emergence_probability', 0.0):.2f}"
                )
            except Exception as e:
                pass  # Silent fail for emergent field logging

    def _log_resources(self, resource_usage: Dict[str, float]):
        """Log resource usage each time we poll it (default every 5s)."""
        self.logger.info(
            f"Resource Usage => CPU: {resource_usage['cpu_percent']:.1f}%, "
            f"Memory: {resource_usage['memory_percent']:.1f}%, "
            f"Avail: {resource_usage['memory_available']:.0f} MB"
        )

    async def handle_command(self, command: str):
        """Process user commands and generate responses."""
        try:
            command = command.lower().strip()

            if command == "status":
                metrics = {
                    'distinction': getattr(self.agent, 'distinction_level', 0),
                    'coherence': getattr(self.agent.quantum_state, 'phase_coherence', 0)
                        if hasattr(self.agent, 'quantum_state') else 0,
                    'stability': getattr(self.agent, 'stability_factor', 0),
                    'surplus_stability': getattr(self.agent.surplus_dynamics.surplus_state, 'stability', 0)
                        if (hasattr(self.agent, 'surplus_dynamics') and
                            hasattr(self.agent.surplus_dynamics, 'surplus_state')) else 0,
                    'surplus_values': getattr(self.agent.surplus_dynamics.surplus_state, 'values', {})
                        if (hasattr(self.agent, 'surplus_dynamics') and
                            hasattr(self.agent.surplus_dynamics, 'surplus_state')) else {},
                    'steps': getattr(self.agent, 'step_counter', 0)
                }

                # Add emergent potential field data
                if hasattr(self.agent, 'emergent_potential_field'):
                    try:
                        field_state = self.agent.emergent_potential_field.get_field_state()
                        metrics['emergent_potential'] = field_state.get('total_potential', 0.0)
                        metrics['emergence_probability'] = field_state.get('emergence_probability', 0.0)
                        metrics['emergence_active'] = field_state.get('emergence_active', False)
                    except Exception:
                        pass  # Silent fail

                self.response_queue.put(f"\nCurrent Status:\n{self._format_metrics(metrics)}")

            elif command == "emergence":
                # Enhanced emergence command
                emergence_metrics = {}
                if (hasattr(self.agent, 'surplus_dynamics') and
                    hasattr(self.agent.surplus_dynamics, 'get_emergence_metrics')):
                    emergence_metrics = self.agent.surplus_dynamics.get_emergence_metrics()

                tracker_metrics = self.emergence_tracker.get_emergence_summary()
                if tracker_metrics:
                    emergence_metrics['dimensional_emergence'] = tracker_metrics

                detailed_analysis = self.emergence_tracker.get_detailed_analysis()
                if detailed_analysis and detailed_analysis.get('total_patterns', 0) > 0:
                    emergence_metrics['pattern_analysis'] = detailed_analysis

                # Add dimension monitor statistics
                if self.dimension_monitor.dimension_changes:
                    emergence_metrics['dimension_changes'] = len(self.dimension_monitor.dimension_changes)
                    emergence_metrics['last_dimensionality'] = self.dimension_monitor.last_dimensionality

                # Add emergent potential field data
                if hasattr(self.agent, 'emergent_potential_field'):
                    try:
                        field_state = self.agent.emergent_potential_field.get_field_state()
                        emergence_metrics['potential_field'] = {
                            'total_potential': field_state.get('total_potential', 0.0),
                            'threshold': field_state.get('threshold', 0.0),
                            'emergence_probability': field_state.get('emergence_probability', 0.0),
                            'emergence_active': field_state.get('emergence_active', False)
                        }
                    except Exception as e:
                        emergence_metrics['potential_field_error'] = str(e)

                # Add statistics from our tracker
                emergence_metrics['stats'] = {
                    'events': self.emergence_stats['events'],
                    'dimensions_reached': list(self.emergence_stats['dimensions_reached']),
                    'total_duration': self.emergence_stats['total_duration'],
                    'peak_resource_usage': self.emergence_stats['peak_resource_usage']
                }

                if emergence_metrics:
                    self.response_queue.put(f"\nEmergence Metrics:\n{self._format_metrics(emergence_metrics)}")
                else:
                    self.response_queue.put("\nNo emergence metrics available yet.")

            elif command == "resources":
                # Enhanced resource command
                if self.resource_history:
                    latest = self.resource_history[-1]
                    avg_cpu = sum(r['cpu_percent'] for r in self.resource_history) / len(self.resource_history)
                    avg_memory = sum(r['memory_percent'] for r in self.resource_history) / len(self.resource_history)
                    peak_cpu = max(r['cpu_percent'] for r in self.resource_history)
                    peak_memory = max(r['memory_percent'] for r in self.resource_history)

                    emergence_info = ""
                    if self.emergence_tracker.is_emergence_active:
                        emergence_summary = self.emergence_tracker.get_emergence_summary()
                        emergence_info = (
                            f"\n\nEmergence Resource Impact:"
                            f"\n  CPU Correlation: {emergence_summary.get('resource_correlation', 0):.3f}"
                            f"\n  Average Impact: {emergence_summary.get('resource_impact', 0):.2f}%"
                            f"\n  Adaptation Multiplier: {self.adaptation_params['dimension_support_multiplier']:.2f}"
                            f"\n  Processing Interval: {self.adaptation_params['processing_interval']:.3f}s"
                        )

                    resource_report = (
                        f"\nResource Usage:"
                        f"\n  Current CPU: {latest['cpu_percent']:.1f}%"
                        f"\n  Current Memory: {latest['memory_percent']:.1f}%"
                        f"\n  Available Memory: {latest['memory_available']:.0f} MB"
                        f"\n  Average CPU: {avg_cpu:.1f}%"
                        f"\n  Average Memory: {avg_memory:.1f}%"
                        f"\n  Peak CPU: {peak_cpu:.1f}%"
                        f"\n  Peak Memory: {peak_memory:.1f}%"
                        f"{emergence_info}"
                    )
                    self.response_queue.put(resource_report)
                else:
                    self.response_queue.put("\nNo resource data available yet.")

            elif command == "patterns":
                # Show emergence patterns
                pattern_analysis = self.emergence_tracker.get_detailed_analysis()
                if pattern_analysis and pattern_analysis.get('total_patterns', 0) > 0:
                    pattern_report = ["\nEmergence Pattern Analysis:"]
                    pattern_report.append(f"Total Patterns: {pattern_analysis['total_patterns']}")
                    pattern_report.append(f"Total Events: {pattern_analysis['total_events']}")
                    pattern_report.append(f"Dominant Pattern: {pattern_analysis.get('dominant_pattern', 'None')}")

                    if pattern_analysis.get('common_sequences'):
                        pattern_report.append(f"Common Sequence: {pattern_analysis['common_sequences']}")
                        pattern_report.append(f"Sequence Frequency: {pattern_analysis['sequence_frequency']}")

                    pattern_report.append("\nPattern Details:")

                    # Sort patterns by count for better display
                    sorted_patterns = sorted(
                        pattern_analysis.get('pattern_details', {}).items(),
                        key=lambda x: x[1].get('count', 0),
                        reverse=True
                    )

                    for shape, details in sorted_patterns:
                        pattern_report.append(f"\n  Pattern: {shape}")
                        pattern_report.append(f"    Count: {details.get('count', 0)}")
                        pattern_report.append(f"    Duration: {details.get('duration', 0):.2f}s")
                        pattern_report.append(f"    Avg CPU Impact: {details.get('avg_cpu_impact', 0):.2f}%")
                        pattern_report.append(f"    Stability Mean: {details.get('stability_mean', 0):.4f}")
                        pattern_report.append(f"    Coherence Mean: {details.get('coherence_mean', 0):.4f}")
                        pattern_report.append(f"    Distinction Mean: {details.get('distinction_mean', 0):.4f}")

                    self.response_queue.put("\n".join(pattern_report))
                else:
                    self.response_queue.put("\nNo emergence patterns detected yet.")

            elif command == "dimensions":
                # Show dimensional emergence details
                dim_changes = self.dimension_monitor.dimension_changes
                dim_transitions = self.dimension_monitor.shape_transitions
                dim_stats = self.dimension_monitor.get_transition_statistics()

                if dim_changes or dim_transitions:
                    dim_report = ["\nDimensional Emergence Analysis:"]
                    dim_report.append(f"Current Dimensionality: {self.dimension_monitor.last_dimensionality}D")
                    dim_report.append(f"Total Dimension Changes: {len(dim_changes)}")
                    dim_report.append(f"Total Shape Transitions: {len(dim_transitions)}")

                    # Add statistics
                    if dim_stats:
                        dim_report.append(f"Most Common Dimensionality: {dim_stats.get('most_common_dimensionality', 'Unknown')}")
                        dim_report.append(f"Most Common Transition: {dim_stats.get('most_common_transition', 'Unknown')}")

                        if 'dimensionality_proportions' in dim_stats:
                            dim_report.append("\nDimensionality Distribution:")
                            for dim, prop in dim_stats['dimensionality_proportions'].items():
                                dim_report.append(f"  {dim}D: {prop*100:.1f}%")

                    # Recent transitions
                    if dim_transitions:
                        dim_report.append("\nRecent Transitions:")
                        for i, transition in enumerate(dim_transitions[-5:]):
                            dim_report.append(f"  {i+1}. {transition['old_dimensionality']}D → {transition['new_dimensionality']}D")
                            dim_report.append(f"     Shape: {transition['shape']}")
                            dim_report.append(f"     Source: {transition['tag']}")
                            if 'timestamp' in transition:
                                time_str = time.strftime('%H:%M:%S', time.localtime(transition['timestamp']))
                                dim_report.append(f"     Time: {time_str}")

                    self.response_queue.put("\n".join(dim_report))
                else:
                    self.response_queue.put("\nNo dimensional changes detected yet.")

            elif command == "symbolic":
                # Get symbolic analysis
                analysis = self._get_symbolic_analysis()
                self.response_queue.put(analysis)

            elif command == "potential":
                # Get emergent potential field data
                if hasattr(self.agent, 'emergent_potential_field'):
                    try:
                        field_state = self.agent.emergent_potential_field.get_field_state()
                        field_data = self.agent.get_emergent_potential_visualization()

                        field_report = ["\nEmergent Potential Field:"]
                        field_report.append(f"Total Potential: {field_state.get('total_potential', 0.0):.4f}")
                        field_report.append(f"Threshold: {field_state.get('threshold', 0.0):.4f}")
                        field_report.append(f"Emergence Probability: {field_state.get('emergence_probability', 0.0):.4f}")
                        field_report.append(f"Field Intensity: {field_state.get('field_intensity', 1.0):.2f}")
                        field_report.append(f"Stability Factor: {field_state.get('stability_factor', 1.0):.2f}")
                        field_report.append(f"Emergence Active: {'Yes' if field_state.get('emergence_active', False) else 'No'}")
                        field_report.append(f"Emergence Count: {field_state.get('emergence_count', 0)}")

                        # Add component information
                        if 'components' in field_data and field_data['components']:
                            field_report.append("\nTop Components:")
                            for i, comp in enumerate(field_data['components'][:5]):  # Show top 5
                                field_report.append(f"  {i+1}. {comp['id']} ({comp['type']}): {comp['potential']:.4f}")

                        # Add emergence events
                        if 'emergence_events' in field_data and field_data['emergence_events']:
                            field_report.append("\nRecent Emergence Events:")
                            for i, event in enumerate(field_data['emergence_events'][-3:]):  # Show latest 3
                                field_report.append(f"  {i+1}. Intensity: {event['intensity']:.2f}, Potential: {event['potential']:.4f}")

                        self.response_queue.put("\n".join(field_report))
                    except Exception as e:
                        self.response_queue.put(f"\nError getting emergent potential field data: {e}")
                else:
                    self.response_queue.put("\nEmergent potential field not available.")

            elif command == "help":
                help_text = """
                Available Commands:
                - status: Get current agent metrics
                - emergence: Get detailed emergence metrics and analysis
                - resources: Get current resource usage
                - patterns: Show emergence pattern analysis
                - dimensions: Show dimensional emergence details
                - symbolic: Show analysis of symbolic expressions and emergence patterns
                - potential: Show emergent potential field status and components
                - help: Show this help message
                - exit: Stop the simulation
                """
                self.response_queue.put(help_text)

            elif command == "exit":
                self.running = False
                self.response_queue.put("Shutting down simulation...")

            else:
                self.response_queue.put(f"Unknown command: {command}")

        except Exception as e:
            self.response_queue.put(f"Error processing command: {e}")
            traceback.print_exc()

    def _format_metrics(self, metrics: Dict[str, Any], indent: int = 0) -> str:
        """Format a dictionary of metrics into a multi-line string for display."""
        if not metrics:
            return "(No metrics to display)"

        lines = []
        indent_str = " " * indent

        for k, v in metrics.items():
            if isinstance(v, dict):
                # Recursively format nested dictionaries with indentation
                lines.append(f"{indent_str}{k}:")
                lines.append(self._format_metrics(v, indent + 2))
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # Handle lists of dictionaries (like history items)
                lines.append(f"{indent_str}{k}:")
                for i, item in enumerate(v[:3]):  # Show only first 3 items
                    lines.append(f"{indent_str}  Item {i+1}:")
                    lines.append(self._format_metrics(item, indent + 4))
                if len(v) > 3:
                    lines.append(f"{indent_str}  ... ({len(v) - 3} more items)")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                # Format numbers with proper precision
                if abs(v) < 0.01 or abs(v) > 1000:
                    lines.append(f"{indent_str}{k}: {v:.6g}")
                else:
                    lines.append(f"{indent_str}{k}: {v:.4f}")
            else:
                # Format everything else
                lines.append(f"{indent_str}{k}: {v}")

        return "\n".join(lines)

def visualize_emergence_potential(agent, display_type='potential'):
    """Helper function to display emergence potential visualization in Colab"""
    try:
        import matplotlib.pyplot as plt

        if hasattr(agent, 'visualize_emergent_field'):
            fig = agent.visualize_emergent_field(display_type)
            if fig:
                plt.show()
        elif hasattr(agent, 'get_emergent_potential_visualization'):
            data = agent.get_emergent_potential_visualization()
            print(f"Emergent Potential Field Status:")
            print(f"- Total Potential: {data['field_state']['total_potential']:.4f}")
            print(f"- Threshold: {data['field_state']['threshold']:.4f}")
            print(f"- Emergence Probability: {data['field_state'].get('emergence_probability', 0.0):.4f}")
            print(f"- Emergence Active: {'Yes' if data['field_state']['emergence_active'] else 'No'}")
            print(f"- Components: {len(data['components'])}")
            print(f"- Emergence Events: {len(data['emergence_events'])}")

            # Create a simple bar chart of components
            if 'components' in data and data['components']:
                plt.figure(figsize=(10, 6))
                components = data['components'][:10]  # Top 10 components
                labels = [f"{c['id'][:10]}..." if len(c['id']) > 10 else c['id'] for c in components]
                values = [c['potential'] for c in components]
                colors = ['blue' if c['type'] == 'surplus' else
                          'green' if c['type'] == 'quantum' else
                          'orange' if c['type'] == 'cognitive' else 'gray'
                          for c in components]

                plt.bar(range(len(components)), values, color=colors)
                plt.xticks(range(len(components)), labels, rotation=45, ha='right')
                plt.title('Top Contributors to Emergent Potential')
                plt.ylabel('Potential')
                plt.tight_layout()
                plt.show()
        else:
            print("Agent does not have emergence potential visualization capabilities")
    except ImportError:
        print("Matplotlib is required for visualization")
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()

def run_interactive_simulation(agent):
    """Run the simulation with interactive command handling."""
    simulation = InteractiveSimulation(agent)

    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Start simulation in separate thread
        sim_thread = threading.Thread(
            target=lambda: loop.run_until_complete(simulation.run_simulation()),
            daemon=True
        )
        sim_thread.start()
        simulation.sim_thread = sim_thread

        simulation.logger.info("\nInteractive Émile-3K Simulation")
        simulation.logger.info("Type 'help' for available commands")
        simulation.logger.info("Use 'emergence' to analyze dimensional emergence")

        # Main input loop
        while simulation.running:
            try:
                command = None  # Initialize command at the start of each loop

                # Use more robust input handling
                if sys.platform != 'win32':
                    # For Unix systems, use select for non-blocking input
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        command = sys.stdin.readline().strip()
                        if command:
                            simulation.command_queue.put(command)
                else:
                    # For Windows, just use regular input with occasional checks for simulation status
                    try:
                        command = input(">> ")
                        if command:
                            simulation.command_queue.put(command)
                    except EOFError:
                        simulation.running = False
                        break

                # Check if simulation is still running
                if not simulation.running:
                    break

                # Wait briefly for responses
                time.sleep(0.1)
                while not simulation.response_queue.empty():
                    response = simulation.response_queue.get()
                    if response:
                        simulation.logger.info(response)

                # Make sure command is defined before checking it
                if command and command.lower() == "exit":
                    break

            except (KeyboardInterrupt, EOFError):
                simulation.logger.info("\nGraceful shutdown initiated...")
                simulation.running = False
                break
            except Exception as input_error:
                simulation.logger.error(f"Error in input handling: {input_error}")
                time.sleep(0.5)  # Brief pause before retrying

        # Wait for thread to terminate with timeout
        if sim_thread.is_alive():
            print("Waiting for simulation thread to terminate...")
            sim_thread.join(timeout=5.0)

            # Force exit if still alive
            if sim_thread.is_alive():
                print("Simulation thread did not terminate gracefully. Forcing exit...")
                if hasattr(simulation, 'sim_task') and simulation.sim_task:
                    try:
                        loop.call_soon_threadsafe(simulation.sim_task.cancel)
                    except Exception as e:
                        print(f"Error cancelling simulation task: {e}")

        # Clean up event loop
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Wait for a short time for cancellation
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=1.0))

            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except (RuntimeError, AttributeError) as e:
            print(f"Error during cleanup: {e}")

    except Exception as e:
        simulation.logger.error(f"Error in simulation runner: {e}")
        traceback.print_exc()  # Goes to log file
    finally:
        simulation.logger.info("Simulation runner terminated")


if __name__ == "__main__":
    from agent_classes import EnhancedSingleAgentFinalEvolution
    import logging
    import traceback
    logger = setup_logging()
    try:
        logger = setup_logging()
        logger.info("\n🔄 Initializing agent...")
        agent = EnhancedSingleAgentFinalEvolution(num_qubits=4)
        logger.info("✅ Agent initialized successfully")
    except Exception as init_error:
        logger = logging.getLogger("emile4.simulation")
        logger.error(f"❌ Error initializing agent: {init_error}")
        traceback.print_exc()  # Goes to log file
        exit(1)

    # Run interactive simulation
    run_interactive_simulation(agent)




