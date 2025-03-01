import time
import numpy as np
import traceback
from typing import Dict, Tuple, List, Any, Optional
from collections import deque

class EmergenceEvent:
    """Represents a dimensional emergence event with enhanced tracking capabilities."""
    def __init__(self, shape, timestamp, metrics=None, resource_data=None):
        """
        Initialize an emergence event with shape, timing, and associated metrics.

        Args:
            shape: The tensor shape where emergence was detected
            timestamp: Time when emergence was detected
            metrics: Optional metrics associated with the emergence event
            resource_data: Optional resource usage data during emergence
        """
        self.shape = shape
        self.dimensionality = len(shape) if isinstance(shape, (tuple, list)) else 0
        self.timestamp = timestamp
        self.metrics = metrics or {}
        self.resource_data = resource_data or {}
        self.duration = 0
        self.previous_dimensionality = 3  # Default assumption
        self.coherence_before = self.metrics.get('coherence', 0.0)
        self.stability_before = self.metrics.get('stability', 0.0)
        self.emergence_intensity = 0.0  # Will be calculated when emergence ends

    def update_with_end_data(self, end_timestamp, end_metrics=None):
        """Update the event with data from when emergence ends."""
        self.duration = end_timestamp - self.timestamp
        if end_metrics:
            self.coherence_after = end_metrics.get('coherence', 0.0)
            self.stability_after = end_metrics.get('stability', 0.0)
            # Calculate emergence intensity from metrics change
            coherence_delta = abs(self.coherence_after - self.coherence_before)
            stability_delta = abs(self.stability_after - self.stability_before)
            self.emergence_intensity = (coherence_delta + stability_delta) / 2

    def __str__(self):
        return f"EmergenceEvent: {self.dimensionality}D {self.shape} at {self.timestamp:.2f}s (duration: {self.duration:.2f}s)"

class EmergenceTracker:
    """Tracks and analyzes emergent computational phenomena with enhanced pattern recognition."""
    def __init__(self):
        self.emergence_events = []
        self.dimension_transitions = []
        self.stability_history = []
        self.last_emergence_time = 0
        self.emergence_duration = 0
        self.is_emergence_active = False
        self.active_emergence_event = None

        # Resource monitoring data
        self.baseline_resource_usage = {'cpu_percent': 0, 'memory_percent': 0}
        self.resource_samples_count = 0

        # Enhanced emergence pattern analysis
        self.emergence_patterns = {}
        self.resource_correlation = []
        self.emergence_prediction_model = None
        self.occurrence_frequency = {}
        self.pattern_sequences = deque(maxlen=10)
        self.emergence_periodicity = None

        # Additional metrics tracking
        self.coherence_during_emergence = []
        self.distinction_during_emergence = []
        self.performance_impact = []

    def monitor_resources(self) -> Dict[str, float]:
        """Enhanced resource monitoring with better memory management."""
        try:
            import psutil
            import gc

            # Run garbage collection to clean up unused objects
            gc.collect()

            # Get process-specific info
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()

            # Get system-wide info
            system_memory = psutil.virtual_memory()

            resource_data = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': system_memory.percent,
                'memory_used_mb': memory_info.rss / (1024 * 1024),  # Process memory in MB
                'memory_available_mb': system_memory.available / (1024 * 1024),  # System available in MB
                'memory_total_mb': system_memory.total / (1024 * 1024)  # Total system memory in MB
            }

            # Track resource history
            if hasattr(self, 'resource_history'):
                self.resource_history.append(resource_data)
                # Limit history size
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]

            return resource_data
        except Exception as e:
            print(f"Error monitoring resources: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available_mb': 0.0
            }

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

    def record_emergence(self, tensor_shape: Tuple, timestamp: float,
                     resource_usage: Dict, agent_metrics: Dict) -> Dict:
        """
        Record and analyze an emergence event with enhanced pattern tracking.

        Args:
            tensor_shape: The tensor shape where emergence was detected
            timestamp: Time when emergence was detected
            resource_usage: Resource usage data during emergence
            agent_metrics: Agent state metrics during emergence

        Returns:
            Dictionary with emergence data
        """
        try:
            # Calculate the total dimensionality and elements
            total_elements = np.prod(tensor_shape)

            # Create emergence data record
            emergence_data = {
                'timestamp': timestamp,
                'tensor_shape': tensor_shape,
                'total_elements': total_elements,
                'resource_usage': resource_usage.copy(),
                'agent_metrics': agent_metrics.copy(),
                'elapsed_time': 0,
                'dimensionality': len(tensor_shape),
                'previous_dimensionality': 3 if not self.dimension_transitions else self.dimension_transitions[-1]['new_dimensionality']
            }

            # Record dimension transition
            if not self.dimension_transitions or self.dimension_transitions[-1]['new_dimensionality'] != len(tensor_shape):
                self.dimension_transitions.append({
                    'timestamp': timestamp,
                    'old_dimensionality': 3 if not self.dimension_transitions else self.dimension_transitions[-1]['new_dimensionality'],
                    'new_dimensionality': len(tensor_shape),
                    'stability_before': agent_metrics.get('stability', 0),
                    'coherence_before': agent_metrics.get('coherence', 0)
                })

            # If this is the start of a new emergence period
            if not self.is_emergence_active:
                self.is_emergence_active = True
                self.last_emergence_time = timestamp
                # Create new emergence event object
                self.active_emergence_event = EmergenceEvent(
                    tensor_shape, timestamp, agent_metrics, resource_usage
                )
                # Record baseline resource usage
                if self.resource_samples_count < 5:  # If we don't have good baseline data yet
                    self.baseline_resource_usage = resource_usage.copy()
                    self.resource_samples_count = 1
                else:
                    # Update baseline with exponential moving average
                    for key in ['cpu_percent', 'memory_percent']:
                        if key in resource_usage and key in self.baseline_resource_usage:
                            self.baseline_resource_usage[key] = 0.9 * self.baseline_resource_usage[key] + 0.1 * resource_usage[key]
            else:
                # Update existing emergence period
                emergence_data['elapsed_time'] = timestamp - self.last_emergence_time
                self.emergence_duration += emergence_data['elapsed_time']
                self.last_emergence_time = timestamp

                # Track metrics during emergence
                if agent_metrics:
                    self.coherence_during_emergence.append(agent_metrics.get('coherence', 0))
                    self.distinction_during_emergence.append(agent_metrics.get('distinction_level', 0))

                # Track performance impact
                if resource_usage and self.baseline_resource_usage:
                    cpu_impact = resource_usage.get('cpu_percent', 0) - self.baseline_resource_usage.get('cpu_percent', 0)
                    memory_impact = resource_usage.get('memory_percent', 0) - self.baseline_resource_usage.get('memory_percent', 0)
                    self.performance_impact.append((cpu_impact, memory_impact))

            # Track and analyze pattern
            shape_key = 'x'.join(str(dim) for dim in tensor_shape)
            if shape_key not in self.emergence_patterns:
                self.emergence_patterns[shape_key] = {
                    'count': 0,
                    'first_seen': timestamp,
                    'resource_impact': [],
                    'associated_metrics': [],
                    'durations': []
                }

            pattern = self.emergence_patterns[shape_key]
            pattern['count'] += 1
            pattern['last_seen'] = timestamp

            # Record resource impact
            for key in ['cpu_percent', 'memory_percent']:
                if key in resource_usage and key in self.baseline_resource_usage:
                    impact = resource_usage[key] - self.baseline_resource_usage[key]
                    pattern['resource_impact'].append(impact)

            # Record associated metrics
            pattern['associated_metrics'].append({
                'distinction': agent_metrics.get('distinction_level', 0),
                'coherence': agent_metrics.get('coherence', 0),
                'stability': agent_metrics.get('stability', 0)
            })

            # Update occurrence frequency for periodicity analysis
            hours_bucket = int(timestamp // 3600)
            minutes_bucket = int(timestamp // 60)
            self.occurrence_frequency[hours_bucket] = self.occurrence_frequency.get(hours_bucket, 0) + 1

            # Add to pattern sequence for sequence analysis
            self.pattern_sequences.append(shape_key)

            # Calculate correlation between emergence and resources
            if len(self.emergence_events) > 5:
                recent_events = self.emergence_events[-5:]
                cpu_trend = [e['resource_usage']['cpu_percent'] for e in recent_events]
                coherence_trend = [e['agent_metrics'].get('coherence', 0) for e in recent_events]

                if len(cpu_trend) > 1 and len(coherence_trend) > 1:
                    cpu_changes = np.diff(cpu_trend)
                    coherence_changes = np.diff(coherence_trend)

                    if len(cpu_changes) > 0 and len(coherence_changes) > 0:
                        try:
                            correlation = self._safe_correlation(cpu_changes, coherence_changes)
                            emergence_data['resource_correlation'] = correlation
                            self.resource_correlation.append(correlation)
                        except Exception as e:
                            emergence_data['resource_correlation'] = 0
                            print(f"Error calculating correlation: {e}")

            self.emergence_events.append(emergence_data)
            self._try_detect_periodicity()
            return emergence_data

        except Exception as e:
            print(f"Error in record_emergence: {e}")
            traceback.print_exc()
            return {'error': str(e), 'timestamp': timestamp}

    def end_emergence(self, timestamp: float, metrics: Dict = None) -> Dict:
        """
        Mark the end of an emergence event and complete the analysis.

        Args:
            timestamp: Time when emergence ended
            metrics: Current agent metrics

        Returns:
            Dictionary with emergence summary data
        """
        if not hasattr(self, 'is_emergence_active') or not self.is_emergence_active or not hasattr(self, 'active_emergence_event') or self.active_emergence_event is None:
            return {"status": "No active emergence to end"}

        try:
            # Calculate final duration
            duration = timestamp - self.active_emergence_event.timestamp

            # Update the active emergence event
            self.active_emergence_event.update_with_end_data(timestamp, metrics)

            # Add the completed event to our events list
            self.emergence_events.append(self.active_emergence_event)

            # Update pattern durations
            if hasattr(self.active_emergence_event, 'shape'):
                shape_key = 'x'.join(str(dim) for dim in self.active_emergence_event.shape)
                if hasattr(self, 'emergence_patterns') and shape_key in self.emergence_patterns:
                    self.emergence_patterns[shape_key]['durations'].append(duration)

            # Reset active emergence tracking
            self.is_emergence_active = False
            self.active_emergence_event = None
            if hasattr(self, 'coherence_during_emergence'):
                self.coherence_during_emergence = []
            if hasattr(self, 'distinction_during_emergence'):
                self.distinction_during_emergence = []
            if hasattr(self, 'performance_impact'):
                self.performance_impact = []

            return {
                "status": "Emergence ended",
                "duration": duration,
                "timestamp": timestamp
            }

        except Exception as e:
            print(f"Error ending emergence: {e}")
            traceback.print_exc()
            self.is_emergence_active = False
            self.active_emergence_event = None
            return {"status": "Error ending emergence", "error": str(e)}

    def update_stability(self, stability: float, timestamp: float):
        """
        Track stability during emergence for correlation analysis.

        Args:
            stability: Current stability metric
            timestamp: Current timestamp
        """
        self.stability_history.append({
            'timestamp': timestamp,
            'stability': stability
        })

    def get_emergence_summary(self) -> Dict:
        """
        Get a comprehensive summary of emergence information with pattern analysis.

        Returns:
            Dictionary of emergence summary statistics and patterns
        """
        if not self.emergence_events:
            return {"status": "No emergence detected"}

        try:
            # Get latest emergence data
            latest = self.emergence_events[-1] if isinstance(self.emergence_events[-1], dict) else vars(self.emergence_events[-1])

            # Count dimensions across events
            dimension_counts = {}
            for event in self.emergence_events:
                if isinstance(event, dict):
                    dim = event['dimensionality']
                else:
                    dim = event.dimensionality
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

            # Calculate average stability change during emergence
            stability_during_emergence = 0
            if len(self.stability_history) > 1:
                stability_changes = [abs(s['stability'] - self.stability_history[i-1]['stability'])
                                   for i, s in enumerate(self.stability_history) if i > 0]
                if stability_changes:
                    stability_during_emergence = sum(stability_changes) / len(stability_changes)

            # Calculate resource impact
            avg_cpu_impact = 0
            avg_memory_impact = 0
            if self.performance_impact:
                cpu_impacts = [impact[0] for impact in self.performance_impact]
                memory_impacts = [impact[1] for impact in self.performance_impact]
                avg_cpu_impact = sum(cpu_impacts) / len(cpu_impacts) if cpu_impacts else 0
                avg_memory_impact = sum(memory_impacts) / len(memory_impacts) if memory_impacts else 0

            # Calculate pattern diversity
            pattern_diversity = len(self.emergence_patterns) / max(1, len(self.emergence_events))

            # Calculate correlation between resource usage and emergence
            resource_correlation = np.mean(self.resource_correlation) if self.resource_correlation else 0

            # Identify most common pattern
            most_common_pattern = max(self.emergence_patterns.items(),
                                    key=lambda x: x[1]['count']) if self.emergence_patterns else (None, {})
            most_common_pattern_key = most_common_pattern[0]
            most_common_pattern_data = most_common_pattern[1]

            return {
                "status": "Active" if self.is_emergence_active else "Inactive",
                "emergence_count": len(self.emergence_events),
                "current_dimensionality": latest.get('dimensionality', 0)
                    if isinstance(latest, dict) else latest.get('dimensionality', 0),
                "dimension_history": dimension_counts,
                "current_shape": latest.get('tensor_shape', None)
                    if isinstance(latest, dict) else latest.get('shape', None),
                "total_duration": self.emergence_duration,
                "dimension_transitions": len(self.dimension_transitions),
                "stability_impact": stability_during_emergence,
                "resource_impact": avg_cpu_impact,
                "memory_impact": avg_memory_impact,
                "pattern_diversity": pattern_diversity,
                "resource_correlation": resource_correlation,
                "most_common_pattern": most_common_pattern_key,
                "most_common_count": most_common_pattern_data.get('count', 0),
                "periodicity": self.emergence_periodicity
            }

        except Exception as e:
            print(f"Error getting emergence summary: {e}")
            traceback.print_exc()
            return {
                "status": "Error in summary generation",
                "emergence_count": len(self.emergence_events),
                "error": str(e)
            }

    def get_detailed_analysis(self) -> Dict:
        """
        Get detailed analysis of emergence patterns with comprehensive metrics.

        Returns:
            Dictionary with detailed pattern analysis
        """
        if not self.emergence_patterns:
            return {"status": "No patterns to analyze"}

        try:
            pattern_analysis = {}
            total_events = sum(pattern.get('count', 0) for pattern in self.emergence_patterns.values())

            for shape_key, pattern in self.emergence_patterns.items():
                if pattern['count'] < 2:
                    continue

                # Calculate stability during this pattern
                stability_values = [m['stability'] for m in pattern['associated_metrics']]
                coherence_values = [m['coherence'] for m in pattern['associated_metrics']]
                distinction_values = [m['distinction'] for m in pattern['associated_metrics']]

                # Calculate time between occurrences if this pattern was seen multiple times
                if 'first_seen' in pattern and 'last_seen' in pattern and pattern['count'] > 1:
                    time_span = pattern['last_seen'] - pattern['first_seen']
                    frequency = pattern['count'] / time_span if time_span > 0 else 0
                else:
                    frequency = 0

                # Calculate average duration if available
                avg_duration = sum(pattern.get('durations', [0])) / len(pattern.get('durations', [1])) if pattern.get('durations') else 0

                pattern_analysis[shape_key] = {
                    "count": pattern['count'],
                    "frequency": frequency,
                    "avg_duration": avg_duration,
                    "proportion": pattern['count'] / total_events if total_events else 0,
                    "duration": pattern.get('last_seen', 0) - pattern.get('first_seen', 0),
                    "avg_cpu_impact": np.mean(pattern['resource_impact']) if pattern['resource_impact'] else 0,
                    "stability_mean": np.mean(stability_values) if stability_values else 0,
                    "stability_variance": np.var(stability_values) if stability_values else 0,
                    "coherence_mean": np.mean(coherence_values) if coherence_values else 0,
                    "coherence_variance": np.var(coherence_values) if coherence_values else 0,
                    "distinction_mean": np.mean(distinction_values) if distinction_values else 0,
                    "distinction_variance": np.var(distinction_values) if distinction_values else 0
                }

            # Find sequences of patterns that commonly occur together
            sequence_patterns = {}
            if len(self.pattern_sequences) >= 3:
                for i in range(len(self.pattern_sequences) - 2):
                    seq = (self.pattern_sequences[i], self.pattern_sequences[i+1], self.pattern_sequences[i+2])
                    sequence_patterns[seq] = sequence_patterns.get(seq, 0) + 1

            # Get most common sequence
            common_sequence = max(sequence_patterns.items(), key=lambda x: x[1]) if sequence_patterns else (None, 0)

            # Find dimension transition patterns
            dimension_transition_patterns = {}
            if len(self.dimension_transitions) >= 2:
                for i in range(len(self.dimension_transitions) - 1):
                    dt_pair = (
                        self.dimension_transitions[i]['new_dimensionality'],
                        self.dimension_transitions[i+1]['new_dimensionality']
                    )
                    dimension_transition_patterns[dt_pair] = dimension_transition_patterns.get(dt_pair, 0) + 1

            return {
                "total_patterns": len(self.emergence_patterns),
                "total_events": total_events,
                "pattern_details": pattern_analysis,
                "dominant_pattern": max(self.emergence_patterns.items(),
                                     key=lambda x: x[1]['count'])[0] if self.emergence_patterns else None,
                "common_sequences": common_sequence[0] if common_sequence[0] else None,
                "sequence_frequency": common_sequence[1] if common_sequence[1] else 0,
                "dimension_transition_patterns": dimension_transition_patterns
            }

        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            traceback.print_exc()
            return {"status": "Error generating detailed analysis", "error": str(e)}

    def _try_detect_periodicity(self):
        """Attempt to detect periodicity in emergence events."""
        try:
            if len(self.emergence_events) < 5:
                return

            # Extract timestamps
            if isinstance(self.emergence_events[0], dict):
                timestamps = [e['timestamp'] for e in self.emergence_events]
            else:
                timestamps = [e.timestamp for e in self.emergence_events]

            # Calculate time differences between consecutive events
            time_diffs = np.diff(timestamps)

            # Check if time differences are consistent
            if len(time_diffs) >= 3:
                mean_diff = np.mean(time_diffs)
                std_diff = np.std(time_diffs)

                # If standard deviation is less than 20% of the mean, we have periodicity
                if std_diff < 0.2 * mean_diff:
                    self.emergence_periodicity = {
                        'period': mean_diff,
                        'confidence': 1.0 - (std_diff / mean_diff),
                        'unit': 'seconds'
                    }
                else:
                    self.emergence_periodicity = None
        except Exception as e:
            print(f"Error detecting periodicity: {e}")
            self.emergence_periodicity = None

class DimensionMonitor:
    """Monitors tensor shapes and captures emergence events with improved transition detection."""
    def __init__(self):
        """Initialize the dimension monitor with empty tracking structures."""
        self.past_shapes = {}  # {id: shape} mapping
        self.shape_transitions = []
        self.dimension_changes = []
        self.last_dimensionality = 3  # Start with standard tensor dimensionality
        self.transition_frequencies = {}  # Track frequency of specific transitions
        self.stable_periods = []  # Track periods of dimensional stability
        self.unstable_periods = []  # Track periods of dimensional instability
        self.stability_threshold = 10  # Number of consecutive identical dimensionality observations
        self.current_stability_count = 0
        self.current_shape = None
        self.dimensionality_histogram = {}  # Track frequency of each dimensionality

        # Transition pattern tracking
        self.transition_sequence = deque(maxlen=20)  # Track recent dimension transitions
        self.transition_patterns = {}  # Common transition sequences
        self.current_stable_period_start = time.time()
        self.last_transition_time = None

    def register_shape(self, tensor_object, tag="unknown"):
        """
        Register a tensor and check for dimensional changes with enhanced transition detection.

        Args:
            tensor_object: The tensor to analyze for dimensional emergence
            tag: Optional tag to identify the source of this tensor

        Returns:
            EmergenceEvent if dimensional emergence detected, None otherwise
        """
        if not hasattr(tensor_object, 'shape'):
            return None

        try:
            shape = tensor_object.shape
            obj_id = id(tensor_object)

            # Check if we've seen this object before with a different shape
            dimensionality = len(shape)
            event = None

            # Update dimensionality histogram
            self.dimensionality_histogram[dimensionality] = self.dimensionality_histogram.get(dimensionality, 0) + 1

            # Check for shape change in this specific tensor object
            if obj_id in self.past_shapes and len(self.past_shapes[obj_id]) != dimensionality:
                # Dimensional emergence detected!
                event = EmergenceEvent(
                    shape=shape,
                    timestamp=time.time(),
                    metrics={'old_shape': self.past_shapes[obj_id], 'tag': tag,
                             'transition_type': 'object_specific'}
                )
                self.dimension_changes.append({
                    'timestamp': time.time(),
                    'obj_id': obj_id,
                    'old_shape': self.past_shapes[obj_id],
                    'new_shape': shape,
                    'old_dimensionality': len(self.past_shapes[obj_id]),
                    'new_dimensionality': dimensionality,
                    'tag': tag,
                    'transition_type': 'object_specific'
                })

                # Track this specific transition
                transition_key = f"{len(self.past_shapes[obj_id])}D→{dimensionality}D"
                self.transition_frequencies[transition_key] = self.transition_frequencies.get(transition_key, 0) + 1

                # Reset stability counter
                self.current_stability_count = 0

                # Record transition time
                current_time = time.time()
                if self.last_transition_time is not None:
                    transition_interval = current_time - self.last_transition_time
                    # Add to unstable periods if transitions are happening rapidly
                    if transition_interval < 5.0:  # Less than 5 seconds between transitions
                        self.unstable_periods.append({
                            'start': self.last_transition_time,
                            'end': current_time,
                            'duration': transition_interval,
                            'transition': transition_key
                        })
                self.last_transition_time = current_time

            # Update shape record
            self.past_shapes[obj_id] = shape

            # If dimensionality changed globally
            if dimensionality != self.last_dimensionality:
                self.shape_transitions.append({
                    'timestamp': time.time(),
                    'old_dimensionality': self.last_dimensionality,
                    'new_dimensionality': dimensionality,
                    'shape': shape,
                    'tag': tag,
                    'transition_type': 'global'
                })

                # Add to transition sequence and track patterns
                transition_str = f"{self.last_dimensionality}→{dimensionality}"
                self.transition_sequence.append(transition_str)

                # Track transition patterns (sequence of 3 transitions)
                if len(self.transition_sequence) >= 3:
                    pattern = tuple(list(self.transition_sequence)[-3:])
                    self.transition_patterns[pattern] = self.transition_patterns.get(pattern, 0) + 1

                # Record transition key
                transition_key = f"{self.last_dimensionality}D→{dimensionality}D"
                self.transition_frequencies[transition_key] = self.transition_frequencies.get(transition_key, 0) + 1

                # End current stable period if it was stable
                if self.current_stability_count >= self.stability_threshold:
                    current_time = time.time()
                    self.stable_periods.append({
                        'start': self.current_stable_period_start,
                        'end': current_time,
                        'duration': current_time - self.current_stable_period_start,
                        'dimensionality': self.last_dimensionality
                    })

                # Start new stable period tracking
                self.current_stable_period_start = time.time()
                self.current_stability_count = 0
                self.last_dimensionality = dimensionality

                # If no event was created yet, create one for global dimensional change
                if event is None:
                    event = EmergenceEvent(
                        shape=shape,
                        timestamp=time.time(),
                        metrics={'global_transition': True, 'tag': tag,
                                 'transition_type': 'global'}
                    )
            else:
                # Increment stability counter for same dimensionality
                self.current_stability_count += 1

                # If we've reached stability threshold, record it
                if self.current_stability_count == self.stability_threshold:
                    print(f"Dimension {dimensionality}D has stabilized")

            # Update current shape
            self.current_shape = shape

            return event

        except Exception as e:
            print(f"Error in register_shape: {e}")
            traceback.print_exc()
            return None

    def get_transition_statistics(self) -> Dict:
        """
        Get statistics about dimensional transitions with pattern analysis.

        Returns:
            Dictionary of transition statistics
        """
        try:
            stats = {
                'total_transitions': len(self.shape_transitions),
                'transition_frequencies': self.transition_frequencies,
                'stable_periods_count': len(self.stable_periods),
                'unstable_periods_count': len(self.unstable_periods),
                'current_dimensionality': self.last_dimensionality,
                'dimensionality_histogram': self.dimensionality_histogram,
            }

            # Average stable period duration
            if self.stable_periods:
                avg_stable_duration = sum(period['duration'] for period in self.stable_periods) / len(self.stable_periods)
                stats['avg_stable_duration'] = avg_stable_duration

            # Average unstable period duration
            if self.unstable_periods:
                avg_unstable_duration = sum(period['duration'] for period in self.unstable_periods) / len(self.unstable_periods)
                stats['avg_unstable_duration'] = avg_unstable_duration

            # Most common transition
            if self.transition_frequencies:
                most_common_transition = max(self.transition_frequencies.items(), key=lambda x: x[1])
                stats['most_common_transition'] = most_common_transition[0]
                stats['most_common_transition_count'] = most_common_transition[1]

            # Most common transition pattern
            if self.transition_patterns:
                most_common_pattern = max(self.transition_patterns.items(), key=lambda x: x[1])
                stats['most_common_pattern'] = most_common_pattern[0]
                stats['most_common_pattern_count'] = most_common_pattern[1]

            # Most common dimensionality
            if self.dimensionality_histogram:
                most_common_dim = max(self.dimensionality_histogram.items(), key=lambda x: x[1])
                stats['most_common_dimensionality'] = most_common_dim[0]
                stats['most_common_dimensionality_count'] = most_common_dim[1]

                # Calculate proportions
                total_observations = sum(self.dimensionality_histogram.values())
                stats['dimensionality_proportions'] = {
                    dim: count / total_observations
                    for dim, count in self.dimensionality_histogram.items()
                }

            return stats

        except Exception as e:
            print(f"Error getting transition statistics: {e}")
            traceback.print_exc()
            return {'error': str(e)}
