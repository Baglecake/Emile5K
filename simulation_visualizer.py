from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SaveStatevector
from qiskit import QuantumCircuit
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Dict, List, Any, Union, Optional, Tuple, TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    from agent_classes import EnhancedSingleAgentFinalEvolution

class SimulationVisualizer:
    """Handles real-time visualization of the quantum consciousness simulation."""
    def __init__(self, figsize: Tuple[int, int] = (15, 10), interactive: bool = True):
        """
        Initialize the visualization system.

        Args:
            figsize: Figure size as (width, height)
            interactive: Whether to use interactive mode for real-time updates
        """
        try:
            self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
            self.history = defaultdict(list)  # Automatically initializes empty lists for keys
            self.max_history_length = 1000  # Maximum number of points to keep
            self.plot_colors = {
                'distinction': 'b',
                'coherence': 'g',
                'entropy': 'r',
                'stability': 'y',
                'surplus_stability': 'm',
                'phase': 'c'
            }

            # Set titles and labels
            self.axes[0, 0].set_title('Distinction Evolution')
            self.axes[0, 1].set_title('Quantum Coherence')
            self.axes[1, 0].set_title('System Entropy')
            self.axes[1, 1].set_title('System Stability')

            # Initialize line objects for faster updating
            self.lines = {}
            for i, key in enumerate(['distinction', 'coherence', 'entropy', 'stability']):
                row, col = i // 2, i % 2
                self.lines[key], = self.axes[row, col].plot(
                    [], [], f'{self.plot_colors[key]}-', label=key.capitalize()
                )
                self.axes[row, col].legend()
                self.axes[row, col].set_ylim(0, 1)
                self.axes[row, col].set_xlabel('Steps')
                self.axes[row, col].set_ylabel('Value')
                self.axes[row, col].grid(True, linestyle='--', alpha=0.7)

            # Enable interactive mode if requested
            if interactive:
                plt.ion()

            plt.tight_layout()

            # Visualization state
            self.paused = False
            self.is_closed = False
            self.update_counter = 0

            print("✅ Visualization system initialized")

        except Exception as e:
            print(f"❌ Error initializing visualization: {e}")
            traceback.print_exc()
            self.is_closed = True

    def _safe_float_conversion(self, value: Any) -> float:
        """
        Safely convert potentially complex values to float with enhanced error handling.

        Args:
            value: Value to convert to float

        Returns:
            Converted float value, or 0.0 if conversion fails
        """
        try:
            if value is None:
                return 0.0

            if isinstance(value, complex) or hasattr(value, 'imag'):
                return float(np.abs(value))  # Use magnitude for complex numbers

            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                # Take first element from sequences
                return self._safe_float_conversion(value[0])

            return float(value)

        except (TypeError, ValueError, IndexError) as e:
            # More specific error handling
            print(f"⚠️ Conversion error ({type(value)}): {e}")
            return 0.0
        except Exception as e:
            print(f"❌ Unexpected error in float conversion: {e}")
            return 0.0

    def _trim_history(self):
        """Trim history to maximum length to prevent memory issues."""
        for key in self.history:
            if len(self.history[key]) > self.max_history_length:
                self.history[key] = self.history[key][-self.max_history_length:]

    def update(self, metrics: Dict[str, Any], update_interval: int = 1):
        """
        Update the visualization with new simulation metrics.

        Args:
            metrics: Dictionary of metrics to visualize
            update_interval: Only update the plot every N calls to reduce overhead
        """
        if self.is_closed or self.paused:
            return

        try:
            # Update counters
            self.update_counter += 1

            # Always update the history
            self._update_history(metrics)

            # Only update the visualization at specified intervals
            if self.update_counter % update_interval != 0:
                return

            # Update plot data
            for key, line in self.lines.items():
                if key in self.history and self.history[key]:
                    data = self.history[key]
                    x_data = list(range(len(data)))
                    line.set_data(x_data, data)

                    # Dynamically update x-axis limits
                    ax = line.axes
                    ax.set_xlim(0, max(100, len(data)))

                    # Update y-axis if values exceed current limits
                    y_min, y_max = ax.get_ylim()
                    data_min, data_max = min(data), max(data)

                    # Add padding to y-axis
                    data_range = max(0.1, data_max - data_min)
                    new_min = max(0, data_min - 0.1 * data_range)
                    new_max = data_max + 0.1 * data_range

                    if new_min < y_min or new_max > y_max:
                        ax.set_ylim(new_min, new_max)

            # Refresh the figure
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"❌ Error updating visualization: {e}")
            traceback.print_exc()

    def _update_history(self, metrics: Dict[str, Any]):
        """
        Update history with new metrics.

        Args:
            metrics: Dictionary of metrics to add to history
        """
        try:
            # Update main metrics
            for key in ['distinction_level', 'coherence', 'entropy', 'stability']:
                # Handle different naming conventions
                actual_key = key
                if key == 'distinction_level':
                    actual_key = 'distinction'

                # Get value using both possible key names
                value = metrics.get(key, metrics.get(actual_key, None))

                if value is not None:
                    self.history[actual_key].append(self._safe_float_conversion(value))

            # Add additional metrics if available
            additional_metrics = [
                ('surplus_stability', ['surplus_stability', 'surplus_state_stability']),
                ('phase', ['phase', 'quantum_phase'])
            ]

            for hist_key, possible_keys in additional_metrics:
                for metric_key in possible_keys:
                    if metric_key in metrics:
                        self.history[hist_key].append(self._safe_float_conversion(metrics[metric_key]))
                        break

            # Trim history to prevent memory issues
            self._trim_history()

        except Exception as e:
            print(f"❌ Error updating history: {e}")
            traceback.print_exc()

    def add_subplot(self, key: str, title: str, color: str = None):
        """
        Add a new subplot for a custom metric.

        Args:
            key: Key of the metric to plot
            title: Title for the subplot
            color: Line color (if None, a default will be chosen)
        """
        if self.is_closed:
            return

        try:
            # Create a new figure for the additional plot
            new_fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)

            # Choose color
            if color is None:
                color = self.plot_colors.get(key, 'b')

            # Create line
            line, = ax.plot([], [], f'{color}-', label=key.capitalize())
            ax.legend()

            # Register the new line
            self.lines[key] = line

            # Ensure history exists for this key
            if key not in self.history:
                self.history[key] = []

            plt.tight_layout()

            return new_fig, ax

        except Exception as e:
            print(f"❌ Error adding subplot: {e}")
            traceback.print_exc()
            return None, None

    def save_plots(self, filename_prefix: str = "simulation"):
        """
        Save all plots to files.

        Args:
            filename_prefix: Prefix for the saved files
        """
        if self.is_closed:
            return

        try:
            # Save main figure
            self.fig.savefig(f"{filename_prefix}_main.png", dpi=300, bbox_inches='tight')
            print(f"✅ Main plot saved to {filename_prefix}_main.png")

            # Create and save a plot of all history data
            if self.history:
                hist_fig, hist_ax = plt.subplots(figsize=(12, 8))
                for key in self.history:
                    if self.history[key]:
                        hist_ax.plot(
                            self.history[key],
                            label=key.capitalize(),
                            color=self.plot_colors.get(key, 'b')
                        )

                hist_ax.set_title('Complete Simulation History')
                hist_ax.set_xlabel('Steps')
                hist_ax.set_ylabel('Value')
                hist_ax.grid(True, linestyle='--', alpha=0.7)
                hist_ax.legend()

                hist_fig.savefig(f"{filename_prefix}_history.png", dpi=300, bbox_inches='tight')
                plt.close(hist_fig)
                print(f"✅ History plot saved to {filename_prefix}_history.png")

        except Exception as e:
            print(f"❌ Error saving plots: {e}")
            traceback.print_exc()

    def pause(self):
        """Pause the visualization updates."""
        self.paused = True

    def resume(self):
        """Resume the visualization updates."""
        self.paused = False

    def close(self):
        """Properly close the visualization."""
        if self.is_closed:
            return

        try:
            plt.close(self.fig)
            # Close any other figures that might have been created
            plt.close('all')
            self.is_closed = True
            print("✅ Visualization closed")
        except Exception as e:
            print(f"❌ Error closing visualization: {e}")
            traceback.print_exc()
