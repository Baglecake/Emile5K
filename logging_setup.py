import logging
import os
import sys
from datetime import datetime
import traceback

def setup_logging(log_dir="logs"):
    """
    Configure logging for Émile-2 simulation.

    This redirects all Qiskit and low-level logs to a file while keeping
    the important simulation metrics in the console.

    Args:
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename based on current date/time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"emile4_sim_{timestamp}.log")

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # File handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler for simulation metrics only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Create logger for simulation metrics
    sim_logger = logging.getLogger("emile4.simulation")
    sim_logger.setLevel(logging.INFO)
    sim_logger.addHandler(console_handler)

    # Set up specific logger configurations

    # Qiskit loggers - redirect to file only
    qiskit_loggers = [
        "qiskit.passmanager",
        "qiskit.compiler",
        "qiskit.qobj",
        "qiskit.providers",
        "qiskit.transpiler"
    ]

    for logger_name in qiskit_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        logger.addHandler(file_handler)

    # Other detailed module loggers - redirect to file only
    detail_loggers = [
        "emile4.utilities",
        "emile4.core_quantum",
        "emile4.base_quantum",
        "emile4.data_classes",
        "emile4.memory_field",
        "emile4.analysis",
        "emile4.transformer_modules"
    ]

    for logger_name in detail_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger
        logger.addHandler(file_handler)

    print(f"Logging configured. Detailed logs will be saved to: {log_file}")
    return sim_logger
