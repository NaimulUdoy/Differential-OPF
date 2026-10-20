"""
Configuration parameters for the Differentiable OPF network model.
"""
import numpy as np

# Network dimensions
NUM_BUSES = 3
NUM_LINES = 2

# Line reactances and limits
X = np.array([0.1, 0.15])
P_LINE_MAX = np.array([50.0, 40.0])

# Nominal active power loads (p.u.)
PD_NOMINAL = np.array([0.0, 1.2, 1.8])