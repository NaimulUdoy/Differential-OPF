# ⚡ Differentiable Optimal Power Flow (OPF) Layer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A differentiable Optimal Power Flow (OPF) framework designed for gradient-based grid optimization and end-to-end power system applications. Implemented on a standard **3-bus test system**, this repository embeds power flow equations and operational constraints within a differentiable computational layer to enable gradient propagation through optimization problems.

---

## 🗂️ Repository Structure

* `main.py` — Entry point script executing the optimization loop and gradient updates.
* `diff_opf.py` — Differentiable OPF layer implementation handling KKT conditions or conic/quadratic relaxations.
* `config.py` — System parameters, network topology, line admittances, and generator limits for the 3-bus network.
* `requirements.txt` — Python package dependencies.
* `results/` — Directory containing convergence plots and optimization trajectory outputs.

---

## 🚀 Getting Started

### Prerequisites & Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/NaimulUdoy/Differential-OPF.git](https://github.com/NaimulUdoy/Differential-OPF.git)
cd Differential-OPF
pip install -r requirements.txt
