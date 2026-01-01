# Ferroelectric Polarization Switching Simulation

This repository contains Python scripts to simulate **ferroelectric polarization switching** using a Landau free energy model, with stochastic effects to mimic device variability.

## Features
- Polarization dynamics under applied electric fields
- Hysteresis (P–E) loop plotting
- Stochastic switching for probabilistic device behavior
- Time-domain polarization analysis

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation script to produce P–E hysteresis and time-domain polarization plots:

```bash
python polarization_sim.py --E0 1.0 --freq 1.0 --duration 10 --dt 1e-3 --noise 0.02
```

The script saves `pe_hysteresis.png` and `polarization_time.png` (prefix configurable with `--out`).

