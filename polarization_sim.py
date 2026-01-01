#!/usr/bin/env python3
"""Ferroelectric polarization switching simulation (Landau model).

Creates P–E hysteresis and time-domain polarization plots using an
overdamped Langevin (Euler-Maruyama) integration of the Landau free energy.

Usage examples:
    python polarization_sim.py --E0 1.0 --freq 1.0 --duration 10 --dt 1e-3 --noise 0.02

"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def landau_force(P: np.ndarray, E: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Compute -dF/dP (driving force) for the Landau free energy."""
    return -(alpha * P + beta * P ** 3 + gamma * P ** 5 - E)


def simulate(
    E_t: np.ndarray,
    t: np.ndarray,
    P0: float = 0.0,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    tau: float = 0.01,
    noise: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate polarization dynamics using Euler-Maruyama integration.

    dP = (1/tau) * landau_force(P, E) * dt + noise * sqrt(dt) * N(0,1)
    """
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    P = np.empty_like(t)
    P[0] = P0
    for i in range(1, len(t)):
        E = E_t[i - 1]
        force = landau_force(P[i - 1], E, alpha, beta, gamma)
        dP_det = (1.0 / tau) * force * dt
        dP_stoch = noise * np.sqrt(dt) * rng.standard_normal()
        P[i] = P[i - 1] + dP_det + dP_stoch
    return P


def make_field(t: np.ndarray, E0: float, freq: float, kind: str = "sin") -> np.ndarray:
    if kind == "sin":
        return E0 * np.sin(2 * np.pi * freq * t)
    elif kind == "tri":
        # triangle wave using sawtooth from numpy
        return E0 * (2 * np.abs(2 * (freq * t - np.floor(freq * t + 0.5))) - 1)
    else:
        raise ValueError("Unknown field kind: choose 'sin' or 'tri'")


def plot_results(t: np.ndarray, E_t: np.ndarray, P: np.ndarray, out_prefix: str = "") -> Tuple[str, str]:
    pe_path = f"{out_prefix}pe_hysteresis.png"
    time_path = f"{out_prefix}polarization_time.png"

    plt.figure(figsize=(6, 5))
    plt.plot(E_t, P, lw=0.8)
    plt.xlabel("Electric field E")
    plt.ylabel("Polarization P")
    plt.title("P–E Hysteresis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pe_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 3.5))
    plt.plot(t, P, label="P(t)")
    plt.plot(t, E_t / np.max(np.abs(E_t)) * np.max(np.abs(P)) * 0.9, "--", label="scaled E(t)")
    plt.xlabel("Time")
    plt.ylabel("Polarization / scaled Field")
    plt.title("Polarization vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(time_path, dpi=200)
    plt.close()

    return pe_path, time_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ferroelectric polarization switching simulation")
    parser.add_argument("--E0", type=float, default=1.0, help="Field amplitude")
    parser.add_argument("--freq", type=float, default=1.0, help="Field frequency (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time-step")
    parser.add_argument("--noise", type=float, default=0.02, help="Noise amplitude")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--out", type=str, default="", help="Output filename prefix")
    parser.add_argument("--field", choices=("sin", "tri"), default="sin", help="Field waveform type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t = np.arange(0.0, args.duration, args.dt)
    E_t = make_field(t, args.E0, args.freq, kind=args.field)

    start = time.time()
    P = simulate(E_t, t, P0=0.0, alpha=-1.0, beta=1.0, gamma=0.0, tau=0.01, noise=args.noise, seed=args.seed)
    elapsed = time.time() - start

    out_prefix = args.out
    if out_prefix and not out_prefix.endswith("_"):
        out_prefix = out_prefix + "_"

    pe_path, time_path = plot_results(t, E_t, P, out_prefix=out_prefix)

    print(f"Simulation finished in {elapsed:.3f} s")
    print(f"Saved: {pe_path}, {time_path}")


if __name__ == "__main__":
    main()

