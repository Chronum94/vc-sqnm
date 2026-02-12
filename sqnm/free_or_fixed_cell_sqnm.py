from __future__ import annotations

import sys

import numpy as np

from sqnm.sqnm import SQNM


class FreeSQNM:
    def __init__(
        self,
        nat: int,
        initial_step_size: float,
        nhist_max: int,
        alpha_min: float,
        eps_subsp: float,
    ):
        self.nat = nat
        self.ndim = 3 * nat
        if self.ndim < nhist_max:
            print(
                "Subspace dimensions exceed problem dimensions. Reducing nhist_max.",
                file=sys.stderr,
            )
            nhist_max = self.ndim
        self.optimizer = SQNM(
            self.ndim, nhist_max, initial_step_size, eps_subsp, alpha_min
        )
        self.fluct = 0.0

    def step(self, pos: np.ndarray, epot: float, forces: np.ndarray) -> np.ndarray:
        fnoise = np.linalg.norm(np.sum(forces, axis=1)) / np.sqrt(3 * self.nat)
        self.fluct = fnoise if self.fluct == 0.0 else 0.8 * self.fluct + 0.2 * fnoise
        if self.fluct > 0.2 * np.max(np.abs(forces)):
            print(
                "Warning: noise in forces is larger than 0.2 times the largest "
                "force component. Convergence is not guaranteed.",
                file=sys.stderr,
            )
        flat_pos = pos.ravel()
        flat_pos = flat_pos + self.optimizer.step(flat_pos, epot, -forces.ravel())
        return flat_pos.reshape(3, self.nat)

    def lower_bound(self) -> float:
        return self.optimizer.lower_bound()
