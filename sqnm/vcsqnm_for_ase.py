from __future__ import annotations

import logging
import os

import numpy as np
from ase.atoms import Atoms
from ase.io import write

from sqnm.free_or_fixed_cell_sqnm import FreeSQNM
from sqnm.periodic_sqnm import PeriodicSQNM


class ASEOptimizer:
    def __init__(
        self,
        initial_structure: Atoms,
        vc_relax: bool = False,
        force_tol: float = 1e-2,
        max_steps: int = 500,
        initial_step_size: float = -0.01,
        nhist_max: int = 10,
        lattice_weight: float = 2.0,
        alpha_min: float = 1e-3,
        eps_subsp: float = 1e-3,
    ):
        self.atoms = initial_structure
        self.vc_relax = vc_relax
        self.force_tol = force_tol
        self.max_steps = max_steps
        self.nat = initial_structure.get_global_number_of_atoms()

        self._extract_info(initial_structure)

        if self.vc_relax:
            self.optimizer = PeriodicSQNM(
                self.nat, self.cell, initial_step_size, nhist_max,
                lattice_weight, alpha_min, eps_subsp,
            )
        else:
            self.optimizer = FreeSQNM(
                self.nat, initial_step_size, nhist_max, alpha_min, eps_subsp,
            )

    def _extract_info(self, atoms: Atoms):
        self.positions = atoms.get_positions().T
        self.forces = atoms.get_forces().T
        self.energy = atoms.get_potential_energy()
        if self.vc_relax:
            self.cell = atoms.get_cell(True).T
            self.stress = atoms.get_stress(voigt=False)
            self.deralat = self._get_lattice_derivative()

    def _get_lattice_derivative(self) -> np.ndarray:
        return -np.linalg.det(self.cell) * self.stress @ np.linalg.inv(self.cell).T

    def step(self, atoms: Atoms):
        self._extract_info(atoms)
        if self.vc_relax:
            self.positions, self.cell = self.optimizer.step(
                self.positions, self.cell, self.energy, self.forces, self.deralat,
            )
            atoms.set_cell(self.cell.T)
        else:
            self.positions = self.optimizer.step(
                self.positions, self.energy, self.forces,
            )
        atoms.set_positions(self.positions.T)

    def _get_derivative_norm(self) -> float:
        force_norm = float(np.max(np.abs(self.forces)))
        if self.vc_relax:
            force_norm = max(force_norm, float(np.max(np.abs(self._get_lattice_derivative()))))
        return force_norm

    def optimize(self, trajectory_filename: str | None = None) -> Atoms:
        traj_file = None
        if trajectory_filename is not None:
            if os.path.exists(trajectory_filename):
                os.remove(trajectory_filename)
            traj_file = open(trajectory_filename, mode="w")

        try:
            for i in range(self.max_steps):
                if self._get_derivative_norm() <= self.force_tol:
                    break
                if self.vc_relax:
                    logging.info(
                        "Step %d: energy=%.6f max_force=%.6f lattice_deriv=%.6f",
                        i, self.energy, np.max(np.abs(self.forces)),
                        np.max(np.abs(self._get_lattice_derivative())),
                    )
                else:
                    logging.info(
                        "Step %d: energy=%.6f max_force=%.6f",
                        i, self.energy, np.max(np.abs(self.forces)),
                    )
                if traj_file is not None:
                    write(traj_file, self.atoms, parallel=False)
                    traj_file.flush()
                self.step(self.atoms)

            if traj_file is not None:
                write(traj_file, self.atoms, parallel=False)
        finally:
            if traj_file is not None:
                traj_file.close()

        return self.atoms
