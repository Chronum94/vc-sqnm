#!/usr/bin/env python3

# Copyright (C) 2022 Moritz Gubler
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import sys

import numpy as np

from sqnm.sqnm import SQNM


class PeriodicSQNM:
    def __init__(
        self,
        nat: int,
        init_lat: np.ndarray,
        initial_step_size: float,
        nhist_max: int,
        lattice_weight: float,
        alpha_min: float,
        eps_subsp: float,
    ):
        self.nat = nat
        self.ndim = 3 * nat + 9
        self.lattice_weight = lattice_weight
        self.initial_lat = np.array(init_lat)
        self.initial_lat_inv = np.linalg.inv(self.initial_lat)
        self.lat_transformer = (
            np.diag(1.0 / np.linalg.norm(self.initial_lat, axis=0))
            * self.lattice_weight
            * np.sqrt(nat)
        )
        self.lat_transformer_inv = np.linalg.inv(self.lat_transformer)
        self.optimizer = SQNM(
            self.ndim, nhist_max, initial_step_size, eps_subsp, alpha_min
        )
        self.fluct = 0.0

    def step(
        self,
        pos: np.ndarray,
        alat: np.ndarray,
        epot: float,
        forces: np.ndarray,
        deralat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fnoise = np.linalg.norm(np.sum(forces, axis=1)) / np.sqrt(3 * self.nat)
        self.fluct = fnoise if self.fluct == 0.0 else 0.8 * self.fluct + 0.2 * fnoise
        if self.fluct > 0.2 * np.max(np.abs(forces)):
            print(
                "Warning: noise in forces is larger than 0.2 times the largest "
                "force component. Convergence is not guaranteed.",
                file=sys.stderr,
            )

        a_inv = np.linalg.inv(alat)
        q = ((self.initial_lat @ a_inv) @ pos).ravel()
        df_dq = (-(alat @ self.initial_lat_inv) @ forces).ravel()
        a_tilde = (alat @ self.lat_transformer).ravel()
        df_da_tilde = (-deralat @ self.lat_transformer_inv).ravel()

        q_and_lat = np.concatenate((q, a_tilde))
        dq_and_dlat = np.concatenate((df_dq, df_da_tilde))

        q_and_lat = q_and_lat + self.optimizer.step(q_and_lat, epot, dq_and_dlat)

        n3 = 3 * self.nat
        alat = q_and_lat[n3:].reshape(3, 3) @ self.lat_transformer_inv
        pos = (alat @ self.initial_lat_inv) @ q_and_lat[:n3].reshape(3, self.nat)
        return pos, alat

    def lower_bound(self) -> float:
        return self.optimizer.lower_bound()
