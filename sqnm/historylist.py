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

import numpy as np


class HistoryList:
    def __init__(self, ndim: int, nhist_max: int):
        self.ndim = ndim
        self.nhist_max = nhist_max
        self.hist = np.zeros((ndim, nhist_max))
        self.diff = np.zeros((ndim, nhist_max))
        self.normalized_diff = np.zeros((ndim, nhist_max))
        self.count = 0
        self._first_full = True

    def add(self, vector: np.ndarray) -> int:
        if self.count < self.nhist_max:
            self.hist[:, self.count] = vector
            self.count += 1
            if self.count > 1:
                idx = self.count - 2
                self.diff[:, idx] = self.hist[:, idx + 1] - self.hist[:, idx]
                self.normalized_diff[:, idx] = self.diff[:, idx] / np.linalg.norm(self.diff[:, idx])
            return self.count - 1

        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = vector

        if self._first_full:
            self._first_full = False
        else:
            self.diff[:, :-1] = self.diff[:, 1:]
            self.normalized_diff[:, :-1] = self.normalized_diff[:, 1:]

        self.diff[:, -1] = self.hist[:, -1] - self.hist[:, -2]
        self.normalized_diff[:, -1] = self.diff[:, -1] / np.linalg.norm(self.diff[:, -1])
        return self.nhist_max
