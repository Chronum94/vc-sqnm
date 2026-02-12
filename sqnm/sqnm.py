from __future__ import annotations

import sys

import numpy as np

from sqnm.historylist import HistoryList


class SQNM:
    def __init__(
        self,
        ndim: int,
        nhist_max: int,
        alpha: float,
        eps_subsp: float,
        alpha_min: float,
    ):
        self.ndim = ndim
        self.nhist_max = nhist_max
        if ndim < nhist_max:
            print("ndim < nhist_max. Setting nhist_max to ndim", file=sys.stderr)
            self.nhist_max = ndim

        self.eps_subsp = eps_subsp
        self.alpha_min = alpha_min
        self.x_list = HistoryList(ndim, self.nhist_max)
        self.f_list = HistoryList(ndim, self.nhist_max)
        self.dim_subspace = 0

        self.estimate_step_size = alpha <= 0
        self.alpha = -alpha if alpha <= 0 else alpha

        n = self.nhist_max
        self.dir_of_descent = np.zeros(ndim)
        self.expected_positions = np.zeros(ndim)
        self.prev_f = 0.0
        self.prev_df = np.zeros(ndim)
        self.s_evec = np.zeros((n, n))
        self.s_eval = np.zeros(n)
        self.dr_subsp = np.zeros((ndim, n))
        self.df_subsp = np.zeros((ndim, n))
        self.h_evec_subsp = np.zeros((n, n))
        self.h_evec = np.zeros((ndim, n))
        self.h_eval = np.zeros(n)
        self.res = np.zeros(n)
        self.gainratio = 0.0
        self.nhist = 0

    def step(self, x: np.ndarray, f: float, df: np.ndarray) -> np.ndarray:
        if np.linalg.norm(df) < 1e-12:
            self.dir_of_descent[:] = 0.0
            return self.dir_of_descent

        self.nhist = self.x_list.add(x)
        self.f_list.add(df)

        if self.nhist == 0:
            self.dir_of_descent = -self.alpha * df
        else:
            if np.max(np.abs(x - self.expected_positions)) > 1e-8:
                print(
                    "SQNM was not called with expected positions. "
                    "Were atoms put back into the cell? This is not allowed.",
                    file=sys.stderr,
                )

            if self.estimate_step_size:
                prev_norm_sq = np.linalg.norm(self.prev_df) ** 2
                l1 = (f - self.prev_f + self.alpha * prev_norm_sq) / (
                    0.5 * self.alpha**2 * prev_norm_sq
                )
                l2 = np.linalg.norm(df - self.prev_df) / (
                    self.alpha * np.sqrt(prev_norm_sq)
                )
                self.alpha = 1.0 / max(l1, l2)
                self.estimate_step_size = False
            else:
                self.gainratio = (f - self.prev_f) / (
                    0.5 * np.dot(self.dir_of_descent, self.prev_df)
                )
                if self.gainratio < 0.5:
                    self.alpha = max(self.alpha_min, self.alpha * 0.65)
                if self.gainratio > 1.05:
                    self.alpha *= 1.05

            nh = self.nhist
            nd = self.x_list.normalized_diff

            self.s_evec[:nh, :nh] = nd[:, :nh].T @ nd[:, :nh]
            self.s_eval[:nh], self.s_evec[:nh, :nh] = np.linalg.eigh(
                self.s_evec[:nh, :nh]
            )

            dim = int(np.sum(self.s_eval[:nh] / self.s_eval[nh - 1] > self.eps_subsp))
            self.dim_subspace = dim
            self.s_eval[:dim] = self.s_eval[nh - dim : nh]
            self.s_evec[:, :dim] = self.s_evec[:, nh - dim : nh]

            inv_sqrt_s = 1.0 / np.sqrt(self.s_eval[:dim])
            diff_inv_norms = 1.0 / np.linalg.norm(self.x_list.diff[:, :nh], axis=0)

            self.dr_subsp[:, :dim] = (
                np.einsum("hi,kh->ki", self.s_evec[:nh, :dim], nd[:, :nh])
                * inv_sqrt_s
            )
            self.df_subsp[:, :dim] = (
                np.einsum(
                    "hi,kh,h->ki",
                    self.s_evec[:nh, :dim],
                    self.f_list.diff[:, :nh],
                    diff_inv_norms,
                )
                * inv_sqrt_s
            )

            self.h_evec_subsp[:dim, :dim] = 0.5 * (
                self.df_subsp[:, :dim].T @ self.dr_subsp[:, :dim]
                + self.dr_subsp[:, :dim].T @ self.df_subsp[:, :dim]
            )
            self.h_eval[:dim], self.h_evec_subsp[:dim, :dim] = np.linalg.eigh(
                self.h_evec_subsp[:dim, :dim]
            )

            self.h_evec[:, :dim] = np.einsum(
                "ki,hk->hi", self.h_evec_subsp[:dim, :dim], self.dr_subsp[:, :dim]
            )

            self.res[:dim] = np.linalg.norm(
                -self.h_eval[:dim] * self.h_evec[:, :dim]
                + self.df_subsp[:, :dim] @ self.h_evec_subsp[:dim, :dim],
                axis=0,
            )

            self.h_eval[:dim] = np.sqrt(self.h_eval[:dim] ** 2 + self.res[:dim] ** 2)

            projections = self.h_evec[:, :dim].T @ df
            self.dir_of_descent = self.alpha * (
                df - np.einsum("i,ki->k", projections, self.h_evec[:, :dim])
            )
            self.dir_of_descent += np.einsum(
                "i,ki,i->k",
                projections,
                self.h_evec[:, :dim],
                1.0 / self.h_eval[:dim],
            )
            self.dir_of_descent = -self.dir_of_descent

        self.expected_positions = x + self.dir_of_descent
        self.prev_f = f
        self.prev_df = df
        return self.dir_of_descent

    def lower_bound(self) -> float:
        if self.nhist < 1:
            print(
                "At least one step needed before lower_bound can be called.",
                file=sys.stderr,
            )
            return 0.0
        return self.prev_f - 0.5 * np.dot(self.prev_df, self.prev_df) / self.h_eval[0]
