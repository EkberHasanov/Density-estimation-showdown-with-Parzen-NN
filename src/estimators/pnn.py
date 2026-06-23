"""Parzen Neural Network (PNN) -- the star method (§3.4).

An MLP ``f_θ: R^d -> R`` is trained by regression (MSE) to reproduce the
Parzen-window density estimate at a set of input locations (training samples +
uniform samples over the support). Two ingredients enforce probabilistic
validity (the Kolmogorov axioms):

  * a **non-negativity output head** (softplus / exp / square) so ``f_θ(x) ≥ 0``;
  * a **unit-integral soft constraint**: a penalty ``λ (∫ f_θ − 1)²`` with the
    integral estimated by numerical quadrature on a grid over the support.

After training, density evaluation is a single forward pass -- O(1) per query,
independent of the training-set size (the punchline of §4.3).

References:
  * E. Trentin & M. Gori, *Parzen Neural Networks*.
  * E. Trentin et al., *soft-constrained ANNs satisfying the Kolmogorov axioms
    of probability*.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .base import DensityEstimator
from .parzen import ParzenWindow
from ..datasets import Grid, make_grid_1d, make_grid_2d


def _support_box(X: np.ndarray, pad: float) -> np.ndarray:
    """Per-dimension [lo, hi] box covering ``X`` with a relative ``pad`` margin."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    return np.column_stack([lo - pad * span, hi + pad * span])  # (d, 2)


def _grid_for_box(box: np.ndarray, res: int) -> Grid:
    """Build a quadrature :class:`Grid` for a 1D or 2D support box."""
    d = box.shape[0]
    if d == 1:
        return make_grid_1d(box[0, 0], box[0, 1], res)
    if d == 2:
        return make_grid_2d(box[0, 0], box[0, 1], box[1, 0], box[1, 1], res)
    raise ValueError("PNN integral constraint supports only 1D/2D supports")


class _MLP(nn.Module):
    """Small feed-forward network with a non-negative output head."""

    def __init__(
        self, in_dim: int, hidden: list[int], activation: str, output_head: str
    ) -> None:
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.output_head = output_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        if self.output_head == "softplus":
            return nn.functional.softplus(raw)
        if self.output_head == "exp":
            return torch.exp(raw)
        if self.output_head == "square":
            return raw**2
        raise ValueError(f"unknown output_head {self.output_head!r}")

    def init_uniform_density(self, area: float) -> None:
        """Initialise the output head so the net starts as ~uniform density 1/area.

        This keeps the initial value of ``∫ f_θ`` close to 1, so the unit-integral
        penalty is a gentle nudge from the start rather than a term that dwarfs the
        MSE and collapses the network to a flat function.
        """
        final = self.net[-1]
        target = 1.0 / area  # desired (roughly constant) initial density value
        if self.output_head == "softplus":
            bias = math.log(math.expm1(target))  # softplus(bias) = target
        elif self.output_head == "exp":
            bias = math.log(target)
        else:  # square
            bias = math.sqrt(target)
        with torch.no_grad():
            final.weight.mul_(0.1)  # flatten the initial surface
            final.bias.fill_(bias)


class ParzenNeuralNetwork(DensityEstimator):
    """MLP trained to reproduce the Parzen estimate, with optional integral constraint.

    Parameters
    ----------
    hidden, activation, output_head :
        MLP architecture and the non-negativity head.
    epochs, lr, weight_decay, batch_size :
        Optimisation settings (Adam).
    n_grid_inputs :
        Number of extra uniform samples over the support added to the training
        inputs (in addition to the data points themselves).
    integral_penalty : float
        ``λ`` for the unit-integral soft constraint. ``0`` disables it (ablation).
    integral_grid_res :
        Per-dimension resolution of the quadrature grid for ``∫ f_θ``.
    parzen_bandwidth :
        Bandwidth rule/value passed to the internal :class:`ParzenWindow` target.
    support_pad : float
        Relative margin of the support box around the data.
    random_state : int | None
        Seed for torch init, input sampling and shuffling.
    """

    def __init__(
        self,
        hidden: list[int] | None = None,
        activation: str = "tanh",
        output_head: str = "softplus",
        epochs: int = 400,
        lr: float = 5e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        n_grid_inputs: int = 1000,
        integral_penalty: float = 1.0,
        integral_grid_res: int = 60,
        parzen_bandwidth: str | float = "silverman",
        support_pad: float = 0.15,
        random_state: int | None = None,
    ) -> None:
        self.hidden = hidden or [64, 64]
        self.activation = activation
        self.output_head = output_head
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_grid_inputs = n_grid_inputs
        self.integral_penalty = integral_penalty
        self.integral_grid_res = integral_grid_res
        self.parzen_bandwidth = parzen_bandwidth
        self.support_pad = support_pad
        self.random_state = random_state

        self.model_: _MLP | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.box_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []

    # -- fitting -------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "ParzenNeuralNetwork":
        X = self._as_2d(X)
        n, d = X.shape
        gen = torch.Generator().manual_seed(
            0 if self.random_state is None else self.random_state
        )
        rng = np.random.default_rng(self.random_state)

        # 1) Parzen target (the teacher).
        parzen = ParzenWindow(bandwidth=self.parzen_bandwidth).fit(X)

        # 2) Support box + extra uniform input locations covering it.
        self.box_ = _support_box(X, self.support_pad)
        if self.n_grid_inputs > 0:
            extra = rng.uniform(
                self.box_[:, 0], self.box_[:, 1], size=(self.n_grid_inputs, d)
            )
            inputs = np.vstack([X, extra])
        else:
            inputs = X
        targets = parzen.score_samples(inputs)

        # 3) Standardize inputs for stable optimisation.
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        inputs_std = (inputs - self.x_mean_) / self.x_std_

        # 4) Quadrature grid for the integral constraint (standardized coords).
        quad = _grid_for_box(self.box_, self.integral_grid_res)
        quad_std = (quad.points - self.x_mean_) / self.x_std_

        # 5) Tensors.
        Xin = torch.as_tensor(inputs_std, dtype=torch.float32)
        Yt = torch.as_tensor(targets, dtype=torch.float32)
        Qt = torch.as_tensor(quad_std, dtype=torch.float32)
        Qw = torch.as_tensor(quad.weights, dtype=torch.float32)  # trapezoidal weights
        # Scale of the regression target. Dividing the MSE by it makes the data
        # term ~O(1) regardless of the density magnitude (which scales as 1/area),
        # so the integral-penalty weight `integral_penalty` has a consistent
        # meaning across 1D and 2D problems.
        target_scale = float(np.mean(targets**2)) + 1e-12

        # 6) Model + optimiser.
        torch.manual_seed(0 if self.random_state is None else self.random_state)
        model = _MLP(d, self.hidden, self.activation, self.output_head)
        model.init_uniform_density(float(np.prod(self.box_[:, 1] - self.box_[:, 0])))
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        mse = nn.MSELoss()

        n_in = Xin.shape[0]
        model.train()
        self.history_ = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n_in, generator=gen)
            epoch_mse = 0.0
            for start in range(0, n_in, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                pred = model(Xin[idx])
                loss_mse = mse(pred, Yt[idx]) / target_scale
                loss = loss_mse
                if self.integral_penalty > 0:
                    integral = (model(Qt) * Qw).sum()
                    loss = loss + self.integral_penalty * (integral - 1.0) ** 2
                loss.backward()
                opt.step()
                epoch_mse += float(loss_mse) * idx.shape[0]
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                with torch.no_grad():
                    integral = float((model(Qt) * Qw).sum())
                self.history_.append(
                    {"epoch": epoch, "mse": epoch_mse / n_in, "integral": integral}
                )

        model.eval()
        self.model_ = model
        self._quad = quad
        self._quad_std = Qt
        self._quad_w = Qw
        return self

    # -- scoring -------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("estimator must be fit before scoring")
        X = self._as_2d(X)
        Xs = (X - self.x_mean_) / self.x_std_
        with torch.no_grad():
            p = self.model_(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
        return p.astype(float)

    def integral_value(self) -> float:
        """Numerically integrate the learned density over the support box."""
        if self.model_ is None:
            raise RuntimeError("estimator must be fit before integrating")
        with torch.no_grad():
            return float((self.model_(self._quad_std) * self._quad_w).sum())
