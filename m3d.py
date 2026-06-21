import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Fixed Kernels (kept for backward compatibility and non-learnable mode)
# ==============================================================================

class RBF(nn.Module):
    """Fixed multi-bandwidth RBF kernel (original implementation)."""

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]
        ).sum(dim=0)


class PoliKernel(nn.Module):
    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree

    def forward(self, X):
        K = (torch.matmul(X, X.t()) + self.constant_term) ** self.degree
        return K


class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        K = torch.matmul(X, X.t())
        return K


class LaplaceKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...] * (self.gammas)[:, None, None]
        ).sum(dim=0)


# ==============================================================================
# Learnable Multi-Kernel (Adaptive)
# ==============================================================================

class LearnableRBF(nn.Module):
    """RBF kernel with learnable bandwidth multipliers and mixture weights.

    Each kernel k has:
      - A learnable log-bandwidth-multiplier controlling its sensitivity scale
      - A learnable mixture logit controlling its contribution weight

    The effective kernel is a weighted sum over the multi-kernel ensemble:
        K(x, y) = Σ_k w_k · exp(-||x - y||² / (σ²_base · m_k))

    where w_k = softmax(logits)_k and m_k = exp(log_multiplier_k).

    When learnable=False, this reduces to the original fixed RBF kernel.
    """

    def __init__(self, n_kernels=5, init_mul_factor=2.0, bandwidth=None, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.n_kernels = n_kernels

        # Initialize bandwidth multipliers as a geometric progression
        # centred at 1.0: [0.25, 0.5, 1.0, 2.0, 4.0] for n_kernels=5, factor=2.0
        init_log_mult = torch.log(
            init_mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        )

        if learnable:
            # Log-space parameterization ensures positivity after exp()
            self.log_bandwidth_multipliers = nn.Parameter(init_log_mult.clone())
            # Kernel mixture logits (un-normalized weights)
            self.kernel_logits = nn.Parameter(torch.zeros(n_kernels))
            # Store initial log-multipliers for mild deviation regularisation
            self.register_buffer('init_log_mult', init_log_mult.clone())
        else:
            self.register_buffer('log_bandwidth_multipliers', init_log_mult)
            self.register_buffer('kernel_logits', torch.zeros(n_kernels))

        self.bandwidth = bandwidth

    def forward(self, X):
        """Compute the multi-kernel Gram matrix.

        Args:
            X: (N, D) feature matrix.

        Returns:
            K_weighted: (N, N) weighted-sum kernel matrix.
        """
        L2_distances = torch.cdist(X, X) ** 2  # (N, N)

        # Base bandwidth: mean pairwise distance (median heuristic)
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            base_bw = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        else:
            base_bw = self.bandwidth

        # Softmax-normalised kernel mixture weights
        kernel_weights = F.softmax(self.kernel_logits, dim=0)  # (K,)

        # Bandwidth multipliers (guaranteed positive via exp)
        multipliers = torch.exp(self.log_bandwidth_multipliers)  # (K,)

        # Multi-kernel tensor: (K, N, N)
        K = torch.exp(
            -L2_distances[None, ...]
            / (base_bw * multipliers.clamp(min=1e-6))[:, None, None]
        )

        # Weighted sum over kernels: (N, N)
        K_weighted = (kernel_weights[:, None, None] * K).sum(dim=0)

        return K_weighted


class LearnableLaplace(nn.Module):
    """Laplace kernel with learnable gamma parameters and mixture weights."""

    def __init__(self, n_kernels=3, init_gammas=None, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.n_kernels = n_kernels

        if init_gammas is None:
            init_gammas = torch.tensor([0.1, 1.0, 5.0])
        init_log_gammas = torch.log(init_gammas.float())

        if learnable:
            self.log_gammas = nn.Parameter(init_log_gammas)
            self.kernel_logits = nn.Parameter(torch.zeros(n_kernels))
            self.register_buffer('init_log_gammas', init_log_gammas.clone())
        else:
            self.register_buffer('log_gammas', init_log_gammas)
            self.register_buffer('kernel_logits', torch.zeros(n_kernels))

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2  # (N, N)

        kernel_weights = F.softmax(self.kernel_logits, dim=0)  # (K,)
        gammas = torch.exp(self.log_gammas)  # (K,)

        K = torch.exp(-L2_distances[None, ...] * gammas.clamp(min=1e-6)[:, None, None])
        K_weighted = (kernel_weights[:, None, None] * K).sum(dim=0)

        return K_weighted


# ==============================================================================
# MMD Loss (Adaptive Multi-Kernel)
# ==============================================================================

class M3DLoss(nn.Module):
    """Multi-Kernel Maximum Mean Discrepancy (M3D) loss.

    Supports both fixed (original) and learnable multi-kernel modes.

    Learnable mode (learnable=True):
        - Bandwidth multipliers / gamma values are nn.Parameters updated via gradient descent.
        - Kernel mixture weights are learned via softmax over logits.
        - Entropy regularisation encourages using the full kernel ensemble rather than
          collapsing to a single kernel.
        - Mild deviation regularisation keeps bandwidths near their initialisation.

    Fixed mode (learnable=False):
        - Reproduces the original M3D behaviour exactly.
    """

    def __init__(self, kernel_type='gaussian', n_kernels=5, learnable=True,
                 ent_weight=0.01, bw_reg_weight=0.001, bandwidth=None):
        """
        Args:
            kernel_type: 'gaussian', 'laplace', 'linear', or 'polynomial'.
            n_kernels: Number of kernels in the ensemble (for gaussian / laplace).
            learnable: Whether kernel parameters are learnable.
            ent_weight: Weight of the entropy regularisation term.
            bw_reg_weight: Weight of the bandwidth deviation regularisation.
            bandwidth: Fixed base bandwidth (None = auto-compute via median heuristic).
        """
        super().__init__()
        self.learnable = learnable
        self.ent_weight = ent_weight
        self.bw_reg_weight = bw_reg_weight
        self.kernel_type = kernel_type

        if kernel_type == 'gaussian':
            self.kernel = LearnableRBF(
                n_kernels=n_kernels,
                bandwidth=bandwidth,
                learnable=learnable,
            )
        elif kernel_type == 'laplace':
            self.kernel = LearnableLaplace(
                n_kernels=n_kernels,
                learnable=learnable,
            )
        elif kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'polynomial':
            self.kernel = PoliKernel()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def forward(self, X, Y):
        """Compute the MMD² between two feature sets.

        Args:
            X: (N, D) feature matrix (e.g., real features).
            Y: (M, D) feature matrix (e.g., synthetic features).

        Returns:
            total_loss: scalar MMD² loss (with regularisation if learnable).
        """
        # Combined Gram matrix over [X; Y]
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]

        # Biased MMD² estimator:  E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
        XX = K[:X_size, :X_size].mean()       # within X
        XY = K[:X_size, X_size:].mean()       # cross
        YY = K[X_size:, X_size:].mean()       # within Y

        mmd_loss = XX - 2 * XY + YY
        total_loss = mmd_loss

        if self.learnable and hasattr(self.kernel, 'kernel_logits'):
            # --- Entropy regularisation ---
            # Maximise entropy of kernel weights → encourages using all kernels
            # L_ent = -H(w) = Σ w_k log w_k   (minimised → high entropy)
            kernel_weights = F.softmax(self.kernel.kernel_logits, dim=0)
            entropy = -(kernel_weights * torch.log(kernel_weights + 1e-8)).sum()
            total_loss = total_loss - self.ent_weight * entropy

            # --- Bandwidth deviation regularisation ---
            # Mild penalty for drifting too far from initial bandwidths
            if hasattr(self.kernel, 'init_log_mult'):
                bw_dev = torch.norm(
                    self.kernel.log_bandwidth_multipliers - self.kernel.init_log_mult
                ) ** 2
                total_loss = total_loss + self.bw_reg_weight * bw_dev
            elif hasattr(self.kernel, 'init_log_gammas'):
                bw_dev = torch.norm(
                    self.kernel.log_gammas - self.kernel.init_log_gammas
                ) ** 2
                total_loss = total_loss + self.bw_reg_weight * bw_dev

        return total_loss

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_kernel_weights(self):
        """Return the current kernel mixture weights (softmax over logits)."""
        if hasattr(self.kernel, 'kernel_logits'):
            return F.softmax(self.kernel.kernel_logits, dim=0).detach().cpu()
        return None

    def get_bandwidth_multipliers(self):
        """Return the current bandwidth multipliers (exp of log-params)."""
        if hasattr(self.kernel, 'log_bandwidth_multipliers'):
            return torch.exp(self.kernel.log_bandwidth_multipliers).detach().cpu()
        return None

    def get_gammas(self):
        """Return the current Laplace gammas (exp of log-params)."""
        if hasattr(self.kernel, 'log_gammas'):
            return torch.exp(self.kernel.log_gammas).detach().cpu()
        return None

    def kernel_parameters(self):
        """Yield all learnable kernel parameters (for the optimizer)."""
        for p in self.kernel.parameters():
            yield p
