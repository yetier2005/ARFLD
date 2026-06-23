"""
Characteristic Function (CF) Loss for Federated Dataset Distillation
=====================================================================

Implements the Neural Characteristic Function Matching (NCFM) loss from:
  "Dataset Distillation with Neural Characteristic Function: A Minmax Perspective"
  (CVPR 2025, Wang et al.)

Adapted for the ARFLD federated learning framework.
Replaces M3D + CORAL with CF-based distribution matching.

Core components:
  - CFLossFunc:   Amplitude + phase distance in the characteristic function domain
  - SampleNet:    Adversarial frequency generator (optional, for minmax training)
  - cf_match_loss: High-level wrapper adapted for ARFLD's model interface
  - cf_calib_loss: Cross-entropy calibration using a trained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper functions
# ==============================================================================

def calculate_norm(x_r, x_i):
    """Compute magnitude of complex numbers represented as (real, imag) pairs."""
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_real(x):
    """Empirical characteristic function: real part = E[cos(t·x)]."""
    return torch.mean(torch.cos(x), dim=1)


def calculate_imag(x):
    """Empirical characteristic function: imaginary part = E[sin(t·x)]."""
    return torch.mean(torch.sin(x), dim=1)


# ==============================================================================
# CFLossFunc — Core Characteristic Function loss
# ==============================================================================

class CFLossFunc(nn.Module):
    """
    Characteristic Function loss combining amplitude and phase differences.

    The empirical characteristic function of a distribution at frequency t is:
        φ(t) = E[exp(i · tᵀx)] ≈ mean(cos(tᵀx)) + i · mean(sin(tᵀx))

    This loss computes the squared distance between two distributions' CFs
    at random projection frequencies, decomposed into amplitude and phase terms.

    Args:
        alpha_for_loss: weight for amplitude term (0-1)
        beta_for_loss:  weight for phase term (0-1)
    """

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss

    def forward(self, feat_tg, feat, t=None, args=None):
        """
        Compute CF distance between target (real) and synthetic features.

        Args:
            feat_tg: target features from real data  [B1 x D]
            feat:    synthetic features              [B2 x D]
            t:       pre-generated frequency matrix  [num_freqs x D]
                     If None, random Gaussian frequencies are sampled.
            args:    must contain `num_freqs` and `device`

        Returns:
            scalar loss
        """
        # Generate random projection frequencies
        if t is None:
            t = torch.randn((args.num_freqs, feat.size(1)), device=feat.device)

        # Project features onto random frequencies: t @ fᵀ → [num_freqs, batch]
        # Real part:  E[cos(t·x)]
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Amplitude difference: (|φ_target| - |φ_x|)²
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        # Phase difference: 2|φ_target||φ_x|(1 - cos(Δθ))
        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )
        loss_pha = loss_pha.clamp(min=1e-12)  # numerical stability

        # Combined loss: sqrt(α·amp + β·phase), averaged over frequencies
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


# ==============================================================================
# SampleNet — Adversarial frequency generator
# ==============================================================================

class SampleNet(nn.Module):
    """
    Adversarial network that learns projection frequencies for CF loss.

    In the minmax formulation:
      - Synthetic images MINIMIZE the CF loss
      - SampleNet MAXIMIZES the CF loss

    SampleNet learns to generate frequencies that best discriminate between
    real and synthetic distributions, forcing the synthetic data to match
    across all discriminative frequencies.

    Architecture: 3-layer MLP with LeakyReLU → Tanh output.

    Args:
        feature_dim:  dimensionality of feature vectors (e.g., 2048 for ConvNet)
        t_batchsize:  number of frequency vectors to generate (= num_freqs)
        t_var:        variance of input noise
    """

    def __init__(self, feature_dim=2048, t_batchsize=4096, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim
        self.t_sigma_num = max(t_batchsize // 16, 1)
        self._input_adv_t_net_dim = feature_dim
        self._input_t_dim = feature_dim
        self._input_t_batchsize = t_batchsize
        self._input_t_var = t_var

        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # 3-layer fully-connected network
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        """Generate frequency vectors.

        Args:
            device: torch device to place output on

        Returns:
            t: frequency matrix [t_batchsize, feature_dim]
        """
        if self.t_sigma_num > 0:
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var ** 0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)
            self._t = a
        else:
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t


# ==============================================================================
# High-level match functions (adapted for ARFLD model interface)
# ==============================================================================

def cf_match_loss(img_real, img_syn, model, cf_loss_func, sampling_net=None, args=None):
    """
    Compute CF matching loss between real and synthetic images.

    Adapted for ARFLD's model interface where model(img, train=True) returns
    (features, logits) instead of NCFM's model(img, return_features=True).

    Args:
        img_real:      real images       [B1, C, H, W]
        img_syn:       synthetic images  [B2, C, H, W]
        model:         ARFLD model (net or new_net), frozen feature extractor
        cf_loss_func:  CFLossFunc instance
        sampling_net:  SampleNet instance or None (None = random frequencies)
        args:          global args (needs num_freqs, cf_loss_scale, device)

    Returns:
        scalar CF loss
    """
    # Extract features — real features without grad, synthetic with grad
    with torch.no_grad():
        ft_real, _ = model(img_real, train=True)
    ft_syn, _ = model(img_syn, train=True)

    # L2 normalization (critical for CF loss — removes magnitude confounds)
    ft_real = F.normalize(ft_real, dim=1)
    ft_syn = F.normalize(ft_syn, dim=1)

    # Generate frequencies: adversarial (SampleNet) or random Gaussian
    if sampling_net is not None:
        t = sampling_net(args.device)
    else:
        t = None

    # CF loss with scaling (matching NCFM's default scale of 300)
    loss = args.cf_loss_scale * cf_loss_func(ft_real, ft_syn, t, args)
    return loss


def cf_calib_loss(img_syn, label_syn, trained_model):
    """
    Calibration loss: cross-entropy on a fully trained model.

    Ensures synthetic images remain classifiable by a trained classifier,
    preventing distribution collapse where features match but semantic content is lost.

    Args:
        img_syn:        synthetic images  [B, C, H, W]
        label_syn:      class labels       [B]
        trained_model:  trained classifier (global model in FL setting)

    Returns:
        scalar CE loss
    """
    logits = trained_model(img_syn, train=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss


# ==============================================================================
# Feature dimension helper
# ==============================================================================

def compute_feature_dim(model, channel, im_size, device='cuda'):
    """
    Auto-detect the feature dimension of a model by doing a dry-run forward pass.

    Args:
        model:   model instance (will be moved to device)
        channel: number of input channels
        im_size: (H, W) tuple
        device:  torch device

    Returns:
        feature_dim: int, dimensionality of the embedding (pre-classifier) layer
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, channel, im_size[0], im_size[1], device=device)
        ft, _ = model(dummy, train=True)
        return ft.shape[1]
