import torch
import torch.nn as nn
from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """
    改进的 Residual Vector Quantizer，使用改进的 VectorQuantizer
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100,
                 ema_decay=0.99, epsilon=1e-5,
                 reset_threshold=1e-5, reset_interval=1000):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_e, e_dim,
                beta=self.beta,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                ema_decay=ema_decay,
                epsilon=epsilon,
                reset_threshold=reset_threshold,
                reset_interval=reset_interval
            )
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True, use_ema=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk, use_ema=use_ema)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    def get_codebook_usage(self):
        """获取所有量化器的码本利用率"""
        usage_stats = []
        for i, quantizer in enumerate(self.vq_layers):
            stats = quantizer.get_codebook_usage()
            stats['quantizer_id'] = i
            usage_stats.append(stats)
        return usage_stats

