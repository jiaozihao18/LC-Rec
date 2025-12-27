import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """
    改进的 Vector Quantizer，支持：
    1. EMA (Exponential Moving Average) 更新码本
    2. Codebook reset 机制（重置未使用的码本向量）
    3. 码本利用率统计
    """

    def __init__(self, n_e, e_dim,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.003, sk_iters=100,
                 ema_decay=0.99, epsilon=1e-5,
                 reset_threshold=1e-5, reset_interval=1000):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        
        # EMA 相关参数
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # Codebook reset 相关参数
        self.reset_threshold = reset_threshold
        self.reset_interval = reset_interval
        self.step_count = 0

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # EMA 统计量
        self.register_buffer('_ema_cluster_size', torch.zeros(n_e))
        self.register_buffer('_ema_w', torch.zeros(n_e, e_dim))
        
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def _reset_unused_codes(self, latent):
        """重置未使用的码本向量"""
        with torch.no_grad():
            # 计算每个码本向量的使用频率
            usage = self._ema_cluster_size / (self._ema_cluster_size.sum() + self.epsilon)
            
            # 找到未使用的码本向量（使用频率低于阈值）
            unused_mask = usage < self.reset_threshold
            num_unused = unused_mask.sum().item()
            
            if num_unused > 0:
                # 从未使用的向量中随机选择一些进行重置
                unused_indices = torch.where(unused_mask)[0]
                
                # 从当前 batch 的 latent 中随机采样来替换未使用的码本向量
                if len(latent) > 0:
                    # 随机选择一些 latent 向量
                    sample_indices = torch.randint(0, len(latent), (num_unused,), device=latent.device)
                    sample_vectors = latent[sample_indices]
                    
                    # 添加一些噪声以避免完全重复
                    noise = torch.randn_like(sample_vectors) * 0.01
                    self.embedding.weight.data[unused_indices] = (sample_vectors + noise).detach()
                    
                    # 重置这些向量的 EMA 统计
                    self._ema_cluster_size[unused_indices] = 0
                    self._ema_w[unused_indices] = 0

    def forward(self, x, use_sk=True, use_ema=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, self.embedding.weight.t())
        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # 计算损失
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # EMA 更新码本（仅在训练时）
        if self.training and use_ema:
            # 统计每个码本向量的使用次数
            encodings = F.one_hot(indices, self.n_e).float()  # [N, n_e]
            cluster_size = encodings.sum(0)  # [n_e]
            
            # 更新 EMA 统计
            self._ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            
            # 计算每个码本向量对应的 latent 向量的加权和
            n = latent.size(0)
            dw = torch.matmul(encodings.t(), latent)  # [n_e, e_dim]
            self._ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            
            # 更新码本向量（使用 EMA）
            # 计算归一化的 EMA 加权和
            normalized_ema_w = self._ema_w / (self._ema_cluster_size.unsqueeze(1) + self.epsilon)
            
            # 使用 EMA 更新 embedding
            # 标准做法：直接使用归一化的 EMA 值更新，但只更新被使用过的码本向量
            with torch.no_grad():
                # 只更新使用频率大于阈值的码本向量
                used_mask = self._ema_cluster_size > self.epsilon
                if used_mask.any():
                    # 对于使用过的码本向量，使用 EMA 更新
                    # 使用较小的更新率，保持平滑
                    update_rate = 1 - self.ema_decay
                    self.embedding.weight.data[used_mask] = (
                        self.embedding.weight.data[used_mask] * (1 - update_rate) +
                        normalized_ema_w[used_mask] * update_rate
                    )
            
            # 定期重置未使用的码本向量
            self.step_count += 1
            if self.step_count % self.reset_interval == 0:
                self._reset_unused_codes(latent)
        else:
            # 不使用 EMA 时，使用标准的梯度更新
            pass

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    def get_codebook_usage(self):
        """获取码本利用率统计"""
        with torch.no_grad():
            usage = self._ema_cluster_size / (self._ema_cluster_size.sum() + self.epsilon)
            used_codes = (usage > self.reset_threshold).sum().item()
            utilization = used_codes / self.n_e
            return {
                'utilization': utilization,
                'used_codes': used_codes,
                'total_codes': self.n_e,
                'usage_distribution': usage.cpu().numpy()
            }

