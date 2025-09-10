"""
models.py - Neural Network Models for Digital Twin Components

This module contains the Transformer-based dynamics model, treatment outcome model,
and Q-network for the RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Works for both (T, B, C) and (B, T, C).
        self.pe shape is (max_len, 1, C) currently.
        """
        if x.dim() == 3 and x.shape[1] >= 1:  # (B, T, C)
            T = x.size(1)
            # pe[:T] -> (T, 1, C)  -> transpose(0,1) -> (1, T, C), broadcast on batch
            return x + self.pe[:T].transpose(0, 1)
        # fallback (T, B, C)
        T = x.size(0)
        return x + self.pe[:T, :]


class TransformerDynamicsModel(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super(TransformerDynamicsModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 改进1: 使用更好的嵌入层初始化
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        
        # 初始化嵌入层
        nn.init.xavier_uniform_(self.state_embedding.weight)
        nn.init.xavier_uniform_(self.action_embedding.weight)
        
        # 改进2: 添加LayerNorm在输入后
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Combine state and action
        self.input_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.xavier_uniform_(self.input_projection.weight)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # 改进3: 减少层数，使用更小的模型
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,  # 减小前馈网络
            dropout=dropout,
            activation='gelu',  # 使用GELU而不是ReLU
            batch_first=True  # 添加batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 减少到2层
        
        # 改进4: 更简单的输出头，添加残差连接
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 改进5: 初始化输出层为小值
        nn.init.xavier_uniform_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        
    def forward(self, states, actions, mask=None):
        """
        Args:
            states:  (B, L, state_dim)
            actions: (B, L)  —— 离散动作index序列
            mask:    (B, L) 的bool张量，True表示有效位（可选）
        Returns:
            next_state_pred: (B, L, state_dim) 逐步下一状态预测
        """
        B, L, _ = states.shape
        # 嵌入
        state_emb  = self.state_embedding(states)           # (B,L,H)
        action_emb = self.action_embedding(actions)         # (B,L,H)
        combined   = torch.cat([state_emb, action_emb], -1) # (B,L,2H)
        embedded   = self.input_projection(combined)        # (B,L,H)
        embedded   = self.input_norm(embedded)
        embedded   = self.pos_encoding(embedded)

        # 因果mask + 可选padding mask
        causal_mask = self._generate_causal_mask(L).to(states.device)  # (L,L), True=block
        src_key_padding_mask = None
        if isinstance(mask, torch.Tensor):
            # 传入的是“有效位=True”，Transformer需要“padding=True”
            src_key_padding_mask = ~mask  # (B,L)

        encoded = self.transformer(
            embedded,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # 关键修复：预测Δ并限制单步幅度，防止多步爆炸/上飘
        # tanh把幅度压到(-1,1)，乘0.05设定“最大单步改变≈±0.05”
        state_delta = torch.tanh(self.output_projection(encoded)) * 0.05

        # 残差 + 生理范围夹紧（所有状态已归一化到[0,1]）
        next_state_pred = torch.clamp(states + state_delta, 0.0, 1.0)
        return next_state_pred

    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        # 布尔上三角（True 表示“不可见”）
        return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    
    def predict_next_state(self, 
                        state_history: torch.Tensor,
                        action_history: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given history
        
        Args:
            state_history: (batch_size, history_len, state_dim)
            action_history: (batch_size, history_len)
            
        Returns:
            next_state: (batch_size, state_dim)
        """
        with torch.no_grad():
            # 确保输入维度正确
            if state_history.dim() == 2:
                state_history = state_history.unsqueeze(0)
            if action_history.dim() == 1:
                action_history = action_history.unsqueeze(0)
            
            # 确保action_history是正确的维度
            if action_history.dim() == 2 and state_history.dim() == 3:
                # action_history应该是(batch_size, history_len)
                pass
            elif action_history.dim() == 1:
                action_history = action_history.unsqueeze(0)
                
            B, T = state_history.size(0), state_history.size(1)
            
            # 创建有效掩码 - 可能这里是None导致的问题
            # valid = torch.ones(B, T, dtype=torch.bool, device=state_history.device)
            
            # 前向传播 - 不传mask试试
            predictions = self.forward(state_history, action_history, mask=None)
            
            # 返回最后一个预测
            return predictions[:, -1, :]

class TreatmentOutcomeModel(nn.Module):
    """
    Treatment outcome prediction model with deconfounding
    
    Implements r_φ(s, a) with representation learning to handle confounders
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 n_hidden_layers: int = 3,
                 dropout: float = 0.1):
        super(TreatmentOutcomeModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Disentangled representation learning
        # Separate encoders for different factors
        self.health_encoder = self._build_encoder(state_dim, hidden_dim // 2, n_hidden_layers)
        self.treatment_encoder = self._build_encoder(state_dim, hidden_dim // 2, n_hidden_layers)
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, hidden_dim // 2)
        
        # Outcome prediction head
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Regularization components
        self.treatment_discriminator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
    def _build_encoder(self, input_dim: int, output_dim: int, n_layers: int) -> nn.Module:
        """Build a multi-layer encoder"""
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            next_dim = output_dim if i == n_layers - 1 else current_dim * 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.LayerNorm(next_dim),  # 改用 LayerNorm 替代 BatchNorm1d
                nn.Dropout(0.1)
            ])
            current_dim = next_dim
            
        return nn.Sequential(*layers)
    
    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor,
                return_representations: bool = False) -> torch.Tensor:
        """
        Forward pass through outcome model
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size,) - action indices
            return_representations: Whether to return learned representations
            
        Returns:
            outcome: (batch_size, 1) - predicted immediate reward
        """
        # Encode state into disentangled representations
        health_repr = self.health_encoder(state)
        treatment_repr = self.treatment_encoder(state)
        
        # Embed action
        action_emb = self.action_embedding(action)
        
        # Concatenate representations
        combined_repr = torch.cat([health_repr, treatment_repr, action_emb], dim=-1)
        
        # Predict outcome
        outcome = self.outcome_head(combined_repr)
        
        if return_representations:
            return outcome, health_repr, treatment_repr
        return outcome
    
    def compute_regularization_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Encourage health representation to be independent of treatment:
        minimize KL( p(a|h) || Uniform )  <=>  maximize entropy of p(a|h)
        """
        with torch.no_grad():
            pass  # 占位以强调不要这里面停梯度

        _, health_repr, _ = self.forward(state, action, return_representations=True)
        logits = self.treatment_discriminator(health_repr)          # (B, A)
        p = F.softmax(logits, dim=-1)
        # 熵 H(p) = - sum p log p
        entropy = -(p * (p + 1e-8).log()).sum(dim=1).mean()
        target_entropy = math.log(self.action_dim + 1e-8)

        # 正则项 = (最大熵 - 当前熵)，始终 >= 0，且在低熵时有梯度
        reg_loss = (target_entropy - entropy)
        return reg_loss


class ConservativeQNetwork(nn.Module):
    """
    Conservative Q-Network for safe offline RL
    
    Implements Q_ψ(s, a) with conservative regularization
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 n_hidden_layers: int = 3,
                 dropout: float = 0.1):
        super(ConservativeQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build Q-network architecture
        layers = []
        current_dim = state_dim
        
        for i in range(n_hidden_layers):
            next_dim = hidden_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.LayerNorm(next_dim),  # 改用 LayerNorm 替代 BatchNorm1d
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for each action (dueling architecture)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(action_dim)
        ])
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            q_values: (batch_size, action_dim) - Q-values for all actions
        """
        # Shared representation
        features = self.shared_layers(state)
        
        # Compute state value
        value = self.value_head(features)
        
        # Compute advantages for each action
        advantages = torch.stack([
            head(features) for head in self.advantage_heads
        ], dim=1).squeeze(-1)
        
        # Dueling Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for specific state-action pairs"""
        q_values = self.forward(state)
        q_value = q_values.gather(1, action.unsqueeze(1))
        return q_value


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for uncertainty estimation
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_ensemble: int = 5,
                 hidden_dim: int = 256):
        super(EnsembleQNetwork, self).__init__()
        
        self.n_ensemble = n_ensemble
        self.q_networks = nn.ModuleList([
            ConservativeQNetwork(state_dim, action_dim, hidden_dim)
            for _ in range(n_ensemble)
        ])
        
    def forward(self, state: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            state: (batch_size, state_dim)
            return_all: Whether to return all Q-values or just mean
            
        Returns:
            q_values: (batch_size, action_dim) or (n_ensemble, batch_size, action_dim)
        """
        q_values_list = [q_net(state) for q_net in self.q_networks]
        q_values_stack = torch.stack(q_values_list, dim=0)
        
        if return_all:
            return q_values_stack
        else:
            # Return mean Q-values
            return q_values_stack.mean(dim=0)
    
    def get_q_value_with_uncertainty(self, 
                                    state: torch.Tensor, 
                                    action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-value with uncertainty estimate"""
        q_values_all = []
        
        for q_net in self.q_networks:
            q_val = q_net.get_q_value(state, action)
            q_values_all.append(q_val)
            
        q_values_tensor = torch.stack(q_values_all, dim=0)
        mean_q = q_values_tensor.mean(dim=0)
        std_q = q_values_tensor.std(dim=0)
        
        return mean_q, std_q


if __name__ == "__main__":
    # Test the models
    batch_size = 32
    seq_len = 10
    state_dim = 10
    action_dim = 5
    
    # Test Transformer Dynamics Model
    print("Testing Transformer Dynamics Model...")
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    
    next_states = dynamics_model(states, actions)
    print(f"Input shape: {states.shape}")
    print(f"Output shape: {next_states.shape}")
    
    # Test Treatment Outcome Model
    print("\nTesting Treatment Outcome Model...")
    outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
    
    single_state = torch.randn(batch_size, state_dim)
    single_action = torch.randint(0, action_dim, (batch_size,))
    
    outcome = outcome_model(single_state, single_action)
    print(f"Outcome shape: {outcome.shape}")
    
    # Test Q-Network
    print("\nTesting Conservative Q-Network...")
    q_network = ConservativeQNetwork(state_dim, action_dim)
    
    q_values = q_network(single_state)
    print(f"Q-values shape: {q_values.shape}")
    
    # Test Ensemble
    print("\nTesting Ensemble Q-Network...")
    ensemble = EnsembleQNetwork(state_dim, action_dim)
    
    ensemble_q = ensemble(single_state)
    print(f"Ensemble Q-values shape: {ensemble_q.shape}")
    
    mean_q, std_q = ensemble.get_q_value_with_uncertainty(single_state, single_action)
    print(f"Q-value uncertainty - Mean shape: {mean_q.shape}, Std shape: {std_q.shape}")