"""
training.py - Training procedures for all components

This module implements the three-stage training process:
1. Digital Twin Model Training
2. Outcome/Reward Model Training  
3. Offline RL Policy Optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
from collections import deque
import copy
import os
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from models import (
    TransformerDynamicsModel,
    TreatmentOutcomeModel,
    ConservativeQNetwork,
    EnsembleQNetwork
)
import json

class PatientTrajectoryDataset(Dataset):
    """Dataset class for patient trajectories"""

    def __init__(self, data: Dict[str, List], seq_len: int = 10):
        self.data = data
        self.seq_len = seq_len
        self.n_samples = len(data['states'])

        # Tensors
        self.states         = torch.FloatTensor(np.array(data['states']))
        self.actions        = torch.LongTensor(data['actions'])
        self.rewards        = torch.FloatTensor(data['rewards'])
        self.next_states    = torch.FloatTensor(np.array(data['next_states']))
        self.trajectory_ids = torch.LongTensor(data['trajectory_ids'])
        self.timesteps      = torch.LongTensor(data['timesteps'])

        # ===== NEW: compute "done" from real trajectory boundaries =====
        traj = self.trajectory_ids.cpu().numpy()
        ts   = self.timesteps.cpu().numpy()
        # 下一个样本（末尾补一个占位）
        next_traj = np.r_[traj[1:], -1]
        next_ts   = np.r_[ts[1:],   -1]
        dones_np  = ((next_traj != traj) | (next_ts != ts + 1)).astype(np.float32)
        self.dones = torch.from_numpy(dones_np)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'state':         self.states[idx],
            'action':        self.actions[idx],
            'reward':        self.rewards[idx],
            'next_state':    self.next_states[idx],
            'trajectory_id': self.trajectory_ids[idx],
            'timestep':      self.timesteps[idx],
            'done':          self.dones[idx],      # NEW
        }

    def get_sequences(self, batch_size: int = 32):
        """Get sequences for transformer training"""
        # Group by trajectory
        trajectories = {}
        for i in range(self.n_samples):
            traj_id = self.trajectory_ids[i].item()
            if traj_id not in trajectories:
                trajectories[traj_id] = {
                    'states': [], 'actions': [], 'rewards': [], 'timesteps': []
                }
            trajectories[traj_id]['states'].append(self.states[i])
            trajectories[traj_id]['actions'].append(self.actions[i])
            trajectories[traj_id]['rewards'].append(self.rewards[i])
            trajectories[traj_id]['timesteps'].append(self.timesteps[i])

        # Create sequences
        sequences = []
        for traj_id, traj_data in trajectories.items():
            traj_len = len(traj_data['states'])
            # overlapping windows
            for start_idx in range(0, traj_len - 1):
                end_idx = min(start_idx + self.seq_len, traj_len)
                seq_states  = torch.stack(traj_data['states'][start_idx:end_idx])
                seq_actions = torch.stack(traj_data['actions'][start_idx:end_idx])
                seq_rewards = torch.stack(traj_data['rewards'][start_idx:end_idx])
                # pad to fixed length
                actual_len = end_idx - start_idx
                if actual_len < self.seq_len:
                    pad_len = self.seq_len - actual_len
                    seq_states  = torch.cat([seq_states,  torch.zeros(pad_len, seq_states.shape[1])])
                    seq_actions = torch.cat([seq_actions, torch.zeros(pad_len, dtype=torch.long)])
                    seq_rewards = torch.cat([seq_rewards, torch.zeros(pad_len)])
                sequences.append({
                    'states':  seq_states,
                    'actions': seq_actions,
                    'rewards': seq_rewards,
                    'length':  actual_len
                })
        return sequences



class DigitalTwinTrainer:
    """Trainer for the Transformer-based dynamics model"""
    
    def __init__(self,
                 model: TransformerDynamicsModel,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self, sequences: List[Dict]) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Stack batch data
            states = torch.stack([seq['states'] for seq in batch]).to(self.device)
            actions = torch.stack([seq['actions'] for seq in batch]).to(self.device)
            lengths = torch.tensor([seq['length'] for seq in batch])
            
            # 创建attention mask
            max_len = states.shape[1] - 1
            mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=self.device)
            for b, length in enumerate(lengths):
                if length > 1:
                    mask[b, :min(length-1, max_len)] = True

            # Forward pass：与输入序列长度一致 (B, L-1)
            predicted_next_states = self.model(states[:, :-1], actions[:, :-1], mask=mask)
            target_next_states = states[:, 1:]
            
            # 修复：使用masked loss
            if mask.sum() > 0:
                # 只在有效位置计算loss
                pred_masked = predicted_next_states[mask]
                target_masked = target_next_states[mask]
                
                # 使用Huber loss代替MSE（对异常值更鲁棒）
                loss = F.smooth_l1_loss(pred_masked, target_masked)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # 更新学习率调度器
        avg_loss = total_loss / max(n_batches, 1)
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def evaluate(self, sequences: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        n_predictions = 0
        
        with torch.no_grad():
            for seq in sequences[:100]:  # Evaluate on subset
                states = seq['states'].unsqueeze(0).to(self.device)
                actions = seq['actions'].unsqueeze(0).to(self.device)
                length = seq['length']
                
                if length > 1:
                    # 输入给模型的是 states[:, :-1]，它的长度是 S = max_len - 1
                    S = states[:, :-1].size(1)
                    valid = torch.zeros(1, S, dtype=torch.bool, device=self.device)
                    # 只有前 (length-1) 是有效步，其余是 padding（保持 False）
                    valid[0, :max(length-1, 0)] = True

                    predicted = self.model(states[:, :-1], actions[:, :-1], mask=valid)
                    target = states[:, 1:length]
                    predicted = predicted[:, :length-1]
                    
                    mse = F.mse_loss(predicted, target).item()
                    mae = F.l1_loss(predicted, target).item()
                    
                    total_mse += mse
                    total_mae += mae
                    n_predictions += 1
        
        return {
            'mse': total_mse / n_predictions,
            'mae': total_mae / n_predictions
        }


class OutcomeModelTrainer:
    """Trainer for the treatment outcome model"""
    
    def __init__(self,
                 model: TreatmentOutcomeModel,
                 learning_rate: float = 1e-3,
                 regularization_weight: float = 0.01,  # Reduced from 0.1
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.regularization_weight = regularization_weight
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5  # 添加权重衰减
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=5,
            factor=0.5
        )
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_outcome_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            
            # Forward pass
            predicted_rewards = self.model(states, actions).squeeze()
            
            # Outcome prediction loss
            outcome_loss = F.mse_loss(predicted_rewards, rewards)
            
            # Regularization loss for deconfounding
            reg_loss = self.model.compute_regularization_loss(states, actions)
            
            # Total loss
            loss = outcome_loss + self.regularization_weight * reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_outcome_loss += outcome_loss.item()
            total_reg_loss += reg_loss.item()
            n_batches += 1
        
        # self.scheduler.step()  <-- THIS LINE HAS BEEN REMOVED
        
        return {
            'total_loss': total_loss / n_batches,
            'outcome_loss': total_outcome_loss / n_batches,
            'regularization_loss': total_reg_loss / n_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                rewards = batch['reward'].to(self.device)
                
                predicted_rewards = self.model(states, actions).squeeze()
                
                mse = F.mse_loss(predicted_rewards, rewards).item()
                mae = F.l1_loss(predicted_rewards, rewards).item()
                
                total_mse += mse
                total_mae += mae
                n_batches += 1
        
        return {
            'mse': total_mse / n_batches,
            'mae': total_mae / n_batches
        }


    
class ConservativeQLearning:
    def __init__(self, q_network, dynamics_models: List[TransformerDynamicsModel], outcome_model,
                 learning_rate: float = 3e-4,
                 cql_weight: float = 0.10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 128,
                 replay_capacity: int = 50000,   # ← 新增参数，默认5万
                 device: str = "cpu"):
        
        self.q_network = q_network.to(device)
        self.q_target = copy.deepcopy(q_network).to(device)
        self.dynamics_models = [model.to(device) for model in dynamics_models]
        self.dynamics_model = self.dynamics_models[0]
        self.outcome_model = outcome_model.to(device)
        
        self.device = device
        self.replay_capacity = replay_capacity
        self.replay_buffer = deque(maxlen=self.replay_capacity)       
        self.cql_weight = cql_weight
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # 保存维度信息
        self.state_dim = q_network.state_dim
        self.action_dim = q_network.action_dim
        
        # 初始化目标网络并设置为评估模式
        self.q_target.eval()
        for param in self.q_target.parameters():
            param.requires_grad = False
            
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-5
        )
        
        self.training_steps = 0
        
        # 经验回放池
        # self.replay_buffer = deque(maxlen=100000)

        # 更保守的 Q 裁剪与 CQL 目标区间
        self.q_min = -5.0
        self.q_max =  5.0
        self.cql_target_low  = 1.2
        self.cql_target_high = 1.8
        self.cql_warmup      = 500

    def add_to_replay_buffer(self, batch):
        first_key = list(batch.keys())[0]
        first_value = batch[first_key]

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # 批量
        if isinstance(first_value, torch.Tensor) and first_value.dim() > 1:
            B = first_value.size(0)
            dones = batch.get('done', torch.zeros(B, device=first_value.device))
            for i in range(B):
                transition = {
                    'state':      to_numpy(batch['state'][i]).astype('float32'),
                    'action':     int(batch['action'][i].item() if isinstance(batch['action'], torch.Tensor) else batch['action'][i]),
                    'reward':     float(batch['reward'][i].item() if isinstance(batch['reward'], torch.Tensor) else batch['reward'][i]),
                    'next_state': to_numpy(batch['next_state'][i]).astype('float32'),
                    'done':       float(dones[i].item() if isinstance(dones, torch.Tensor) else dones[i]),
                }
                self.replay_buffer.append(transition)
        else:
            dones = batch.get('done', 0.0)
            transition = {
                'state':      to_numpy(batch['state']).astype('float32'),
                'action':     int(batch['action'].item() if hasattr(batch['action'], 'item') else batch['action']),
                'reward':     float(batch['reward'].item() if hasattr(batch['reward'], 'item') else batch['reward']),
                'next_state': to_numpy(batch['next_state']).astype('float32'),
                'done':       float(dones.item() if hasattr(dones, 'item') else dones),
            }
            self.replay_buffer.append(transition)
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        idxs = np.random.choice(len(self.replay_buffer), batch_size, replace=False)

        states      = np.stack([self.replay_buffer[i]['state']      for i in idxs]).astype(np.float32)
        actions     = np.array( [self.replay_buffer[i]['action']     for i in idxs]).astype(np.int64)
        rewards     = np.array( [self.replay_buffer[i]['reward']     for i in idxs]).astype(np.float32)
        next_states = np.stack([self.replay_buffer[i]['next_state'] for i in idxs]).astype(np.float32)
        dones       = np.array( [self.replay_buffer[i].get('done',0) for i in idxs]).astype(np.float32)

        d = self.device
        return {
            'state':      torch.as_tensor(states,      device=d),
            'action':     torch.as_tensor(actions,     device=d),
            'reward':     torch.as_tensor(rewards,     device=d),
            'next_state': torch.as_tensor(next_states, device=d),
            'done':       torch.as_tensor(dones,       device=d),
        }


    def compute_cql_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_values = self.q_network(states)                          # (B, A)
        q_values = torch.clamp(q_values, self.q_min, self.q_max)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        logsumexp_all = torch.logsumexp(q_values, dim=1)
        cql = torch.relu(logsumexp_all - q_taken).mean()           # 非负
        cql = cql + 1e-3 * q_values.abs().mean()                   # 轻正则
        return cql


    def train_step(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.q_network.train()
        batch = self.sample_batch(self.batch_size)

        states      = batch['state']
        actions     = batch['action'].long()
        rewards     = batch['reward'].view(-1)
        next_states = batch['next_state']
        dones       = batch['done'].view(-1)

        # Double DQN target with done mask
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q = next_q.clamp(self.q_min, self.q_max)
            r = rewards.clamp(-10.0, 10.0)
            target_q = (r + self.gamma * (1.0 - dones) * next_q).clamp(self.q_min, self.q_max)

        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        bellman_loss = F.smooth_l1_loss(current_q, target_q)
        cql_loss = self.compute_cql_loss(states, actions)

        # Behavior Cloning warmup
        bc_loss = torch.tensor(0.0, device=self.device)
        bc_warmup    = 500
        # 在warmup期间，bc_coef从0.1线性衰减到0
        bc_coef = 0.1 * max(0.0, 1.0 - self.training_steps / bc_warmup) 
        # 在warmup之后，始终保留一个微小的bc_tail_coef
        bc_tail_coef = 0.01 
        
        # 始终计算BC Loss
        bc_loss = F.cross_entropy(current_q_values, actions)

        # 总损失：在warmup期间使用较强的BC，之后使用微弱的BC
        if self.training_steps < bc_warmup:
            total_loss = bellman_loss + self.cql_weight * cql_loss + bc_coef * bc_loss
        else:
            total_loss = bellman_loss + self.cql_weight * cql_loss + bc_tail_coef * bc_loss

        # 轻量 Q 正则
        total_loss = total_loss + 1e-4 * (current_q_values ** 2).mean()

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0) # 从 10.0 修改为 1.0
        self.optimizer.step()
        self.scheduler.step()      # 余弦退火

        # 每步软更新
        self._soft_update_target()
        self.training_steps += 1

        return {
            'total_loss':   float(total_loss.item()),
            'bellman_loss': float(bellman_loss.item()),
            'cql_loss':     float(cql_loss.item()),
            'mean_q':       float(current_q.mean().item()),
            'max_q':        float(current_q.max().item()),
            'min_q':        float(current_q.min().item()),
            'cql_weight':   float(self.cql_weight),
        }


    def _soft_update_target(self):
        """软更新目标网络的权重"""
        for target_param, local_param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_policy(self, state: torch.Tensor) -> int:
        """根据学习到的策略获取动作"""
        self.q_network.eval()
        with torch.no_grad():
            # 确保输入状态是正确的设备和维度
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            q_values = self.q_network(state)
            action = q_values.argmax(dim=1).item()
        return action

    def evaluate_policy(
        self,
        dataset: 'PatientTrajectoryDataset',
        n_episodes: int = 50,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        max_horizon: int = 50,
        spo2_idx: int = 8,
        spo2_threshold: float = 0.80
    ) -> Dict[str, float]:
        """使用动力学模型集成评估策略，和 evaluation.py 的设置保持一致"""
        self.q_network.eval()
        self.outcome_model.eval()
        for m in self.dynamics_models:
            m.eval()

        returns, lengths = [], []

        with torch.no_grad():
            for _ in range(n_episodes):
                idx = np.random.randint(len(dataset))
                state = dataset[idx]['state'].to(self.device)

                ep_ret = 0.0
                state_hist = [state]
                action_hist: List[int] = []

                for t in range(max_horizon):
                    # 贪心动作
                    qv = self.q_network(state.unsqueeze(0))
                    action = int(qv.argmax(dim=1).item())
                    action_hist.append(action)

                    # 奖励（保持与训练一致：标准化）
                    raw_reward = self.outcome_model(
                        state.unsqueeze(0),
                        torch.tensor([action], device=self.device, dtype=torch.long)
                    ).item()
                    reward = (raw_reward - reward_mean) / max(reward_std, 1e-6)
                    ep_ret += reward * (0.99 ** t)

                    # 集成预测下一状态
                    states_seq  = torch.stack(state_hist).unsqueeze(0)
                    actions_seq = torch.tensor(action_hist, device=self.device, dtype=torch.long).unsqueeze(0)
                    preds = [m.predict_next_state(states_seq, actions_seq) for m in self.dynamics_models]
                    next_state = torch.stack(preds).mean(dim=0).squeeze(0).squeeze(0)
                    next_state = torch.clamp(next_state, 0.0, 1.0)

                    # 与评估一致的安全终止
                    if next_state[spo2_idx] < spo2_threshold:
                        lengths.append(t + 1)
                        break

                    state = next_state
                    state_hist.append(state)
                    if len(state_hist) > 10:
                        state_hist.pop(0)
                        action_hist.pop(0)
                else:
                    lengths.append(max_horizon)

                returns.append(ep_ret)

        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_episode_length': float(np.mean(lengths))
        }



def train_digital_twin(data: Dict[str, List],
                      state_dim: int,
                      action_dim: int,
                      n_epochs: int = 50,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                      save_dir: str = '.') -> TransformerDynamicsModel:
    """Train the Transformer-based dynamics model"""
    print("Training Digital Twin Model...")
    
    # Prepare data
    dataset = PatientTrajectoryDataset(data)
    sequences = dataset.get_sequences()
    
    # Split train/val
    n_train = int(0.8 * len(sequences))
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:]
    
    # Initialize model and trainer
    model = TransformerDynamicsModel(state_dim, action_dim, dropout=0.2)
    trainer = DigitalTwinTrainer(model, device=device)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        # Train
        train_loss = trainer.train_epoch(train_sequences)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_sequences)
        
        # Log
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val MSE: {val_metrics['mse']:.4f}")
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            save_path = os.path.join(save_dir, 'best_dynamics_model.pth')
            torch.save(model.state_dict(), save_path)
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model


def train_outcome_model(data: Dict[str, List],
                       state_dim: int,
                       action_dim: int,
                       n_epochs: int = 30,
                       batch_size: int = 256,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                       save_dir: str = '.') -> TreatmentOutcomeModel:
    """Train the treatment outcome model"""
    print("\nTraining Treatment Outcome Model...")
    
    # Prepare data
    dataset = PatientTrajectoryDataset(data)
    
    # Split train/val
    n_train = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = TreatmentOutcomeModel(state_dim, action_dim)
    trainer = OutcomeModelTrainer(model, device=device)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_metrics['total_loss']:.4f}, Val MSE: {val_metrics['mse']:.4f}")
        
        # Save best model
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            save_path = os.path.join(save_dir, 'best_outcome_model.pth')
            torch.save(model.state_dict(), save_path)
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model


def train_rl_policy(data: Dict[str, List],
                   dynamics_models: List[TransformerDynamicsModel],  # 注意是列表
                   outcome_model: TreatmentOutcomeModel,
                   state_dim: int,
                   action_dim: int,
                   n_iterations: int = 50000,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   save_dir: str = '.') -> ConservativeQNetwork:
    """Train the RL policy using Conservative Q-Learning with validation"""
    print("\nTraining RL Policy...")
    
    # Initialize Q-network and trainer
    q_network = ConservativeQNetwork(state_dim, action_dim)
    trainer = ConservativeQLearning(
        q_network, dynamics_model, outcome_model,
        learning_rate=3e-4,  # 可以稍微提高学习率
        cql_weight=0.1,
        device=device
    )
    
    # Prepare dataset
    dataset = PatientTrajectoryDataset(data)
    
    # 分割训练集和验证集
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=pin, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=pin, persistent_workers=True
    )
    reward_mean = float(np.mean(data['rewards']))
    reward_std  = float(np.std(data['rewards']) + 1e-6)
    print(f"Normalizing rewards: mean={reward_mean:.3f}, std={reward_std:.3f}")

    # Fill replay buffer with training data only
    print("Filling replay buffer...")
    for batch in tqdm(train_loader):
        # normalize rewards in-place (Tensor on CPU here)
        batch['reward'] = (batch['reward'] - reward_mean) / reward_std
        trainer.add_to_replay_buffer(batch)

    stats_path = os.path.join(save_dir, 'reward_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({'mean': float(reward_mean), 'std': float(reward_std)}, f)
    print(f"Saved reward stats to {stats_path}")

    # Training loop with early stopping
    losses = deque(maxlen=100)
    best_val_loss = float('inf')
    patience = 5000
    patience_counter = 0
    
    for iteration in tqdm(range(n_iterations)):
        # Train step
        metrics = trainer.train_step()
        
        if metrics:
            losses.append(metrics['total_loss'])
            
            # Validation and logging every 1000 iterations
            if iteration % 1000 == 0 and iteration > 0:
                # 验证性能
                val_loss = 0.0
                n_val_batches = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        states = val_batch['state'].to(device)
                        actions = val_batch['action'].to(device)
                        rewards = val_batch['reward'].to(device)
                        next_states = val_batch['next_state'].to(device)
                        
                        # 计算验证损失
                        q_values = trainer.q_network(states)
                        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                        
                        with torch.no_grad():
                            next_q_values = trainer.q_target(next_states)
                            next_q_max = next_q_values.max(dim=1)[0]
                            target_q = rewards + trainer.gamma * next_q_max
                        
                        val_bellman_loss = F.mse_loss(q_taken, target_q)
                        val_loss += val_bellman_loss.item()
                        n_val_batches += 1
                
                avg_val_loss = val_loss / max(n_val_batches, 1)
                
                print(f"\nIteration {iteration} - Train Loss: {np.mean(losses):.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(q_network.state_dict(), 
                             os.path.join(save_dir, 'best_q_network_checkpoint.pth'))
                else:
                    patience_counter += 1000
                    
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
    
    # 加载最佳模型
    if os.path.exists(os.path.join(save_dir, 'best_q_network_checkpoint.pth')):
        q_network.load_state_dict(torch.load(os.path.join(save_dir, 'best_q_network_checkpoint.pth')))
    
    # Final evaluation
    print("\nFinal evaluation...")
    # 传入 reward_mean 和 reward_std
    final_metrics = trainer.evaluate_policy(dataset, n_episodes=100, reward_mean=reward_mean, reward_std=reward_std) 
    print(f"Final Mean Return: {final_metrics['mean_return']:.4f} ± {final_metrics['std_return']:.4f}")
    print(f"Mean Episode Length: {final_metrics['mean_episode_length']:.2f}")
    
    # Save model
    save_path = os.path.join(save_dir, 'best_q_network.pth')
    torch.save(q_network.state_dict(), save_path)
    
    return q_network


if __name__ == "__main__":
    # Example usage
    from data import PatientDataGenerator
    
    # Generate data
    print("Generating patient data...")
    generator = PatientDataGenerator(n_patients=1000)
    data = generator.generate_dataset()
    
    state_dim = generator.n_features
    action_dim = generator.n_actions
    
    # Stage 1: Train dynamics model
    dynamics_model = train_digital_twin(data, state_dim, action_dim)
    
    # Stage 2: Train outcome model
    outcome_model = train_outcome_model(data, state_dim, action_dim)
    
    # Stage 3: Train RL policy
    q_network = train_rl_policy(data, dynamics_model, outcome_model, state_dim, action_dim)
    
    print("\nTraining complete!")