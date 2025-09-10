"""
test_offline_training.py - 综合测试和诊断离线训练的效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import time
from collections import defaultdict, deque
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
# 导入必要的模块
from data import PatientDataGenerator
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork, EnsembleQNetwork
from training import (
    train_digital_twin,
    train_outcome_model,
    train_rl_policy,
    DigitalTwinTrainer,
    OutcomeModelTrainer,
    ConservativeQLearning,
    PatientTrajectoryDataset
)
from torch.utils.data import DataLoader
import torch.nn.functional as F


class OfflineTrainingDiagnostics:
    """离线训练诊断工具"""
    
    def __init__(self, output_dir: str = "./training_diagnostics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建子目录
        self.plots_dir = os.path.join(output_dir, f"plots_{self.timestamp}")
        self.logs_dir = os.path.join(output_dir, f"logs_{self.timestamp}")
        self.models_dir = os.path.join(output_dir, f"models_{self.timestamp}")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 训练历史
        self.training_history = defaultdict(lambda: defaultdict(list))
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def test_data_generation(self, n_patients: int = 100) -> Dict:
        """测试数据生成质量"""
        print("\n" + "="*60)
        print("1. Testing Data Generation")
        print("="*60)
        
        generator = PatientDataGenerator(n_patients=n_patients, seed=42)
        data = generator.generate_dataset()
        
        # 统计信息
        stats = {
            'n_transitions': len(data['states']),
            'n_patients': n_patients,
            'avg_trajectory_length': len(data['states']) / n_patients,
            'state_dim': len(data['states'][0]) if data['states'] else 0,
            'action_dim': len(set(data['actions'])),
            'reward_mean': np.mean(data['rewards']),
            'reward_std': np.std(data['rewards']),
            'reward_min': np.min(data['rewards']),
            'reward_max': np.max(data['rewards'])
        }
        
        # 绘制数据分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 动作分布
        ax = axes[0, 0]
        action_counts = pd.Series(data['actions']).value_counts().sort_index()
        ax.bar(action_counts.index, action_counts.values)
        ax.set_xlabel('Action')
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        
        # 2. 奖励分布
        ax = axes[0, 1]
        ax.hist(data['rewards'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(data['rewards']), color='red', linestyle='--', label=f"Mean: {np.mean(data['rewards']):.3f}")
        ax.axvline(np.median(data['rewards']), color='green', linestyle='--', label=f"Median: {np.median(data['rewards']):.3f}")
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        
        # 3. 轨迹长度分布
        ax = axes[0, 2]
        traj_lengths = pd.Series(data['trajectory_ids']).value_counts().values
        ax.hist(traj_lengths, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Trajectory Length')
        ax.set_ylabel('Count')
        ax.set_title('Trajectory Length Distribution')
        
        # 4. 状态特征相关性
        ax = axes[1, 0]
        states_array = np.array(data['states'])
        corr_matrix = np.corrcoef(states_array.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('State Feature Correlations')
        plt.colorbar(im, ax=ax)
        
        # 5. 状态转移示例
        ax = axes[1, 1]
        sample_traj_id = data['trajectory_ids'][0]
        sample_indices = [i for i, tid in enumerate(data['trajectory_ids']) if tid == sample_traj_id][:20]
        sample_states = np.array([data['states'][i] for i in sample_indices])
        
        for feature_idx in [2, 4, 8]:  # BP, Glucose, O2
            ax.plot(sample_states[:, feature_idx], label=f'Feature {feature_idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Sample State Trajectory')
        ax.legend()
        
        # 6. 数据统计表
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""Data Statistics:
        
Total Transitions: {stats['n_transitions']}
Number of Patients: {stats['n_patients']}
Avg Trajectory Length: {stats['avg_trajectory_length']:.1f}
State Dimension: {stats['state_dim']}
Action Space: {stats['action_dim']}

Reward Statistics:
Mean: {stats['reward_mean']:.3f}
Std: {stats['reward_std']:.3f}
Min: {stats['reward_min']:.3f}
Max: {stats['reward_max']:.3f}"""
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Data Generation Diagnostics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'data_diagnostics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Data generated successfully")
        print(f"  - Transitions: {stats['n_transitions']}")
        print(f"  - Reward range: [{stats['reward_min']:.3f}, {stats['reward_max']:.3f}]")
        
        return data, stats
    
    def test_dynamics_model(self, data: Dict, epochs: int = 20, n_ensemble: int = 5) -> List[TransformerDynamicsModel]:
        """测试Dynamics模型训练（集成版本）"""
        print("\n" + "="*60)
        print(f"2. Testing Dynamics Model Training (Ensemble of {n_ensemble})")
        print("="*60)
        
        state_dim = len(data['states'][0])
        action_dim = len(set(data['actions']))
        
        # 准备数据
        dataset = PatientTrajectoryDataset(data)
        sequences = dataset.get_sequences()
        n_train = int(0.8 * len(sequences))
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:]
        
        ensemble_models = []
        for i in range(n_ensemble):
            print(f"  Training model {i+1}/{n_ensemble}...")
            # 为每个模型使用不同的随机初始化
            model = TransformerDynamicsModel(state_dim, action_dim)
            nn.init.xavier_uniform_(model.output_projection[-1].weight) # 确保每次初始化都正确
            nn.init.zeros_(model.output_projection[-1].bias)

            trainer = DigitalTwinTrainer(model)
            
            # 训练循环
            for epoch in range(epochs):
                train_loss = trainer.train_epoch(train_sequences)
                if (epoch + 1) % 10 == 0: # 每10轮打印一次日志
                    val_metrics = trainer.evaluate(val_sequences)
                    print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val MSE: {val_metrics['mse']:.4f}")
            
            model.eval() # 设为评估模式
            ensemble_models.append(model)
            print(f"  ✓ Model {i+1} trained.")

        # 仅为最后一个模型绘制预测质量图
        self._test_dynamics_predictions(ensemble_models[-1], val_sequences)
        
        print(f"✓ Dynamics model ensemble trained successfully.")
        return ensemble_models
    
    def test_outcome_model(self, data: Dict, epochs: int = 20) -> TreatmentOutcomeModel:
        """测试Outcome模型训练"""
        print("\n" + "="*60)
        print("3. Testing Outcome Model Training")
        print("="*60)
        
        state_dim = len(data['states'][0])
        action_dim = len(set(data['actions']))
        
        # 初始化模型
        model = TreatmentOutcomeModel(state_dim, action_dim)
        trainer = OutcomeModelTrainer(model, regularization_weight=0.01)
        
        # 准备数据
        dataset = PatientTrajectoryDataset(data)
        n_train = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train]
        )
        
        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=4, pin_memory=pin, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=pin, persistent_workers=True
        )
        
        # 训练循环
        train_losses = []
        val_losses = []
        reg_losses = []
        
        for epoch in range(epochs):
            # 训练
            epoch_metrics = trainer.train_epoch(train_loader)
            train_losses.append(epoch_metrics['outcome_loss'])
            reg_losses.append(epoch_metrics['regularization_loss'])
            
            # 验证
            val_metrics = trainer.evaluate(val_loader)
            val_losses.append(val_metrics['mse'])
            
            print(f"  Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {epoch_metrics['outcome_loss']:.4f}, "
                  f"Reg Loss: {epoch_metrics['regularization_loss']:.4f}, "
                  f"Val MSE: {val_metrics['mse']:.4f}")
            trainer.scheduler.step(val_metrics['mse'])
            # 记录
            self.training_history['outcome']['train_loss'].append(epoch_metrics['outcome_loss'])
            self.training_history['outcome']['val_loss'].append(val_metrics['mse'])
            self.training_history['outcome']['reg_loss'].append(epoch_metrics['regularization_loss'])
        
        # 绘制训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss曲线
        ax = axes[0]
        ax.plot(train_losses, label='Train Loss', linewidth=2)
        ax.plot(val_losses, label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Outcome Model - Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 正则化损失
        ax = axes[1]
        ax.plot(reg_losses, label='Regularization Loss', color='orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reg Loss')
        ax.set_title('Deconfounding Regularization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Outcome Model Training Diagnostics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'outcome_training.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Outcome model trained")
        print(f"  - Final train loss: {train_losses[-1]:.4f}")
        print(f"  - Final val loss: {val_losses[-1]:.4f}")
        
        return model

    def _plot_enhanced_rl_diagnostics(self, metrics_history: Dict):
        """绘制增强的RL训练诊断图"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. 总损失
        ax = axes[0, 0]
        if 'total_loss' in metrics_history and metrics_history['total_loss']:
            ax.plot(metrics_history['total_loss'], linewidth=2, alpha=0.7)
            # 添加移动平均
            if len(metrics_history['total_loss']) > 50:
                ma = pd.Series(metrics_history['total_loss']).rolling(50).mean()
                ax.plot(ma, linewidth=2, label='MA(50)', color='red')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Total Loss')
            ax.set_title('Total Loss Over Training')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 2. Bellman损失
        ax = axes[0, 1]
        if 'bellman_loss' in metrics_history and metrics_history['bellman_loss']:
            ax.plot(metrics_history['bellman_loss'], color='green', linewidth=2, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Bellman Loss')
            ax.set_title('TD Error (Bellman Loss)')
            ax.grid(True, alpha=0.3)
        
        # 3. CQL损失
        ax = axes[0, 2]
        if 'cql_loss' in metrics_history and metrics_history['cql_loss']:
            ax.plot(metrics_history['cql_loss'], color='orange', linewidth=2, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('CQL Loss')
            ax.set_title('Conservative Q-Learning Penalty')
            ax.grid(True, alpha=0.3)
        
        # 4. Q值统计（均值、最大、最小）
        ax = axes[1, 0]
        if all(k in metrics_history for k in ['mean_q', 'max_q', 'min_q']):
            if metrics_history['mean_q']:
                ax.plot(metrics_history['mean_q'], label='Mean Q', linewidth=2)
                ax.plot(metrics_history['max_q'], label='Max Q', linewidth=1, alpha=0.5)
                ax.plot(metrics_history['min_q'], label='Min Q', linewidth=1, alpha=0.5)
                ax.fill_between(range(len(metrics_history['mean_q'])), 
                            metrics_history['min_q'], 
                            metrics_history['max_q'], 
                            alpha=0.2)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Q-Value')
                ax.set_title('Q-Value Statistics')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 5. 返回值（Policy Performance）
        ax = axes[1, 1]
        if 'mean_return' in metrics_history and metrics_history['mean_return']:
            eval_steps = np.linspace(500, len(metrics_history.get('total_loss', [])), 
                                    len(metrics_history['mean_return']))
            ax.plot(eval_steps, metrics_history['mean_return'], 
                'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Mean Return')
            ax.set_title('Policy Performance (Return)')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(metrics_history['mean_return']) > 2:
                z = np.polyfit(eval_steps, metrics_history['mean_return'], 1)
                p = np.poly1d(z)
                ax.plot(eval_steps, p(eval_steps), "b--", alpha=0.5, label='Trend')
                ax.legend()
        
        # 6. 损失比率
        ax = axes[1, 2]
        if 'bellman_loss' in metrics_history and 'cql_loss' in metrics_history:
            if metrics_history['bellman_loss'] and metrics_history['cql_loss']:
                cql = np.array(metrics_history['cql_loss'])
                td  = np.array(metrics_history['bellman_loss']) + 1e-8
                ax.plot(cql / td, color='brown', linewidth=2, alpha=0.7, label='raw')
                if 'cql_weight' in metrics_history and metrics_history['cql_weight']:
                    w = np.array(metrics_history['cql_weight'])
                    ax.plot((w * cql) / td, 'k--', linewidth=2, alpha=0.7, label='weighted')
                    ax.legend()
                ax.set_xlabel('Training Step')
                ax.set_ylabel('CQL/Bellman Ratio')
                ax.set_title('Conservative vs TD Loss Ratio')
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3)
        
        # 7. Q值方差（稳定性指标）
        ax = axes[2, 0]
        if 'q_variance' in metrics_history and metrics_history['q_variance']:
            ax.plot(metrics_history['q_variance'], color='purple', linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Q-Value Variance')
            ax.set_title('Q-Value Stability (Lower is Better)')
            ax.grid(True, alpha=0.3)
        
        # 8. 梯度范数
        ax = axes[2, 1]
        if 'gradient_norm' in metrics_history and metrics_history['gradient_norm']:
            ax.plot(metrics_history['gradient_norm'], color='teal', linewidth=2, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Magnitude')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 9. 训练诊断总结
        ax = axes[2, 2]
        ax.axis('off')
        
        # 生成诊断总结
        summary_text = "RL Training Summary\n" + "="*25 + "\n\n"
        
        if metrics_history.get('total_loss'):
            final_loss = metrics_history['total_loss'][-1]
            initial_loss = metrics_history['total_loss'][0]
            loss_change = ((final_loss - initial_loss) / initial_loss) * 100
            summary_text += f"Loss Change: {loss_change:+.1f}%\n"
            summary_text += f"Final Loss: {final_loss:.4f}\n\n"
        
        if metrics_history.get('mean_q'):
            final_q = metrics_history['mean_q'][-1]
            min_q = min(metrics_history['min_q']) if metrics_history.get('min_q') else 0
            max_q = max(metrics_history['max_q']) if metrics_history.get('max_q') else 0
            summary_text += f"Q-Value Range: [{min_q:.2f}, {max_q:.2f}]\n"
            summary_text += f"Final Mean Q: {final_q:.2f}\n\n"
            
            # 健康检查
            if final_q < -10:
                summary_text += "⚠️ Warning: Q-values very negative\n"
            elif final_q > 10:
                summary_text += "⚠️ Warning: Q-values possibly too high\n"
            else:
                summary_text += "✓ Q-values in reasonable range\n"
        
        if metrics_history.get('mean_return'):
            returns = metrics_history['mean_return']
            if len(returns) > 1:
                improvement = returns[-1] - returns[0]
                summary_text += f"\nReturn Improvement: {improvement:+.3f}\n"
                summary_text += f"Final Return: {returns[-1]:.3f}\n"
                
                if improvement > 0:
                    summary_text += "✓ Policy improving\n"
                else:
                    summary_text += "⚠️ Policy degrading\n"
        
        # 渐变范数分析
        if metrics_history.get('gradient_norm'):
            avg_grad = np.mean(metrics_history['gradient_norm'][-100:])
            summary_text += f"\nAvg Gradient Norm: {avg_grad:.4f}\n"
            if avg_grad < 0.0001:
                summary_text += "⚠️ Gradients vanishing\n"
            elif avg_grad > 10:
                summary_text += "⚠️ Gradients exploding\n"
            else:
                summary_text += "✓ Gradients stable\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                family='monospace')
        
        plt.suptitle('Enhanced RL Training Diagnostics (CQL)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rl_diagnostics_enhanced.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Enhanced RL diagnostics plot saved")


    def test_rl_training(self, data: Dict, dynamics_models: List[TransformerDynamicsModel],
                        outcome_model: TreatmentOutcomeModel, iterations: int = 5000) -> ConservativeQNetwork:
        """增强的RL训练测试"""
        print("\n" + "="*60)
        print("4. Testing RL Policy Training (CQL)")
        print("="*60)
        
        state_dim = len(data['states'][0])
        action_dim = len(set(data['actions']))
        
        # 初始化Q网络
        q_network = ConservativeQNetwork(state_dim, action_dim)
        trainer = ConservativeQLearning(
            q_network, dynamics_models, outcome_model,
            learning_rate=1e-4,  # 更低的学习率
            cql_weight=0.1  # 降低CQL权重
        )
        
        # 准备数据
        dataset = PatientTrajectoryDataset(data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # 填充replay buffer
        print("  Filling replay buffer...")
        # 计算数据集奖励统计量
        all_rewards = []
        for b in dataloader:
            all_rewards.append(b['reward'].numpy())
        r = np.concatenate(all_rewards, axis=0)
        r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
        print(f"  Normalizing rewards: mean={r_mean:.3f}, std={r_std:.3f}")
        # 归一化后再入池
        for batch in dataloader:
            batch['reward'] = (batch['reward'] - r_mean) / r_std
            # 不再覆盖 done
            trainer.add_to_replay_buffer(batch)

        # 增强的指标追踪
        metrics_history = {
            'total_loss': [], 'bellman_loss': [], 'cql_loss': [],
            'mean_q': [], 'max_q': [], 'min_q': [],
            'mean_return': [], 'q_variance': [], 'gradient_norm': [],
            'cql_weight': []
        }

        print("  Starting training...")
        for i in range(iterations):
            metrics = trainer.train_step()
            if not metrics:
                continue

            # 记录标量
            for key in ('total_loss','bellman_loss','cql_loss',
                        'mean_q','max_q','min_q','mean_return',
                        'q_variance','gradient_norm'):
                if key in metrics:
                    metrics_history[key].append(float(metrics[key]))

            # cql_weight 单独追加一次（不要放到上面的 for 循环里）
            metrics_history['cql_weight'].append(
                float(metrics.get('cql_weight', trainer.cql_weight))
            )

            # 计算 Q 值方差（最近10步）；步数不够时先补 NaN 以对齐长度
            if len(metrics_history['mean_q']) >= 10:
                recent_q = metrics_history['mean_q'][-10:]
                metrics_history['q_variance'].append(float(np.var(recent_q)))
            else:
                metrics_history['q_variance'].append(float('nan'))

            # 计算梯度范数
            total_norm = 0.0
            for p in q_network.parameters():
                if p.grad is not None:
                    g = p.grad.data.norm(2).item()
                    total_norm += g * g
            total_norm = total_norm ** 0.5
            metrics_history['gradient_norm'].append(total_norm)

            # 定期评估
            if (i + 1) % 500 == 0:
                eval_metrics = trainer.evaluate_policy(dataset, n_episodes=20, reward_mean=r_mean, reward_std=r_std)
                metrics_history['mean_return'].append(eval_metrics['mean_return'])
                print(
                    f"  Iteration {i+1}/{iterations} - "
                    f"Loss: {metrics['total_loss']:.4f}, "
                    f"Q: [{metrics['min_q']:.2f}, {metrics['mean_q']:.2f}, {metrics['max_q']:.2f}], "
                    f"Return: {eval_metrics['mean_return']:.3f}"
                )
                # 早停检查
                if metrics['mean_q'] < -50:
                    print("  ⚠️ Q-values collapsing, stopping early")
                    break
        
        # 绘制增强的诊断图
        self._plot_enhanced_rl_diagnostics(metrics_history)
        
        return q_network
    
    def _plot_training_curve(self, train_losses: List, val_losses: List, 
                           title: str, filename: str):
        """绘制训练曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        # 标记最佳验证loss
        best_epoch = np.argmin(val_losses)
        ax.scatter([best_epoch + 1], [val_losses[best_epoch]], 
                  color='green', s=100, zorder=5)
        ax.annotate(f'Best: {val_losses[best_epoch]:.4f}',
                   xy=(best_epoch + 1, val_losses[best_epoch]),
                   xytext=(best_epoch + 1, val_losses[best_epoch] + 0.01),
                   ha='center')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _test_dynamics_predictions(self, model: TransformerDynamicsModel, sequences: List):
        """测试dynamics模型的预测质量"""
        model.eval()
        device = next(model.parameters()).device
        
        # 随机选择一些序列进行可视化
        sample_sequences = np.random.choice(sequences, min(5, len(sequences)), replace=False)
        
        fig, axes = plt.subplots(len(sample_sequences), 2, figsize=(12, 3*len(sample_sequences)))
        if len(sample_sequences) == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for idx, seq in enumerate(sample_sequences):
                if seq['length'] < 2:
                    continue
                    
                states = seq['states'].unsqueeze(0).to(device)
                actions = seq['actions'].unsqueeze(0).to(device)

                # 预测（传有效步 mask）
                S = states[:, :-1].size(1)
                valid = torch.zeros(1, S, dtype=torch.bool, device=device)
                valid[0, :seq['length']-1] = True

                predicted = model(states[:, :-1], actions[:, :-1], mask=valid)
                predicted = predicted[:, :seq['length']-1].squeeze(0).cpu().numpy()
                actual = states[:, 1:seq['length']].squeeze(0).cpu().numpy()

                
                # 绘制关键特征的预测vs实际
                ax = axes[idx, 0]
                feature_idx = 4  # Glucose
                ax.plot(predicted[:, feature_idx], 'b-', label='Predicted')
                ax.plot(actual[:, feature_idx], 'r--', label='Actual')
                ax.set_title(f'Seq {idx}: Glucose Prediction')
                ax.set_xlabel('Time Step')
                ax.legend()
                
                ax = axes[idx, 1]
                feature_idx = 8  # O2
                ax.plot(predicted[:, feature_idx], 'b-', label='Predicted')
                ax.plot(actual[:, feature_idx], 'r--', label='Actual')
                ax.set_title(f'Seq {idx}: O2 Saturation Prediction')
                ax.set_xlabel('Time Step')
                ax.legend()
        
        plt.suptitle('Dynamics Model Prediction Quality')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dynamics_predictions.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rl_diagnostics(self, metrics_history: Dict):
        """绘制RL训练诊断图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 总损失
        ax = axes[0, 0]
        ax.plot(metrics_history['total_loss'], linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss Over Training')
        ax.grid(True, alpha=0.3)
        
        # 2. Bellman损失
        ax = axes[0, 1]
        ax.plot(metrics_history['bellman_loss'], color='green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Bellman Loss')
        ax.set_title('TD Error (Bellman Loss)')
        ax.grid(True, alpha=0.3)
        
        # 3. CQL损失
        ax = axes[0, 2]
        ax.plot(metrics_history['cql_loss'], color='orange', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('CQL Loss')
        ax.set_title('Conservative Q-Learning Penalty')
        ax.grid(True, alpha=0.3)
        
        # 4. 平均Q值
        ax = axes[1, 0]
        ax.plot(metrics_history['mean_q'], color='purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Q-Value')
        ax.set_title('Average Q-Value Evolution')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 5. 返回值（如果有）
        ax = axes[1, 1]
        if metrics_history['mean_return']:
            eval_steps = np.linspace(500, len(metrics_history['total_loss']), 
                                    len(metrics_history['mean_return']))
            ax.plot(eval_steps, metrics_history['mean_return'], 
                   'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Mean Return')
            ax.set_title('Policy Performance (Return)')
            ax.grid(True, alpha=0.3)
        
        # 6. 损失比率
        ax = axes[1, 2]
        if len(metrics_history['bellman_loss']) > 0:
            cql_ratio = np.array(metrics_history['cql_loss']) / (
                np.array(metrics_history['bellman_loss']) + 1e-8
            )
            ax.plot(cql_ratio, color='brown', linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('CQL/Bellman Ratio')
            ax.set_title('Conservative vs TD Loss Ratio')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('RL Training Diagnostics (CQL)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'rl_diagnostics.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_convergence_analysis(self):
        """分析收敛性"""
        print("\n" + "="*60)
        print("5. Convergence Analysis")
        print("="*60)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Dynamics模型收敛分析
        ax = axes[0]
        if 'dynamics' in self.training_history:
            losses = self.training_history['dynamics']['train_loss']
            if len(losses) > 5:
                # 计算移动平均
                window = 3
                ma = pd.Series(losses).rolling(window=window).mean()
                ax.plot(losses, alpha=0.3, label='Raw')
                ax.plot(ma, linewidth=2, label=f'MA({window})')
                
                # 检测收敛
                if len(losses) > 10:
                    recent_std = np.std(losses[-5:])
                    converged = recent_std < 0.01
                    ax.set_title(f'Dynamics: {"Converged" if converged else "Not Converged"}')
                else:
                    ax.set_title('Dynamics Model')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        
        # Outcome模型收敛分析
        ax = axes[1]
        if 'outcome' in self.training_history:
            losses = self.training_history['outcome']['train_loss']
            if len(losses) > 5:
                window = 3
                ma = pd.Series(losses).rolling(window=window).mean()
                ax.plot(losses, alpha=0.3, label='Raw')
                ax.plot(ma, linewidth=2, label=f'MA({window})')
                
                if len(losses) > 10:
                    recent_std = np.std(losses[-5:])
                    converged = recent_std < 0.01
                    ax.set_title(f'Outcome: {"Converged" if converged else "Not Converged"}')
                else:
                    ax.set_title('Outcome Model')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
        
        # 整体训练健康度
        ax = axes[2]
        ax.axis('off')
        
        health_report = self._generate_health_report()
        ax.text(0.1, 0.5, health_report, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.suptitle('Convergence Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'convergence_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(health_report)
    
    def _generate_health_report(self) -> str:
        """生成训练健康报告"""
        report = "Training Health Report\n" + "="*30 + "\n\n"
        
        # Dynamics模型
        if 'dynamics' in self.training_history:
            train_losses = self.training_history['dynamics']['train_loss']
            val_losses = self.training_history['dynamics']['val_loss']
            
            if train_losses and val_losses:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                eps = max(1e-4, 0.1 * min(final_train, final_val))
                overfit_ratio = (final_val + eps) / (final_train + eps)
                
                report += f"Dynamics Model:\n"
                report += f"  Final Train Loss: {final_train:.4f}\n"
                report += f"  Final Val Loss: {final_val:.4f}\n"
                report += f"  Overfit Ratio: {overfit_ratio:.2f}\n"
                report += f"  Status: {'⚠ Overfitting' if overfit_ratio > 1.5 else '✓ Good'}\n\n"
        
        # Outcome模型
        if 'outcome' in self.training_history:
            train_losses = self.training_history['outcome']['train_loss']
            val_losses = self.training_history['outcome']['val_loss']
            
            if train_losses and val_losses:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                overfit_ratio = final_val / (final_train + 1e-8)
                
                report += f"Outcome Model:\n"
                report += f"  Final Train Loss: {final_train:.4f}\n"
                report += f"  Final Val Loss: {final_val:.4f}\n"
                report += f"  Overfit Ratio: {overfit_ratio:.2f}\n"
                report += f"  Status: {'⚠ Overfitting' if overfit_ratio > 1.5 else '✓ Good'}\n\n"
        
        # 总体建议
        report += "Recommendations:\n"
        if any('dynamics' in self.training_history for _ in [1]):
            if self.training_history.get('dynamics', {}).get('train_loss', [float('inf')])[-1] > 0.1:
                report += "  - Dynamics loss high, increase epochs\n"
        
        return report
    
    def save_results(self):
        """保存所有结果"""
        # 保存训练历史
        history_file = os.path.join(self.logs_dir, 'training_history.json')
        
        # 转换为可序列化格式
        serializable_history = {}
        for model_name, model_history in self.training_history.items():
            serializable_history[model_name] = {}
            for metric_name, metric_values in model_history.items():
                serializable_history[model_name][metric_name] = [
                    float(v) for v in metric_values
                ]
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"\n✓ Results saved to {self.output_dir}")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Logs: {self.logs_dir}")
        print(f"  - Models: {self.models_dir}")
    
    def run_complete_diagnostics(self, epochs: int, rl_iterations: int):
        """运行完整的诊断流程"""
        print("\n" + "="*60)
        print("OFFLINE TRAINING DIAGNOSTICS")
        print("="*60)
        print(f"Output Directory: {self.output_dir}")
        
        # 1. 测试数据生成
        data, data_stats = self.test_data_generation(n_patients=100)
        
        # 2. 测试Dynamics模型
        dynamics_models = self.test_dynamics_model(data, epochs=50, n_ensemble=5) # 从 10 增加到 50
        
        # 3. 测试Outcome模型
        # 同样增加 outcome 模型的训练轮数
        outcome_model = self.test_outcome_model(data, epochs=30) # 从 10 增加到 30
        
        # 4. 测试RL训练
        # 给予 RL 算法更长的训练时间来探索
        q_network = self.test_rl_training(
            data, dynamics_models, outcome_model, 
            iterations=rl_iterations
        )
        
        # 5. 收敛性分析
        self.test_convergence_analysis()
        
        # 6. 保存结果
        self.save_results()
        
        print("\n" + "="*60)
        print("DIAGNOSTICS COMPLETE")
        print("="*60)
       
        return {
            'data_stats': data_stats,
            'models': {
                'dynamics': dynamics_models,
                'outcome': outcome_model,
                'q_network': q_network
            },
            'training_history': self.training_history
        }


def run_quick_test():
    """快速测试函数，验证主要修复是否生效"""
    print("\n" + "="*60)
    print("QUICK FIX VALIDATION TEST")
    print("="*60)

    # 1. 测试CQL Loss修复
    print("\n1. Testing CQL Loss Fix...")
    from training import ConservativeQLearning

    # 创建小型测试数据
    state_dim, action_dim = 10, 5
    batch_size = 32

    q_network = ConservativeQNetwork(state_dim, action_dim)
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    outcome_model = TreatmentOutcomeModel(state_dim, action_dim)

    cql_trainer = ConservativeQLearning(
        q_network, dynamics_model, outcome_model,
        learning_rate=3e-4, cql_weight=0.1
    )

    # 测试CQL loss计算
    test_states = torch.randn(batch_size, state_dim)
    test_actions = torch.randint(0, action_dim, (batch_size,))

    try:
        cql_loss = cql_trainer.compute_cql_loss(test_states, test_actions)
        print(f"  ✓ CQL Loss: {cql_loss.item():.4f} (should be positive and stable)")
        
        # 检查是否在合理范围
        if 0 < cql_loss.item() < 100:
            print("  ✓ CQL Loss in reasonable range")
        else:
            print(f"  ⚠ CQL Loss out of range: {cql_loss.item()}")
    except Exception as e:
        print(f"  ✗ CQL Loss computation failed: {e}")

    # 2. 测试目标网络软更新
    print("\n2. Testing Target Network Soft Update...")

    # 获取初始参数
    initial_target_params = []
    for param in cql_trainer.q_target.parameters():
        initial_target_params.append(param.clone())

    # 修改Q网络参数
    for param in cql_trainer.q_network.parameters():
        param.data += 0.1

    # 执行软更新
    cql_trainer._soft_update_target()

    # 检查是否更新
    updated = False
    for initial, current in zip(initial_target_params, cql_trainer.q_target.parameters()):
        if not torch.allclose(initial, current):
            updated = True
            break

    if updated:
        print("  ✓ Target network soft update working")
    else:
        print("  ✗ Target network not updating")

    # 3. 测试Dynamics模型的masked loss
    print("\n3. Testing Dynamics Model Masked Loss...")

    from training import DigitalTwinTrainer
    trainer = DigitalTwinTrainer(dynamics_model)

    # 创建测试序列
    test_sequences = []
    for i in range(5):
        seq_len = np.random.randint(2, 10)
        test_sequences.append({
            'states': torch.randn(seq_len, state_dim),
            'actions': torch.randint(0, action_dim, (seq_len,)),
            'length': seq_len
        })

    try:
        loss = trainer.train_epoch(test_sequences)
        print(f"  ✓ Dynamics training loss: {loss:.4f}")
        
        if not np.isnan(loss) and not np.isinf(loss):
            print("  ✓ Loss is stable (not NaN or Inf)")
        else:
            print(f"  ✗ Unstable loss detected: {loss}")
    except Exception as e:
        print(f"  ✗ Dynamics training failed: {e}")

    # 4. 测试Replay Buffer数据类型处理
    print("\n4. Testing Replay Buffer Data Handling...")

    # 测试不同数据类型
    test_transitions = [
        {
            'state': np.random.randn(state_dim),  # numpy array
            'action': 0,  # int
            'reward': 0.5,  # float
            'next_state': torch.randn(state_dim)  # tensor
        },
        {
            'state': torch.randn(state_dim),  # tensor
            'action': torch.tensor(1),  # tensor
            'reward': torch.tensor(0.3),  # tensor
            'next_state': np.random.randn(state_dim)  # numpy array
        }
    ]

    try:
        for trans in test_transitions:
            cql_trainer.add_to_replay_buffer(trans)
        
        # 尝试采样
        batch = cql_trainer.sample_batch(batch_size=2)
        
        print(f"  ✓ Replay buffer handles mixed data types")
        print(f"    - Sampled batch shape: {batch['state'].shape}")
    except Exception as e:
        print(f"  ✗ Replay buffer failed: {e}")

    # 5. 测试学习率调度器
    print("\n5. Testing Learning Rate Schedulers...")

    from training import OutcomeModelTrainer
    outcome_trainer = OutcomeModelTrainer(outcome_model, regularization_weight=0.01)

    initial_lr = outcome_trainer.optimizer.param_groups[0]['lr']

    # 模拟几个epoch
    for i in range(5):
        # 模拟loss
        outcome_trainer.scheduler.step(metrics=0.1 * (i + 1))  # 递增的loss

    current_lr = outcome_trainer.optimizer.param_groups[0]['lr']

    if current_lr < initial_lr:
        print(f"  ✓ Learning rate scheduler working: {initial_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"  ⚠ Learning rate not decreasing: {initial_lr:.6f} -> {current_lr:.6f}")

    print("\n" + "="*60)
    print("QUICK TEST COMPLETE")
    print("="*60)


def compare_before_after():
    """比较修复前后的训练效果"""
    print("\n" + "="*60)
    print("BEFORE/AFTER COMPARISON")
    print("="*60)

    # 生成测试数据
    generator = PatientDataGenerator(n_patients=50, seed=42)
    data = generator.generate_dataset()

    state_dim = len(data['states'][0])
    action_dim = len(set(data['actions']))

    # 准备数据
    dataset = PatientTrajectoryDataset(data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    results = {}

    # 测试配置
    configs = {
        'Original': {
            'cql_weight': 1.0,
            'tau': 0.005,
            'learning_rate': 3e-4,
            'target_update_freq': 1000
        },
        'Fixed': {
            'cql_weight': 1.0,
            'tau': 0.005,
            'learning_rate': 3e-4,
            'target_update_freq': 1  # 每步软更新
        }
    }

    for config_name, config in configs.items():
        print(f"\nTesting {config_name} Configuration...")
        
        # 初始化模型
        q_network = ConservativeQNetwork(state_dim, action_dim)
        dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
        outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
        
        trainer = ConservativeQLearning(
            q_network, dynamics_model, outcome_model,
            learning_rate=config['learning_rate'],
            cql_weight=config['cql_weight'],
            target_update_freq=config['target_update_freq']
        )
        trainer.tau = config['tau']
        
        # 填充replay buffer
        for batch in dataloader:
            trainer.add_to_replay_buffer(batch)
        
        # 训练
        losses = []
        for i in range(500):
            metrics = trainer.train_step()
            if metrics:
                losses.append(metrics['total_loss'])
        
        results[config_name] = {
            'final_loss': losses[-1] if losses else float('inf'),
            'loss_std': np.std(losses[-100:]) if len(losses) > 100 else float('inf'),
            'converged': np.std(losses[-50:]) < 0.01 if len(losses) > 50 else False
        }
        
        print(f"  Final Loss: {results[config_name]['final_loss']:.4f}")
        print(f"  Loss Std (last 100): {results[config_name]['loss_std']:.4f}")
        print(f"  Converged: {results[config_name]['converged']}")
   
   # 比较结果
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    if 'Fixed' in results and 'Original' in results:
        improvement = (results['Original']['final_loss'] - results['Fixed']['final_loss']) / results['Original']['final_loss'] * 100
        stability_improvement = (results['Original']['loss_std'] - results['Fixed']['loss_std']) / results['Original']['loss_std'] * 100
        
        print(f"Loss Improvement: {improvement:.1f}%")
        print(f"Stability Improvement: {stability_improvement:.1f}%")
        
        if results['Fixed']['converged'] and not results['Original']['converged']:
            print("✓ Fixed version converges while original doesn't")
        elif results['Fixed']['converged'] and results['Original']['converged']:
            print("✓ Both versions converge")
        else:
            print("⚠ Neither version converges in 500 iterations")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Offline Training Fixes')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['quick', 'full', 'compare'],
                        help='Test mode: quick test, full diagnostics, or before/after comparison')
    parser.add_argument('--output-dir', type=str, default='./training_diagnostics',
                        help='Output directory for results')
    parser.add_argument('--n-patients', type=int, default=100,
                        help='Number of patients for data generation')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--rl-iterations', type=int, default=5000,
                        help='Number of RL training iterations')

    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║          OFFLINE TRAINING DIAGNOSTICS TOOL                ║
    ║                                                           ║
    ║  Mode: {args.mode:^47}║
    ║  Output: {args.output_dir:^45}║
    ╚═══════════════════════════════════════════════════════════╝
    """)
   
    if args.mode == 'quick':
        run_quick_test()
    elif args.mode == 'compare':
        compare_before_after()
    else:  # full
        diagnostics = OfflineTrainingDiagnostics(args.output_dir)
        results = diagnostics.run_complete_diagnostics(
            epochs=args.epochs, 
            rl_iterations=args.rl_iterations
        )
        
        # 生成最终报告
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        if 'data_stats' in results:
            print(f"✓ Data Generation: {results['data_stats']['n_transitions']} transitions")
        
        if 'models' in results:
            print(f"✓ Models Trained: {len(results['models'])} models")
        
        if 'training_history' in results:
            for model_name in results['training_history']:
                if 'train_loss' in results['training_history'][model_name]:
                    final_loss = results['training_history'][model_name]['train_loss'][-1]
                    print(f"  - {model_name}: Final loss = {final_loss:.4f}")
        
        print(f"\n📊 Check plots in: {diagnostics.plots_dir}")
        print(f"📝 Check logs in: {diagnostics.logs_dir}")
        print(f"💾 Models saved in: {diagnostics.models_dir}")


if __name__ == "__main__":
   main()