"""
train_and_evaluate.py - 完整的训练和评估流程
"""
import os
import torch
import torch.nn as nn
import traceback
import numpy as np
from data import PatientDataGenerator
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork
from training import (
    DigitalTwinTrainer,
    OutcomeModelTrainer,
    ConservativeQLearning,
    PatientTrajectoryDataset,
    train_outcome_model,
    train_rl_policy
)
from torch.utils.data import DataLoader
from evaluation import run_evaluation
import json

def train_dynamics_ensemble(data, state_dim, action_dim, n_ensemble=5, epochs=50, save_dir='.'):
    """训练动力学模型集成 - 参照test_offline_training.py的实现"""
    print(f"\n2. Training Ensemble of {n_ensemble} Dynamics Models...")
    
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
        
        # 确保每次初始化都正确（参照test_offline_training.py）
        nn.init.xavier_uniform_(model.output_projection[-1].weight)
        nn.init.zeros_(model.output_projection[-1].bias)
        
        trainer = DigitalTwinTrainer(model)
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_sequences)
            
            if (epoch + 1) % 10 == 0:  # 每10轮打印一次日志
                val_metrics = trainer.evaluate(val_sequences)
                print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val MSE: {val_metrics['mse']:.4f}")
                
                # 保存最佳模型
                if val_metrics['mse'] < best_val_loss:
                    best_val_loss = val_metrics['mse']
                    save_path = os.path.join(save_dir, f'dynamics_model_{i}.pth')
                    torch.save(model.state_dict(), save_path)
        
        model.eval()  # 设为评估模式
        ensemble_models.append(model)
        print(f"  ✓ Model {i+1} trained.")
    
    print(f"✓ Dynamics model ensemble trained successfully.")
    return ensemble_models

def main():
    try:
        # 1. 训练阶段
        print("="*60)
        print("PHASE 1: TRAINING")
        print("="*60)
        
        # 生成训练数据
        print("\n1. Generating training data...")
        generator = PatientDataGenerator(n_patients=100, seed=42)
        train_data = generator.generate_dataset()
        print(f"   Generated {len(train_data['states'])} transitions")
        
        state_dim = 10
        action_dim = 5
        model_dir = './output/models'
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练 Dynamics Model 集成（5个独立模型）
        try:
            dynamics_models = train_dynamics_ensemble(
                train_data, 
                state_dim, 
                action_dim,
                n_ensemble=5,
                epochs=50,
                save_dir=model_dir
            )
            print("   ✓ Dynamics ensemble training complete")
        except Exception as e:
            print(f"   ✗ Dynamics ensemble training failed: {e}")
            traceback.print_exc()
            return
        
        # 训练 Outcome Model
        print("\n3. Training Outcome Model...")
        try:
            outcome_model = train_outcome_model(
                train_data, 
                state_dim, 
                action_dim,
                n_epochs=30,
                batch_size=256,
                save_dir=model_dir
            )
            print("   ✓ Outcome model training complete")
        except Exception as e:
            print(f"   ✗ Outcome model training failed: {e}")
            traceback.print_exc()
            return
        
        # 训练 RL Policy（使用5个独立的模型）
        print("\n4. Training RL Policy with ensemble...")
        try:
            # 初始化Q网络
            q_network = ConservativeQNetwork(state_dim, action_dim)
            
            # 创建CQL训练器，传入5个独立的dynamics模型
            trainer = ConservativeQLearning(
                q_network, 
                dynamics_models,  # 5个独立训练的模型列表
                outcome_model,
                learning_rate=3e-4,
                cql_weight=0.1
            )
            
            # 准备数据
            dataset = PatientTrajectoryDataset(train_data)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            # 填充replay buffer
            print("  Filling replay buffer...")
            all_rewards = []
            for b in dataloader:
                all_rewards.append(b['reward'].numpy())
            r = np.concatenate(all_rewards, axis=0)
            r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
            print(f"  Normalizing rewards: mean={r_mean:.3f}, std={r_std:.3f}")
            # NEW: persist reward normalization so evaluation uses the same scale
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, 'reward_stats.json'), 'w') as f:
                json.dump({'mean': r_mean, 'std': r_std}, f)
                        
            # 归一化后再入池
            for batch in dataloader:
                batch['reward'] = (batch['reward'] - r_mean) / r_std
                
                if 'timestep' in batch:
                    ts = batch['timestep']
                    if isinstance(ts, torch.Tensor):
                        batch['done'] = (ts >= 49).float()
                    else:
                        batch['done'] = (np.asarray(ts) >= 49).astype(np.float32)
                else:
                    if isinstance(batch['reward'], torch.Tensor):
                        B = batch['reward'].shape[0]
                        batch['done'] = torch.zeros(B, dtype=torch.float32)
                    else:
                        batch['done'] = np.zeros_like(batch['reward'], dtype=np.float32)
                
                trainer.add_to_replay_buffer(batch)
            
            # 训练RL策略
            print("  Starting RL training...")
            n_iterations = 10000
            for i in range(n_iterations):
                metrics = trainer.train_step()
                
                if (i + 1) % 1000 == 0 and metrics:
                    print(f"  Iteration {i+1}/{n_iterations} - "
                          f"Loss: {metrics['total_loss']:.4f}, "
                          f"Mean Q: {metrics['mean_q']:.2f}")
                    
                    # 定期评估
                    if (i + 1) % 2000 == 0:
                        eval_metrics = trainer.evaluate_policy(
                            dataset, n_episodes=20, 
                            reward_mean=r_mean, reward_std=r_std
                        )
                        print(f"    Policy Return: {eval_metrics['mean_return']:.3f}")
            
            # 保存Q网络
            save_path = os.path.join(model_dir, 'best_q_network.pth')
            torch.save(q_network.state_dict(), save_path)
            print("   ✓ RL policy training complete")
            
        except Exception as e:
            print(f"   ✗ RL policy training failed: {e}")
            traceback.print_exc()
            return
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE - All models saved to:", model_dir)
        print("="*60)
        
        # 2. 评估阶段
        print("\n" + "="*60)
        print("PHASE 2: EVALUATION")
        print("="*60)
        
        try:
            results = run_evaluation(
                model_dir=model_dir,
                output_dir='./evaluation_results'
            )
            print("\n✅ Training and evaluation complete!")
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            traceback.print_exc()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()