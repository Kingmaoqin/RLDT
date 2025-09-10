"""
online_experiments.py - 论文中描述的三个实验场景
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from data import PatientDataGenerator
from online_loop import create_online_training_system



class OnlineExperimentRunner:
    """运行论文中的三个实验场景"""
    
    def __init__(self, base_models: Dict, device: str = 'cuda'):
        self.base_models = base_models
        self.device = device
        self.results = {}
        
    def scenario1_steady_state_performance(self, 
                                         duration: int = 10000,
                                         stream_rate: float = 10.0) -> Dict:
        """场景1：稳态性能评估"""
        print("\n=== Scenario 1: Steady-State Performance ===")
        
        # 创建在线系统
        system = create_online_training_system(
            self.base_models,
            sampler_type='hybrid',
            tau=0.05,
            stream_rate=stream_rate
        )
        
        # 启动系统
        system['stream'].start_stream()
        
        # 收集指标
        metrics_history = {
            'clinical_performance': [],
            'query_rates': [],
            'inference_times': [],
            'training_times': []
        }
        
        # 运行实验
        checkpoint_interval = 100
        for i in range(duration // checkpoint_interval):
            time.sleep(checkpoint_interval / stream_rate)
            
            # 收集当前统计
            stats = system['trainer'].get_statistics()
            al_stats = system['active_learner'].get_statistics()
            
            # 记录指标
            metrics_history['query_rates'].append(al_stats.get('query_rate', 0))
            
            # 测试推理时间
            inference_times = []
            for _ in range(100):
                start = time.time()
                state = np.random.rand(10)
                with torch.no_grad():
                    q_values = system['trainer'].q_ensemble(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                inference_times.append(time.time() - start)
            
            metrics_history['inference_times'].extend(inference_times)
            
            print(f"Progress: {(i+1)*checkpoint_interval}/{duration} samples")
        
        # 停止系统
        system['stream'].stop_stream()
        system['trainer'].stop()
        system['expert'].stop()
        
        return {
            'metrics_history': metrics_history,
            'final_stats': stats,
            'system': system
        }
    
    def scenario2_distribution_shift(self,
                                   pre_shift_samples: int = 5000,
                                   post_shift_samples: int = 5000) -> Dict:
        """场景2：分布偏移适应"""
        print("\n=== Scenario 2: Distribution Shift Adaptation ===")
        
        # 创建两个不同的数据生成器
        pre_shift_gen = PatientDataGenerator(n_patients=1000, seed=42)
        post_shift_gen = PatientDataGenerator(n_patients=1000, seed=123)
        
        # 修改后偏移生成器以创建不同的分布
        class ShiftedDataStream:
            def __init__(self, pre_gen, post_gen, shift_point):
                self.pre_data = pre_gen.generate_dataset()
                self.post_data = post_gen.generate_dataset()
                self.shift_point = shift_point
                self.counter = 0
                
            def __call__(self):
                if self.counter < self.shift_point:
                    data = self.pre_data
                else:
                    data = self.post_data
                    
                idx = self.counter % len(data['states'])
                transition = {
                    'state': data['states'][idx],
                    'action': data['actions'][idx],
                    'reward': data['rewards'][idx],
                    'next_state': data['next_states'][idx]
                }
                
                # 后偏移：增加老年患者比例
                if self.counter >= self.shift_point:
                    transition['state'][0] = np.clip(transition['state'][0] + 0.3, 0, 1)
                
                self.counter += 1
                return transition
        
        # 创建系统
        data_stream = ShiftedDataStream(pre_shift_gen, post_shift_gen, pre_shift_samples)
        
        system = create_online_training_system(
            self.base_models,
            sampler_type='hybrid',
            tau=0.05,
            stream_rate=10.0
        )
        
        # 替换数据源
        system['stream'].data_source = data_stream
        
        # 运行实验
        system['stream'].start_stream()
        
        performance_history = []
        shift_detected = False
        
        total_samples = pre_shift_samples + post_shift_samples
        checkpoint_interval = 100
        
        for i in range(total_samples // checkpoint_interval):
            time.sleep(checkpoint_interval / 10.0)
            
            # 评估当前性能
            test_data = pre_shift_gen.generate_dataset() if i < pre_shift_samples // checkpoint_interval else post_shift_gen.generate_dataset()
            
            # 简化的性能评估
            correct = 0
            total = min(100, len(test_data['states']))
            
            for j in range(total):
                state = test_data['states'][j]
                with torch.no_grad():
                    q_values = system['trainer'].q_ensemble(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    pred_action = q_values.mean(dim=0).argmax().item()
                    
                if pred_action == test_data['actions'][j]:
                    correct += 1
                    
            performance = correct / total
            performance_history.append(performance)
            
            # 检测分布偏移
            if i == pre_shift_samples // checkpoint_interval:
                print(f"\n!!! Distribution shift at sample {i * checkpoint_interval} !!!\n")
                shift_detected = True
            
            print(f"Sample {(i+1)*checkpoint_interval}: Performance = {performance:.3f}")
        
        # 停止系统
        system['stream'].stop_stream()
        system['trainer'].stop()
        system['expert'].stop()
        
        return {
            'performance_history': performance_history,
            'shift_point': pre_shift_samples // checkpoint_interval,
            'recovery_analysis': self._analyze_recovery(performance_history, pre_shift_samples // checkpoint_interval)
        }
    
    def scenario3_active_learning_efficiency(self,
                                           tau_values: List[float] = [0.01, 0.05, 0.1, 0.15, 0.2],
                                           samples_per_tau: int = 5000) -> Dict:
        """场景3：主动学习效率分析"""
        print("\n=== Scenario 3: Active Learning Efficiency ===")
        
        results = {}
        
        for tau in tau_values:
            print(f"\nTesting tau = {tau}")
            
            # 创建系统
            system = create_online_training_system(
                self.base_models,
                sampler_type='hybrid',
                tau=tau,
                stream_rate=50.0  # 加快实验
            )
            
            # 运行实验
            system['stream'].start_stream()
            
            time.sleep(samples_per_tau / 50.0)
            
            # 收集统计
            al_stats = system['active_learner'].get_statistics()
            trainer_stats = system['trainer'].get_statistics()
            
            # 评估性能
            test_gen = PatientDataGenerator(n_patients=100)
            test_data = test_gen.generate_dataset()
            
            correct = 0
            for i in range(min(500, len(test_data['states']))):
                state = test_data['states'][i]
                with torch.no_grad():
                    q_values = system['trainer'].q_ensemble(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    pred_action = q_values.mean(dim=0).argmax().item()
                    
                if pred_action == test_data['actions'][i]:
                    correct += 1
            
            performance = correct / min(500, len(test_data['states']))
            
            results[tau] = {
                'query_rate': al_stats.get('query_rate', 0),
                'performance': performance,
                'total_queries': al_stats.get('total_queries', 0),
                'total_updates': trainer_stats.get('total_updates', 0)
            }
            
            # 停止系统
            system['stream'].stop_stream()
            system['trainer'].stop()
            system['expert'].stop()
            
            print(f"tau={tau}: Query Rate={results[tau]['query_rate']:.3f}, Performance={performance:.3f}")
        
        return results
    
    def _analyze_recovery(self, performance_history: List[float], shift_point: int) -> Dict:
        """分析分布偏移后的恢复"""
        if shift_point >= len(performance_history):
            return {}
            
        pre_shift_perf = np.mean(performance_history[max(0, shift_point-10):shift_point])
        post_shift_perf = performance_history[shift_point:]
        
        # 找到恢复点
        recovery_threshold = pre_shift_perf * 0.9
        recovery_point = None
        
        for i, perf in enumerate(post_shift_perf):
            if perf >= recovery_threshold:
                recovery_point = i
                break
                
        return {
            'pre_shift_performance': pre_shift_perf,
            'min_performance': min(post_shift_perf) if post_shift_perf else 0,
            'recovery_samples': recovery_point * 100 if recovery_point else None,
            'performance_drop': (pre_shift_perf - min(post_shift_perf)) / pre_shift_perf if post_shift_perf else 0
        }
    
    def plot_all_results(self, save_dir: str = './experiment_results'):
        """绘制所有实验结果"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 场景1: 稳态性能图表
        if 'scenario1' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            s1_data = self.results['scenario1']
            
            # 1.1 查询率随时间变化
            ax = axes[0, 0]
            query_rates = s1_data['metrics_history']['query_rates']
            ax.plot(range(len(query_rates)), query_rates, 'b-', linewidth=2)
            ax.axhline(y=0.15, color='r', linestyle='--', label='Target (15%)')
            ax.fill_between(range(len(query_rates)), query_rates, alpha=0.3)
            ax.set_xlabel('Checkpoints (×100 samples)')
            ax.set_ylabel('Query Rate')
            ax.set_title('Active Learning Query Rate Over Time')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # 1.2 推理时间直方图
            ax = axes[0, 1]
            inference_times = [t*1000 for t in s1_data['metrics_history']['inference_times']]  # 转换为ms
            ax.hist(inference_times, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(x=50, color='r', linestyle='--', linewidth=2, label='50ms threshold')
            ax.set_xlabel('Inference Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Latency Distribution')
            ax.legend()
            
            # 1.3 训练时间箱线图
            ax = axes[1, 0]
            if 'training_times' in s1_data['metrics_history']:
                training_times = s1_data['metrics_history']['training_times']
                if training_times:
                    ax.boxplot(training_times, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'),
                            medianprops=dict(color='red', linewidth=2))
                    ax.set_ylabel('Training Time (seconds)')
                    ax.set_title('Training Time Distribution')
                    ax.set_xticklabels(['Updates'])
            
            # 1.4 性能指标汇总
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"""Steady-State Performance Summary:
            
    - Average Query Rate: {np.mean(query_rates):.2%}
    - P95 Inference Time: {np.percentile(inference_times, 95):.2f}ms
    - Meets <50ms Target: {np.percentile(inference_times, 95) < 50}
    - Total Samples Processed: {s1_data['final_stats'].get('total_transitions', 0)}
    - Total Updates: {s1_data['final_stats'].get('total_updates', 0)}
    - Labeling Reduction: {(1 - np.mean(query_rates)):.1%}
            """
            ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'scenario1_steady_state.png'), dpi=300)
            plt.close()
        
        # 场景2: 分布偏移适应
        if 'scenario2' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            s2_data = self.results['scenario2']
            
            # 2.1 性能曲线与分布偏移
            ax = axes[0, 0]
            perf_history = s2_data['performance_history']
            shift_point = s2_data['shift_point']
            
            ax.plot(range(len(perf_history)), perf_history, 'b-', linewidth=2, label='Performance')
            ax.axvline(x=shift_point, color='r', linestyle='--', linewidth=2, label='Distribution Shift')
            
            # 标记恢复区域
            if s2_data['recovery_analysis'].get('recovery_samples'):
                recovery_point = shift_point + s2_data['recovery_analysis']['recovery_samples'] // 100
                ax.fill_between(range(shift_point, min(recovery_point, len(perf_history))), 
                            0, 1, alpha=0.2, color='yellow', label='Recovery Period')
            
            ax.set_xlabel('Checkpoints (×100 samples)')
            ax.set_ylabel('Performance')
            ax.set_title('Performance Under Distribution Shift')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # 2.2 性能下降与恢复分析
            ax = axes[0, 1]
            recovery_data = s2_data['recovery_analysis']
            
            metrics = ['Pre-shift\nPerformance', 'Minimum\nPerformance', 'Current\nPerformance']
            values = [
                recovery_data.get('pre_shift_performance', 0),
                recovery_data.get('min_performance', 0),
                perf_history[-1] if perf_history else 0
            ]
            
            bars = ax.bar(metrics, values, color=['green', 'red', 'blue'], alpha=0.7)
            ax.set_ylabel('Performance')
            ax.set_title('Performance Impact Analysis')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
            
            # 2.3 滑动窗口性能变化率
            ax = axes[1, 0]
            if len(perf_history) > 2:
                window_size = 5
                perf_changes = []
                for i in range(window_size, len(perf_history)):
                    window = perf_history[i-window_size:i]
                    if len(window) > 1:
                        change_rate = (window[-1] - window[0]) / window_size
                        perf_changes.append(change_rate)
                
                ax.plot(range(window_size, len(perf_history)), perf_changes, 'g-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=shift_point, color='r', linestyle='--', linewidth=2)
                ax.set_xlabel('Checkpoints')
                ax.set_ylabel('Performance Change Rate')
                ax.set_title(f'Performance Change Rate (window={window_size})')
            
            # 2.4 恢复统计
            ax = axes[1, 1]
            ax.axis('off')
            recovery_summary = f"""Distribution Shift Recovery Analysis:
            
    - Performance Drop: {recovery_data.get('performance_drop', 0):.2%}
    - Recovery Time: {recovery_data.get('recovery_samples', 'N/A')} samples
    - Pre-shift Performance: {recovery_data.get('pre_shift_performance', 0):.3f}
    - Minimum Performance: {recovery_data.get('min_performance', 0):.3f}
    - Final Performance: {perf_history[-1] if perf_history else 0:.3f}
            """
            ax.text(0.1, 0.5, recovery_summary, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'scenario2_distribution_shift.png'), dpi=300)
            plt.close()
        
        # 场景3: 主动学习效率
        if 'scenario3' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            s3_data = self.results['scenario3']
            
            # 提取数据
            tau_values = sorted(s3_data.keys())
            query_rates = [s3_data[tau]['query_rate'] for tau in tau_values]
            performances = [s3_data[tau]['performance'] for tau in tau_values]
            total_queries = [s3_data[tau]['total_queries'] for tau in tau_values]
            
            # 3.1 查询率 vs Tau
            ax = axes[0, 0]
            ax.plot(tau_values, query_rates, 'bo-', linewidth=2, markersize=8)
            ax.fill_between(tau_values, query_rates, alpha=0.3)
            ax.set_xlabel('Uncertainty Threshold (τ)')
            ax.set_ylabel('Query Rate')
            ax.set_title('Query Rate vs Uncertainty Threshold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            
            # 3.2 性能 vs Tau
            ax = axes[0, 1]
            ax.plot(tau_values, performances, 'go-', linewidth=2, markersize=8)
            ax.fill_between(tau_values, performances, alpha=0.3, color='green')
            ax.set_xlabel('Uncertainty Threshold (τ)')
            ax.set_ylabel('Performance')
            ax.set_title('Model Performance vs Uncertainty Threshold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_ylim(0, 1)
            
            # 3.3 性能-查询率权衡曲线
            ax = axes[1, 0]
            ax.plot(query_rates, performances, 'ro-', linewidth=2, markersize=8)
            
            # 添加τ值标签
            for i, tau in enumerate(tau_values):
                ax.annotate(f'τ={tau}', (query_rates[i], performances[i]),
                        textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
            
            # 添加理想点
            ax.plot([0], [1], 'g*', markersize=15, label='Ideal (0% queries, 100% performance)')
            
            ax.set_xlabel('Query Rate')
            ax.set_ylabel('Performance')
            ax.set_title('Performance vs Query Rate Trade-off')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0, 1.05)
            
            # 3.4 效率指标
            ax = axes[1, 1]
            
            # 计算效率得分 (performance per query)
            efficiency_scores = [p/(q+0.001) for p, q in zip(performances, query_rates)]
            
            bars = ax.bar(range(len(tau_values)), efficiency_scores, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(tau_values))))
            ax.set_xticks(range(len(tau_values)))
            ax.set_xticklabels([f'{tau}' for tau in tau_values])
            ax.set_xlabel('Uncertainty Threshold (τ)')
            ax.set_ylabel('Efficiency Score\n(Performance / Query Rate)')
            ax.set_title('Active Learning Efficiency')
            
            # 标记最佳τ值
            best_idx = np.argmax(efficiency_scores)
            ax.get_children()[best_idx].set_color('red')
            ax.text(best_idx, efficiency_scores[best_idx] + 0.1, 
                f'Best\nτ={tau_values[best_idx]}', ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'scenario3_active_learning.png'), dpi=300)
            plt.close()
        
        # 综合对比图
        if all(scenario in self.results for scenario in ['scenario1', 'scenario2', 'scenario3']):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 创建雷达图
            categories = ['Query\nEfficiency', 'Performance\nStability', 'Adaptation\nSpeed', 
                        'Inference\nSpeed', 'Training\nEfficiency']
            
            # 计算各项指标（归一化到0-1）
            s1_query_eff = 1 - np.mean(self.results['scenario1']['metrics_history']['query_rates'])
            s2_stability = 1 - self.results['scenario2']['recovery_analysis'].get('performance_drop', 0)
            s2_adapt_speed = 1 / (1 + self.results['scenario2']['recovery_analysis'].get('recovery_samples', 1000)/1000)
            s1_inference = 1 if np.percentile(self.results['scenario1']['metrics_history']['inference_times'], 95)*1000 < 50 else 0.5
            s1_train_eff = min(1, self.results['scenario1']['final_stats'].get('total_updates', 0) / 1000)
            
            values = [s1_query_eff, s2_stability, s2_adapt_speed, s1_inference, s1_train_eff]
            
            # 绘制雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 完成圆圈
            angles += angles[:1]
            
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values, 'b-', linewidth=2, label='DRIVE-Online')
            ax.fill(angles, values, 'b', alpha=0.25)
            ax.set_ylim(0, 1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title('DRIVE-Online System Performance Overview', size=16, pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'system_overview_radar.png'), dpi=300)
            plt.close()
        
        print(f"All plots saved to {save_dir}/")
        
    def run_all_experiments(self) -> Dict:
        """运行所有实验"""
        print("Running all DRIVE-Online experiments...")
        
        # 场景1
        self.results['scenario1'] = self.scenario1_steady_state_performance(
            duration=10000,
            stream_rate=10.0
        )

         # 场景2
        self.results['scenario2'] = self.scenario2_distribution_shift(
            pre_shift_samples=5000,
            post_shift_samples=5000
            )
       # 场景3
        self.results['scenario3'] = self.scenario3_active_learning_efficiency(
            tau_values=[0.01, 0.05, 0.1, 0.15, 0.2],
            samples_per_tau=5000
        )

        # 生成综合报告
        self.generate_final_report()

        return self.results
   
    def generate_final_report(self):
        """生成最终实验报告"""
        report = "# DRIVE-Online Experimental Results\n\n"
        
        # 场景1结果
        if 'scenario1' in self.results:
            s1 = self.results['scenario1']
            report += "## Scenario 1: Steady-State Performance\n"
            report += f"- Final Query Rate: {s1['final_stats'].get('query_rate', 0):.3f}\n"
            report += f"- Total Updates: {s1['final_stats'].get('total_updates', 0)}\n"
            report += f"- Average Inference Time: {np.mean(s1['metrics_history']['inference_times'])*1000:.2f}ms\n\n"
        
        # 场景2结果
        if 'scenario2' in self.results:
            s2 = self.results['scenario2']
            recovery = s2['recovery_analysis']
            report += "## Scenario 2: Distribution Shift Adaptation\n"
            report += f"- Performance Drop: {recovery.get('performance_drop', 0):.2%}\n"
            report += f"- Recovery Time: {recovery.get('recovery_samples', 'N/A')} samples\n\n"
        
        # 场景3结果
        if 'scenario3' in self.results:
            s3 = self.results['scenario3']
            report += "## Scenario 3: Active Learning Efficiency\n"
            report += "| Tau | Query Rate | Performance |\n"
            report += "|-----|------------|-------------|\n"
            for tau, metrics in s3.items():
                report += f"| {tau} | {metrics['query_rate']:.3f} | {metrics['performance']:.3f} |\n"
        
        # 保存报告
        with open('experiment_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + report)


# def run_complete_online_evaluation():
#     """运行完整的在线评估流程"""
#     from models import TransformerDynamicsModel, TreatmentOutcomeModel, EnsembleQNetwork
#     import torch
    
#     # 加载预训练模型
#     state_dim = 10
#     action_dim = 5
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 初始化模型
#     dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
#     outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
#     q_ensemble = EnsembleQNetwork(state_dim, action_dim)
    
#     # 加载权重
#     model_paths = {
#         "dynamics": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_dynamics_model.pth",
#         "outcome": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_outcome_model.pth",
#         "q_network": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_q_network.pth"
#     }
    
#     # 加载 dynamics model
#     dynamics_model.load_state_dict(torch.load(model_paths["dynamics"], map_location=device))
    
#     # 修复：加载 outcome model 时处理 BatchNorm 兼容性
#     outcome_state_dict = torch.load(model_paths["outcome"], map_location=device)
#     keys_to_remove = [k for k in outcome_state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
#     for k in keys_to_remove:
#         del outcome_state_dict[k]
#     outcome_model.load_state_dict(outcome_state_dict, strict=False)
    
#     # 修复：加载 q_network 时处理 BatchNorm 兼容性
#     q_state_dict = torch.load(model_paths["q_network"], map_location=device)
#     keys_to_remove = [k for k in q_state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
#     for k in keys_to_remove:
#         del q_state_dict[k]
    
#     # 为ensemble的每个网络加载权重并添加扰动
#     for i, q_net in enumerate(q_ensemble.q_networks):
#         q_net.load_state_dict(q_state_dict, strict=False)
#         # 添加小扰动以创建多样性
#         with torch.no_grad():
#             for param in q_net.parameters():
#                 param.add_(torch.randn_like(param) * 0.001)
   
#     base_models = {
#         'dynamics_model': dynamics_model,
#         'outcome_model': outcome_model,
#         'q_ensemble': q_ensemble
#     }
   
#    # 运行实验
#     runner = OnlineExperimentRunner(base_models, device)
#     results = runner.run_all_experiments()

#     return results
def run_complete_online_evaluation(quick_params=None):
    """运行完整的在线评估流程"""
    print("🚀 Starting complete online evaluation...")
    
    try:
        from run_complete_evaluation import run_enhanced_evaluation
        import drive_tools
        import numpy as np
        
        results = {}
        
        # Scenario 1: Conservative Settings
        print("\n=== Scenario 1: Conservative Settings ===")
        drive_tools.update_hyperparams({"tau": 0.8, "alpha": 1.5})
        time.sleep(2)
        result1 = run_enhanced_evaluation(duration_seconds=100)
        
        # 修复：正确计算overall_score
        if result1:
            score1 = np.mean([r['score'] for r in result1.values()])
            results['scenario1'] = {'compliance_results': result1, 'overall_score': score1}
        else:
            results['scenario1'] = {'overall_score': 0.0}
        
        # Scenario 2: Balanced Settings  
        print("\n=== Scenario 2: Balanced Settings ===")
        drive_tools.update_hyperparams({"tau": 0.5, "alpha": 1.0})
        time.sleep(2)
        result2 = run_enhanced_evaluation(duration_seconds=100)
        
        if result2:
            score2 = np.mean([r['score'] for r in result2.values()])
            results['scenario2'] = {'compliance_results': result2, 'overall_score': score2}
        else:
            results['scenario2'] = {'overall_score': 0.0}
        
        # Scenario 3: Aggressive Settings
        print("\n=== Scenario 3: Aggressive Settings ===")
        drive_tools.update_hyperparams({"tau": 0.2, "alpha": 0.5})
        time.sleep(2)
        result3 = run_enhanced_evaluation(duration_seconds=100)

        if result3:
            score3 = np.mean([r['score'] for r in result3.values()])
            results['scenario3'] = {'compliance_results': result3, 'overall_score': score3}
        else:
            results['scenario3'] = {'overall_score': 0.0}
        
        # 创建结果目录
        import os
        os.makedirs('./experiment_results', exist_ok=True)
        
        # 保存结果
        import json
        with open('./experiment_results/full_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n=== Final Results ===")
        for scenario, result in results.items():
            score = result.get('overall_score', 0)
            print(f"{scenario}: Score {score:.1%}")
        
        return results
        
    except Exception as e:
        print(f"Error in full evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'scenario1': {'overall_score': 0},
            'scenario2': {'overall_score': 0}, 
            'scenario3': {'overall_score': 0}
        }

if __name__ == "__main__":
   results = run_complete_online_evaluation()        