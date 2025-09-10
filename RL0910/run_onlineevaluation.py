import torch
import numpy as np
import os
import time # 导入 time 模块

# ... (其他 import 保持不变)
from online_evaluation import create_online_evaluation_pipeline
import drive_tools
from inference import DigitalTwinInference, ClinicalDecisionSupport
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork
from collections import deque
from comprehensive_evaluation import run_comprehensive_evaluation
from online_experiments import run_complete_online_evaluation
from online_monitor import OnlineSystemMonitor
from system_health_check import SystemHealthChecker
# ... (MODEL_PATHS 和 setup_system 函数保持不变)
MODEL_PATHS = {
    "dynamics_model": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_dynamics_model.pth",
    "outcome_model": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_outcome_model.pth",
    "q_network": "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_q_network.pth"
}
def test_expert_labeling():
    """快速测试专家标注是否工作"""
    print("\nTesting expert labeling system...")
    
    # 创建简单的测试
    from online_loop import ExpertSimulator
    expert = ExpertSimulator(label_delay=0.1)
    
    test_transition = {
        'state': np.random.rand(10),
        'action': 0,
        'reward': 0.5,
        'next_state': np.random.rand(10)
    }
    
    label_received = [False]
    
    def callback(labeled):
        print(f"Label received! Original reward: {test_transition['reward']}, "
              f"Expert reward: {labeled['reward']}")
        label_received[0] = True
    
    expert.request_label(test_transition, callback)
    
    # 等待标注完成
    time.sleep(0.5)
    
    if label_received[0]:
        print("✓ Expert labeling system working correctly")
    else:
        print("✗ Expert labeling system NOT working")
        
    expert.stop()

def setup_system():
    # ... (此函数内容完全不变)
    print("Setting up the evaluation environment...")
    state_dim = 10
    action_dim = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
    q_network = ConservativeQNetwork(state_dim, action_dim)
    print("Loading pre-trained models...")
    for model_name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Please check the path.")
    dynamics_model.load_state_dict(torch.load(MODEL_PATHS["dynamics_model"], map_location=device))
    state_dict = torch.load(MODEL_PATHS["outcome_model"], map_location=device)
    keys_to_remove = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    for k in keys_to_remove:
        del state_dict[k]
    outcome_model.load_state_dict(state_dict, strict=False)
    state_dict = torch.load(MODEL_PATHS["q_network"], map_location=device)
    keys_to_remove = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
    for k in keys_to_remove:
        del state_dict[k]
    q_network.load_state_dict(state_dict, strict=False)
    print("Models loaded successfully.")
    inference_engine = DigitalTwinInference(dynamics_model, outcome_model, q_network, state_dim, action_dim, device)
    cds = ClinicalDecisionSupport(inference_engine)
    drive_tools.initialize_tools(inference_engine, cds)
    print("System initialized successfully.")

drive_tools.update_hyperparams({"tau": 0.2, "batch_size": 8})

def run_evaluation():
    """
    运行评估流程
    """
    # 准备测试数据
    print("\nPreparing test data...")
    test_data = {
        'states': np.random.randn(100, 10),
        'actions': np.random.randint(0, 5, 100)
    }

    if drive_tools._inference_engine is None or drive_tools._online_system is None:
        raise RuntimeError("System is not initialized. Call setup_system() first.")

    # 创建评估管道
    print("Creating evaluation pipeline...")
    evaluation_pipeline = create_online_evaluation_pipeline(
        models={
            'dynamics': drive_tools._inference_engine.dynamics_model,
            'outcome': drive_tools._inference_engine.outcome_model,
            'q_network': drive_tools._inference_engine.q_network
        },
        test_data=test_data
    )

    # 收集系统状态 - 这次我们全部使用实时数据
    print("Collecting REAL-TIME system statistics...")
    trainer_stats = drive_tools._online_system['trainer'].get_statistics()
    active_learner_stats = drive_tools._online_system['active_learner'].get_statistics()

    # 从 trainer 的统计中提取真实的历史数据（如果存在）
    # 注意: online_loop.py 中没有内置 history, 这里的 history 会是空的，除非你添加了相关逻辑
    # 我们将直接使用当前的统计数据来生成报告
    system_stats = {
        'active_learning': active_learner_stats,
        'trainer': trainer_stats,
        'model': drive_tools._inference_engine.q_network,
        # 这里只是示例，实际应该从评估中获取
        'current_performance': np.mean(list(trainer_stats.get('training_times', [0]))) if trainer_stats.get('training_times') else 0,
        'metrics_history': { # 绘图数据也应来自真实系统，但需要修改代码来收集历史
             'query_rates': [active_learner_stats.get('query_rate', 0)], # 仅使用当前值
             'performance': [],
             'uncertainties': [],
             'learning_times': list(trainer_stats.get('training_times', []))
        },
        'response_times': [] # 同样需要真实数据
    }
    print("\nRunning system health check...")
    health_checker = SystemHealthChecker(drive_tools._online_system)
    health_results = health_checker.run_all_checks()
    
    # 如果有检查失败，给出警告
    failed_checks = [name for name, result in health_results.items() if not result['passed']]
    if failed_checks:
        print(f"\nWARNING: The following checks failed: {', '.join(failed_checks)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    # 运行评估
    print("Running evaluation...")
    results = evaluation_pipeline['evaluate'](system_stats)
    
    # 打印和保存报告
    print("\n--- Evaluation Report ---")
    print(results['online_report'])
    print("--------------------------\n")

    report_path = 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(results['online_report'])
    print(f"Evaluation report saved to: {report_path}")
    print("Metrics plot saved to: online_metrics.png (Note: plot may be sparse with short run time)")


if __name__ == "__main__":
    # 1. 首先设置系统
    test_expert_labeling()     
    setup_system()
    print("\nStarting system monitor...")
    monitor = OnlineSystemMonitor(drive_tools._online_system)
    monitor.start()
    
    # 修改等待循环
    print(f"\nOnline system is running. Waiting for {run_duration_seconds} seconds to collect statistics...")
    
    try:
        for i in range(run_duration_seconds):
            time.sleep(1)
            
            if i % 10 == 0:
                stats = drive_tools.get_online_stats()
                summary = monitor.get_summary()
                
                print(f"\n[{i}s] System Status:")
                print(f"  Transitions: {stats.get('total_transitions', 0)}")
                print(f"  Updates: {stats.get('total_updates', 0)}")
                print(f"  Labeled Buffer: {stats.get('labeled_buffer_size', 0)}")
                print(f"  Avg Trans/sec: {summary['avg_transitions_per_sec']:.2f}")
                print(f"  Training Active: {summary['total_training_steps'] > 0}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        monitor.stop()    
    # 2. **关键改动**: 等待一段时间让系统收集数据
    run_duration_seconds = 600  # 运行600秒
    print(f"\nOnline system is running. Waiting for {run_duration_seconds} seconds to collect statistics...")
    try:
        for i in range(run_duration_seconds):
            time.sleep(1)
            if i % 10 == 0:  # 每10秒打印一次状态
                stats = drive_tools.get_online_stats()
                print(f"\n[{i}s] Transitions: {stats.get('total_transitions', 0)}, "
                    f"Updates: {stats.get('total_updates', 0)}, "
                    f"Buffer: {stats.get('labeled_buffer_size', 0)}")
            print(f"  ... {run_duration_seconds - i - 1} seconds remaining", end='\r')
    except KeyboardInterrupt:
        print("\nSkipping wait time.")

    # 3. 然后运行评估
    run_evaluation()

    # 4. 停止后台线程（好习惯）
    print("Stopping online system...")
    if drive_tools._online_system:
        drive_tools._online_system['stream'].stop_stream()
        drive_tools._online_system['trainer'].stop()
    print("System stopped.")