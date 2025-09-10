"""
online_evaluation.py - Evaluation methods for online learning system
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from sklearn.metrics import roc_auc_score, precision_recall_curve
import time


class OnlineEvaluator:
    """Comprehensive evaluation for online learning systems"""
    
    def __init__(self, 
                 window_size: int = 1000,
                 checkpoint_freq: int = 100):
        """
        Initialize online evaluator
        
        Args:
            window_size: Size of sliding window for metrics
            checkpoint_freq: Frequency of metric checkpoints
        """
        self.window_size = window_size
        self.checkpoint_freq = checkpoint_freq
        
        # Metric buffers
        self.prediction_errors = deque(maxlen=window_size)
        self.query_decisions = deque(maxlen=window_size)
        self.learning_times = deque(maxlen=window_size)
        self.uncertainty_values = deque(maxlen=window_size)
        
        # Checkpoint storage
        self.checkpoints = []
        self.transition_count = 0
        
    def evaluate_active_learning(self, 
                               sampler_stats: Dict,
                               ground_truth_labels: Optional[List] = None) -> Dict:
        """
        Evaluate active learning performance
        
        Metrics:
        - Query efficiency: % of queries that were truly uncertain
        - Coverage: Diversity of queried samples
        - Label efficiency: Performance gain per labeled sample
        """
        metrics = {}
        
        # Query rate over time
        metrics['query_rate'] = sampler_stats.get('query_rate', 0)
        metrics['total_queries'] = sampler_stats.get('total_queries', 0)
        
        # Query efficiency (if we have ground truth)
        if ground_truth_labels and 'query_history' in sampler_stats:
            # Check if queried samples were actually difficult
            query_history = sampler_stats['query_history']
            if query_history:
                uncertainties = [h['uncertainty'] for h in query_history[-100:]]
                metrics['avg_query_uncertainty'] = np.mean(uncertainties)
                metrics['query_uncertainty_std'] = np.std(uncertainties)
        
        # Threshold adaptation
        metrics['current_threshold'] = sampler_stats.get('current_threshold', 0)
        
        return metrics
    
    def evaluate_online_learning(self,
                               trainer_stats: Dict,
                               model_performance: Dict) -> Dict:
        """
        Evaluate online learning performance
        
        Metrics:
        - Learning stability: Parameter drift over time
        - Adaptation speed: How quickly model improves
        - Catastrophic forgetting: Performance on old data
        """
        metrics = {}
        
        # Training efficiency
        metrics['updates_per_second'] = trainer_stats.get('total_updates', 0) / max(
            trainer_stats.get('training_time', 1), 1
        )
        
        if 'training_times' in trainer_stats:
            times = list(trainer_stats['training_times'])
            if times:
                metrics['avg_update_time'] = np.mean(times)
                metrics['max_update_time'] = np.max(times)
        
        # Buffer utilization
        metrics['labeled_buffer_efficiency'] = trainer_stats.get('labeled_buffer_size', 0) / max(
            trainer_stats.get('total_transitions', 1), 1
        )
        
        # Model performance trends
        if 'performance_history' in model_performance:
            perf_history = model_performance['performance_history']
            if len(perf_history) > 1:
                # Calculate performance improvement rate
                recent_perf = np.mean(perf_history[-10:])
                old_perf = np.mean(perf_history[:10])
                metrics['performance_improvement'] = recent_perf - old_perf
        
        return metrics
    
    def evaluate_parameter_adaptation(self,
                                    param_history: List[Dict],
                                    performance_history: List[float]) -> Dict:
        """
        Evaluate how well parameters adapt to data
        
        Metrics:
        - Parameter stability
        - Correlation with performance
        - Adaptation efficiency
        """
        if len(param_history) < 2:
            return {}
        
        metrics = {}
        
        # Parameter drift
        param_changes = []
        for i in range(1, len(param_history)):
            changes = {}
            for key in param_history[i]:
                if key in param_history[i-1]:
                    change = abs(param_history[i][key] - param_history[i-1][key])
                    changes[key] = change
            param_changes.append(changes)
        
        # Average drift per parameter
        avg_drift = {}
        for key in param_history[-1]:
            drifts = [c.get(key, 0) for c in param_changes]
            avg_drift[key] = np.mean(drifts) if drifts else 0
        
        metrics['parameter_drift'] = avg_drift
        
        # Stability score (lower is more stable)
        metrics['stability_score'] = np.mean(list(avg_drift.values()))
        
        # Performance correlation
        if len(performance_history) == len(param_history):
            # Find which parameters correlate with performance
            correlations = {}
            for key in param_history[0]:
                param_values = [p.get(key, 0) for p in param_history]
                if np.std(param_values) > 0:
                    corr = np.corrcoef(param_values, performance_history)[0, 1]
                    correlations[key] = corr
            
            metrics['param_performance_correlation'] = correlations
        
        return metrics
    
    def evaluate_real_time_performance(self,
                                     response_times: List[float],
                                     decision_times: List[float]) -> Dict:
        """
        Evaluate real-time system performance
        
        Metrics:
        - Latency: Time to make decisions
        - Throughput: Decisions per second
        - Resource utilization
        """
        metrics = {}
        
        if response_times:
            metrics['avg_response_time'] = np.mean(response_times)
            metrics['p95_response_time'] = np.percentile(response_times, 95)
            metrics['p99_response_time'] = np.percentile(response_times, 99)
        
        if decision_times:
            metrics['avg_decision_time'] = np.mean(decision_times)
            metrics['decisions_per_second'] = 1.0 / metrics['avg_decision_time']
        
        return metrics
    
    def create_evaluation_report(self,
                               system_stats: Dict,
                               save_path: Optional[str] = None) -> str:
        """
        Create comprehensive evaluation report
        """
        report = "# Online Learning System Evaluation Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        al_metrics = {}
        ol_metrics = {}
        pa_metrics = {}
        rt_metrics = {} 
        # Active Learning Performance
        if 'active_learning' in system_stats:
            al_metrics = self.evaluate_active_learning(system_stats['active_learning'])
            report += "## Active Learning Performance\n"
            report += f"- Query Rate: {al_metrics.get('query_rate', 0):.2%}\n"
            report += f"- Total Queries: {al_metrics.get('total_queries', 0)}\n"
            report += f"- Average Query Uncertainty: {al_metrics.get('avg_query_uncertainty', 0):.4f}\n"
            report += f"- Current Threshold: {al_metrics.get('current_threshold', 0):.4f}\n\n"
        
        # Online Learning Performance
        if 'trainer' in system_stats:
            ol_metrics = self.evaluate_online_learning(
                system_stats['trainer'],
                system_stats.get('model_performance', {})
            )
            report += "## Online Learning Performance\n"
            report += f"- Updates per Second: {ol_metrics.get('updates_per_second', 0):.2f}\n"
            report += f"- Average Update Time: {ol_metrics.get('avg_update_time', 0):.4f}s\n"
            report += f"- Buffer Efficiency: {ol_metrics.get('labeled_buffer_efficiency', 0):.2%}\n"
            report += f"- Performance Improvement: {ol_metrics.get('performance_improvement', 0):.4f}\n\n"
        
        # Parameter Adaptation
        if 'param_history' in system_stats:
            pa_metrics = self.evaluate_parameter_adaptation(
                system_stats['param_history'],
                system_stats.get('performance_history', [])
            )
            report += "## Parameter Adaptation\n"
            report += f"- Stability Score: {pa_metrics.get('stability_score', 0):.4f}\n"
            report += "- Parameter Drift:\n"
            for param, drift in pa_metrics.get('parameter_drift', {}).items():
                report += f"  - {param}: {drift:.6f}\n"
            report += "\n"
        
        # Real-time Performance
        if 'response_times' in system_stats:
            rt_metrics = self.evaluate_real_time_performance(
                system_stats['response_times'],
                system_stats.get('decision_times', [])
            )
            report += "## Real-time Performance\n"
            report += f"- Average Response Time: {rt_metrics.get('avg_response_time', 0):.4f}s\n"
            report += f"- P95 Response Time: {rt_metrics.get('p95_response_time', 0):.4f}s\n"
            report += f"- Decisions per Second: {rt_metrics.get('decisions_per_second', 0):.2f}\n\n"
        
        # Recommendations
        report += "## Recommendations\n"
        
        # Query rate recommendation
        if al_metrics.get('query_rate', 0) > 0.5:
            report += "- Consider increasing uncertainty threshold τ to reduce query rate\n"
        elif al_metrics.get('query_rate', 0) < 0.05:
            report += "- Consider decreasing uncertainty threshold τ to capture more uncertain cases\n"
        
        # Stability recommendation
        if pa_metrics.get('stability_score', 0) > 0.1:
            report += "- Parameters showing high drift; consider reducing learning rate\n"
        
        # Performance recommendation
        if ol_metrics.get('avg_update_time', 0) > 0.1:
            report += "- Update time is high; consider reducing batch size or model complexity\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_online_metrics(self,
                           metrics_history: Dict[str, List],
                           save_path: Optional[str] = None):
        """Create visualization of online learning metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Query rate over time
        if 'query_rates' in metrics_history:
            axes[0, 0].plot(metrics_history['query_rates'])
            axes[0, 0].set_title('Query Rate Over Time')
            axes[0, 0].set_xlabel('Checkpoint')
            axes[0, 0].set_ylabel('Query Rate')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Performance over time
        if 'performance' in metrics_history:
            axes[0, 1].plot(metrics_history['performance'])
            axes[0, 1].set_title('Model Performance')
            axes[0, 1].set_xlabel('Checkpoint')
            axes[0, 1].set_ylabel('Performance Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Uncertainty distribution
        if 'uncertainties' in metrics_history:
            axes[1, 0].hist(metrics_history['uncertainties'][-1000:], bins=50, alpha=0.7)
            axes[1, 0].axvline(x=metrics_history.get('threshold', 0.05), 
                               color='r', linestyle='--', label='Threshold')
            axes[1, 0].set_title('Uncertainty Distribution')
            axes[1, 0].set_xlabel('Uncertainty')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Learning efficiency
        if 'learning_times' in metrics_history:
            axes[1, 1].plot(metrics_history['learning_times'])
            axes[1, 1].set_title('Learning Time per Update')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class ContinualEvaluator:
    """Evaluate continual learning aspects"""
    
    def __init__(self, test_dataset: Dict):
        """
        Initialize with a held-out test set
        
        Args:
            test_dataset: Fixed test data to measure forgetting
        """
        self.test_dataset = test_dataset
        self.performance_history = []
        
    def evaluate_catastrophic_forgetting(self,
                                       model,
                                       current_performance: float) -> Dict:
        """
        Measure how much the model forgets old knowledge
        """
        # Evaluate on original test set
        old_task_performance = self._evaluate_on_test_set(model)
        
        # Calculate forgetting
        if self.performance_history:
            initial_performance = self.performance_history[0]
            forgetting = initial_performance - old_task_performance
        else:
            forgetting = 0
        
        self.performance_history.append(old_task_performance)
        
        return {
            'current_performance': current_performance,
            'old_task_performance': old_task_performance,
            'forgetting_amount': forgetting,
            'performance_ratio': old_task_performance / max(current_performance, 1e-6)
        }
    
    def _evaluate_on_test_set(self, model) -> float:
        """Evaluate model on fixed test set"""
        # Implement based on your model type
        # This is a placeholder
        return np.random.rand()  # Replace with actual evaluation
    
    def evaluate_forward_transfer(self,
                                model,
                                new_task_data: Dict) -> Dict:
        """
        Measure how well model transfers to new tasks
        """
        # Baseline: random initialization performance
        baseline_performance = 0.5  # Placeholder
        
        # Current model performance on new task
        transfer_performance = self._evaluate_on_new_task(model, new_task_data)
        
        # Forward transfer metric
        forward_transfer = transfer_performance - baseline_performance
        
        return {
            'baseline_performance': baseline_performance,
            'transfer_performance': transfer_performance,
            'forward_transfer': forward_transfer,
            'transfer_efficiency': forward_transfer / baseline_performance
        }
    
    def _evaluate_on_new_task(self, model, task_data: Dict) -> float:
        """Evaluate on new task"""
        # Implement based on your task
        return np.random.rand()  # Replace with actual evaluation


def create_online_evaluation_pipeline(models: Dict,
                                     test_data: Dict) -> Dict:
    """
    Create complete evaluation pipeline for online system
    """
    evaluators = {
        'online': OnlineEvaluator(window_size=1000),
        'continual': ContinualEvaluator(test_data)
    }
    
    def evaluate_system(system_stats: Dict) -> Dict:
        """Run all evaluations"""
        results = {}
        
        # Online learning evaluation
        online_report = evaluators['online'].create_evaluation_report(system_stats)
        results['online_report'] = online_report
        
        # Continual learning evaluation
        if 'model' in system_stats:
            forgetting_metrics = evaluators['continual'].evaluate_catastrophic_forgetting(
                system_stats['model'],
                system_stats.get('current_performance', 0)
            )
            results['forgetting_metrics'] = forgetting_metrics
        
        # Create visualizations
        if 'metrics_history' in system_stats:
            evaluators['online'].plot_online_metrics(
                system_stats['metrics_history'],
                save_path='online_metrics.png'
            )
        
        return results
    
    return {
        'evaluators': evaluators,
        'evaluate': evaluate_system
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Online Evaluation System...")
    
    # Mock system stats
    mock_stats = {
        'active_learning': {
            'query_rate': 0.15,
            'total_queries': 150,
            'current_threshold': 0.05,
            'query_history': [{'uncertainty': 0.08} for _ in range(100)]
        },
        'trainer': {
            'total_updates': 500,
            'training_time': 600,
            'labeled_buffer_size': 1000,
            'total_transitions': 10000,
            'training_times': [0.01 + np.random.normal(0, 0.002) for _ in range(100)]
        },
        'param_history': [
            {'alpha': 1.0, 'gamma': 0.99, 'lr': 1e-3},
            {'alpha': 1.1, 'gamma': 0.99, 'lr': 8e-4},
            {'alpha': 1.2, 'gamma': 0.98, 'lr': 6e-4}
        ],
        'performance_history': [0.7, 0.75, 0.78],
        'response_times': [0.05 + np.random.normal(0, 0.01) for _ in range(100)]
    }
    
    # Create evaluator
    evaluator = OnlineEvaluator()
    
    # Generate report
    report = evaluator.create_evaluation_report(mock_stats, save_path='evaluation_report.md')
    print("\nEvaluation Report Generated:")
    print(report)
    
    # Test metrics visualization
    metrics_history = {
        'query_rates': [0.1 + 0.05 * np.sin(i/10) for i in range(100)],
        'performance': [0.7 + 0.001 * i + 0.05 * np.random.randn() for i in range(100)],
        'uncertainties': np.random.exponential(0.05, 1000),
        'learning_times': [0.01 + 0.005 * np.random.randn() for _ in range(100)]
    }
    
    evaluator.plot_online_metrics(metrics_history, save_path='test_metrics.png')
    print("\nMetrics visualization saved to test_metrics.png")