"""
online_monitor.py - 实时监控在线学习系统
"""

import time
import threading
from collections import deque
from typing import Dict, Optional


class OnlineSystemMonitor:
    """监控在线学习系统的各个组件"""
    
    def __init__(self, system: Dict, update_interval: float = 1.0):
        self.system = system
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 监控指标
        self.metrics = {
            'transitions_processed': deque(maxlen=100),
            'queries_made': deque(maxlen=100),
            'labels_received': deque(maxlen=100),
            'training_steps': deque(maxlen=100),
            'buffer_sizes': deque(maxlen=100),
            'performance': deque(maxlen=100)
        }
        
        self.last_stats = {}
        
    def start(self):
        """开始监控"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("System monitoring started")
        
    def stop(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("System monitoring stopped")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            # 收集当前统计
            trainer_stats = self.system['trainer'].get_statistics()
            al_stats = self.system['active_learner'].get_statistics()
            
            # 计算增量
            transitions_delta = trainer_stats.get('total_transitions', 0) - self.last_stats.get('transitions', 0)
            queries_delta = al_stats.get('total_queries', 0) - self.last_stats.get('queries', 0)
            updates_delta = trainer_stats.get('total_updates', 0) - self.last_stats.get('updates', 0)
            
            # 记录指标
            self.metrics['transitions_processed'].append(transitions_delta)
            self.metrics['queries_made'].append(queries_delta)
            self.metrics['training_steps'].append(updates_delta)
            self.metrics['buffer_sizes'].append({
                'labeled': trainer_stats.get('labeled_buffer_size', 0),
                'weak': trainer_stats.get('weak_buffer_size', 0),
                'query': trainer_stats.get('query_buffer_size', 0)
            })
            
            # 更新last_stats
            self.last_stats = {
                'transitions': trainer_stats.get('total_transitions', 0),
                'queries': al_stats.get('total_queries', 0),
                'updates': trainer_stats.get('total_updates', 0)
            }
            
            # 打印实时状态
            if transitions_delta > 0 or updates_delta > 0:
                print(f"\r[Monitor] Trans: +{transitions_delta} | Queries: +{queries_delta} | "
                      f"Updates: +{updates_delta} | Buffers: L={trainer_stats.get('labeled_buffer_size', 0)} "
                      f"W={trainer_stats.get('weak_buffer_size', 0)}", end='')
            
            time.sleep(self.update_interval)
    
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        return {
            'avg_transitions_per_sec': sum(self.metrics['transitions_processed']) / len(self.metrics['transitions_processed']) if self.metrics['transitions_processed'] else 0,
            'avg_queries_per_sec': sum(self.metrics['queries_made']) / len(self.metrics['queries_made']) if self.metrics['queries_made'] else 0,
            'total_training_steps': sum(self.metrics['training_steps']),
            'current_buffers': self.metrics['buffer_sizes'][-1] if self.metrics['buffer_sizes'] else {}
        }