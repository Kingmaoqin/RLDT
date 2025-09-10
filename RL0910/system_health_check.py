"""
system_health_check.py - 系统健康检查
"""

import time
from typing import Dict, List, Tuple


class SystemHealthChecker:
    """检查在线学习系统的健康状态"""
    
    def __init__(self, system: Dict):
        self.system = system
        self.checks = []
        
    def check_data_flow(self) -> Tuple[bool, str]:
        """检查数据流是否正常"""
        trainer = self.system['trainer']
        initial_count = trainer.stats['total_transitions']
        time.sleep(2)
        final_count = trainer.stats['total_transitions']
        
        if final_count > initial_count:
            return True, f"Data flow OK: {final_count - initial_count} transitions in 2s"
        else:
            return False, "Data flow FAILED: No new transitions"
    
    def check_active_learning(self) -> Tuple[bool, str]:
        """检查主动学习是否工作"""
        al = self.system['active_learner']
        stats = al.get_statistics()
        
        if stats['total_seen'] > 0:
            query_rate = stats.get('query_rate', 0)
            if 0 < query_rate < 1:
                return True, f"Active learning OK: Query rate = {query_rate:.2%}"
            else:
                return False, f"Active learning issue: Query rate = {query_rate:.2%}"
        else:
            return False, "Active learning not started"
    
    def check_training(self) -> Tuple[bool, str]:
        """检查训练是否发生"""
        trainer = self.system['trainer']
        initial_updates = trainer.stats['total_updates']
        
        # 等待足够时间让标注完成
        time.sleep(5)
        
        final_updates = trainer.stats['total_updates']
        
        if final_updates > initial_updates:
            return True, f"Training OK: {final_updates - initial_updates} updates"
        else:
            buffer_size = len(trainer.labeled_buffer)
            return False, f"Training not occurring. Buffer size: {buffer_size}"
    
    def check_expert_labeling(self) -> Tuple[bool, str]:
        """检查专家标注系统"""
        expert = self.system['expert']
        
        if expert.is_running:
            queue_size = expert.label_queue.qsize()
            return True, f"Expert system OK: Queue size = {queue_size}"
        else:
            return False, "Expert system not running"
    
    def run_all_checks(self) -> Dict:
        """运行所有健康检查"""
        print("\n=== System Health Check ===")
        
        checks = [
            ("Data Flow", self.check_data_flow),
            ("Active Learning", self.check_active_learning),
            ("Expert Labeling", self.check_expert_labeling),
            ("Training System", self.check_training),
        ]
        
        results = {}
        all_passed = True
        
        for name, check_func in checks:
            passed, message = check_func()
            results[name] = {'passed': passed, 'message': message}
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{name}: {status} - {message}")
            
            if not passed:
                all_passed = False
        
        print(f"\nOverall: {'✓ All checks passed' if all_passed else '✗ Some checks failed'}")
        
        return results