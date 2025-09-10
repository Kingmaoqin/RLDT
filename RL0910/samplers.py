"""
samplers.py - Active Learning samplers for intelligent data selection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import os 

class UncertaintySampler:
    """Uncertainty-based active learning sampler using ensemble variance"""
    
    def __init__(self, 
                 q_ensemble: 'EnsembleQNetwork',
                 tau: float = 0.05,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize uncertainty sampler
        
        Args:
            q_ensemble: Ensemble Q-network for uncertainty estimation
            tau: Threshold for querying (higher = more selective)
            device: Computing device
        """
        self.q_ensemble = q_ensemble
        self.tau = tau
        self.device = device
        self.query_history = deque(maxlen=1000)
        self.rejection_history = deque(maxlen=10000)
        print(f"UncertaintySampler initialized with tau={self.tau}")

    @torch.no_grad()
    def need_query(self, state: Union[np.ndarray, torch.Tensor]) -> bool:
        """
        Determine if we need to query expert for this state
        
        Args:
            state: Current state vector
            
        Returns:
            True if uncertainty exceeds threshold
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Ensure correct shape and device
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        
        # Get Q-values from all ensemble members
        # 检查是否使用BCQ（BCQ没有ensemble，需要特殊处理）
        bcq_path = os.path.join('./output/models', 'best_bcq_policy.d3')
        if os.path.exists(bcq_path):
            try:
                from drive_tools import BCQ_AVAILABLE
                if BCQ_AVAILABLE:
                    # 对于BCQ，使用基于状态复杂度的不确定性估计
                    state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
                    
                    # 计算状态的标准差作为不确定性代理
                    state_variance = np.var(state_np)
                    
                    # 将方差映射到[0,1]区间
                    normalized_uncertainty = min(1.0, state_variance * 10)
                    normalized_uncertainty = max(0.0, normalized_uncertainty)
                    
                    should_query = normalized_uncertainty > self.tau
                    
                    if should_query:
                        self.query_history.append({
                            'state': state_np,
                            'uncertainty': normalized_uncertainty
                        })
                    else:
                        self.rejection_history.append({
                            'state': state_np,
                            'uncertainty': normalized_uncertainty
                        })
                    
                    return should_query
            except Exception as e:
                print(f"BCQ uncertainty estimation failed: {e}")
        
        # 原有的ensemble方法（CQL情况）
        q_values_all = self.q_ensemble(state, return_all=True)  # (n_ensemble, batch_size, action_dim)
        
        # Compute variance across ensemble for each action
        q_std = torch.std(q_values_all, dim=0)    # (batch_size, action_dim)
        q_mean = torch.mean(q_values_all, dim=0)  # (batch_size, action_dim)
        
        coefficient_of_variation = q_std / (torch.abs(q_mean) + 1e-6)
        
        max_relative_uncertainty = coefficient_of_variation.max(dim=1)[0]
        
        normalized_uncertainty = torch.tanh(max_relative_uncertainty).item()

        # Decision
        should_query = normalized_uncertainty > self.tau
        
        # Record for analysis
        if should_query:
            self.query_history.append({
                'state': state.cpu().numpy(),
                'uncertainty': normalized_uncertainty # Store the normalized value
            })
        else:
            self.rejection_history.append({
                'state': state.cpu().numpy(),
                'uncertainty': normalized_uncertainty # Store the normalized value
            })
        
        return should_query
    
    def get_uncertainty(self, state: Union[np.ndarray, torch.Tensor]) -> float:
        """Get uncertainty value for a state - 符合论文公式"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        
        raw_uncertainty_value = 0.0 # Initialize for the print statement
        normalized_uncertainty = 0.0


        # BCQ特殊处理
        bcq_path = os.path.join('./output/models', 'best_bcq_policy.d3')
        if os.path.exists(bcq_path):
            try:
                from drive_tools import BCQ_AVAILABLE
                if BCQ_AVAILABLE:
                    state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
                    if state_np.ndim == 1:
                        state_variance = np.var(state_np)
                        normalized_uncertainty = min(1.0, state_variance * 10)
                        return max(0.0, normalized_uncertainty)
            except Exception as e:
                print(f"BCQ uncertainty calculation failed: {e}")

        with torch.no_grad():
            # 按照论文公式: u(st) = max_a σ{Qψk(st, a)}^K_k=1
            q_values_all = self.q_ensemble(state, return_all=True)  # (K, batch, actions)
            
            # FIX: 使用更科学的相对不确定性度量 (变异系数)
            q_std_per_action = torch.std(q_values_all, dim=0)    # (batch, actions)
            q_mean_per_action = torch.mean(q_values_all, dim=0)  # (batch, actions)
            
            # 加上一个极小值避免除以零
            coefficient_of_variation = q_std_per_action / (torch.abs(q_mean_per_action) + 1e-6)
            
            # 取最大不确定性
            max_relative_uncertainty = torch.max(coefficient_of_variation, dim=1)[0]
            
            # FIX: Capture the raw value before normalization for debugging
            raw_uncertainty_value = max_relative_uncertainty.item()

            # 使用tanh进行最终的压缩
            normalized_uncertainty = torch.tanh(max_relative_uncertainty).item()
        
        # 调试信息
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 200 == 0:
            # FIX: Use the correct variable names in the print statement
            print(f"[Debug] Raw uncertainty: {raw_uncertainty_value:.4f}, Normalized: {normalized_uncertainty:.4f}, Threshold: {self.tau}")
        
        return normalized_uncertainty
    
    def update_threshold(self, new_tau: float):
        """Update query threshold"""
        self.tau = new_tau
        print(f"Updated uncertainty threshold to {new_tau}")
    
    def get_query_stats(self) -> Dict:
        """Get statistics about querying behavior"""
        total_seen = len(self.query_history) + len(self.rejection_history)
        
        if total_seen == 0:
            query_rate = 0.0
            avg_queried_uncertainty = 0.0
            avg_rejected_uncertainty = 0.0
        else:
            query_rate = len(self.query_history) / total_seen
            avg_queried_uncertainty = np.mean([h['uncertainty'] for h in self.query_history]) if self.query_history else 0.0
            avg_rejected_uncertainty = np.mean([h['uncertainty'] for h in self.rejection_history]) if self.rejection_history else 0.0
        
        # 修复：确保总是返回所有键，特别是 current_threshold
        return {
            'query_rate': query_rate,
            'total_queries': len(self.query_history),
            'total_seen': total_seen,
            'avg_queried_uncertainty': avg_queried_uncertainty if avg_queried_uncertainty > 0 else 0.1,  # 设置默认值
            'avg_rejected_uncertainty': avg_rejected_uncertainty if avg_rejected_uncertainty > 0 else 0.05,
            'current_threshold': self.tau if self.tau > 0 else 0.05  # 确保不为0
        }


class DiversitySampler:
    """Diversity-based sampler using k-center algorithm"""
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 k: int = 10,
                 distance_metric: str = 'euclidean'):
        """
        Initialize diversity sampler
        
        Args:
            buffer_size: Size of candidate buffer
            k: Number of diverse samples to select
            distance_metric: Distance metric to use
        """
        self.buffer_size = buffer_size
        self.k = k
        self.distance_metric = distance_metric
        self.candidate_buffer = deque(maxlen=buffer_size)
        self.selected_centers = []
        
    def add_candidate(self, state: np.ndarray, uncertainty: float):
        """Add a candidate to the buffer"""
        self.candidate_buffer.append({
            'state': state,
            'uncertainty': uncertainty
        })
    
    def select_diverse_batch(self, force_k: Optional[int] = None) -> List[Dict]:
        """
        Select k diverse samples using k-center algorithm
        
        Args:
            force_k: Override default k value
            
        Returns:
            List of selected diverse samples
        """
        if len(self.candidate_buffer) == 0:
            return []
        
        k = force_k or self.k
        k = min(k, len(self.candidate_buffer))
        
        # Extract states
        states = np.array([c['state'] for c in self.candidate_buffer])
        uncertainties = np.array([c['uncertainty'] for c in self.candidate_buffer])
        
        # Initialize with highest uncertainty point
        selected_indices = [np.argmax(uncertainties)]
        selected_centers = [states[selected_indices[0]]]
        
        # Iteratively select farthest points
        for _ in range(k - 1):
            # Compute distances to nearest selected center
            min_distances = np.full(len(states), np.inf)
            
            for center in selected_centers:
                distances = np.linalg.norm(states - center, axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Weight by uncertainty (prefer high uncertainty + far from centers)
            scores = min_distances * uncertainties
            
            # Select farthest point
            next_idx = np.argmax(scores)
            selected_indices.append(next_idx)
            selected_centers.append(states[next_idx])
        
        # Return selected samples
        selected = [self.candidate_buffer[i] for i in selected_indices]
        
        # Update internal state
        self.selected_centers = selected_centers
        
        # Clear buffer after selection
        self.candidate_buffer.clear()
        
        return selected
    
    def visualize_selection(self, selected_samples: List[Dict], save_path: Optional[str] = None):
        """Visualize the diversity selection (2D projection)"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            if len(selected_samples) < 2:
                return
            
            # Project to 2D for visualization
            all_states = np.array([s['state'] for s in selected_samples])
            if all_states.shape[1] > 2:
                pca = PCA(n_components=2)
                states_2d = pca.fit_transform(all_states)
            else:
                states_2d = all_states
            
            plt.figure(figsize=(8, 6))
            plt.scatter(states_2d[:, 0], states_2d[:, 1], 
                       c=[s['uncertainty'] for s in selected_samples],
                       cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(label='Uncertainty')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Diverse Sample Selection')
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()
            
        except ImportError:
            print("Matplotlib/sklearn not available for visualization")


class HybridSampler:
    """Combines uncertainty and diversity sampling"""
    
    def __init__(self,
                 q_ensemble: 'EnsembleQNetwork',
                 tau: float = 0.05,
                 diversity_k: int = 10,
                 diversity_weight: float = 0.3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize hybrid sampler
        
        Args:
            q_ensemble: Ensemble Q-network
            tau: Uncertainty threshold
            diversity_k: Number of diverse samples to select
            diversity_weight: Weight for diversity vs uncertainty (0-1)
            device: Computing device
        """
        self.uncertainty_sampler = UncertaintySampler(q_ensemble, tau, device)
        self.diversity_sampler = DiversitySampler(k=diversity_k)
        self.diversity_weight = diversity_weight
        self.batch_candidates = []
        
    def route(self, transition: Dict) -> str:
        """
        Route a transition to appropriate queue
        
        Args:
            transition: Dict with state, action, reward, next_state
            
        Returns:
            'query': needs expert labeling
            'replay': can use for training directly
            'discard': not useful
        """
        state = transition['state']
        
        # First check uncertainty
        uncertainty = self.uncertainty_sampler.get_uncertainty(state)
        
        if uncertainty > self.uncertainty_sampler.tau:
            # High uncertainty - add to diversity candidate pool
            self.diversity_sampler.add_candidate(state, uncertainty)
            self.batch_candidates.append(transition)
            
            # If we have enough candidates, select diverse batch
            if len(self.batch_candidates) >= self.diversity_sampler.k:
                return 'query_batch'
            else:
                return 'buffer'  # Wait for more candidates
        else:
            # Low uncertainty - use directly
            return 'replay'
    
    def get_query_batch(self) -> List[Dict]:
        """Get diverse batch of high-uncertainty samples"""
        selected = self.diversity_sampler.select_diverse_batch()
        
        # Map back to full transitions
        batch = []
        for sel in selected:
            # Find matching transition
            for trans in self.batch_candidates:
                if np.allclose(trans['state'], sel['state']):
                    batch.append(trans)
                    break
        
        # Clear candidates
        self.batch_candidates.clear()
        
        return batch
    def get_query_stats(self) -> Dict:
        """
        Pass through statistics from the internal uncertainty sampler.
        """
        return self.uncertainty_sampler.get_query_stats()

    def update_parameters(self, tau: Optional[float] = None, 
                          diversity_weight: Optional[float] = None):
        """Update sampler parameters"""
        if tau is not None:
            self.uncertainty_sampler.update_threshold(tau)
        if diversity_weight is not None:
            self.diversity_weight = diversity_weight


class StreamActiveLearner:
    """Main active learning coordinator for streaming data"""
    
    def __init__(self,
                 q_ensemble: 'EnsembleQNetwork',
                 sampler_type: str = 'hybrid',
                 tau: float = 0.05,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize stream active learner
        
        Args:
            q_ensemble: Ensemble Q-network
            sampler_type: 'uncertainty', 'diversity', or 'hybrid'
            tau: Uncertainty threshold
            device: Computing device
        """
        self.q_ensemble = q_ensemble
        self.device = device
        
        # Initialize appropriate sampler
        if sampler_type == 'uncertainty':
            self.sampler = UncertaintySampler(q_ensemble, tau, device)
        elif sampler_type == 'hybrid':
            self.sampler = HybridSampler(q_ensemble, tau, device=device)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Queues
        self.query_queue = deque(maxlen=1000)
        self.weak_label_pool = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            'total_seen': 0,
            'total_queried': 0,
            'total_weak': 0,
            'total_discarded': 0
        }
    
    def process_transition(self, transition: Dict) -> Tuple[str, Optional[List[Dict]]]:
        """
        Process incoming transition
        
        Args:
            transition: State transition dict
            
        Returns:
            (decision, batch) where:
                decision: 'query', 'weak', 'query_batch'
                batch: List of transitions if query_batch, else None
        """
        self.stats['total_seen'] += 1
        
        if hasattr(self.sampler, 'route'):
            # Hybrid sampler
            decision = self.sampler.route(transition)
            
            if decision == 'query_batch':
                batch = self.sampler.get_query_batch()
                self.stats['total_queried'] += len(batch)
                return 'query_batch', batch
            elif decision == 'buffer':
                return 'buffer', None
            elif decision == 'replay':  # 这里应该是 'replay' 而不是 'weak'
                self.weak_label_pool.append(transition)
                self.stats['total_weak'] += 1
                return 'weak', None
            # 添加单个查询的情况
            elif decision == 'query':
                self.query_queue.append(transition)
                self.stats['total_queried'] += 1
                return 'query', None
        else:
            # Simple uncertainty sampler
            if self.sampler.need_query(transition['state']):
                self.query_queue.append(transition)
                self.stats['total_queried'] += 1
                return 'query', None
            else:
                self.weak_label_pool.append(transition)
                self.stats['total_weak'] += 1
                return 'weak', None
    
    def get_statistics(self) -> Dict:
        """Get active learning statistics"""
        stats = self.stats.copy()
        
        if hasattr(self.sampler, 'get_query_stats'):
            sampler_stats = self.sampler.get_query_stats()
            
            # 3. 【关键修复】在更新前，移除 sampler_stats 中可能覆盖主计数的键。
            #    因为 sampler_stats 中的计数值是基于有限历史的，不准确。
            sampler_stats.pop('total_queries', None)
            sampler_stats.pop('total_seen', None)
            
            # 4. 用剩下的补充信息（如平均不确定度）更新主统计数据。
            stats.update(sampler_stats)
        
        # 5. 基于准确的计数值，重新计算比率。
        if stats['total_seen'] > 0:
            stats['query_rate'] = stats.get('total_queried', 0) / stats['total_seen']
            stats['weak_rate'] = stats.get('total_weak', 0) / stats['total_seen']
        else:
            stats['query_rate'] = 0.0
            stats['weak_rate'] = 0.0
            
        return stats
    
    def update_threshold(self, new_tau: float):
        """Update uncertainty threshold"""
        if hasattr(self.sampler, 'update_threshold'):
            self.sampler.update_threshold(new_tau)
        elif hasattr(self.sampler, 'update_parameters'):
            self.sampler.update_parameters(tau=new_tau)


if __name__ == "__main__":
    # Test samplers
    print("Testing Active Learning Samplers...")
    
    # Mock ensemble
    class MockEnsemble:
        def __init__(self, n_ensemble=5):
            self.n_ensemble = n_ensemble
            
        def __call__(self, state, return_all=False):
            batch_size = state.shape[0]
            action_dim = 5
            
            # Generate mock Q-values with varying uncertainty
            base_q = torch.randn(batch_size, action_dim)
            
            if return_all:
                # Add noise for each ensemble member
                q_all = []
                for _ in range(self.n_ensemble):
                    noise = torch.randn_like(base_q) * 0.5
                    q_all.append(base_q + noise)
                return torch.stack(q_all)
            else:
                return base_q
    
    # Test uncertainty sampler
    ensemble = MockEnsemble()
    sampler = UncertaintySampler(ensemble, tau=0.5)
    
    # Test states
    test_states = [
        np.random.randn(10),  # Random state
        np.ones(10) * 0.5,    # Average state
        np.zeros(10),         # Edge state
    ]
    
    for i, state in enumerate(test_states):
        needs_query = sampler.need_query(state)
        uncertainty = sampler.get_uncertainty(state)
        print(f"State {i}: uncertainty={uncertainty:.3f}, query={needs_query}")
    
    print("\nQuery stats:", sampler.get_query_stats())
    
    # Test diversity sampler
    div_sampler = DiversitySampler(k=3)
    
    # Add candidates
    for _ in range(10):
        state = np.random.randn(10)
        uncertainty = np.random.rand()
        div_sampler.add_candidate(state, uncertainty)
    
    # Select diverse batch
    selected = div_sampler.select_diverse_batch()
    print(f"\nSelected {len(selected)} diverse samples")
    
    print("\nSamplers tested successfully!")