"""
utils.py - Utility functions and helpers

This module contains helper functions for data processing, visualization,
evaluation metrics, and other utilities.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import json
from datetime import datetime


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_treatment_effect_heterogeneity(
    q_network: torch.nn.Module,
    patient_states: np.ndarray,
    action_pairs: Tuple[int, int] = (0, 3),
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    Compute heterogeneous treatment effects between two actions
    
    Args:
        q_network: Trained Q-network
        patient_states: Array of patient states
        action_pairs: Tuple of (treatment, control) actions
        device: Computing device
        
    Returns:
        Array of treatment effects
    """
    q_network.eval()
    states_tensor = torch.FloatTensor(patient_states).to(device)
    
    with torch.no_grad():
        q_values = q_network(states_tensor)
        treatment_values = q_values[:, action_pairs[0]]
        control_values = q_values[:, action_pairs[1]]
        treatment_effects = (treatment_values - control_values).cpu().numpy()
    
    return treatment_effects


def visualize_treatment_effects(
    treatment_effects: np.ndarray,
    patient_features: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """Visualize heterogeneous treatment effects"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot treatment effects vs key features
    names = feature_names or [f"f{i}" for i in range(patient_features.shape[1])]
    top = min(6, len(names))
    for i, (ax, feature_name) in enumerate(zip(axes[:top], names[:top])):
        ax.scatter(patient_features[:, i], treatment_effects, alpha=0.5, s=20)
        # 趋势线
        z = np.polyfit(patient_features[:, i], treatment_effects, 1)
        p = np.poly1d(z)
        x_line = np.linspace(patient_features[:, i].min(), patient_features[:, i].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Treatment Effect')
        ax.set_title(f'Treatment Effect vs {feature_name}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Heterogeneous Treatment Effects Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def evaluate_calibration(
    predicted_outcomes: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate calibration of outcome predictions
    
    Args:
        predicted_outcomes: Predicted outcomes
        actual_outcomes: Actual observed outcomes
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    # Create bins
    bin_edges = np.percentile(predicted_outcomes, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(predicted_outcomes, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute calibration statistics
    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted = predicted_outcomes[mask].mean()
            mean_actual = actual_outcomes[mask].mean()
            calibration_data.append({
                'bin': i,
                'mean_predicted': mean_predicted,
                'mean_actual': mean_actual,
                'count': mask.sum()
            })
    
    calibration_df = pd.DataFrame(calibration_data)
    
    # Expected Calibration Error (ECE)
    ece = 0.0
    for _, row in calibration_df.iterrows():
        weight = row['count'] / len(predicted_outcomes)
        ece += weight * abs(row['mean_predicted'] - row['mean_actual'])
    
    # Maximum Calibration Error (MCE)
    mce = (calibration_df['mean_predicted'] - calibration_df['mean_actual']).abs().max()
    
    return {
        'ece': ece,
        'mce': mce,
        'calibration_data': calibration_df
    }


def plot_calibration_curve(
    calibration_data: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot calibration curve"""
    
    plt.figure(figsize=(8, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Actual calibration
    plt.scatter(calibration_data['mean_predicted'], 
               calibration_data['mean_actual'],
               s=calibration_data['count'] * 5,  # Size by count
               alpha=0.7,
               label='Model Calibration')
    
    # Connect points
    plt.plot(calibration_data['mean_predicted'], 
            calibration_data['mean_actual'],
            'b-', alpha=0.5)
    
    plt.xlabel('Mean Predicted Outcome', fontsize=12)
    plt.ylabel('Mean Actual Outcome', fontsize=12)
    plt.title('Calibration Plot', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text with ECE
    plt.text(0.05, 0.95, f"ECE: {calibration_data['ece'].iloc[0]:.3f}",
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_policy_value_bounds(
    q_network: torch.nn.Module,
    dataset: Dict[str, List],
    gamma: float = 0.99,
    n_bootstrap: int = 100
) -> Dict[str, float]:
    """
    Compute confidence bounds on policy value using bootstrap
    
    Args:
        q_network: Trained Q-network
        dataset: Patient trajectory data
        gamma: Discount factor
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with value estimates and confidence bounds
    """
    q_network.eval()
    device = next(q_network.parameters()).device
    
    states = torch.FloatTensor(np.array(dataset['states'])).to(device)
    
    # Compute Q-values for all states
    with torch.no_grad():
        q_values = q_network(states)
        values = q_values.max(dim=1)[0].cpu().numpy()
    
    # Bootstrap for confidence intervals
    bootstrap_values = []
    n_samples = len(values)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_mean = values[indices].mean()
        bootstrap_values.append(bootstrap_mean)
    
    bootstrap_values = np.array(bootstrap_values)
    
    return {
        'mean_value': values.mean(),
        'std_value': values.std(),
        'bootstrap_mean': bootstrap_values.mean(),
        'bootstrap_std': bootstrap_values.std(),
        'ci_lower': np.percentile(bootstrap_values, 2.5),
        'ci_upper': np.percentile(bootstrap_values, 97.5)
    }


def analyze_feature_importance(
    model: torch.nn.Module,
    dataset: Dict[str, List],
    feature_names: List[str],
    n_samples: int = 1000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Analyze feature importance using permutation
    
    Args:
        model: Trained model (Q-network or outcome model)
        dataset: Patient data
        feature_names: Names of features
        n_samples: Number of samples to use
        device: Computing device
        
    Returns:
        DataFrame with feature importance scores
    """
    model.eval()
    
    # Sample data
    indices = np.random.choice(len(dataset['states']), 
                              min(n_samples, len(dataset['states'])), 
                              replace=False)
    
    states = torch.FloatTensor(np.array(dataset['states'])[indices]).to(device)
    
    # Baseline predictions
    with torch.no_grad():
        if hasattr(model, 'forward'):
            baseline_output = model(states).cpu().numpy()
        else:
            # For outcome model that needs actions
            actions = torch.LongTensor(np.array(dataset['actions'])[indices]).to(device)
            baseline_output = model(states, actions).cpu().numpy()
    
    # Permutation importance
    importance_scores = []
    
    for feature_idx, feature_name in enumerate(feature_names):
        permuted_states = states.clone()
        
        # Permute single feature
        perm_indices = torch.randperm(states.shape[0])
        permuted_states[:, feature_idx] = states[perm_indices, feature_idx]
        
        # Get predictions with permuted feature
        with torch.no_grad():
            if hasattr(model, 'forward'):
                permuted_output = model(permuted_states).cpu().numpy()
            else:
                permuted_output = model(permuted_states, actions).cpu().numpy()
        
        # Compute importance as change in output
        if len(baseline_output.shape) > 1:
            importance = np.abs(baseline_output - permuted_output).mean()
        else:
            importance = np.abs(baseline_output - permuted_output).mean()
        
        importance_scores.append({
            'feature': feature_name,
            'importance': importance,
            'relative_importance': importance  # Will normalize later
        })
    
    # Create DataFrame and normalize
    importance_df = pd.DataFrame(importance_scores)
    total_importance = importance_df['importance'].sum()
    importance_df['relative_importance'] = importance_df['importance'] / total_importance
    
    return importance_df.sort_values('importance', ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 10,
    save_path: Optional[str] = None
):
    """Plot feature importance"""
    
    plt.figure(figsize=(10, 6))
    
    # Select top features
    top_features = importance_df.head(top_k)
    
    # Create bar plot
    bars = plt.bar(range(len(top_features)), 
                   top_features['relative_importance'],
                   color='steelblue')
    
    # Customize plot
    plt.xticks(range(len(top_features)), 
              top_features['feature'], 
              rotation=45, ha='right')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Relative Importance', fontsize=12)
    plt.title(f'Top {top_k} Feature Importances', fontsize=14)
    
    # Add value labels on bars
    for bar, importance in zip(bars, top_features['relative_importance']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{importance:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_summary_statistics(
    dataset: Dict[str, List],
    feature_names: List[str]
) -> pd.DataFrame:
    """Create summary statistics for the dataset"""
    
    states = np.array(dataset['states'])
    actions = np.array(dataset['actions'])
    rewards = np.array(dataset['rewards'])
    
    # Feature statistics
    feature_stats = []
    for i, feature_name in enumerate(feature_names):
        feature_data = states[:, i]
        feature_stats.append({
            'Feature': feature_name,
            'Mean': feature_data.mean(),
            'Std': feature_data.std(),
            'Min': feature_data.min(),
            'Q1': np.percentile(feature_data, 25),
            'Median': np.percentile(feature_data, 50),
            'Q3': np.percentile(feature_data, 75),
            'Max': feature_data.max()
        })
    
    feature_df = pd.DataFrame(feature_stats)
    
    # Action distribution
    action_counts = pd.Series(actions).value_counts().sort_index()
    
    # Reward statistics
    reward_stats = {
        'Mean Reward': rewards.mean(),
        'Std Reward': rewards.std(),
        'Min Reward': rewards.min(),
        'Max Reward': rewards.max()
    }
    
    return feature_df, action_counts, reward_stats


def visualize_state_distributions(
    dataset: Dict[str, List],
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """Visualize distributions of state features"""
    
    states = np.array(dataset['states'])
    n_features = min(len(feature_names), states.shape[1])
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Plot histogram
        ax.hist(states[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(states[:, i].mean(), color='red', linestyle='--', 
                  label=f'Mean: {states[:, i].mean():.3f}')
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('State Feature Distributions', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compute_overlap_statistics(
    dataset: Dict[str, List],
    action_dim: int
) -> Dict[str, float]:
    """
    Compute overlap statistics for actions in similar states
    
    This helps assess positivity assumption violations
    """
    states = np.array(dataset['states'])
    actions = np.array(dataset['actions'])
    
    # Compute pairwise distances between states
    n_samples = min(1000, len(states))  # Subsample for efficiency
    sample_indices = np.random.choice(len(states), n_samples, replace=False)
    
    sampled_states = states[sample_indices]
    sampled_actions = actions[sample_indices]
    
    # Find nearest neighbors for each state
    overlap_scores = []
    
    for i in range(n_samples):
        # Compute distances to other states
        distances = np.linalg.norm(sampled_states - sampled_states[i], axis=1)
        
        # Find k nearest neighbors (excluding self)
        k = min(10, n_samples - 1)
        nearest_indices = np.argsort(distances)[1:k+1]
        
        # Check action diversity in neighborhood
        neighbor_actions = sampled_actions[nearest_indices]
        unique_actions = len(np.unique(neighbor_actions))
        
        overlap_scores.append(unique_actions / action_dim)
    
    return {
        'mean_overlap': np.mean(overlap_scores),
        'std_overlap': np.std(overlap_scores),
        'min_overlap': np.min(overlap_scores),
        'max_overlap': np.max(overlap_scores)
    }


def save_experiment_config(
    config: Dict,
    save_path: str = 'experiment_config.json'
):
    """Save experiment configuration"""
    config_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)


def load_experiment_config(load_path: str = 'experiment_config.json') -> Dict:
    """Load experiment configuration"""
    with open(load_path, 'r') as f:
        config_with_metadata = json.load(f)
    return config_with_metadata['config']


def create_model_card(
    model_performance: Dict,
    dataset_stats: Dict,
    save_path: str = 'model_card.md'
):
    """Create a model card documenting the digital twin system"""
    
    card_content = f"""# Digital Twin Model Card

## Model Details
- **Model Type**: Reinforcement Learning-based Digital Twin for Treatment Optimization
- **Components**: 
  - Transformer-based Dynamics Model
  - Treatment Outcome Model with Deconfounding
  - Conservative Q-Learning Policy
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Intended Use
- **Primary Use**: Clinical decision support for treatment selection
- **Users**: Healthcare providers and researchers
- **Out-of-Scope**: Direct patient use without clinical supervision

## Performance Metrics
- **Dynamics Model MSE**: {model_performance.get('dynamics_mse', 'N/A')}
- **Outcome Model MSE**: {model_performance.get('outcome_mse', 'N/A')}
- **Policy Mean Return**: {model_performance.get('policy_return', 'N/A')}

## Training Data
- **Number of Patients**: {dataset_stats.get('n_patients', 'N/A')}
- **Number of Transitions**: {dataset_stats.get('n_transitions', 'N/A')}
- **Feature Dimensions**: {dataset_stats.get('state_dim', 'N/A')}
- **Action Space Size**: {dataset_stats.get('action_dim', 'N/A')}

## Ethical Considerations
- Model should augment, not replace, clinical judgment
- Regular monitoring needed for distribution shifts
- Ensure representative patient populations in training data

## Limitations
- Trained on simulated data (for this implementation)
- Assumes Markovian dynamics with finite history
- Conservative bias may underestimate novel treatment benefits

## Additional Notes
This implementation demonstrates the algorithm from the paper
"Reinforcement Learning-Based Interaction Modeling in a Digital Twin"
"""
    
    with open(save_path, 'w') as f:
        f.write(card_content)


if __name__ == "__main__":
    # Example usage of utilities
    print("Testing utility functions...")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Create dummy data for testing
    n_samples = 1000
    state_dim = 10
    action_dim = 5
    
    dummy_dataset = {
        'states': np.random.randn(n_samples, state_dim),
        'actions': np.random.randint(0, action_dim, n_samples),
        'rewards': np.random.randn(n_samples),
        'next_states': np.random.randn(n_samples, state_dim)
    }
    
    feature_names = [f'Feature_{i}' for i in range(state_dim)]
    
    # Test summary statistics
    print("\nComputing summary statistics...")
    feature_df, action_counts, reward_stats = create_summary_statistics(
        dummy_dataset, feature_names
    )
    print("Feature Statistics:")
    print(feature_df.head())
    print(f"\nAction Distribution:\n{action_counts}")
    print(f"\nReward Statistics: {reward_stats}")
    
    # Test overlap statistics
    print("\nComputing overlap statistics...")
    overlap_stats = compute_overlap_statistics(dummy_dataset, action_dim)
    print(f"Overlap Statistics: {overlap_stats}")
    
    # Test visualization
    print("\nCreating visualizations...")
    visualize_state_distributions(dummy_dataset, feature_names, 
                                 save_path='state_distributions.png')
    print("Saved state distributions plot")
    
    # Test configuration saving
    config = {
        'n_patients': 1000,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'learning_rate': 1e-3,
        'batch_size': 256
    }
    save_experiment_config(config)
    print("\nSaved experiment configuration")
    
    # Create model card
    model_performance = {
        'dynamics_mse': 0.023,
        'outcome_mse': 0.045,
        'policy_return': 8.76
    }
    
    dataset_stats = {
        'n_patients': 1000,
        'n_transitions': 25000,
        'state_dim': state_dim,
        'action_dim': action_dim
    }
    
    create_model_card(model_performance, dataset_stats)
    print("Created model card")