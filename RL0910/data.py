"""
data.py - Simulated Patient Data Generation Module

This module generates synthetic patient trajectories for training the digital twin model.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random

class PatientDataGenerator:
    """Generates synthetic patient trajectories with complex interactions"""
    
    def __init__(self, 
                 n_patients: int = 1000,
                 max_timesteps: int = 50,
                 n_features: int = 10,
                 n_actions: int = 5,
                 seed: int = 42):
        """
        Initialize the patient data generator
        
        Args:
            n_patients: Number of patients to simulate
            max_timesteps: Maximum number of timesteps per patient
            n_features: Dimension of patient state (covariates)
            n_actions: Number of possible treatment actions
            seed: Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.max_timesteps = max_timesteps
        self.n_features = n_features
        self.n_actions = n_actions
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Define feature names (demographics, labs, vitals, etc.)
        self.feature_names = [
            'age', 'gender', 'blood_pressure', 'heart_rate', 
            'glucose_level', 'creatinine', 'hemoglobin',
            'temperature', 'oxygen_saturation', 'bmi'
        ][:n_features]
        
    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial patient state with realistic distributions"""
        state = np.zeros(self.n_features)
        
        # Age (normalized between 0-1, representing 18-90 years)
        state[0] = np.random.beta(5, 5)  
        
        # Gender (0 or 1)
        state[1] = np.random.binomial(1, 0.5)
        
        # Physiological measurements (normalized)
        # Blood pressure
        state[2] = np.random.normal(0.5, 0.15)
        # Heart rate  
        state[3] = np.random.normal(0.5, 0.1)
        # Glucose level
        state[4] = np.random.normal(0.5, 0.2)
        # Creatinine
        state[5] = np.random.normal(0.5, 0.1)
        # Hemoglobin
        state[6] = np.random.normal(0.6, 0.1)
        # Temperature
        state[7] = np.random.normal(0.5, 0.05)
        # Oxygen saturation
        state[8] = np.random.normal(0.95, 0.05)
        # BMI
        state[9] = np.random.normal(0.5, 0.15) if self.n_features > 9 else 0
        
        # Clip values to [0, 1] range
        state = np.clip(state, 0, 1)
        state[1] = int(state[1])  # Ensure gender is binary
        
        return state
    
    def _compute_treatment_effect(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute treatment effect with complex interactions
        
        This simulates how different treatments affect patient state based on
        their current condition, incorporating high-order interactions
        """
        effect = np.zeros(self.n_features)
        
        # Base treatment effects
        treatment_effects = {
            0: np.array([0, 0, -0.05, -0.03, -0.02, 0, 0.01, -0.01, 0.02, 0]),  # Medication A
            1: np.array([0, 0, -0.03, -0.02, -0.05, -0.01, 0.02, 0, 0.01, -0.01]),  # Medication B
            2: np.array([0, 0, -0.02, -0.04, -0.01, -0.02, 0, -0.02, 0.03, 0]),  # Medication C
            3: np.array([0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),  # Placebo
            4: np.array([0, 0, -0.04, -0.05, -0.03, -0.03, 0.03, -0.01, 0.04, -0.02]),  # Combination
        }
        
        base_effect = treatment_effects.get(action, np.zeros(self.n_features))[:self.n_features]
        
        # Add interaction effects
        # Example: Treatment 0 is more effective for younger patients with high glucose
        if action == 0:
            age_factor = 1 - state[0]  # Younger patients respond better
            glucose_factor = state[4]   # High glucose patients benefit more
            interaction = age_factor * glucose_factor * 0.1
            effect[4] -= interaction  # Additional glucose reduction
            
        # Example: Treatment 1 works better for patients with specific biomarker combination
        elif action == 1:
            # Complex 3-way interaction: gender, creatinine, hemoglobin
            if state[1] == 1 and state[5] > 0.6 and state[6] < 0.5:
                effect[5] -= 0.05  # Extra creatinine reduction
                effect[6] += 0.03  # Hemoglobin improvement
                
        # Example: Treatment 4 (combination) has synergistic effects
        elif action == 4:
            # 4-way interaction: age, BP, heart rate, oxygen
            if state[0] > 0.5 and state[2] > 0.6 and state[3] > 0.6 and state[8] < 0.9:
                effect[2] -= 0.08  # Strong BP reduction
                effect[3] -= 0.06  # Heart rate reduction
                effect[8] += 0.05  # Oxygen improvement
        
        # Add base effects
        effect += base_effect
        
        # Add some noise
        effect += np.random.normal(0, 0.01, self.n_features)
        
        return effect
    
    def _transition_dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Simulate patient state transition given current state and treatment
        
        This implements P(s_{t+1} | s_t, a_t) with complex dynamics
        """
        # Natural progression (without treatment)
        natural_change = np.zeros(self.n_features)
        
        # Some features naturally deteriorate
        natural_change[2] += 0.01  # BP tends to increase
        natural_change[4] += 0.01  # Glucose tends to increase
        natural_change[8] -= 0.005  # Oxygen tends to decrease slightly
        
        # Apply treatment effects
        treatment_effect = self._compute_treatment_effect(state, action)
        
        # Compute next state
        next_state = state + natural_change + treatment_effect
        
        # Ensure physiological constraints
        next_state = np.clip(next_state, 0, 1)
        next_state[1] = state[1]  # Gender doesn't change
        next_state[0] = state[0]  # Age doesn't change (in this short timeframe)
        
        return next_state
    
    def _compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Compute immediate reward/outcome for the transition
        
        This implements R(s, a) - the clinical outcome
        """
        # Health score based on vital signs and biomarkers
        health_score = 0.0
        
        # Penalize abnormal values (assuming 0.5 is normal for most features)
        for i in [2, 3, 4, 5, 6, 7]:  # Key health indicators
            health_score -= abs(next_state[i] - 0.5) * 2
            
        # Bonus for high oxygen saturation
        health_score += (next_state[8] - 0.9) * 5 if next_state[8] > 0.9 else (next_state[8] - 0.9) * 10
        
        # Improvement bonus
        improvement = 0.0
        for i in [2, 3, 4, 5]:  # Focus on key metrics
            if abs(next_state[i] - 0.5) < abs(state[i] - 0.5):
                improvement += 0.5
                
        # Treatment cost (some treatments have side effects)
        treatment_cost = [0, 0.1, 0.1, 0, 0.3][action] if action < 5 else 0
        
        # Total reward
        reward = health_score + improvement - treatment_cost
        
        return reward
    
    def generate_dataset(self) -> Dict[str, List]:
        """
        Generate the complete dataset of patient trajectories
        
        Returns:
            Dictionary containing:
                - states: List of state arrays
                - actions: List of actions taken
                - rewards: List of immediate rewards
                - next_states: List of next state arrays
                - trajectory_ids: List of patient IDs
                - timesteps: List of timestep indices
        """
        data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'trajectory_ids': [],
            'timesteps': []
        }
        
        for patient_id in range(self.n_patients):
            # Random trajectory length
            trajectory_length = np.random.randint(10, self.max_timesteps)
            
            # Initialize patient state
            state = self._generate_initial_state()
            
            for t in range(trajectory_length):
                # Simulate physician's treatment decision (behavior policy)
                # This creates a realistic action distribution in the data
                action = self._behavior_policy(state)
                
                # Transition to next state
                next_state = self._transition_dynamics(state, action)
                
                # Compute reward
                reward = self._compute_reward(state, action, next_state)
                
                # Store transition
                data['states'].append(state.copy())
                data['actions'].append(action)
                data['rewards'].append(reward)
                data['next_states'].append(next_state.copy())
                data['trajectory_ids'].append(patient_id)
                data['timesteps'].append(t)
                
                # Update state
                state = next_state
                
                # Early termination if patient reaches critical condition
                if state[8] < 0.8 or np.random.random() < 0.05:
                    break
        
        return data
    
    def _behavior_policy(self, state: np.ndarray) -> int:
        """
        Simulate physician's treatment selection (behavior policy)
        
        This creates a realistic distribution of actions in the dataset
        """
        # Physicians tend to be conservative
        action_probs = np.array([0.3, 0.3, 0.2, 0.15, 0.05])
        
        # Adjust based on patient condition
        if state[4] > 0.7:  # High glucose
            action_probs[0] += 0.2  # Prefer medication A
            action_probs[3] -= 0.1  # Less placebo
            
        if state[2] > 0.7:  # High BP
            action_probs[1] += 0.2  # Prefer medication B
            action_probs[3] -= 0.1
            
        if state[8] < 0.9:  # Low oxygen
            action_probs[4] += 0.3  # More likely to use combination
            action_probs[3] -= 0.15
        
        # Normalize
        action_probs = np.clip(action_probs, 0.01, 1)
        action_probs /= action_probs.sum()
        
        return np.random.choice(self.n_actions, p=action_probs)
    
    def create_dataframe(self, data: Dict[str, List]) -> pd.DataFrame:
        """Convert dataset dictionary to pandas DataFrame"""
        df_data = []
        
        for i in range(len(data['states'])):
            row = {
                'trajectory_id': data['trajectory_ids'][i],
                'timestep': data['timesteps'][i],
                'action': data['actions'][i],
                'reward': data['rewards'][i]
            }
            
            # Add state features
            for j, feature in enumerate(self.feature_names):
                row[f'state_{feature}'] = data['states'][i][j]
                row[f'next_state_{feature}'] = data['next_states'][i][j]
                
            df_data.append(row)
            
        return pd.DataFrame(df_data)


if __name__ == "__main__":
    # Example usage
    generator = PatientDataGenerator(n_patients=1000, max_timesteps=50)
    data = generator.generate_dataset()
    
    print(f"Generated {len(data['states'])} transitions from {generator.n_patients} patients")
    print(f"State dimension: {generator.n_features}")
    print(f"Action space size: {generator.n_actions}")
    
    # Convert to DataFrame for easier analysis
    df = generator.create_dataframe(data)
    print("\nDataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Save data
    df.to_csv('patient_trajectories.csv', index=False)
    print("\nData saved to patient_trajectories.csv")