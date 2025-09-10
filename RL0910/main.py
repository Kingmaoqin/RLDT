"""
main.py - Main entry point for the Digital Twin system

This script provides a complete pipeline for training and deploying
the RL-based digital twin for treatment optimization.
"""

import argparse
import os
import logging
from datetime import datetime
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import PatientDataGenerator
from training import (
    train_digital_twin,
    train_outcome_model,
    train_rl_policy
)
from inference import (
    DigitalTwinInference,
    ClinicalDecisionSupport,
    load_trained_models
)
from utils import (
    set_random_seeds,
    compute_treatment_effect_heterogeneity,
    visualize_treatment_effects,
    analyze_feature_importance,
    plot_feature_importance,
    create_summary_statistics,
    visualize_state_distributions,
    save_experiment_config,
    create_model_card
)


def setup_logging(log_dir: str = 'logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'digital_twin_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_pipeline(args):
    """Complete training pipeline"""
    logger = setup_logging()
    logger.info("Starting Digital Twin training pipeline")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    
    # Generate or load data
    logger.info("Generating patient data...")
    generator = PatientDataGenerator(
        n_patients=args.n_patients,
        max_timesteps=args.max_timesteps,
        n_features=args.state_dim,
        n_actions=args.action_dim,
        seed=args.seed
    )
    
    data = generator.generate_dataset()
    logger.info(f"Generated {len(data['states'])} transitions from {args.n_patients} patients")
    
    # Save dataset
    if args.save_data:
        df = generator.create_dataframe(data)
        data_path = os.path.join(args.output_dir, 'patient_data.csv')
        df.to_csv(data_path, index=False)
        logger.info(f"Saved data to {data_path}")
    
    # Analyze and visualize data
    logger.info("Analyzing dataset...")
    feature_names = generator.feature_names
    
    # Summary statistics
    feature_stats, action_counts, reward_stats = create_summary_statistics(data, feature_names)
    logger.info(f"Action distribution:\n{action_counts}")
    logger.info(f"Reward stats: {reward_stats}")
    
    # Visualize state distributions
    visualize_state_distributions(
        data, feature_names,
        save_path=os.path.join(args.output_dir, 'figures', 'state_distributions.png')
    )
    
    # Stage 1: Train dynamics model
    logger.info("\n=== Stage 1: Training Dynamics Model ===")
    dynamics_model = train_digital_twin(
        data,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        n_epochs=args.dynamics_epochs,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'models')
    )
    
    # Save dynamics model
    dynamics_path = os.path.join(args.output_dir, 'models', 'dynamics_model.pth')
    torch.save(dynamics_model.state_dict(), dynamics_path)
    logger.info(f"Saved dynamics model to {dynamics_path}")
    
    # Stage 2: Train outcome model
    logger.info("\n=== Stage 2: Training Outcome Model ===")
    outcome_model = train_outcome_model(
        data,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        n_epochs=args.outcome_epochs,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'models')
    )
    
    # Save outcome model
    outcome_path = os.path.join(args.output_dir, 'models', 'outcome_model.pth')
    torch.save(outcome_model.state_dict(), outcome_path)
    logger.info(f"Saved outcome model to {outcome_path}")
    
    # Stage 3: Train RL policy
    logger.info("\n=== Stage 3: Training RL Policy ===")
    q_network = train_rl_policy(
        data,
        dynamics_model,
        outcome_model,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        n_iterations=args.rl_iterations,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'models')
    )
    
    # Save Q-network
    q_network_path = os.path.join(args.output_dir, 'models', 'q_network.pth')
    torch.save(q_network.state_dict(), q_network_path)
    logger.info(f"Saved Q-network to {q_network_path}")
    
    # Analyze learned policy
    logger.info("\n=== Analyzing Learned Policy ===")
    
    # Feature importance for Q-network
    q_importance = analyze_feature_importance(
        q_network, data, feature_names,
        n_samples=1000, device=args.device
    )
    
    plot_feature_importance(
        q_importance,
        save_path=os.path.join(args.output_dir, 'figures', 'q_network_importance.png')
    )
    
    # Treatment effect heterogeneity
    patient_states = np.array(data['states'])[:1000]  # Sample
    treatment_effects = compute_treatment_effect_heterogeneity(
        q_network, patient_states,
        action_pairs=(0, 3),  # Compare treatment 0 vs placebo
        device=args.device
    )
    
    visualize_treatment_effects(
        treatment_effects, patient_states, feature_names,
        save_path=os.path.join(args.output_dir, 'figures', 'treatment_effects.png')
    )
    
    # Save configuration
    config = {
        'n_patients': args.n_patients,
        'max_timesteps': args.max_timesteps,
        'state_dim': args.state_dim,
        'action_dim': args.action_dim,
        'dynamics_epochs': args.dynamics_epochs,
        'outcome_epochs': args.outcome_epochs,
        'rl_iterations': args.rl_iterations,
        'batch_size': args.batch_size,
        'seed': args.seed
    }
    
    save_experiment_config(
        config,
        save_path=os.path.join(args.output_dir, 'experiment_config.json')
    )
    
    # Create model card
    model_performance = {
        'dynamics_mse': 'See training logs',
        'outcome_mse': 'See training logs',
        'policy_return': 'See training logs'
    }
    
    dataset_stats = {
        'n_patients': args.n_patients,
        'n_transitions': len(data['states']),
        'state_dim': args.state_dim,
        'action_dim': args.action_dim
    }
    
    create_model_card(
        model_performance, dataset_stats,
        save_path=os.path.join(args.output_dir, 'model_card.md')
    )
    
    logger.info("\n=== Training Pipeline Complete ===")
    logger.info(f"All outputs saved to {args.output_dir}")
    
    return dynamics_model, outcome_model, q_network


def inference_pipeline(args):
    """Inference pipeline for treatment recommendations"""
    logger = setup_logging()
    logger.info("Starting Digital Twin inference pipeline")
    
    # Load models
    logger.info("Loading trained models...")
    inference_engine = load_trained_models(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=args.device
    )
    
    # Create clinical decision support system
    cds = ClinicalDecisionSupport(inference_engine)
    
    # Example patient cases
    test_patients = [
        {
            'patient_id': 'P001',
            'description': 'Young patient with high glucose',
            'age': 30,
            'gender': 0,
            'blood_pressure': 0.5,
            'heart_rate': 0.5,
            'glucose': 0.8,
            'creatinine': 0.5,
            'hemoglobin': 0.6,
            'temperature': 0.5,
            'oxygen_saturation': 0.95,
            'bmi': 0.5
        },
        {
            'patient_id': 'P002',
            'description': 'Elderly patient with multiple conditions',
            'age': 75,
            'gender': 1,
            'blood_pressure': 0.75,
            'heart_rate': 0.7,
            'glucose': 0.65,
            'creatinine': 0.7,
            'hemoglobin': 0.4,
            'temperature': 0.5,
            'oxygen_saturation': 0.88,
            'bmi': 0.6
        },
        {
            'patient_id': 'P003',
            'description': 'Middle-aged patient with low oxygen',
            'age': 50,
            'gender': 0,
            'blood_pressure': 0.6,
            'heart_rate': 0.6,
            'glucose': 0.55,
            'creatinine': 0.5,
            'hemoglobin': 0.5,
            'temperature': 0.52,
            'oxygen_saturation': 0.85,
            'bmi': 0.65
        }
    ]
    
    # Generate reports for test patients
    os.makedirs(args.output_dir, exist_ok=True)
    reports_dir = os.path.join(args.output_dir, 'patient_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    for patient in test_patients:
        logger.info(f"\nProcessing patient {patient['patient_id']}: {patient['description']}")
        
        # Generate report
        report_path = os.path.join(reports_dir, f"{patient['patient_id']}_report.html")
        cds.create_patient_report(patient, report_path)
        logger.info(f"Generated report: {report_path}")
        
        # Get detailed explanation
        state = cds._extract_patient_state(patient)
        explanation = inference_engine.explain_recommendation(state, n_simulations=50)
        
        logger.info(f"Recommendation: {explanation['recommendation']['recommended_treatment']}")
        logger.info(f"Confidence: {explanation['recommendation']['confidence']:.3f}")
        
        # Save trajectory visualization
        trajectory = inference_engine.simulate_treatment_trajectory(
            state,
            [explanation['recommendation']['recommended_action']] * 20
        )
        
        fig_path = os.path.join(reports_dir, f"{patient['patient_id']}_trajectory.png")
        inference_engine.visualize_treatment_trajectory(trajectory, save_path=fig_path)
    
    logger.info("\n=== Inference Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description='Digital Twin RL System for Treatment Optimization'
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'both'],
                       default='both', help='Running mode')
    
    # Data parameters
    parser.add_argument('--n_patients', type=int, default=1000,
                       help='Number of patients to simulate')
    parser.add_argument('--max_timesteps', type=int, default=50,
                       help='Maximum timesteps per patient')
    parser.add_argument('--state_dim', type=int, default=10,
                       help='Dimension of patient state')
    parser.add_argument('--action_dim', type=int, default=5,
                       help='Number of treatment actions')
    
    # Training parameters
    parser.add_argument('--dynamics_epochs', type=int, default=50,
                       help='Epochs for dynamics model')
    parser.add_argument('--outcome_epochs', type=int, default=30,
                       help='Epochs for outcome model')
    parser.add_argument('--rl_iterations', type=int, default=50000,
                       help='Iterations for RL training')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computing device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--save_data', action='store_true',
                       help='Save generated data')
    
    args = parser.parse_args()
    
    # Run appropriate pipeline
    if args.mode == 'train' or args.mode == 'both':
        dynamics_model, outcome_model, q_network = train_pipeline(args)
    
    if args.mode == 'inference' or args.mode == 'both':
        inference_pipeline(args)


if __name__ == "__main__":
    main()