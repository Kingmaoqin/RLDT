"""
enhanced_chat_ui.py - Enhanced UI for DRIVE with online learning and hot parameter updates
"""

import gradio as gr
import json
import argparse
import sys
import os
import pandas as pd
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent_graph import drive_agent, AgentState
from drive_tools import (
    initialize_tools, 
    describe_parameter,
    recommend_parameters,
    update_reward_parameters,
    retrain_model,
    get_patient_list,
    get_patient_data,
    analyze_patient,
    load_data_source,
    get_cohort_stats,
    update_hyperparams,  # New hot update function
    online_finetune,      # New online finetuning
    get_online_stats,     # New online stats
    pause_online_training,    # New pause function
    resume_online_training    # New resume function
)
from data_manager import data_manager
from inference import DigitalTwinInference, ClinicalDecisionSupport
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from online_evaluation import OnlineEvaluator, ContinualEvaluator, create_online_evaluation_pipeline
from run_complete_evaluation import (
    run_enhanced_evaluation,
    test_safety_compliance,
    trigger_distribution_shift_test,
    generate_paper_compliance_report
)
from online_monitor import OnlineSystemMonitor
from system_health_check import SystemHealthChecker

from drive_tools import load_data_source, generate_patient_report, generate_patient_report_ui

# ---- Optional BCQ loader ----
def _load_bcq_policy(path: str):
    if not path or not os.path.exists(path):
        print(f"[BCQ] policy not found: {path}")
        return None
    try:
        from d3rlpy.algos import load_learnable as ll
        policy = ll(path)
    except Exception:
        try:
            import d3rlpy
            policy = d3rlpy.load_learnable(path)
        except Exception as e:
            print(f"[BCQ] failed to load: {e}")
            return None
    print("[BCQ] policy loaded")
    return policy

# Global variables for model paths and current parameters
MODEL_PATHS = {
    "dynamics_model": "./output/models/dynamics_model_0.pth",
    "outcome_model": "./output/models/best_outcome_model.pth",
    "q_network": "./output/models/best_q_network.pth",
    "bcq_policy": "./output/models/best_bcq_policy.d3"
}

CURRENT_PARAMS = {
    "alpha": 1.0,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "regularization_weight": 0.01,
    "batch_size": 256,
    "n_epochs": 50
}

# Demo patient state
DEMO_PATIENT = {
    "age": 55,
    "gender": 1,
    "blood_pressure": 0.7,
    "heart_rate": 0.65,
    "glucose": 0.75,
    "creatinine": 0.6,
    "hemoglobin": 0.45,
    "temperature": 0.5,
    "oxygen_saturation": 0.88,
    "bmi": 0.55
}


def load_models_and_initialize():
    """Load pre-trained models and initialize tools"""
    print("Loading models...")
    
    # Initialize models
    state_dim = 10
    action_dim = 5
    
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
    q_network = ConservativeQNetwork(state_dim, action_dim)
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATHS["dynamics_model"]):
        dynamics_model.load_state_dict(torch.load(MODEL_PATHS["dynamics_model"]))
        print("Loaded dynamics model")
    else:
        print(f"Warning: {MODEL_PATHS['dynamics_model']} not found, using random weights")
    
    if os.path.exists(MODEL_PATHS["outcome_model"]):
        # Load with compatibility for BatchNorm -> LayerNorm conversion
        state_dict = torch.load(MODEL_PATHS["outcome_model"])
        # Remove BatchNorm specific keys
        keys_to_remove = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
        for k in keys_to_remove:
            del state_dict[k]
        # Try to load the modified state dict
        try:
            outcome_model.load_state_dict(state_dict, strict=False)
            print("Loaded outcome model (converted from BatchNorm to LayerNorm)")
        except Exception as e:
            print(f"Warning: Could not fully load outcome model: {e}")
            print("Using randomly initialized outcome model")
    else:
        print(f"Warning: {MODEL_PATHS['outcome_model']} not found, using random weights")
    
    if os.path.exists(MODEL_PATHS["q_network"]):
        # Load with compatibility for BatchNorm -> LayerNorm conversion
        state_dict = torch.load(MODEL_PATHS["q_network"])
        # Remove BatchNorm specific keys
        keys_to_remove = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
        for k in keys_to_remove:
            del state_dict[k]
        # Try to load the modified state dict
        try:
            q_network.load_state_dict(state_dict, strict=False)
            print("Loaded Q-network (converted from BatchNorm to LayerNorm)")
        except Exception as e:
            print(f"Warning: Could not fully load Q-network: {e}")
            print("Using randomly initialized Q-network")
    else:
        print(f"Warning: {MODEL_PATHS['q_network']} not found, using random weights")
    
    # Create inference engine
    inference_engine = DigitalTwinInference(
        dynamics_model, outcome_model, q_network,
        state_dim, action_dim
    )
    
    # Create CDS
    cds = ClinicalDecisionSupport(inference_engine)
    
    # Initialize tools
    initialize_tools(inference_engine, cds)

    # Try loading BCQ policy and attach to inference engine
    bcq = _load_bcq_policy(MODEL_PATHS.get("bcq_policy", ""))
    if bcq is not None:
        inference_engine.bcq_policy = bcq
        print("[BCQ] Using BCQ for action selection in UI.")
    # Ensure online stream starts
    # try:
    #     from drive_tools import resume_online_training
    #     resume_online_training()
    # except Exception as e:
    #     print(f"[WARN] Could not start online stream: {e}")

    
    print("Models loaded and tools initialized.")
    return inference_engine, cds


def chat_function(message, history):
    """Process chat message and return response"""
    try:
        # Create state
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            human_review=False
        )
        
        # Run agent
        result = drive_agent.invoke(initial_state)
        
        # Extract response
        response = ""
        for msg in result["messages"]:
            if msg.type == "ai":
                response = msg.content
                break
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def get_param_description(param_name):
    """Get parameter description for display"""
    result = describe_parameter(param_name)
    if "error" in result:
        return "Unknown parameter"
    return f"**{result['name']}**\n\n{result['description']}\n\nMechanism: {result['mechanism']}\n\nImpact: {result['impact']}"


def update_preset(preset_name):
    """Update parameters based on preset selection"""
    presets = {
        "Conservative": {
            "alpha": 1.5,
            "gamma": 0.99,
            "learning_rate": 5e-4,
            "regularization_weight": 0.05,
            "batch_size": 256,
            "n_epochs": 30
        },
        "Balanced": {
            "alpha": 1.0,
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "regularization_weight": 0.01,
            "batch_size": 128,
            "n_epochs": 50
        },
        "Aggressive": {
            "alpha": 0.5,
            "gamma": 0.95,
            "learning_rate": 2e-3,
            "regularization_weight": 0.005,
            "batch_size": 64,
            "n_epochs": 70
        },
        "Custom": CURRENT_PARAMS
    }
    
    params = presets.get(preset_name, CURRENT_PARAMS)
    
    # Return updated slider values
    return (
        params["alpha"],
        params["gamma"],
        params["learning_rate"],
        params["regularization_weight"],
        params["batch_size"],
        params["n_epochs"],
        gr.update(visible=(preset_name == "Custom"))  # Show sliders only for custom
    )


def recommend_config():
    """Get recommended configuration based on patient state"""
    result = recommend_parameters(DEMO_PATIENT)
    if "error" in result:
        return "Error getting recommendations"
    
    recommendations = result["recommendations"]
    output = "## Recommended Configurations\n\n"
    
    for rec in recommendations:
        output += f"**{rec['preset'].title()} Configuration**\n"
        output += f"- Reason: {rec['reason']}\n"
        output += f"- Expected Effect: {rec['expected_effect']}\n"
        output += f"- Parameters:\n"
        for param, value in rec['parameters'].items():
            output += f"  - {param}: {value}\n"
        output += "\n"
    
    output += "\n💡 **With online learning, these parameters can be adjusted on-the-fly without full retraining!**"
    
    return output


def apply_config(alpha, gamma, lr, reg_weight, batch_size, n_epochs, confirm):
    """Apply configuration with HOT UPDATE instead of full retrain"""
    if not confirm:
        return "Please check the confirmation box to apply changes."
    
    # Use hot update mechanism
    params_to_update = {
        "alpha": alpha,
        "gamma": gamma,
        "learning_rate": lr,
        "regularization_weight": reg_weight,
        "batch_size": int(batch_size),
        "tau": CURRENT_PARAMS.get("tau", 0.05)  # Keep current tau if not changed
    }
    
    # Call hot update function
    from drive_tools import update_hyperparams
    result = update_hyperparams(params_to_update)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    # Format response based on update type
    output = "## ✅ Configuration Applied\n\n"
    
    if result.get("tier1_instant"):
        output += f"**Instant Updates**: {', '.join(result['tier1_instant'])}\n"
    
    if result.get("tier2_adapting"):
        output += f"**Parameters requiring fast adaptation**: {', '.join(result['finetuning'])}\n"
        output += f"**Job ID**: {result.get('job_id', 'N/A')}\n"
        output += f"**Estimated time**: {result.get('estimated_time', 'Unknown')}\n\n"
        output += "🔄 Tier-2 adaptation in progress. The model is adapting to new parameters.\n"
    else:
        output += "All parameters updated instantly. No finetuning required.\n"
    
    if result.get("updated"):
        output += "\n**Changes applied**:\n"
        for change in result["updated"]:
            output += f"- {change}\n"
    
    return output


def retrain_with_params(preset, alpha, gamma, lr, reg_weight, batch_size, n_epochs, confirm_retrain):
    """Updated to recommend online finetuning instead of full retrain"""
    if not confirm_retrain:
        return "Please check the confirmation box to proceed."
    
    # First try online finetuning
    from drive_tools import online_finetune
    
    training_params = {
        "preset": preset.lower() if preset != "Custom" else "custom",
        "parameters": {
            "alpha": alpha,
            "gamma": gamma,
            "learning_rate": lr,
            "regularization_weight": reg_weight,
            "batch_size": int(batch_size)
        }
    }
    
    # Call online finetune
    result = online_finetune(training_params)
    
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    # Format response
    output = f"## 🚀 Online Adaptation Started\n\n"
    output += f"**Status**: {result.get('status', 'Unknown')}\n"
    output += f"**Type**: Online Finetuning (Much faster than full retraining)\n"
    output += f"**Job ID**: {result.get('job_id', 'N/A')}\n"
    output += f"**Duration**: {result.get('duration', 'Unknown')}\n\n"
    
    output += "### Why Online Finetuning?\n"
    output += "- ⚡ 10-100x faster than full retraining\n"
    output += "- 📊 Preserves existing knowledge while adapting\n"
    output += "- 🔄 Continuous learning from new data\n\n"
    
    output += "The system is now adapting to your configuration. You can continue using it while it updates."
    
    return output


def create_gradio_interface():
    """Create the full Gradio interface with chat, parameter control, and online learning monitor"""
    
    # Initialize models
    print("Initializing system with online learning...")
    inference_engine, cds = load_models_and_initialize()
    
    # Initialize data manager with virtual data
    print("Generating initial virtual data...")
    data_manager.generate_virtual_data(n_patients=100)
    
    print("Initializing evaluation system...")
    evaluation_pipeline = create_online_evaluation_pipeline(
        models={
            'dynamics': inference_engine.dynamics_model,
            'outcome': inference_engine.outcome_model,
            'q_network': inference_engine.q_network
        },
        test_data={'states': [], 'actions': []}  # 初始为空，实际使用时会更新
    )    
    with gr.Blocks(title="Real-time Interactive Clinical Navigator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 Real-time Interactive Clinical Navigator
        
        **AI-powered treatment recommendations with explainable causal reasoning**
        
        ✨ **New Features**:
        - 🔄 Online Learning: Continuously adapts to new data
        - ⚡ Hot Parameter Updates: Change settings without retraining
        - 🎯 Active Learning: Only queries uncertain cases
        - 📊 Real-time Training Monitor
        """)
        
        with gr.Tabs():
            # Tab 1: Data Management
            with gr.Tab("📊 Data Management"):
                gr.Markdown("### Data Source Configuration")
                
                with gr.Row():
                    data_source_radio = gr.Radio(
                        choices=["Virtual Data", "Real Data"],
                        value="Virtual Data",
                        label="Data Source"
                    )
                    current_source_text = gr.Textbox(
                        value="Current: Virtual Data",
                        label="Active Source",
                        interactive=False
                    )
                
                # Virtual data options
                with gr.Column(visible=True) as virtual_data_options:
                    with gr.Row():
                        n_patients_slider = gr.Slider(
                            minimum=10,
                            maximum=10000,
                            value=100,
                            step=10,
                            label="Number of Patients"
                        )
                        generate_btn = gr.Button("Generate Virtual Data", variant="primary")
                
                # Real data options
                with gr.Column(visible=False) as real_data_options:
                    with gr.Row():
                        file_upload = gr.File(label="Upload Data File", file_types=[".csv", ".parquet", ".xlsx", ".xls"])
                        schema_upload = gr.File(label="Upload Schema YAML (optional but recommended)", file_types=[".yaml", ".yml"])
                        load_real_btn = gr.Button("Load Real Data", variant="primary")


                
                # Data statistics
                gr.Markdown("### Dataset Overview")
                current_source_text = gr.Textbox(label="Active Source", interactive=False)
                stats_display = gr.Image(label="Dataset Statistics", interactive=False)
                action_legend = gr.HTML(label="Action Legend", visible=False)

                # Patient selection
                gr.Markdown("### Patient Selection")
                with gr.Row():
                    patient_dropdown = gr.Dropdown(
                        label="Select Patient",
                        choices=[],
                        value=None
                    )
                    refresh_patients_btn = gr.Button("🔄 Refresh List")
                
                # Patient details
                with gr.Row():
                    patient_info_display = gr.Plot(label="Patient Information")
                    generate_report_btn = gr.Button("🧾 Generate Patient Report", variant="primary")
                    patient_report_html = gr.HTML(visible=True)
                    patient_analysis_display = gr.Image(label="Treatment Analysis", visible=False)
                    report_download = gr.File(label="Report (HTML)", visible=False)

                    def _on_generate_report(pid):
                        try:
                            if not pid:
                                return "No patient selected", None, gr.update(visible=False)
                            html, img, path = generate_patient_report_ui(pid, topk=3, fmt="html")
                            return html, img, gr.update(value=path, visible=bool(path))
                        except Exception as e:
                            return f"<p>Report error: {e}</p>", None, gr.update(visible=False)


                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Patient", variant="secondary", visible=False)
                    export_btn = gr.Button("Export Patient Data")
                
                export_output = gr.File(label="Exported Data", visible=False)
            
            # Tab 2: Chat Interface
            with gr.Tab("💬 Clinical Consultation"):
                with gr.Row():
                    active_patient_display = gr.Textbox(
                        label="Active Patient",
                        value="No patient selected",
                        interactive=False
                    )
                
                chatbot = gr.Chatbot(height=400, type="messages")
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="E.g., What's the best treatment? Why not use placebo? Simulate 7-day trajectory...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "What's the recommended treatment for this patient?",
                        "Compare all treatment options",
                        "Why recommend Medication A over Medication B?",
                        "Simulate 7-day trajectory with Medication A",
                        "What are the key factors in this decision?",
                        "How would lowering the learning rate affect the model?",
                        "Recommend parameters for this patient"
                    ],
                    inputs=msg
                )
            
            # Tab 3: Parameter Control
            with gr.Tab("⚙️ Parameter Control"):
                gr.Markdown("### Model Parameter Configuration")
                
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=["Conservative", "Balanced", "Aggressive", "Custom"],
                        value="Balanced",
                        label="Parameter Preset",
                        info="Select a preset or choose Custom to adjust manually"
                    )
                    recommend_btn = gr.Button("📊 Get Recommendations", scale=2)
                
                with gr.Column(visible=False) as custom_params:
                    gr.Markdown("### Custom Parameters")
                    
                    with gr.Row():
                        alpha_slider = gr.Slider(
                            minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                            label="Alpha (α) - Conservative Q-Learning Weight",
                            info="Higher = more conservative recommendations"
                        )
                        alpha_info = gr.Button("❓", scale=0)
                    
                    with gr.Row():
                        gamma_slider = gr.Slider(
                            minimum=0.9, maximum=0.999, value=0.99, step=0.001,
                            label="Gamma (γ) - Discount Factor",
                            info="Higher = values long-term outcomes more"
                        )
                        gamma_info = gr.Button("❓", scale=0)
                    
                    with gr.Row():
                        lr_slider = gr.Slider(
                            minimum=1e-4, maximum=1e-2, value=1e-3, step=1e-4,
                            label="Learning Rate",
                            info="Lower = slower but more stable learning"
                        )
                        lr_info = gr.Button("❓", scale=0)
                    
                    with gr.Row():
                        reg_slider = gr.Slider(
                            minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                            label="Regularization Weight (λ)",
                            info="Higher = stronger deconfounding"
                        )
                        reg_info = gr.Button("❓", scale=0)
                    
                    with gr.Row():
                        batch_slider = gr.Slider(
                            minimum=32, maximum=512, value=256, step=32,
                            label="Batch Size",
                            info="Larger = smoother gradients"
                        )
                        batch_info = gr.Button("❓", scale=0)
                    
                    with gr.Row():
                        epoch_slider = gr.Slider(
                            minimum=10, maximum=100, value=50, step=5,
                            label="Number of Epochs",
                            info="More epochs = better learning (with overfit risk)"
                        )
                        epoch_info = gr.Button("❓", scale=0)
                
                recommendations_display = gr.Markdown("Click 'Get Recommendations' to see suggested configurations")
                
                with gr.Row():
                    confirm_apply = gr.Checkbox(label="I confirm these parameter changes", value=False)
                    apply_btn = gr.Button("✅ Apply Configuration", variant="primary")
                
                apply_output = gr.Textbox(label="Configuration Status", lines=3)
                
                gr.Markdown("### Model Retraining")
                gr.Markdown("""
                ⚡ **Note**: With the new online learning system, full retraining is rarely needed!
                
                - **Hot Updates**: Parameter changes like α, γ, learning rate apply instantly or with quick finetuning
                - **Online Learning**: The model continuously learns from new data
                - **Active Learning**: Only queries uncertain cases, reducing labeling cost
                
                Full retraining is now only recommended for major architectural changes.
                """)
                
                with gr.Row():
                    confirm_retrain = gr.Checkbox(label="I want to trigger online adaptation (5-10 min)", value=False)
                    retrain_btn = gr.Button("🔄 Start Online Adaptation", variant="primary")
                
                retrain_output = gr.Markdown()
            
            # Tab 4: Online Learning Monitor
            with gr.Tab("📊 Online Learning Monitor"):
                gr.Markdown("### Real-time Training Statistics")
                
                # 按钮行
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="primary")
                    pause_btn = gr.Button("⏸️ Pause Training")
                    resume_btn = gr.Button("▶️ Resume Training")
                    evaluate_btn = gr.Button("📊 Run Evaluation", variant="secondary") # 添加这个
                
                # 统计数据显示和 Active Learning Statistics JSON 显示在同一行，分成两列
                with gr.Row():
                    with gr.Column(scale=1): # 左侧的统计数字列
                        total_transitions = gr.Number(label="Total Transitions Seen", value=0)
                        query_rate = gr.Number(label="Query Rate (%)", value=0)
                        buffer_size = gr.Number(label="Labeled Buffer Size", value=0)
                        
                        avg_uncertainty = gr.Number(label="Average Uncertainty", value=0)
                        current_tau = gr.Number(label="Current Threshold (τ)", value=0.05)
                        training_updates = gr.Number(label="Total Updates", value=0)
                    
                    with gr.Column(scale=2): # 右侧的 Active Learning Statistics 显示列
                        # 使用图表替代JSON
                        al_stats_plot = gr.Plot(label="Active Learning Statistics")
                        al_stats_table = gr.Dataframe(
                            headers=["Metric", "Value"],
                            label="Statistics Summary",
                            interactive=False
                        )

                gr.Markdown("### ⚙️ Evaluation Settings")
                with gr.Row():
                    eval_duration_slider = gr.Slider(
                        minimum=30,
                        maximum=600,
                        value=60,
                        step=30,
                        label="Evaluation Duration (seconds)",
                        info="How long to run the compliance evaluation"
                    )
                    eval_scenario_dropdown = gr.Dropdown(
                        choices=["Quick Test", "Standard Evaluation", "Full Compliance Check"],
                        value="Standard Evaluation",
                        label="Evaluation Scenario"
                    )
                    start_eval_btn = gr.Button("🚀 Start Custom Evaluation", variant="primary")
                   
                gr.Markdown("### 👨‍⚕️ Expert Feedback Mode")
                gr.Markdown("""
                **Instructions for Expert Labeling:**
                - The system queries uncertain cases where the model needs expert input
                - You can choose between automatic simulation or manual expert labeling
                - For manual labeling, adjust the reward value based on clinical outcome
                
                **Reward Guidelines:**
                - **Positive values (0 to +5)**: Patient improved (higher = better improvement)
                - **Near zero (-1 to +1)**: Stable condition, minimal change
                - **Negative values (-5 to 0)**: Patient deteriorated (lower = worse deterioration)
                - Consider: vital sign changes, symptom relief, adverse events
                """)
                
                with gr.Row():
                    expert_mode = gr.Radio(
                        choices=["Automatic Simulation", "Manual Expert Input"],
                        value="Automatic Simulation",
                        label="Expert Feedback Mode",
                        info="Choose how to handle uncertain cases requiring expert labels"
                    )
                    expert_queue_size = gr.Number(
                        label="Pending Expert Reviews",
                        value=0,
                        interactive=False
                    )
                
                with gr.Row():
                    # 显示当前待标注的案例
                    with gr.Column(scale=2):
                        current_case_display = gr.JSON(
                            label="Current Case for Review",
                            value={}
                        )
                    
                    # 专家输入控制
                    with gr.Column(scale=1):
                        expert_reward_slider = gr.Slider(
                            minimum=-5.0,
                            maximum=5.0,
                            value=0.0,
                            step=0.1,
                            label="Expert Reward Assessment",
                            info="Slide to set the clinical outcome value",
                            interactive=False
                        )
                        
                        reward_interpretation = gr.Textbox(
                            label="Interpretation",
                            value="Neutral outcome",
                            interactive=False
                        )
                        
                        submit_expert_label_btn = gr.Button(
                            "✅ Submit Expert Label",
                            variant="primary",
                            interactive=False
                        )
                        
                        skip_case_btn = gr.Button(
                            "⏭️ Skip Case",
                            variant="secondary",
                            interactive=False
                        )
                        refresh_case_btn = gr.Button(
                            "🔄 Get Next Case",
                            variant="secondary"
                        )


                with gr.Row():
                    expert_stats = gr.Textbox(
                        label="Expert Labeling Statistics",
                        value="No labels submitted yet",
                        lines=3,
                        interactive=False
                    )


                # Uncertainty threshold adjustment 部分
                gr.Markdown("### Adjust Active Learning Threshold")
                with gr.Row():
                    tau_slider = gr.Slider(
                        minimum=0.01, maximum=0.2, value=0.05, step=0.01,
                        label="Uncertainty Threshold (τ)",
                        info="Lower = query more samples, Higher = query fewer samples"
                    )
                    update_tau_btn = gr.Button("Update Threshold")
                
                tau_update_output = gr.Textbox(label="Update Status", lines=2)
                
                # 评估结果部分
                gr.Markdown("### Evaluation Results")
                with gr.Row():
                    evaluation_text = gr.Textbox(
                        label="Evaluation Report",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        visible=False
                    )
                    evaluation_plot = gr.Image(label="Performance Metrics", visible=False)
                
                # 下载报告按钮
                with gr.Row():
                    download_report_btn = gr.Button("📥 Download Report", visible=False)
                    report_file = gr.File(label="Downloaded Report", visible=False)                
                
                # 如何工作说明
                gr.Markdown("""
                ### How Online Learning Works
                
                1. **Active Learning**: Only uncertain samples are queried for expert labels
                2. **Incremental Updates**: Models update continuously without full retraining
                3. **Hot Parameters**: Change α, γ, lr instantly without stopping the system
                4. **Auto-save**: Models checkpoint every 10 minutes
                
                The system is learning in real-time while you use it!
                """)
            
            # Tab 5: Model Info (renumbered from Tab 4)
            with gr.Tab("📈 Model Information"):
                gr.Markdown("""
                ### Current Model Configuration
                
                **Model Paths**:
                - Dynamics Model: `{}`
                - Outcome Model: `{}`
                - Q-Network: `{}`
                
                **Architecture**:
                - State Dimension: 10 (patient features)
                - Action Space: 5 treatments
                - Dynamics Model: Transformer-based
                - Outcome Model: Deconfounded reward prediction
                - Policy: Conservative Q-Learning
                
                **Treatment Options**:
                1. Medication A - Primary glucose control
                2. Medication B - Blood pressure focus
                3. Medication C - Balanced approach
                4. Placebo - No active treatment
                5. Combination Therapy - Multi-drug approach
                
                **Patient Features**:
                - Age (normalized)
                - Gender (binary)
                - Blood Pressure
                - Heart Rate
                - Glucose Level
                - Creatinine
                - Hemoglobin
                - Temperature
                - Oxygen Saturation
                - BMI
                """.format(
                    MODEL_PATHS["dynamics_model"],
                    MODEL_PATHS["outcome_model"],
                    MODEL_PATHS["q_network"]
                ))
        
        # State variables
        active_patient_id = gr.State(None)
        def create_dataset_stats_image(stats: dict):
            """
            四宫格：
            - 左上：Treatment Distribution（饼图）
            - 右上：Key Metrics（柱图：Total Patients / Total Records / Avg Trajectory / Avg Reward(或N/A)）
            - 左下：Feature Statistics（mean/std/min/max 的热图；最多取10个特征）
            - 右下：Dataset Summary（文本卡片）
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import io
            from PIL import Image
            from data_manager import data_manager

            # 拿到当前数据与 meta
            try:
                df = data_manager.get_current_data()
            except Exception:
                df = None
            meta = data_manager.get_current_meta()
            id2name = (meta.get("action_map")
                    or {i: n for i, n in enumerate(meta.get("action_names") or [])})

            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)
            fig.suptitle("Dataset Overview", fontsize=16)

            # 左上：饼图
            ax = fig.add_subplot(gs[0, 0])
            ac = stats.get("action_counts", {}) or {}
            if ac:
                labels_raw = list(ac.keys())
                values = [ac[k] for k in labels_raw]
                def _name(k):
                    try:
                        return id2name.get(int(k), str(k))
                    except Exception:
                        return str(k)
                labels = [_name(k) for k in labels_raw]
                ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
                ax.set_title("Treatment Distribution")
            else:
                ax.axis("off"); ax.text(0.5,0.5,"No actions",ha="center",va="center")

            # 右上：关键指标柱图
            ax = fig.add_subplot(gs[0, 1])
            tp = stats.get("total_patients", 0)
            tr = stats.get("total_records", 0)
            th = stats.get("traj_len_hist", {}) or {}
            avg_len = th.get("mean", 0.0)
            # Avg Reward：若数据没有 reward 列则用 NaN -> 显示 0 或 N/A
            if df is not None and "reward" in df.columns and len(df):
                avg_r = float(np.nanmean(df["reward"].values))
                avg_r_label = f"{avg_r:.3f}"
            else:
                avg_r = 0.0
                avg_r_label = "N/A"
            bars = ["Total Patients", "Total Records", "Avg Trajectory", "Avg Reward"]
            vals = [tp, tr, avg_len, avg_r]
            ax.bar(bars, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            ax.set_title("Key Metrics")
            for i, v in enumerate(vals):
                label = f"{v:.1f}" if i < 3 else avg_r_label
                ax.text(i, v, label, ha="center", va="bottom", fontsize=9)

            # 左下：特征统计热图（mean/std/min/max）
            ax = fig.add_subplot(gs[1, 0])
            # 选特征列
            feature_cols = meta.get("feature_columns") or ([c for c in (df.columns if df is not None else []) if str(c).startswith("state_")])
            feature_cols = feature_cols[:10]  # 最多10个，避免过密
            if df is not None and feature_cols:
                data = df[feature_cols].copy()
                # 标准化到 [0,1]（避免量纲影响视觉）
                normalized = (data - data.min()) / (data.max() - data.min() + 1e-12)
                stats_mat = np.vstack([
                    np.nanmean(normalized.values, axis=0),
                    np.nanstd(normalized.values, axis=0),
                    np.nanmin(normalized.values, axis=0),
                    np.nanmax(normalized.values, axis=0),
                ])
                im = ax.imshow(stats_mat, aspect="auto", interpolation="nearest")
                ax.set_yticks(range(4)); ax.set_yticklabels(["mean", "std", "min", "max"])
                ax.set_xticks(range(len(feature_cols))); ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
                ax.set_title("Feature Statistics")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off"); ax.text(0.5,0.5,"No feature columns",ha="center",va="center")

            # 右下：文本摘要卡
            ax = fig.add_subplot(gs[1, 1])
            ax.axis("off")
            import datetime
            lines = [
                f"- Total Patients: {tp}",
                f"- Total Records: {tr}",
                f"- Avg Trajectory Length: {avg_len:.1f}",
                f"- Avg Reward: {avg_r_label}",
                "",
                f"Data Source: {getattr(data_manager, 'current_source', 'unknown')}",
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ]
            text = "\n".join(lines)
            ax.text(0.02, 0.98, "Dataset Summary:", fontsize=12, weight="bold", va="top")
            ax.text(0.04, 0.9, text, fontsize=11, va="top",
                    bbox=dict(boxstyle="round,pad=0.6", fc="#f0f5ff", ec="#b3c7ff", alpha=0.95))

            plt.tight_layout(rect=[0,0,1,0.96])
            buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, bbox_inches="tight"); buf.seek(0); plt.close(fig)
            return Image.open(buf)


        def create_stats_visualization(stats):
            """Create visualization for dataset statistics"""
            if not stats or "error" in stats:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Dataset Overview', fontsize=16)
            
            # 1. Action distribution pie chart
            ax = axes[0, 0]
            if 'action_distribution' in stats:
                actions = list(stats['action_distribution'].keys())
                counts = list(stats['action_distribution'].values())
                try:
                    # 如果 stats 里已经是 name->count，就直接用 keys
                    if all(not str(k).isdigit() for k in actions):
                        action_names = actions
                    else:
                        # id->name 的映射：如果 data_manager.current_meta 有 action_names / action_map，cohort 会按 ID 统计
                        meta = getattr(__import__('data_manager'), 'data_manager').get_current_meta()
                        id2name = meta.get('action_map') or {i: n for i, n in enumerate(meta.get('action_names', []))}
                        action_names = [id2name.get(int(a), str(a)) for a in actions]
                except Exception:
                    action_names = [str(a) for a in actions]

                ax.pie(counts, labels=action_names, autopct='%1.1f%%', startangle=90)
                ax.set_title('Treatment Distribution')
            
            # 2. Key metrics bar chart
            ax = axes[0, 1]
            metrics = {
                'Total Patients': stats.get('total_patients', 0),
                'Total Records': stats.get('total_records', 0) / 100,  # Scale down
                'Avg Trajectory': stats.get('avg_trajectory_length', 0),
                'Avg Reward': (stats.get('avg_reward', 0) + 5) * 10  # Scale up
            }
            ax.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel('Value')
            ax.set_title('Key Metrics')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 3. Feature statistics heatmap
            ax = axes[1, 0]
            if 'feature_stats' in stats:
                features = list(stats['feature_stats'].keys())[:8]  # Limit to 8 features
                metrics = ['mean', 'std', 'min', 'max']
                data = []
                for metric in metrics:
                    row = [stats['feature_stats'][f].get(metric, 0) for f in features]
                    data.append(row)
                
                im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(features)))
                ax.set_yticks(range(len(metrics)))
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.set_yticklabels(metrics)
                ax.set_title('Feature Statistics')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Value')
            
            # 4. Summary text
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"""Dataset Summary:
            
        - Total Patients: {stats.get('total_patients', 0)}
        - Total Records: {stats.get('total_records', 0)}
        - Avg Trajectory Length: {stats.get('avg_trajectory_length', 0):.1f}
        - Avg Reward: {stats.get('avg_reward', 0):.3f}

        Data Source: {data_manager.current_source}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
            
            ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return Image.open(buf)

        def _on_change_source(src):
            if src == "Virtual Data":
                return gr.update(visible=True), gr.update(visible=False), "Current: Virtual Data"
            else:
                return gr.update(visible=False), gr.update(visible=True), "Current: Real Data"

        data_source_radio.change(
            fn=_on_change_source,
            inputs=[data_source_radio],
            outputs=[virtual_data_options, real_data_options, current_source_text]
        )

        # 2) 生成虚拟数据
        def _on_generate_virtual(n_pat):
            stats = data_manager.generate_virtual_data(n_patients=int(n_pat))
            plist = get_patient_list()
            img = create_dataset_stats_image(get_cohort_stats())

            # 构建动作图例
            from drive_tools import get_action_legend_html
            legend_html = get_action_legend_html()
            from drive_tools import get_action_legend_html
            legend_html = get_action_legend_html()
            return (
                gr.update(choices=plist, value=(plist[0] if plist else None)),
                img,
                gr.update(value=legend_html, visible=bool(legend_html)),
                f"Loaded {len(plist)} patients from Real Data"
            )


        # 3) 加载真实数据
        def _on_load_real(data_file, schema_file):
            # 真正加载：drive_tools.load_data_source 会同步 meta
            path = data_file.name if data_file else None
            schema = schema_file.name if schema_file else None

            # 正确的调用方式：明确 source_type="real"
            res = load_data_source("real", file_path=path, schema_path=schema)

            plist = get_patient_list()
            patient_choices = plist.get("patients", []) if isinstance(plist, dict) else plist

            img = create_dataset_stats_image(get_cohort_stats())
            return (
                gr.update(choices=patient_choices, value=(patient_choices[0] if patient_choices else None)),
                img,
                f"Loaded {len(patient_choices)} patients from Real Data"
            )

        # 4) 刷新患者列表
        def _on_refresh_patients():
            plist = get_patient_list()
            patient_choices = plist.get("patients", []) if isinstance(plist, dict) else plist
            return gr.update(choices=patient_choices)

        refresh_patients_btn.click(
            fn=_on_refresh_patients,
            inputs=[],
            outputs=[patient_dropdown]
        )

        # 5) 选择患者 -> 刷新顶部“Active Patient”和左侧基本信息图
        def _on_select_patient(pid):
            if not pid:
                return "No patient selected", None
            info = get_patient_data(pid)
            fig = create_patient_visualization(info)
            return f"Active: {pid}", fig

        patient_dropdown.change(
            fn=_on_select_patient,
            inputs=[patient_dropdown],
            outputs=[active_patient_display, patient_info_display]
        )

        def create_patient_visualization(patient_info):
            """Create visualization for patient information"""
            if not patient_info or "error" in patient_info:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(f'Patient {patient_info.get("patient_id", "Unknown")} Overview', fontsize=14)
            
            # 1. Current vital signs
            ax = axes[0, 0]
            current_state = patient_info.get('current_state', {})
            vitals = ['blood_pressure', 'heart_rate', 'glucose', 'oxygen_saturation']
            values = [current_state.get(v, 0.5) for v in vitals]
            colors = ['red' if abs(v - 0.5) > 0.2 else 'green' for v in values]
            
            bars = ax.bar(range(len(vitals)), values, color=colors, alpha=0.7)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Normal')
            ax.set_xticks(range(len(vitals)))
            ax.set_xticklabels(['BP', 'HR', 'Glucose', 'O2 Sat'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Normalized Value')
            ax.set_title('Current Vital Signs')
            ax.legend()
            
            # 2. Treatment history
            ax = axes[0, 1]
            treatment_history = patient_info.get('treatment_history', [])[-20:]  # Last 20
            if treatment_history:
                ax.plot(treatment_history, 'o-', markersize=6)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Treatment')
                ax.set_yticks([0, 1, 2, 3, 4])
                ax.set_yticklabels(['Med A', 'Med B', 'Med C', 'Placebo', 'Combo'])
                ax.set_title('Recent Treatment History')
                ax.grid(True, alpha=0.3)
            
            # 3. Outcome history
            ax = axes[1, 0]
            outcome_history = patient_info.get('outcome_history', [])[-20:]  # Last 20
            if outcome_history:
                ax.plot(outcome_history, 'b-', linewidth=2)
                ax.fill_between(range(len(outcome_history)), outcome_history, alpha=0.3)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Reward/Outcome')
                ax.set_title('Outcome Trajectory')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 4. Patient details
            ax = axes[1, 1]
            ax.axis('off')
            details_text = f"""Patient Details:
            
        - ID: {patient_info.get('patient_id', 'Unknown')}
        - Age: {int(current_state.get('age', 45))} years
        - Gender: {'Male' if current_state.get('gender', 0) == 1 else 'Female'}
        - Total Records: {patient_info.get('total_records', 0)}
        - Current Step: {current_state.get('timestep', 0)}
        - Last Action: {current_state.get('last_action', -1)}
        - Last Reward: {current_state.get('last_reward', 0):.3f}"""
            
            ax.text(0.1, 0.5, details_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            return fig

        # def create_analysis_visualization(analysis):
        #     """Create visualization for treatment analysis"""
        #     if not analysis or "error" in analysis:
        #         return None
            
        #     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        #     fig.suptitle('Treatment Analysis', fontsize=14)
            
        #     # 1. Treatment comparison
        #     ax = axes[0, 0]
        #     if 'all_options' in analysis and 'action_values' in analysis['all_options']:
        #         actions = [av['action'] for av in analysis['all_options']['action_values']]
        #         q_values = [av['q_value'] for av in analysis['all_options']['action_values']]
        #         colors = ['red' if av['action'] == analysis['recommendation'].get('recommended_action', '') else 'blue' 
        #                 for av in analysis['all_options']['action_values']]
                
        #         bars = ax.bar(range(len(actions)), q_values, color=colors, alpha=0.7)
        #         ax.set_xticks(range(len(actions)))
        #         ax.set_xticklabels([a.replace('Medication', 'Med') for a in actions], rotation=45, ha='right')
        #         ax.set_ylabel('Q-Value')
        #         ax.set_title('Treatment Options Comparison')
                
        #         # Highlight recommended
        #         if 'recommendation' in analysis:
        #             ax.text(0.02, 0.98, f"Recommended: {analysis['recommendation'].get('recommended_action', 'Unknown')}", 
        #                     transform=ax.transAxes, verticalalignment='top',
        #                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
        #     # 2. Predicted trajectory preview
        #     ax = axes[0, 1]
        #     if 'predicted_trajectory' in analysis and 'trajectory' in analysis['predicted_trajectory']:
        #         trajectory = analysis['predicted_trajectory']['trajectory'][:7]  # 7 days
        #         if trajectory:
        #             # Extract key vitals
        #             steps = [t['step'] for t in trajectory]
        #             glucose = [t['state'].get('glucose', 0.5) for t in trajectory]
        #             o2_sat = [t['state'].get('oxygen_saturation', 0.95) for t in trajectory]
                    
        #             ax2 = ax.twinx()
        #             line1 = ax.plot(steps, glucose, 'b-o', label='Glucose', markersize=6)
        #             line2 = ax2.plot(steps, o2_sat, 'r-s', label='O2 Sat', markersize=6)
                    
        #             ax.set_xlabel('Days')
        #             ax.set_ylabel('Glucose Level', color='b')
        #             ax2.set_ylabel('O2 Saturation', color='r')
        #             ax.tick_params(axis='y', labelcolor='b')
        #             ax2.tick_params(axis='y', labelcolor='r')
        #             ax.set_title('7-Day Predicted Trajectory')
        #             ax.grid(True, alpha=0.3)
                    
        #             # Combine legends
        #             lines = line1 + line2
        #             labels = [l.get_label() for l in lines]
        #             ax.legend(lines, labels, loc='best')
            
        #     # 3. Confidence visualization
        #     ax = axes[1, 0]
        #     ax.text(0.5, 0.5, 'Confidence Analysis\n(Placeholder for uncertainty visualization)', 
        #             ha='center', va='center', fontsize=12)
        #     ax.set_xlim(0, 1)
        #     ax.set_ylim(0, 1)
        #     ax.set_title('Model Confidence')
            
        #     # 4. Summary
        #     ax = axes[1, 1]
        #     ax.axis('off')
        #     summary_text = f"""Analysis Summary:
            
        # - Recommended: {analysis.get('recommendation', {}).get('recommended_action', 'Unknown')}
        # - Confidence: {analysis.get('recommendation', {}).get('confidence', 0):.3f}
        # - Analysis Time: {analysis.get('analysis_timestamp', 'Unknown')}

        # Key Insights:
        # - Best treatment based on long-term outcomes
        # - Prediction covers next 7 days
        # - Model confidence is based on Q-value differences"""
            
        #     ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
        #             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
        #     plt.tight_layout()
            
        #     # Convert to image
        #     buf = io.BytesIO()
        #     plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        #     buf.seek(0)
        #     plt.close()
            
        #     return Image.open(buf)

        def generate_report_for_ui(patient_id):
            """
            返回 (markdown_text, PIL.Image 或 None)
            """
            if not patient_id:
                return "No patient selected", None
            try:
                md, img = generate_patient_report_ui(patient_id, topk=3)
                return md, img
            except Exception as e:
                return f"Error generating report: {e}", None


        def create_analysis_visualization(analysis):
            """Create visualization for treatment analysis (robust & returns PIL.Image)"""
            if not analysis or "error" in analysis:
                return None

            import io, math
            from PIL import Image
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('Treatment Analysis', fontsize=14)

            all_opts = (analysis.get('all_options') or {})
            avs = all_opts.get('action_values') or []
            action_catalog = {av.get('action_id'): av.get('action') for av in avs}
            rec = analysis.get('recommendation', {}) if isinstance(analysis, dict) else {}
            act = rec.get('recommended_action')
            rt = rec.get('recommended_treatment')
            if isinstance(rt, str) and rt.isdigit():
                rec_name = action_catalog.get(int(rt), f"Action {rt}")
            elif rt:
                rec_name = str(rt)
            elif isinstance(act, (int, float)):
                rec_name = action_catalog.get(int(act), f"Action {int(act)}")
            elif isinstance(act, str):
                rec_name = act
            else:
                rec_name = 'Unknown'
            rec_idx = act

            # 1) Treatment comparison
            ax = axes[0, 0]
            if avs:
                actions, q_values, colors = [], [], []
                for av in avs:
                    a = av.get('action', '')
                    q = av.get('q_value', 0.0)
                    try:
                        q = float(q)
                        if math.isnan(q) or math.isinf(q):
                            q = 0.0
                    except Exception:
                        q = 0.0
                    actions.append(str(a))
                    q_values.append(q)
                    if ((rec_idx is not None and av.get('action_id') == rec_idx) or
                            (rec_name and a == rec_name)):
                        colors.append('red')
                    else:
                        colors.append('blue')

                ax.bar(range(len(actions)), q_values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(actions)))
                ax.set_xticklabels([a.replace('Medication', 'Med') for a in actions],
                                rotation=45, ha='right')
                ax.set_ylabel('Q-Value')
                ax.set_title('Treatment Options Comparison')

                if rec_name or rec_idx is not None:
                    label = rec_name if rec_name else str(rec_idx)
                    ax.text(0.02, 0.98, f"Recommended: {label}",
                            transform=ax.transAxes, va='top',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No action values available',
                        ha='center', va='center', fontsize=12, alpha=0.8)

            # 2) Predicted trajectory preview (robust)
            ax = axes[0, 1]
            pred = analysis.get('predicted_trajectory') or {}
            traj = pred.get('trajectory') or []
            traj = traj[:7]  # show next 7 steps
            if traj:
                steps, glucose, o2_sat = [], [], []
                last_g, last_o = 0.5, 0.95

                def clean(val, last):
                    try:
                        v = float(val)
                        if math.isnan(v) or math.isinf(v):
                            return last
                        return v
                    except Exception:
                        return last

                for i, t in enumerate(traj):
                    step = t.get('step', i)
                    try:
                        step = int(step)
                    except Exception:
                        step = i
                    steps.append(step)

                    st = t.get('state') or {}
                    g = clean(st.get('glucose', last_g), last_g); last_g = g
                    o = clean(st.get('oxygen_saturation', last_o), last_o); last_o = o
                    glucose.append(g); o2_sat.append(o)

                ax2 = ax.twinx()
                l1, = ax.plot(steps, glucose, 'o-', label='Glucose', markersize=6)
                l2, = ax2.plot(steps, o2_sat, 's--', label='O2 Sat', markersize=6)

                # If series are in [0,1], clamp for nicer view
                if all(0.0 <= v <= 1.0 for v in glucose): ax.set_ylim(0, 1)
                if all(0.0 <= v <= 1.0 for v in o2_sat): ax2.set_ylim(0, 1)

                ax.set_xlabel('Days')
                ax.set_ylabel('Glucose Level')
                ax2.set_ylabel('O2 Saturation')
                ax.set_title('7-Day Predicted Trajectory')
                ax.grid(True, alpha=0.3)
                ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='best')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No trajectory available',
                        ha='center', va='center', fontsize=12, alpha=0.8)

            # 3) Confidence visualization
            ax = axes[1, 0]
            ax.text(0.5, 0.5, 'Confidence Analysis\n(Placeholder)',
                    ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title('Model Confidence')

            # 4) Summary
            ax = axes[1, 1]
            ax.axis('off')
            conf = analysis.get('recommendation', {}).get('confidence', 0.0)
            ts = analysis.get('analysis_timestamp', 'Unknown')

            summary_text = f"""Analysis Summary:

        - Recommended: {rec_name}
        - Confidence: {conf:.3f}
        - Analysis Time: {ts}

        Key Insights:
        - Best treatment based on long-term outcomes
        - Prediction covers next 7 days
        - Model confidence is based on Q-value differences"""
            ax.text(0.1, 0.5, summary_text, fontsize=11, va='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            # === return PIL.Image (what Gradio Image expects) ===
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return Image.open(buf)


        # Event handlers for Data Management
        def toggle_data_options(source):
            if source == "Virtual Data":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
                
        def generate_virtual_data(n_patients):
            from data_manager import data_manager
            from drive_tools import get_cohort_stats, get_action_legend_html
            try:
                data_manager.generate_virtual_data(n_patients=int(n_patients))
                stats = get_cohort_stats()
                img = create_dataset_stats_image(stats)
                legend_html = get_action_legend_html()

                lst = get_patient_list()
                choices = lst.get("patients", []) if isinstance(lst, dict) else lst

                # 返回顺序必须和 outputs 对齐：
                # [current_source_text, stats_display, action_legend, patient_dropdown]
                return (
                    f"Loaded {len(choices)} patients (virtual)",
                    img,
                    gr.update(value=legend_html, visible=bool(legend_html)),
                    gr.update(choices=choices, value=(choices[0] if choices else None)),
                )
            except Exception as e:
                # 占位图片，防止 UI 崩
                import matplotlib.pyplot as plt, io
                from PIL import Image
                fig, ax = plt.subplots(1,1,figsize=(6,3)); ax.axis("off")
                ax.text(0.5,0.5,f"Error: {e}",ha="center",va="center"); buf=io.BytesIO()
                plt.tight_layout(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight"); buf.seek(0); plt.close(fig)
                return (
                    f"❌ Generate error: {e}",
                    Image.open(buf),
                    gr.update(value="", visible=False),
                    gr.update(),  # 下拉不更新
                )


        def load_real_data(file, schema_file):
            from drive_tools import load_data_source, get_cohort_stats, get_action_legend_html
            try:
                if file is None:
                    return "Please upload a data file", None, gr.update(value="", visible=False), gr.update()

                schema_path = schema_file.name if schema_file else None
                res = load_data_source("real", file_path=file.name, schema_path=schema_path)

                stats = get_cohort_stats()
                img = create_dataset_stats_image(stats)
                legend_html = get_action_legend_html()

                lst = get_patient_list()
                choices = lst.get("patients", []) if isinstance(lst, dict) else lst
                msg = f"Loaded {res.get('patients', len(choices))} patients ({res.get('records','?')} records)" if isinstance(res, dict) else "Loaded"

                return (
                    msg,
                    img,
                    gr.update(value=legend_html, visible=bool(legend_html)),
                    gr.update(choices=choices, value=(choices[0] if choices else None)),
                )
            except Exception as e:
                import matplotlib.pyplot as plt, io
                from PIL import Image
                fig, ax = plt.subplots(1,1,figsize=(6,3)); ax.axis("off")
                ax.text(0.5,0.5,f"Error: {e}",ha="center",va="center"); buf=io.BytesIO()
                plt.tight_layout(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight"); buf.seek(0); plt.close(fig)
                return (
                    f"❌ Load error: {e}",
                    Image.open(buf),
                    gr.update(value="", visible=False),
                    gr.update(),
                )

        
        def refresh_patient_list():
            patients = get_patient_list()
            if "error" not in patients:
                patient_choices = patients.get("patients", [])
                return gr.update(choices=patient_choices, value=patient_choices[0] if patient_choices else None)
            return gr.update()
        
        def display_patient_info(patient_id):
            if not patient_id:
                return None, None, "No patient selected"
            
            patient_info = get_patient_data(patient_id)
            if "error" in patient_info:
                return None, None, patient_id
            
            patient_viz = create_patient_visualization(patient_info)
            return patient_viz, None, f"Patient: {patient_id}"
        
        def analyze_patient_fn(patient_id):
            if not patient_id:
                return None
            
            analysis = analyze_patient(patient_id)
            analysis_viz = create_analysis_visualization(analysis)
            return analysis_viz
        
        def export_patient_fn(patient_id):
            if not patient_id:
                return None
            
            try:
                output_path = f"patient_{patient_id}_export.json"
                data_manager.export_patient_data(patient_id, output_path)
                return gr.update(value=output_path, visible=True)
            except Exception as e:
                gr.Warning(f"Export failed: {e}")
                return None
        
        # Updated chat function to use active patient
        def user_message_with_context(message, history, patient_id):
            if patient_id:
                context_msg = f"[当前患者: {patient_id}] {message}"
            else:
                context_msg = message
            # Use messages format
            history = history + [{"role": "user", "content": message}]
            return "", history
        
        def bot_response_with_context(history, patient_id):
            if history and history[-1]["role"] == "user":
                user_msg = history[-1]["content"]
                
                # If patient is selected, add context
                if patient_id:
                    patient_state = data_manager.get_patient_state(patient_id)
                    context = f"当前患者ID: {patient_id}, 状态: {json.dumps(patient_state, ensure_ascii=False)}\n问题: {user_msg}"
                    response = chat_function(context, history[:-1])
                else:
                    response = chat_function(user_msg, history[:-1])
                
                # Add assistant response
                history = history + [{"role": "assistant", "content": response}]
            return history

        def create_stats_visualization(stats_dict):

            if not stats_dict or "error" in stats_dict:
                return None, []

            # 兼容嵌套结构
            al = stats_dict.get('active_learning', {}) if isinstance(stats_dict, dict) else {}
            hyp = stats_dict.get('current_hyperparams', {}) if isinstance(stats_dict, dict) else {}

            # 顶层或嵌套取值（有则用，没有则回退）
            query_rate = al.get('query_rate', stats_dict.get('query_rate', 0) if isinstance(stats_dict, dict) else 0) * 100
            labeled_size = stats_dict.get('labeled_buffer_size', al.get('labeled_buffer_size', 0)) if isinstance(stats_dict, dict) else al.get('labeled_buffer_size', 0)
            weak_size = stats_dict.get('weak_buffer_size', al.get('weak_buffer_size', 0)) if isinstance(stats_dict, dict) else al.get('weak_buffer_size', 0)
            query_size = stats_dict.get('query_buffer_size', al.get('query_buffer_size', 0)) if isinstance(stats_dict, dict) else al.get('query_buffer_size', 0)

            avg_queried = al.get('avg_queried_uncertainty', stats_dict.get('avg_queried_uncertainty', 0) if isinstance(stats_dict, dict) else 0)
            avg_rejected = al.get('avg_rejected_uncertainty', stats_dict.get('avg_rejected_uncertainty', 0) if isinstance(stats_dict, dict) else 0)

            threshold = al.get('current_threshold', stats_dict.get('current_threshold', hyp.get('tau', 0.05)) if isinstance(stats_dict, dict) else hyp.get('tau', 0.05))

            total_seen = stats_dict.get('total_transitions', al.get('total_transitions', 0)) if isinstance(stats_dict, dict) else al.get('total_transitions', 0)
            total_queries = stats_dict.get('total_queries', al.get('total_queries', 0)) if isinstance(stats_dict, dict) else al.get('total_queries', 0)
            total_updates = stats_dict.get('total_updates', 0) if isinstance(stats_dict, dict) else 0

            # 创建图表
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # 1. 查询率对比
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.bar(['Current', 'Target'], [query_rate, 15], color=['#3498db', '#95a5a6'])
            ax1.set_ylabel('Query Rate (%)')
            ax1.set_title('Query Rate vs Target')
            ax1.set_ylim(0, 100)
            for i, v in enumerate([query_rate, 15]):
                ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

            # 2. 缓冲区分布
            ax2 = fig.add_subplot(gs[0, 1])
            sizes = [labeled_size or 0, weak_size or 0, query_size or 0]
            labels = ['Labeled', 'Weak', 'Query']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            if sum(sizes) > 0:
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
                ax2.set_title('Buffer Distribution')
            else:
                ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Buffer Distribution')

            # 3. 不确定性分布
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.bar(['Queried', 'Rejected'], [avg_queried, avg_rejected], color=['#e74c3c', '#3498db'])
            ax3.axhline(y=threshold, color='#2c3e50', linestyle='--', label=f'Threshold: {threshold:.3f}')
            ax3.set_ylabel('Uncertainty')
            ax3.set_title('Uncertainty Analysis')
            ax3.legend()

            # 4. 关键指标概览
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            stats_text = f"""Key Metrics:
            
        • Total Samples: {total_seen:,}
        • Total Queries: {total_queries:,}
        • Query Rate: {query_rate:.1f}%
        • Total Updates: {total_updates:,}
        • Labeling Reduction: {max(0.0, 100 - query_rate):.1f}%

        Performance:
        • Avg Uncertainty: {avg_queried:.4f}
        • Current τ: {threshold:.3f}"""
            ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle('Active Learning Statistics Dashboard', fontsize=14, fontweight='bold')

            table_data = [
                ["Total Transitions", f"{total_seen:,}"],
                ["Query Rate", f"{query_rate:.2f}%"],
                ["Labeled Buffer", f"{(labeled_size or 0):,}"],
                ["Weak Buffer", f"{(weak_size or 0):,}"],
                ["Query Buffer", f"{(query_size or 0):,}"],
                ["Avg Uncertainty", f"{avg_queried:.4f}"],
                ["Threshold (τ)", f"{threshold:.3f}"],
                ["Total Updates", f"{total_updates:,}"],
                ["Labeling Reduction", f"{max(0.0, 100 - query_rate):.1f}%"]
            ]

            return fig, table_data

                # Online Learning Monitor events
        def refresh_online_stats():
            """Refresh online learning statistics"""
            from drive_tools import get_online_stats
            stats = get_online_stats()
            
            if "error" in stats:
                gr.Warning(f"Error getting stats: {stats['error']}")
                return 0, 0, 0, 0, 0.05, 0, gr.update(visible=False), []  # hide plot when error
            
            # Extract key metrics
            total_trans = stats.get('total_transitions', 0)
            query_rate = stats.get('active_learning', {}).get('query_rate', 0) * 100
            buffer_size = stats.get('labeled_buffer_size', 0)
            avg_uncertainty = stats.get('active_learning', {}).get('avg_queried_uncertainty', 0)
            current_tau = stats.get('current_hyperparams', {}).get('tau', 0.05)
            updates = stats.get('total_updates', 0)
            
            # 创建可视化
            fig, table_data = create_stats_visualization(stats)
            
            return (
                total_trans,
                round(query_rate, 2),
                buffer_size,
                round(avg_uncertainty, 4),
                current_tau,
                updates,
                fig,  # 返回图表而不是JSON
                table_data  # 返回表格数据
            )
        
        def pause_online_training():
            """Pause online training"""
            from drive_tools import pause_online_training
            result = pause_online_training()
            if "error" in result:
                gr.Warning(f"Error: {result['error']}")
            else:
                gr.Info("Online training paused")
        
        def resume_online_training():
            """Resume online training"""
            from drive_tools import resume_online_training
            result = resume_online_training()
            if "error" in result:
                gr.Warning(f"Error: {result['error']}")
            else:
                gr.Info("Online training resumed")
        
        def update_tau_threshold(new_tau):
            """Update active learning threshold"""
            from drive_tools import update_hyperparams
            result = update_hyperparams({"tau": new_tau})
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            return f"✅ Threshold updated to {new_tau}. {result.get('message', '')}"
        def run_system_evaluation(duration_seconds=60):
            """运行增强的系统评估 - 使用 run_complete_evaluation 的逻辑"""
            try:
                from drive_tools import _online_system
                
                if not _online_system:
                    return "❌ Online system not initialized. Please wait for system to start.", None
                
                # 使用 run_enhanced_evaluation 函数
                compliance_results = run_enhanced_evaluation(duration_seconds)
                
                if not compliance_results:
                    return "❌ Evaluation failed. Please check system status.", None
                
                # 生成文本报告
                report_text = "# 📊 Paper Compliance Evaluation Report\n\n"
                report_text += f"**Duration**: {duration_seconds} seconds\n"
                report_text += f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # 添加整体得分
                overall_score = np.mean([r.get('score', 0) for r in compliance_results.values()])
                passed_count = sum(1 for r in compliance_results.values() if r.get('passed', False))
                total_count = len(compliance_results)
                
                report_text += f"## 📈 Overall Performance\n"
                report_text += f"- **Overall Score**: {overall_score:.1%}\n"
                report_text += f"- **Tests Passed**: {passed_count}/{total_count}\n"
                report_text += f"- **Grade**: {'A' if overall_score > 0.9 else 'B' if overall_score > 0.8 else 'C' if overall_score > 0.7 else 'F'}\n\n"
                
                # 详细结果
                report_text += "## 📋 Detailed Results\n\n"
                for metric, result in compliance_results.items():
                    status = "✅" if result.get('passed', False) else "❌"
                    report_text += f"### {metric.replace('_', ' ').title()}\n"
                    report_text += f"- **Value**: {result.get('value', 0):.4f}\n"
                    report_text += f"- **Target**: {result.get('target', 0):.4f}\n"
                    report_text += f"- **Status**: {status}\n"
                    report_text += f"- **Score**: {result.get('score', 0):.2%}\n\n"
                
                # 如果有评估指标图片，返回它
                if os.path.exists('evaluation_metrics.png'):
                    from PIL import Image
                    metrics_plot_image = Image.open('evaluation_metrics.png')
                else:
                    metrics_plot_image = None
                
                return report_text, metrics_plot_image
                
            except Exception as e:
                import traceback
                error_msg = f"❌ Evaluation error: {str(e)}\n{traceback.format_exc()}"
                return error_msg, None

        def download_evaluation_report():
            """下载评估报告"""
            try:
                # 运行评估并保存
                report_text, _ = run_system_evaluation()
                
                # 保存报告
                filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(filename, 'w') as f:
                    f.write(report_text)
                
                return gr.update(value=filename, visible=True)
            except Exception as e:
                gr.Warning(f"Failed to generate report: {e}")
                return gr.update(visible=False)        
        # Connect event handlers
        
        # Data Management events
        data_source_radio.change(
            toggle_data_options,
            inputs=[data_source_radio],
            outputs=[virtual_data_options, real_data_options]
        )
        
        generate_btn.click(
            fn=generate_virtual_data,
            inputs=[n_patients_slider],
            outputs=[current_source_text, stats_display, action_legend, patient_dropdown]  # ← 顺序固定
        )


        generate_report_btn.click(
            fn=_on_generate_report,
            inputs=[patient_dropdown],
            outputs=[patient_report_html, patient_analysis_display, report_download]
        )


        load_real_btn.click(
            fn=load_real_data,
            inputs=[file_upload, schema_upload],
            outputs=[current_source_text, stats_display, action_legend, patient_dropdown]  # ← 顺序固定
        )

        
        refresh_patients_btn.click(
            refresh_patient_list,
            outputs=[patient_dropdown]
        )
        
        patient_dropdown.change(
            display_patient_info,
            inputs=[patient_dropdown],
            outputs=[patient_info_display, patient_analysis_display, active_patient_display]
        ).then(
            lambda x: x,
            inputs=[patient_dropdown],
            outputs=[active_patient_id]
        )
        
        # analyze_btn.click(
        #     analyze_patient_fn,
        #     inputs=[patient_dropdown],
        #     outputs=[patient_analysis_display]
        # )
        
        export_btn.click(
            export_patient_fn,
            inputs=[patient_dropdown],
            outputs=[export_output]
        )
        
        # Chat events with patient context
        msg.submit(
            user_message_with_context, 
            [msg, chatbot, active_patient_id], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot_response_with_context, 
            [chatbot, active_patient_id], 
            chatbot
        )
        
        submit.click(
            user_message_with_context, 
            [msg, chatbot, active_patient_id], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot_response_with_context, 
            [chatbot, active_patient_id], 
            chatbot
        )
        
        clear.click(lambda: [], None, chatbot, queue=False)
        
        # Parameter control events
        preset_dropdown.change(
            update_preset,
            inputs=[preset_dropdown],
            outputs=[
                alpha_slider, gamma_slider, lr_slider,
                reg_slider, batch_slider, epoch_slider,
                custom_params
            ]
        )
        
        recommend_btn.click(
            recommend_config,
            outputs=[recommendations_display]
        )
        
        apply_btn.click(
            apply_config,
            inputs=[
                alpha_slider, gamma_slider, lr_slider,
                reg_slider, batch_slider, epoch_slider,
                confirm_apply
            ],
            outputs=[apply_output]
        )
        
        retrain_btn.click(
            retrain_with_params,
            inputs=[
                preset_dropdown, alpha_slider, gamma_slider,
                lr_slider, reg_slider, batch_slider, 
                epoch_slider, confirm_retrain
            ],
            outputs=[retrain_output]
        )
        
        # Parameter info buttons
        alpha_info.click(
            lambda: gr.Info(get_param_description("alpha")),
            outputs=[]
        )
        gamma_info.click(
            lambda: gr.Info(get_param_description("gamma")),
            outputs=[]
        )
        lr_info.click(
            lambda: gr.Info(get_param_description("learning_rate")),
            outputs=[]
        )
        reg_info.click(
            lambda: gr.Info(get_param_description("regularization_weight")),
            outputs=[]
        )
        batch_info.click(
            lambda: gr.Info(get_param_description("batch_size")),
            outputs=[]
        )
        epoch_info.click(
            lambda: gr.Info(get_param_description("n_epochs")),
            outputs=[]
        )
        
        # Connect Online Learning Monitor events
        refresh_stats_btn.click(
            refresh_online_stats,
            outputs=[
                total_transitions,
                query_rate,
                buffer_size,
                avg_uncertainty,
                current_tau,
                training_updates,
                al_stats_plot,  # 改为plot
                al_stats_table  # 添加table
            ]
        )
        
        pause_btn.click(pause_online_training)
        resume_btn.click(resume_online_training)
        
        def run_custom_evaluation(duration, scenario):
            """运行自定义评估"""
            # 根据场景调整参数
            if scenario == "Quick Test":
                actual_duration = min(duration, 30)
            elif scenario == "Full Compliance Check":
                actual_duration = max(duration, 300)
            else:
                actual_duration = duration
            
            return run_system_evaluation(actual_duration)
        
        start_eval_btn.click(
            run_custom_evaluation,
            inputs=[eval_duration_slider, eval_scenario_dropdown],
            outputs=[evaluation_text, evaluation_plot]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)),
            outputs=[evaluation_text, evaluation_plot, download_report_btn]
        )

        
        # 健康检查按钮处理
        # def run_health_check_ui():
        #     """运行系统健康检查并返回结果"""
        #     try:
        #         from drive_tools import _online_system
                
        #         if not _online_system:
        #             return {"error": "System not initialized"}
                
        #         checker = SystemHealthChecker(_online_system)
        #         results = checker.run_all_checks()
                
        #         # 格式化结果
        #         formatted_results = {
        #             "overall_status": "✅ Healthy" if all(r['passed'] for r in results.values()) else "⚠️ Issues Detected",
        #             "checks": results,
        #             "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #         }
                
        #         return formatted_results
                
        #     except Exception as e:
        #         return {"error": str(e)}
        
        # health_check_btn.click(
        #     run_health_check_ui,
        #     outputs=[health_check_output]
        # )
        def update_live_monitoring():
            """更新实时监控指标"""
            try:
                from drive_tools import _online_system, get_response_time_stats
                
                if not _online_system:
                    return 0, 0, 0, "Not Started", 0, "System Not Initialized"
                
                # 获取实时统计
                stats = _online_system['trainer'].get_statistics()
                al_stats = _online_system['active_learner'].get_statistics()
                response_stats = get_response_time_stats()
                
                # 计算实时指标
                query_rate = al_stats.get('query_rate', 0) * 100
                recent_transitions = stats.get('total_transitions', 0)
                throughput = recent_transitions / max(1, stats.get('total_updates', 1))
                avg_latency = response_stats.get('avg_response_time', 0) * 1000 if response_stats else 0
                
                shift_status = "Normal"
                safety_score = 95.0
                
                if throughput > 5 and query_rate < 30 and avg_latency < 100:
                    health_status = "✅ Healthy"
                elif throughput > 0:
                    health_status = "⚠️ Degraded"
                else:
                    health_status = "❌ Critical"
                
                return query_rate, throughput, avg_latency, shift_status, safety_score, health_status
                
            except Exception as e:
                return 0, 0, 0, "Error", 0, f"Error: {str(e)}"
        
        # refresh_live_monitoring_btn.click(
        #     update_live_monitoring,
        #     outputs=[
        #         live_query_rate,
        #         live_throughput,
        #         live_latency,
        #         distribution_shift_indicator,
        #         safety_compliance,
        #         system_health
        #     ]
        # )
      
        # 分布偏移触发按钮处理
        # def trigger_shift_ui():
        #     """触发分布偏移"""
        #     try:
        #         trigger_distribution_shift_test()
        #         return "✅ Distribution shift triggered. Monitor the system response."
        #     except Exception as e:
        #         return f"❌ Error: {str(e)}"
        
        # trigger_shift_btn.click(
        #     trigger_shift_ui,
        #     outputs=[shift_status]
        # )

        def interpret_reward_value(reward_value):
            """解释reward值的含义"""
            if reward_value >= 3:
                return "🎉 Excellent outcome - Significant improvement"
            elif reward_value >= 1:
                return "✅ Good outcome - Moderate improvement"
            elif reward_value >= -1:
                return "➖ Neutral outcome - Stable condition"
            elif reward_value >= -3:
                return "⚠️ Poor outcome - Moderate deterioration"
            else:
                return "❌ Critical outcome - Severe deterioration"
        
        # 更新reward滑块的解释
        expert_reward_slider.change(
            interpret_reward_value,
            inputs=[expert_reward_slider],
            outputs=[reward_interpretation]
        )
        
        # 专家模式切换
        def change_expert_mode(mode):
            """切换专家模式（稳健导入）"""
            try:
                import drive_tools as dt
                set_fn = getattr(dt, "set_expert_mode", None)
                mode_map = {
                    "Automatic Simulation": "automatic",
                    "Manual Expert Input": "manual"
                }
                if set_fn is not None:
                    result = set_fn(mode_map[mode])
                else:
                    # 后备：直接改在线系统里的 expert 标志，至少不报错
                    if getattr(dt, "_online_system", None) and 'expert' in dt._online_system:
                        dt._online_system['expert'].manual_mode = (mode == "Manual Expert Input")
                        result = {"status": "success"}
                    else:
                        result = {"error": "set_expert_mode not available"}
            except Exception as e:
                result = {"error": str(e)}

            is_manual = (mode == "Manual Expert Input")
            return (
                gr.update(interactive=is_manual),
                gr.update(interactive=is_manual),
                gr.update(interactive=is_manual),
                f"Mode changed to: {mode}" if "error" not in result else f"Error: {result['error']}"
            )


        def fetch_next_case():
            """获取下一个需要审核的案例（兼容 drive_tools 不同命名 & 稳定返回类型）"""
            # —— 惰性导入 + 兼容函数名 —— #
            try:
                import drive_tools as dt
            except Exception as e:
                print("[ExpertUI] drive_tools import error:", e)
                return {}, "0", "System not initialized"

            fn_case = getattr(dt, "get_next_expert_case", None) \
                    or getattr(dt, "get_next_case", None) \
                    or getattr(dt, "pop_next_case", None)
            fn_stats = getattr(dt, "get_expert_stats", None) \
                    or getattr(dt, "get_stats", None)

            if fn_case is None or fn_stats is None:
                # 不抛异常到 UI，返回占位值
                return {}, "0", "API not found in drive_tools"

            # —— 调用并兜底 —— #
            try:
                case = fn_case()
                stats = fn_stats()
            except Exception as e:
                print(f"[ExpertUI] An exception occurred while fetching data: {e}")
                return {}, "0", "Error fetching case. Please check logs."

            # —— JSON 序列化（包含 numpy 等类型）—— #
            def _to_serializable(obj):
                import numpy as _np
                if isinstance(obj, dict):
                    return {str(k): _to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_serializable(v) for v in obj]
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                return obj

            # 解析统计
            qs, labeled, avg_r = 0, 0, 0.0
            if isinstance(stats, dict):
                qs = int(stats.get("queue_size", stats.get("pending", 0)) or 0)
                labeled = int(stats.get("total_labeled", 0) or 0)
                try:
                    avg_r = float(stats.get("average_reward", 0.0) or 0.0)
                except Exception:
                    avg_r = 0.0
            stats_text = f"Total Labeled: {labeled}\nAverage Reward: {avg_r:.2f}\nQueue Size: {qs}"

            # 空队列/提示信息：返回占位值（不要抛异常/不要返回 None）
            if not isinstance(case, dict) or ("error" in case) or ("message" in case):
                return {}, qs, "No cases available"

            # 正常返回（三个值，类型固定）
            return _to_serializable(case), qs, stats_text

        refresh_case_btn.click(
            fetch_next_case,
            outputs=[
                current_case_display,
                expert_queue_size,
                expert_stats
            ]
        )


        expert_mode.change(
            change_expert_mode,
            inputs=[expert_mode],
            outputs=[
                submit_expert_label_btn,
                skip_case_btn,
                expert_reward_slider,
                expert_stats
            ]
        ).then(
            fetch_next_case,  # 直接引用本地函数（已在上方定义）
            outputs=[current_case_display, expert_queue_size, expert_stats]
        )


        


        
        # 提交专家标注
        def submit_expert_feedback(case_data, reward_value):
            """提交专家标注；无病例/异常时不抛错，返回占位值（三个输出）"""
            try:
                import drive_tools as dt
            except Exception as e:
                return {}, 0, f"Error: {e}"
            fn_submit = getattr(dt, "submit_expert_label", None) \
                        or getattr(dt, "submit_label", None)
            if fn_submit is None:
                return {}, 0, "submit API not found in drive_tools"


            # 1) 无样例：返回 3 个占位值（JSON、pending、stats 文本）
            if not isinstance(case_data, dict) or 'case_id' not in case_data:
                return {}, 0, "No case selected"

            # 2) 提交；任何异常都不抛出到 UI
            try:
                result = fn_submit(case_data['case_id'], float(reward_value))
            except Exception as e:
                return {}, 0, f"Error: {e}"

            if isinstance(result, dict) and "error" in result:
                return {}, 0, f"Error: {result['error']}"

            # 3) 成功后拉下一例
            return fetch_next_case()

        
        submit_expert_label_btn.click(
            submit_expert_feedback,
            inputs=[current_case_display, expert_reward_slider],
            outputs=[
                current_case_display,
                expert_queue_size,
                expert_stats
            ]
        )
        
        # 跳过案例
        skip_case_btn.click(
            fetch_next_case,
            outputs=[
                current_case_display,
                expert_queue_size,
                expert_stats
            ]
        )

        # 系统重置按钮处理
        # def reset_system_ui():
        #     """重置系统"""
        #     try:
        #         # 这里可以添加系统重置逻辑
        #         return "✅ System reset completed."
        #     except Exception as e:
        #         return f"❌ Error: {str(e)}"
        
        # reset_system_btn.click(
        #     reset_system_ui,
        #     outputs=[shift_status]
        # )

        update_tau_btn.click(
            update_tau_threshold,
            inputs=[tau_slider],
            outputs=[tau_update_output]
        )
        evaluate_btn.click(
            run_system_evaluation,
            outputs=[evaluation_text, evaluation_plot]
        )
        
        download_report_btn.click(
            download_evaluation_report,
            outputs=[report_file]
        )        
        # Auto-refresh stats every 5 seconds when the tab is active
        demo.load(
            refresh_online_stats,
            outputs=[
                total_transitions,
                query_rate,
                buffer_size,
                avg_uncertainty,
                current_tau,
                training_updates,
                al_stats_plot,
                al_stats_table
            ]
        )
        def update_live_monitoring():
            """更新实时监控指标"""
            try:
                from drive_tools import _online_system, get_response_time_stats
                from drive_tools import get_cohort_stats
                
                if not _online_system:
                    return 0, 0, 0, "Unknown", 100, "Not Started"
                
                # 获取实时统计
                stats = _online_system['trainer'].get_statistics()
                al_stats = _online_system['active_learner'].get_statistics()
                response_stats = get_response_time_stats()
                
                # 计算实时指标
                query_rate = al_stats.get('query_rate', 0) * 100
                
                # 计算吞吐量
                recent_transitions = stats.get('total_transitions', 0)
                throughput = recent_transitions / max(time.time() - stats.get('start_time', time.time()), 1)
                
                # 获取延迟
                avg_latency = response_stats.get('avg_response_time', 0) * 1000 if response_stats else 0
                
                # 检测分布偏移
                shift_status = "Normal"  # 默认状态
                
                # 计算安全合规性
                safety_score = 95.0  # 默认值
                
                # 系统健康状态
                if throughput > 5 and query_rate < 30 and avg_latency < 100:
                    health_status = "✅ Healthy"
                elif throughput > 0:
                    health_status = "⚠️ Degraded"
                else:
                    health_status = "❌ Critical"
                
                return query_rate, throughput, avg_latency, shift_status, safety_score, health_status
                
            except Exception as e:
                return 0, 0, 0, "Error", 0, f"Error: {str(e)}"
            
        # Load initial data on startup
        demo.load(
            generate_virtual_data,
            inputs=[n_patients_slider],
            outputs=[current_source_text, stats_display, action_legend, patient_dropdown]
        )
    
    return demo


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DRIVE Clinical Decision Support Interface")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
