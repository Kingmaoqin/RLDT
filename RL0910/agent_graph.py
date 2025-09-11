"""
agent_graph.py - 改进版：支持多种免费LLM API
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
import os
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

# 根据可用的API选择合适的LLM
def get_llm():
    """根据环境变量选择合适的LLM"""
    
    # 选项0: 本地LLM服务器（优先）
    if os.getenv("LOCAL_LLM_ENDPOINT"):
        from langchain_openai import ChatOpenAI
        endpoint = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:8080")
        return ChatOpenAI(
            model="local-model",
            openai_api_key="not-needed",
            openai_api_base=f"{endpoint}/v1",
            temperature=0.7,
            max_tokens=150
        )
    
    # 选项1: OpenAI兼容的免费API (如Groq)
    elif os.getenv("GROQ_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="llama2-70b-4096",  # 或使用 "mixtral-8x7b-32768"
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=150,
            request_timeout=30  # 添加超时设置
        )
    
    # 选项2: Google Gemini (有免费层)
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_output_tokens=150
        )
    
    # 选项3: Anthropic Claude (如果有API key)
    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            max_tokens=150
        )
    
    # 选项4: Cohere (有免费层)
    elif os.getenv("COHERE_API_KEY"):
        from langchain_community.chat_models import ChatCohere
        return ChatCohere(
            model="command",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            temperature=0.7,
            max_tokens=150
        )
    
    # 默认: 使用本地模型
    else:
        print("警告: 未找到API密钥，使用本地小模型。")
        print("建议设置以下环境变量之一：")
        print("- GROQ_API_KEY (推荐，完全免费)")
        print("- GOOGLE_API_KEY (Gemini，有免费额度)")
        print("- COHERE_API_KEY (有免费层)")
        
        from langchain_community.llms import Ollama
        try:
            # 尝试使用Ollama (需要先安装并运行)
            return Ollama(model="llama2", temperature=0.7)
        except:
            # 最后的备选：HuggingFace
            from langchain_community.llms import HuggingFacePipeline
            return HuggingFacePipeline.from_model_id(
                model_id="google/flan-t5-base",
                task="text2text-generation",
                model_kwargs={"temperature": 0.7, "max_length": 150}
            )

# Import tool functions
from drive_tools import (
    get_optimal_recommendation,
    get_all_action_values,
    calculate_treatment_effect,
    simulate_future_trajectory,
    get_feature_importance,
    get_immediate_reward,
    describe_parameter,
    recommend_parameters,
    update_reward_parameters,
    retrain_model,
    get_patient_list,
    get_patient_data,
    analyze_patient,
    get_cohort_stats
)


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    human_review: bool

class EnhancedDriveAgent:
    """增强的DRIVE代理，实现论文中的双向LLM-RL集成"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.context_memory = deque(maxlen=10)  # 上下文记忆
        
    def process_clinical_query(self, query: str, patient_context: Dict) -> str:
        """处理临床查询，实现双向信息流"""
        
        # 1. 从RL系统获取定量洞察
        rl_insights = self._extract_rl_insights(patient_context)
        
        # 2. LLM生成临床解释
        llm_response = self._generate_clinical_explanation(query, patient_context, rl_insights)
        
        # 3. 反馈给RL系统用于改进
        self._provide_llm_feedback(llm_response, patient_context)
        
        return llm_response
    
    def _extract_rl_insights(self, patient_context: Dict) -> Dict:
        """从RL系统提取定量洞察"""
        insights = {}
        
        try:
            # 获取治疗推荐和不确定性
            recommendation = get_optimal_recommendation(patient_context)
            all_actions = get_all_action_values(patient_context)
            
            insights = {
                'primary_recommendation': recommendation,
                'action_rankings': all_actions.get('action_values', []),
                'confidence_level': recommendation.get('confidence', 0),
                'uncertainty_estimate': self._calculate_uncertainty(all_actions),
                'risk_assessment': self._assess_clinical_risk(patient_context)
            }
            
        except Exception as e:
            insights = {'error': str(e)}
        
        return insights
    
    def _calculate_uncertainty(self, all_actions: Dict) -> float:
        """计算决策不确定性"""
        if 'action_values' not in all_actions:
            return 1.0
        
        q_values = [av['q_value'] for av in all_actions['action_values']]
        if len(q_values) < 2:
            return 1.0
        
        # 使用Q值的方差作为不确定性度量
        return float(np.var(q_values))
    
    def _assess_clinical_risk(self, patient_state: Dict) -> Dict:
        """评估临床风险"""
        risk_factors = []
        risk_score = 0.0
        
        # 检查关键生命体征
        if patient_state.get('oxygen_saturation', 1.0) < 0.9:
            risk_factors.append("Low oxygen saturation")
            risk_score += 0.3
        
        if patient_state.get('blood_pressure', 0.5) > 0.8:
            risk_factors.append("High blood pressure")
            risk_score += 0.2
        
        if patient_state.get('glucose', 0.5) > 0.8:
            risk_factors.append("High glucose level")
            risk_score += 0.1
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'risk_level': 'High' if risk_score > 0.5 else 'Medium' if risk_score > 0.2 else 'Low'
        }
    
    def _generate_clinical_explanation(self, query: str, patient_context: Dict, 
                                     rl_insights: Dict) -> str:
        """生成临床解释"""
        
        # 构建增强的提示
        enhanced_prompt = f"""
You are DRIVE, an AI clinical decision support system. You have access to:

1. **Patient Context**: {json.dumps(patient_context, indent=2)}

2. **Quantitative RL Insights**:
   - Primary Recommendation: {rl_insights.get('primary_recommendation', {})}
   - Confidence Level: {rl_insights.get('confidence_level', 0):.3f}
   - Decision Uncertainty: {rl_insights.get('uncertainty_estimate', 0):.3f}
   - Risk Assessment: {rl_insights.get('risk_assessment', {})}

3. **Treatment Rankings**: {rl_insights.get('action_rankings', [])}

**Clinical Query**: {query}

**Instructions**:
- Integrate quantitative RL insights with clinical reasoning
- Explain recommendations in terms clinicians understand
- Address uncertainty and risk explicitly
- Provide actionable guidance
- Cite specific Q-values and confidence scores when relevant
- Keep response under 1200 words

**Response**:
"""
        
        try:
            response = self.llm.invoke(enhanced_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating clinical explanation: {str(e)}"
    
    def _provide_llm_feedback(self, llm_response: str, patient_context: Dict):
        """将LLM反馈提供给RL系统"""
        
        # 分析LLM响应中的关键词和偏好
        feedback_signals = self._extract_feedback_signals(llm_response)
        
        # 存储上下文以改进未来决策
        self.context_memory.append({
            'patient_context': patient_context,
            'llm_response': llm_response,
            'feedback_signals': feedback_signals,
            'timestamp': time.time()
        })
        
        # 如果有在线系统，可以用这些信号调整参数
        if feedback_signals.get('suggests_conservative'):
            # 增加保守性
            if hasattr(drive_tools, '_online_system'):
                try:
                    drive_tools.update_hyperparams({'alpha': 1.2})  # 更保守的CQL
                except:
                    pass
    
    def _extract_feedback_signals(self, llm_response: str) -> Dict:
        """从LLM响应中提取反馈信号"""
        signals = {}
        
        response_lower = llm_response.lower()
        
        # 检测保守倾向
        conservative_keywords = ['conservative', 'cautious', 'careful', 'risk', 'safe']
        signals['suggests_conservative'] = any(kw in response_lower for kw in conservative_keywords)
        
        # 检测激进倾向
        aggressive_keywords = ['aggressive', 'intensive', 'maximum', 'optimal']
        signals['suggests_aggressive'] = any(kw in response_lower for kw in aggressive_keywords)
        
        # 检测不确定性关注
        uncertainty_keywords = ['uncertain', 'unclear', 'ambiguous', 'consider']
        signals['highlights_uncertainty'] = any(kw in response_lower for kw in uncertainty_keywords)
        
        return signals

def create_enhanced_drive_agent():
    """创建增强的DRIVE代理"""
    llm = get_llm()
    enhanced_agent = EnhancedDriveAgent(llm, tools)
    
    def enhanced_chat_function(message, history, patient_context=None):
        """增强的聊天函数"""
        if patient_context:
            return enhanced_agent.process_clinical_query(message, patient_context)
        else:
            # 回退到标准处理
            return chat_function(message, history)
    
    return enhanced_chat_function
    
# Tool definitions with better descriptions
@tool
def optimal_recommendation(patient_state: str) -> str:
    """获取最佳治疗建议。输入：患者状态的JSON字符串，包含age、gender、blood_pressure等字段。"""
    try:
        state_dict = json.loads(patient_state)
        result = get_optimal_recommendation(state_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def all_action_values(patient_state: str) -> str:
    """比较所有可能治疗方案的长期价值(Q值)。输入：患者状态JSON。"""
    try:
        state_dict = json.loads(patient_state)
        result = get_all_action_values(state_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def treatment_effect(patient_state: str, treatment_a: str, treatment_b: str) -> str:
    """计算两种治疗方案之间的因果效应差异(CATE)。treatment_a/b可选: 'Medication A/B/C', 'Placebo', 'Combination Therapy'"""
    try:
        state_dict = json.loads(patient_state)
        result = calculate_treatment_effect(state_dict, treatment_a, treatment_b)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def simulate_trajectory(patient_state: str, action_sequence: str, horizon: int) -> str:
    """模拟患者在治疗序列下的未来健康轨迹。action_sequence: 逗号分隔的治疗名称。"""
    try:
        state_dict = json.loads(patient_state)
        actions = [a.strip() for a in action_sequence.split(',')]
        result = simulate_future_trajectory(state_dict, actions, horizon)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def feature_importance() -> str:
    """获取影响治疗决策的关键患者特征排名。"""
    try:
        result = get_feature_importance()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def immediate_reward(patient_state: str, action: str) -> str:
    """预测某个治疗的即时临床效果。"""
    try:
        state_dict = json.loads(patient_state)
        result = get_immediate_reward(state_dict, action)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def describe_param(param_name: str) -> str:
    """描述模型参数的作用机制和量化影响。参数名如: alpha, gamma, learning_rate等。"""
    try:
        result = describe_parameter(param_name)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def recommend_params(patient_state: str) -> str:
    """根据患者状态推荐最佳参数配置。"""
    try:
        state_dict = json.loads(patient_state)
        result = recommend_parameters(state_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def update_params(new_params: str) -> str:
    """更新模型参数。输入JSON格式的参数字典，如: {'alpha': 1.5, 'gamma': 0.99}"""
    try:
        params_dict = json.loads(new_params)
        result = update_reward_parameters(params_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def retrain(training_params: str) -> str:
    """触发模型重训练。输入JSON格式的训练参数，如: {'preset': 'conservative'} 或自定义参数。"""
    try:
        params_dict = json.loads(training_params)
        result = retrain_model(params_dict)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def list_patients() -> str:
    """获取当前数据集中的所有患者列表。"""
    try:
        result = get_patient_list()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def patient_info(patient_id: str) -> str:
    """获取特定患者的详细信息，包括病史和当前状态。"""
    try:
        result = get_patient_data(patient_id)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def analyze_specific_patient(patient_id: str) -> str:
    """分析特定患者并获取个性化治疗建议。"""
    try:
        result = analyze_patient(patient_id)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def cohort_statistics() -> str:
    """获取当前数据集的队列统计信息。"""
    try:
        result = get_cohort_stats()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Tool list
tools = [
    optimal_recommendation,
    all_action_values,
    treatment_effect,
    simulate_trajectory,
    feature_importance,
    immediate_reward,
    describe_param,
    recommend_params,
    update_params,
    retrain,
    list_patients,
    patient_info,
    analyze_specific_patient,
    cohort_statistics
]


# System prompt - 更清晰的指令
SYSTEM_PROMPT = """你是DRIVE，一个临床决策支持AI助手。你帮助医生理解基于因果AI模型的治疗建议。

重要规则：
1. 回答必须简洁(每次回复≤1200字)
2. 只使用工具返回的数据，绝不编造信息
3. 如果不确定，明确说明
4. 聚焦关键洞察而非冗长解释
5. 你的回复必须是英文

可用工具：
- optimal_recommendation: 获取最佳治疗
- all_action_values: 比较所有方案
- treatment_effect: 计算治疗差异
- simulate_trajectory: 预测未来变化
- feature_importance: 关键决策因素
- immediate_reward: 即时效果
- describe_param: 解释参数作用
- recommend_params: 推荐参数配置
- update_params: 更新参数设置
- retrain: 触发模型重训练
- list_patients: 列出所有患者
- patient_info: 获取患者详情
- analyze_specific_patient: 分析特定患者
- cohort_statistics: 队列统计信息

当提到特定患者ID时，使用patient_info或analyze_specific_patient工具。
参数调节问题使用describe_param、recommend_params等工具。

请根据医生的问题，选择合适的工具获取数据后回答。"""


def create_drive_agent():
    """创建DRIVE代理"""
    
    # 获取LLM
    llm = get_llm()
    
    # 将工具绑定到LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """主推理节点 - 使用LLM with tools"""
        messages = state["messages"]
        
        # 添加系统消息
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # 调用LLM
        response = llm_with_tools.invoke(full_messages)
        
        # 检查是否需要人工审核
        human_review_needed = False
        if hasattr(response, 'content'):
            content = response.content
            human_review_needed = any(word in content.lower() for word in ["确认", "批准", "confirm", "approve"])
        
        return {
            "messages": [response],
            "human_review": human_review_needed
        }
    
    def human_review_node(state: AgentState) -> Dict[str, Any]:
        """人工审核节点"""
        # 实际应用中会暂停等待人工输入
        # 演示中自动批准
        return {
            "messages": [HumanMessage(content="已批准")],
            "human_review": False
        }
    
    def should_continue(state: AgentState) -> str:
        """决定下一步"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # 检查是否需要人工审核
        if state.get("human_review", False):
            return "human_review"
        
        # 检查是否有工具调用
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # 否则结束
        return "end"
    
    # 构建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("human_review", human_review_node)
    
    # 设置入口
    workflow.set_entry_point("agent")
    
    # 条件路由
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "human_review": "human_review",
            "end": END
        }
    )
    
    # 工具执行后返回agent
    workflow.add_edge("tools", "agent")
    
    # 人工审核后返回agent
    workflow.add_edge("human_review", "agent")
    
    # 编译图
    app = workflow.compile()
    
    return app


# 导出编译后的代理
drive_agent = create_drive_agent()