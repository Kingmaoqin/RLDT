# Hugging Face Spaces 免费部署指南

本指南演示如何将 RLDT 的 Gradio 前端零成本发布到 **Hugging Face Spaces**，并使用 Hugging Face Inference 免费额度作为 LLM 后端。

## 1. 目录与关键文件
- `app.py`：Spaces 的入口脚本，直接调用现有的 `create_gradio_interface()` 并在 `PORT`（Spaces 自动注入）上启动，无需命令行参数。
- `requirements.txt`：Python 依赖列表，Spaces 会自动安装。
- `RL0910/enhanced_chat_ui.py`：主界面与业务逻辑，保持不变。

> 如果需要自带示例模型，可将模型权重放在仓库的 `output/models/` 下或在启动脚本中下载。

## 2. 准备 Hugging Face 账号与 Token
1. 访问 https://huggingface.co/settings/tokens 创建 **Write** 权限的 Access Token。
2. 复制 Token，稍后在 Space 中配置为 `HUGGINGFACEHUB_API_TOKEN`。Hugging Face Inference 免费额度足以支撑轻量对话演示。

## 3. 创建 Space
1. 在 Hugging Face 主页点击 **New Space**。
2. 选择模板 **Gradio**（`app.py` 会被自动识别）。
3. 选择 **Public**（免费额度）或 **Private**（需 Pro）。
4. 创建完成后，获取 Space 的 Git 远程地址（形如 `https://huggingface.co/spaces/<org>/<space-name>`）。

## 4. 推送代码到 Space
在本地或 CI 中执行：
```bash
# 1) 克隆当前仓库并进入目录
cd /path/to/RLDT

# 2) 添加 Space 作为远程
git remote add space https://huggingface.co/spaces/<org>/<space-name>

# 3) 推送 main 分支（或当前分支）
git push space main
```
推送完成后，Spaces 会自动构建：安装依赖 → 运行 `app.py` → 暴露页面。

## 5. 配置环境变量（LLM 走 Hugging Face Inference）
在 Space 页面打开 **Settings → Variables and secrets**，添加：
- `HUGGINGFACEHUB_API_TOKEN`：步骤 2 获取的 Token。
- 可选：
  - `HUGGINGFACE_MODEL`（默认 `meta-llama/Meta-Llama-3-8B-Instruct`）
  - `HUGGINGFACE_TASK`（默认 `text-generation`）
  - `HUGGINGFACE_TEMPERATURE`、`HUGGINGFACE_MAX_TOKENS`
  - 如需其它云端模型，也可设置 `GOOGLE_API_KEY`、`COHERE_API_KEY`、`ANTHROPIC_API_KEY` 等，优先级在 `RL0910/agent_graph.py` 中定义。

> 无需显式设置端口，Spaces 会自动注入 `PORT` 环境变量，`app.py` 会读取并绑定到 `0.0.0.0`。

## 6. 数据与模型的放置方式
- **轻量演示**：直接使用内置虚拟数据生成流程，无需上传文件。
- **自带模型/数据**：
  - 将模型权重放入仓库（小体积）并随代码推送；或
  - 在 `app.py` 中添加启动时的下载逻辑（如从 `hf://` 或公开 URL）；或
  - 在 Space 的 **Persistent storage**（启用后）中手动上传，再在代码中读取 `/data` 路径。
- 若使用本地训练功能，请在 **Variables and secrets** 中设置：
  - `DYNAMICS_MODEL_PATH`、`OUTCOME_MODEL_PATH`、`Q_NETWORK_PATH`
  - `DATA_DIR`（数据或导出报告目录）

## 7. 验证部署
- 等待构建完成后，点击 Space 页面右上角的 **App**，应显示 RLDT 的多标签 Gradio 界面。
- 打开 “Chat with DRIVE Agent” 输入指令，确认能返回由 Hugging Face Inference 提供的回复。
- 查看 **Logs**（左侧栏）可排查依赖安装或模型加载问题。

## 8. 常见问题
- **构建超时/内存不足**：优先使用虚拟数据与远程 LLM，避免在免费套餐加载过大的本地模型。
- **LLM 无响应**：确认 `HUGGINGFACEHUB_API_TOKEN` 和 `HUGGINGFACE_MODEL` 配置正确，或尝试更小的模型（如 `mistralai/Mistral-7B-Instruct-v0.2`）。
- **端口相关错误**：无需手动指定端口，确保未在代码中硬编码端口；`app.py` 已读取 `PORT`。

部署完成后即可分享 Space 链接给他人免费体验，无需自建服务器。
