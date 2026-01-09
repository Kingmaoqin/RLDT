# 云端部署与 Hugging Face LLM 接入指南

本文档说明如何把 RLDT 的 Gradio 前端部署到云端，使团队通过公网链接访问，并使用 Hugging Face Inference 取代本地 LLM。

## 部署目标概览
- 在云服务器上启动 `enhanced_chat_ui.py`，通过 `--share` 或自定义反向代理暴露公网地址。
- 使用环境变量配置训练/推理所需路径，并通过 Hugging Face Token 调用远程大模型。
- 推荐使用 Docker 封装运行环境，便于一键启动和升级。

## 准备工作
1. **云主机**：任选支持持久网络访问的云服务（如 AWS EC2、GCP、阿里云）。开放应用端口（默认 7860）。
2. **Python 运行环境**：Python 3.10+，并安装项目依赖。
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **模型与数据路径**：将训练好的模型权重（`dynamics_model.pth`、`outcome_model.pth`、`q_network.pth` 等）上传到云端，并在启动前通过环境变量指向这些路径。

## 依赖安装与封装步骤
1. **创建隔离环境**（推荐）：`python -m venv .venv && source .venv/bin/activate`。
2. **安装依赖**：`pip install -r requirements.txt`，其中包含 Gradio、LangChain（含 Hugging Face/OpenAI/Cohere/Gemini/Anthropic 客户端）、PyTorch、d3rlpy、数据科学工具等。
3. **可选 LLM 依赖**：
   - 若使用 Google Gemini / Cohere / Anthropic，对应包已在 `requirements.txt`；只需提供 API Key。
   - 如需 Ollama 本地推理，需在系统层预装 Ollama 并拉取模型。
4. **运行健康检查**：`python RL0910/system_health_check.py`，确认模型路径、端口和网络正常。
5. **打包运行资产**：
   - 保留 `requirements.txt`、`RL0910/`、`OVdata.py`、`docs/` 以及 `output/models`（或挂载外部模型目录）。
   - 将 `.env`（存放密钥与路径）与 `Dockerfile` 一同提交/上传到云端，便于一键构建。

## 关键环境变量
将以下变量写入 `.env` 或云服务的“环境变量”面板中：

- **LLM 相关**
  - `HUGGINGFACEHUB_API_TOKEN`：Hugging Face Inference API Token。
  - `HUGGINGFACE_MODEL`（可选）：使用的仓库名，默认 `meta-llama/Meta-Llama-3-8B-Instruct`。
  - `HUGGINGFACE_TASK`（可选）：推理任务，默认 `text-generation`。
  - `HUGGINGFACE_TEMPERATURE`、`HUGGINGFACE_MAX_TOKENS`（可选）：生成温度与最大输出长度。
  - 其他可选开关：`LOCAL_LLM_ENDPOINT`、`GROQ_API_KEY`、`GOOGLE_API_KEY`、`COHERE_API_KEY`，当同时存在时优先级见 `RL0910/agent_graph.py`。

- **模型权重与数据**
  - `DYNAMICS_MODEL_PATH`、`OUTCOME_MODEL_PATH`、`Q_NETWORK_PATH`：推理所需模型文件路径（若使用在线推理/训练功能）。
  - `DATA_DIR`：数据集或导出报告的存放目录。

## 直接运行（快速验证）
在云主机上拉取代码后执行：
```bash
cd /path/to/RLDT
export HUGGINGFACEHUB_API_TOKEN="<你的 Token>"
python RL0910/enhanced_chat_ui.py --share --port 7860
```
- `--share` 会创建一个 Gradio 提供的公网链接，适合快速演示。
- 若需绑定固定域名，可省略 `--share`，改由 Nginx/Caddy 做反向代理并配置 HTTPS。

## Docker 部署（推荐生产环境）
1. 创建镜像（示例 Dockerfile）：
   ```Dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY . /app
   RUN pip install --no-cache-dir -r requirements.txt
   ENV PORT=7860
   EXPOSE 7860
   CMD ["python", "RL0910/enhanced_chat_ui.py", "--port", "7860"]
   ```

2. 构建与运行：
   ```bash
   docker build -t rldt-app .
   docker run -d \
     -p 7860:7860 \
     --env-file .env \
     -v /path/on/host/models:/app/models \
     --name rldt rldt-app
   ```
   - `.env` 中写入前述环境变量；`-v` 将主机模型目录挂载到容器。

3. **上线访问**：
   - 直接暴露端口：配置安全组放行 7860（或映射到 80/443）。
   - 使用反向代理：在 Nginx 中将域名流量转发到容器 `localhost:7860`，并开启 TLS。

## 免费云端部署：Hugging Face Spaces（零成本上线）
Hugging Face Spaces 提供免费额度，可直接托管 Gradio 前端并使用 Hugging Face Inference。

1. **准备仓库**：确保仓库包含 `app.py`、`requirements.txt`、`RL0910/` 和模型权重（或在 Space 中通过挂载/下载）。
2. **创建 Space**：在 Hugging Face 主页选择 *New Space*，选择 **Gradio** 模板，勾选 Public（免费额度）。
3. **推送代码**：将当前仓库（含 `requirements.txt`）推送到 Space 的 Git 远端。
4. **配置变量**：在 Space 的 *Settings → Variables and secrets* 设置 `HUGGINGFACEHUB_API_TOKEN`、`HUGGINGFACE_MODEL` 等环境变量；如模型体积较大，可改为在启动脚本中从外部 URL 下载。
5. **入口应用**：Space 会自动识别 `app.py` 中的 `demo`（Gradio 对象）并启动，无需手动指定 `--share` 或端口；`app.py` 会读取 `PORT` 环境变量并绑定 `0.0.0.0`。
6. **访问链接**：构建完成后会生成永久公开链接，免费分享给团队；若需私有访问，可设置 Space 为 Private（需付费额度）。

## 使用远程 Hugging Face LLM
- 只需在部署环境设置 `HUGGINGFACEHUB_API_TOKEN`（及可选模型、参数变量），`agent_graph.get_llm` 会自动选择 Hugging Face Chat 模型，替代默认本地 LLM。
- 更换模型时无需改代码：更新环境变量后重启服务即可。

## 运行健康检查与日志
- 启动脚本会在控制台输出加载信息。可通过 Docker `logs rldt -f` 或云监控查看。
- 如需系统级自检，可运行 `python RL0910/system_health_check.py`，确保依赖模型、数据路径、网络连通性均正常。

## 常见问题
- **端口无法访问**：确认云安全组/防火墙放行对应端口；如用反向代理，检查 upstream 配置。
- **LLM 无响应或超时**：核对 Token 是否正确、模型是否支持 Inference API；可调小 `HUGGINGFACE_MAX_TOKENS`、增大超时。
- **显存不足**：使用 Hugging Face Inference 即可避免本地显存限制；若仍需本地推理，可改用轻量模型并启用 `LOCAL_LLM_ENDPOINT`。
