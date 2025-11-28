# RLDT

数字孪生强化学习 (DRIVE) 系统代码与临床决策支持 UI。

## 文档
- [系统流程说明](docs/system_workflow.md)
- [部署指南：云端发布与 Hugging Face LLM 接入](docs/deployment_guide.md)
- [Hugging Face Spaces 免费部署步骤](docs/huggingface_space_deploy.md)

## 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python RL0910/enhanced_chat_ui.py --share --port 7860  # 生成一个可分享的公网链接
```

- 免费部署示例请参考部署指南中的 **Hugging Face Spaces** 章节，无需服务器即可发布。
