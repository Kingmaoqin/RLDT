# 系统算法与 UI 运作流程说明

## 总览
本系统围绕 **数据接入 → 快速建模 → 推理/可视化** 三个阶段设计：

1. **数据管理 (`data_manager`)** 负责生成虚拟数据或读取用户上传的数据，并根据提供的 Schema 规范进行字段映射、数值归一。
2. **推理与训练工具 (`drive_tools`)** 提供快速训练、在线推理、诊断等工具，核心使用 Transformer 动力学模型、回归式奖励模型、保守 Q 网络。
3. **交互界面 (`enhanced_chat_ui`)** 将上述能力串联成分步 UI：先选择/上传数据 → 查看预览 → 触发快速训练 → 进入患者探索与策略分析。

下文按照流程详细说明，并总结近期排查出的自定义数据问题与解决方案。

## 数据流：从上传到结构化表

1. **虚拟数据**：调用 `DataManager.generate_virtual_data` 直接生成标准化字段（`patient_id`、`timestep`、`action`、`reward`、`state_*`）。
2. **真实数据（无 Schema）**：`load_real_data_schema_less` 尝试根据常见列名推断轨迹、时间步与动作，适合快速调试。
3. **真实数据（推荐，带 Schema）**：
   - UI 上传数据文件及 YAML Schema 字符串/文件。
   - `SchemaSpec.from_yaml_text` 解析 YAML → dataclass。新增的 `from_yaml_text`/`from_yaml_file` 修复了此前报错 `SchemaSpec` 无此方法的问题。
   - `TabularAdapter.load` 基于 Schema 执行列映射、奖励派生与特征提取，输出统一的张量字典与元数据。
   - `DataManager.load_real_data_with_schema` 将张量回填为 UI 友好的 DataFrame，并记录：
     * `feature_columns`：全部 `state_*` 列，现统一转换为 `float32` 以避免训练阶段的 dtype 冲突。
     * `feature_dtypes`、`schema_source` 等诊断信息，便于 UI 侧提示。

> **问题记录**：用户上传 OV 数据时遇到 `SchemaSpec` 缺少 `from_yaml_text`。原因是旧版 `schema.py` 未提供 YAML 解析入口。现已新增 `from_yaml_text/from_yaml_file`，允许直接传入字符串或路径。
>
> 同期排查还发现：
> 1. **患者展示数值全部相同**——原 UI 假设特征已经 0~1 归一，导致真实数值（尤其是年龄与实验室指标）被错误放缩。现在 `data_manager` 会为每个 `state_*` 列记录真实统计量，患者面板使用这些统计量动态归一并显示原始值，因此不同患者能够呈现正确区分度。
> 2. **动作图例始终显示 “Action 0/1/2”**——缺乏 Schema 时未写入动作名称映射。最新实现优先沿用数据集中出现的原始标签（或 YAML 中定义的 `action_names`），同时 UI 图例支持字符串 ID，避免含义不明的占位文案。
> 3. **难以确认是否真的训练过**——轻量训练完成后会把样本数、状态/动作维度、耗时写入 `data_manager.training_runs`，UI 也在提示中附带最新记录时间，便于复查训练是否发生。

## 快速训练与推理流程

1. 数据加载成功后，`drive_tools.quick_train_on_current_data` 可对当前 DataFrame 进行轻量训练：
   - 依据 `patient_id`、`timestep` 排序生成轨迹。
   - 构建状态矩阵与下一状态矩阵，计算 Monte-Carlo 累积回报。
   - 训练三个模型：
     * **TransformerDynamicsModel**：学习 $s_{t+1}$。
     * **TreatmentOutcomeModel**：监督预测即时奖励。
     * **ConservativeQNetwork**：拟合累计回报，形成策略价值估计。
   - 训练完成后刷新全局推理引擎与 UI 元数据。
2. **dtype 报错排查**：自定义数据触发 `Quick training failed: Found dtype Double but expected Float`。定位到奖励序列以 `float64` 进入 torch，而模型参数为 `float32`。现将奖励与回报张量统一转换为 `float32`，同时对特征列执行 `float32` 归一，问题已解决。

## UI 分步交互逻辑

1. **数据选择面板**：默认仅展示“选择数据源”区域；在没有数据前，训练与分析板块保持折叠/禁用，避免空页面困扰。
2. **数据预览卡片**：载入成功后显示基础统计（记录数、患者数）及前几行数据，便于快速确认映射是否正确。
3. **快速训练卡片**：允许配置 epoch、batch size，触发 `quick_train_on_current_data`。失败信息直接展示来自 `drive_tools` 的诊断文字（现包含 dtype、列缺失提示）。
4. **推理 / 患者探索**：完成训练后开启，包括患者列表、行动建议、轨迹模拟等组件。

## 自定义数据使用建议

1. **准备 Schema**：确保 YAML 中 `mapping`、`reward_spec` 等字段完整。若 Schema 嵌套在 `schema:` 顶层，新接口也会自动提取。
2. **数据类型**：
   - 动作列建议为类别或整数；系统会自动映射到连续 id。
   - 特征列建议为数值。当前版本会自动尝试 `pd.to_numeric(...).astype(float32)`，非数值将转为 NaN 并在训练日志中提示。
3. **奖励定义**：若原始数据缺少 `reward` 列，可通过 `reward_spec.expression` 或 `label_to_reward` 派生。
4. **诊断查看**：UI 的“数据统计”区域会引用 `feature_dtypes`、`schema_source` 等信息，帮助定位潜在映射错误。

> **当前排查结论**：
> - `SchemaSpec.from_yaml_text` 缺失导致自定义 Schema 无法解析 —— 已实现并在此文档登记。
> - 快速训练阶段的 `float64` → `float32` 不匹配导致失败 —— 现已在数据导入与训练两处强制转换。
> - UI 患者视图和动作图例现依据最新元数据动态渲染，杜绝“全员同龄”“Action 0” 等展示错误。
> - 若仍遇到空白预览，请检查 `mapping.feature_cols` 是否覆盖所有需要展示的状态列。
> - 在线学习监控面板在未启动训练或返回异常时会渲染占位图表与状态说明，避免出现空白卡片。
> - 所有核心操作按钮均附加了悬浮问号提示（`enhanced_chat_ui` 中的 `help-target` 样式），鼠标悬停即可查看使用指导。

## 后续改进方向

- 将 `feature_dtypes`、缺失值计数等诊断信息直接渲染到 UI 提示卡。
- 根据 Schema 中的 `normalization` 配置自动应用归一化，并在 UI 中展示处理方式。
- 持续完善错误信息，使其指向具体的列名、行号，降低自定义数据调试成本。

如需进一步排查，请参考：
- `RL0910/schema.py`（Schema 定义与 YAML 解析）
- `RL0910/data_manager.py`（数据载入与元数据存储）
- `RL0910/drive_tools.py`（快速训练逻辑与 dtype 修复）
- `RL0910/enhanced_chat_ui.py`（交互流程与提示）
