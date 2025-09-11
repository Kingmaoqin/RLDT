# reports.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from PIL import Image
import io
import math
import numpy as np
import matplotlib.pyplot as plt

def build_action_catalog(meta: Dict, dataset: Dict) -> Dict[int, str]:
    """
    优先级：
      1) meta['action_map'] {int->name}
      2) meta['action_names'] [str,...]
      3) 从数据中推断 unique action id => "Action <id>"
    """
    # 1) 显式 action_map
    if isinstance(meta, dict) and isinstance(meta.get('action_map'), dict):
        # 规范化 key => int
        return {int(k): str(v) for k, v in meta['action_map'].items()}

    # 2) action_names 列表
    if isinstance(meta, dict) and isinstance(meta.get('action_names'), (list, tuple)):
        return {i: str(name) for i, name in enumerate(meta['action_names'])}

    # 3) 从数据推断
    actions = None
    for k in ('actions', 'action', 'y_action', 'a'):
        if k in dataset and dataset[k] is not None:
            actions = dataset[k]
            break
    if actions is None:
        # 最保守：默认 0,1
        return {0: 'Action 0', 1: 'Action 1'}

    ids = sorted(set(map(int, np.array(actions).reshape(-1))))
    return {i: f'Action {i}' for i in ids}

def _q_values_from_ensemble(state: np.ndarray, model_handles: Dict) -> Optional[np.ndarray]:
    """
    尝试从在线的 q_ensemble 拿一帧 Q(s,·)
    model_handles:
      - 'q_ensemble' : torch.nn.Module
      - 'device'     : str
    """
    try:
        import torch
        q_ens = model_handles.get('q_ensemble', None)
        if q_ens is None:
            return None
        x = torch.as_tensor(state, dtype=torch.float32, device=model_handles.get('device', 'cpu'))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            # q_ensemble(x) -> (batch, action_dim) 或 (K,batch,action_dim) 看你的实现
            out = q_ens(x)
            if hasattr(out, 'shape') and out.ndim == 2:
                return out.cpu().numpy()[0]  # (A,)
            if out.ndim == 3:
                # (K,B,A) => 均值
                return out.mean(dim=0).cpu().numpy()[0]
        return None
    except Exception:
        return None

def _scores_from_policy_only(state: np.ndarray, model_handles: Dict, action_dim: int) -> np.ndarray:
    """
    当只有 BCQ (或 DQN baseline) 只能 predict 动作时，构造一个“伪分数”：
    预测动作给 1.0，其它给 0.5 / 0.2 衰减，目的是在报告里能排个序+展示相对优势。
    """
    try:
        # 先试 bcq_online_trainer
        bcq = model_handles.get('bcq_trainer', None)
        if bcq:
            a = int(bcq.predict(state))
            s = np.full(action_dim, 0.2, dtype=np.float32)
            s[a] = 1.0
            return s
        # 再试 baseline algo
        base = model_handles.get('baseline_trainer', None)
        if base:
            a = int(base.predict(state))
            s = np.full(action_dim, 0.2, dtype=np.float32)
            s[a] = 1.0
            return s
    except Exception:
        pass
    return np.zeros(action_dim, dtype=np.float32)

def _estimate_uncertainty(state: np.ndarray, model_handles: Dict) -> float:
    """
    与 StreamActiveLearner/UncertaintySampler 的逻辑保持一致：
      - 若使用 BCQ：用状态方差代理，再映射到 [0,1]
      - 否则：如果能拿到 ensemble 的 (K,B,A) 则用方差/变异系数 + tanh 压缩
      - 最后兜底 0.1
    """
    try:
        # 优先尝试从 sampler 拿
        sampler = model_handles.get('uncertainty_sampler', None)
        if sampler and hasattr(sampler, 'get_uncertainty'):
            return float(sampler.get_uncertainty(state))
    except Exception:
        pass

    # 简化兜底：与 samplers.py 中 BCQ 的近似一致
    try:
        import numpy as np
        return float(min(1.0, max(0.0, np.var(state) * 10.0)))
    except Exception:
        return 0.1

def compute_recommendation(state: np.ndarray,
                           model_handles: Dict,
                           action_catalog: Dict[int, str],
                           meta: Dict,
                           topk: int = 3) -> Dict:
    """
    返回:
      {
        'ranked': [(aid, name, score), ...],
        'uncertainty': float,
        'q_span': float,  # top 与 次优 的差
        'action_dim': int
      }
    """
    # 先尝试 ensemble 的真实 Q
    q = _q_values_from_ensemble(state, model_handles)
    if q is None:
        # 退化：用 policy-only 的打分
        action_dim = len(action_catalog) if action_catalog else 2
        q = _scores_from_policy_only(state, model_handles, action_dim)
    else:
        action_dim = q.shape[0]

    # 排序
    idx = np.argsort(-q)  # desc
    ranked = []
    for i in idx[:min(topk, len(idx))]:
        ranked.append((int(i), action_catalog.get(int(i), f'Action {i}'), float(q[i])))

    # 不确定性（对报告用）
    u = _estimate_uncertainty(state, model_handles)
    q_span = float(q[idx[0]] - q[idx[1]]) if len(idx) > 1 else float(q[idx[0]])

    return {'ranked': ranked, 'uncertainty': u, 'q_span': q_span, 'action_dim': int(action_dim)}

def _eval_critical_rules(state: np.ndarray, meta: Dict) -> List[str]:
    """
    评估 schema/meta 中的 'critical_features' 规则（之前在 adapters/schema 已支持）
    期望 meta['critical_features'] 是列表： [{'feature':'SpO2','op':'<','value':0.9,'message':'...'}, ...]
    如果只有 feature index，则使用 meta['feature_columns'] 做回填
    """
    msgs = []
    rules = meta.get('critical_features') or []
    feat_cols = meta.get('feature_columns') or []
    name_to_idx = {n: i for i, n in enumerate(feat_cols)}

    for r in rules:
        try:
            f = r.get('feature')
            idx = name_to_idx.get(f) if isinstance(f, str) else int(f)
            val = float(state[idx])
            op = r.get('op', '<')
            thr = float(r.get('value', 0))
            ok = (val < thr) if op == '<' else (val > thr) if op == '>' else False
            if ok:
                msgs.append(r.get('message') or f"Rule: {f} {op} {thr} (val={val:.4f})")
        except Exception:
            continue
    return msgs

def _safety_checks(state: np.ndarray, meta: Dict) -> List[str]:
    out = []
    stats = meta.get('feature_stats') or {}
    cols = meta.get('feature_columns') or []
    for i, name in enumerate(cols):
        info = stats.get(name) or {}
        lo, hi = info.get('min', None), info.get('max', None)
        try:
            v = float(state[i])
            if lo is not None and v < lo:
                out.append(f"{name} below min ({v:.4f} < {lo})")
            if hi is not None and v > hi:
                out.append(f"{name} above max ({v:.4f} > {hi})")
        except Exception:
            pass
    return out

def render_patient_report(patient: Dict,
                          state: np.ndarray,
                          rec: Dict,
                          meta: Dict,
                          action_catalog: Dict[int, str]) -> str:
    """
    统一渲染 Markdown 报告
    """
    pid = patient.get('id') or patient.get('trajectory_id') or 'N/A'
    cols = meta.get('feature_columns') or [f'feat{i}' for i in range(len(state))]
    pairs = ", ".join([f"{c}={float(state[i]):.4f}" for i, c in enumerate(cols[: min(8, len(cols))])])

    # 触发规则与安全检查
    fired = _eval_critical_rules(state, meta)
    safety = _safety_checks(state, meta)

    # 推荐段
    lines = []
    lines.append(f"# Patient Report (ID: {pid})\n")
    lines.append("## Snapshot")
    lines.append(f"- Latest {len(cols)} features (first 8 shown): {pairs}")
    lines.append(f"- Uncertainty: **{rec['uncertainty']:.3f}**")
    lines.append("")

    lines.append("## Recommendations")
    for rank, (aid, name, score) in enumerate(rec['ranked'], 1):
        lines.append(f"{rank}. **{name}**  \n   - Score: {score:.4f}")

    if rec.get('q_span') is not None:
        lines.append(f"- Advantage over next best: **{rec['q_span']:.4f}**")
    lines.append("")

    if fired:
        lines.append("## Triggered Clinical Rules")
        for m in fired[:10]:
            lines.append(f"- {m}")
        lines.append("")
    if safety:
        lines.append("## Safety Checks")
        for m in safety[:10]:
            lines.append(f"- {m}")
        lines.append("")
    lines.append("## Notes")
    lines.append("- This report is generated from your uploaded schema & data; action labels are dynamic.")
    return "\n".join(lines)




def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def render_patient_report_html(patient: Dict,
                               analysis: Dict,
                               action_catalog: Dict[int, str]) -> str:
    """Render an HTML patient report using available analysis data."""
    pid = str(patient.get("patient_id") or patient.get("id") or "Unknown")
    state = patient.get("current_state", {}) or {}
    rec = (analysis or {}).get("recommendation", {}) or {}
    act = rec.get("recommended_action")
    rt = rec.get("recommended_treatment")
    if isinstance(rt, str) and rt.isdigit():
        rec_name = action_catalog.get(int(rt), f"Action {rt}")
    elif rt:
        rec_name = str(rt)
    elif isinstance(act, (int, float)):
        rec_name = action_catalog.get(int(act), f"Action {int(act)}")
    elif isinstance(act, str):
        rec_name = act
    else:
        rec_name = "Unknown"
    conf_val = rec.get("confidence")
    conf = f"{_safe_float(conf_val):.3f}" if conf_val is not None else "N/A"
    exp_val = rec.get("expected_immediate_outcome")
    exp_out = f"{_safe_float(exp_val):.3f}" if exp_val is not None else "N/A"

    # Top-K comparison if present
    all_opts = (analysis or {}).get("all_options", {}) or {}
    avs = all_opts.get("action_values") or []
    def _score(av):
        return _safe_float(av.get("q_value", av.get("expected_outcome", 0.0)))
    avs_sorted = sorted(avs, key=_score, reverse=True)[:5]

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "body{font-family:Arial,sans-serif;margin:20px;color:#000;}"
        "h1,h2{color:#2c3e50;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
        "th{background:#34495e;color:#fff;}"
        ".section{margin:20px 0;padding:10px;border-radius:8px;}"
        ".recommendation{background:#e8f4f8;}"
        "</style></head><body>",
        f"<h1>Patient Report (ID: {pid})</h1>",
    ]

    # State snapshot
    html.append("<div class='section'>")
    html.append("<h2>Current State</h2>")
    if state:
        html.append("<table><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>")
        keys = list(state.keys())[: min(12, len(state))]
        for k in keys:
            v = state[k]
            try:
                val = f"{float(v):.3f}" if isinstance(v, float) or isinstance(v, int) else str(v)
            except Exception:
                val = str(v)
            html.append(f"<tr><td>{k}</td><td>{val}</td></tr>")
        html.append("</tbody></table>")
    else:
        html.append("<p>No structured state information.</p>")
    html.append("</div>")

    # Recommendation
    html.append("<div class='section recommendation'>")
    html.append("<h2>Treatment Recommendation</h2>")
    html.append(f"<p><strong>Recommended:</strong> {rec_name}</p>")
    html.append(f"<p><strong>Confidence:</strong> {conf}</p>")
    html.append(f"<p><strong>Expected Immediate Outcome:</strong> {exp_out}</p>")
    html.append("</div>")

    # Option ranking
    if avs_sorted:
        html.append("<div class='section'>")
        html.append("<h2>Top Options</h2>")
        html.append("<table><thead><tr><th>Action</th><th>Score</th></tr></thead><tbody>")
        for av in avs_sorted:
            nm = str(av.get("action"))
            sc = _score(av)
            html.append(f"<tr><td>{nm}</td><td>{sc:.3f}</td></tr>")
        html.append("</tbody></table></div>")

    html.append("<p><em>Report generated by AI system; consult professionals for decisions.</em></p>")
    html.append("</body></html>")
    return "".join(html)

def render_patient_report_md(patient: Dict, analysis: Dict, cohort_stats: Optional[Dict] = None) -> str:
    pid = str(patient.get('patient_id', 'Unknown'))
    state = patient.get('current_state', {})
    rec = (analysis or {}).get('recommendation', {}) or {}
    all_opts = (analysis or {}).get('all_options', {}) or {}
    avs = all_opts.get('action_values') or []
    action_catalog = {av.get('action_id'): av.get('action') for av in avs}
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
    conf = _safe_float(rec.get('confidence', 0.0))
    exp_out = _safe_float(rec.get('expected_immediate_outcome', 0.0))
    # 排序（按 q 或 outcome）
    def _score(av):
        return _safe_float(av.get('q_value', av.get('expected_outcome', 0.0)))
    avs_sorted = sorted(avs, key=_score, reverse=True)[:5]

    lines = []
    lines.append(f"# 患者报告（ID: {pid}）")
    lines.append("")
    lines.append("## 一、当前状态概览")
    if state:
        # 取前若干关键值
        keys = list(state.keys())
        keys = keys[: min(12, len(keys))]
        for k in keys:
            v = state[k]
            try:
                if isinstance(v, float):
                    lines.append(f"- **{k}**：{v:.3f}")
                else:
                    lines.append(f"- **{k}**：{v}")
            except Exception:
                lines.append(f"- **{k}**：{v}")
    else:
        lines.append("- 暂无结构化状态信息")

    lines.append("")
    lines.append("## 二、治疗建议")
    lines.append(f"- **推荐**：{rec_name}")
    lines.append(f"- **信心**：{conf:.3f}")
    lines.append(f"- **期望即时收益/结局**：{exp_out:.3f}")

    if avs_sorted:
        lines.append("")
        lines.append("### 候选方案排序（前 5）")
        for i, av in enumerate(avs_sorted, 1):
            nm = str(av.get('action', f'Option-{i}'))
            qv = _safe_float(av.get('q_value', av.get('expected_outcome', 0.0)))
            lines.append(f"{i}. {nm}（评分 {qv:.3f}）")

    # 关键驱动因素（若 explain 返回）
    abn = (analysis or {}).get('abnormal_features') or []
    if abn:
        lines.append("")
        lines.append("## 三、关键影响因素")
        for item in abn[:10]:
            feat = str(item.get('feature'))
            val = _safe_float(item.get('value', 0.0))
            status = item.get('status', '')
            lines.append(f"- {feat}：{status}（{val:.3f}）")

    # 反事实对比（若有）
    cf = (analysis or {}).get('counterfactuals') or {}
    if cf:
        lines.append("")
        lines.append("## 四、反事实评估（不同治疗的长期收益）")
        # 只展示均值
        # 格式：治疗：均值（5%-95%）
        for k, v in cf.items():
            mean = _safe_float(v.get('mean_outcome', 0.0))
            ci = v.get('confidence_interval', [None, None])
            lo = _safe_float(ci[0], default=mean)
            hi = _safe_float(ci[1], default=mean)
            lines.append(f"- {k}：{mean:.3f}（95%CI {lo:.3f} ~ {hi:.3f}）")

    # 安全校验（若有）
    sc = rec.get('safety_check')
    if isinstance(sc, dict):
        lines.append("")
        lines.append("## 五、用药安全校验")
        ok = sc.get('approved', True)
        reason = sc.get('reason', '无')
        alt = sc.get('alternative')
        lines.append(f"- 通过：{'是' if ok else '否'}；原因：{reason}")
        if (not ok) and alt:
            lines.append(f"- 已切换安全备选：{alt}")

    # 队列/队群统计（可选）
    if isinstance(cohort_stats, dict) and cohort_stats:
        lines.append("")
        lines.append("## 六、队列统计（简要）")
        tp = int(cohort_stats.get('total_patients', 0))
        tr = int(cohort_stats.get('total_records', 0))
        lines.append(f"- 总患者数：{tp}")
        lines.append(f"- 总记录数：{tr}")

    lines.append("")
    lines.append("> 说明：报告为模型推理与仿真所得，不替代临床判断。")
    return "\n".join(lines)


def make_treatment_analysis_figure(analysis: Dict) -> Image.Image:
    """
    生成稳健的治疗分析图（PIL Image），包括动作评分柱状图 + 7步轨迹预览。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # 左：动作评分
    ax = axes[0]
    all_opts = (analysis or {}).get('all_options') or {}
    avs = all_opts.get('action_values') or []
    action_catalog = {av.get('action_id'): av.get('action') for av in avs}
    rec = (analysis or {}).get('recommendation', {}) or {}
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
    rec_action = act
    if avs:
        acts = []
        vals = []
        cols = []
        for av in avs:
            a = str(av.get('action', 'A?'))
            a_id = av.get('action_id', None)
            q = _safe_float(av.get('q_value', av.get('expected_outcome', 0.0)))
            acts.append(a)
            vals.append(q)
            if (rec_action is not None and a_id == rec_action) or (rec_name and a == rec_name):
                cols.append('tomato')
            else:
                cols.append('steelblue')
        ax.bar(range(len(acts)), vals, color=cols, alpha=0.85)
        ax.set_xticks(range(len(acts)))
        ax.set_xticklabels(acts, rotation=30, ha='right')
        ax.set_title('Treatment Options')
        ax.set_ylabel('Score')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No action values', ha='center', va='center')

    # 右：7步轨迹
    ax = axes[1]
    traj = ((analysis or {}).get('predicted_trajectory') or {}).get('trajectory') or []
    traj = traj[:7]
    if traj:
        steps = list(range(len(traj)))
        # 尝试取常见两项：glucose / oxygen_saturation；找不到就取第 0 与最后一维
        def _pick(dct, key, fallback):
            v = dct.get(key, fallback)
            return _safe_float(v, default=fallback)
        g = []
        o = []
        last_g, last_o = 0.5, 0.95
        for t in traj:
            st = t.get('state', {})
            gv = st.get('glucose', last_g)
            ov = st.get('oxygen_saturation', last_o)
            gv = _safe_float(gv, default=last_g); ov = _safe_float(ov, default=last_o)
            g.append(gv); o.append(ov)
            last_g, last_o = gv, ov
        ax2 = ax.twinx()
        l1 = ax.plot(steps, g, marker='o', label='Glucose')
        l2 = ax2.plot(steps, o, marker='s', linestyle='--', label='O2 Sat')
        ax.set_xlabel('Step'); ax.set_ylabel('Glucose')
        ax2.set_ylabel('O2 Sat')
        ax.set_title('7-step Projection')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        ax.grid(alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No trajectory', ha='center', va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def _html_escape(s: str) -> str:
    import html
    return html.escape(str(s), quote=True)

def render_patient_report_html(patient: Dict,
                               analysis: Dict,
                               action_catalog: Dict[int, str],
                               meta: Optional[Dict] = None,
                               cohort_stats: Optional[Dict] = None) -> str:
    """
    生成自然语言 HTML 报告（兼容任意数据列；动作名来自 action_catalog）
    """
    pid = str(patient.get('patient_id', 'Unknown'))
    state = patient.get('current_state', {}) or {}
    rec  = (analysis or {}).get('recommendation', {}) or {}
    all_opts_dict = (analysis or {}).get('all_options') or {}
    all_opts = all_opts_dict.get('action_values') or []
    traj = ((analysis or {}).get('predicted_trajectory') or {}).get('trajectory') or []
    warning = (analysis or {}).get('all_options_error') or all_opts_dict.get('error')

    act = rec.get("recommended_action")
    rt  = rec.get("recommended_treatment")
    if isinstance(rt, str) and rt.isdigit():
        rec_name = action_catalog.get(int(rt), f"Action {rt}")
    elif rt:
        rec_name = str(rt)
    elif isinstance(act, (int, float)):
        rec_name = action_catalog.get(int(act), f"Action {int(act)}")
    elif isinstance(act, str):
        rec_name = act
    else:
        rec_name = "Unknown"
    conf = rec.get('confidence', None)
    exp  = rec.get('expected_immediate_outcome', None)

    # 状态表（仅展示若干关键项）
    keys = list(state.keys())[:12]
    rows: List[str] = []
    abnormal: List[str] = []
    stats = (meta or {}).get('feature_stats') if isinstance(meta, dict) else {}
    for k in keys:
        v = state[k]
        status = "Normal"
        try:
            fv = float(v)
            vv = f"{fv:.3f}"
            info = stats.get(k) if isinstance(stats, dict) else None
            lo = info.get('min') if isinstance(info, dict) else None
            hi = info.get('max') if isinstance(info, dict) else None
            if (lo is not None and fv < lo) or (hi is not None and fv > hi):
                status = "Review"
                abnormal.append(k)
        except Exception:
            vv = _html_escape(v)
            status = "N/A"
        badge_class = 'badge-normal' if status == 'Normal' else 'badge-alert'
        rows.append(
            f"<tr><td>{_html_escape(k)}</td><td>{vv}</td><td><span class='badge {badge_class}'>{status}</span></td></tr>"
        )
    state_table = "\n".join(rows) if rows else "<tr><td colspan='3'>No structured state</td></tr>"

    # 候选治疗排序
    def _score(d):
        return float(d.get('q_value', d.get('expected_outcome', 0.0)) or 0.0)
    avs_sorted = sorted(all_opts, key=_score, reverse=True)[:5]
    comp_rows = []
    for it in avs_sorted:
        act = it.get('action', 'Option')
        nm = _html_escape(action_catalog.get(int(act), act)) if isinstance(act, (int, float)) else _html_escape(str(act))
        qv = _score(it)
        comp_rows.append(f"<tr><td>{nm}</td><td>{qv:.3f}</td></tr>")
    comp_table = "\n".join(comp_rows)

    explanation = ""
    if avs_sorted:
        best = avs_sorted[0]
        best_act = best.get('action', 'Unknown')
        if isinstance(best_act, (int, float)):
            best_name = action_catalog.get(int(best_act), 'Unknown')
        else:
            best_name = str(best_act)
        best_score = _score(best)
        if len(avs_sorted) > 1:
            second = avs_sorted[1]
            second_act = second.get('action', 'Unknown')
            if isinstance(second_act, (int, float)):
                second_name = action_catalog.get(int(second_act), 'Unknown')
            else:
                second_name = str(second_act)
            diff = best_score - _score(second)
            explanation = (f"Model favors <strong>{_html_escape(best_name)}</strong> "
                            f"(score {best_score:.3f}), outperforming { _html_escape(second_name) } "
                            f"by {diff:.3f}.")
        else:
            explanation = (f"Model recommends <strong>{_html_escape(best_name)}</strong> "
                            f"with estimated score {best_score:.3f}.")

    if abnormal:
        vitals_comment = "Attention: " + ", ".join(map(_html_escape, abnormal)) + " outside normal range." 
    else:
        vitals_comment = "No abnormal vital signs detected."

    # 生成时间戳
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Assemble HTML report
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\"/>",
        "<title>Patient Treatment Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 32px; }",
        "h1, h2 { color: #2c3e50; }",
        ".card { background:#f9fbfd; border:1px solid #e6ecf5; border-radius:10px; padding:16px; margin:12px 0; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border:1px solid #ddd; padding:10px; text-align:left; }",
        "th { background:#34495e; color:#fff; }",
        ".badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; }",
        ".badge-normal { background:#2ecc71; color:#fff; }",
        ".badge-alert { background:#e74c3c; color:#fff; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Digital Twin Treatment Recommendation Report</h1>",
        f"<p><strong>Generated:</strong> {now}</p>",
        "<h2>Patient</h2>",
        f"<div class='card'><p><strong>ID:</strong> {pid}</p></div>",
        "<h2>Current State (snapshot)</h2>",
        "<table>",
        "  <thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>",
        "  <tbody>",
        f"  {state_table}",
        "  </tbody>",
        "</table>",
        "<h2>Treatment Recommendation</h2>",
        f"<div class='card'><p><strong>Recommended Treatment:</strong> { _html_escape(rec_name) }</p>"
        f"<p><strong>Confidence:</strong> { (f'{conf:.3f}' if isinstance(conf,(int,float)) else 'N/A') }</p>"
        f"<p><strong>Expected Immediate Outcome:</strong> { (f'{float(exp):+.3f}' if isinstance(exp,(int,float)) else 'N/A') }</p></div>",
    ]
    if warning:
        html_parts.append(f"<p style='color:#e74c3c;'><strong>Warning:</strong> {_html_escape(warning)}</p>")
    if avs_sorted:
        html_parts.extend([
            "<h2>Treatment Comparison</h2>",
            "<table>",
            "  <thead><tr><th>Plan</th><th>Score</th></tr></thead>",
            "  <tbody>",
            f"    {comp_table}",
            "  </tbody>",
            "</table>",
            "<h2>Treatment Explanation</h2>",
            f"<p class='card'>{explanation}</p>",
        ])
    html_parts.extend([
        f"<p>{vitals_comment}</p>",
        "<h2>Notes</h2>",
        "<p class='card'>This report is generated from your uploaded schema & data. Action labels are dynamic and resolved from your dataset or YAML schema.</p>",
        "</body>",
        "</html>",
    ])
    return "".join(html_parts)
