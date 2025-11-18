# my_reward_qwen_prm_official.py
import os, json, math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "/home/ma-user/work/models/Qwen/Qwen2.5-Math-PRM-7B"
DEVICE_POLICY = "auto"  # "auto"/"cpu"/具体设备
STEP_TOKEN = "<extra_0>"

tokenizer = None
model = None
step_id = None


def lazy_load():
    global tokenizer, model, step_id
    if model is not None:
        return
    # 官方要求 transformers>=4.40 且 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map=DEVICE_POLICY, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    ids = tokenizer.encode(STEP_TOKEN, add_special_tokens=False)
    if not ids:
        raise RuntimeError("Tokenizer cannot encode <extra_0>.")
    step_id = ids[0]


def chat_to_text(chat_like: Any) -> str:
    # collab 的 prompt 是 chat 列（[{role, content}, ...]）；展开为纯文本上下文
    if isinstance(chat_like, list) and chat_like and isinstance(chat_like[0], dict):
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in chat_like)
    if isinstance(chat_like, str):
        return chat_like
    return json.dumps(chat_like, ensure_ascii=False)


def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: Optional[str],
                  extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    VERL 奖励入口：返回单标量 reward ∈ [0,1]
    - 只对最后一轮回答 solution_str 打分；
    - 多轮历史 + 最后一问从 extra_info['prompt'] 读入，放到 user 上下文。
    """
    lazy_load()

    # user：历史上下文
    chat = (extra_info or {}).get("prompt")
    user_text = chat_to_text(chat) if chat is not None else "(no context)"

    # assistant：整段“最后一轮回答”作为单一步，末尾插入一次 <extra_0>
    response = solution_str if isinstance(solution_str, str) else str(solution_str)
    assistant_text = response + STEP_TOKEN  # 单一步 → 仅一次 <extra_0>

    # 官方风格：apply_chat_template → encode → model(input_ids=...)  :contentReference[oaicite:1]{index=1}
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    conv_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 与模型卡一致：直接 encode 成 input_ids Tensor（不额外传 attention_mask）
    input_ids = tokenizer.encode(conv_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits  # [1, L, 2]
        probs = F.softmax(logits, dim=-1)  # [1, L, 2]
        mask = (input_ids == step_id)[0]  # [L]
        if mask.sum().item() == 0:
            return 0.0
        # 正类通道 index=1；单一步只有一个 <extra_0>，mean 等价于该位置
        score = float(probs[0, mask, 1].mean().item())

    if not math.isfinite(score):
        score = 0.0
    return score
