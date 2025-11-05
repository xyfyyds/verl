# file: aqmi_cumi_reward.py
# Reward = AQMI + CUMI  (equal weights, no extra terms)
import os
import re
import math
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer

hasSentenceEncoder = True
tokenizerCache = None
policyForScoring = None
sentenceEncoder = None


def lazyInit(modelNameOrPath: str, sentenceEncoderName: str = None):
    global tokenizerCache, policyForScoring, sentenceEncoder
    if tokenizerCache is None or policyForScoring is None:
        tokenizerCache = AutoTokenizer.from_pretrained(modelNameOrPath, use_fast=True)
        policyForScoring = AutoModelForCausalLM.from_pretrained(
            modelNameOrPath,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    if sentenceEncoder is None and hasSentenceEncoder:
        # 轻量句向量模型：效果/速度平衡良好（官方卡片建议）:contentReference[oaicite:2]{index=2}
        sentenceEncoder = SentenceTransformer(sentenceEncoderName)


@torch.no_grad()
def avgLogprob(contextText: str, answerText: str, maxLen: int = 4096) -> float:
    """
    不使用 BOS。
    若 context 为空，则跳过答案的第一个 token（因为它没有有效的条件前缀），
    仅对答案的后续 token 计算平均 log-prob。
    """
    tok = tokenizerCache
    model = policyForScoring

    ctx_ids = tok(contextText, add_special_tokens=False, return_tensors="pt").input_ids
    ans_ids = tok(answerText, add_special_tokens=False, return_tensors="pt").input_ids

    # 只截断历史，保留答案完整
    total_len = ctx_ids.size(1) + ans_ids.size(1)
    if total_len > maxLen:
        keep_ctx = max(0, maxLen - ans_ids.size(1))
        ctx_ids = ctx_ids[:, -keep_ctx:]

    input_ids = torch.cat([ctx_ids, ans_ids], dim=1)
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]  # 预测下一个 token 的分布
    shift_labels = input_ids[:, 1:]  # 对齐到“要预测的目标 token”

    ctx_len = ctx_ids.size(1)
    ans_len = ans_ids.size(1)

    # 不用 BOS：当 context 为空时，跳过答案首 token
    if ctx_len == 0:
        # 首个可对齐的位置从 0 开始，但跳过第一个答案 token
        start = 0
        use_len = max(0, ans_len - 1)  # 只统计后续 token
        if use_len == 0:
            # 答案只有 1 个 token，且无上下文时无法对齐任何 token，返回 0 分
            return 0.0
        ans_logits = logits[:, start: start + use_len, :]
        ans_labels = shift_labels[:, start: start + use_len]
    else:
        # 有上下文：答案第一个 token 的分布位于 ctx_len-1 处开始
        start = ctx_len - 1
        use_len = ans_len
        ans_logits = logits[:, start: start + use_len, :]
        ans_labels = shift_labels[:, start: start + use_len]

    logprobs = torch.log_softmax(ans_logits, dim=-1)
    token_lp = logprobs.gather(dim=-1, index=ans_labels.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.mean().item())


def splitToFragments(text: str, maxFragChars: int = 80, strideChars: int = 60):
    """
    将一段文本切成**片段（span）**：
      1) 先按句/分隔符粗切：中英句末与常见停顿符号。
      2) 对过长的片段再做字符级滑窗（避免整段太长）。
    返回: List[(start, end, fragText)]
    说明：不依赖分词工具，中英混排通用；对中文按字符数滑窗，对英文同样适用。
    """
    # 句/子句粗切
    seps = r"[。！？!?；;：:，,、\n]+"
    chunks = [c for c in re.split(seps, text) if c and c.strip()]
    spans = []
    cursor = 0
    # 用原文索引回填 start/end
    # 逐个在原文中定位（保留多次出现时的顺序匹配）
    idx = 0
    for chunk in chunks:
        # 在 text[cursor:] 中找 chunk 的起点
        pos = text.find(chunk, cursor)
        if pos == -1:
            continue
        start = pos
        end = pos + len(chunk)
        cursor = end  # 向后推进

        # 对过长片段做滑窗
        length = end - start
        if length <= maxFragChars:
            spans.append((start, end, text[start:end]))
        else:
            s = start
            while s < end:
                e = min(s + maxFragChars, end)
                spans.append((s, e, text[s:e]))
                if e == end:
                    break
                s = s + strideChars

    return spans


def pickFragmentsToRemove(messages, questionText: str, topK: int = 6, charBudget: int = None):
    """
    在整个历史对话（去掉最后一条 user）范围内，挑选**最相关的片段**集合 R_t。
    评分：句向量余弦相似度（若无 sentence encoder，可退化到 TF/关键词重叠）。
    选择策略：优先 Top-K；若提供 charBudget（字符预算），则在 Top-K 中截断到预算上限。
    返回: List[(msgIdx, startChar, endChar)]
    """
    # 1) 汇总所有片段，并记录其在各自 message 内的字符区间
    allFrags = []  # (score占位, msgIdx, start, end)
    flatTexts = []
    indices = []
    for mi, m in enumerate(messages):
        content = m.get("content", "") or ""
        if not content.strip():
            continue
        spans = splitToFragments(content)  # [(s,e,fragText)]
        for (s, e, frag) in spans:
            flatTexts.append(frag)
            indices.append((mi, s, e))

    if not flatTexts:
        return []

    # 2) 打分：嵌入余弦；若没有 sentenceEncoder，则退化为简单词交集得分
    if sentenceEncoder is not None:
        qEmb = sentenceEncoder.encode([questionText], normalize_embeddings=True)
        fEmb = sentenceEncoder.encode(flatTexts, normalize_embeddings=True)
        sims = (qEmb @ fEmb.T)[0]  # [num_frags]
        scores = sims.tolist()
    else:
        # 退化：用字符/词集合交集比率（非常简化，但不依赖额外包）
        def lexScore(a, b):
            # 统一小写，去掉标点，取集合交并比例
            na = re.sub(r"[\W_]+", " ", a).lower().split()
            nb = re.sub(r"[\W_]+", " ", b).lower().split()
            sa, sb = set(na), set(nb)
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb);
            union = len(sa | sb)
            return inter / union

        scores = [lexScore(questionText, t) for t in flatTexts]

    # 3) 选 Top-K 片段（可选字符预算）
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = order[:topK]
    # 若给了预算，就按顺序累加长度直到达到预算
    if charBudget is not None:
        acc = 0
        budgetKeep = []
        for i in keep:
            mi, s, e = indices[i]
            l = max(0, e - s)
            if acc + l > charBudget and budgetKeep:
                break
            acc += l
            budgetKeep.append(i)
        if budgetKeep:
            keep = budgetKeep

    # 4) 返回消息内的字符坐标
    toRemove = [(indices[i][0], indices[i][1], indices[i][2]) for i in keep]
    # 可按出现顺序排序，便于后续从后向前删除
    toRemove.sort(key=lambda x: (x[0], x[1]))
    return toRemove


def buildContextsWithTemplate(messages, topK: int = 6, charBudget: int = None):
    """
    基于 Qwen chat template 生成三份上下文：
      fullCtx   : 历史 + 本轮 user
      ctxWoutQ  : 去掉最后一条 user 的历史
      ablatedCtx: 从历史中**删除若干片段**（跨消息的 span），再加回本轮 user
    """
    tok = tokenizerCache
    assert messages and messages[-1]["role"] == "user"
    # 完整上下文
    fullCtx = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 无本轮问句
    messagesWoutQ = messages[:-1]
    ctxWoutQ = "" if len(messagesWoutQ) == 0 else tok.apply_chat_template(
        messagesWoutQ, tokenize=False, add_generation_prompt=False
    )

    if len(messagesWoutQ) == 0:
        # 没历史：消融上下文同样为空
        ablatedCtx = ""
        return fullCtx, ctxWoutQ, ablatedCtx

    # 片段级选择：在“历史整体”上，选 Top-K 片段进行删除
    questionText = messages[-1]["content"].strip()
    removeSpans = pickFragmentsToRemove(messagesWoutQ, questionText, topK=topK, charBudget=charBudget)

    # 在每条 message 的 content 中按字符坐标删掉片段
    # ——按每个 msg 内从后往前删除，避免坐标失效
    newHist = []
    curIdx = 0
    grouped = {}
    for (mi, s, e) in removeSpans:
        grouped.setdefault(mi, []).append((s, e))
    for mi, m in enumerate(messagesWoutQ):
        content = m.get("content", "") or ""
        spans = grouped.get(mi, [])
        if spans:
            # 从后往前删
            spans.sort(key=lambda x: x[0], reverse=True)
            buf = content
            for (s, e) in spans:
                if 0 <= s < e <= len(buf):
                    buf = buf[:s] + buf[e:]
            # 清理多余空白
            buf = re.sub(r"\s+\n", "\n", buf).strip()
            if buf:
                newHist.append({"role": m["role"], "content": buf})
            else:
                # 整条被删空则跳过
                pass
        else:
            newHist.append(m)

    messagesAbl = newHist + [messages[-1]]
    ablatedCtx = tok.apply_chat_template(messagesAbl, tokenize=False, add_generation_prompt=False)
    return fullCtx, ctxWoutQ, ablatedCtx


def aqmiWithTemplate(messages, answerText: str) -> float:
    fullCtx, ctxWoutQ, _ = buildContextsWithTemplate(messages, topK=3)
    lpWithQ = avgLogprob(fullCtx, answerText)
    lpNoQ = avgLogprob(ctxWoutQ, answerText)
    return lpWithQ - lpNoQ


def cumiWithTemplate(messages, answerText: str, topK: int = 3) -> float:
    if not messages or len(messages) <= 1:
        return 0.0

    fullCtx, _, ablatedCtx = buildContextsWithTemplate(messages, topK=topK)
    lpFull = avgLogprob(fullCtx, answerText)
    lpAbl = avgLogprob(ablatedCtx, answerText)
    return lpFull - lpAbl


def compute_score(dataSource: str, solutionStr: str, groundTruth: str, extra_info: dict = None):
    """
    VERL 契约：返回一个标量作为奖励（或列表）。此处返回 AQMI + CUMI。
    你需要在 config 中把 custom_reward_function.name 设为 computeScore。:contentReference[oaicite:4]{index=4}

    extra_info 需至少包含：
      - "prompt": List[{"role":"user"|"assistant", "content": str}, ...]
                   最后一个元素的 role 必为 "user"
      - "model_name_or_path": 用于打分的（冻结）语言模型
      - "device": "auto"（可选，默认自动）
      - "top_k": 选 R_t 的句子数（可选，默认 3）
    """
    assert extra_info is not None and "prompt" in extra_info, "extra_info.prompt 缺失"

    modelName = os.environ.get("MODEL_PATH", "/home/ma-user/work/models/Qwen/Qwen3-8B")
    assert modelName, "extra_info.model_name_or_path 缺失"

    sentenceEncoderName = os.environ.get("SENTENCE_ENCODER_PATH", "sentence-transformers/all-MiniLM-L6-v2")

    lazyInit(modelNameOrPath=modelName, sentenceEncoderName=sentenceEncoderName)

    messages = extra_info["prompt"]  # 最后一条 role=user
    # 等权：AQMI + CUMI
    valueAqmi = aqmiWithTemplate(messages, solutionStr)
    valueCumi = cumiWithTemplate(messages, solutionStr, topK=extra_info.get("top_k", 3))
    return float(valueAqmi + valueCumi)
