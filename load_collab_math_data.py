#!/usr/bin/env python3
# raw_9_1_to_jsonl_and_parquet.py
# 原样下载 collabllm/collabllm-multiturn-math-hard-large → 随机打乱(seed=42) → 9/1 切分
# 然后分别保存为 JSONL 和 Parquet（字段不做任何修改）

import os, argparse, json
from typing import Dict, Any
from datasets import load_dataset, Dataset, DatasetDict

DATASET_ID = "collabllm/collabllm-multiturn-math-hard-large"
OUT_DIR = "./data/collabllm-math-hard-large"
SEED = 42
TEST_RATIO = 0.1  # 9/1


def add_extra(example: Dict[str, Any], prompt_key: str) -> Dict[str, Any]:
    # 将原 prompt（可以是str或聊天列表）放进 extra_info；原字段不动
    p = example.get(prompt_key, None)
    return {"extra_info": {"prompt": p}}


SUFFIX = " /no_think"


def append_to_last_user(prompt: Any) -> Any:
    # collab 的 prompt 可能是 list[{"role","content"}, ...] 或 str
    if isinstance(prompt, list):
        # 找最后一个 role=="user" 的条目；找不到就对最后一条 content 追加
        idx = None
        for i in range(len(prompt) - 1, -1, -1):
            if isinstance(prompt[i], dict) and prompt[i].get("role") == "user":
                idx = i
                break
        if idx is None:
            # 兜底：最后一条
            if prompt and isinstance(prompt[-1], dict) and "content" in prompt[-1]:
                prompt[-1]["content"] = str(prompt[-1]["content"]) + SUFFIX
            return prompt
        prompt[idx]["content"] = str(prompt[idx].get("content", "")) + SUFFIX
        return prompt
    elif isinstance(prompt, str):
        return prompt + SUFFIX
    else:
        # 其他格式：转成字符串后直接追加
        return json.dumps(prompt, ensure_ascii=False) + SUFFIX


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 从 HF 原样读取（不做 map/filter/重命名）
    ds_all: Dataset = load_dataset(DATASET_ID,
                                   split="train")  # 官方: load_dataset 从 Hub 加载。:contentReference[oaicite:1]{index=1}

    ds_all: Dataset = ds_all.map(lambda ex: add_extra(ex, "prompt"))

    def _map_fn(ex):
        ex = dict(ex)
        ex["prompt"] = append_to_last_user(ex.get("prompt"))
        return ex

    ds_all = ds_all.map(_map_fn, num_proc=1)

    # 2) 打乱 + 9/1 切分（仅改变顺序，不改内容）
    ds_all = ds_all.shuffle(seed=SEED)  # 官方: shuffle 支持固定 seed。:contentReference[oaicite:2]{index=2}
    dsd: DatasetDict = ds_all.train_test_split(
        test_size=TEST_RATIO, seed=SEED, shuffle=False  # 先 shuffle 再 split。:contentReference[oaicite:3]{index=3}
    )

    # 3) 保存 JSONL（默认就是 JSON Lines，字段原样）
    train_jsonl = os.path.join(OUT_DIR, "train.jsonl")
    val_jsonl = os.path.join(OUT_DIR, "test.jsonl")
    dsd["train"].to_json(train_jsonl)  # 官方: Dataset.to_json 默认 JSONL。:contentReference[oaicite:4]{index=4}
    dsd["test"].to_json(val_jsonl)

    # 4) 保存 Parquet（字段原样）
    train_parquet = os.path.join(OUT_DIR, "train.parquet")
    val_parquet = os.path.join(OUT_DIR, "test.parquet")
    dsd["train"].to_parquet(train_parquet)  # 官方: Dataset.to_parquet 导出 Parquet。:contentReference[oaicite:5]{index=5}
    dsd["test"].to_parquet(val_parquet)

    print(f"[OK] train.jsonl → {train_jsonl} (rows={len(dsd['train'])})")
    print(f"[OK] test.jsonl → {val_jsonl} (rows={len(dsd['test'])})")
    print(f"[OK] train.parquet → {train_parquet}")
    print(f"[OK] test.parquet → {val_parquet}")


if __name__ == "__main__":
    main()
