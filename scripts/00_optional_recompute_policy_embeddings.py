from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from common import RAW, ensure_output_dir


CARBON_QUERY = [
    "碳中和",
    "碳达峰",
    "绿色低碳发展",
    "节能减排",
    "低碳转型",
    "零碳排放",
    "双碳战略",
    "可持续发展",
    "碳捕集与封存",
    "碳捕集与利用",
    "碳足迹",
    "清洁能源",
    "能源结构调整",
    "绿色制造",
    "绿色建筑",
    "工业脱碳",
    "电气化转型",
    "储能技术",
    "碳市场",
    "碳交易",
    "碳配额",
    "碳税",
    "碳核查",
    "碳定价",
    "碳信用",
    "自愿减排机制",
    "绿色金融",
    "碳金融",
    "碳资产",
    "碳中和基金",
    "碳披露",
    "ESG报告",
    "碳汇",
    "森林碳汇",
    "土壤碳汇",
    "自然解决方案",
    "生态修复",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optionally recompute BGE embeddings and carbon similarity.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5")
    args = parser.parse_args()

    source = RAW / "combined_policy_data.xlsx"
    if not source.exists():
        raise FileNotFoundError(f"Missing raw policy file: {source}")

    outdir = ensure_output_dir("00_optional_recompute_policy_embeddings")
    df = pd.read_excel(source)
    if "policy_content" not in df.columns:
        raise ValueError("combined_policy_data.xlsx must contain a policy_content column.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    policy_embeddings = model.encode(
        df["policy_content"].fillna("").astype(str).tolist(),
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        batch_size=64,
    )
    query_embeddings = model.encode(CARBON_QUERY, convert_to_tensor=True, device=device)
    similarities = torch.max(util.pytorch_cos_sim(policy_embeddings, query_embeddings), dim=1).values.cpu().numpy()
    df["carbon_similarity"] = similarities

    embeddings_np = policy_embeddings.cpu().numpy()
    df["embedding"] = [np.asarray(v, dtype=float).tolist() for v in embeddings_np]
    df.to_csv(outdir / "policies_with_carbon_similarity_recomputed.csv", index=False)
    df[df["carbon_similarity"] > args.threshold].to_csv(
        outdir / f"filtered_policies_above_{int(args.threshold * 100)}pct_recomputed.csv",
        index=False,
    )
    print(f"Saved recomputed embedding outputs to {outdir}")


if __name__ == "__main__":
    main()
