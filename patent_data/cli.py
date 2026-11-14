"""命令行入口：资料检查、训练验证、批量评分、基线对照、复算。

示例：
  python -m patent_data.cli check demo_data/demo.json
  python -m patent_data.cli validate \
      --manifest demo_data/demo.json --history demo_data/catalog-history.json \
      --taxonomy demo_data/demo-taxonomy.jsonl --truth demo_data/hidden-truth.jsonl \
      --out reports/run
  python -m patent_data.cli score --bundle reports/run/bundle-v1 --manifest demo_data/demo.json \
      --history demo_data/catalog-history.json --taxonomy demo_data/demo-taxonomy.jsonl
  python -m patent_data.cli baseline --manifest demo_data/demo.json \
      --history demo_data/catalog-history.json --taxonomy demo_data/demo-taxonomy.jsonl
"""
from __future__ import annotations

import argparse
import json

from .baseline import CoarseKeywordBaseline, KeywordSearchBaseline, baseline_error_report
from .catalog import load_history
from .data import attach_hidden_truth, build_dataset
from .domain import inspect_tables
from .io import DataError, load
from .pipeline import (
    DEFAULT_FOLDS,
    run_pipeline,
    score_manifest,
)
from .text import ConceptLexicon


def _load_history(args):
    rows = [
        json.loads(line)
        for line in open(args.taxonomy, encoding="utf-8")
        if line.strip()
    ]
    return load_history(args.history, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="技术专利层级分类与新主题拒识")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="资料完整性与身份检查").add_argument("manifest")

    validate = sub.add_parser("validate", help="前推训练与分层验证")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--history", required=True)
    validate.add_argument("--taxonomy", required=True)
    validate.add_argument("--truth", default=None)
    validate.add_argument("--out", required=True)
    validate.add_argument("--epochs", type=int, default=100)
    validate.add_argument("--min-train-pos", type=int, default=3)
    validate.add_argument("--precision-floor", type=float, default=0.85)
    validate.add_argument("--mode", choices=["words", "chars", "both"], default="words")
    validate.add_argument("--stem-latin", action="store_true")

    score = sub.add_parser("score", help="冻结模型包批量评分")
    score.add_argument("--bundle", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--history", required=True)
    score.add_argument("--taxonomy", required=True)
    score.add_argument("--out", default=None)

    baseline = sub.add_parser("baseline", help="历史做法基线对照")
    baseline.add_argument("--manifest", required=True)
    baseline.add_argument("--history", required=True)
    baseline.add_argument("--taxonomy", required=True)
    baseline.add_argument("--version", type=int, default=1)
    baseline.add_argument("--truth", default=None)

    args = parser.parse_args()
    try:
        if args.command == "check":
            print(json.dumps(inspect_tables(load(args.manifest)), ensure_ascii=False, indent=2))
            return
        if args.command == "validate":
            from .model import ModelConfig
            from .text import VectorizerConfig

            config = ModelConfig(
                vectorizer=VectorizerConfig(mode=args.mode, stem_latin=args.stem_latin),
                epochs=args.epochs,
                min_train_pos=args.min_train_pos,
                precision_floor=args.precision_floor,
            )
            report = run_pipeline(
                manifest=args.manifest,
                history_spec=args.history,
                taxonomy_jsonl=args.taxonomy,
                output_dir=args.out,
                truth_path=args.truth,
                folds=DEFAULT_FOLDS,
                config=config,
            )
            _print_validation_summary(report)
            return
        if args.command == "score":
            records = score_manifest(
                bundle_dir=args.bundle,
                history_spec=args.history,
                taxonomy_jsonl=args.taxonomy,
                manifest=args.manifest,
            )
            payload = json.dumps(records, ensure_ascii=False, indent=2)
            if args.out:
                open(args.out, "w", encoding="utf-8").write(payload)
                print(f"已写出 {len(records)} 条预测到 {args.out}")
            else:
                print(payload)
            return
        if args.command == "baseline":
            history = _load_history(args)
            dataset = build_dataset(args.manifest, history)
            if args.truth:
                attach_hidden_truth(dataset, args.truth)
            lexicon = ConceptLexicon.load()
            catalog = history.catalog(args.version)
            family_ids = tuple(dataset.family_ids_by_date())
            coarse = baseline_error_report(
                CoarseKeywordBaseline(catalog, lexicon), dataset, family_ids, args.version
            )
            keyword = baseline_error_report(
                KeywordSearchBaseline(catalog, lexicon), dataset, family_ids, args.version
            )
            print(json.dumps({"coarse": coarse, "keyword": keyword}, ensure_ascii=False, indent=2))
            return
    except (DataError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


def _print_validation_summary(report: dict) -> None:
    summary = {"folds": [], "catalog_diff": None, "replay_change_counts": None}
    for fold in report["folds"]:
        evaluation = fold["evaluation"]
        summary["folds"].append(
            {
                "fold": fold["fold"]["name"],
                "version": fold["fold"]["version"],
                "coverage": evaluation["coverage"],
                "level_metrics": evaluation["level_metrics"],
                "abstention_reasons": evaluation["abstention_reasons"],
                "error_triage_counts": {
                    key: len(value) for key, value in evaluation["error_triage"].items()
                },
                "cross_lingual": evaluation["cross_lingual"],
                "brier": evaluation["calibration"]["brier"],
            }
        )
    if report["catalog_diff"]:
        summary["catalog_diff"] = {
            "added": report["catalog_diff"]["added"],
            "removed": report["catalog_diff"]["removed"],
            "multi_parent_nodes": report["catalog_diff"]["multi_parent_nodes"],
            "migrations": [m["op"] for m in report["catalog_diff"]["migrations"]],
        }
    if report["replay"]:
        summary["replay_change_counts"] = report["replay"]["change_counts"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
