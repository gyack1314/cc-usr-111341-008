"""批量评分管线：训练、前推验证、双版本复算与报告落盘。

所有选择（词表、校准、阈值、拒识参数）只在训练/开发段完成，验证段在冻结后打开。
CLI 与 Python 接口共用本模块，输出到 reports/，模型包写入 models/，全部可复算。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import CatalogHistory, load_history
from .data import PatentDataset, attach_hidden_truth, build_dataset
from .evaluation import EvaluationResult, evaluate
from .model import HierarchicalModel, ModelConfig
from .split import TemporalSplit, assert_family_isolation, make_split
from .text import ConceptLexicon
from .versioning import diff_catalogs, load_bundle, replay_on_frozen_families, save_bundle


@dataclass
class FoldSpec:
    name: str
    version: int
    train_end: str
    dev_end: str
    test_end: str | None = None


DEFAULT_FOLDS = (
    FoldSpec("v1_fold", 1, "2021-06-30", "2021-09-30", "2022-12-31"),
    FoldSpec("v2_fold", 2, "2022-12-31", "2023-06-30", None),
)


def load_inputs(
    manifest: str | Path,
    history_spec: str | Path,
    taxonomy_jsonl: str | Path,
    truth_path: str | Path | None = None,
) -> tuple[PatentDataset, CatalogHistory, ConceptLexicon]:
    taxonomy_rows = [
        json.loads(line)
        for line in Path(taxonomy_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = load_history(history_spec, taxonomy_rows)
    dataset = build_dataset(manifest, history)
    if truth_path is not None:
        attach_hidden_truth(dataset, truth_path)
    lexicon = ConceptLexicon.load()
    return dataset, history, lexicon


def run_fold(
    dataset: PatentDataset,
    history: CatalogHistory,
    lexicon: ConceptLexicon,
    spec: FoldSpec,
    config: ModelConfig | None = None,
) -> tuple[HierarchicalModel, TemporalSplit, EvaluationResult]:
    config = config or ModelConfig(min_train_pos=3, rare_train_pos=8, epochs=100)
    split = make_split(
        dataset,
        spec.train_end,
        spec.dev_end,
        name=spec.name,
        test_end=spec.test_end,
    )
    assert_family_isolation([split])
    model = HierarchicalModel.train(
        dataset, split.train, split.dev, version=spec.version, config=config, lexicon=lexicon
    )
    result = evaluate(model, dataset, split.test)
    return model, split, result


def run_pipeline(
    manifest: str | Path,
    history_spec: str | Path,
    taxonomy_jsonl: str | Path,
    output_dir: str | Path,
    truth_path: str | Path | None = None,
    folds: tuple[FoldSpec, ...] = DEFAULT_FOLDS,
    config: ModelConfig | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, history, lexicon = load_inputs(
        manifest, history_spec, taxonomy_jsonl, truth_path
    )
    report: dict = {
        "schema": 1,
        "folds": [],
        "catalog_diff": None,
        "replay": None,
    }
    models: dict[int, HierarchicalModel] = {}
    for spec in folds:
        model, split, result = run_fold(dataset, history, lexicon, spec, config)
        models[spec.version] = model
        bundle_dir = output_dir / f"bundle-v{spec.version}"
        save_bundle(model, bundle_dir, extra={"fold": asdict(spec)})
        fold_report = {
            "fold": asdict(spec),
            "split": split.summary(),
            "evaluation": result.to_dict(),
            "bundle": str(bundle_dir),
        }
        report["folds"].append(fold_report)
    # 目录升级差异 + 同冻结家族双版本复算
    if len(models) >= 2:
        versions = sorted(models)
        old_v, new_v = versions[0], versions[-1]
        report["catalog_diff"] = diff_catalogs(history, old_v, new_v).to_dict()
        frozen_ids = tuple(
            fid
            for fid in dataset.family_ids_by_date()
            if dataset.families[fid].published >= history.catalog(new_v).effective
        )
        report["replay"] = replay_on_frozen_families(
            models[old_v], models[new_v], dataset, frozen_ids
        )
    report_path = output_dir / "validation-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def score_manifest(
    bundle_dir: str | Path,
    history_spec: str | Path,
    taxonomy_jsonl: str | Path,
    manifest: str | Path,
) -> list[dict]:
    """冻结模型包的批量评分接口：加载已发布模型，对新清单逐家族输出预测记录。"""
    taxonomy_rows = [
        json.loads(line)
        for line in Path(taxonomy_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = load_history(history_spec, taxonomy_rows)
    lexicon = ConceptLexicon.load()
    model, _manifest_meta = load_bundle(bundle_dir, history, lexicon)
    dataset = build_dataset(manifest, history)
    records = []
    for fid in dataset.family_ids_by_date():
        prediction = model.predict_family(dataset.families[fid])
        records.append(prediction.to_record())
    return records
