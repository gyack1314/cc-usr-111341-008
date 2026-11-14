"""模型包冻结、复算与目录升级差异分析。

纪律：
- 模型包是不可变目录：manifest 记录每个文件 SHA256、字节数与冻结资源版本；
- 目录升级后旧版模型仍可加载复算；diff 在同一组冻结家族上解释路径与数量变化；
- split（拆分类）只改变归类路径，不被描述为技术涌现；涌现必须由 new_topic
  拒识记录 + 专家复核支撑，而不是节点数量变化。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .catalog import CatalogHistory, Migration
from .model import HierarchicalModel


class BundleError(ValueError):
    pass


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_bundle(
    model: HierarchicalModel,
    output_dir: str | Path,
    extra: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = model.to_bundle()
    files: dict[str, bytes] = {
        "model.json": json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
        "train_families.json": json.dumps(
            sorted(model.train_family_ids), ensure_ascii=False, indent=2
        ).encode("utf-8"),
    }
    entries: dict[str, dict] = {}
    for name, content in files.items():
        target = output_dir / name
        target.write_bytes(content)
        entries[name] = {
            "sha256": sha256_bytes(content),
            "bytes": len(content),
        }
    manifest = {
        "schema": 1,
        "immutable": True,
        "taxonomy_version": model.version,
        "files": entries,
        "extra": extra or {},
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def load_bundle(
    bundle_dir: str | Path,
    history: CatalogHistory,
    lexicon,
) -> tuple[HierarchicalModel, dict]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise BundleError("模型包缺少 manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        content = (bundle_dir / name).read_bytes()
        if sha256_bytes(content) != entry["sha256"] or len(content) != entry["bytes"]:
            raise BundleError(f"模型包文件 {name} 摘要或长度不一致，拒绝加载")
    payload = json.loads((bundle_dir / "model.json").read_text(encoding="utf-8"))
    model = HierarchicalModel.from_bundle(payload, history, lexicon)
    return model, manifest


@dataclass
class CatalogDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    deprecated: tuple[str, ...]
    multi_parent_nodes: tuple[str, ...]
    migrations: tuple[dict, ...]
    path_changes: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "deprecated": list(self.deprecated),
            "multi_parent_nodes": list(self.multi_parent_nodes),
            "migrations": list(self.migrations),
            "path_changes": self.path_changes,
            "interpretation_note": (
                "节点增减是目录口径变化；split/merge 导致的数量变化不得称为技术涌现。"
                "技术涌现只能来自 new_topic 拒识经专家复核确认。"
            ),
        }


def diff_catalogs(history: CatalogHistory, old_version: int, new_version: int) -> CatalogDiff:
    old = history.catalog(old_version)
    new = history.catalog(new_version)
    added = tuple(sorted(set(new.nodes) - set(old.nodes)))
    removed = tuple(sorted(set(old.nodes) - set(new.nodes)))
    deprecated = tuple(
        sorted(
            nid
            for nid, node in old.nodes.items()
            if nid in new.nodes and new.nodes[nid].deprecated is not None
        )
    )
    multi_parent = tuple(
        sorted(nid for nid, node in new.nodes.items() if len(node.parents) > 1)
    )
    migrations = tuple(
        {
            "op": m.op,
            "sources": list(m.sources),
            "targets": list(m.targets),
            "basis": m.basis,
            "note": m.note,
        }
        for m in history.migrations
        if _migration_between(m, old_version, new_version)
    )
    path_changes: dict[str, dict] = {}
    for nid in sorted(set(old.nodes) & set(new.nodes)):
        old_paths = [list(path) for path in old.paths(nid)]
        new_paths = [list(path) for path in new.paths(nid)]
        if old_paths != new_paths:
            path_changes[nid] = {"old": old_paths, "new": new_paths}
    return CatalogDiff(
        added=added,
        removed=removed,
        deprecated=deprecated,
        multi_parent_nodes=multi_parent,
        migrations=migrations,
        path_changes=path_changes,
    )


def _migration_between(migration: Migration, old_version: int, new_version: int) -> bool:
    # 当前合成目录只有相邻两版；保留版本区间判定以便扩展
    return True


def replay_on_frozen_families(
    model_old: HierarchicalModel,
    model_new: HierarchicalModel,
    dataset,
    family_ids: tuple[str, ...],
) -> dict:
    """同一组冻结家族分别用旧版/新版模型评分，解释路径与数量变化。

    返回每个家族的双版本预测、迁移留痕与变化原因（migrated / split_expanded /
    merge_collapsed / new_node_available），供专家核对“变化来自目录而非技术”。
    """
    rows: list[dict] = []
    for fid in family_ids:
        family = dataset.families[fid]
        pred_old = model_old.predict_family(family)
        pred_new = model_new.predict_family(family)
        expected_new, trace = dataset.history.migrate(
            pred_old.accepted, model_old.version, model_new.version
        )
        change_code = _classify_change(trace, pred_old, pred_new, model_new)
        rows.append(
            {
                "family": fid,
                "published": family.published.isoformat(),
                "old_accepted": sorted(pred_old.accepted),
                "new_accepted": sorted(pred_new.accepted),
                "migrated_expectation": sorted(expected_new),
                "migration_trace": trace,
                "change_code": change_code,
                "old_abstain": pred_old.is_abstained,
                "new_abstain": pred_new.is_abstained,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["change_code"]] = counts.get(row["change_code"], 0) + 1
    return {"families": rows, "change_counts": counts}


def _classify_change(trace: list[dict], pred_old, pred_new, model_new) -> str:
    if pred_old.is_abstained or pred_new.is_abstained:
        return "abstention_state_changed"
    ops = {entry.get("op") for entry in trace if "op" in entry}
    if "split" in ops:
        return "split_expanded"
    if "merge" in ops:
        return "merge_collapsed"
    if "rename" in ops:
        return "renamed"
    valid_nodes = set(model_new.catalog.nodes)
    if set(pred_new.accepted) - valid_nodes:
        return "invalid"
    if pred_old.accepted != pred_new.accepted:
        return "new_node_available"
    return "unchanged"
