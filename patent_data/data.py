"""文档、家族与人工标签的载入；family 是训练隔离与统计计量的唯一口径。

同一发明的多语种公开（同 family）在评分与划分时整体移动，翻译文本不会横跨
训练与验证；指标按家族计数，避免把一次创新算成多次。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .catalog import CatalogHistory, _parse_day
from .io import DataError, load


@dataclass
class Document:
    id: str
    family: str
    language: str
    published: date
    priority: date
    text: str

    @classmethod
    def from_row(cls, row: dict) -> "Document":
        return cls(
            id=row["id"],
            family=row["family"],
            language=row["language"],
            published=_parse_day(row["published"]),
            priority=_parse_day(row["priority"]),
            text=row["text"],
        )


@dataclass
class Label:
    id: str
    document: str
    node: str
    taxonomy_version: int
    confirmed_at: date


@dataclass
class Family:
    id: str
    docs: list[Document]
    labels: dict[str, set[str]] = field(default_factory=dict)
    """document_id -> 人工确认节点集合（未列入即未知，不是负类）。"""
    truth: set[str] = field(default_factory=set)
    """隐藏真值节点（原始登记口径，通常为最新版节点名；仅评估使用，训练不读取）。"""
    truth_version: int | None = None
    """隐藏真值登记时使用的目录版本。"""
    truth_note: str = ""
    truth_status: str = "labeled"
    """labeled / missing_annotation / new_topic / noisy_label 等评估口径。"""

    @property
    def published(self) -> date:
        return min(doc.published for doc in self.docs)

    @property
    def priority(self) -> date:
        return min(doc.priority for doc in self.docs)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted({doc.language for doc in self.docs}))

    def primary_doc(self) -> Document:
        """主文档：公开最早；并列时取 id 最小，保证确定性。"""
        return sorted(self.docs, key=lambda d: (d.published, d.id))[0]

    def doc(self, doc_id: str) -> Document:
        for doc in self.docs:
            if doc.id == doc_id:
                return doc
        raise KeyError(doc_id)


@dataclass
class PatentDataset:
    families: dict[str, Family]
    history: CatalogHistory
    documents: dict[str, Document]
    raw_labels: list[Label]

    def family_ids_by_date(self) -> list[str]:
        return sorted(self.families, key=lambda fid: (self.families[fid].published, fid))

    def annotated_families(self) -> list[Family]:
        return [f for f in self.families.values() if any(f.labels.values())]

    def labels_for(self, doc_id: str) -> list[Label]:
        return [label for label in self.raw_labels if label.document == doc_id]

    def check_references(self) -> None:
        """在 io/domain 的引用检查之外，补充版本归属与家族一致性检查。

        标签节点必须属于其登记的 taxonomy_version 目录。注意不要求文档公开日晚于
        该版本生效日：目录上线后对存量家族的回填标注是合法操作（confirmed_at 在
        生效后即可），“当时还没有这个类”由 _node_available 在训练采样时点判断，
        而不是在资料完整性检查阶段拒绝回填。
        """
        for family in self.families.values():
            for doc in family.docs:
                # 公开日必须存在生效目录，否则数据本身有口径缺口
                self.history.version_at(doc.published)
            for doc_id in family.labels:
                family.doc(doc_id)
                for label in self.labels_for(doc_id):
                    catalog = self.history.catalog(label.taxonomy_version)
                    if label.node not in catalog.nodes:
                        raise DataError(
                            f"标签节点 {label.node} 不属于版本 {label.taxonomy_version} 目录"
                        )

    def family_label_vector(self, family: Family, version: int) -> frozenset[str]:
        """把家族上登记的人工标签迁移到指定版本并取祖先闭包。

        不同文档可能登记不同版本标签：逐条迁移后合并；迁移依据在 trace 中留痕。
        未标注家族不在训练样本中使用（PU 语义在 model.py 处理），本方法对无标签
        家族返回空集合，调用方需自行区分“空”与“未知”。
        """
        catalog = self.history.catalog(version)
        merged: set[str] = set()
        for doc_id in family.labels:
            for label in self.labels_for(doc_id):
                moved, _trace = self.history.migrate(
                    {label.node}, label.taxonomy_version, version
                )
                merged.update(moved)
        unknown = merged - set(catalog.nodes)
        if unknown:
            raise DataError(
                f"家族 {family.id} 标签迁移后出现未知节点 {sorted(unknown)}"
            )
        return catalog.closure(merged)

    def migration_trace(self, family: Family, version: int) -> list[dict]:
        """与 family_label_vector 同源，但返回逐条迁移依据供复核。"""
        trace: list[dict] = []
        for doc_id in family.labels:
            for label in self.labels_for(doc_id):
                _, part = self.history.migrate(
                    {label.node}, label.taxonomy_version, version
                )
                trace.extend(part)
        return trace


def build_dataset(manifest_path: str | Path, history: CatalogHistory) -> PatentDataset:
    tables = load(manifest_path)
    documents = {row["id"]: Document.from_row(row) for row in tables["documents"]}
    families: dict[str, Family] = {}
    for doc in documents.values():
        family = families.setdefault(doc.family, Family(id=doc.family, docs=[]))
        family.docs.append(doc)
    raw_labels: list[Label] = []
    for row in tables["labels"]:
        label = Label(
            id=row["id"],
            document=row["document"],
            node=row["node"],
            taxonomy_version=int(row["taxonomy_version"]),
            confirmed_at=_parse_day(row["confirmed_at"]),
        )
        raw_labels.append(label)
        family = families[documents[label.document].family]
        family.labels.setdefault(label.document, set()).add(label.node)
    dataset = PatentDataset(
        families=families,
        history=history,
        documents=documents,
        raw_labels=raw_labels,
    )
    dataset.check_references()
    return dataset


def attach_hidden_truth(dataset: PatentDataset, truth_path: str | Path) -> None:
    """载入隐藏真值（合成数据生成时写出；真实场景由专家复核库提供）。

    训练管线不读取本文件；仅评估模块使用。
    """
    path = Path(truth_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        family = dataset.families[row["family"]]
        family.truth = set(row["nodes"])
        family.truth_version = row.get("version")
        family.truth_status = row.get("status", "labeled")
        family.truth_note = row.get("note", "")
