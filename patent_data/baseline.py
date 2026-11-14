"""历史做法基线：粗大类映射与单语关键词搜索，用于对照展示错在哪里。

两个基线刻意保留业务里真实出现过的失败模式：
1. CoarseKeywordBaseline：只在宏观类层面匹配，交叉技术（edge_ai 同时需要
   network + ai 证据）被漏到任一单一大类；
2. KeywordSearchBaseline：直接搜“网络/network”，神经网络专利因词面命中被混入
   通信网络类——对应人工搜索的污染问题。

正式模型见 model.HierarchicalModel；本模块只做对照，不参与训练选择。
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog
from .data import Family
from .text import ConceptLexicon, normalize


@dataclass
class BaselinePrediction:
    family_id: str
    accepted: frozenset[str]
    reasons: tuple[str, ...]
    evidence_terms: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "family": self.family_id,
            "accepted": sorted(self.accepted),
            "reasons": list(self.reasons),
            "evidence_terms": list(self.evidence_terms),
        }


class CoarseKeywordBaseline:
    """只认识宏观类：命中任一单词就归大类，交叉技术必然漏边。"""

    def __init__(self, catalog: Catalog, lexicon: ConceptLexicon):
        self.catalog = catalog
        self.lexicon = lexicon
        self.macro_roots = [node for node in catalog.roots()]
        # 宏观类 -> 概念词（不含交叉概念 edge_ai，模拟目录太粗）
        self.rules = {
            "network": ("net_transport", "wireless_radio", "low_power", "iot_sensing"),
            "ai": ("ml_general", "neural"),
            "security": ("security",),
        }

    def predict(self, family: Family) -> BaselinePrediction:
        text = normalize(family.primary_doc().text)
        language = family.primary_doc().language
        hits = self.lexicon.match(family.primary_doc().text, language)
        accepted: set[str] = set()
        reasons: list[str] = []
        terms: list[str] = []
        for macro, concepts in self.rules.items():
            if macro not in self.catalog.nodes:
                continue
            matched = [cid for cid in concepts if cid in hits]
            if matched:
                accepted.add(macro)
                terms.extend(term for cid in matched for term in hits[cid])
        # 交叉技术检测：同时出现网络与 AI 证据，但基线没有细类可放
        if "net_transport" in hits and ("ml_general" in hits or "neural" in hits):
            if "edge_ai" not in self.catalog.nodes:
                reasons.append("cross_technology_dropped: 目录缺少交叉细类，仅归入宏观类")
        return BaselinePrediction(
            family_id=family.id,
            accepted=frozenset(self.catalog.closure(accepted)),
            reasons=tuple(reasons),
            evidence_terms=tuple(sorted(set(terms))),
        )


class KeywordSearchBaseline:
    """模拟人工搜索“网络/network”：词面命中即归入 network，不区分语境。"""

    def __init__(self, catalog: Catalog, lexicon: ConceptLexicon):
        self.catalog = catalog
        self.lexicon = lexicon
        self.keywords = {
            "zh": ["网络"],
            "en": ["network"],
            "de": ["Netzwerk"],
            "ja": ["ネットワーク"],
        }

    def predict(self, family: Family) -> BaselinePrediction:
        doc = family.primary_doc()
        text = normalize(doc.text)
        keywords = self.keywords.get(doc.language, [])
        hit_terms = [word for word in keywords if normalize(word) in text]
        accepted: set[str] = set()
        reasons: list[str] = []
        if hit_terms and "network" in self.catalog.nodes:
            accepted.add("network")
            # 神经网络语境检测：只提示不拦截，复刻人工搜索的污染
            neural_hits = self.lexicon.match(doc.text, doc.language).get("neural", [])
            if neural_hits:
                reasons.append(
                    "keyword_pollution: 歧义词命中但语境为神经网络，仍被计入网络类"
                )
        return BaselinePrediction(
            family_id=family.id,
            accepted=frozenset(self.catalog.closure(accepted)),
            reasons=tuple(reasons),
            evidence_terms=tuple(hit_terms),
        )


def baseline_error_report(
    baseline,
    dataset,
    family_ids: tuple[str, ...],
    version: int,
) -> dict:
    """按家族统计基线的漏标（交叉技术）与污染（歧义词误收）。

    真值里目录外的节点（如 v1 时代的 edge_ai）不剔除：它们正是“目录太粗、交叉
    技术被漏掉”的证据，单独标记为 out_of_catalog，与目录内漏判区分。
    """
    dropped: list[dict] = []
    polluted: list[dict] = []
    catalog = dataset.history.catalog(version)
    any_version_nodes = set()
    for cat in dataset.history.versions.values():
        any_version_nodes.update(cat.nodes)
    cross_downgraded: list[dict] = []
    for fid in family_ids:
        family = dataset.families[fid]
        prediction = baseline.predict(family)
        raw_truth = set(family.truth or ())
        # 交叉技术降级：真值含多父/交叉节点（如 edge_ai），旧版目录只能落到宏观祖先。
        # 宏观类可能预测正确，但细类路径信息丢失——这正是“大类太粗、交叉技术被漏”。
        cross_nodes = set()
        for node in raw_truth:
            for cat in dataset.history.versions.values():
                if node in cat.nodes and len(cat.nodes[node].parents) > 1:
                    cross_nodes.add(node)
        if cross_nodes:
            cross_downgraded.append(
                {
                    "family": fid,
                    "cross_nodes": sorted(cross_nodes),
                    "baseline_accepted": sorted(prediction.accepted),
                }
            )
        truly_outside = {node for node in raw_truth if node not in any_version_nodes}
        migratable = raw_truth - truly_outside
        if family.truth_version is not None and family.truth_version != version and migratable:
            moved, _ = dataset.history.migrate(migratable, family.truth_version, version)
            truth = set(moved)
        else:
            truth = {node for node in migratable if node in catalog.nodes}
        # 迁移规则未把某节点带到目标版本（如 v2 新增细类无法回到 v1）：
        # 属于“当时目录不够细”，与 truly_outside 合并展示
        truth_outside = set(truly_outside)
        missed = truth - set(prediction.accepted)
        if missed or truth_outside:
            dropped.append(
                {
                    "family": fid,
                    "missed": sorted(missed),
                    "out_of_catalog": sorted(truth_outside),
                    "reasons": list(prediction.reasons),
                }
            )
        if any("pollution" in reason for reason in prediction.reasons):
            polluted.append({"family": fid, "terms": list(prediction.evidence_terms)})
    return {
        "baseline": baseline.__class__.__name__,
        "families": len(family_ids),
        "missed_families": dropped,
        "polluted_families": polluted,
        "cross_technology_downgraded": cross_downgraded,
        "missed_count": len(dropped),
        "polluted_count": len(polluted),
        "cross_technology_count": len(cross_downgraded),
    }
