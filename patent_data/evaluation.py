"""验证评估：按家族口径，分层报告，区分三类错误，拒绝不追求全对。

指标分层：
- 宏观类（深度 0-1）与叶子/稀有子类分开报告，避免大类指标掩盖细类失败；
- 按语种分别报告（家族主文档语种），暴露跨语一致性问题；
- 拒绝单独统计覆盖率与拒绝原因分布，目标是高精度接受 + 可解释拒绝，而非全拒绝。

错误三分法（需要隐藏真值 truth_status）：
- missing_annotation：人工漏标，系统接受了真值节点；
- known_misclassification：目录可容纳但判错（含 noisy_label）；
- new_topic：目录无法容纳的新技术，正确行为是拒识而非硬归。
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Family, PatentDataset
from .model import HierarchicalModel, Prediction


@dataclass
class MetricAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted_positive: bool, actual_positive: bool) -> None:
        if predicted_positive and actual_positive:
            self.tp += 1
        elif predicted_positive and not actual_positive:
            self.fp += 1
        elif not predicted_positive and actual_positive:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or p + r == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
        }


def _round(value: float) -> float | None:
    return None if value != value else round(value, 4)


@dataclass
class EvaluationResult:
    node_metrics: dict
    level_metrics: dict
    language_metrics: dict
    rare_metrics: dict
    coverage: dict
    abstention_reasons: dict
    error_triage: dict
    calibration: dict
    constraint_violations: list
    cross_lingual: dict

    def to_dict(self) -> dict:
        return {
            "node_metrics": self.node_metrics,
            "level_metrics": self.level_metrics,
            "language_metrics": self.language_metrics,
            "rare_metrics": self.rare_metrics,
            "coverage": self.coverage,
            "abstention_reasons": self.abstention_reasons,
            "error_triage": self.error_triage,
            "calibration": self.calibration,
            "constraint_violations": self.constraint_violations,
            "cross_lingual": self.cross_lingual,
        }


def evaluate(
    model: HierarchicalModel,
    dataset: PatentDataset,
    family_ids: tuple[str, ...],
    use_truth: bool = True,
) -> EvaluationResult:
    catalog = model.catalog
    # 模型只对其版本生效期内的家族负责；更晚目录时代的家族由对应版本模型考核。
    # 跨版本家族仍可评分（versioning.replay_on_frozen_families），但不进版本内指标。
    scoped_ids = tuple(
        fid
        for fid in family_ids
        if dataset.history.version_at(dataset.families[fid].published) == model.version
    )
    predictions: dict[str, Prediction] = {}
    for fid in scoped_ids:
        predictions[fid] = model.predict_family(dataset.families[fid])

    node_metrics = {node: MetricAccumulator() for node in catalog.nodes}
    level_metrics = {"macro": MetricAccumulator(), "leaf": MetricAccumulator()}
    rare_nodes = {nid for nid, state in model.nodes.items() if state.is_rare}
    rare_metrics = {nid: MetricAccumulator() for nid in sorted(rare_nodes)}
    language_metrics: dict[str, MetricAccumulator] = {}
    brier_sum = 0.0
    brier_count = 0
    violations: list[dict] = []
    abstention_reasons: dict[str, int] = {}
    triage: dict[str, list[dict]] = {
        "missing_annotation": [],
        "known_misclassification": [],
        "noisy_label": [],
        "new_topic_handled": [],
        "new_topic_force_fit": [],
    }

    evaluated = 0
    accepted_count = 0
    for fid in scoped_ids:
        family = dataset.families[fid]
        prediction = predictions[fid]
        truth = _reference_set(family, dataset, model.version, use_truth)
        if truth is None:
            continue
        evaluated += 1
        if prediction.is_abstained:
            for reason in prediction.hard_reasons:
                abstention_reasons[reason["code"]] = abstention_reasons.get(reason["code"], 0) + 1
        else:
            accepted_count += 1
        # 节点级与层级指标：家族口径（一次创新只计一次）。
        # 硬拒识时 accepted 为空集合，成员判断自然全为假，无需额外开关。
        for node in catalog.nodes:
            actual = node in truth
            predicted = node in prediction.accepted
            node_metrics[node].add(predicted, actual)
            bucket = "leaf" if node in catalog.leaves() else "macro"
            level_metrics[bucket].add(predicted, actual)
            if node in rare_nodes:
                rare_metrics[node].add(predicted, actual)
            # Brier 校准误差（仅对有分类器的节点）
            if node in prediction.node_scores:
                probability = prediction.node_scores[node]["probability"]
                brier_sum += (probability - actual) ** 2
                brier_count += 1
        # 语种指标：同族每个语种文档单独预测，对照同一份家族真值
        for doc in family.docs:
            single = Family(
                id=family.id,
                docs=[doc],
                labels=family.labels,
                truth=family.truth,
                truth_version=family.truth_version,
                truth_status=family.truth_status,
                truth_note=family.truth_note,
            )
            doc_prediction = model.predict_family(single)
            accumulator = language_metrics.setdefault(doc.language, MetricAccumulator())
            for node in catalog.nodes:
                actual = node in truth
                predicted = node in doc_prediction.accepted
                accumulator.add(predicted, actual)
        # 祖先约束违规（理论上为 0，作为护栏）
        bad = catalog.violations(prediction.accepted)
        if bad:
            violations.append({"family": fid, "violations": bad})
        # 错误三分法
        _triage_family(family, prediction, truth, triage)

    coverage = {
        "families": evaluated,
        "accepted": accepted_count,
        "abstained": evaluated - accepted_count,
        "accept_rate": _round(accepted_count / evaluated) if evaluated else None,
        "scoped_out_of_version": len(family_ids) - len(scoped_ids),
        "taxonomy_version": model.version,
    }
    calibration = {
        "brier": _round(brier_sum / brier_count) if brier_count else None,
        "samples": brier_count,
    }
    cross_lingual = cross_lingual_agreement(model, dataset, scoped_ids)
    return EvaluationResult(
        node_metrics={k: v.to_dict() for k, v in sorted(node_metrics.items())},
        level_metrics={k: v.to_dict() for k, v in sorted(level_metrics.items())},
        language_metrics={k: v.to_dict() for k, v in sorted(language_metrics.items())},
        rare_metrics={k: v.to_dict() for k, v in sorted(rare_metrics.items())},
        coverage=coverage,
        abstention_reasons=abstention_reasons,
        error_triage={k: v for k, v in triage.items()},
        calibration=calibration,
        constraint_violations=violations,
        cross_lingual=cross_lingual,
    )


def family_language(family: Family) -> str:
    return family.primary_doc().language


def _reference_set(
    family: Family,
    dataset: PatentDataset,
    version: int,
    use_truth: bool,
) -> frozenset[str] | None:
    """返回模型版本口径下的参考节点集合。

    隐藏真值按其登记版本（通常为最新版）通过显式迁移规则投影到当前模型版本；
    new_topic 状态家族的真值是目录外节点，投影后自然为空，其行为由
    _triage_family 单独考核。没有隐藏真值时退回迁移后的人工标签。
    未标注不是负类的语义由调用方（仅评估有参考集的家族）保证。
    """
    catalog = dataset.history.catalog(version)
    if use_truth and family.truth:
        inside = {node for node in family.truth if _node_exists_any_version(dataset, node)}
        if family.truth_version is not None and family.truth_version != version:
            moved, _ = dataset.history.migrate(inside, family.truth_version, version)
            inside = set(moved)
        inside = {node for node in inside if node in catalog.nodes}
        return catalog.closure(inside)
    if any(family.labels.values()):
        return dataset.family_label_vector(family, version)
    return None


def _node_exists_any_version(dataset: PatentDataset, node_id: str) -> bool:
    return any(node_id in catalog.nodes for catalog in dataset.history.versions.values())


def _triage_family(
    family: Family,
    prediction: Prediction,
    truth: frozenset[str],
    triage: dict[str, list[dict]],
) -> None:
    status = family.truth_status
    accepted = set() if prediction.is_abstained else set(prediction.accepted)
    if status == "new_topic":
        bucket = "new_topic_handled" if prediction.is_abstained else "new_topic_force_fit"
        triage[bucket].append(
            {
                "family": family.id,
                "accepted": sorted(accepted),
                "reasons": [r["code"] for r in prediction.abstain_reasons],
                "note": family.truth_note,
            }
        )
        return
    if not accepted:
        return
    if status == "missing_annotation":
        # 人工漏标：系统接受集合覆盖了真值即视为成功捕获
        if truth and truth <= accepted:
            triage["missing_annotation"].append(
                {
                    "family": family.id,
                    "accepted": sorted(accepted),
                    "truth": sorted(truth),
                    "note": family.truth_note,
                }
            )
        return
    if status == "noisy_label":
        # 噪声标签：人工登记与真值冲突。系统若按真值（而非错误人工标签）归类，
        # 说明模型没有被脏标签带偏；这里记录供专家核对，不算模型误判。
        matches_truth = bool(truth) and truth <= accepted
        triage["noisy_label"].append(
            {
                "family": family.id,
                "accepted": sorted(accepted),
                "truth": sorted(truth),
                "model_matches_truth": matches_truth,
                "note": family.truth_note,
            }
        )
        return
    # 已知类别误判：接受了不在真值中的目录节点
    wrong = accepted - set(truth)
    if wrong:
        triage["known_misclassification"].append(
            {
                "family": family.id,
                "wrong_nodes": sorted(wrong),
                "truth": sorted(truth),
                "status": status,
            }
        )


def cross_lingual_agreement(
    model: HierarchicalModel,
    dataset: PatentDataset,
    family_ids: tuple[str, ...],
) -> dict:
    """同族跨语一致性：把每个文档单独过模型，测量同家族预测 Jaccard 一致率。

    翻译文本在训练/验证切分中已整体隔离，这里仅在验证家族内部比较，不产生泄漏。
    """
    agreements: list[float] = []
    per_language_prob: dict[str, list[tuple[float, float]]] = {}
    for fid in family_ids:
        family = dataset.families[fid]
        if len(family.docs) < 2:
            continue
        doc_sets: list[frozenset[str]] = []
        for doc in family.docs:
            single = Family(
                id=family.id,
                docs=[doc],
                labels=family.labels,
                truth=family.truth,
                truth_version=family.truth_version,
                truth_status=family.truth_status,
                truth_note=family.truth_note,
            )
            prediction = model.predict_family(single)
            doc_sets.append(frozenset() if prediction.is_abstained else prediction.accepted)
        for i in range(len(doc_sets)):
            for j in range(i + 1, len(doc_sets)):
                union = doc_sets[i] | doc_sets[j]
                jaccard = (
                    len(doc_sets[i] & doc_sets[j]) / len(union) if union else 1.0
                )
                agreements.append(jaccard)
    return {
        "family_pairs": len(agreements),
        "mean_jaccard": _round(sum(agreements) / len(agreements)) if agreements else None,
        "perfect_agreement_rate": _round(
            sum(1 for value in agreements if value == 1.0) / len(agreements)
        )
        if agreements
        else None,
    }
