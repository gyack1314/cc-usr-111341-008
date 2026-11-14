"""层级多标签分类 + 校准 + 拒识（纯标准库实现，确定性可复算）。

方法组合（在开发段冻结，验证段不参与任何选择）：
1. 文本：text.TfidfVectorizer，词/字符两种表示可比较；
2. 层级学习：每个节点一个局部二项逻辑回归（local classifier per node），
   PU 语义下未标注不进负样本，负样本只取“该时点该节点已可选且被标为其他类”的家族；
3. 校准：开发段（稀有节点在子树内池化）Platt scaling，输出概率而非裸分数；
4. 拒识：短摘要 / OOD（质心余弦 + 词典覆盖）/ 证据不足（阈值下校准概率不足），
   全部带原因；排序高分本身不构成归类证据。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import Catalog, CatalogHistory
from .data import Family, PatentDataset
from .text import (
    ConceptLexicon,
    SparseVector,
    TfidfVectorizer,
    VectorizerConfig,
    dot,
    evidence_support,
)


class ModelError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# 纯 Python 二项逻辑回归
# --------------------------------------------------------------------------- #
@dataclass
class LogisticModel:
    weights: dict[int, float] = field(default_factory=dict)
    bias: float = 0.0
    n_features: int = 0
    n_train_pos: int = 0
    n_train_neg: int = 0

    def score(self, vector: SparseVector) -> float:
        return self.bias + sum(
            self.weights.get(idx, 0.0) * value
            for idx, value in zip(vector.indices, vector.values)
        )

    def predict_proba_raw(self, vector: SparseVector) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.score(vector)))))

    def fit(
        self,
        rows: list[tuple[SparseVector, int]],
        epochs: int = 120,
        learning_rate: float = 0.15,
        l2: float = 1e-3,
    ) -> "LogisticModel":
        """加权 SGD。类别不平衡用逆频率权重，遍历顺序固定保证确定性。"""
        if not rows:
            raise ModelError("训练样本为空")
        self.n_features = max(max(vector.indices, default=-1) for vector, _ in rows) + 1
        pos = sum(label for _, label in rows)
        neg = len(rows) - pos
        self.n_train_pos, self.n_train_neg = pos, neg
        if pos == 0 or neg == 0:
            raise ModelError("节点训练数据只有一个类别，无法拟合二分类器")
        weight_pos = len(rows) / (2.0 * pos)
        weight_neg = len(rows) / (2.0 * neg)
        rng = _DeterministicRng(seed=17)
        order = list(range(len(rows)))
        for epoch in range(epochs):
            rng.shuffle(order)
            decay = learning_rate / (1.0 + 0.02 * epoch)
            for row_index in order:
                vector, label = rows[row_index]
                prediction = self.predict_proba_raw(vector)
                instance_weight = weight_pos if label else weight_neg
                residual = (prediction - label) * instance_weight
                for idx, value in zip(vector.indices, vector.values):
                    current = self.weights.get(idx, 0.0)
                    self.weights[idx] = current - decay * (
                        residual * value + l2 * current
                    )
                self.bias -= decay * residual
        return self

    def to_dict(self) -> dict:
        return {
            "weights": {str(idx): value for idx, value in sorted(self.weights.items())},
            "bias": self.bias,
            "n_features": self.n_features,
            "n_train_pos": self.n_train_pos,
            "n_train_neg": self.n_train_neg,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LogisticModel":
        model = cls(
            weights={int(idx): float(value) for idx, value in payload["weights"].items()},
            bias=float(payload["bias"]),
            n_features=int(payload["n_features"]),
            n_train_pos=int(payload["n_train_pos"]),
            n_train_neg=int(payload["n_train_neg"]),
        )
        return model


class _DeterministicRng:
    """固定种子的 LCG，避免依赖 random 全局状态，保证跨机器逐位一致。"""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_int(self, bound: int) -> int:
        # Numerical Recipes LCG
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state % bound

    def shuffle(self, items: list[int]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.next_int(i + 1)
            items[i], items[j] = items[j], items[i]


# --------------------------------------------------------------------------- #
# Platt 校准与阈值
# --------------------------------------------------------------------------- #
@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    pooled: bool = False

    def calibrate(self, raw_score: float) -> float:
        # 标准 Platt：p = sigmoid(a*s + b)，a 应近似为正
        return 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, -(self.a * raw_score + self.b)))))

    def fit(self, scores: list[tuple[float, int]], prior_shift: bool = True) -> "PlattCalibrator":
        """L2 正则梯度下降拟合 Platt 参数（比裸 Newton 稳定，参数有界）。

        开发段只有单一类别时不做伪校准：退化为带先验偏置的 sigmoid，使概率保留
        区分度但整体偏向已知类别，阈值选择会自然收紧。
        """
        if not scores:
            self.a, self.b = 1.0, 0.0
            return self
        positives = sum(label for _, label in scores)
        negatives = len(scores) - positives
        if positives == 0 or negatives == 0:
            # 单类开发集：保留原始排序（a=1），偏置反映先验
            self.a = 1.0
            self.b = -math.log((negatives + 1.0) / (positives + 1.0))
            return self
        # Platt 1999 的先验修正，避免开发段类别比例偏差
        if prior_shift:
            hi = (positives + 1.0) / (positives + 2.0)
            lo = 1.0 / (negatives + 2.0)
        else:
            hi, lo = 1.0, 0.0
        targets = [hi if label else lo for _, label in scores]
        score_scale = max(1e-6, max(abs(s) for s, _ in scores))
        a, b = 0.0, -math.log((negatives + 1.0) / (positives + 1.0))
        l2 = 1.0 / max(1, len(scores))
        learning_rate = 0.5
        for _ in range(300):
            ga = gb = 0.0
            for (score, _), target in zip(scores, targets):
                normalized = score / score_scale
                p = 1.0 / (
                    1.0 + math.exp(max(-30.0, min(30.0, -(a * normalized + b))))
                )
                residual = p - target
                ga += residual * normalized + l2 * a
                gb += residual + l2 * b
            a -= learning_rate * ga / len(scores)
            b -= learning_rate * gb / len(scores)
        # 恢复到原始分数量纲，并对极端参数设护栏；a 必须非负（概率随分数单调）
        self.a = max(0.0, min(10.0, a / score_scale))
        self.b = max(-20.0, min(20.0, b))
        return self

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "pooled": self.pooled}

    @classmethod
    def from_dict(cls, payload: dict) -> "PlattCalibrator":
        return cls(a=float(payload["a"]), b=float(payload["b"]), pooled=bool(payload["pooled"]))


@dataclass
class NodeState:
    node_id: str
    model: LogisticModel
    calibrator: PlattCalibrator
    threshold: float
    train_pos_families: int
    pooled: bool
    centroid: SparseVector
    ood_quantile: float
    lex_coverage_quantile: float
    is_rare: bool
    centroids_by_language: dict[str, SparseVector] = field(default_factory=dict)
    ood_quantile_by_language: dict[str, float] = field(default_factory=dict)
    lex_quantile_by_language: dict[str, float] = field(default_factory=dict)

    def ood_profile(self, language: str) -> tuple[SparseVector, float, float]:
        """返回该语种的 OOD 剖面；训练段未见过的语种回退全局剖面。"""
        if language in self.centroids_by_language:
            return (
                self.centroids_by_language[language],
                self.ood_quantile_by_language[language],
                self.lex_quantile_by_language[language],
            )
        return self.centroid, self.ood_quantile, self.lex_coverage_quantile


# --------------------------------------------------------------------------- #
# 配置与层级模型
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    epochs: int = 120
    learning_rate: float = 0.15
    min_train_pos: int = 4
    rare_train_pos: int = 8
    precision_floor: float = 0.85
    min_tokens: int = 8
    ood_quantile: float = 0.05
    lex_coverage_quantile: float = 0.05
    ood_floor_scale: float = 0.8
    """OOD 地板缩放：实际拒识线 = 分位数 × 该系数，给小样本分位估计留稳健裕度。"""
    enforce_hierarchy: bool = True
    calibration_pool: bool = True


def family_text(family: Family) -> str:
    doc = family.primary_doc()
    return doc.text


def family_language(family: Family) -> str:
    return family.primary_doc().language


@dataclass
class HierarchicalModel:
    config: ModelConfig
    history: CatalogHistory
    version: int
    vectorizer: TfidfVectorizer
    nodes: dict[str, NodeState]
    lexicon: ConceptLexicon | None
    train_family_ids: tuple[str, ...]
    supported_concept_indices: frozenset[int] = frozenset()

    @property
    def catalog(self) -> Catalog:
        return self.history.catalog(self.version)

    # ---- 训练 ----
    @classmethod
    def train(
        cls,
        dataset: PatentDataset,
        train_ids: tuple[str, ...],
        dev_ids: tuple[str, ...],
        version: int,
        config: ModelConfig,
        lexicon: ConceptLexicon | None = None,
    ) -> "HierarchicalModel":
        catalog = dataset.history.catalog(version)
        train_families = [dataset.families[fid] for fid in train_ids]
        dev_families = [dataset.families[fid] for fid in dev_ids]
        # 1) 词表拟合训练段全部语种文档（同家族翻译整体在训练段内，不泄漏到验证）。
        #    只看主文档会让非主语种词面饿死；多语文本共享概念通道以保证跨语一致。
        vectorizer = TfidfVectorizer(config=config.vectorizer, lexicon=lexicon)
        fit_records = [
            (doc.text, doc.language)
            for family in train_families
            for doc in family.docs
        ]
        vectorizer.fit(fit_records)
        # 训练/开发向量按 (family, document) 建：每个语种文档都有向量，
        # 家族标签复制到同族全部文档（翻译不是独立创新，标签口径仍按家族）。
        train_vectors = {
            (f.id, doc.id): vectorizer.transform_one(doc.text, doc.language)
            for f in train_families
            for doc in f.docs
        }
        dev_vectors = {
            (f.id, doc.id): vectorizer.transform_one(doc.text, doc.language)
            for f in dev_families
            for doc in f.docs
        }
        # 2) 训练标签向量（迁移 + 闭包），未标注家族不进任何节点的训练集
        train_labels = {
            f.id: dataset.family_label_vector(f, version)
            for f in train_families
            if any(f.labels.values())
        }
        dev_labels = {
            f.id: dataset.family_label_vector(f, version)
            for f in dev_families
            if any(f.labels.values())
        }
        if not train_labels or not dev_labels:
            raise ModelError("训练/开发段缺少人工标注家族")
        nodes: dict[str, NodeState] = {}
        for node_id in sorted(catalog.nodes):
            state = _train_node(
                node_id=node_id,
                catalog=catalog,
                config=config,
                train_families=train_families,
                train_vectors=train_vectors,
                train_labels=train_labels,
                dev_families=dev_families,
                dev_vectors=dev_vectors,
                dev_labels=dev_labels,
                dataset=dataset,
                version=version,
                feature_kinds=vectorizer.feature_kinds,
            )
            if state is not None:
                nodes[node_id] = state
        # 被至少一个节点逻辑回归正向承接的概念特征。
        # 用模型权重而非质心：质心会混入负文档的偶然共现（如生物技术词与通用词同句）。
        # 一个概念只有在某节点取得正权重时才是该节点的判别证据；若它在所有节点
        # 权重都 <= 0，说明目录里没有任何类能解释它，即“孤儿概念”（新技术信号）。
        supported: set[int] = set()
        for state in nodes.values():
            for idx, weight in state.model.weights.items():
                if (
                    vectorizer.feature_kinds[idx] in ("concept", "marker")
                    and weight > 0.0
                ):
                    supported.add(idx)
        return cls(
            config=config,
            history=dataset.history,
            version=version,
            vectorizer=vectorizer,
            nodes=nodes,
            lexicon=lexicon,
            train_family_ids=tuple(sorted(train_ids)),
            supported_concept_indices=frozenset(supported),
        )

    # ---- 推理 ----
    def score_family(self, family: Family) -> dict:
        vector = self.vectorizer.transform_one(family_text(family), family_language(family))
        node_scores: dict[str, dict] = {}
        for node_id, state in self.nodes.items():
            raw_score = state.model.score(vector)
            calibrated = state.calibrator.calibrate(raw_score)
            node_scores[node_id] = {
                "raw_score": raw_score,
                "probability": calibrated,
                "threshold": state.threshold,
                "is_rare": state.is_rare,
            }
        return {"vector": vector, "nodes": node_scores}

    def predict_family(self, family: Family) -> "Prediction":
        scored = self.score_family(family)
        vector: SparseVector = scored["vector"]
        language = family_language(family)
        abstain_reasons = self._global_abstention_reasons(family, vector, scored["nodes"])
        # 孤儿概念（词表认识但无目录节点承接）是硬拒识信号：
        # 不接受任何节点，直接作为目录无法容纳的新技术上交专家。
        hard_orphan = any(
            reason.get("orphan_concepts") for reason in abstain_reasons
        )
        accepted: set[str] = set()
        rejected_nodes: dict[str, str] = {}
        if not hard_orphan:
            for node_id, metrics in scored["nodes"].items():
                state = self.nodes[node_id]
                # 节点级 OOD：该文档相对此节点质心落在训练分布之外，则不接受该节点，
                # 但不阻止其他节点接受（跨主题文档对错误节点天然 OOD）。
                centroid, ood_floor, _ = state.ood_profile(language)
                similarity = dot(vector, centroid)
                node_floor = ood_floor * self.config.ood_floor_scale
                if similarity < node_floor:
                    rejected_nodes[node_id] = (
                        f"质心相似度 {similarity:.3f} < OOD 地板 {node_floor:.3f}"
                    )
                    continue
                if metrics["probability"] >= metrics["threshold"]:
                    accepted.add(node_id)
        if self.config.enforce_hierarchy:
            accepted = set(self.catalog.closure(accepted))
        # 阈值过滤后再做祖先闭包；闭包加入的祖先标记为 propagated
        direct = set(accepted)
        propagated = set()
        if self.config.enforce_hierarchy:
            for node_id in list(direct):
                propagated.update(self.catalog.ancestors(node_id))
            propagated -= direct
        if rejected_nodes and not accepted:
            abstain_reasons.append(
                {
                    "code": "uncovered_topic",
                    "detail": "全部候选节点均未通过节点级 OOD 检查",
                    "rejected_nodes": rejected_nodes,
                }
            )
        elif rejected_nodes:
            # 有接受也有节点级 OOD 拒绝：记录为软警告，不改变整体“已接受”状态
            abstain_reasons.append(
                {
                    "code": "node_ood_warning",
                    "detail": "部分候选节点未通过节点级 OOD，已单独排除",
                    "rejected_nodes": rejected_nodes,
                }
            )
        evidence = self._evidence(family, vector, accepted)
        return Prediction(
            family_id=family.id,
            version=self.version,
            accepted=frozenset(accepted),
            direct=frozenset(direct - propagated),
            propagated=frozenset(propagated),
            node_scores=scored["nodes"],
            abstain_reasons=tuple(abstain_reasons),
            evidence=evidence,
            vector=vector,
        )

    def _global_abstention_reasons(
        self, family: Family, vector: SparseVector, node_scores: dict
    ) -> list[dict]:
        """全局拒识：短摘要、全局 OOD、孤儿概念、整体证据不足。

        全局 OOD 用所有节点中“最宽松”的地板：文档只要贴近任一已学主题的训练分布
        就放行，具体归到哪个节点由节点级 OOD + 阈值决定。只有与全部已知主题都偏离
        的文档（真正的新技术）才在这一层被拦截。
        """
        reasons: list[dict] = []
        text_value = family_text(family)
        tokens = len(text_value.strip())
        if tokens < self.config.min_tokens:
            reasons.append(
                {
                    "code": "short_abstract",
                    "detail": f"摘要长度 {tokens} 字符 < 最小 {self.config.min_tokens}",
                }
            )
        language = family_language(family)
        global_floor = None
        global_nearest = None
        for node_id, state in self.nodes.items():
            centroid, ood_floor, _ = state.ood_profile(language)
            effective_floor = ood_floor * self.config.ood_floor_scale
            similarity = dot(vector, centroid)
            if global_floor is None or effective_floor < global_floor:
                global_floor = effective_floor
            if global_nearest is None or similarity > global_nearest[1]:
                global_nearest = (node_id, similarity)
        if global_floor is not None and global_nearest[1] < global_floor:
            reasons.append(
                {
                    "code": "uncovered_topic",
                    "detail": (
                        f"与最宽松已知主题地板的最大相似度 {global_nearest[1]:.3f} "
                        f"< 全局 OOD 地板 {global_floor:.3f}，不属于任何已训练主题"
                    ),
                    "nearest_node": global_nearest[0],
                }
            )
        # 已知概念但全部落在训练期无节点承接的概念上（如生物技术）：
        # 概念被冻结词表识别，却不对应任何已训练节点的正向特征分布。
        orphan = self._orphan_concepts(family, vector)
        if orphan:
            reasons.append(
                {
                    "code": "uncovered_topic",
                    "detail": (
                        f"命中概念 {sorted(orphan)} 在训练段没有任何目录节点承接，"
                        "属于目录无法容纳的主题"
                    ),
                    "orphan_concepts": sorted(orphan),
                }
            )
        # 证据不足：最高校准概率仍低于其阈值
        if node_scores:
            best = max(node_scores.items(), key=lambda item: item[1]["probability"])
            if best[1]["probability"] < best[1]["threshold"]:
                reasons.append(
                    {
                        "code": "insufficient_evidence",
                        "detail": (
                            f"最高候选 {best[0]} 校准概率 {best[1]['probability']:.3f} "
                            f"< 阈值 {best[1]['threshold']:.3f}"
                        ),
                    }
                )
        # 同类原因去重（保留信息最丰富的一条），避免重复计数
        deduped: dict[str, dict] = {}
        for reason in reasons:
            deduped.setdefault(reason["code"], reason)
        return list(deduped.values())

    def _lexicon_coverage(self, family: Family, vector: SparseVector) -> float:
        if not vector.indices:
            return 0.0
        concept_hits = 0
        for idx in vector.indices:
            if self.vectorizer.feature_kinds[idx] in ("concept", "marker"):
                concept_hits += 1
        return concept_hits / len(vector.indices)

    def _orphan_concepts(self, family: Family, vector: SparseVector) -> set[str]:
        """命中的概念特征里，没有任何训练节点正向承接的概念。

        典型场景：生物技术概念被冻结词表识别，但目录里没有对应节点，训练段从未
        把该概念作为任何节点的正向证据——它的出现本身就是“目录无法容纳”的信号。
        """
        if self.lexicon is None:
            return set()
        text_value = family_text(family)
        language = family_language(family)
        hits = self.lexicon.match(text_value, language)
        orphans: set[str] = set()
        for concept_id in hits:
            key = f"k:{concept_id}"
            idx = self.vectorizer.vocabulary.get(key)
            if idx is None:
                continue
            if idx not in self.supported_concept_indices:
                orphans.add(concept_id)
        return orphans

    def _evidence(
        self, family: Family, vector: SparseVector, accepted: set[str]
    ) -> dict[str, list[dict]]:
        output: dict[str, list[dict]] = {}
        for node_id in sorted(accepted):
            state = self.nodes.get(node_id)
            if state is None:
                output[node_id] = []
                continue
            output[node_id] = evidence_support(
                family_text(family),
                family_language(family),
                self.vectorizer,
                state.model.weights,
            )
        return output

    # ---- 序列化 ----
    def to_bundle(self) -> dict:
        return {
            "schema": 1,
            "config": {
                "vectorizer": self.vectorizer.to_dict()["config"],
                "epochs": self.config.epochs,
                "min_train_pos": self.config.min_train_pos,
                "rare_train_pos": self.config.rare_train_pos,
                "precision_floor": self.config.precision_floor,
                "min_tokens": self.config.min_tokens,
                "ood_quantile": self.config.ood_quantile,
                "lex_coverage_quantile": self.config.lex_coverage_quantile,
                "ood_floor_scale": self.config.ood_floor_scale,
                "enforce_hierarchy": self.config.enforce_hierarchy,
                "calibration_pool": self.config.calibration_pool,
            },
            "version": self.version,
            "train_family_ids": list(self.train_family_ids),
            "supported_concept_indices": sorted(self.supported_concept_indices),
            "vectorizer": self.vectorizer.to_dict(),
            "nodes": {
                node_id: {
                    "model": state.model.to_dict(),
                    "calibrator": state.calibrator.to_dict(),
                    "threshold": state.threshold,
                    "train_pos_families": state.train_pos_families,
                    "pooled": state.pooled,
                    "centroid": {
                        "indices": list(state.centroid.indices),
                        "values": list(state.centroid.values),
                    },
                    "ood_quantile": state.ood_quantile,
                    "lex_coverage_quantile": state.lex_coverage_quantile,
                    "is_rare": state.is_rare,
                    "centroids_by_language": {
                        language: {
                            "indices": list(centroid.indices),
                            "values": list(centroid.values),
                        }
                        for language, centroid in sorted(state.centroids_by_language.items())
                    },
                    "ood_quantile_by_language": dict(sorted(state.ood_quantile_by_language.items())),
                    "lex_quantile_by_language": dict(sorted(state.lex_quantile_by_language.items())),
                }
                for node_id, state in sorted(self.nodes.items())
            },
        }

    @classmethod
    def from_bundle(
        cls, payload: dict, history: CatalogHistory, lexicon: ConceptLexicon | None
    ) -> "HierarchicalModel":
        vectorizer_payload = payload["vectorizer"]
        vectorizer = TfidfVectorizer.from_dict(vectorizer_payload, lexicon)
        config = ModelConfig(
            vectorizer=VectorizerConfig(**vectorizer_payload["config"]),
            epochs=payload["config"]["epochs"],
            min_train_pos=payload["config"]["min_train_pos"],
            rare_train_pos=payload["config"]["rare_train_pos"],
            precision_floor=payload["config"]["precision_floor"],
            min_tokens=payload["config"]["min_tokens"],
            ood_quantile=payload["config"]["ood_quantile"],
            lex_coverage_quantile=payload["config"]["lex_coverage_quantile"],
            ood_floor_scale=payload["config"].get("ood_floor_scale", 0.8),
            enforce_hierarchy=payload["config"]["enforce_hierarchy"],
            calibration_pool=payload["config"]["calibration_pool"],
        )
        nodes: dict[str, NodeState] = {}
        for node_id, entry in payload["nodes"].items():
            centroid = SparseVector(
                indices=tuple(entry["centroid"]["indices"]),
                values=tuple(entry["centroid"]["values"]),
            )
            centroids_by_language = {
                language: SparseVector(
                    indices=tuple(part["indices"]),
                    values=tuple(part["values"]),
                )
                for language, part in entry.get("centroids_by_language", {}).items()
            }
            nodes[node_id] = NodeState(
                node_id=node_id,
                model=LogisticModel.from_dict(entry["model"]),
                calibrator=PlattCalibrator.from_dict(entry["calibrator"]),
                threshold=float(entry["threshold"]),
                train_pos_families=int(entry["train_pos_families"]),
                pooled=bool(entry["pooled"]),
                centroid=centroid,
                ood_quantile=float(entry["ood_quantile"]),
                lex_coverage_quantile=float(entry["lex_coverage_quantile"]),
                is_rare=bool(entry["is_rare"]),
                centroids_by_language=centroids_by_language,
                ood_quantile_by_language={
                    k: float(v) for k, v in entry.get("ood_quantile_by_language", {}).items()
                },
                lex_quantile_by_language={
                    k: float(v) for k, v in entry.get("lex_quantile_by_language", {}).items()
                },
            )
        return cls(
            config=config,
            history=history,
            version=int(payload["version"]),
            vectorizer=vectorizer,
            nodes=nodes,
            lexicon=lexicon,
            train_family_ids=tuple(payload["train_family_ids"]),
            supported_concept_indices=frozenset(
                int(idx) for idx in payload.get("supported_concept_indices", ())
            ),
        )


@dataclass
class Prediction:
    family_id: str
    version: int
    accepted: frozenset[str]
    direct: frozenset[str]
    propagated: frozenset[str]
    node_scores: dict
    abstain_reasons: tuple[dict, ...]
    evidence: dict[str, list[dict]]
    vector: SparseVector

    @property
    def is_abstained(self) -> bool:
        """硬拒识：没有任何接受节点。带 node_ood_warning 的接受仍是有效分类。"""
        if self.accepted:
            return False
        return bool(self.abstain_reasons)

    @property
    def hard_reasons(self) -> tuple[dict, ...]:
        """导致硬拒识的阻断性原因（不含软警告）。"""
        return tuple(r for r in self.abstain_reasons if r["code"] != "node_ood_warning")

    def to_record(self) -> dict:
        return {
            "family": self.family_id,
            "taxonomy_version": self.version,
            "accepted": sorted(self.accepted),
            "direct": sorted(self.direct),
            "propagated": sorted(self.propagated),
            "abstain": self.is_abstained,
            "abstain_reasons": list(self.abstain_reasons),
            "scores": {
                node: {
                    "probability": round(metrics["probability"], 6),
                    "threshold": round(metrics["threshold"], 6),
                    "is_rare": metrics["is_rare"],
                }
                for node, metrics in sorted(self.node_scores.items())
            },
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- #
# 节点级训练辅助
# --------------------------------------------------------------------------- #
def _node_available(family: Family, dataset: PatentDataset, node_id: str, version: int) -> bool:
    """节点在给定版本目录中是否可作为该家族的训练候选（正或负）。

    口径：节点属于该版本目录即对该版本的全部有标签家族可见——目录一旦生效，
    审查员就可以对任何存量家族判断“属于/不属于”，回填标签正是这种判断。
    PU 语义（未标注不是负类）由 _doc_records 只采样有标签家族独立保证，
    不在这里通过收窄可见性来实现，否则会让细类节点拿不到负样本。
    """
    return node_id in dataset.history.catalog(version).nodes



def _doc_records(
    families: list[Family],
    vectors: dict[tuple[str, str], SparseVector],
    labels: dict[str, frozenset[str]],
    dataset: PatentDataset,
    node_id: str,
    version: int,
) -> list[tuple[str, str, SparseVector, int]]:
    """产出 (family_id, doc_id, vector, label) 记录；未标注家族不进样本（PU 语义）。"""
    records: list[tuple[str, str, SparseVector, int]] = []
    for family in families:
        if family.id not in labels or not _node_available(family, dataset, node_id, version):
            continue
        label = 1 if node_id in labels[family.id] else 0
        for doc in family.docs:
            key = (family.id, doc.id)
            if key in vectors:
                records.append((family.id, doc.id, vectors[key], label))
    return records


def _train_node(
    node_id: str,
    catalog: Catalog,
    config: ModelConfig,
    train_families: list[Family],
    train_vectors: dict[tuple[str, str], SparseVector],
    train_labels: dict[str, frozenset[str]],
    dev_families: list[Family],
    dev_vectors: dict[tuple[str, str], SparseVector],
    dev_labels: dict[str, frozenset[str]],
    dataset: PatentDataset,
    version: int,
    feature_kinds: list[str],
) -> NodeState | None:
    records = _doc_records(
        train_families, train_vectors, train_labels, dataset, node_id, version
    )
    pos_family_ids = {fid for fid, _, _, label in records if label == 1}
    pos_family_count = len(pos_family_ids)
    # 正样本门槛按家族口径（翻译不算独立创新），防止同家族多语重复计数
    if pos_family_count < config.min_train_pos:
        return None
    rows = [(vector, label) for _, _, vector, label in records]
    if not any(label == 0 for _, label in rows):
        # 可用样本全部为正（极稀有叶子常见）：无法拟合二分类器，跳过
        return None
    model = LogisticModel().fit(
        rows, epochs=config.epochs, learning_rate=config.learning_rate
    )
    is_rare = pos_family_count < config.rare_train_pos
    # 校准：开发段文档级分数；稀有节点在子树内池化
    dev_scores = _dev_scores(
        node_id, catalog, model, dev_families, dev_vectors, dev_labels, dataset, version,
        pool=config.calibration_pool and is_rare,
    )
    calibrator = PlattCalibrator().fit(dev_scores)
    calibrator.pooled = config.calibration_pool and is_rare
    threshold = _choose_threshold(
        node_id, model, calibrator, dev_families, dev_vectors, dev_labels,
        dataset, version, precision_floor=config.precision_floor,
    )
    (
        centroid,
        ood_quantile,
        lex_quantile,
        centroids_by_language,
        ood_by_language,
        lex_by_language,
    ) = _ood_profile(
        node_id, train_families, train_vectors, train_labels, config, dataset, version,
        feature_kinds=feature_kinds,
    )
    return NodeState(
        node_id=node_id,
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        train_pos_families=pos_family_count,
        pooled=calibrator.pooled,
        centroid=centroid,
        ood_quantile=ood_quantile,
        lex_coverage_quantile=lex_quantile,
        is_rare=is_rare,
        centroids_by_language=centroids_by_language,
        ood_quantile_by_language=ood_by_language,
        lex_quantile_by_language=lex_by_language,
    )


def _dev_scores(
    node_id: str,
    catalog: Catalog,
    model: LogisticModel,
    dev_families: list[Family],
    dev_vectors: dict[tuple[str, str], SparseVector],
    dev_labels: dict[str, frozenset[str]],
    dataset: PatentDataset,
    version: int,
    pool: bool,
) -> list[tuple[float, int]]:
    records = _doc_records(
        dev_families, dev_vectors, dev_labels, dataset, node_id, version
    )
    scores = [
        (model.score(vector), label) for _, _, vector, label in records
    ]
    if not pool and len({label for _, label in scores}) > 1:
        return scores
    # 池化：子树相关节点的开发文档分数合并，缓解稀有节点正样本太少
    subtree = {node_id} | set(catalog.descendants(node_id)) | set(catalog.ancestors(node_id))
    pooled: list[tuple[float, int]] = []
    for family in dev_families:
        if family.id not in dev_labels:
            continue
        if not any(_node_available(family, dataset, nid, version) for nid in subtree):
            continue
        label = 1 if node_id in dev_labels[family.id] else 0
        for doc in family.docs:
            key = (family.id, doc.id)
            if key in dev_vectors:
                pooled.append((model.score(dev_vectors[key]), label))
    if len({label for _, label in pooled}) > 1:
        return pooled
    return scores


def _choose_threshold(
    node_id: str,
    model: LogisticModel,
    calibrator: PlattCalibrator,
    dev_families: list[Family],
    dev_vectors: dict[tuple[str, str], SparseVector],
    dev_labels: dict[str, frozenset[str]],
    dataset: PatentDataset,
    version: int,
    precision_floor: float,
) -> float:
    """在开发段选阈值：满足 precision_floor 的候选中取 F1 最高。

    指标在文档级分数上计算但按家族聚合判断，避免同家族翻译把同一次创新重复计数。
    没有任何阈值能满足精度底线时，阈值设为开发段最高概率之上——该节点在运行期
    一律以 insufficient_evidence 拒识。宁可节点拒识交给专家，也不制造低精度接受。
    """
    records = _doc_records(
        dev_families, dev_vectors, dev_labels, dataset, node_id, version
    )
    if not records:
        return 1.0
    candidates = [
        (calibrator.calibrate(model.score(vector)), label)
        for _, _, vector, label in records
    ]

    def stats(threshold: float) -> tuple[float, float, float, int]:
        tp = sum(1 for p, label in candidates if p >= threshold and label == 1)
        fp = sum(1 for p, label in candidates if p >= threshold and label == 0)
        fn = sum(1 for p, label in candidates if p < threshold and label == 1)
        if tp + fp == 0:
            return 0.0, 0.0, 0.0, tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1, tp

    grid = sorted({round(p, 4) for p, _ in candidates})
    best_threshold, best_f1 = None, -1.0
    for threshold in grid:
        precision, _, f1, tp = stats(threshold)
        if tp == 0:
            continue  # 全拒绝点不参与正常选择
        if precision < precision_floor:
            continue
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    if best_threshold is not None:
        return best_threshold
    # 精度底线无法满足：节点级拒识阈值（高于开发段任何观测概率）
    return min(1.0, max(p for p, _ in candidates) + 1e-4)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[position]


def _ood_profile(
    node_id: str,
    train_families: list[Family],
    train_vectors: dict[tuple[str, str], SparseVector],
    train_labels: dict[str, frozenset[str]],
    config: ModelConfig,
    dataset: PatentDataset,
    version: int,
    feature_kinds: list[str],
) -> tuple[SparseVector, float, float, dict, dict, dict]:
    """返回全局剖面与分语种剖面（均基于正类训练文档）。

    分语种质心避免高资源语种（中文 CJK n-gram 密集）主导余弦距离，使小语种
    文档不会因特征空间稀疏被误判为 OOD；未见语种或单样本语种回退全局剖面。
    """
    positive_docs: list[tuple[str, SparseVector]] = []
    for family in train_families:
        if family.id not in train_labels or node_id not in train_labels[family.id]:
            continue
        for doc in family.docs:
            key = (family.id, doc.id)
            if key in train_vectors:
                positive_docs.append((doc.language, train_vectors[key]))
    if not positive_docs:
        empty = SparseVector((), ())
        return empty, 0.0, 0.0, {}, {}, {}

    def profile(vectors: list[SparseVector]) -> tuple[SparseVector, float, float]:
        summed: dict[int, float] = {}
        for vector in vectors:
            for idx, value in zip(vector.indices, vector.values):
                summed[idx] = summed.get(idx, 0.0) + value
        norm = math.sqrt(sum(v * v for v in summed.values())) or 1.0
        centroid = SparseVector.from_counts(
            {idx: value / norm for idx, value in summed.items()}
        )
        similarities = sorted(dot(vector, centroid) for vector in vectors)
        coverages = sorted(
            sum(
                1
                for idx in vector.indices
                if feature_kinds[idx] in ("concept", "marker")
            )
            / max(1, len(vector.indices))
            for vector in vectors
        )
        return (
            centroid,
            _quantile(similarities, config.ood_quantile),
            _quantile(coverages, config.lex_coverage_quantile),
        )

    global_centroid, global_ood, global_lex = profile([v for _, v in positive_docs])
    by_language: dict[str, list[SparseVector]] = {}
    for language, vector in positive_docs:
        by_language.setdefault(language, []).append(vector)
    centroids_by_language: dict[str, SparseVector] = {}
    ood_by_language: dict[str, float] = {}
    lex_by_language: dict[str, float] = {}
    for language, vectors in sorted(by_language.items()):
        if len(vectors) < 2:
            continue  # 单样本语种不估分位数，运行期回退全局
        centroid, ood_floor, lex_floor = profile(vectors)
        centroids_by_language[language] = centroid
        ood_by_language[language] = ood_floor
        lex_by_language[language] = lex_floor
    return (
        global_centroid,
        global_ood,
        global_lex,
        centroids_by_language,
        ood_by_language,
        lex_by_language,
    )
