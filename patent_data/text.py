"""多语短摘要的文本表示（纯标准库、离线、可冻结）。

两种可比较的表示：
- words：语种感知词形 + 冻结概念特征，跨语一致性好但依赖词表覆盖；
- chars：字符 n-gram，细类区分强但跨语迁移弱。
训练前在开发段选择，运行期词表与 IDF 冻结并随模型包序列化。

关键纪律：is_marker 概念（如中文“网络”）只作为普通特征参与打分，证据模块会
显式标记其歧义，禁止把单个歧义词命中当作技术归类结论（见 evidence_support）。
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

_TOKEN_RE = re.compile(r"[0-9]+|[a-zäöüß]+(?:[-'][a-zäöüß]+)*", re.IGNORECASE)
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

# 轻量屈折后缀（德语/英语常见），仅用于词干化特征，不改动原文证据片段
_INFLECTION_SUFFIXES = (
    "ierung", "ierung", "schafter", "schaften",
    "ungen", "heit", "keit", "isch", "lich",
    "ierung", "ens", "ern", "est", "ing", "ung",
    "en", "es", "em", "er", "st",
)


def _stem(token: str) -> str:
    """非常轻量的后缀词干化：长度足够才剥离，避免过度合并。"""
    if len(token) < 7:
        return token
    for suffix in _INFLECTION_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 5:
            return token[: -len(suffix)]
    return token


def _is_latin_term(normalized_term: str) -> bool:
    return not any(_CJK_RE.match(char) for char in normalized_term)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.casefold()


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Unicode 感知分词：拉丁词按词边界，CJK 按单字切分并保留相邻 bigram。

    stem=True 时对拉丁词做轻量词干化（见 _stem），CJK 不做词干化。
    """
    normalized = normalize(text)
    tokens: list[str] = []
    # 拉丁/数字词
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        tokens.append(_stem(token) if stem else token)
    # CJK：单字 + bigram（让“神经网络”与“网络路由”在词面可区分）
    cjk_runs: list[str] = []
    buf: list[str] = []
    for char in normalized:
        if _CJK_RE.match(char):
            buf.append(char)
        else:
            if buf:
                cjk_runs.append("".join(buf))
                buf = []
    if buf:
        cjk_runs.append("".join(buf))
    for run in cjk_runs:
        tokens.extend(run)
        tokens.extend(f"{run[i]}{run[i + 1]}" for i in range(len(run) - 1))
    return tokens


def char_ngrams(text: str, n_values: tuple[int, ...] = (3, 4)) -> list[str]:
    normalized = normalize(re.sub(r"\s+", " ", text)).strip()
    out: list[str] = []
    for n in n_values:
        if len(normalized) >= n:
            out.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
    return out


@dataclass(frozen=True)
class ConceptLexicon:
    concepts: dict[str, dict]
    """concept_id -> {terms: {lang: [term...]}, is_marker: bool}"""
    marker_concepts: frozenset[str]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ConceptLexicon":
        if path is None:
            raw = resources.files("patent_data.resources").joinpath(
                "concept_lexicon.json"
            ).read_text(encoding="utf-8")
            spec = json.loads(raw)
        else:
            spec = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = {entry["id"]: entry for entry in spec["concepts"]}
        markers = frozenset(
            cid for cid, entry in concepts.items() if entry.get("is_marker")
        )
        return cls(concepts=concepts, marker_concepts=markers)

    def match(self, text: str, language: str) -> dict[str, list[str]]:
        """返回 concept_id -> 在原文中命中的表面词（保留原词用于证据片段）。

        匹配规则按书写系统区分，避免短词子串误命中：
        - 拉丁词形要求 token 边界（德语 Gen 不得命中 EnergiesparGen）；
        - CJK 词形允许子串（中文无空格分词）。
        """
        lowered = normalize(text)
        latin_tokens = set(_TOKEN_RE.findall(lowered))
        hits: dict[str, list[str]] = {}
        for cid, entry in self.concepts.items():
            terms = entry["terms"].get(language, [])
            matched = []
            for term in terms:
                term_norm = normalize(term)
                if not term_norm:
                    continue
                if _is_latin_term(term_norm):
                    # 多词术语做短语子串匹配，单词做 token 精确匹配
                    if " " in term_norm or "-" in term_norm:
                        if term_norm in lowered:
                            matched.append(term)
                    elif term_norm in latin_tokens:
                        matched.append(term)
                elif term_norm in lowered:
                    matched.append(term)
            if matched:
                hits[cid] = sorted(set(matched))
        return hits


@dataclass
class VectorizerConfig:
    mode: str = "words"  # words | chars | both
    min_df: int = 1
    max_features: int = 4000
    use_concepts: bool = True
    char_n: tuple[int, ...] = (3, 4)
    concept_min_df: int = 1
    """概念/标记特征的独立文档频率门槛：跨语对齐通道即使只在少数语种出现也保留。"""
    stem_latin: bool = False
    """拉丁词轻量词干化（应对德语复合/屈折）；默认关闭，实验对比时显式开启。"""


@dataclass
class SparseVector:
    """dict 形式稀疏向量；索引在 TfidfVectorizer 冻结后稳定。"""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def from_counts(cls, counts: dict[int, float]) -> "SparseVector":
        items = sorted(counts.items())
        return cls(
            indices=tuple(i for i, _ in items),
            values=tuple(v for _, v in items),
        )


@dataclass
class TfidfVectorizer:
    config: VectorizerConfig
    lexicon: ConceptLexicon | None
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: list[float] = field(default_factory=list)
    feature_kinds: list[str] = field(default_factory=list)
    """与 vocabulary 索引对齐：word/char/concept/marker。"""
    num_docs: int = 0

    # ---- 特征抽取 ----
    def _raw_features(self, text: str, language: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
        words: list[str] = []
        chars: list[str] = []
        if self.config.mode in ("words", "both"):
            words = tokenize(text, stem=self.config.stem_latin)
        if self.config.mode in ("chars", "both"):
            chars = char_ngrams(text, self.config.char_n)
        concept_hits: dict[str, list[str]] = {}
        if self.config.use_concepts and self.lexicon is not None:
            concept_hits = self.lexicon.match(text, language)
        return words, chars, concept_hits

    def fit(self, records: list[tuple[str, str]]) -> "TfidfVectorizer":
        """records: [(text, language), ...]，只能用训练段拟合。"""
        document_frequency: dict[str, int] = {}
        feature_kind: dict[str, str] = {}
        # 冻结词表中的全部概念键先登记，df=0 也保留：
        # 这是运行期识别“训练段从未出现的新概念”的前提，否则新技术词面无法进入向量。
        if self.config.use_concepts and self.lexicon is not None:
            for cid in self.lexicon.concepts:
                key = f"k:{cid}"
                feature_kind[key] = (
                    "marker" if cid in self.lexicon.marker_concepts else "concept"
                )
                document_frequency.setdefault(key, 0)
        for text, language in records:
            words, chars, concept_hits = self._raw_features(text, language)
            present = set()
            for token in words:
                present.add(f"w:{token}")
                feature_kind.setdefault(f"w:{token}", "word")
            for gram in chars:
                present.add(f"c:{gram}")
                feature_kind.setdefault(f"c:{gram}", "char")
            for cid, surfaces in concept_hits.items():
                key = f"k:{cid}"
                present.add(key)
                feature_kind[key] = (
                    "marker" if cid in self.lexicon.marker_concepts else "concept"
                )
            for key in present:
                document_frequency[key] = document_frequency.get(key, 0) + 1
        # 概念/标记特征：冻结词表概念无条件保留（含 df=0 的未见概念，用于新主题识别）；
        # 词与字符 n-gram 用主文档频率门槛。
        candidates = []
        for key, df in document_frequency.items():
            kind = feature_kind[key]
            if kind in ("concept", "marker"):
                floor = 0
            else:
                floor = self.config.min_df
            if df >= floor:
                candidates.append((key, df))
        # 确定性排序：先 df 降序再键名升序，截断到 max_features
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[: self.config.max_features]
        self.vocabulary = {key: idx for idx, (key, _) in enumerate(candidates)}
        self.num_docs = len(records)
        self.idf = [0.0] * len(self.vocabulary)
        self.feature_kinds = [""] * len(self.vocabulary)
        for key, idx in self.vocabulary.items():
            df = document_frequency[key]
            self.idf[idx] = math.log((1 + self.num_docs) / (1 + df)) + 1.0
            self.feature_kinds[idx] = feature_kind[key]
        return self

    def transform_one(self, text: str, language: str) -> SparseVector:
        words, chars, concept_hits = self._raw_features(text, language)
        counts: dict[int, float] = {}
        for token in words:
            idx = self.vocabulary.get(f"w:{token}")
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        for gram in chars:
            idx = self.vocabulary.get(f"c:{gram}")
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        for cid in concept_hits:
            idx = self.vocabulary.get(f"k:{cid}")
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        weighted = {idx: value * self.idf[idx] for idx, value in counts.items()}
        norm = math.sqrt(sum(v * v for v in weighted.values())) or 1.0
        return SparseVector.from_counts(
            {idx: value / norm for idx, value in weighted.items()}
        )

    def transform(self, records: list[tuple[str, str]]) -> list[SparseVector]:
        return [self.transform_one(text, lang) for text, lang in records]

    # ---- 序列化（词表冻结，随模型包发布）----
    def to_dict(self) -> dict:
        return {
            "config": {
                "mode": self.config.mode,
                "min_df": self.config.min_df,
                "max_features": self.config.max_features,
                "use_concepts": self.config.use_concepts,
                "char_n": list(self.config.char_n),
                "concept_min_df": self.config.concept_min_df,
                "stem_latin": self.config.stem_latin,
            },
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "feature_kinds": self.feature_kinds,
            "num_docs": self.num_docs,
        }

    @classmethod
    def from_dict(cls, payload: dict, lexicon: ConceptLexicon | None) -> "TfidfVectorizer":
        config = VectorizerConfig(
            mode=payload["config"]["mode"],
            min_df=payload["config"]["min_df"],
            max_features=payload["config"]["max_features"],
            use_concepts=payload["config"]["use_concepts"],
            char_n=tuple(payload["config"]["char_n"]),
            concept_min_df=payload["config"].get("concept_min_df", 1),
            stem_latin=payload["config"].get("stem_latin", False),
        )
        vectorizer = cls(config=config, lexicon=lexicon)
        vectorizer.vocabulary = {key: int(idx) for key, idx in payload["vocabulary"].items()}
        vectorizer.idf = [float(v) for v in payload["idf"]]
        vectorizer.feature_kinds = list(payload["feature_kinds"])
        vectorizer.num_docs = int(payload["num_docs"])
        return vectorizer


def dot(vector_a: SparseVector, vector_b: SparseVector) -> float:
    if len(vector_a.indices) > len(vector_b.indices):
        vector_a, vector_b = vector_b, vector_a
    lookup = dict(zip(vector_b.indices, vector_b.values))
    return sum(value * lookup.get(idx, 0.0) for idx, value in zip(vector_a.indices, vector_a.values))


def evidence_support(
    text: str,
    language: str,
    vectorizer: TfidfVectorizer,
    weights: dict[int, float],
    top_k: int = 6,
) -> list[dict]:
    """返回可核对的局部特征：命中词面 + 权重贡献 + 是否歧义标记。

    排序纪律：概念/标记特征优先（它们是跨语对齐的判别证据），词与字符 n-gram
    按权重绝对值补充。marker 特征显式标注 ambiguous=True——排序高分本身不是
    技术归类证据，歧义词必须与非 marker 证据共同出现才可采信。
    """
    words, chars, concept_hits = vectorizer._raw_features(text, language)
    contributions: dict[str, dict] = {}
    for token in words:
        idx = vectorizer.vocabulary.get(f"w:{token}")
        if idx is not None and idx in weights:
            contributions[f"w:{token}"] = {
                "feature": f"w:{token}",
                "kind": vectorizer.feature_kinds[idx],
                "surface": token,
                "weight": weights[idx],
            }
    for gram in chars:
        idx = vectorizer.vocabulary.get(f"c:{gram}")
        if idx is not None and idx in weights:
            contributions[f"c:{gram}"] = {
                "feature": f"c:{gram}",
                "kind": vectorizer.feature_kinds[idx],
                "surface": gram,
                "weight": weights[idx],
            }
    for cid, surfaces in concept_hits.items():
        idx = vectorizer.vocabulary.get(f"k:{cid}")
        if idx is not None and idx in weights:
            contributions[f"k:{cid}"] = {
                "feature": f"k:{cid}",
                "kind": vectorizer.feature_kinds[idx],
                "surface": " / ".join(surfaces),
                "weight": weights[idx],
                "ambiguous": cid in vectorizer.lexicon.marker_concepts,
            }
    conceptual = [
        item
        for item in contributions.values()
        if item["kind"] in ("concept", "marker")
    ]
    lexical = sorted(
        (item for item in contributions.values() if item["kind"] not in ("concept", "marker")),
        key=lambda item: (-abs(item["weight"]), item["feature"]),
    )
    conceptual.sort(key=lambda item: (-abs(item["weight"]), item["feature"]))
    ranked = (conceptual + lexical)[:top_k]
    for item in ranked:
        item.setdefault("ambiguous", False)
        item["snippet"] = _snippet(text, item["surface"])
    return ranked


def _snippet(text: str, surface: str, window: int = 12) -> str:
    if not surface:
        return text[:40]
    lowered = text.casefold()
    position = lowered.find(surface.casefold())
    if position < 0:
        return text[:40]
    start = max(0, position - window)
    end = min(len(text), position + len(surface) + window)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"
