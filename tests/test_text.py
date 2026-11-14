"""文本表示、概念词表与证据的单元测试。"""
import unittest

from patent_data.text import (
    ConceptLexicon,
    TfidfVectorizer,
    VectorizerConfig,
    char_ngrams,
    evidence_support,
    normalize,
    tokenize,
)


class TokenizeTests(unittest.TestCase):
    def test_latin_and_cjk_tokens(self):
        tokens = tokenize("Edge nodes 边缘节点")
        self.assertIn("edge", tokens)
        self.assertIn("nodes", tokens)
        # CJK 单字与 bigram 都保留
        self.assertIn("边", tokens)
        self.assertIn("边缘", tokens)

    def test_normalize_casefold(self):
        self.assertEqual(normalize("Network"), "network")
        self.assertEqual(normalize("ÜBER"), "über")

    def test_char_ngrams(self):
        grams = char_ngrams("abcdef", (3,))
        self.assertEqual(grams, ["abc", "bcd", "cde", "def"])

    def test_stemming_optional(self):
        raw = tokenize("Trainingsdaten", stem=False)
        stemmed = tokenize("Trainingsdaten", stem=True)
        self.assertIn("trainingsdaten", raw)
        self.assertTrue(any(len(t) < len("trainingsdaten") for t in stemmed))


class LexiconTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = ConceptLexicon.load()

    def test_multilingual_alignment(self):
        self.assertIn("net_transport", self.lexicon.concepts)
        self.assertTrue(
            {"zh", "en", "de", "ja"}
            <= set(self.lexicon.concepts["net_transport"]["terms"])
        )

    def test_latin_token_boundary_no_substring_false_hit(self):
        # 德语 Gen 不得命中 EnergiesparGen（历史误判回归）
        hits = self.lexicon.match(
            "Ein Energiespar-Aufweckmechanismus für Sensorknoten", "de"
        )
        self.assertNotIn("biotech", hits)

    def test_latin_single_word_exact_match(self):
        hits = self.lexicon.match("Gen und Protein Analyse", "de")
        self.assertIn("biotech", hits)

    def test_cjk_substring_allowed(self):
        hits = self.lexicon.match("一种神经网络的卷积训练方法", "zh")
        self.assertIn("neural", hits)

    def test_marker_flag(self):
        self.assertIn("net_transport", self.lexicon.marker_concepts)
        self.assertNotIn("neural", self.lexicon.marker_concepts)


class VectorizerTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = ConceptLexicon.load()

    def test_unseen_concept_kept_in_vocab(self):
        # 训练段没有生物技术文档，概念特征仍须保留（新主题识别前提）
        records = [
            ("一种通信网络路由方法", "zh"),
            ("A network routing protocol", "en"),
        ]
        vectorizer = TfidfVectorizer(
            config=VectorizerConfig(mode="words"), lexicon=self.lexicon
        ).fit(records)
        self.assertIn("k:biotech", vectorizer.vocabulary)

    def test_transform_normalized(self):
        records = [("网络路由协议", "zh"), ("network routing", "en")]
        vectorizer = TfidfVectorizer(
            config=VectorizerConfig(mode="words"), lexicon=self.lexicon
        ).fit(records)
        vector = vectorizer.transform_one("网络路由协议", "zh")
        # L2 归一化后模长为 1
        norm = sum(v * v for v in vector.values) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_evidence_marks_marker_ambiguous(self):
        records = [("神经网络与网络路由", "zh")]
        vectorizer = TfidfVectorizer(
            config=VectorizerConfig(mode="words"), lexicon=self.lexicon
        ).fit(records)
        idx = vectorizer.vocabulary["k:net_transport"]
        evidence = evidence_support(
            "神经网络与网络路由", "zh", vectorizer, {idx: 1.0}
        )
        marker = next(item for item in evidence if item["feature"] == "k:net_transport")
        self.assertTrue(marker["ambiguous"])
        self.assertTrue(marker["snippet"])

    def test_serialization_roundtrip(self):
        records = [("网络路由", "zh"), ("network routing", "en")]
        vectorizer = TfidfVectorizer(
            config=VectorizerConfig(mode="both", stem_latin=True), lexicon=self.lexicon
        ).fit(records)
        restored = TfidfVectorizer.from_dict(vectorizer.to_dict(), self.lexicon)
        self.assertEqual(restored.config.mode, "both")
        self.assertTrue(restored.config.stem_latin)
        self.assertEqual(restored.vocabulary, vectorizer.vocabulary)


if __name__ == "__main__":
    unittest.main()
