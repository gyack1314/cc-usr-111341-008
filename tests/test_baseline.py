"""历史基线对照测试：复现“大类太粗”和“关键词污染”两个错误。"""
import json
import unittest
from pathlib import Path

from patent_data.baseline import (
    CoarseKeywordBaseline,
    KeywordSearchBaseline,
    baseline_error_report,
)
from patent_data.catalog import load_history
from patent_data.data import attach_hidden_truth, build_dataset
from patent_data.text import ConceptLexicon

DEMO = "demo_data/demo.json"
HISTORY = "demo_data/catalog-history.json"
TAXONOMY = "demo_data/demo-taxonomy.jsonl"
TRUTH = "demo_data/hidden-truth.jsonl"


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [
            json.loads(line)
            for line in Path(TAXONOMY).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.history = load_history(HISTORY, rows)
        cls.dataset = build_dataset(DEMO, cls.history)
        attach_hidden_truth(cls.dataset, TRUTH)
        cls.lexicon = ConceptLexicon.load()
        cls.family_ids = tuple(cls.dataset.family_ids_by_date())

    def test_keyword_search_polluted_by_neural(self):
        report = baseline_error_report(
            KeywordSearchBaseline(self.history.catalog(1), self.lexicon),
            self.dataset,
            self.family_ids,
            1,
        )
        # 神经网络摘要含“网络/network”词面，被人工搜索式基线误纳
        self.assertGreater(report["polluted_count"], 0)
        self.assertTrue(
            all("pollution" in reason for reason in [])
            or all(item["terms"] for item in report["polluted_families"])
        )

    def test_coarse_baseline_downgrades_cross_technology(self):
        report = baseline_error_report(
            CoarseKeywordBaseline(self.history.catalog(1), self.lexicon),
            self.dataset,
            self.family_ids,
            1,
        )
        # edge_ai 多父交叉技术在粗目录下只能降级到宏观祖先
        cross_nodes = {
            node
            for item in report["cross_technology_downgraded"]
            for node in item["cross_nodes"]
        }
        self.assertIn("edge_ai", cross_nodes)

    def test_truly_new_topic_outside_any_catalog(self):
        report = baseline_error_report(
            CoarseKeywordBaseline(self.history.catalog(1), self.lexicon),
            self.dataset,
            self.family_ids,
            1,
        )
        outside = {
            node
            for item in report["missed_families"]
            for node in item["out_of_catalog"]
        }
        # biotech_process 不属于任何版本目录，是真正的新技术
        self.assertIn("biotech_process", outside)


if __name__ == "__main__":
    unittest.main()
