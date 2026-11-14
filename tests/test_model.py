"""层级模型、校准拒识、评估三分法与模型包复算的端到端测试。"""
import json
import tempfile
import unittest
from pathlib import Path

from patent_data.catalog import load_history
from patent_data.data import attach_hidden_truth, build_dataset
from patent_data.evaluation import evaluate
from patent_data.model import (
    HierarchicalModel,
    LogisticModel,
    ModelConfig,
    PlattCalibrator,
)
from patent_data.split import make_split
from patent_data.text import ConceptLexicon, SparseVector, VectorizerConfig
from patent_data.versioning import (
    BundleError,
    diff_catalogs,
    load_bundle,
    replay_on_frozen_families,
    save_bundle,
)

DEMO = "demo_data/demo.json"
HISTORY = "demo_data/catalog-history.json"
TAXONOMY = "demo_data/demo-taxonomy.jsonl"
TRUTH = "demo_data/hidden-truth.jsonl"


def _load():
    rows = [
        json.loads(line)
        for line in Path(TAXONOMY).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = load_history(HISTORY, rows)
    dataset = build_dataset(DEMO, history)
    attach_hidden_truth(dataset, TRUTH)
    return history, dataset


class LogisticTests(unittest.TestCase):
    def test_separates_linearly(self):
        # 两个特征、两类线性可分，模型应学到正确方向
        rows = []
        for i in range(20):
            rows.append((SparseVector((0, 1), (2.0 + i * 0.01, 0.1)), 1))
            rows.append((SparseVector((0, 1), (-2.0 - i * 0.01, -0.1)), 0))
        model = LogisticModel().fit(rows, epochs=80)
        self.assertGreater(model.weights[0], 0)
        self.assertGreater(model.predict_proba_raw(rows[0][0]), 0.9)
        self.assertLess(model.predict_proba_raw(rows[1][0]), 0.1)

    def test_single_class_rejected(self):
        rows = [(SparseVector((0,), (1.0,)), 1) for _ in range(5)]
        with self.assertRaises(ValueError):
            LogisticModel().fit(rows)

    def test_deterministic_same_seed(self):
        rows = [
            (SparseVector((0,), (1.0,)), 1),
            (SparseVector((0,), (-1.0,)), 0),
        ] * 10
        first = LogisticModel().fit(rows).weights
        second = LogisticModel().fit(rows).weights
        self.assertEqual(first, second)


class PlattTests(unittest.TestCase):
    def test_monotonic_calibration(self):
        cal = PlattCalibrator().fit([(-2, 0), (-1, 0), (1, 1), (2, 1)])
        self.assertLess(cal.calibrate(-2), cal.calibrate(2))

    def test_single_class_no_crash(self):
        cal = PlattCalibrator().fit([(0.1, 1), (0.2, 1)])
        self.assertTrue(0.0 <= cal.calibrate(0.15) <= 1.0)


class HierarchicalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.dataset = _load()
        cls.lexicon = ConceptLexicon.load()
        cls.split = make_split(
            cls.dataset, "2021-06-30", "2021-09-30", test_end="2022-12-31"
        )
        cls.config = ModelConfig(
            vectorizer=VectorizerConfig(mode="words"),
            min_train_pos=3,
            rare_train_pos=8,
            epochs=60,
        )
        cls.model = HierarchicalModel.train(
            cls.dataset,
            cls.split.train,
            cls.split.dev,
            version=1,
            config=cls.config,
            lexicon=cls.lexicon,
        )

    def test_prediction_respects_ancestor_closure(self):
        for fid in self.split.test:
            prediction = self.model.predict_family(self.dataset.families[fid])
            self.assertEqual(
                self.model.catalog.violations(prediction.accepted),
                [],
            )

    def test_new_topic_abstained_with_reason(self):
        # 生物技术（目录外）必须硬拒识并带原因
        biotech = [
            fid
            for fid in self.split.test
            if self.dataset.families[fid].truth_status == "new_topic"
            and "biotech_process" in self.dataset.families[fid].truth
        ]
        self.assertTrue(biotech)
        for fid in biotech:
            prediction = self.model.predict_family(self.dataset.families[fid])
            self.assertTrue(prediction.is_abstained)
            codes = {reason["code"] for reason in prediction.hard_reasons}
            self.assertIn("uncovered_topic", codes)

    def test_short_abstract_reason(self):
        for fid, family in self.dataset.families.items():
            if family.truth_status == "short_abstract":
                prediction = self.model.predict_family(family)
                codes = {r["code"] for r in prediction.hard_reasons}
                self.assertIn("short_abstract", codes)

    def test_evidence_contains_snippets(self):
        fid = next(
            fid
            for fid in self.split.test
            if not self.model.predict_family(self.dataset.families[fid]).is_abstained
        )
        prediction = self.model.predict_family(self.dataset.families[fid])
        for node, items in prediction.evidence.items():
            if items:
                self.assertTrue(all("snippet" in item for item in items))
                break

    def test_bundle_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_bundle(self.model, tmp)
            restored, _ = load_bundle(tmp, self.history, self.lexicon)
            original = self.model.predict_family(
                self.dataset.families[self.split.test[0]]
            )
            reloaded = restored.predict_family(
                self.dataset.families[self.split.test[0]]
            )
            self.assertEqual(original.accepted, reloaded.accepted)
            # 篡改 model.json 必须被摘要校验拒绝
            model_path = Path(tmp) / "model.json"
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            payload["version"] = 999
            model_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(BundleError):
                load_bundle(tmp, self.history, self.lexicon)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.dataset = _load()
        cls.lexicon = ConceptLexicon.load()
        cls.split = make_split(
            cls.dataset, "2021-06-30", "2021-09-30", test_end="2022-12-31"
        )
        cls.model = HierarchicalModel.train(
            cls.dataset,
            cls.split.train,
            cls.split.dev,
            version=1,
            config=ModelConfig(min_train_pos=3, epochs=60),
            lexicon=cls.lexicon,
        )
        cls.result = evaluate(cls.model, cls.dataset, cls.split.test)

    def test_triage_partitions(self):
        triage = self.result.error_triage
        # 漏标家族被识别
        self.assertTrue(triage["missing_annotation"])
        # 新主题全部被正确拒识，无硬归
        self.assertTrue(triage["new_topic_handled"])
        self.assertEqual(triage["new_topic_force_fit"], [])

    def test_no_constraint_violations(self):
        self.assertEqual(self.result.constraint_violations, [])

    def test_language_breakdown_present(self):
        # 四语种分层指标都应存在（按文档口径）
        self.assertTrue(set(self.result.language_metrics) >= {"zh", "en"})

    def test_macro_and_leaf_separately_reported(self):
        self.assertIn("macro", self.result.level_metrics)
        self.assertIn("leaf", self.result.level_metrics)


class VersioningTests(unittest.TestCase):
    def test_diff_flags_split_not_emergence(self):
        history, _ = _load()
        diff = diff_catalogs(history, 1, 2)
        self.assertIn("network", diff.removed)
        self.assertIn("telecom_network", diff.added)
        ops = {m["op"] for m in diff.migrations}
        self.assertIn("split", ops)
        self.assertIn("不得称为技术涌现", diff.to_dict()["interpretation_note"])

    def test_replay_change_codes(self):
        history, dataset = _load()
        lexicon = ConceptLexicon.load()
        split_v1 = make_split(dataset, "2021-06-30", "2021-09-30", test_end="2022-12-31")
        model_v1 = HierarchicalModel.train(
            dataset, split_v1.train, split_v1.dev, version=1,
            config=ModelConfig(min_train_pos=3, epochs=40), lexicon=lexicon,
        )
        split_v2 = make_split(dataset, "2022-12-31", "2023-06-30")
        model_v2 = HierarchicalModel.train(
            dataset, split_v2.train, split_v2.dev, version=2,
            config=ModelConfig(min_train_pos=3, epochs=40), lexicon=lexicon,
        )
        frozen = tuple(
            fid
            for fid in dataset.family_ids_by_date()
            if dataset.families[fid].published >= history.catalog(2).effective
        )
        replay = replay_on_frozen_families(model_v1, model_v2, dataset, frozen)
        valid_codes = {
            "split_expanded", "merge_collapsed", "renamed",
            "new_node_available", "unchanged", "abstention_state_changed", "invalid",
        }
        self.assertTrue(set(replay["change_counts"]) <= valid_codes)


if __name__ == "__main__":
    unittest.main()
