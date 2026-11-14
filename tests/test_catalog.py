"""目录 DAG、多父路径、版本迁移与前推切分的单元测试。"""
import unittest
from datetime import date

from patent_data.catalog import (
    CatalogError,
    build_catalog,
    load_history,
)
from patent_data.split import assert_family_isolation, make_split


def _history_spec():
    return {
        "versions": [
            {"version": 1, "effective": "2019-01-01"},
            {"version": 2, "effective": "2023-01-01"},
        ],
        "migrations": [
            {
                "op": "split",
                "sources": ["network"],
                "targets": ["telecom_network", "iot_network"],
                "basis": "修订第3条",
            },
            {
                "op": "merge",
                "sources": ["wireless", "radio_link"],
                "targets": ["telecom_network"],
                "basis": "修订第5条",
            },
            {"op": "rename", "sources": ["security"], "targets": ["cyber_security"], "basis": "修订第1条"},
            {"op": "add", "sources": [], "targets": ["edge_ai"], "basis": "新增交叉类"},
        ],
    }


def _taxonomy_rows():
    return [
        {"id": "digital", "parents": [], "version": 1, "introduced": 1},
        {"id": "network", "parents": ["digital"], "version": 1, "introduced": 1, "deprecated": 2},
        {"id": "wireless", "parents": ["network"], "version": 1, "introduced": 1, "deprecated": 2},
        {"id": "radio_link", "parents": ["network"], "version": 1, "introduced": 1, "deprecated": 2},
        {"id": "security", "parents": ["digital"], "version": 1, "introduced": 1, "deprecated": 2},
        {"id": "telecom_network", "parents": ["digital"], "version": 2, "introduced": 2},
        {"id": "iot_network", "parents": ["digital"], "version": 2, "introduced": 2},
        {"id": "cyber_security", "parents": ["digital"], "version": 2, "introduced": 2},
        {"id": "ai", "parents": ["digital"], "version": 1, "introduced": 1},
        {"id": "edge_ai", "parents": ["iot_network", "ai"], "version": 2, "introduced": 2},
    ]


class CatalogStructureTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        spec_path = Path(self.tmp.name) / "history.json"
        spec_path.write_text(json.dumps(_history_spec()), encoding="utf-8")
        self.history = load_history(spec_path, _taxonomy_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def test_multi_parent_paths(self):
        catalog = self.history.catalog(2)
        paths = catalog.paths("edge_ai")
        self.assertIn(("digital", "iot_network", "edge_ai"), paths)
        self.assertIn(("digital", "ai", "edge_ai"), paths)

    def test_ancestor_closure_multi_path(self):
        catalog = self.history.catalog(2)
        closure = catalog.closure(["edge_ai"])
        self.assertEqual(closure, frozenset({"edge_ai", "iot_network", "ai", "digital"}))

    def test_violation_detects_missing_ancestor(self):
        catalog = self.history.catalog(2)
        violations = catalog.violations(["edge_ai"])  # 缺 iot_network/ai/digital
        self.assertTrue(violations)

    def test_cycle_rejected(self):
        rows = [
            {"id": "a", "parents": ["b"], "version": 1, "introduced": 1},
            {"id": "b", "parents": ["a"], "version": 1, "introduced": 1},
        ]
        with self.assertRaises(CatalogError):
            build_catalog(1, date(2019, 1, 1), rows)

    def test_unknown_parent_rejected(self):
        rows = [{"id": "a", "parents": ["ghost"], "version": 1, "introduced": 1}]
        with self.assertRaises(CatalogError):
            build_catalog(1, date(2019, 1, 1), rows)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        spec_path = Path(self.tmp.name) / "history.json"
        spec_path.write_text(json.dumps(_history_spec()), encoding="utf-8")
        self.history = load_history(spec_path, _taxonomy_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_expands_labels(self):
        moved, trace = self.history.migrate({"network"}, 1, 2)
        self.assertEqual(moved, frozenset({"telecom_network", "iot_network"}))
        self.assertTrue(all(t.get("basis") for t in trace if "op" in t))

    def test_merge_collapses_labels(self):
        moved, _ = self.history.migrate({"wireless", "radio_link"}, 1, 2)
        self.assertEqual(moved, frozenset({"telecom_network"}))

    def test_rename_preserves_identity(self):
        moved, _ = self.history.migrate({"security"}, 1, 2)
        self.assertEqual(moved, frozenset({"cyber_security"}))

    def test_reverse_migration_roundtrip(self):
        forward, _ = self.history.migrate({"network"}, 1, 2)
        backward, _ = self.history.migrate(forward, 2, 1)
        self.assertEqual(backward, frozenset({"network"}))

    def test_migration_requires_basis(self):
        import json
        import tempfile
        from pathlib import Path

        bad = _history_spec()
        bad["migrations"][0]["basis"] = "  "
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "h.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(CatalogError):
            load_history(path, _taxonomy_rows())
        tmp.cleanup()

    def test_version_at_published_date(self):
        self.assertEqual(self.history.version_at("2020-06-01"), 1)
        self.assertEqual(self.history.version_at("2023-06-01"), 2)


class SplitTests(unittest.TestCase):
    def _dataset(self):
        from patent_data.data import build_dataset
        import json
        import tempfile
        from pathlib import Path

        tmp = tempfile.TemporaryDirectory()
        spec = Path(tmp.name) / "h.json"
        spec.write_text(json.dumps(_history_spec()), encoding="utf-8")
        # 用演示数据的真实目录行（含 neural_net/edge_ai 等全部被标签引用的节点），
        # 但历史规格仍用本测试夹具，保证迁移断言独立可控。
        full_rows = [
            json.loads(line)
            for line in Path("demo_data/demo-taxonomy.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        full_spec = json.loads(Path("demo_data/catalog-history.json").read_text(encoding="utf-8"))
        full_spec_path = Path(tmp.name) / "full-history.json"
        full_spec_path.write_text(json.dumps(full_spec), encoding="utf-8")
        history = load_history(full_spec_path, full_rows)
        return build_dataset("demo_data/demo.json", history), tmp

    def test_family_isolation(self):
        dataset, tmp = self._dataset()
        try:
            split = make_split(dataset, "2021-06-30", "2021-09-30", test_end="2022-12-31")
            assert_family_isolation([split])
            # 同一家族不会横跨训练与验证
            self.assertFalse(set(split.train) & set(split.test))
            self.assertFalse(set(split.train) & set(split.dev))
        finally:
            tmp.cleanup()

    def test_empty_segment_rejected(self):
        dataset, tmp = self._dataset()
        try:
            with self.assertRaises(ValueError):
                make_split(dataset, "2018-01-01", "2018-02-01")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
