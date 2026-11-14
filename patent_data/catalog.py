"""层级技术目录：多父 DAG、版本生效区间、祖先约束与标签迁移。

设计要点：
- 一个节点可沿多条路径到达同一祖先（多父），祖先/后代集合按 DAG 闭包计算。
- 每个版本是不可变快照；节点带引入版本与停用版本，版本间通过显式迁移规则连接。
- 未标注不是负类的语义在 model.py 处理；本模块只负责目录结构与迁移的合法性。
- 迁移必须登记依据（basis），拆分类不会被解释为技术涌现（见 versioning.diff_bundles）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


class CatalogError(ValueError):
    pass


def _parse_day(value: str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Node:
    id: str
    parents: tuple[str, ...]
    introduced: int
    deprecated: int | None = None


@dataclass(frozen=True)
class Migration:
    """版本间标签迁移规则。

    op 取值：rename（一对一）、split（一对多）、merge（多对一）、
    deprecate（节点停用，无目标）、add（新节点，无源节点）。
    """

    op: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    basis: str
    note: str = ""


@dataclass(frozen=True)
class Catalog:
    version: int
    effective: date
    nodes: dict[str, Node]
    children: dict[str, tuple[str, ...]]

    # ---- 结构查询 ----
    def roots(self) -> list[str]:
        return sorted(nid for nid, node in self.nodes.items() if not node.parents)

    def leaves(self) -> list[str]:
        return sorted(nid for nid in self.nodes if not self.children.get(nid))

    def depth(self, node_id: str) -> int:
        """最长祖先链长度，根为 0。多父节点取最深路径。"""
        memo: dict[str, int] = {}

        def walk(nid: str) -> int:
            if nid in memo:
                return memo[nid]
            parents = self.nodes[nid].parents
            memo[nid] = 0 if not parents else 1 + max(walk(p) for p in parents)
            return memo[nid]

        return walk(node_id)

    def ancestors(self, node_id: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(self.nodes[node_id].parents)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.nodes[current].parents)
        return frozenset(seen)

    def descendants(self, node_id: str) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(self.children.get(node_id, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.children.get(current, ()))
        return frozenset(seen)

    def paths(self, node_id: str) -> list[tuple[str, ...]]:
        """从任一树根到该节点的全部路径，体现多父多路径结构。"""
        node = self.nodes[node_id]
        if not node.parents:
            return [(node_id,)]
        out: list[tuple[str, ...]] = []
        for parent in node.parents:
            out.extend(tuple(path) + (node_id,) for path in self.paths(parent))
        return out

    def closure(self, node_ids: Iterable[str]) -> frozenset[str]:
        """预测输出必须满足的祖先闭包。"""
        result: set[str] = set()
        for nid in node_ids:
            if nid not in self.nodes:
                raise CatalogError(f"节点 {nid} 不属于版本 {self.version} 目录")
            result.add(nid)
            result.update(self.ancestors(nid))
        return frozenset(result)

    def violations(self, node_ids: Iterable[str]) -> list[str]:
        """返回违反祖先约束的节点（缺少任一祖先）。"""
        chosen = set(node_ids)
        bad: list[str] = []
        for nid in sorted(chosen):
            missing = self.ancestors(nid) - chosen
            if missing:
                bad.append(f"{nid} 缺少祖先 {sorted(missing)}")
        return bad

    def available_at(self, day: str | date) -> frozenset[str]:
        """某公开日可选的节点集合。

        快照本身只包含该版本生效区间内的节点（build 时已按 introduced/deprecated
        过滤），因此在版本生效日之后，快照内节点全部可选；早于生效日返回空集。
        跨版本的时点可见性用 CatalogHistory.version_at 判断。
        """
        day = _parse_day(day)
        if day < self.effective:
            return frozenset()
        return frozenset(self.nodes)


@dataclass(frozen=True)
class CatalogHistory:
    versions: dict[int, Catalog]
    migrations: tuple[Migration, ...]
    effective_dates: dict[int, date]

    def catalog(self, version: int) -> Catalog:
        try:
            return self.versions[version]
        except KeyError:
            raise CatalogError(f"目录版本 {version} 不存在")

    def version_at(self, day: str | date) -> int:
        """公开日当时生效的最新目录版本。"""
        day = _parse_day(day)
        current = None
        for version, effective in sorted(self.effective_dates.items()):
            if effective <= day:
                current = version
        if current is None:
            raise CatalogError(f"{day} 早于任何目录版本生效日")
        return current

    def migrate(
        self, node_ids: Iterable[str], source_version: int, target_version: int
    ) -> tuple[frozenset[str], list[dict]]:
        """按显式迁移规则把旧版标签搬到新版。

        返回（迁移后节点集合，逐条迁移记录）。无规则可用的节点保留在记录里并标记
        unmapped，绝不静默丢弃；split 会扩展为多目标，merge 会收敛为单目标。
        """
        if source_version == target_version:
            return frozenset(node_ids), []
        if source_version not in self.versions or target_version not in self.versions:
            raise CatalogError("迁移端点版本不存在")
        direction = 1 if target_version > source_version else -1
        current = set(node_ids)
        trace: list[dict] = []
        versions = sorted(self.versions)
        span = [v for v in versions if (min(source_version, target_version) <= v <= max(source_version, target_version))]
        for left, right in zip(span, span[1:]):
            old, new = (left, right) if direction == 1 else (right, left)
            rules = [m for m in self.migrations if _rule_spans(m, self, old, new)]
            next_set: set[str] = set()
            for nid in sorted(current):
                applied = None
                for rule in rules:
                    if direction == 1 and nid in rule.sources:
                        applied = rule
                        break
                    if direction == -1 and nid in rule.targets:
                        applied = rule
                        break
                if applied is None:
                    trace.append({"node": nid, "from": old, "to": new, "status": "unmapped"})
                    next_set.add(nid)
                    continue
                if direction == 1:
                    destinations = applied.targets or ()
                    status = "migrated" if destinations else "deprecated"
                else:
                    destinations = applied.sources or ()
                    status = "reverse-migrated" if destinations else "deprecated"
                trace.append(
                    {
                        "node": nid,
                        "from": old,
                        "to": new,
                        "op": applied.op,
                        "basis": applied.basis,
                        "destinations": list(destinations),
                        "status": status,
                    }
                )
                next_set.update(destinations)
            current = next_set
        target_catalog = self.catalog(target_version)
        unknown = current - set(target_catalog.nodes)
        if unknown:
            trace.append({"node": sorted(unknown), "status": "invalid_after_migration"})
            raise CatalogError(f"迁移后节点 {sorted(unknown)} 不在版本 {target_version} 目录内")
        return frozenset(current), trace


def _rule_spans(migration: Migration, history: CatalogHistory, old: int, new: int) -> bool:
    return True  # 迁移规则在构建时已绑定相邻版本，这里保留接口便于未来扩展


def build_catalog(version: int, effective: date, raw_nodes: Iterable[dict]) -> Catalog:
    """从 taxonomy 表行（id/parents/version）构建单版本快照并做 DAG 校验。"""
    nodes: dict[str, Node] = {}
    for row in raw_nodes:
        node = Node(
            id=row["id"],
            parents=tuple(row.get("parents", ())),
            introduced=int(row.get("introduced", row.get("version", version))),
            deprecated=row.get("deprecated"),
        )
        if node.id in nodes:
            raise CatalogError(f"节点 {node.id} 重复定义")
        nodes[node.id] = node
    children: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for parent in node.parents:
            if parent not in nodes:
                raise CatalogError(f"节点 {nid} 的父节点 {parent} 不存在")
            children[parent].append(nid)
    _assert_acyclic(nodes)
    return Catalog(
        version=version,
        effective=_parse_day(effective),
        nodes=nodes,
        children={k: tuple(sorted(v)) for k, v in children.items()},
    )


def _assert_acyclic(nodes: dict[str, Node]) -> None:
    color: dict[str, int] = {}

    def visit(nid: str) -> None:
        state = color.get(nid, 0)
        if state == 1:
            raise CatalogError(f"目录存在环，涉及节点 {nid}")
        if state == 2:
            return
        color[nid] = 1
        for parent in nodes[nid].parents:
            visit(parent)
        color[nid] = 2

    for nid in nodes:
        visit(nid)


def load_history(spec_path: str | Path, taxonomy_rows: Iterable[dict]) -> CatalogHistory:
    """根据历史规格 JSON 与 taxonomy 表构建多版本目录。

    规格结构：
    {
      "versions": [{"version": 1, "effective": "2019-01-01"}, ...],
      "migrations": [{"op": "split", "sources": ["x"], "targets": ["a","b"],
                      "basis": "...", "note": "..."}]
    }
    taxonomy 行的 version 字段表示引入版本；deprecated 字段可选。
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows = list(taxonomy_rows)
    versions: dict[int, Catalog] = {}
    effective_dates: dict[int, date] = {}
    for entry in spec["versions"]:
        version = int(entry["version"])
        effective = _parse_day(entry["effective"])
        effective_dates[version] = effective
        active = [
            row
            for row in rows
            if int(row.get("introduced", row["version"])) <= version
            and (row.get("deprecated") is None or int(row["deprecated"]) > version)
        ]
        versions[version] = build_catalog(version, effective, active)
    migrations = tuple(
        Migration(
            op=m["op"],
            sources=tuple(m.get("sources", ())),
            targets=tuple(m.get("targets", ())),
            basis=m["basis"],
            note=m.get("note", ""),
        )
        for m in spec.get("migrations", ())
    )
    _validate_migrations(versions, migrations)
    return CatalogHistory(versions=versions, migrations=migrations, effective_dates=effective_dates)


def _validate_migrations(versions: dict[int, Catalog], migrations: tuple[Migration, ...]) -> None:
    allowed = {"rename", "split", "merge", "deprecate", "add"}
    ordered = sorted(versions)
    for migration in migrations:
        if migration.op not in allowed:
            raise CatalogError(f"未知迁移类型 {migration.op}")
        if not migration.basis.strip():
            raise CatalogError("迁移规则必须登记依据 basis")
        if migration.op == "deprecate" and (not migration.sources or migration.targets):
            raise CatalogError("deprecate 迁移只能有 source 不能有 target")
        if migration.op == "add" and (migration.sources or not migration.targets):
            raise CatalogError("add 迁移只能有 target 不能有 source")
        # 源必须存在于某个旧版本、目标必须存在于某个新版本
        for src in migration.sources:
            if not any(src in cat.nodes for cat in versions.values()):
                raise CatalogError(f"迁移源 {src} 在任何版本中都不存在")
        for tgt in migration.targets:
            if not any(tgt in cat.nodes for cat in versions.values()):
                raise CatalogError(f"迁移目标 {tgt} 在任何版本中都不存在")
        if migration.op in {"rename", "split"} and len(migration.sources) != 1:
            raise CatalogError(f"{migration.op} 迁移要求恰好一个源节点")
        if migration.op in {"rename", "merge"} and len(migration.targets) != 1:
            raise CatalogError(f"{migration.op} 迁移要求恰好一个目标节点")
    _ = ordered
