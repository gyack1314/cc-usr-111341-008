"""按公开时间前推（rolling-origin）划分，family 整体移动，翻译文本不跨段。

纪律：
- 训练/开发/验证按家族最早公开日排序后切分，同 family 的全部语种文档同进同出。
- 词表（text.TfidfVectorizer.fit）只看训练段；阈值、校准、拒识参数只看开发段；
  验证段在所有选择冻结后才打开。
- 切分边界显式记录日期，报告中可核对；支持多折前推以便稳定性观察。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .catalog import _parse_day
from .data import Family, PatentDataset


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    train_end: date
    dev_end: date
    test_end: date

    def summary(self) -> dict:
        return {
            "name": self.name,
            "train_families": len(self.train),
            "dev_families": len(self.dev),
            "test_families": len(self.test),
            "train_end": self.train_end.isoformat(),
            "dev_end": self.dev_end.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def make_split(
    dataset: PatentDataset,
    train_end: str | date,
    dev_end: str | date,
    name: str = "split_1",
    test_end: str | date | None = None,
) -> TemporalSplit:
    train_end = _parse_day(train_end)
    dev_end = _parse_day(dev_end)
    if not train_end < dev_end:
        raise ValueError("开发段必须晚于训练段结束日")
    ordered = dataset.family_ids_by_date()
    train, dev, test = [], [], []
    latest = train_end
    for fid in ordered:
        published = dataset.families[fid].published
        if published <= train_end:
            train.append(fid)
        elif published <= dev_end:
            dev.append(fid)
        else:
            if test_end is not None and published > _parse_day(test_end):
                continue
            test.append(fid)
    if not train or not dev or not test:
        raise ValueError("前推切分后存在空段，请放宽边界日期")
    actual_test_end = max(dataset.families[fid].published for fid in test)
    return TemporalSplit(
        name=name,
        train=tuple(train),
        dev=tuple(dev),
        test=tuple(test),
        train_end=train_end,
        dev_end=dev_end,
        test_end=_parse_day(test_end) if test_end else actual_test_end,
    )


def rolling_splits(
    dataset: PatentDataset,
    boundaries: list[tuple[str, str]],
) -> list[TemporalSplit]:
    """多折前推：boundaries 为 [(train_end, dev_end), ...]，测试段自动取后续。"""
    return [
        make_split(dataset, train_end, dev_end, name=f"split_{idx + 1}")
        for idx, (train_end, dev_end) in enumerate(boundaries)
    ]


def families_of(dataset: PatentDataset, ids: tuple[str, ...]) -> list[Family]:
    return [dataset.families[fid] for fid in ids]


def assert_family_isolation(splits: list[TemporalSplit]) -> None:
    """测试护栏：任何 family 不得同时出现在训练与验证侧。"""
    for split in splits:
        train_set = set(split.train)
        overlap = train_set & set(split.dev)
        overlap |= train_set & set(split.test)
        if overlap:
            raise AssertionError(f"家族横跨训练与验证：{sorted(overlap)}")
