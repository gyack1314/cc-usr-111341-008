# 端到端操作手册（从输入到专家复核）

所有命令离线运行，仅用 Python 标准库。`tools/run.py` 以隔离模式（`-I`）启动，
显式定位仓库根目录，避免从工作目录加载非受信模块。

## 0. 生成演示资料（确定性，可重复）

```sh
python3 -I tools/generate_demo.py
```

输出到 `demo_data/`：三表 JSONL + 清单（SHA256/字节数绑定）、两版目录历史规格、
隐藏真值 `hidden-truth.jsonl`（仅评估读取，训练管线不接触）。

资料覆盖：同族四语翻译、目录 split/merge/rename/add、多父节点 edge_ai、
人工漏标、噪声标签、目录外生物技术、量子密钥（v2 前无类）、过短摘要、
“网络/network”歧义词。

## 1. 资料完整性检查

```sh
python3 -I tools/run.py check demo_data/demo.json
```

校验表集合、引用完整性、清单摘要与行数；内容被改动即拒绝加载。

## 2. 前推训练与分层验证

```sh
python3 -I tools/run.py validate \
  --manifest demo_data/demo.json \
  --history demo_data/catalog-history.json \
  --taxonomy demo_data/demo-taxonomy.jsonl \
  --truth demo_data/hidden-truth.jsonl \
  --out reports/run
```

- 终端打印两折摘要；完整结果写入 `reports/run/validation-report.json`。
- 冻结模型包写入 `reports/run/bundle-v1/`、`reports/run/bundle-v2/`（含 manifest 摘要）。
- 可加 `--mode chars|both`、`--stem-latin`、`--precision-floor`、`--epochs` 做对照实验，
  选择只在开发段生效。

报告读法（专家视角）：
- `coverage`：接受率与各拒绝原因计数；拒绝不追求全对，关注原因是否合理。
- `level_metrics`：macro 与 leaf 分开；`language_metrics` 暴露跨语差异。
- `error_triage`：漏标捕获 / 噪声标签抵抗 / 已知误判 / 新主题拒识与硬归。
- `cross_lingual`：同族翻译预测 Jaccard 一致率。
- `replay.change_counts`：目录升级在冻结家族上的路径变化归因。

## 3. 历史错误基线对照

```sh
python3 -I tools/run.py baseline \
  --manifest demo_data/demo.json \
  --history demo_data/catalog-history.json \
  --taxonomy demo_data/demo-taxonomy.jsonl \
  --version 1 --truth demo_data/hidden-truth.jsonl
```

- `coarse.cross_technology_downgraded`：粗大类目录把 edge_ai 交叉技术降级为宏观祖先。
- `coarse.missed_families[*].out_of_catalog`：生物技术等目录外新技术。
- `keyword.polluted_count`：人工搜“网络/network”混入的神经网络专利数量与命中词。

## 4. 冻结模型批量评分

```sh
python3 -I tools/run.py score \
  --bundle reports/run/bundle-v1 \
  --manifest demo_data/demo.json \
  --history demo_data/catalog-history.json \
  --taxonomy demo_data/demo-taxonomy.jsonl \
  --out reports/predictions.json
```

每条预测含 `accepted/direct/propagated`、节点校准概率与阈值、`abstain_reasons`
（带原因细节）、`evidence`（概念优先的支持片段与歧义标记）。

## 5. 专家复核行为示例

### 5.1 同族翻译（不横跨训练/验证）
同 family 的 zh/en/de/ja 文档整体进入同一折；`cross_lingual` 报告它们预测的
Jaccard 一致率，不一致的家族优先复核。

### 5.2 类别拆并（split 不是涌现）
v1 的 `network` 在 v2 拆为 `telecom_network`/`iot_network`。`replay` 对同一冻结家族
给出 `split_expanded` 与迁移留痕，数量变化归因于目录修订第 3 条，而非新技术。

### 5.3 多父目录
`edge_ai` 经两条路径（digital→iot_network→edge_ai 与 digital→ai→edge_ai）到达祖先；
预测闭包同时满足两条路径，`constraint_violations` 为 0。

### 5.4 真正新主题
生物技术家族命中 `biotech` 概念但无节点承接，触发孤儿概念硬拒识；
量子家族在 v1 时代因 `quantum_key` 尚未生效而拒识，v2 有类后正常归类。

### 5.5 摘要太短 / 证据不足
短摘要返回 `short_abstract`；概率低于开发段阈值返回 `insufficient_evidence`，
均不强行归类。

## 6. Python 接口

```python
from patent_data.pipeline import load_inputs, run_fold, FoldSpec
from patent_data.model import ModelConfig
from patent_data.text import VectorizerConfig

dataset, history, lexicon = load_inputs(
    "demo_data/demo.json", "demo_data/catalog-history.json",
    "demo_data/demo-taxonomy.jsonl", "demo_data/hidden-truth.jsonl",
)
model, split, result = run_fold(
    dataset, history, lexicon,
    FoldSpec("my_fold", 1, "2021-06-30", "2021-09-30", "2022-12-31"),
    ModelConfig(vectorizer=VectorizerConfig(mode="words"), min_train_pos=3),
)
print(result.to_dict()["level_metrics"])
```

## 7. 测试

```sh
python3 -I -c "import sys, unittest; sys.path.insert(0,'.'); \
  unittest.TextTestRunner().run(unittest.TestLoader().discover('tests'))"
```
