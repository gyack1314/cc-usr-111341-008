# 技术专利多语文本的层级分类与新主题拒识

面向创新监测的离线 Python 能力：把多语短摘要映射为符合目录版本祖先约束的层级多标签，
并对目录无法容纳的新技术、过短摘要与证据不足给出带原因的拒识，让专家把精力放在
真正的新主题与歧义上。纯标准库实现，无需在线接口或第三方依赖。

## 解决的两个历史错误
- **专利大类太粗、交叉技术被漏**：粗大类基线只能把多父交叉类（如 edge_ai）降级为
  宏观祖先；本系统用节点级层级分类 + 祖先闭包保留细类路径。
- **人工搜“网络”混入神经网络**：歧义词（网络/network）在冻结词表中标记为 marker，
  证据里显式标注 `ambiguous`，排序高分本身不作为归类证据。

## 核心纪律
- 按公开时间前推；词表、校准、阈值、拒识参数只在训练/开发段选择，验证段冻结后打开。
- family 整体参与切分，同族翻译不横跨训练与验证；指标按家族口径计数。
- 未标注不是负类（PU 语义）；“当时还没有这个类”不等于判为不相关。
- 标签迁移必须登记依据；输出符合所用版本的祖先约束。
- 拒识区分人工漏标、已知类别误判与目录外新技术；不追求把全部拒绝做成高精度。
- 模型包与资源冻结并做 SHA256 校验，旧版目录/模型可复算；拆分类不称为技术涌现。

## 目录结构
```
patent_data/
  catalog.py     多父 DAG、版本生效区间、祖先闭包、有依据的标签迁移
  data.py        文档/家族/标签载入，家族口径与标签投影
  text.py        多语分词、冻结概念词表、TF-IDF、证据片段与歧义标记
  model.py       纯 Python 逻辑回归、Platt 校准、层级局部分类、两级 OOD 拒识
  evaluation.py  宏观/稀有/语种分层指标、错误三分法、跨语一致性、约束护栏
  baseline.py    粗大类与关键词搜索两个历史基线对照
  versioning.py  模型包冻结哈希、目录升级差异、同冻结家族复算
  pipeline.py    批量训练/验证/评分的 Python 接口
  cli.py         命令行入口（check / validate / score / baseline）
  resources/     冻结多语概念词表
tools/
  generate_demo.py 确定性演示数据生成器
  run.py           隔离模式启动器
docs/
  data-contract.md  原始资料约定
  methodology.md    方法与取舍说明
  runbook.md        端到端操作手册
```

## 快速开始
```sh
python3 -I tools/generate_demo.py
python3 -I tools/run.py check demo_data/demo.json
python3 -I tools/run.py validate \
  --manifest demo_data/demo.json --history demo_data/catalog-history.json \
  --taxonomy demo_data/demo-taxonomy.jsonl --truth demo_data/hidden-truth.jsonl \
  --out reports/run
python3 -I tools/run.py baseline --manifest demo_data/demo.json \
  --history demo_data/catalog-history.json --taxonomy demo_data/demo-taxonomy.jsonl \
  --version 1 --truth demo_data/hidden-truth.jsonl
```
详见 [docs/runbook.md](docs/runbook.md) 与 [docs/methodology.md](docs/methodology.md)。

## 测试
```sh
python3 -I -c "import sys, unittest; sys.path.insert(0,'.'); \
  unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().discover('tests'))"
python3 -m compileall -q patent_data tools tests
```

## 资料说明
`data/` 保存原始业务资料（normal/boundary/broken-digest），其读取与身份检查逻辑保持不变；
`demo_data/` 由生成器产出，覆盖同族翻译、类别拆并、多父目录、漏标/噪声标签与真正新主题。
JSONL 文件按清单内 SHA256 与字节长度绑定，未知版本或内容变化需重新登记。
运行环境 Python 3.11+，仅使用标准库。
