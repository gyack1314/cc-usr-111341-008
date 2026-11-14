"""生成演示用专利资料包（确定性、离线）。

覆盖任务要求的全部边界情形：
- 同家族多语种翻译（zh/en/de/ja），family 整体参与切分；
- 目录 v1→v2：split / merge / rename / add / 多父节点（edge_ai）；
- 人工漏标（missing_annotation）、噪声标签（noisy_label）；
- 目录无法容纳的新技术（biotech 始终无类；quantum 在 v2 生效前无类）；
- 过短摘要（short_abstract 拒识）；
- “网络/network”歧义词：神经网络与通信网络词面同形。

输出到 demo_data/，并按 io.load 的清单约定写 SHA256 与字节数。
隐藏真值单独写 hidden-truth.jsonl，训练管线不读取。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "demo_data"


class Lcg:
    def __init__(self, seed: int = 20261114) -> None:
        self.state = seed & 0xFFFFFFFF

    def below(self, bound: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state % bound

    def pick(self, items):
        return items[self.below(len(items))]


# --------------------------------------------------------------------------- #
# 目录两版
# --------------------------------------------------------------------------- #
TAXONOMY = [
    # v1 only
    ("digital", [], 1, None),
    ("network", ["digital"], 1, 2),
    ("wireless", ["network"], 1, 2),
    ("radio_link", ["network"], 1, 2),
    ("ai", ["digital"], 1, None),
    ("security", ["digital"], 1, 2),
    # v2 only
    ("telecom_network", ["digital"], 2, None),
    ("iot_network", ["digital"], 2, None),
    ("neural_net", ["ai"], 2, None),
    ("cyber_security", ["digital"], 2, None),
    ("quantum_key", ["cyber_security"], 2, None),
    ("edge_ai", ["iot_network", "ai"], 2, None),  # 多父
]

HISTORY_SPEC = {
    "versions": [
        {"version": 1, "effective": "2019-01-01"},
        {"version": 2, "effective": "2023-01-01"},
    ],
    "migrations": [
        {
            "op": "split",
            "sources": ["network"],
            "targets": ["telecom_network", "iot_network"],
            "basis": "2023目录修订第3条：通信网络与物联网络口径分离",
            "note": "拆分不代表技术涌现",
        },
        {
            "op": "merge",
            "sources": ["wireless", "radio_link"],
            "targets": ["telecom_network"],
            "basis": "2023目录修订第5条：无线链路类并入电信网络",
        },
        {
            "op": "rename",
            "sources": ["security"],
            "targets": ["cyber_security"],
            "basis": "2023目录修订第1条：安全类更名以涵盖量子密钥子类",
        },
        {"op": "add", "sources": [], "targets": ["edge_ai"], "basis": "2023新增交叉技术类"},
        {"op": "add", "sources": [], "targets": ["neural_net"], "basis": "2023新增神经网络细类"},
        {"op": "add", "sources": [], "targets": ["quantum_key"], "basis": "2023新增量子密钥细类"},
    ],
}

# --------------------------------------------------------------------------- #
# 多语摘要模板：每个主题 4 个句式，共享冻结词表中的核心术语
# --------------------------------------------------------------------------- #
TEMPLATES = {
    "telecom": {
        "zh": [
            "一种通信网络中的分组路由方法，通过协议优化报文传输与链路调度。",
            "网络路由设备根据协议转发报文，降低传输延迟并提升链路利用率。",
            "面向分组交换网络的路由协议，改善数据报传输与通信链路稳定性。",
            "通信网络的链路层传输机制，利用路由表加速报文分组转发。",
        ],
        "en": [
            "A packet routing method in a communication network using a protocol to optimize datagram transmission.",
            "A network routing device forwards packets by protocol, reducing transmission delay over the link.",
            "A routing protocol for packet-switched networks improves datagram communication and link stability.",
            "A link-layer transmission mechanism for communication networks accelerates packet forwarding via routing tables.",
        ],
        "de": [
            "Ein Paket-Routing-Verfahren in einem Kommunikationsnetzwerk mit Protokoll zur Optimierung der Datenpaket-Übertragung.",
            "Ein Netzwerk-Routing-Gerät leitet Pakete per Protokoll weiter und senkt die Übertragungsverzögerung auf der Verbindung.",
            "Ein Routing-Protokoll für paketvermittelte Netzwerke verbessert die Kommunikation und Leitungsstabilität.",
            "Ein Sicherungsverfahren auf der Sicherungsschicht von Netzwerken beschleunigt die Paketweiterleitung über Routing-Tabellen.",
        ],
        "ja": [
            "通信ネットワークにおけるパケットルーティング方法であり、プロトコルにより伝送を最適化する。",
            "ネットワークのルーティング装置がプロトコルでパケットを転送し、リンク上の伝送遅延を低減する。",
            "パケット交換ネットワーク向けルーティングプロトコルが通信とリンクの安定性を改善する。",
            "通信ネットワークのリンク層伝送機構がルーティングテーブルでパケット転送を高速化する。",
        ],
    },
    "iot": {
        "zh": [
            "低功耗物联网节点的传感器采集与休眠唤醒方法，边缘节点压缩传感数据。",
            "一种物联网边缘节点的低功耗传感器采集装置，支持休眠唤醒与电池供电。",
            "面向传感节点的低功耗唤醒机制，在边缘侧采集传感器信号并节能运行。",
            "嵌入式物联网节点通过低功耗休眠与唤醒策略延长电池寿命并采集传感信息。",
        ],
        "en": [
            "A low-power IoT node method for sensor acquisition with sleep and wake-up, where edge nodes compress sensor data.",
            "A low-power sensor acquisition device for IoT edge nodes supporting sleep, wake-up and battery supply.",
            "A low-power wake-up mechanism for sensing nodes acquires sensor signals at the edge with energy saving.",
            "An embedded IoT node extends battery life through low-power sleep and wake-up scheduling while collecting sensing data.",
        ],
        "de": [
            "Ein energieeffizientes IoT-Knoten-Verfahren zur Sensor-Erfassung mit Schlafmodus und Aufwecken, Edge-Knoten komprimieren Sensordaten.",
            "Eine energieeffiziente Sensor-Erfassungsvorrichtung für IoT-Edge-Knoten mit Schlafmodus, Aufwecken und Batteriebetrieb.",
            "Ein Energiespar-Aufweckmechanismus für Sensorknoten erfasst Signale am Edge und arbeitet energieeffizient.",
            "Ein eingebetteter IoT-Knoten verlängert die Batterielebensdauer durch Energieeffizient-Schlaf- und Aufweckpläne bei der Erfassung.",
        ],
        "ja": [
            "低消費電力IoTノードのセンサー取得とスリープ・起床方法であり、エッジノードがセンサーデータを圧縮する。",
            "IoTエッジノード向け低消費電力センサー取得装置で、スリープと起床、バッテリー給電を支援する。",
            "センサーノード向け省エネ起床機構がエッジ側でセンサー信号を取得し省エネ動作する。",
            "組み込みIoTノードが低消費電力のスリープと起床策略でバッテリー寿命を延ばしセンサー情報を取得する。",
        ],
    },
    "wireless": {
        "zh": [
            "一种无线射频天线的频谱调制方法，降低无线链路干扰并稳定射频传输。",
            "面向无线通信的射频天线调制装置，通过频谱分配优化无线信号质量。",
            "无线射频链路的天线调制与频谱感知机制，提升无线传输的抗干扰能力。",
            "利用射频天线与频谱调制实现的低干扰无线通信传输系统。",
        ],
        "en": [
            "A spectrum modulation method for wireless radio antennas that reduces wireless link interference and stabilizes radio transmission.",
            "A radio antenna modulation device for wireless communication optimizes signal quality through spectrum allocation.",
            "An antenna modulation and spectrum sensing mechanism for wireless radio links improves interference resistance.",
            "A low-interference wireless communication transmission system using radio antennas and spectrum modulation.",
        ],
        "de": [
            "Ein Spektrum-Modulationsverfahren für drahtlose Funk-Antennen reduziert Störungen auf drahtlosen Funkverbindungen.",
            "Eine Funkantennen-Modulationsvorrichtung für drahtlose Kommunikation optimiert die Signalqualität über Spektrumszuweisung.",
            "Ein Antennenmodulations- und Spektrumserkennungsmechanismus für drahtlose Funkverbindungen verbessert die Störfestigkeit.",
            "Ein störungsarmes drahtloses Kommunikationssystem mit Funk-Antennen und Spektrummodulation.",
        ],
        "ja": [
            "無線無線周波数アンテナのスペクトル変調方法で、無線リンクの干渉を低減し無線伝送を安定化する。",
            "無線通信向け無線周波数アンテナ変調装置がスペクトル割り当てで信号品質を最適化する。",
            "無線リンクのアンテナ変調とスペクトル検知機構が無線伝送の耐干渉性を高める。",
            "無線周波数アンテナとスペクトル変調を用いた低干渉の無線通信伝送システム。",
        ],
    },
    "ml": {
        "zh": [
            "一种机器学习模型的训练与分类推理方法，基于训练数据优化模型参数。",
            "面向分类任务的机器学习训练流程，提升模型推理精度与学习效率。",
            "利用机器学习进行模型训练与推理分类的特征学习方法。",
            "一种模型训练框架，通过机器学习迭代分类器并输出推理结果。",
        ],
        "en": [
            "A machine learning model training and classification inference method optimizing model parameters on training data.",
            "A machine learning training pipeline for classification tasks improves model inference accuracy and learning efficiency.",
            "A feature learning approach using machine learning for model training and inference classification.",
            "A model training framework iterates classifiers via machine learning and outputs inference results.",
        ],
        "de": [
            "Ein maschinelles Lernverfahren zum Training und zur Klassifizierungsinferenz von Modellen optimiert Parameter an Trainingsdaten.",
            "Eine Trainingspipeline des maschinellen Lernens für Klassifizierungsaufgaben verbessert Inferenzgenauigkeit und Lerneffizienz.",
            "Ein Merkmalslernverfahren mit maschinellem Lernen für Modelltraining und Klassifizierungsinferenz.",
            "Ein Modelltrainingsrahmen iteriert Klassifikatoren durch maschinelles Lernen und gibt Inferenzergebnisse aus.",
        ],
        "ja": [
            "機械学習モデルの訓練と分類推論方法であり、訓練データでモデルパラメータを最適化する。",
            "分類タスク向け機械学習訓練フローがモデルの推論精度と学習効率を向上する。",
            "機械学習によるモデル訓練と推論分類の特徴学習方法。",
            "機械学習で分類器を反復訓練し推論結果を出力するモデル訓練フレームワーク。",
        ],
    },
    "neural": {
        "zh": [
            "一种深度神经网络的卷积训练方法，通过神经元权重学习图像分类特征。",
            "面向神经网络的深度模型训练装置，调整神经元权重并优化卷积层。",
            "利用深度神经网络进行权重学习与神经元推理的分类方法。",
            "神经网络模型的卷积权重训练流程，提升深度神经元网络的推理表现。",
        ],
        "en": [
            "A convolutional training method for deep neural networks learns image classification features through neuron weights.",
            "A deep model training apparatus for neural networks adjusts neuron weights and optimizes convolutional layers.",
            "A classification approach using deep neural networks for weight learning and neuron-based inference.",
            "A convolutional weight training pipeline for neural network models improves deep neuron network inference.",
        ],
        "de": [
            "Ein faltendes Trainingsverfahren für tiefe neuronale Netze erlernt Bildklassifizierungsmerkmale über Neuron-Gewichte.",
            "Eine Trainingsvorrichtung für tiefe Modelle neuronaler Netze justiert Neuron-Gewichte und optimiert Faltungsschichten.",
            "Ein Klassifizierungsverfahren mit tiefen neuronalen Netzen für Gewichtslernen und neuronenbasierte Inferenz.",
            "Eine Faltungsgewicht-Trainingspipeline für neuronale Netzmodelle verbessert die Inferenz tiefer Neuronennetze.",
        ],
        "ja": [
            "深層ニューラルネットワークの畳み込み訓練方法で、ニューロン重みにより画像分類特徴を学習する。",
            "ニューラルネットワーク向け深層モデル訓練装置がニューロン重みを調整し畳み込み層を最適化する。",
            "深層ニューラルネットワークによる重み学習とニューロン推論の分類方法。",
            "ニューラルネットワークモデルの畳み込み重み訓練フローが深層ニューロンネットワークの推論を改善する。",
        ],
    },
    "edge_ai": {
        "zh": [
            "边缘计算节点上的本地推理与边缘智能方法，在端侧运行机器学习模型并压缩通信量。",
            "一种边缘AI的本地推理装置，边缘计算节点结合传感网络与模型推理降低传输流量。",
            "面向端侧的边缘智能本地推理框架，在物联网节点上执行机器学习分类并减少通信。",
            "边缘计算与边缘AI协同的本地推理系统，于网络边缘节点完成模型推理与传感分析。",
        ],
        "en": [
            "A local inference and edge AI method on edge computing nodes runs machine learning models on-device and compresses communication traffic.",
            "An edge AI local inference apparatus combines sensor networks with model inference at edge computing nodes to cut traffic.",
            "An on-device edge intelligence local inference framework executes machine learning classification on IoT nodes and reduces communication.",
            "A local inference system combining edge computing and edge AI performs model inference and sensing analysis at network edge nodes.",
        ],
        "de": [
            "Ein lokales Inferenz- und Edge-KI-Verfahren auf Edge-Computing-Knoten führt Modelle des maschinellen Lernens On-Device aus und komprimiert den Verkehr.",
            "Eine Edge-KI-lokale-Inferenz-Vorrichtung verbindet Sensornetzwerke mit Modellinferenz an Edge-Computing-Knoten zur Verkehrsreduktion.",
            "Ein On-Device-Edge-Intelligence-Rahmen für lokale Inferenz führt Klassifizierung des maschinellen Lernens auf IoT-Knoten aus.",
            "Ein lokales Inferenzsystem aus Edge-Computing und Edge-KI führt Modellinferenz und Sensoranalyse an Netzwerk-Edge-Knoten aus.",
        ],
        "ja": [
            "エッジコンピューティングノード上のローカル推論とエッジAI方法で、オンデバイスで機械学習モデルを実行し通信量を圧縮する。",
            "エッジAIのローカル推論装置がエッジコンピューティングノードでセンサーネットワークとモデル推論を組み合わせトラフィックを削減する。",
            "オンデバイスのエッジインテリジェンスローカル推論フレームワークがIoTノードで機械学習分類を実行し通信を削減する。",
            "エッジコンピューティングとエッジAI協調のローカル推論システムがネットワークエッジノードでモデル推論とセンサー分析を行う。",
        ],
    },
    "security": {
        "zh": [
            "一种数据加密与密钥认证方法，通过数字签名保障通信安全与身份认证。",
            "面向网络通信的加密认证装置，利用密钥签名机制提供安全认证服务。",
            "基于加密密钥与数字签名的安全认证协议，防止伪造并强化身份认证。",
            "一种通信安全加密方法，结合密钥管理与签名验证完成双向认证。",
        ],
        "en": [
            "A data encryption and key authentication method using digital signatures to ensure communication security and identity authentication.",
            "An encryption and authentication apparatus for network communication provides secure authentication via key signature mechanisms.",
            "A secure authentication protocol based on encryption keys and digital signatures prevents forgery and strengthens identity authentication.",
            "A communication security encryption method combines key management with signature verification for mutual authentication.",
        ],
        "de": [
            "Ein Datenverschlüsselungs- und Schlüsselauthentifizierungsverfahren mit digitalen Signaturen sichert Kommunikation und Identitätsauthentifizierung.",
            "Eine Verschlüsselungs- und Authentifizierungsvorrichtung für Netzwerkkommunikation bietet sichere Authentifizierung über Schlüsselsignaturen.",
            "Ein sicheres Authentifizierungsprotokoll auf Basis von Verschlüsselungsschlüsseln und digitalen Signaturen verhindert Fälschungen.",
            "Ein Kommunikationssicherheits-Verschlüsselungsverfahren kombiniert Schlüsselverwaltung und Signaturprüfung zur gegenseitigen Authentifizierung.",
        ],
        "ja": [
            "データ暗号化と鍵認証方法であり、デジタル署名により通信セキュリティと本人認証を保障する。",
            "ネットワーク通信向け暗号化認証装置が鍵署名機構でセキュアな認証サービスを提供する。",
            "暗号化鍵とデジタル署名に基づくセキュア認証プロトコルが偽造を防止し本人認証を強化する。",
            "通信セキュリティ暗号化方法が鍵管理と署名検証を組み合わせ双方向認証を完了する。",
        ],
    },
    "quantum": {
        "zh": [
            "一种基于量子纠缠的量子密钥分发方法，通过量子通信信道协商量子密钥。",
            "利用量子纠缠态实现的量子密钥协商装置，在量子通信链路中检测窃听。",
            "面向量子通信的量子密钥分发协议，基于纠缠光子保障密钥安全。",
            "一种量子密钥管理方法，借助量子纠缠与量子信道完成密钥分发。",
        ],
        "en": [
            "A quantum key distribution method based on quantum entanglement negotiates quantum keys over a quantum communication channel.",
            "A quantum key agreement apparatus using quantum entanglement states detects eavesdropping on quantum communication links.",
            "A quantum key distribution protocol for quantum communication secures keys via entangled photons.",
            "A quantum key management method completes key distribution using quantum entanglement and a quantum channel.",
        ],
        "de": [
            "Ein Quantenschlüssel-Verteilverfahren auf Basis der Quantenverschränkung handelt Quantenschlüssel über einen Quantenkommunikationskanal aus.",
            "Eine Quantenschlüssel-Vereinbarungsvorrichtung mit Quantenverschränkung erkennt Abhören auf Quantenkommunikationsverbindungen.",
            "Ein Quantenschlüssel-Verteilprotokoll für Quantenkommunikation sichert Schlüssel über verschränkte Photonen.",
            "Ein Quantenschlüssel-Managementverfahren führt die Schlüsselverteilung mit Quantenverschränkung und einem Quantenkanal durch.",
        ],
        "ja": [
            "量子もつれに基づく量子鍵配送方法で、量子通信チャネルを通じ量子鍵を合意する。",
            "量子もつれ状態を用いた量子鍵合意装置が量子通信リンクで盗聴を検出する。",
            "量子通信向け量子鍵配送プロトコルがもつれ光子により鍵のセキュリティを確保する。",
            "量子もつれと量子チャネルを用いて鍵配送を完了する量子鍵管理方法。",
        ],
    },
    "biotech": {
        "zh": [
            "一种基因测序与蛋白表达分析方法，通过细胞培养与发酵优化生物制备流程。",
            "利用基因编辑调控细胞蛋白合成的生物发酵装置，提升测序通量。",
            "面向生物细胞的基因测序试剂与蛋白发酵工艺，用于生物制剂生产。",
            "一种蛋白测序与基因检测结合的生物细胞培养发酵技术。",
        ],
        "en": [
            "A gene sequencing and protein expression analysis method optimizing biological preparation via cell culture and fermentation.",
            "A biological fermentation apparatus using gene editing to regulate cellular protein synthesis improves sequencing throughput.",
            "Gene sequencing reagents and protein fermentation processes for biological cells support biologic production.",
            "A biological cell culture and fermentation technique combining protein sequencing with gene detection.",
        ],
        "de": [
            "Ein Gen-Sequenzierungs- und Proteinexpressionsanalyseverfahren optimiert die biologische Herstellung über Zellkultur und Fermentation.",
            "Eine biologische Fermentationsvorrichtung mit Gen-Editierung zur Regulation der zellulären Proteinsynthese erhöht den Sequenzierungsdurchsatz.",
            "Gen-Sequenzierungsreagenzien und Proteinfermentationsverfahren für biologische Zellen dienen der Biologika-Produktion.",
            "Eine biologische Zellkultur- und Fermentationstechnik aus Proteinsequenzierung und Gen-Nachweis.",
        ],
        "ja": [
            "遺伝子シーケンシングとタンパク質発現分析方法で、細胞培養と発酵により生物製造フローを最適化する。",
            "遺伝子編集で細胞のタンパク質合成を制御する生物発酵装置がシーケンシング処理量を向上する。",
            "生物細胞向け遺伝子シーケンシング試薬とタンパク質発酵プロセスが生物製剤の生産に用いられる。",
            "タンパク質シーケンシングと遺伝子検出を組み合わせた生物細胞培養発酵技術。",
        ],
    },
}

JITTER = {
    "zh": ["该方案具备工程可行性。", "实验结果显示性能稳定。", "适用于多种部署场景。", "整体流程可重复实施。"],
    "en": [
        "The scheme is engineering feasible.",
        "Experiments show stable performance.",
        "It applies to multiple deployment scenarios.",
        "The overall pipeline is reproducible.",
    ],
    "de": [
        "Das Verfahren ist ingenieurtechnisch umsetzbar.",
        "Experimente zeigen stabile Leistung.",
        "Es eignet sich für mehrere Einsatzszenarien.",
        "Der Gesamtablauf ist reproduzierbar.",
    ],
    "ja": [
        "本方案は工学的に実現可能である。",
        "実験結果は安定した性能を示す。",
        "複数の展開シナリオに適用できる。",
        "全体フローは再現可能である。",
    ],
}

# --------------------------------------------------------------------------- #
# 家族计划：(主题, 家族数, 语种组合, 起始日, 状态, v2真值)
# 状态：labeled / missing_annotation / noisy_label / new_topic / short
# --------------------------------------------------------------------------- #
PLAN = [
    ("telecom", 10, ["zh", "en", "de"], date(2020, 2, 1), "labeled", ["telecom_network"]),
    ("telecom", 4, ["en", "ja"], date(2023, 3, 1), "labeled", ["telecom_network"]),
    ("iot", 10, ["zh", "en", "ja"], date(2020, 5, 1), "labeled", ["iot_network"]),
    ("iot", 4, ["zh", "de"], date(2023, 6, 1), "labeled", ["iot_network"]),
    ("wireless", 8, ["zh", "en", "de", "ja"], date(2020, 8, 1), "labeled", ["telecom_network"]),
    ("ml", 10, ["zh", "en"], date(2020, 3, 1), "labeled", ["ai"]),
    ("ml", 3, ["de", "ja"], date(2023, 2, 1), "labeled", ["ai"]),
    ("neural", 8, ["zh", "en", "de"], date(2020, 6, 1), "labeled", ["ai", "neural_net"]),
    ("neural", 3, ["zh", "en"], date(2023, 4, 1), "labeled", ["ai", "neural_net"]),
    ("edge_ai", 6, ["zh", "en", "ja"], date(2023, 2, 1), "labeled", ["iot_network", "ai", "edge_ai"]),
    ("edge_ai", 3, ["zh", "en"], date(2021, 6, 1), "labeled", ["network", "ai", "edge_ai"]),
    ("security", 8, ["zh", "en", "de"], date(2020, 4, 1), "labeled", ["cyber_security"]),
    ("security", 3, ["en", "ja"], date(2023, 5, 1), "labeled", ["cyber_security"]),
    # 2022 窗口保证有标注家族（开发段校准/阈值需要）
    ("telecom", 3, ["zh", "en"], date(2022, 2, 1), "labeled", ["telecom_network"]),
    ("iot", 3, ["zh", "en", "ja"], date(2022, 4, 1), "labeled", ["iot_network"]),
    ("ml", 3, ["en", "de"], date(2022, 6, 1), "labeled", ["ai"]),
    ("neural", 2, ["zh", "en"], date(2022, 8, 1), "labeled", ["ai", "neural_net"]),
    ("security", 2, ["zh", "en"], date(2022, 9, 1), "labeled", ["cyber_security"]),
    ("quantum", 4, ["zh", "en"], date(2021, 2, 1), "new_topic", ["quantum_key"]),
    ("quantum", 4, ["zh", "en", "de"], date(2023, 3, 1), "labeled", ["cyber_security", "quantum_key"]),
    ("biotech", 4, ["zh", "en"], date(2022, 2, 1), "new_topic", ["biotech_process"]),
    ("biotech", 3, ["zh", "en", "ja"], date(2022, 7, 1), "new_topic", ["biotech_process"]),
    ("biotech", 2, ["zh", "en", "ja"], date(2023, 7, 1), "new_topic", ["biotech_process"]),
    ("iot", 2, ["zh", "en"], date(2021, 9, 1), "missing_annotation", ["iot_network"]),
    ("neural", 2, ["zh", "en"], date(2021, 10, 1), "noisy_label", ["ai", "neural_net"]),
    ("security", 1, ["zh"], date(2024, 2, 1), "short", ["cyber_security"]),
]

SHORT_TEXTS = {"zh": "加密。", "en": "Crypto.", "de": "Kryptographie.", "ja": "暗号。"}

# v1 主题 -> v1 标签节点
V1_LABEL = {
    "telecom": "network",
    "iot": "network",
    "wireless": "wireless",
    "ml": "ai",
    "neural": "ai",
    "edge_ai": "network",  # v1 无交叉细类，只能挂 network（专家另补 ai 见下）
    "security": "security",
}


def build() -> None:
    rng = Lcg()
    OUT.mkdir(parents=True, exist_ok=True)
    taxonomy_rows, doc_rows, label_rows, truth_rows = [], [], [], []
    family_seq = 0
    doc_seq = 0
    label_seq = 0
    family_topics: dict[str, tuple[str, date, str]] = {}

    for topic, count, languages, start_day, status, truth_nodes in PLAN:
        for k in range(count):
            family_seq += 1
            fid = f"fam{family_seq:03d}"
            family_topics[fid] = (topic, start_day, status)
            priority = start_day + timedelta(days=20 * k + rng.below(12))
            for li, language in enumerate(languages):
                doc_seq += 1
                doc_id = f"doc{doc_seq:04d}"
                published = priority + timedelta(days=25 + 18 * li)
                if status == "short":
                    text = SHORT_TEXTS[language]
                else:
                    variant = TEMPLATES[topic][language][(k + li) % 4]
                    text = f"{variant}{rng.pick(JITTER[language])}"
                doc_rows.append(
                    {
                        "id": doc_id,
                        "family": fid,
                        "language": language,
                        "published": published.isoformat(),
                        "priority": priority.isoformat(),
                        "text": text,
                    }
                )
            first_published = priority + timedelta(days=25)
            catalog_version = 1 if first_published < date(2023, 1, 1) else 2
            confirmed = first_published + timedelta(days=60)
            if status == "labeled":
                if catalog_version == 1:
                    nodes = [V1_LABEL[topic]]
                    if topic == "edge_ai":
                        nodes = ["network", "ai"]  # v1 交叉技术只能双挂宏观类
                else:
                    nodes = list(truth_nodes)
                    nodes = [n for n in nodes if n not in ("biotech_process",)]
                for node in nodes:
                    label_seq += 1
                    label_rows.append(
                        {
                            "id": f"lab{label_seq:04d}",
                            "document": doc_rows[doc_seq - len(languages)]["id"],
                            "node": node,
                            "taxonomy_version": catalog_version,
                            "confirmed_at": confirmed.isoformat(),
                        }
                    )
            elif status == "noisy_label":
                # 人工错挂到 network（实际是神经网络），复刻“网络”关键词污染
                label_seq += 1
                label_rows.append(
                    {
                        "id": f"lab{label_seq:04d}",
                        "document": doc_rows[doc_seq - len(languages)]["id"],
                        "node": "network",
                        "taxonomy_version": 1,
                        "confirmed_at": confirmed.isoformat(),
                    }
                )
            # missing_annotation / new_topic / short 不写标签（short 仍有真值）
            status_map = {
                "new_topic": "new_topic",
                "missing_annotation": "missing_annotation",
                "noisy_label": "noisy_label",
                "short": "short_abstract",
            }
            truth_rows.append(
                {
                    "family": fid,
                    "nodes": truth_nodes,
                    "version": 2,
                    "status": status_map.get(status, "labeled"),
                    "note": {
                        "new_topic": "目录无法容纳或当时尚无该类",
                        "missing_annotation": "专家确认相关但漏标",
                        "noisy_label": "人工标签与真值冲突",
                        "short_abstract": "摘要过短应触发拒识",
                        "labeled": "",
                    }[status_map.get(status, "labeled")],
                }
            )

    # ---- v2 上线回填：对 2023 年前公开的已标注家族补登 v2 细类标签 ----
    # 真实目录升级时，审查员会把存量案例按迁移规则重新归类；confirmed_at 在 v2
    # 生效之后，但文档公开日更早——时点检查只要求公开日不早于节点生效日的标签
    # 登记版本，回填标签的 taxonomy_version=2 且节点在 v2 存在即合法。
    backfill_nodes = {
        "telecom": ["telecom_network"],
        "iot": ["iot_network"],
        "wireless": ["telecom_network"],
        "ml": ["ai"],
        "neural": ["ai", "neural_net"],
        "edge_ai": ["iot_network", "ai", "edge_ai"],
        "security": ["cyber_security"],
        "quantum": ["cyber_security", "quantum_key"],
    }
    docs_by_family: dict[str, list[str]] = {}
    for row in doc_rows:
        docs_by_family.setdefault(row["family"], []).append(row["id"])
    backfill_cutoff = date(2023, 1, 1)
    for fid, (topic, start_day, status) in sorted(family_topics.items()):
        if status != "labeled" or topic not in backfill_nodes:
            continue
        first_doc = docs_by_family[fid][0]
        first_pub = next(r["published"] for r in doc_rows if r["id"] == first_doc)
        if date.fromisoformat(first_pub) >= backfill_cutoff:
            continue  # 2023 后家族已直接登记 v2 标签
        for node in backfill_nodes[topic]:
            label_seq += 1
            label_rows.append(
                {
                    "id": f"lab{label_seq:04d}",
                    "document": first_doc,
                    "node": node,
                    "taxonomy_version": 2,
                    "confirmed_at": "2023-02-15",
                }
            )

    for node_id, parents, introduced, deprecated in TAXONOMY:
        taxonomy_rows.append(
            {
                "id": node_id,
                "parents": parents,
                "version": 1,
                "introduced": introduced,
                "deprecated": deprecated,
            }
        )

    _write_jsonl_with_manifest(taxonomy_rows, doc_rows, label_rows)
    (OUT / "catalog-history.json").write_text(
        json.dumps(HISTORY_SPEC, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "hidden-truth.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in truth_rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "families": family_seq,
                "documents": doc_seq,
                "labels": label_seq,
                "truth_rows": len(truth_rows),
                "output": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_jsonl_with_manifest(taxonomy_rows, doc_rows, label_rows) -> None:
    specs = [
        ("taxonomy", "demo-taxonomy.jsonl", taxonomy_rows),
        ("documents", "demo-documents.jsonl", doc_rows),
        ("labels", "demo-labels.jsonl", label_rows),
    ]
    manifest = {"schema": 1, "tables": {}}
    for name, filename, rows in specs:
        content = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
        (OUT / filename).write_bytes(content)
        manifest["tables"][name] = {
            "file": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": len(rows),
            "fields": list(rows[0].keys()),
        }
    (OUT / "demo.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    build()
