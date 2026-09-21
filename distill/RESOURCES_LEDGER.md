# 资源提炼总账（2026-09-21 五批清单全量验证）

## 今日战果（已标注入库）

| 批次 | 行数 | 教师 |
| --- | ---: | --- |
| 工单域 POC+v2+v3+v4 | 19,360 | jev-1.13.0 |
| verdict T01 钓鱼+银行 | 1,500 | jev-1.13.0 |
| verdict T02-T05 | 2,000 | jev-1.13.0 |
| 书块（Sutton 70MB + MIT AlgDM 12MB） | 9,962 | jev-1.13.0 |
| 多语言书块（NLPBookTranslations，6 语言） | 4,980 | jev-1.13.0 |
| **今日合计** | **~37,800** | unified 总题数（37,802） |

## 种子队列（待标注，cron 逐日消化）

| 队列文件 | 规模 | 说明 |
| --- | ---: | --- |
| `seeds_c13k.jsonl` | 14,568 | choices13k 风险选择（jcpeterson，真实学术集），A/B 彩票自然语言化，含 EV 金标 |
| `seeds_mple.jsonl` | 240 | MultiPL-E 代码决策（nuprl，8 语言 humaneval） |
| mlbooks 剩余 | ~37,000 | 6 语言均衡续标 |
| verdict3 收尾 | ~2,500 | T06-T10 |

## 真实仓库（已克隆 /root/research/，25 个）

**决策模型实现（配方参考）**
- bespokelabsai/nimble（1344★）：本地 typed decisions + 对比数据整理
- Zefan-Cai/Open-Jev（39★）：Qwen3.5-2B/9B/27B 决策头+校准温度
- scienthoon/luce（7★）：init→synth→train→eval→serve 完整配方，acc 91.1% / ECE 0.022
- zhihz/openjev（26★）：双语中英概率决策，Apple M3 本地
- intikhab49/open-jev-typed-decision-engine + Heman10x-NGU/openJev-verdict：150M 编码器路线
- TianyuCodings/NanoJev：当前学生架构底座

**代码推理**
- hkust-nlp/CodeIO（573★，ICML 2025 Oral）：代码 I/O 预测浓缩推理模式
- griffinbholt/decisionmaking-code-py（59★）：MIT AlgDM 全书 Python 实现
- Anfeather/Logic-Distillation（IJCAI 2025）、nweir127/CoGEX

**多语言方法（阶段 5b RLCD/对齐备选）**
- smoothie-qwen（109★）：token 概率平滑增强多语言，轻量可集成
- NJUNLP/MAPO（44★）：多语言对齐=偏好优化
- ZNLP/Language-Imbalance-Driven-Rewarding（25★，ICLR 2025）：语言不平衡做奖励
- cisnlp/COPSD、NJUNLP/RP-OPSD：跨语言在线策略自蒸馏
- mbzuai-nlp/bactrian-x（97★）：52 语言 3.4M 指令对（种子源）
- Helsinki-NLP/OpusDistillery：多语言 NMT 蒸馏流水线

**场景源（对话/议价种子）**
- shaxiu/XianyuAutoAgent（9220★）：闲鱼智能议价，中文对话场景
- wjl8636/AI-Customer-Service-Assistant（13★）：电商客服 10 章演进代码
- aliyun/qwen-dianjin：金融大模型，内含 DianJin-CSC/CSConv 数据管线
- haimianxing/CCSE-CS-Empathetic-Culture-AI：76,847 轮中文客服（4 域，EMNLP 2026）
- daseinlabs/open-jev、NiuTrans/NLPBookTranslations（12 个多语言 PDF）、jcpeterson/choices13k

## 幻觉记录（清单点名但不存在，防重复踩坑）

- "Good Thinking Corpus v1.0"（27,252 条）——全网无踪
- Aphelios-Tang/Decision_CaT —— 不存在
- eilab-gt/wopr —— 不存在
- SHerZH/IMLJP —— 不存在

## 待验证/待接入（下一步队列）

- Newcomb-like 数据集：arXiv 2411.10588（Oesterheld 2024，真实论文）配 repo
- UniMoral（6 语言道德推理）、HALT/UAVBench/AgentDrive（智能体基准，名字未验证）
- OpenCodeReasoning（NVIDIA 736,712 样本，HF: nvidia/OpenCodeReasoning）
- JDDC 2.1（京东 24.6 万多模态对话）
- Open Bandit Dataset（ZOZOTOWN 2600 万行）
- 开放决策书：Zollman《Theories of Rational Decisions》、《Real-Life Decision-Making》(CC BY-NC-ND)、Chernoff《Elementary Decision Theory》

## 版权原则（持续有效）

书文本提取物与版权保留数据**不入 git**；Apache-2.0/MIT/学术公开数据可入；
标注产物含 state 原文的按来源许可决定。GitHub 只入脚本、统计与许可明确的数据。
