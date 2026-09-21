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

## 补充批次（2026-09-21 深夜，第六~九批清单）

### 新克隆（真实，已验证）
- zwhong714/Hybrid-Policy-Distillation（25★，ICML 2026）：Qwen2.5-1.5B←7B-Thinking 推理蒸馏，LlamaFactory+verl 双实现
- horus-ai-labs/DistillFlow（169★）：logits/注意力/中间层三策略 + 动态资源分配
- songmzhang/DSKD（65★，EMNLP 2024）：跨 tokenizer 白盒蒸馏（72B→1.5B 实证）
- jingyaogong/minimind（61.9k★）：64M 全流程（Pretrain/SFT/LoRA/DPO）
- bojieli/ai-infra-book（4.8k★）：AI Infra 量化推导开源书
- 另验证为真未克隆：rasbt/LLMs-from-scratch（105k★）、system-design-primer、UniMoral、OpusDistillery

### HF 数据集直接拉取（现成指令-响应对，无需教师，PID 8856 进行中）
- Mxode/Chinese-Instruct（中文大规模指令集，含 dpsk-r1-distil/chinese-reasoning-distil 配置）
- Mxode/Chinese-Reasoning-Distil-Data（中文推理蒸馏）
- Mxode/Meow-Reasoning-100K（10 万中文推理）
- 落位 /root/materials/Mxode__*，本地训练用

### 果蝇数据集（用户点名，科学数据，记录在案）
- FlyWire 全脑连接组（14 万神经元/5000 万突触，Nature 2024）+ FAFB/Hemibrain
- 30,000+ 果蝇行为轨迹（Scientific Data 2025, DOI:10.1038/s41597-025-04724-3）
- Tephritid26（38,081 张实蝇图像，26 种）
- 8 阶段发育分类（Drosophila_stages_models，ResNet-50 85%）
- 用途：远期多模态任务 #16 的视觉分类种子源

### 人类决策神经科学数据（行为决策研究链）
- DfE-DB（380 万条经验决策记录）/ ITC Database（117 万条跨期选择）
- 100K Choice Dilemmas（10 万真实选择困境，PNAS 2025 属性分析）——文本决策直接素材
- IGT/囚徒困境 EEG、混合赌博 fMRI（OpenNeuro）——神经基线参考
- 用途：choice 题型与"人类偏好 vs 理性最优"的校准分析素材

### 蒸馏模型成果（可直接下载权重，按需拉取）
- DeepSeek-R1 蒸馏系列（91.9k★，MIT）：6 规格 SFT 权重（HF: deepseek-ai/*）
- DistilBERT/TinyBERT/MiniLLM：经典小模型蒸馏检查点
- Align-TI-1B（25★，ICML 2026 token 交互蒸馏，HF 权重）
- Siglino（60★，CVPR 2026 视觉编码器蒸馏 5 检查点）
- Multi-Level-OT（39★，AAAI 2025 Oral，跨 tokenizer 学生检查点）
- Byrne-VLM-131M（131M VLM，571MB）
- 记录：mimi_0.6b、PolyDistill、DistillDetect 名字未经验证，拉取前需 probe

### 蒸馏文本数据集（直接 SFT 用，验证后拉取）
- Chinese-DeepSeek-R1-Distill-data-110k（11 万中文，数学 36,568+通用 58,352，Math-Verify 校验）——P0，与中文学生对口
- "十万个为什么"中文百科（120 万+指令，general/preference/reasoning 三子集）——P1
- OpenMathInstruct-1（180 万数学对，商业友好）——P1
- TART（344k 推理轨迹）/ Reasoning Corpus（500 万条，SupraLabs）——量级大，探测后择要
- open-distillation-datasets（蒸馏数据目录，持续发现入口）
