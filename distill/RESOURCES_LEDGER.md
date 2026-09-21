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

## 第十~十二批（CoT/数学/书籍世界观，HF 探测 8/8 真实）

### 已拉取（第三通道，PID 21824）
- oncu/LiteCoT：10 万简短推理（难度感知蒸馏 DAD，10万 LiteCoT > 80万长 CoT）——效率优先教材
- Rorosko/Open-CoT-Reasoning-Mini：10,200 条，专为 10B 以下学生设计（input+output 双标签）——**与 0.6B 学生直接对口**
- AI-MO/NuminaMath-CoT：86 万数学竞赛 CoT——数学域主力

### 验证为真，择要后拉（磁盘约束）
- nvidia/OpenMathInstruct-2（1400 万对）、nvidia/Nemotron-Math（750 万轨迹，100% maj@16）
- PAI-2026/OmniThought（200 万 CoT，RV/CD 双指标标注，ACL 2026）
- openthoughts/OpenThoughts2-1M（ICLR 2026）
- harvard-lil/institutional-books-1.0（98.3 万本/2420 亿 token，250+ 语言）——世界观语料主力，需分片拉取
- claude-opus-4.6-4.7-reasoning-8.7k（8,706 条 28 类，Apache-2.0）
- CODI（EMNLP 2025，CoT 压缩到连续空间，gpt2/llama3.2-1b 权重）
- mCoT-MATH（630 万 11 语言数学 CoT）

### 书籍世界观语料清单（按版权稳妥度排序）
1. Project Gutenberg（7 万本公版）+ PG-19 长文本
2. harvard-lil/institutional-books-1.0（公域，分片拉）
3. Common Corpus（200-300 万本书）
4. Academic Textbook Corpora（3.9 万教材/5000 学科/15 语言）
5. OWL（20 书 × 10 语言对齐摘录）、EvolvingWorld（57 书 13.8 万样本）
6. ⚠️ Books3（19.7 万本混合版权）——记录但不用，合规风险

### 配比经验（清单给出）
高质量书籍/教材在预训练中需 3-5 倍过采样；CoT 课程学习（短→长）用全局排序。

## 第十三~二十批（游戏/Galgame/Minecraft/代码审查/ClaudeCode 学习资源，全量克隆）

**仓库总量：62 个（/root/research/），覆盖九批清单全部点名项。**

### 游戏/Galgame/剧本杀（决策事件语料）
- Rushes（44,226 决策事件）、MACHIAVELLI（134 游戏 50 万场景）、CHADPOD（1,462 决策点分类）
- FIREBALL（D&D 800 万话语）、DDD（5,600 万 token RPG）、Klondike 28,904 轨迹
- Jubensha 剧本杀数据集（1,100+ 实例，中文）、YMgal data-export（JSONL galgame 数据）
- Orak/SpireEval/Game Reasoning Arena/TowerMind/LudoBench：战略决策基准族
- 中文游戏：SVFSearch、ZZZDialog/GenshinDialog 提取器、Mini Trade Game NPC（繁中 JSON NPC）

### Minecraft（服务端反编译代码，用户点名"jar 已无混淆"）
- MGOA 25,000 视频/3,000 万三元组（CVPR 2025）、PLAICraft 万小时五模态、VPT 5,390 万帧
- MineDojo 知识库（73 万视频/6 千 Wiki/34 万 Reddit）、MineMA 39 万指令、Odyssey 数据全开
- Forge Coder（22,916 Java 文件微调）、MinecraftModSources（2,000 插件 50 万源文件）
- CraftGround（300 TPS 高性能 RL 环境）、MineLand 48 智能体、MineStudio 全流程包
- 提炼动作：官方无混淆 jar → CFR/Vineflower 反编译 → 代码决策语料（用户确认可行，排入队列）

### 代码审查（与 decide 层/OCR 工作流对口）
- alibaba/open-code-review（工业级混合架构）、aacr-bench（200 PR/10 语言金标）
- CuREV/PR-Review-Bench/HistoryCR（17.7 万 PR）等 8 个数据集
- reviewdog/semgrep/static-analysis 静态分析链

### Claude Code 学习资源（lintsinghua/claude-code-book + zhang588/Claude-Code-OrangeBook 已克隆）
- claude-code-book：四部分结构（基础/核心系统/高级模式/工程实践）
- OrangeBook：橙皮书 v2.0.0 PDF
- 社区生态：claude-code-tips(6k★)/Learn Claude Code(26.7k★)/internals-orange-book 等 20+ 指南记录

### 世界观开源项目（第十批补充）
- EvolvingWorld（57 书 13.8 万样本）、RLVR-World、stable-worldmodel、LingBot-World 2.0（1.3B/14B）
- CSKG/ConceptNet/FactNet（17 亿断言/30 亿证据指针）、NovaCOMET、Theogony
- Cosmopedia（3000 万样本/250 亿 token）、Common Corpus（1.99T token）
- 评估：EWoK-core-1.0（4,374 条 11 域）、KoLA、WorldVQA、BeQu

### 四阶段整合策略（采纳为提炼路线图）
1. 基础决策：Good Thinking(逻辑)+choices13k(交互)+SpireEval(战略)
2. 领域注入：电商 CASCADE + 客服 CSConv/CCSE-CS/BANKING77 + 烹饪 RecipeLLM
3. 多模态推理：LongPerceptualThoughts + DriveLMM-oL + Cosmopedia
4. 世界模型+蒸馏：stable-worldmodel + EasyDistill 收口；多语言 CUTE/WanJuanSiLu

### 格式统一铁律（提炼终点）
全部数据 → prompt + response + 可选 reasoning 三元组 → 3-5 倍过采样决策/CoT 数据 → GAMEBoT/Arena 定期评估闭环。
