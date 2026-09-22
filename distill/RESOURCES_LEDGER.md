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

## qwen3.txt 资料总集（2026-09-22，2740 行多主题对话存档）
原文存档：distill/qwen3_resources_archive.txt；工程陷阱原文：/tmp/upcycle_risks.md

### 最高优先级：AlexWortega/moe-600m-qwen3-upcycle 的 UPCYCLE.md
Qwen3-0.6B upcycle 工程陷阱的社区记录，与我们的管线直接对应：
1. RoPE half-split 陷阱：interleaved vs half-split 静默打乱 attention（偏差 2.58，
   症状="upcycling 效果不好"）。我们用 transformers 原生实现，不受影响。
2. **专家构建策略差异（升级方向）**：他们用重要性切分（E[a_j^2]*||down[:,j]||^2 排序，
   shared expert 拿 top-768 承载 47% 重要度质量，routed 蛇形均分 CV=0.0001），
   我们用复制+微扰（专家同质化风险）。upcycle_to_moe.py 的下一版改造方向。
3. parity check 门禁：RoPE/logits 与 HF 逐位对比（max|d|=0）后才允许训练——该加。
4. 其他静默雷：muP 缩放、weight_decay 侵蚀 embedding/norm（~20%）、EMA 冗余。

### 代码审查素材（已验证本地有 aacr-bench）
- AACR-Bench（评估+训练素材）、CuREV（17.7 万 PR）——代码审查决策直接素材。

### 大规模语料（超本土化需求时启用）
HPLT 3.0(30T tokens/200 语言)、Common Corpus(1.99T)、MNBVC(60TB 中文)、
CCI 4.0(35TB 中文)、Institutional Books 1.0(98.3 万本书)、FineTranslations(万亿级平行)。

### CoT/蒸馏全家桶
claude-opus-4.6-4.7-reasoning-8.7k、OmniThought(RV/CD 双指标)、CoT-Trace-Inverted-28K
(课程学习：短→长渐进)、Open-CoT-Reasoning-Mini(10B 以下专用)、TART(34.4 万)、
Dolci-Think-SFT-32B-Multilingual(32K 长序列六语言)、mCoT-MATH(630 万/11 语言)、
CODI(连续空间自蒸馏)、TRS(技能卡片蒸馏, ACL 2026 Oral)。

### 决策理论电子书（合成数据种子语料）
Foundations of Computational Decision Analysis、MIT《Algorithms for Decision Making》
(代码库配套)、MDP+RL(Puterman)、Good Thinking Corpus v1.0(27,252 条/182 分类码，
Zenodo)、Risky Choices(=choices13k 自然语言重构,已用)、Newcomb-like(已用)、
《Theories of Rational Decisions》(Zollman)、《An Introduction to Cognitive Economics》。

### 搜索增强（此前判定不适用，维持）
Search-R1/ASearcher/SAIL/s3/StepSearch 等——decide 层无检索循环，架构不适用。
GuarantRAG/RE-IAG/对比解码——RAG 质量技术，若做 RAG 路线再启用。

### Minecraft 生态（与决策产品弱关联，暂存档）
CraftGround(300 TPS RL 环境)、MineStudio、Agent Society Distill(1,355 SFT)、
Forge Coder(22,916 Java 文件)、MinecraftModSources(50 万源文件)。

## qwen3.txt 增量补录（250-2059 区段，此前跳读遗漏的内容）

### 决策数据集重磅清单（直接对口产品核心，此前漏读）
- **DfE-DB：380 万条人类决策记录**（系统性数据库，多任务经验决策）——决策域最大单一源。
- **ITC Database：117 万条跨期选择**（intertemporal choice，直接对应我们的风险决策题式）。
- **MACHIAVELLI：50 万场景**（research/ 本地已有仓库，未利用！）。
- **HALT Benchmark**：1,248 实例，"何时停止/调查/升级/拒绝"——与 decide 层的弃答
  （Seal [REJ] token）设计直接对应。
- **LISTEN Benchmark**：LLM 从大量候选集引出用户偏好并选最佳——set-attention 的评估面。
- **UAVBench 5 万无人机场景 / AgentDrive 30 万驾驶场景**——高可靠性决策域数据。
- **GameTheory-Bench 2,146 题专为 RLVR 设计**、QualGames（行为博弈论）、
  Dictator Game Benchmark（12 国）、CaSiNo（谈判对话）、WereBench（狼人杀 15 规则变体）。
- **SportD 1,415 足球决策场景 / NFL 4th Down（1999-2025 逐场）**——体育决策。
- **FiFAR**：50 名欺诈分析师对 3 万实例的预测——人机决策"学习延迟"素材。
- 法律域：LFPBench、IMLJP（中文刑事判决预测）、MultiJustice、Arabic-LJP。
- 医疗域：llm-alignable-dm（62 分诊场景+决策者属性）、MIMIC-SR-ICD11、
  Clinical-Tool-Learning（RiskCalcs/RiskQA）。
- 金融域：SME Credit Risk 1,200 份、BD-SME-Credit-Risk、UCI_sft_3000
  （**首个中文信用违约 SFT，结构化数据序列化为自然语言对话**——与我们的
  state+questions 格式高度同构）。

### 果蝇资料全景（用户保留项，非"删除"）
发育阶段图像（300 张/类×8 阶段）、Tephritid26（38,081 图/26 种检疫实蝇）、
SpaceAnimal（中国空间站多动物姿态）、30,000 只行为轨迹（Scientific Data 2025）、
FlyWire 全脑连接组（14 万神经元/5,000 万突触）、FAFB、雄性 CNS、Hemibrain、
FlyAtlas 2、衰老细胞图谱、Genome Nexus。**用户说"去掉果蝇"指的是管线移除，
资料保留在存档中备用。**

### 推理蒸馏工具包（比 CODI 更轻的选项）
- **R-Chain（modelscope/r-chain）**：轻量级，系统复现 R1 蒸馏流程，含 MathR 数据集。
- **ReasonLite-0.6B（AMD）**：6 亿参数数学推理模型，全开源（权重+数据+代码），
  课程蒸馏——与我们 0.6B 学生完全同规格的先行者。
- **DRP**：数学技能感知步骤分解蒸馏，GSM8K token 917→328 且准确率 91.7→94.1。
- **Chinese-Data-Distill-From-R1**：11 万条中文蒸馏（数学 36,568），Math-Verify 校验。
- **Chinese Reasoning Dataset v1**：10,140 条中文数学逻辑（100% 验证无幻觉）。
- **CAPC-CG**：中文政策指令开放数据集 330 万段落（research/ 本地已有）。
- **OpenRobotHarness-Data-v0.1**：中文机器人 Harness 决策层（澄清/重规划）。

### 中文小说语料全景（stage 1 长文本热身备选）
webnovel-chinese（9B tokens 已清洗 jsonl）、webnovel_cn（2,170 万条）、
MNBVC 60TB、Qidian-Webnovel-Corpus（110 本+读者评论）、MultiGenre-ChineseNovel
（13 体裁）、BeyondDialogue（123 本小说角色对话）、novel-agent-sft-dataset
（669 本场景分割/角色归因标注）、Dxniz/Novelist（文笔/世界观/张力评分标注）。

### 训练方法论参考（校准方向）
- **Luce**：校准决策模型完整配方（init→synth→train→eval→serve），输出诚实概率。
- **ifllm-learn / Bespoke Nimble**：本地类型化 LLM 决策 + 概率校准——与我们的
  "类型化决策引擎"同型项目，配方可参考。
- **Logic-Distillation（IJCAI 2025）**：逐函数从代码学习，决策任务逻辑蒸馏。
- **Superior-Reasoning-SFT**：DASD pipeline（温度调度学习+散度感知采样+混合策略
  蒸馏），HF 趋势榜 #1，少数据 SOTA——与我们"质量优先"路线同向。
- **The Smol Training Playbook（HF）**：何时从头训练的决策流程图 + SFT/DPO/GRPO 踩坑。

## 模型正式命名（2026-09-22 用户拍板）
**Apeireth-Decis-2.6B-128k**
- 2.6B = 总参数（checkpoint 2,446M + tied lm_head 156M；激活 ~1.02B，8 专家 top-2）
- 128k = YaRN 上下文（原生 40,960 × factor 3.2，theta 1e6）
- 底座：Qwen3-0.6B 稠密 → 8 专家 upcycling（α 校准 + gate [E,H]）
- 部署：不量化；APK 下载接口 → 本地加载（FP32 9.8G 或加载时转 BF16 4.9G）
- HF 归属：congyu778/duan-nanojev（best.safetensors + config）
- 血统：v9 管线（upcycle→热身→600 步决策微调，143k 题教师数据）

## NanoJev README 情报（github.com/TianyuCodings/NanoJev，2026-09-22 全读）
- **定位**：Jev 的 nano 复刻（0.6B 并行决策模型，zero output-token decoding）。
- **成绩**：ViZDoom Basic 128/128（Jev 56/128，2.3x）；Maze 225 attempts（Jev 2,738）；
  Predict Position 27/128；Snake 8/8。548 评估案例/模型。
- **数据已接入**：unified/hard 五分区 18,760 题（10,898 训练）已合并进 unified 主线 ✓。
- **未用增量**：896 专家 episodes（17,498 记录决策，512 给训练）——predict_position
  软标签来源，下一轮蒸馏可出题。
- **训练配方**：hard_lr1e5 臂（one-hot），混合权重 Maze 1/3 Snake 1/3 Basic 1/6
  Predict 1/6；决策头 lr 1e-4、backbone lr 1e-5（与我们一致）。
- **JevHarness（新项目）**：LLM 构建任务特定决策 harness + 奖励/轨迹精炼 +
  Pokémon demo——设计参考。
- **tokenizer 警告**：fix_mistral_regex flag 存在（comment 6 证实），但 Qwen3 预训练
  分词行为未确认——决策：不加 flag，保持基座一致性（训练/推理闭环自洽）。

## NanoJev-Data soft 臂（2026-09-22 接入，本轮真实落盘）
- **soft/train.jsonl 70MB**：teacher.native_probs 是 jev 教师完整软分布
  （如 west 0.7/south 0.24/east 0.03）。已转换合并 2,421 题（含评估分区）。
- **SONIC_PREDICT_POSITION.md**：专家=sonic_doom 的 CNN+GRU 视觉策略，
  双 ViZDoom 实例同步采集（tick 级对齐、独立 replay 验证）。
  17,498 决策记录；frozen cohort 896 episodes（train 512/dev 64/cal 64/test 128/OOD 128）。
  OOD 换决策节奏（8-tick）；监督集中在正弹药状态的可影响轨迹决策。
- **意义**：predict_position 11,173 题中 6,788 训练题的专家级监督链路完整
  （专家 logits→四动作分布→replay 验证），episodes 可直接复用出软标签。
- **fix_mistral_regex**：flag 存在（已证实），决策不加——保持 Qwen3 基座分词一致性。

## 反幻觉/反谄媚方案（2026-09-22 用户资料，登记待启用）
- **TruthRL 三元奖励（正确/幻觉/弃权）**——与 decide 层的 Seal [REJ] 弃答机制
  直接对应：让"我不知道"成为可奖励选项。优先级 P1（stage 2 损失改造）。
- **RLCR 校准奖励**：输出置信度与真实正确率匹配——我们的 teacher_confidence
  已是现成校准目标。
- **Confessions 自白机制**：主答案后附自白，仅按诚实度奖励。
- 评估：HaluEval（3 万）、C-FAITH（中文细粒度 6 万）、TruthfulQA。
- **反谄媚**：对比偏好对（坚持事实 vs 迎合用户）。
- 采纳判定：我们的模型是单次前向分类决策（非生成），幻觉形态=错选项而非编造。
  最直接的干预：TruthRL 三元（弃答可奖励）+ RLCR（置信度校准），二者都
  落在现有 loss 结构内，改造成本低。

## Light-MER 框架解剖（2026-09-22，多模态线启动，任务 #16）
- **仓库**：research/light-mer（20MB，含 SWD-H + M-GRPO 双阶段）
- **SWD-H 核心**：
  - my_affectgpt/models/ot_loss.py:141 SWDProjector（教师正交冻结投影→学生维度）
  - ot_loss.py:163 sliced_wasserstein_loss（100 随机 1D 投影→排序→W-p 距离，
    answer_mask 隔离有效位）
  - affectgpt.py:930 接线：ot_hidden_loss = sliced_wasserstein_loss(...)
- **架构配对**：教师 Qwen3-8B + CLIP-ViT-L-14 + HuBERT-L（9B/20GB）
  → 学生 Qwen3-0.6B + CLIP-ViT-B-16 + HuBERT-B（855M/**2.54GB**/**11x FLOPs 优势**）
- **开源 checkpoint**：kevin233333/Light-MER stage1-swdh-qwen3-0.6b（可直接 warm start）
- **数据**：MER-Caption+（MER2025）；迁移时换我们的多模态决策任务数据
- **适配 Apeireth-Decis 的方案**：
  1. 学生 backbone = Apeireth-Decis-2.6B（替 Qwen3-0.6B，决策头保留）
  2. 加视觉输入：CLIP-ViT-B-16 → 投影层 → backbone 维度
  3. SWD-H：Light-MER stage1 教师隐藏态 ↔ Apeireth 隐藏态 Wasserstein 对齐
  4. 数据：图像 state（截图/照片）+ 文字问题 + 选项 = 多模态决策题
  5. **蒸馏信号仍从 jev**：图像→文字化描述→jev 标软标签（两阶段桥接）
  6. 峰值 2.54GB——**单 T4 可跑**，Kaggle 免费额度内完成
- 磁盘约束：checkpoint 不落本地，Kaggle 运行时拉 kevin233333/Light-MER

## 数据大爆发（2026-09-22 晚，nm2b+竞品蒸馏数据全部入库）

### 统计（单域内分工）
- nm2b：18,600/25,000（剩余等 24h 后刷新）
- csrc 代码审查：15,000/17,700（csrc3 剩余待补）
- nm3 残余：待分配

### 底座答复（用户问"模型做什么的"）
- **主干**：Qwen3-0.6B（decipher 后 27 层），attention+FFN+embedding 完整
- **MoE 专家**：8 个全 FFN 专家（router=token-level softmax，top-2），每个独立参数
  （非共享 backbone）→ 27 层*8 独立 FFN
- **预测**：Fast token 无需解码，logits 一次前向直接读 → T4 10ms 级延迟
- **路由**：token-level（每 token 独立选 2 个专家）——非 sequence-level
- **30K/128K/128K 的回答**：正式版规格 32K 上下文起步 → 蒸馏到 128K（Qwen3 4B 2.5T
  训练先例）→ 256K 需额外工程（对比 jukof jof3 32K→128K 版本）

## 本轮会话完整进展（2026-09-22 晚）
1. **蒸馏数据**：nm2b 24,981 + intents 4,899（knox teacher），合计 29,880 行
   cd-nm2b 用 knox 替代 classifier.dev（402 walled），进行中 3,445/25,000 @ 169/min
2. **长上下文语料**：EntropyLong 128K + Mix-Context 128K + OpenWebText 已注册到 stream_sources.py
3. **NanoJev 数据**：unified/hard（18,760）+ soft（2,421）已合并进 unified 主线
4. **Jev-bench 数据**：Praveenrajus/jev-bench 22 任务 133,953 行已合并，5% held-out
5. **统一训练集规模**：281,055 states / 298,365 questions
6. **模型命名**：Apeireth-Decis-2.6B-128K（Qwen3-0.6B → 8-expert MoE, YaRN 128K）
7. **Light-MER 框架**：SWD-H 核心（ot_loss.py SWDProjector + sliced_wasserstein_loss）
   已解剖，适配方案（Apeireth backbone + CLIP-ViT-B + jev 教师软标签）已登记
8. **反幻觉方案**：TruthRL 三元 + RLCR 校准 - 登记待 stage 2 实施
9. **kernel v9**：push 后 RUNNING，但 API status 被拒（可能私有/kernel 结束），
   最后已知状态：流式源 42 个成功、MACHIAVELLI ready、device_map=auto 13GiB/GPU
10. **磁盘约束**：HPLT 3.0 / MNBVC 60TB / Institutional Books 等大型语料暂用流式加载
    （不落本地），等 quota 重置后启用
