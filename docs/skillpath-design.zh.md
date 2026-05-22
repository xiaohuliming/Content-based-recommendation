# SkillPath — 项目设计文档

**日期：** 2026-05-22
**课程：** 大数据分析（Big Data Analytics），北师港浸大（UIC）
**执行模式：** 个人独立完成
**状态：** 初稿 v1 — 待 review

> 本文档为 [`skillpath-design.md`](skillpath-design.md) 的中文版，便于快速阅读与思考。**英文版是项目对外交付的 canonical 文档**（演讲、报告、代码注释皆用英文，与 UIC 英文授课环境一致）。

---

## 1. 项目定位

**标题：** SkillPath：基于大数据分析的职业目标驱动型个性化课程推荐系统

**一句话定位：** 根据学生的职业目标与当前技能画像，推荐 UIC 课程，并给出可解释的"技能缺口"分析。

**与"普通课程推荐"的本质区别：**

- **Skill-Gap Driven（技能缺口驱动）** —— 系统建模"职业需要什么技能"、"课程教授什么技能"、"学生掌握什么技能"，推荐能补齐缺口的课程。不是"和你相似的人也选了……"
- **异构图 + Personalized PageRank** 是算法主轴，直接对应 Lecture 3
- **可解释** —— 每条推荐都附带它能帮你解锁哪些与职业相关的技能、以及在图上的推理链路

**演讲与 demo 受众：** UIC 学生与教职。所有交付物（slides、demo UI、代码注释）均为英文，匹配 UIC 英文授课环境。

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  数据层（Data Layer）                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ LinkedIn     │  │ UIC 课程     │  │ 合成学生         │   │
│  │ 12.4 万岗位  │  │ (Excel+PDF)  │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  处理层（Processing Layer）                                    │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │ 技能抽取        │ ─▶ │ 统一技能词典                  │    │
│  │ (LLM 辅助)      │    │ (~500 个技能，6 大类)         │    │
│  └─────────────────┘    └──────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  算法层（Algorithm Layer）                                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 异构图                                                  │  │
│  │   Career   ──needs──▶  Skill   （职业需要技能）         │  │
│  │   Course   ──teaches─▶ Skill   （课程教授技能）         │  │
│  │   Course   ──prereq──▶ Course  （课程的先修课）         │  │
│  │   Student  ──has────▶  Skill   （学生掌握技能）         │  │
│  │   Student  ──took───▶  Course  （学生已修课程）         │  │
│  │                                                          │  │
│  │ Personalized PageRank                                   │  │
│  │   teleport set = {目标职业} ∪ {缺失技能}                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────┐
│  展示层（Presentation Layer）                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Demo UI（扩展现有 prototype/SkillPath-Demo.html）       │  │
│  │  • 输入：目标职业 + 已修课程 + 技能自评                  │  │
│  │  • 输出：带逐课理由的排序课程列表                        │  │
│  │  • 可视化：技能缺口图 + PageRank 解释路径                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 数据来源

### 3.1 岗位数据 —— LinkedIn 2023–2024 招聘数据集

- **位置：** `data/archive.zip`（Kaggle 公开数据集）
- **规模：** 12.4 万真实岗位，含完整英文 JD 文本
- **使用字段：** `job_id, title, description, normalized_salary, location, formatted_experience_level, listed_time, skills_desc`
- **过滤策略：** 不过滤 —— 全量保留 12.4 万条，故事是"国际化职业市场，跨所有职能"
- **注意点：** LinkedIn 自带的 `skills` 字段只有 ~30 个高层"职业大类"（FIN、MRKT、ENG…），**不是技术技能**。技术技能需要我们自己从 `description` 文本中抽取。

### 3.2 课程数据 —— UIC（双源 join）

**课程描述 PDF** — `data/Course Descriptions_20260421.pdf`

- 172 页，约 1300 门课程（UIC 完整课程目录）
- 单门字段：课程代码、课程名、学分、先修课、完整描述文本
- 格式高度规整，可用正则解析

**课程时间表 Excel** — `data/Course List and Timetable_Semester 2 of AY2025-26_20260224.xls`

- 2443 行 session 记录，约 600-800 门本学期实际开设课程
- 字段：代码、标题、专业、学分、上课时间、教室、教师、要求

**Join 策略：** PDF 是 canonical 课程目录；Excel 在本学期开设的课程上补充时间表与教师信息。Join key = 课程代码。

**Master 课程记录 schema：**

```
course_id, code, name, units, programme, department,
description,                  # 来自 PDF
prerequisites_text,           # PDF 原始字符串
prerequisite_ids[],           # 解析出的课程代码
schedule, classroom, teacher, # 本学期开设课程，来自 Excel
is_offered_current_sem (bool),
extracted_skills[]            # 来自技能抽取 pipeline
```

### 3.3 学生数据 —— 合成

- **种子：** 1-2 个手工编写的真实画像（自己 + 想象中的同学）
- **生成：** 用 LLM（Claude / GPT-4）基于 6-8 个 persona 合成 500-1000 个学生：Data Analyst / ML Engineer / Backend / NLP / Quant / Product / Design / Business Analytics
- **Schema：**

```
student_id, major, year, target_career,
completed_courses[],
skill_ratings { skill_id : 1-5 },
interest_prefs { theoretical|applied, research|career, ... },
max_credits, constraints[]
```

### 3.4 技能词典 —— 关键的横切骨架

这是整个系统的胶水。技能词典不干净，图就建不起来。

**构建流程：**

1. 用 LLM 从 LinkedIn `description` 字段抽取候选技术技能（采样 5K 条代表性岗位以控制成本）
2. 用 LLM 从 PDF 课程描述抽取候选技术技能
3. 合并、小写归一、去重
4. 同义词合并（Python / PYTHON / py → "Python"）；人工 review top ~200
5. 归入 6 大类：编程语言 / ML & AI / 数据系统 / 数学统计 / 领域知识 / 软技能

**目标规模：** 500-800 个唯一技能。

**Schema：**

```
skill_id, name, category, aliases[], parent_skill (可选)
```

---

## 4. 图边的派生（来自数据）

| 边 | 数据源 | 计算方式 |
|---|---|---|
| career → skill（带权） | LinkedIn | 按归一化职业标题聚类岗位 → 在 JD 文本中统计技能频率 → 归一化为权重 |
| course → skill（带权） | PDF 课程描述 | LLM 抽取 + TF-IDF 校验；权重 = 置信度 |
| course → course（先修） | PDF 先修课字段 | 正则解析 "Pre-requisite(s): X" |
| student → skill（掌握度） | 自评 + 已修课程推断 | rating ∈ {0..5} |
| student → course（已修） | 学生画像 | 直接 |

---

## 5. 核心算法：Personalized PageRank

**为什么 PageRank 是合适的主轴：**

- 直接对应 Lecture 3 —— "PageRank、teleport、个性化"三件套就是本课程的招牌内容
- 天然处理异构节点（career / skill / course / student 都活在同一张图里）
- teleport 机制是"按学生职业目标做个性化"的完美旋钮
- 可解释 —— 课程的高 PR 分可以通过 skill 节点反向追溯到 teleport set

**公式：**

- `r = β · M·r + (1−β) · p`
- `M` = 异构图邻接矩阵的列归一化
- `p` = teleport 分布，质量集中在 `{目标职业节点} ∪ {学生缺失技能节点}`
- `β = 0.85`（标准值）
- 迭代约 50 次至收敛

**输出：** 按 PageRank 分数排序课程节点。再应用约束过滤：

- 去掉已修课程
- 去掉先修条件未满足的课程
- 去掉与已选课程时间冲突的课程（用 Excel 时间表）
- 限制在 `max_credits` 学分预算内

**解释生成：**

对每门推荐课程，在图上反向追溯，找出从该课程到 teleport set 的 top-3 路径。渲染为自然语言：

> "推荐理由：
> 1. 教授 **Python** —— Data Analyst 职业对该技能权重 0.42
> 2. 桥接到 **Machine Learning**，你目标职业 top-10 技能中有 4 个依赖它
> 3. 是 AI3013（Machine Learning）的先修课，而你标记了 ML 为兴趣方向"

---

## 6. 其他 lecture 知识点的覆盖策略

课程共 8 个 lecture。一个人执行下，全部认真做会变成"每个都浅"。策略：**深做 2 个，其余在演讲里框定为架构延展**。

| Lecture | 覆盖策略 | 是否实做 |
|---|---|---|
| L1 KDD（知识发现） | 项目整体框架 —— "valid / useful / unexpected / understandable" 贯穿叙事 | 融入叙事 |
| L2 MapReduce | 12.4 万岗位上的技能频率聚合就是天然用例。演讲："本 demo 单机内存运行 12.4 万规模可承受；同样的 map/reduce 逻辑可扩展到 Spark/Hadoop 上的亿级数据" | 仅框定 —— 展示 map/reduce 伪代码，单机跑 |
| **L3 PageRank** | **主轴。实做。** | ✅ 实做 |
| L4 Online Matching / AdWords | 一张 slide：课程名额分配 = online bipartite matching，提 BALANCE 算法 | 仅框定 |
| L5 MinHash / LSH | 课程描述相似度 → "相关课程"功能 | Optional 加分项（1-2 天） |
| L6 Stream Data | 用 LinkedIn `listed_time` 模拟流；滑动窗口统计技能趋势 | Optional 加分项（2 天） |
| **L7 Clustering** | K-means 学生技能向量 → 学生原型聚类 | ✅ 实做（便宜，~1 天） |
| **L8 Recommendation** | 整个系统就是推荐系统。Content-based + Graph-based 混合 | ✅ 实做 |

**净覆盖：** L3、L7、L8 完全实做；L1 贯穿；L2、L4、L5、L6 框定。最终演讲对"实做 vs 框定"应**诚实表述**。

---

## 7. Demo UI

- 在现有 `prototype/SkillPath-Demo.html` 基础上扩展
- **输入表单：** 目标职业下拉、已修课程多选、技能自评滑块
- **输出：**
  - Top 10 推荐课程及其 PageRank 分数
  - 技能缺口雷达图（目标职业技能分布 vs 当前掌握）
  - 每门课：3 条推理 bullet 解释
  - "为什么不推荐它们？"面板 —— 展示 3 门高 PR 但被过滤掉的课程，并说明原因（先修未满足 / 时间冲突 / 与目标弱相关）
- **实现要点：** 保持 UI 为静态 HTML + JS；后端计算**离线预计算**为 JSON 加载。Demo 不需要实时后端。

---

## 8. 评估策略

**没有 ground truth → 不假装做 Precision@K。** 用以下替代：

1. **技能覆盖率提升 Δ（可计算的硬指标）：**
   - 推荐前：学生当前 rating ≥ 3 的技能在目标职业 top-20 技能中的占比
   - 模拟修完 top-5 推荐课程后：投影后的占比
   - 在 500-1000 合成学生上取平均，报告 Δ
2. **Baseline 对比：** 按热门度排序课程（按选课人数 / 或均匀 teleport 的 PageRank） vs Personalized PageRank。比较两者的技能覆盖率 Δ。
3. **定性叙事：** 演讲时走 2-3 个手工精雕的学生 persona，叙述为什么推荐合理
4. **可解释性检查：** 每条推荐都有理由。统计"无法生成解释"的失败案例（应为 0）

**砍掉：** 用户满意度问卷 —— 一个人没时间做、受众规模太小、形式大于实质。

---

## 9. 范围：In 和 Out

**In（实做并演示）：**
- 完整数据 pipeline（PDF 解析、Excel 解析、LinkedIn 加载、技能抽取、建图）
- Personalized PageRank + 约束过滤 + 解释生成
- K-means 学生聚类（第二个算法技术）
- Demo UI，演讲时走 2-3 个学生 persona
- 技能覆盖率 Δ 评估 + baseline 对比

**Out（演讲中作为"架构延展"提及，不实做）：**
- 真正的分布式 MapReduce 执行（单机版本顶替）
- 课程名额分配的 Online Matching 算法
- 真实用户研究
- Collaborative Filtering（没有真实历史选课数据）

**Optional 加分项**（Phase 1-5 提前完成时可加）：
- LSH 课程相似度（~1-2 天）
- DGIM-style 滑动窗口流处理（~2 天）

**诚实原则：** 演讲讨论未实做模块时，明确标注为"架构延展点"而非"已实现"。这比假装更打动人 —— 它表明你**理解这些技术的适用范围**。

---

## 10. 实施阶段

时间宽松。自然执行顺序：

| Phase | 交付物 | 估时（一人天） |
|---|---|---|
| 1 | 数据基础：PDF/Excel/LinkedIn 解析器、技能抽取、技能词典 | 3-5 |
| 2 | 异构图 + Personalized PageRank + 约束过滤 + 解释生成器 | 3-5 |
| 3 | 合成学生、批量推荐计算、persona 精雕 | 2-3 |
| 4 | Demo UI 扩展 + 预计算 JSON + 端到端打通 3 个 persona | 2-3 |
| 5 | 评估：技能覆盖率 Δ + baseline 对比 + K-means 聚类加分项 | 2-3 |
| 6 | （Optional）LSH 和/或 Stream 模块 | 2-4 |
| 7 | Slides + 排练 + 屏幕录制兜底 | 2-3 |

**核心（Phase 1-5、7）：** 一人 14-22 天，可持续节奏下完成。

---

## 11. 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 技能抽取产出噪声 / 重复技能 | 高 | 预留显式 cleanup 时间；人工 review top-200；迭代 LLM prompt |
| PDF 解析对部分非规范格式的课程失败 | 中 | 抽检 50 门课程解析结果；失败时降级为"仅用标题"模式 |
| LinkedIn 12.4 万被非技术岗位主导，扭曲职业技能权重 | 中 | 权重派生时按职业标题聚类，每个职业的权重只用它**自己**子集的岗位计算，不用全局平均 |
| 合成学生在 demo 中显得人工痕迹太重 | 中 | demo 展示的 2-3 个 persona 手工精雕；大批量合成数据只用于聚合评估 |
| PageRank 不收敛 / 排名异常 | 低 | 标准算法；先在 10 节点 toy 图上验证；检查列随机矩阵 dead-end 修复 |
| 范围膨胀 —— 加 L4/L5/L6 模块导致核心都没做完 | 中 | 严格 gate：Phase 1-5 demo-ready 之前不动 optional 加分项 |

---

## 12. 仓库结构（建议）

```
小组项目/
├── data/
│   ├── archive.zip
│   ├── Course List and Timetable_Semester 2 of AY2025-26_20260224.xls
│   └── Course Descriptions_20260421.pdf
├── docs/
│   ├── skillpath-design.md           # 英文版（canonical）
│   └── skillpath-design.zh.md        # 本文档
├── prototype/
│   └── SkillPath-Demo.html           # 现有 UI 原型
├── src/
│   ├── parsers/                      # PDF + Excel + LinkedIn 加载器
│   ├── skills/                       # 技能抽取 + 词典
│   ├── graph/                        # 异构图 + PageRank
│   ├── recommend/                    # 推荐 + 约束 + 解释
│   ├── students/                     # 合成学生生成
│   └── eval/                         # 技能覆盖率指标 + baseline
├── notebooks/                        # 探索 + 出图
└── output/
    ├── courses_master.csv
    ├── jobs_processed.csv
    ├── skill_taxonomy.csv
    ├── students_synthetic.csv
    └── recommendations/              # 每个 persona 一个 JSON
```

---

## 13. 待决问题

无阻塞项。以下战术决策延后到 implementation plan 决定：

- 技能抽取用 Claude 还是 GPT-4，及 API 成本预算
- 图存储用 NetworkX（轻量、纯内存）还是 Neo4j（功能丰富、演示更出彩）
- 可视化用 D3 / Plotly / 静态 matplotlib 输出

这些会在 implementation plan 里基于"哪个对个人执行最快"来定。

---

## 下一步

本设计文档 review 通过后，转入 **implementation plan**：把每个 Phase 拆成具体的文件路径、函数签名、验证检查点 —— 那是真正可以一行行执行的执行清单。
