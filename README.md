# 新疆旅游线路多场景优化

[![Verify optimization results](https://github.com/x29981412-maker/xinjiang-tourism-optimization/actions/workflows/verify.yml/badge.svg)](https://github.com/x29981412-maker/xinjiang-tourism-optimization/actions/workflows/verify.yml)

把旅行规划拆成可检验的决策问题：面对覆盖、费用、工期和接待资源的冲突，如何选方案，如何证明方案满足约束，又如何解释它在哪些条件下失效。

**[研究报告](paper/main.pdf)** · **[可复算结果](results/generated/reproduction_report.md)** · **[模型设计](docs/model_design.md)** · **[个人贡献](docs/contributions.md)**

## 我参与的工作

在团队项目中参与目标与约束设计、Python求解实现、情景仿真和结果验证，负责数据整理与分析。本仓库近期进一步补齐了模块化求解、独立约束核验、权重敏感性与需求压力测试，使用AI辅助编程推进实现，并用可复算检查验证结果。

团队研究开展于2026年4月至5月；当前求解工具与展示文档于2026年11月补强。具体分工及本轮实现范围见[贡献说明](docs/contributions.md)。

## 已复算的核心结果

![线路组合优化](results/generated/portfolio.svg)

| 决策问题 | 本仓库实现 | 本次复算结果 |
| --- | --- | --- |
| 63条候选线路中选择18条 | 0-1 MILP，保持产品配额、区域覆盖、容量与质量下界 | 综合分921.989204 → 993.392727，约提升7.745%；模型容量160190 → 187550人 |
| 14个文化点分配给3组 | Held–Karp子集路径动态规划 + 固定政策下的精确分组枚举 | 最晚完成99.187小时，组间差距9.994小时；对称去重后检查237127种分配 |

![三团队调度](results/generated/schedule.svg)

这两项是实际运行结果，详细数值、输入哈希、软件版本和核验记录保存在[reproduction.json](results/generated/reproduction.json)。最优性限定于固定候选集合或特殊准入政策。

## 运行

Python 3.11及以上，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m xinjiang_decision --verify
python -m unittest discover -s tests -v
```

无需地图密钥或外部服务。求解使用随SciPy提供的HiGHS。运行后生成：

- 18条入选线路及其分数、容量和质量指标；
- 三团队访问顺序、交通时间和现场作业时间；
- 5组偏好权重敏感性分析；
- 6组需求压力测试；
- SVG结果图、中文分析报告和带输入哈希的JSON证据。

可用 `--output-dir` 指定输出目录。CI在独立环境运行9项测试，并复算两项基准最优值。

## 从结果到判断

**权重变化会改变选择。** 容量优先情景选择的组合容量为189980人；舒适度优先时为180540人、质量代理通过数16条。各情景的评分权重不同，应比较方案构成和结果取舍，而非直接比较跨权重的综合分。

**需求上升会触及政策边界。** 在固定候选、配额和质量约束下，需求增至基准的2倍时无可行组合。此时需要讨论扩充候选、增加运力或调整政策，不能把不可行结果包装成成功方案。

**总量可接待不代表分时均可执行。** 模型容量不是实时库存，仍需结合景区分时利用率、住宿与车辆等局部瓶颈解释落地条件。

[查看完整敏感性与压力测试](results/generated/reproduction_report.md)

## 四类研究场景与当前复算范围

| 场景 | 团队研究问题 | 当前仓库状态 |
| --- | --- | --- |
| 30天个人旅行 | 覆盖、费用、缓冲日与扰动风险 | 保留团队报告与结果摘要 |
| 两年分期旅行 | 费用、年度负担与交通衔接 | 保留团队报告与结果摘要 |
| 三组文化考察 | 特殊准入、组内路径与最晚完成时间 | 已提供可运行的子集路径和精确分组 |
| 节假日接待 | 产品配额、区域、容量与质量约束 | 已提供可运行的MILP与敏感性、压力测试 |

输入是课程研究中的情景数据，部分参数经过模拟或校准。[数据来源与字段解释](data/SOURCES.md)保留了参考材料、版本和本地哈希。源代码在本仓库重新组织实现；参考工程的程序未整包迁入。

## 代码结构

```text
xinjiang_decision/
  portfolio.py       MILP、独立约束审计、敏感性与压力测试
  scheduling.py      子集路径DP、精确分组、路线还原
  reporting.py       SVG图表与报告表格
  __main__.py        一键复算、参考核验与证据导出
data/reference/      固定版本的数值样例与参考结果
tests/              穷举对照、异常输入及约束测试
results/generated/  当前运行结果、图表和研究报告
docs/               数学模型、分工与实验解释
paper/              原团队研究报告及排版源码
```
