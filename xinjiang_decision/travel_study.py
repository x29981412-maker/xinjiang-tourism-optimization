"""Complete input -> network -> itinerary -> audit -> simulation evidence pipeline."""

from dataclasses import asdict
import json

from .data_pipeline import export_spots, load_inputs
from .itinerary import CalendarPolicy, solve_personal, solve_two_year
from .network import TransportNetwork, leg_rows
from .simulation import uncertainty_sweep, wilson_interval


def run_travel_study(data, output_dir, write_csv, repetitions=500, seed=20261104):
    spots, edges, audit = load_inputs(data)
    network, policy = TransportNetwork(edges), CalendarPolicy()
    matrices = [network.matrix([s.spot_id for s in spots], value) for value in (10, 60, 200)]
    personal, personal_matrix, choices = solve_personal(spots, matrices, policy)
    economy, _ = solve_two_year(spots, matrices[:2], policy)
    balanced, balanced_matrix = solve_two_year(spots, matrices[:2], policy, max_day_gap=7)
    simulations = [{"route_case": "personal", **s} for s in uncertainty_sweep(personal["route"], spots, personal_matrix, policy, repetitions, seed)]
    for year in balanced["years"]:
        simulations.extend({"route_case": f"year_{year['year']}", **s}
                           for s in uncertainty_sweep(year["route"], spots, balanced_matrix, policy, repetitions, seed + year["year"]))
    reference = next(s for s in simulations if s["route_case"] == "personal" and s["case"] == "reference")
    budget_curve = []
    for horizon in (24, 26, 28, 30):
        successes = sum(count for day, count in reference["completion_day_counts"].items() if int(day) <= horizon)
        low, high = wilson_interval(successes, repetitions)
        budget_curve.append({"available_days": horizon, "successes": successes, "repetitions": repetitions,
                             "completion_rate": successes / repetitions, "wilson_95_low": low, "wilson_95_high": high})
    assumptions = ["This is a newly implemented scenario model, not reproduction of the original Q1/Q2 full formulation.",
                   "Dijkstra minimizes fare + value_of_time * duration; cost and duration come from the same path.",
                   "Path-level mode switching is allowed; timetable connections and car custody/return fees are not modeled.",
                   "Travel can be split at abstract overnight stops; hotel availability and intermediate stop locations are not modeled.",
                   "Multiday visits may resume the following day. Reentry fees are not modeled.",
                   "Fares are for two people, exclude tickets, meals, accommodation and travel to/from Xinjiang.",
                   "Priority objectives: personal coverage then priority then transport fare; two-year fare subject to annual constraints.",
                   "Lognormal delay CVs are declared scenario assumptions, not estimates fitted from operational observations.",
                   "A common trip factor correlates travel delays. Missed departures, cancellations and fare shocks are not simulated.",
                   "Heuristic solutions are not claimed as globally optimal or as evidence that an unlocated solution is infeasible."]
    result = {"policy": asdict(policy), "data_audit": audit, "personal": personal, "personal_candidates": choices,
              "two_year_economy": economy, "two_year_balanced": balanced,
              "simulations": simulations, "budget_curve": budget_curve, "assumptions": assumptions}
    (output_dir / "travel_study.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "clean_spots.csv", export_spots(spots))
    write_csv(output_dir / "personal_candidates.csv", choices)
    write_csv(output_dir / "travel_legs.csv", leg_rows(personal["route"], personal_matrix, "personal")
              + [row for year in balanced["years"] for row in leg_rows(year["route"], balanced_matrix, f"year_{year['year']}")])
    write_csv(output_dir / "travel_calendar.csv", [{"case": case, **event} for case, solution in
              [("personal", personal)] + [(f"year_{year['year']}", year) for year in balanced["years"]] for event in solution["events"]])
    write_csv(output_dir / "travel_simulation.csv", [{k: v for k, v in s.items() if k != "completion_day_counts"} for s in simulations])
    write_csv(output_dir / "buffer_sensitivity.csv", budget_curve)
    return result


def travel_report(study):
    from .reporting import markdown_table
    personal, economy, balanced = study["personal"], study["two_year_economy"], study["two_year_balanced"]
    years = balanced["years"]
    return f'''## 个人旅行与两年分期旅行（新增情景模型）

从40个景点和1900条有向交通边出发，排除2个需特殊审批的点，形成38个普通游客候选点。
清洗过程校验主键、外键、有限非负数值和开放区间；保留拆分开放时段，逐条记录低置信度映射等数据问题。

- 个人旅行：启发式选择{len(personal['route'])}个点，排程{personal['days']}天，境内双人交通代理费用{personal['transport_cost_for_two']:.2f}元。
- 费用优先的两年方案：{economy['transport_cost_for_two']:.2f}元，年度排程{' / '.join(str(y['days']) for y in economy['years'])}天。
- 年度天数差不超过7天的方案：{balanced['transport_cost_for_two']:.2f}元，年度排程{years[0]['days']} / {years[1]['days']}天；约束代价为{balanced['transport_cost_for_two'] - economy['transport_cost_for_two']:.2f}元。
- 基础排程每天08:00–20:00内最多活动10小时，各年30天中至少预留2天；38个点在两年间无遗漏、无重复。
- 交通费用和时间来自同一条最短路径，不拼接两条不同路径各自的最小值。逐腿交通方式、原始边ID与逐日活动可在CSV中追溯。

### 延误仿真与缓冲价值

每条路线做轻度、基准和重度3组扰动，每组{study['simulations'][0]['repetitions']}次；固定随机种子，重排开放时段和每日活动，输出Wilson区间。
下表对同一条个人路线、同一组基准仿真样本改变可用天数，隔离观察预算变化的影响。

{markdown_table(study['budget_curve'], [('available_days','可用天数'),('completion_rate','情景内完成比例'),('wilson_95_low','95%下界'),('wilson_95_high','95%上界')])}

这些比例来自假定的对数正态扰动，不是实测完成概率。交通边仍含校准/模拟参数，班次衔接、车辆保有、酒店可用性与中途住宿位置未建模。
因此本节为新实现的情景决策模型，**不得与原报告的Q1/Q2数字直接比较，也不声称复现其完整模型或取得全局最优**。

独立审计逐项重算开放时段、每日活动、游览时长、交通、费用和覆盖。审计明细、候选比较及模型边界见travel_study.json。

'''
