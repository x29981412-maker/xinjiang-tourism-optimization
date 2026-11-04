"""One-command reproduction, sensitivity analysis, and evidence generation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy

from . import __version__
from .portfolio import load_routes, read_rows, policy_from_baseline, score_routes, solve_portfolio, sensitivity_analysis, demand_stress_test, audit_selection
from .reporting import portfolio_figure, schedule_figure, markdown_table
from .scheduling import solve_cultural_schedule
from .travel_study import run_travel_study, travel_report


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir, verify=False, simulations=500, seed=20261104):
    data = ROOT / "data/reference"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    travel = run_travel_study(data, output_dir, write_csv, simulations, seed)
    routes = load_routes(data / "route_products.csv")
    baseline_ids = [r["route_id"] for r in read_rows(data / "baseline_portfolio.csv")]
    policy = policy_from_baseline(routes, baseline_ids)
    all_scores = score_routes(routes)
    baseline_routes = [r for r in routes if r.route_id in set(baseline_ids)]
    baseline = {"score": float(sum(v for r, v in zip(routes, all_scores) if r.route_id in set(baseline_ids))),
                "capacity": sum(r.capacity for r in baseline_routes), "quality_pass_count": sum(r.quality_pass for r in baseline_routes)}
    if not all(audit_selection(baseline_routes, policy).values()):
        raise ValueError("Baseline fails its own policy constraints")
    optimized = solve_portfolio(routes, policy)
    gain = optimized["score"] / baseline["score"] - 1
    schedule = solve_cultural_schedule(data / "cultural_sites.csv", data / "cultural_travel.csv")
    checks = {}
    expected_p = read_rows(data / "portfolio_expected.csv")[0]
    expected_s = read_rows(data / "cultural_expected.csv")[0]
    checks["portfolio_score_matches_reference"] = math.isclose(optimized["score"], float(expected_p["milp_exact_score"]), abs_tol=.00001)
    checks["portfolio_capacity_matches_reference"] = optimized["capacity"] == float(expected_p["milp_capacity"])
    checks["portfolio_optimal"] = optimized["proved_optimal"] and optimized["mip_gap"] < 1e-8
    checks["schedule_makespan_matches_reference"] = math.isclose(schedule["max_completion_hours"], float(expected_s["exact_max_completion_hours"]), abs_tol=.00001)
    checks["schedule_imbalance_matches_reference"] = math.isclose(schedule["balance_gap_hours"], float(expected_s["exact_balance_gap_hours"]), abs_tol=.00001)
    checks["personal_calendar_audited"] = all(travel["personal"]["checks"].values())
    checks["two_year_economy_audited"] = all(travel["two_year_economy"]["checks"].values())
    checks["two_year_balanced_audited"] = all(travel["two_year_balanced"]["checks"].values())
    if verify and not all(checks.values()):
        raise AssertionError(f"Reference verification failed: {checks}")
    sensitivity = sensitivity_analysis(routes, policy, optimized["selected_ids"])
    stress = demand_stress_test(routes, policy)
    evidence = {"package_version": __version__, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.csv"))},
                "baseline": baseline, "portfolio": optimized, "relative_score_gain": gain, "scheduling": schedule, "reference_checks": checks,
                "sensitivity": sensitivity, "demand_stress": stress,
                "travel_study": {"artifact": "travel_study.json", "personal_spots": len(travel["personal"]["route"]),
                                 "personal_days": travel["personal"]["days"], "two_year_cost": travel["two_year_balanced"]["transport_cost_for_two"]},
                "model_scope": "Q1/Q2 newly implemented multimodal itinerary scenario models (heuristic). Q3 fixed-policy 14-site scheduling and Q4 fixed 63-candidate portfolio (exact). Original full formulations remain in the archived report.",
                "scenario_note": "Sensitivity/stress cases are deterministic modeling assumptions; they are not observed visitor demand or estimated real-world probabilities."}
    (output_dir / "reproduction.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    chosen = set(optimized["selected_ids"])
    write_csv(output_dir / "selected_portfolio.csv", [{"route_id": r.route_id, "route_name": r.name, "product_type": r.product_type, "capacity": r.capacity, "quality_pass": r.quality_pass, "model_score": float(s)} for r, s in zip(routes, all_scores) if r.route_id in chosen])
    write_csv(output_dir / "sensitivity.csv", sensitivity)
    write_csv(output_dir / "demand_stress.csv", stress)
    write_csv(output_dir / "team_schedule.csv", [{"group": g["group"], "sites": ";".join(g["site_names"]), "travel_hours": g["travel_hours"], "service_hours": g["service_hours"], "completion_hours": g["completion_hours"]} for g in schedule["groups"]])
    (output_dir / "portfolio.svg").write_text(portfolio_figure(baseline, optimized, gain), encoding="utf-8")
    (output_dir / "schedule.svg").write_text(schedule_figure(schedule), encoding="utf-8")
    report = f'''# 求解复算与决策分析

{travel_report(travel)}

## 固定候选空间中的线路组合

63条候选线路中选18条，保持产品类型配额、原有区域覆盖、基准容量下界106358人及质量代理通过数下界。

- 基准综合分：{baseline['score']:.6f}；优化综合分：{optimized['score']:.6f}；相对提升：{gain:.6%}。
- 模型容量：{baseline['capacity']:.0f} → {optimized['capacity']:.0f}人；求解状态：{optimized['status']}；MILP gap：{optimized['mip_gap']}。
- 选择后独立重算唯一性、数量、产品配额、区域、容量和质量六类约束，全部通过。

## 固定特殊准入政策中的三团队调度

14个点分配给3组；特殊准入点固定到第1组，每组至少2个点，各组从乌鲁木齐往返。

- 最晚完成时间：{schedule['max_completion_hours']:.3f}小时；组间差距：{schedule['balance_gap_hours']:.3f}小时。
- 子集动态规划计算各子集最短往返路径；消除普通两组交换对称后，完整检查{schedule['checked_canonical_partitions']:,}个可行分配。
- 二级目标为组间差距、总完成时间；不要求与参考工程的风险/费用二级目标得到同一条路线。

## 权重敏感性

{markdown_table(sensitivity, [('case','偏好情景'),('capacity','模型容量'),('quality_pass_count','质量代理通过数'),('overlap_with_reference','与基准优化组合重合数'),('status','状态')])}

各情景使用同一候选数据和政策约束；评分随权重变化，不跨权重直接比较综合分大小。

## 需求压力测试

{markdown_table(stress, [('demand_multiplier','需求倍率'),('required_capacity','容量下界'),('selected_capacity','所选模型容量'),('status','状态')])}

不可行表示在当前固定产品配额、质量与候选集合内无法满足需求，应讨论扩充候选、运力或政策调整。

## 核验与解释边界

本次运行的参考核验全部通过：{all(checks.values())}。输入文件哈希、软件版本与各项检查详见reproduction.json。
分数与容量为模型指标；方案总容量充足并不保证每个分时资源均不超载。参考数据保留模拟/校准参数，不能直接作为实时排班。
'''
    (output_dir / "reproduction_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"portfolio_score": optimized["score"], "score_gain_pct": gain * 100, "capacity": optimized["capacity"], "schedule_makespan_hours": schedule["max_completion_hours"], "reference_checks": checks, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generated")
    parser.add_argument("--verify", action="store_true", help="Fail if the two benchmark optima differ from the reference values")
    parser.add_argument("--simulations", type=int, default=500, help="Simulation draws per route and uncertainty scenario")
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    run(args.output_dir, args.verify, args.simulations, args.seed)


if __name__ == "__main__":
    main()
