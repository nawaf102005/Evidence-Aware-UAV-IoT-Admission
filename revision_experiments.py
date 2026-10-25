"""Execute reviewer-requested experiments without changing legacy results.

python revision_experiments.py --instances 200 --replications 128
Development runs must use --output tmp/revision_dev (never the manuscript path).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import binomtest, norm, t as student_t

from secure_drop_research import (Config, RadioTable, generate_instance, feasible_routes,
                                  make_plan, save_csv, interval)
from wireless_extensions import (ExtendedRadio, extended_plan, adaptive_plan, evaluate,
                                 gaussian_envelope_cap, generate_umi_instance, local_eopt_cut,
                                 packet_payload_budget)


METHODS = ("SDW-power", "SPEB-joint", "Robust-shortest", "ERGO-joint",
           "GRE-joint", "Adaptive-E", "Adaptive-GRE")
LABELS = {"SDW-power":"SDW power", "SPEB-joint":"Robust SPEB", "Robust-shortest":"Robust shortest",
          "ERGO-joint":"ERGO", "GRE-joint":"Gaussian reserve", "Adaptive-E":"Adaptive E",
          "Adaptive-GRE":"Adaptive Gaussian"}


def research_config(**changes):
    return replace(Config(pilot_control_s=40e-6), **changes)


def table_for(instance, cfg, codebook):
    tables = [RadioTable(instance, n, cfg, codebook) for n in range(cfg.stops+2)]
    for table in tables:
        table.capacity_bits=table.bits.copy()
        table.bits=packet_payload_budget(table)
    return tables, [ExtendedRadio(table) for table in tables]


def ci99(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float(values.mean()) if len(values) else np.nan, np.nan)
    return float(values.mean()), float(student_t.ppf(.995, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))


def wilson99(successes, total):
    z = float(norm.ppf(.995)); p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return center-half, center+half


def summarize(rows, methods=METHODS):
    result = []
    metrics = ("violation", "pilot_fraction", "goodput_bits", "nees", "radial_union_bound",
               "eigen_union_bound", "budget_exceed_mission", "budget_exceed_epoch",
               "mean_losses", "unresolved", "flight_kj", "objective", "planning_ms")
    for method in methods:
        all_rows = [r for r in rows if r["method"] == method]
        accepted = [r for r in all_rows if r["admitted"]]
        if not all_rows:
            continue
        entry = dict(method=method, instances=len(all_rows), admitted=len(accepted))
        entry["admission_low99"], entry["admission_high99"] = wilson99(len(accepted), len(all_rows))
        for key in metrics:
            mean, half = interval([r[key] for r in accepted])
            entry[key], entry[key+"_ci95"] = mean, half
        success = [1-r["violation"] if r["admitted"] else 0. for r in all_rows]
        entry["unconditional_success"], entry["success_ci95"] = interval(success)
        result.append(entry)
    return result


def record(seed, method, plan, metrics, planning_ms):
    row = dict(seed=seed, method=method, admitted=int(plan is not None))
    keys = ("violation", "pilot_fraction", "goodput_bits", "nees", "radial_union_bound",
            "eigen_union_bound", "budget_exceed_mission", "budget_exceed_epoch", "mean_losses", "unresolved")
    row.update(metrics if plan is not None else {k:float("nan") for k in keys})
    row.update(flight_kj=plan.route.flight_j/1000 if plan else np.nan,
               objective=plan.objective if plan else np.nan, planning_ms=planning_ms)
    return row


def paired(rows, first, second):
    a = {r["seed"]:r for r in rows if r["method"] == first}
    b = {r["seed"]:r for r in rows if r["method"] == second}
    common = [seed for seed in a if a[seed]["admitted"] and b[seed]["admitted"]]
    gains = sum(a[s]["admitted"] and not b[s]["admitted"] for s in a)
    losses = sum(b[s]["admitted"] and not a[s]["admitted"] for s in a)
    row = dict(first=first, second=second, common=len(common), gains=gains, losses=losses,
               mcnemar_p=binomtest(gains, gains+losses, .5).pvalue if gains+losses else 1.)
    for key in ("violation", "pilot_fraction", "goodput_bits", "flight_kj", "objective"):
        values = [a[s][key]-b[s][key] for s in common]
        row[key+"_difference"], row[key+"_ci95"] = interval(values)
        row[key+"_ci99"] = ci99(values)[1]
        row[first+"_"+key] = interval([a[s][key] for s in common])[0]
        row[second+"_"+key] = interval([b[s][key] for s in common])[0]
    return row


def baseline_and_conservatism(args, cfg, output):
    codebook = np.array(list(itertools.product(range(len(cfg.powers_w)), repeat=cfg.anchors)), dtype=int)
    rows, packet_rows, csi_rows, status_rows, phy_rows = [], [], [], [], []
    begin = time.perf_counter()
    for trial in range(args.instances):
        instance = generate_instance(args.seed+trial, cfg)
        tables, extra = table_for(instance, cfg, codebook)
        routes = feasible_routes(instance, cfg)
        plans = {}
        for method in METHODS:
            clock = time.perf_counter()
            if method.startswith("Adaptive"):
                base = plans["GRE-joint" if method == "Adaptive-GRE" else "ERGO-joint"]
                plan = adaptive_plan(base, method, extra, tables, cfg)
            else:
                plan = extended_plan(method, routes, tables, extra, cfg)
            plans[method] = plan
            planning_ms = 1000*(time.perf_counter()-clock)
            metrics = evaluate(instance, plan, tables, extra, cfg, args.replications) if plan else None
            rows.append(record(instance.seed, method, plan, metrics, planning_ms))
        # Split 0.10 total risk equally between centered localization error
        # and exceeding the PHY-induced erasure budget. Same catalog/powers.
        phy_cfg = replace(cfg, mission_risk=.05)
        phy_tables, phy_extra = table_for(instance, phy_cfg, codebook)
        clock=time.perf_counter()
        phy_plan=extended_plan("PHY-GRE",routes,phy_tables,phy_extra,phy_cfg)
        phy_ms=1000*(time.perf_counter()-clock)
        for mode in ("independent","correlated"):
            metrics=evaluate(instance,phy_plan,phy_tables,phy_extra,phy_cfg,args.replications,
                             loss_mode=mode) if phy_plan else None
            row=record(instance.seed,"PHY-GRE",phy_plan,metrics,phy_ms)
            row["loss_mode"]=mode;phy_rows.append(row)
        for node, ext in enumerate(extra):
            status_rows.append(dict(seed=instance.seed, node=node, status=ext.sdw_status,
                                    continuous_power_w=float(ext.sdw_power.sum()) if ext.sdw_power is not None else np.nan))
        # Loss-process comparison on identical plans. No adversarial losses
        # are added to the stochastic tests; samples use fresh packet uniforms.
        for loss_mode in ("independent", "correlated"):
            for method in ("SPEB-joint", "ERGO-joint", "GRE-joint"):
                plan = plans[method]
                metrics = evaluate(instance, plan, tables, extra, cfg, args.replications,
                                   loss_mode=loss_mode) if plan else None
                row = record(instance.seed, method, plan, metrics, 0.)
                row["loss_mode"] = loss_mode; packet_rows.append(row)
        # A declared bounded map error is distinguished from unbounded
        # Gaussian mismatch. A 3-dB design margin is deliberately tested.
        for margin in (0., 3.):
            design_instance = replace(instance, gains=instance.gains*10**(-margin/10))
            design_tables, design_extra = table_for(design_instance, cfg, codebook)
            plan = make_plan("ERGO-joint", routes, design_tables, cfg)
            metrics = evaluate(design_instance, plan, design_tables, design_extra, cfg,
                               args.replications, csi_error_db=2., csi_margin_db=margin) if plan else None
            row = record(instance.seed, "ERGO-joint", plan, metrics, 0.)
            row["margin_db"] = margin; csi_rows.append(row)
        if (trial+1) % 10 == 0 or trial+1 == args.instances:
            print(f"Revision comparison: {trial+1}/{args.instances}; elapsed {time.perf_counter()-begin:.1f}s", flush=True)
    save_csv(output/"revision_trials.csv", rows)
    save_csv(output/"revision_summary.csv", summarize(rows))
    comparisons = [paired(rows, a, b) for a, b in (
        ("ERGO-joint", "SPEB-joint"), ("ERGO-joint", "SDW-power"),
        ("ERGO-joint", "Robust-shortest"), ("GRE-joint", "ERGO-joint"),
        ("Adaptive-E", "ERGO-joint"), ("Adaptive-GRE", "GRE-joint"))]
    # One predeclared admission-test family for these six comparisons.
    order = np.argsort([r["mcnemar_p"] for r in comparisons]); running=0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(comparisons)-rank)*comparisons[index]["mcnemar_p"]))
        comparisons[index]["holm_p"] = running
    # Different paired column names require a unified schema.
    fieldnames = list(dict.fromkeys(k for r in comparisons for k in r))
    save_csv(output/"revision_paired.csv", [{k:r.get(k, "") for k in fieldnames} for r in comparisons])
    save_csv(output/"packet_trials.csv", packet_rows)
    packet_summary = []
    for mode in ("independent", "correlated"):
        for row in summarize([r for r in packet_rows if r["loss_mode"] == mode],
                             ("SPEB-joint", "ERGO-joint", "GRE-joint")):
            row["loss_mode"] = mode; packet_summary.append(row)
    save_csv(output/"packet_summary.csv", packet_summary)
    save_csv(output/"phy_chance_trials.csv",phy_rows)
    phy_summary=[]
    for mode in ("independent","correlated"):
        for row in summarize([r for r in phy_rows if r["loss_mode"]==mode],("PHY-GRE",)):
            row["loss_mode"]=mode;phy_summary.append(row)
    save_csv(output/"phy_chance_summary.csv",phy_summary)
    save_csv(output/"csi_trials.csv", csi_rows)
    csi_summary=[]
    for margin in (0., 3.):
        row=summarize([r for r in csi_rows if r["margin_db"]==margin], ("ERGO-joint",))[0]
        row["margin_db"] = margin; csi_summary.append(row)
    save_csv(output/"csi_summary.csv", csi_summary)
    save_csv(output/"sdw_solver_status.csv", status_rows)


def standards_channel(args, output):
    cfg=research_config(altitude_m=10.)
    codebook=np.array(list(itertools.product(range(4), repeat=cfg.anchors)), dtype=int)
    rows=[]
    for trial in range(args.instances):
        instance=generate_umi_instance(550001+trial, cfg)
        tables, extra=table_for(instance, cfg, codebook)
        routes=feasible_routes(instance, cfg)
        for method in ("SDW-power", "SPEB-joint", "ERGO-joint", "GRE-joint"):
            clock=time.perf_counter()
            plan=extended_plan(method, routes, tables, extra, cfg)
            planning_ms=1000*(time.perf_counter()-clock)
            metrics=evaluate(instance, plan, tables, extra, cfg, args.replications) if plan else None
            rows.append(record(instance.seed, method, plan, metrics, planning_ms))
        if (trial+1)%25==0 or trial+1==args.instances:
            print(f"38.901 UMi large-scale subset: {trial+1}/{args.instances}", flush=True)
    save_csv(output/"umi_trials.csv", rows)
    save_csv(output/"umi_summary.csv", summarize(rows, ("SDW-power", "SPEB-joint", "ERGO-joint", "GRE-joint")))


def sensitivity(args, output):
    settings = [("floor_m", x, research_config(range_floor_m=x)) for x in (.15,.3,.6,.9)]
    settings += [("drift_m2_per_m", x, research_config(drift_m2_per_m=x)) for x in (.001,.002,.004)]
    settings += [("efficiency", x, research_config(ranging_efficiency=x)) for x in (.5,1.,2.)]
    settings += [("rician_k", x, research_config(rician_k=x)) for x in (0.,5.,10.)]
    settings += [("pathloss_exponent", x, research_config(pathloss_exponent=x)) for x in (2.2,2.6,3.)]
    settings += [("radio_weight", x, research_config(radio_weight=x)) for x in (.1,.5,1.)]
    settings += [("stops", x, research_config(stops=x)) for x in (5,10,20)]
    codebook=np.array(list(itertools.product(range(4),repeat=6)), dtype=int)
    rows=[]
    for name, value, cfg in settings:
        for j in range(args.sensitivity_instances):
            instance=generate_instance(660001+j,cfg)
            tables=[RadioTable(instance,n,cfg,codebook) for n in range(cfg.stops+2)]
            for table in tables: table.bits=packet_payload_budget(table)
            plan=make_plan("ERGO-joint",feasible_routes(instance,cfg),tables,cfg)
            rows.append(dict(seed=instance.seed, parameter=name, value=value,
                             admitted=int(plan is not None),
                             pilot_fraction=plan.mean_pilot_fraction if plan else np.nan,
                             flight_kj=plan.route.flight_j/1000 if plan else np.nan,
                             objective=plan.objective if plan else np.nan))
        print(f"Sensitivity: {name}={value} ({args.sensitivity_instances} layouts)",flush=True)
    save_csv(output/"wireless_sensitivity_trials.csv",rows)
    summary=[]
    for name, value, _ in settings:
        selected=[r for r in rows if r["parameter"]==name and r["value"]==value]
        accepted=[r for r in selected if r["admitted"]]
        row=dict(parameter=name,value=value,admitted=len(accepted),instances=len(selected))
        row["admission_low99"],row["admission_high99"]=wilson99(len(accepted),len(selected))
        for key in ("pilot_fraction","flight_kj","objective"):
            row[key],row[key+"_ci99"]=ci99([r[key] for r in accepted])
        summary.append(row)
    save_csv(output/"wireless_sensitivity_summary.csv",summary)


def optimizer_benchmark(args, output):
    rows=[]
    for M in (4,6,8,10):
        cfg=research_config(anchors=M)
        for j in range(args.solver_instances):
            instance=generate_instance(770001+j,cfg)
            length=float(np.linalg.norm(instance.nodes[1]-instance.nodes[0]))
            result=local_eopt_cut(instance,1,length,cfg,time_limit=args.solver_seconds)
            clock=time.perf_counter(); oracle=None
            if M<=8:
                codebook=np.array(list(itertools.product(range(4),repeat=M)),dtype=int)
                table=RadioTable(instance,1,cfg,codebook)
                table.bits=packet_payload_budget(table)
                index=table.select(cfg.covariance_cap+cfg.drift_m2_per_m*length,cfg.erasure_budget)
                oracle=float(table.cost[index]) if index is not None else np.inf
            result.update(anchors=M,seed=instance.seed,oracle_cost=oracle,
                          enumeration_ms=1000*(time.perf_counter()-clock) if M<=8 else np.nan)
            result["levels"]=json.dumps(result["levels"])
            if M<=8 and result["status"]=="optimal" and abs(result["upper_bound"]-oracle)>1e-6:
                raise RuntimeError("Independent E-opt solver disagrees with enumeration.")
            rows.append(result)
            print(f"E-opt MILP cuts: M={M}, seed={instance.seed}, {result['status']}, gap={result['gap']:.4g}",flush=True)
    save_csv(output/"eopt_solver_benchmark.csv",rows)


def read_csv(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try: row[key]=float(value)
            except (ValueError,TypeError): pass
    return rows


def reports(output):
    summary=read_csv(output/"revision_summary.csv")
    paired_rows=read_csv(output/"revision_paired.csv")
    umi=read_csv(output/"umi_summary.csv")
    packet=read_csv(output/"packet_summary.csv")
    packet+=read_csv(output/"phy_chance_summary.csv")
    sensitivity_rows=read_csv(output/"wireless_sensitivity_summary.csv")
    figures=output.parent/"figures"; figures.mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(1,3,figsize=(9.4,3.0),layout="constrained")
    selected=[r for r in summary if r["method"]!="Robust-shortest"]
    labels=[LABELS[r["method"]] for r in selected]
    colors=["#8a8a8a","#c08b48","#217c78","#387fae","#61a78d","#665397"]
    for ax,key,title,scale in zip(axes,("violation","pilot_fraction","goodput_bits"),
                                 ("Mission violation","Pilot airtime fraction","Decoded payload (kbit)"),(1,1,.001)):
        ax.bar(range(len(selected)),[r[key]*scale for r in selected],color=colors,
               yerr=[r[key+"_ci95"]*scale for r in selected],capsize=2)
        ax.set_xticks(range(len(selected)),labels,rotation=38,ha="right",fontsize=8)
        ax.set_ylabel(title,fontsize=10); ax.grid(axis="y",alpha=.2)
        if key=="violation": ax.axhline(.1,color="black",ls="--",lw=.8)
    fig.savefig(figures/"revision_comparison.pdf",bbox_inches="tight")
    fig.savefig(figures/"revision_comparison.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9.3,2.8),layout="constrained")
    for ax,param,label in zip(axes,("floor_m","drift_m2_per_m","pathloss_exponent"),
                             ("Range floor (m)","Drift (m$^2$/m)","Path-loss exponent")):
        selected=[r for r in sensitivity_rows if r["parameter"]==param]
        x=[r["value"] for r in selected]; y=[r["admitted"]/r["instances"] for r in selected]
        lo=[max(0.,v-r["admission_low99"]) for v,r in zip(y,selected)]
        hi=[max(0.,r["admission_high99"]-v) for v,r in zip(y,selected)]
        ax.errorbar(x,y,yerr=[lo,hi],fmt="o-",capsize=3,color="#217c78")
        ax.set_ylim(-.05,1.05); ax.set_xlabel(label); ax.set_ylabel("Admission fraction")
        ax.grid(alpha=.2)
    fig.savefig(figures/"wireless_sensitivity.pdf",bbox_inches="tight")
    fig.savefig(figures/"wireless_sensitivity.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    table=[r"\begin{table*}[t]",r"\centering\small",
           r"\caption{Literature-criterion and covariance comparisons: 200 paired layouts, 128 realizations; 95\% across-layout intervals. Decoded payload uses a common rate dictionary and the AWGN normal approximation.}",
           r"\label{tab:revision}",r"\begin{tabular}{lrrrrr}",r"\toprule",
           r"Policy & Admitted & Violation & Pilot fraction & Payload (kbit) & Risk upper bound \\",r"\midrule"]
    for row in summary:
        table.append(f"{LABELS[row['method']]} & {int(row['admitted'])}/{int(row['instances'])} & "
                     f"${row['violation']:.3f}\\pm{row['violation_ci95']:.3f}$ & "
                     f"${row['pilot_fraction']:.3f}\\pm{row['pilot_fraction_ci95']:.3f}$ & "
                     f"${row['goodput_bits']/1000:.2f}\\pm{row['goodput_bits_ci95']/1000:.2f}$ & "
                     f"{row['radial_union_bound']:.3f} " + chr(92)*2)
    table += [r"\bottomrule\end{tabular}\end{table*}"]
    (output/"revision_table.tex").write_text("\n".join(table)+"\n")
    table=[r"\begin{table}[t]",r"\centering\small\setlength{\tabcolsep}{3pt}",
           r"\caption{Common-admission paired differences (first minus second), with 95\% intervals.}",
           r"\label{tab:common}",r"\begin{tabular}{lrrr}",r"\toprule",
           r"Comparison & $N$ & Violation diff. & Pilot diff. \\",r"\midrule"]
    for r in paired_rows:
        a={"ERGO-joint":"ERGO","GRE-joint":"GRE","Adaptive-E":"A-E","Adaptive-GRE":"A-GRE"}[r["first"]]
        b={"SPEB-joint":"SPEB","SDW-power":"SDW","Robust-shortest":"shortest","ERGO-joint":"ERGO","GRE-joint":"GRE"}[r["second"]]
        table.append(f"{a}--{b} & {int(r['common'])} & "
                     f"${r['violation_difference']:.3f}\\pm{r['violation_ci95']:.3f}$ & "
                     f"${r['pilot_fraction_difference']:.3f}\\pm{r['pilot_fraction_ci95']:.3f}$ " + chr(92)*2)
    table += [r"\bottomrule\end{tabular}\end{table}"]
    (output/"common_table.tex").write_text("\n".join(table)+"\n")
    table=[r"\begin{table}[t]",r"\centering\small\setlength{\tabcolsep}{3pt}",
           r"\caption{38.901 UMi large-scale subset; terminal/anchor heights 20/10 m.}",
           r"\label{tab:umi}",r"\begin{tabular}{lrrr}",r"\toprule",
           r"Policy & Admitted & Violation & Pilot frac. \\",r"\midrule"]
    for row in umi:
        table.append(f"{LABELS[row['method']]} & {int(row['admitted'])}/{int(row['instances'])} & "
                     f"${row['violation']:.3f}\\pm{row['violation_ci95']:.3f}$ & {row['pilot_fraction']:.3f} " +chr(92)*2)
    table += [r"\bottomrule\end{tabular}\end{table}"]
    (output/"umi_table.tex").write_text("\n".join(table)+"\n")
    table=[r"\begin{table}[t]",r"\centering\small\setlength{\tabcolsep}{2pt}",
           r"\caption{SINR-induced losses. $N$ is the admitted count out of 200; $P_{>r}$ is the conditional mission probability of exceeding the erasure budget.}",
           r"\label{tab:packet}",r"\begin{tabular}{llrrr}",r"\toprule",
           r"Loss process & Policy & $N$ & Violation & $P_{>r}$ \\",r"\midrule"]
    for row in packet:
        mode="Independent" if row["loss_mode"]=="independent" else "+ common burst"
        name="PHY-GRE" if row['method']=="PHY-GRE" else LABELS[row['method']]
        table.append(f"{mode} & {name} & {int(row['admitted'])} & {row['violation']:.3f} & {row['budget_exceed_mission']:.3f} " +chr(92)*2)
    table += [r"\bottomrule\end{tabular}\end{table}"]
    (output/"packet_table.tex").write_text("\n".join(table)+"\n")
    by_method={r["method"]:r for r in summary}
    e,g,ae,ag=(by_method[x] for x in ("ERGO-joint","GRE-joint","Adaptive-E","Adaptive-GRE"))
    text=(f"The Gaussian reserve (GRE) admits {int(g['admitted'])}/{int(g['instances'])} missions, "
          f"with conditional violation ${g['violation']:.3f}\\pm{g['violation_ci95']:.3f}$ and "
          f"pilot fraction ${g['pilot_fraction']:.3f}\\pm{g['pilot_fraction_ci95']:.3f}$. "
          f"On the respective pre-admitted routes, adaptive E and adaptive GRE have pilot fractions "
          f"{ae['pilot_fraction']:.3f} and {ag['pilot_fraction']:.3f}. "
          f"For ERGO, the actual-covariance eigenvalue union bound averages {e['eigen_union_bound']:.3f}, "
          f"the radial-integral union bound {e['radial_union_bound']:.3f}, and observed violation {e['violation']:.3f}, "
          "separating the implemented bound components on the evaluated histories.\n")
    (output/"revision_text.tex").write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances",type=int,default=200)
    parser.add_argument("--replications",type=int,default=128)
    parser.add_argument("--sensitivity-instances",type=int,default=200)
    parser.add_argument("--solver-instances",type=int,default=5)
    parser.add_argument("--solver-seconds",type=float,default=3.)
    parser.add_argument("--seed",type=int,default=20261014)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--reports-only",action="store_true")
    args=parser.parse_args()
    if min(args.instances,args.replications,args.sensitivity_instances,args.solver_instances)<2:
        parser.error("All sample counts must be at least 2.")
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    if not args.reports_only:
        start=time.perf_counter(); cfg=research_config()
        baseline_and_conservatism(args,cfg,output)
        standards_channel(args,output)
        sensitivity(args,output)
        optimizer_benchmark(args,output)
        metadata=dict(config=asdict(cfg),gaussian_envelope_cap=gaussian_envelope_cap(cfg),
                      instances=args.instances,replications=args.replications,
                      sensitivity_instances=args.sensitivity_instances,solver_instances=args.solver_instances,
                      seed_start=args.seed,umi_seed_start=550001,sensitivity_seed_start=660001,
                      solver_seed_start=770001,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      elapsed_s=time.perf_counter()-start,
                      sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__),Path(__file__).with_name("wireless_extensions.py"),Path(__file__).with_name("secure_drop_research.py"))},
                      propagation="TR 38.901 V18.0.0 UMi pathloss, LOS probability and correlated shadowing subset; not full clustered channel or measurements",
                      packet_model="AWGN normal approximation; 48/800 control bits/symbols; 608 data bits and 320/640/1280/2560/5120 symbol dictionary; 0.02 control floor; correlated burst 0.03; PHY-GRE risk split 0.05+0.05",
                      certificate="Known unbiased affine Gaussian model; Riemann upper bound; noise-independent losses; union bound retained")
        (output/"revision_metadata.json").write_text(json.dumps(metadata,indent=2))
    reports(output)
    print("Reviewer-requested experiment outputs and LaTeX tables written.",flush=True)


if __name__=="__main__":
    main()
