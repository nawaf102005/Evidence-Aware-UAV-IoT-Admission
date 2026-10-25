"""Estimator-consistent radio reserves for GNSS-denied UAV--IoT delivery.

Run: python secure_drop_research.py --instances 200 --replications 128

This implements a finite-route / finite-power-codebook research experiment.
The theorem concerns the declared affine, Gaussian observation model only.
Actual nonlinear ranges are evaluated separately with an EKF; no unconditional
hardware, NLOS-bias, spoofing, or continuous-flight safety claim is made.
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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Config:
    anchors: int = 6
    stops: int = 5
    route_orders: int = 8
    area_m: float = 1200.0
    altitude_m: float = 60.0
    powers_w: tuple[float, ...] = (0.0, 0.02, 0.08, 0.32)
    bandwidth_hz: float = 20e6
    pilot_s: float = 1e-4
    pilot_control_s: float = 0.0  # Legacy runs use all exchange symbols for ranging.
    frame_s: float = 1e-3
    data_power_w: float = 0.2
    data_min_bits: float = 512.0
    range_floor_m: float = 0.30
    ranging_efficiency: float = 1.0
    noise_w: float = 3.98e-21 * 20e6 * 5.0
    external_interference_w: float = 7.94e-13  # -91 dBm; before local interferer
    reference_gain: float = 4.66e-5  # free-space 3.5 GHz at 1 m
    pathloss_exponent: float = 2.6
    shadow_sd_db: float = 4.0
    nlos_probability: float = 0.15
    nlos_extra_db: float = 12.0
    rician_k: float = 5.0
    drift_m2_per_m: float = 0.002
    radius_m: float = 2.0
    mission_risk: float = 0.10
    erasure_budget: int = 1
    initial_sd_m: float = 0.30
    flight_j_per_m: float = 70.0
    payload_j_per_m_kg: float = 20.0
    capacity_j: float = 400e3
    reserve_j: float = 25e3
    speed_m_s: float = 10.0
    service_s: float = 5.0
    charge_s: float = 300.0
    radio_weight: float = 0.5
    energy_scale_j: float = 1e5
    time_weight: float = 0.2
    time_scale_s: float = 1e3
    pilot_energy_price_per_j: float = 1000.0

    @property
    def epochs_max(self) -> int:
        return self.stops + 2  # deliveries, at most one charge, return to depot

    @property
    def chi2_threshold(self) -> float:
        # Exact upper quantile of chi-square with 2 degrees of freedom.
        return -2.0 * math.log(self.mission_risk / self.epochs_max)

    @property
    def covariance_cap(self) -> float:
        return self.radius_m**2 / self.chi2_threshold


METHODS = (
    "Distance-fixed", "Geometry-uniform", "Nominal-joint",
    "Robust-shortest", "Robust-uniform", "ERGO-joint", "Perfect-position",
)
LABELS = {
    "Distance-fixed": "Distance/fixed",
    "Geometry-uniform": "Geometry/uniform",
    "Nominal-joint": "Nominal joint",
    "Robust-shortest": "Robust shortest",
    "Robust-uniform": "Robust uniform",
    "ERGO-joint": "ERGO joint",
    "Perfect-position": "Position oracle",
}


def eigmin2(matrix: np.ndarray) -> np.ndarray:
    """Vectorized minimum eigenvalue of real symmetric 2 x 2 matrices."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1]
    disc = np.hypot(matrix[..., 0, 0] - matrix[..., 1, 1], 2 * matrix[..., 0, 1])
    return (trace - disc) / 2


def inv2(matrix: np.ndarray) -> np.ndarray:
    """Vectorized inverse, used for small positive-definite covariance matrices."""
    det = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    if np.any(det <= 0):
        raise ValueError("A covariance/information matrix is not positive definite.")
    out = np.empty_like(matrix)
    out[..., 0, 0] = matrix[..., 1, 1] / det
    out[..., 1, 1] = matrix[..., 0, 0] / det
    out[..., 0, 1] = -matrix[..., 0, 1] / det
    out[..., 1, 0] = -matrix[..., 1, 0] / det
    return out


@dataclass
class Instance:
    seed: int
    anchors: np.ndarray
    nodes: np.ndarray  # 0 depot, 1..S deliveries, S+1 charger
    payload_kg: np.ndarray
    deadlines_s: np.ndarray
    gains: np.ndarray  # node x anchor; identical across every method
    interference: np.ndarray
    orders: list[tuple[int, ...]]


def generate_instance(seed: int, cfg: Config) -> Instance:
    rng = np.random.default_rng(seed)
    angles = np.arange(cfg.anchors) * 2 * np.pi / cfg.anchors + rng.uniform(-0.15, 0.15, cfg.anchors)
    angles += rng.uniform(0, 2 * np.pi)
    radii = rng.uniform(0.43, 0.53, cfg.anchors) * cfg.area_m
    anchors = cfg.area_m / 2 + np.c_[np.cos(angles), np.sin(angles)] * radii[:, None]
    depot = np.array([[100.0, 100.0]])
    stops = rng.uniform(170, cfg.area_m - 170, (cfg.stops, 2))
    charger = np.array([[cfg.area_m * 0.52, cfg.area_m * 0.50]])
    nodes = np.vstack((depot, stops, charger))
    payload = rng.uniform(0.2, 0.6, cfg.stops)
    deadlines = rng.uniform(850, 1250, cfg.stops)
    ranges = np.sqrt(np.sum((nodes[:, None, :] - anchors[None, :, :])**2, axis=2) + cfg.altitude_m**2)
    shadow = rng.normal(0, cfg.shadow_sd_db, ranges.shape)
    blocked = rng.random(ranges.shape) < cfg.nlos_probability
    k = cfg.rician_k
    fading = np.abs(np.sqrt(k / (k + 1)) + (rng.normal(size=ranges.shape) + 1j * rng.normal(size=ranges.shape)) / np.sqrt(2 * (k + 1)))**2
    gains = cfg.reference_gain * ranges**(-cfg.pathloss_exponent)
    gains *= 10**(-(shadow + blocked * cfg.nlos_extra_db) / 10) * fading
    jammer_pos = np.array([0.72, 0.28]) * cfg.area_m
    jammer_dist = np.sqrt(np.sum((nodes - jammer_pos)**2, axis=1) + cfg.altitude_m**2)
    interference = cfg.external_interference_w + 0.01 * cfg.reference_gain * jammer_dist**(-cfg.pathloss_exponent)

    remaining = set(range(1, cfg.stops + 1))
    nearest: list[int] = []
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda j: float(np.linalg.norm(nodes[j] - nodes[current])))
        nearest.append(nxt)
        remaining.remove(nxt)
        current = nxt
    candidates = [tuple(nearest), tuple(nearest[::-1]), tuple(np.argsort(deadlines) + 1)]
    candidates += [tuple(rng.permutation(np.arange(1, cfg.stops + 1))) for _ in range(10 * cfg.route_orders)]
    orders = list(dict.fromkeys(candidates))[:cfg.route_orders]
    return Instance(seed, anchors, nodes, payload, deadlines, gains, interference, orders)


class RadioTable:
    """Exhaustive codebook oracle with analytically exact 2-D geometry tests."""
    def __init__(self, instance: Instance, node: int, cfg: Config, codebook: np.ndarray):
        self.cfg = cfg
        self.codebook = codebook
        self.powers = np.asarray(cfg.powers_w)[codebook]
        delta = instance.nodes[node] - instance.anchors
        ranges = np.sqrt(np.sum(delta**2, axis=1) + cfg.altitude_m**2)
        self.H = delta / ranges[:, None]
        self.sinr = np.asarray(cfg.powers_w)[None, :] * instance.gains[node, :, None]
        self.sinr /= cfg.noise_w + instance.interference[node]
        beta = cfg.bandwidth_hz / np.sqrt(12.0)
        if not (0 <= cfg.pilot_control_s < cfg.pilot_s):
            raise ValueError("Control duration must be less than the ranging exchange.")
        enr = self.sinr * cfg.bandwidth_hz * (cfg.pilot_s-cfg.pilot_control_s)
        self.variance = np.full_like(enr, np.inf)
        self.variance[:, 1:] = cfg.range_floor_m**2 + cfg.ranging_efficiency * 299792458.0**2 / (8 * np.pi**2 * beta**2 * enr[:, 1:])
        self.weights = 1 / self.variance[np.arange(cfg.anchors)[None, :], codebook]
        contributions = self.weights[:, :, None, None] * self.H[None, :, :, None] * self.H[None, :, None, :]
        self.total_info = contributions.sum(axis=1)
        self.min_info: dict[int, np.ndarray] = {0: eigmin2(self.total_info)}
        for erasures in {1, cfg.erasure_budget}:
            minimum = self.min_info[0].copy()
            for erased in itertools.combinations(range(cfg.anchors), erasures):
                remaining_info = self.total_info - contributions[:, erased, :, :].sum(axis=1)
                minimum = np.minimum(minimum, eigmin2(remaining_info))
            self.min_info[erasures] = minimum
        self.count = (codebook > 0).sum(axis=1)
        self.energy = cfg.pilot_s * self.powers.sum(axis=1)
        self.cost = self.count + cfg.pilot_energy_price_per_j * self.energy
        self.rate = cfg.bandwidth_hz * np.log2(1 + cfg.data_power_w * instance.gains[node].max() / (cfg.noise_w + instance.interference[node]))
        self.bits = self.rate * np.maximum(cfg.frame_s - self.count * cfg.pilot_s, 0)
        self.sorted_indices = np.argsort(self.cost, kind="stable")
        self.uniform_mask = np.all((codebook == 0) | (codebook == 2), axis=1)
        strongest = np.argsort(instance.gains[node])[-3:]
        levels = np.zeros(cfg.anchors, dtype=int)
        levels[strongest] = 2
        self.fixed_index = int(np.flatnonzero(np.all(codebook == levels, axis=1))[0])
        uniform_three = self.uniform_mask & (self.count == 3)
        indices = np.flatnonzero(uniform_three)
        self.geometry_index = int(indices[np.argmax(self.min_info[0][indices])])

    def select(self, prior_variance: float, erasures: int, uniform: bool = False) -> int | None:
        required = 1 / self.cfg.covariance_cap - 1 / prior_variance
        valid = (self.min_info[erasures] + 1e-10 >= required) & (self.bits >= self.cfg.data_min_bits)
        if uniform:
            valid &= self.uniform_mask
        ordered = self.sorted_indices[valid[self.sorted_indices]]
        return int(ordered[0]) if len(ordered) else None

    def worst_erasure(self, index: int, prior: np.ndarray, number: int) -> tuple[int, ...]:
        active = np.flatnonzero(self.codebook[index])
        precision = inv2(prior)
        full = precision + self.total_info[index]
        worst = ()
        worst_value = float(eigmin2(full))
        for erased in itertools.combinations(active, min(number, len(active))):
            removal = sum((self.weights[index, j] * np.outer(self.H[j], self.H[j]) for j in erased), start=np.zeros((2, 2)))
            value = float(eigmin2(full - removal))
            if value < worst_value:
                worst_value, worst = value, tuple(int(j) for j in erased)
        return worst


@dataclass
class Route:
    events: tuple[int, ...]
    lengths: np.ndarray
    flight_j: float
    completion_s: float
    recharges: int


def feasible_routes(instance: Instance, cfg: Config) -> list[Route]:
    routes: list[Route] = []
    charger = cfg.stops + 1
    for order in instance.orders:
        for gap in (-1, *range(1, cfg.stops + 1)):
            events = list(order)
            if gap >= 0:
                events.insert(gap, charger)
            events.append(0)
            remaining = float(instance.payload_kg.sum())
            battery = cfg.capacity_j
            energy = elapsed = 0.0
            previous = 0
            lengths = []
            valid = True
            for node in events:
                distance = float(np.linalg.norm(instance.nodes[node] - instance.nodes[previous]))
                leg_energy = distance * (cfg.flight_j_per_m + cfg.payload_j_per_m_kg * remaining)
                battery -= leg_energy
                energy += leg_energy
                elapsed += distance / cfg.speed_m_s + cfg.frame_s
                lengths.append(distance)
                if battery < cfg.reserve_j:
                    valid = False
                    break
                if node == charger:
                    battery = cfg.capacity_j
                    elapsed += cfg.charge_s
                elif node != 0:
                    elapsed += cfg.service_s
                    if elapsed > instance.deadlines_s[node - 1]:
                        valid = False
                        break
                    remaining -= instance.payload_kg[node - 1]
                previous = node
            if valid:
                routes.append(Route(tuple(events), np.asarray(lengths), energy, elapsed, int(gap >= 0)))
    return routes


@dataclass
class Plan:
    method: str
    route: Route
    schedules: tuple[int, ...]
    objective: float
    bits: float
    mean_pilot_fraction: float
    pilot_energy_j: float


def make_plan(method: str, routes: list[Route], tables: list[RadioTable], cfg: Config) -> Plan | None:
    if not routes:
        return None
    if method != "Perfect-position" and cfg.initial_sd_m**2 > cfg.covariance_cap:
        return None  # Theorem's initial-envelope condition is not available.
    if method in {"Distance-fixed", "Geometry-uniform", "Robust-shortest"}:
        candidate_routes = [min(routes, key=lambda r: r.flight_j + cfg.time_weight * r.completion_s * cfg.energy_scale_j / cfg.time_scale_s)]
    else:
        candidate_routes = routes
    best: Plan | None = None
    for route in candidate_routes:
        schedules = []
        cost = bits = energy = count = 0.0
        for node, distance in zip(route.events, route.lengths):
            table = tables[node]
            prior = cfg.covariance_cap + cfg.drift_m2_per_m * distance
            if method == "Distance-fixed":
                index = table.fixed_index
            elif method == "Geometry-uniform":
                index = table.geometry_index
            elif method == "Perfect-position":
                index = 0  # the all-zero codebook entry
            else:
                budget = 0 if method == "Nominal-joint" else cfg.erasure_budget
                index = table.select(prior, budget, uniform=(method == "Robust-uniform"))
            if index is None or table.bits[index] < cfg.data_min_bits:
                break
            schedules.append(index)
            cost += table.cost[index]
            count += table.count[index]
            energy += table.energy[index]
            if 1 <= node <= cfg.stops:
                bits += table.bits[index]
        if len(schedules) != len(route.events):
            continue
        objective = route.flight_j / cfg.energy_scale_j + cfg.time_weight * route.completion_s / cfg.time_scale_s + cfg.radio_weight * cost
        candidate = Plan(method, route, tuple(schedules), float(objective), float(bits), float(count * cfg.pilot_s / (len(schedules) * cfg.frame_s)), float(energy))
        if best is None or candidate.objective < best.objective:
            best = candidate
    return best


def filter_experiment(instance: Instance, plan: Plan, tables: list[RadioTable], cfg: Config,
                      samples: int, mode: str = "affine", loss_budget: int = 1,
                      mismatch: float = 1.0, bias_m: float = 0.0) -> dict[str, float]:
    """Execute a real KF/EKF recursion and evaluate measured estimation errors.

    Worst-subset losses depend on the covariance and selected codeword, never
    on a measurement value in the affine experiment. Paired methods reuse
    initial/process/measurement normal draws indexed by the same physical node.
    """
    if plan.method == "Perfect-position":
        return dict(violation=0.0, rmse_m=0.0, nees=0.0, coverage95=1.0, certificate=1.0)
    rng = np.random.default_rng(instance.seed + 90_000_000)
    initial = rng.normal(size=(samples, 2))
    process = rng.normal(size=(cfg.stops + 2, samples, 2))
    measurement = rng.normal(size=(cfg.stops + 2, samples, cfg.anchors))
    errors = cfg.initial_sd_m * initial
    P = np.eye(2) * cfg.initial_sd_m**2
    any_violation = np.zeros(samples, dtype=bool)
    sum_squared = sum_nees = sum_covered = 0.0
    cert = True
    for node, distance, index in zip(plan.route.events, plan.route.lengths, plan.schedules):
        table = tables[node]
        Q = cfg.drift_m2_per_m * distance
        errors += np.sqrt(mismatch * Q) * process[node]
        P = P + np.eye(2) * Q
        erasure_prior = np.eye(2) * (cfg.covariance_cap + Q)
        erased = table.worst_erasure(index, erasure_prior, loss_budget)
        active = [int(j) for j in np.flatnonzero(table.codebook[index]) if j not in erased]
        if active:
            H = table.H[active]
            levels = table.codebook[index, active]
            Rdiag = table.variance[active, levels]
            noise = measurement[node][:, active] * np.sqrt(Rdiag)[None, :]
            if mode == "affine":
                posterior = inv2(inv2(P) + H.T @ (H / Rdiag[:, None]))
                K = posterior @ (H.T / Rdiag[None, :])
                errors = errors @ (np.eye(2) - K @ H).T - noise @ K.T
                P = posterior
            else:
                # Batch EKF with nonlinear slant ranges. Each trial is centered
                # on its predicted position by an ideal, error-preserving command.
                if P.ndim == 2:
                    P = np.broadcast_to(P, (samples, 2, 2)).copy()
                residual_pos = instance.nodes[node][None, :] + errors
                nominal = np.broadcast_to(instance.nodes[node], (samples, 2)).copy()
                for column, anchor in enumerate(active):
                    a = instance.anchors[anchor]
                    delta = nominal - a
                    predicted_range = np.sqrt(np.sum(delta**2, axis=1) + cfg.altitude_m**2)
                    Hi = delta / predicted_range[:, None]
                    true_range = np.sqrt(np.sum((residual_pos - a)**2, axis=1) + cfg.altitude_m**2)
                    innovation = true_range + noise[:, column] + bias_m - predicted_range
                    PH = np.einsum("nij,nj->ni", P, Hi)
                    S = np.einsum("ni,ni->n", Hi, PH) + Rdiag[column]
                    Ki = PH / S[:, None]
                    nominal += Ki * innovation[:, None]
                    A = np.eye(2)[None, :, :] - Ki[:, :, None] * Hi[:, None, :]
                    P = A @ P @ np.swapaxes(A, 1, 2) + Rdiag[column] * Ki[:, :, None] * Ki[:, None, :]
                errors = residual_pos - nominal
        # Certificate here uses actual affine covariance (or EKF proxy in stress).
        max_eigen = np.linalg.eigvalsh(P)[..., -1]
        cert &= bool(np.all(max_eigen <= cfg.covariance_cap + 1e-8))
        squared = np.sum(errors**2, axis=1)
        any_violation |= squared > cfg.radius_m**2
        precision = inv2(P)
        nees = (np.einsum("ni,ij,nj->n", errors, precision, errors) if P.ndim == 2
                else np.einsum("ni,nij,nj->n", errors, precision, errors))
        sum_squared += squared.mean()
        sum_nees += nees.mean()
        sum_covered += np.mean(nees <= -2 * np.log(0.05))
    n = len(plan.route.events)
    return dict(violation=float(any_violation.mean()), rmse_m=float(np.sqrt(sum_squared / n)),
                nees=float(sum_nees / n), coverage95=float(sum_covered / n), certificate=float(cert))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan")
    return float(values.mean()), float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def aggregate(rows: list[dict]) -> list[dict]:
    summary = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        admitted = [r for r in subset if r["admitted"]]
        entry = dict(method=method, instances=len(subset), admitted=len(admitted), admission_fraction=len(admitted) / len(subset))
        for key in ("violation", "rmse_m", "nees", "coverage95", "certificate", "bits", "pilot_fraction", "flight_kj", "objective", "completion_s", "planning_ms"):
            mean, half = interval([r[key] for r in admitted])
            entry[key], entry[key + "_ci95"] = mean, half
        summary.append(entry)
    return summary


def plot_results(summary: list[dict], stress: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    main = [r for r in summary if r["method"] != "Perfect-position"]
    colors = ["#9c9c9c", "#c5a15a", "#d95f59", "#759bc9", "#8780b0", "#167b75"]
    labels = [LABELS[r["method"]] for r in main]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.1), layout="constrained")
    for axis, key, label, scale in zip(axes.flat, ["violation", "pilot_fraction", "bits", "flight_kj"],
                                      ["Mission violation rate", "Ranging airtime fraction", "Telemetry capacity (kbit)", "Propulsion energy (kJ)"], [1, 1, 1e-3, 1]):
        axis.bar(range(len(main)), [r[key] * scale for r in main], color=colors,
                 yerr=[r[key + "_ci95"] * scale for r in main], capsize=3)
        axis.set_xticks(range(len(main)), labels, rotation=35, ha="right", fontsize=10)
        axis.set_ylabel(label, fontsize=11)
        axis.grid(axis="y", alpha=.2)
        if key == "violation":
            axis.axhline(0.1, color="#333333", linestyle="--", linewidth=1, label="Design risk = 0.10")
            axis.legend(fontsize=10)
    fig.savefig(folder / "comparison.pdf", bbox_inches="tight")
    fig.savefig(folder / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    grouped = []
    for name in dict.fromkeys(r["condition"] for r in stress):
        values = [r["violation"] for r in stress if r["condition"] == name and r["admitted"]]
        grouped.append((name, *interval(values)))
    fig, ax = plt.subplots(figsize=(4.1, 3.2), layout="constrained")
    ax.bar(range(len(grouped)), [r[1] for r in grouped], yerr=[r[2] for r in grouped], capsize=4, color="#167b75")
    ax.set_xticks(range(len(grouped)), ["Affine 1 loss", "EKF", "EKF + bias", "2 losses", "4x drift"],
                  rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mission error violation", fontsize=11)
    ax.axhline(0.1, color="#333333", ls="--", lw=1)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(folder / "stress.pdf", bbox_inches="tight")
    fig.savefig(folder / "stress.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_suite(args) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    codebook = np.array(list(itertools.product(range(len(cfg.powers_w)), repeat=cfg.anchors)), dtype=int)
    rows: list[dict] = []
    stress_rows: list[dict] = []
    time_start = time.perf_counter()
    for trial in range(args.instances):
        instance = generate_instance(args.seed + trial, cfg)
        start = time.perf_counter()
        tables = [RadioTable(instance, node, cfg, codebook) for node in range(cfg.stops + 2)]
        routes = feasible_routes(instance, cfg)
        shared_ms = (time.perf_counter() - start) * 1000
        proposed = None
        for method in METHODS:
            start = time.perf_counter()
            plan = make_plan(method, routes, tables, cfg)
            planning_ms = shared_ms + (time.perf_counter() - start) * 1000
            record = dict(seed=instance.seed, method=method, admitted=int(plan is not None),
                          routes=len(routes), planning_ms=planning_ms)
            if plan is None:
                record.update({key: float("nan") for key in ("violation", "rmse_m", "nees", "coverage95", "certificate", "bits", "pilot_fraction", "flight_kj", "objective", "completion_s")})
            else:
                record.update(filter_experiment(instance, plan, tables, cfg, args.replications,
                                                loss_budget=cfg.erasure_budget))
                record.update(bits=plan.bits, pilot_fraction=plan.mean_pilot_fraction,
                              flight_kj=plan.route.flight_j / 1000, objective=plan.objective,
                              completion_s=plan.route.completion_s)
            rows.append(record)
            if method == "ERGO-joint":
                proposed = plan
        for condition, mode, losses, mismatch, bias in (
            ("Affine / 1 erasure", "affine", 1, 1., 0.),
            ("EKF / true ranges", "ekf", 1, 1., 0.),
            ("EKF / +1m range bias", "ekf", 1, 1., 1.),
            ("Affine / 2 erasures", "affine", 2, 1., 0.),
            ("Affine / 4x drift", "affine", 1, 4., 0.),
        ):
            result = dict(seed=instance.seed, condition=condition, admitted=int(proposed is not None))
            metrics = (filter_experiment(instance, proposed, tables, cfg, args.stress_replications,
                                         mode=mode, loss_budget=losses, mismatch=mismatch, bias_m=bias)
                       if proposed is not None else {k: float("nan") for k in ("violation", "rmse_m", "nees", "coverage95", "certificate")})
            result.update(metrics)
            stress_rows.append(result)
        if (trial + 1) % 25 == 0 or trial + 1 == args.instances:
            print(f"Completed {trial + 1}/{args.instances} paired instances", flush=True)
    summary = aggregate(rows)
    save_csv(output / "raw_trials.csv", rows)
    save_csv(output / "summary.csv", summary)
    save_csv(output / "stress_trials.csv", stress_rows)
    metadata = dict(config=asdict(cfg), instances=args.instances, seed_start=args.seed,
                    replications=args.replications, stress_replications=args.stress_replications,
                    numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
                    platform=platform.platform(), elapsed_s=time.perf_counter() - time_start,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    guarantee="Conditional affine-Gaussian estimation error at service/charge/return epochs; not flight or hardware safety.")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(summary, stress_rows, output.parent / "figures")
    write_latex_results(summary, stress_rows, args, output)
    print("\nConditional affine-Gaussian results (not field measurements):")
    for row in summary:
        print(f"{row['method']:20s} admit={row['admitted']:3d}/{row['instances']} "
              f"violate={row['violation']:.4f} +/- {row['violation_ci95']:.4f} "
              f"pilot={row['pilot_fraction']:.3f} NEES={row['nees']:.3f}")


def write_latex_results(summary, stress_rows, args, output) -> None:
    rows = [r for r in summary if r["method"] != "Perfect-position"]
    table = ["% Automatically generated from executed experiment; do not edit numbers manually.",
             r"\begin{table*}[t]", r"\centering", r"\caption{Paired synthetic experiments. Values are means with 95\% across-instance Student-$t$ intervals; outcomes are conditional on admission.}",
             r"\label{tab:results}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Policy & Admitted & Mission error violation & Ranging fraction & IoT bits (kbit) & Energy (kJ) & NEES \\", r"\midrule"]
    for r in rows:
        table.append(f"{LABELS[r['method']]} & {r['admitted']}/{r['instances']} & "
                     f"${r['violation']:.3f}\\pm{r['violation_ci95']:.3f}$ & "
                     f"${r['pilot_fraction']:.3f}\\pm{r['pilot_fraction_ci95']:.3f}$ & "
                     f"${r['bits']/1000:.2f}\\pm{r['bits_ci95']/1000:.2f}$ & "
                     f"${r['flight_kj']:.1f}\\pm{r['flight_kj_ci95']:.1f}$ & {r['nees']:.2f} " + chr(92) * 2)
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (output / "results_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    by_method = {r["method"]: r for r in summary}
    proposed, nominal, fixed = (by_method[k] for k in ("ERGO-joint", "Nominal-joint", "Distance-fixed"))
    text = (f"We execute {args.instances} independently seeded scenarios with {args.replications} Gaussian state/noise realizations per scenario and policy. "
            f"ERGO admits {proposed['admitted']} missions. Its conditional probability of at least one service-epoch error exceeding the 2-m radius is "
            f"${proposed['violation']:.3f}\\pm{proposed['violation_ci95']:.3f}$, compared with "
            f"${nominal['violation']:.3f}\\pm{nominal['violation_ci95']:.3f}$ for nominal joint design and "
            f"${fixed['violation']:.3f}\\pm{fixed['violation_ci95']:.3f}$ for distance/fixed allocation. "
            f"The proposed ranging fraction is ${proposed['pilot_fraction']:.3f}\\pm{proposed['pilot_fraction_ci95']:.3f}$. "
            f"Its mean normalized estimation-error squared (NEES) is {proposed['nees']:.2f}; the affine model predicts 2. "
            "These are model-based simulation results, not flight measurements.\n\n")
    for name in dict.fromkeys(r["condition"] for r in stress_rows):
        values = [r["violation"] for r in stress_rows if r["condition"] == name and r["admitted"]]
        mean, half = interval(values)
        text += f"For {name.replace('+', 'plus ')}, the conditional mission violation is ${mean:.3f}\\pm{half:.3f}$. "
    text += "No affine-model guarantee is extended to these nonlinear, biased, excessive-erasure, or drift-mismatch conditions.\n"
    (output / "results_text.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--replications", type=int, default=128)
    parser.add_argument("--stress-replications", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if min(args.instances, args.replications, args.stress_replications) < 2:
        parser.error("All sample counts must be at least 2.")
    run_suite(args)


if __name__ == "__main__":
    main()
