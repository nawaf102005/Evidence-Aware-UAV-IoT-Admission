"""Literature comparators, Gaussian-risk reserves and wireless channel tests.

SPEB follows Shen, Dai and Win, IEEE/ACM ToN 2014 (doi:2278984).
SDW-power solves their single-agent linear-EFIM minimum-power problem, extended
to a finite family of erasure subsets. SLSQP solves the convex formulation;
this is not a claim to reproduce their complete robust-angle algorithm.
E-opt-cut implements the discrete E-opt constraint using MILP eigenvector
cuts. The constraints are equivalent to the 2-D MISOCP representation; this
is an adaptation of an established criterion, not the anchor-placement
experiment of Duggal et al., arXiv:2604.00863.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.special import ndtr, ndtri

from secure_drop_research import (
    Config, Instance, Plan, RadioTable, Route, eigmin2, inv2,
    generate_instance, make_plan,
)


def erasure_sets(anchors: int, budget: int):
    """Exactly r subsets suffice: inactive terms are zero and information is PSD."""
    return tuple(itertools.combinations(range(anchors), min(budget, anchors)))


def contributions(table: RadioTable) -> np.ndarray:
    return (table.weights[..., None, None] * table.H[None, :, :, None]
            * table.H[None, :, None, :])


def trace_inverse2(matrix: np.ndarray) -> np.ndarray:
    determinant = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1]**2
    return (matrix[..., 0, 0] + matrix[..., 1, 1]) / determinant


def radial_tail_bounds(covariance: np.ndarray, radius: float,
                       terms: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Rigorous left/right Riemann bounds for a centered 2-D Gaussian radius.

    For uniform polar angle of a standard normal, the radial tail is the
    angular mean of exp[-R^2/(2*(lambda_max*cos^2+lambda_min*sin^2))].
    On [0, pi/2] this integrand decreases. Numerical quadrature thus has a
    known bracket instead of an unquantified integration tolerance.
    """
    if terms < 1 or radius < 0:
        raise ValueError("Positive quadrature count and nonnegative radius required.")
    small = eigmin2(covariance)
    large = np.trace(covariance, axis1=-2, axis2=-1) - small
    if np.any(small <= 0):
        raise ValueError("Gaussian covariance must be positive definite.")
    cos2 = np.cos(np.arange(terms) * (np.pi / (2 * terms)))**2
    variance = small[..., None] + (large-small)[..., None] * cos2
    upper = np.exp(-radius**2 / (2 * variance)).mean(axis=-1)
    first = np.exp(-radius**2 / (2 * large))
    last = np.exp(-radius**2 / (2 * small))
    lower = np.maximum(upper + (last-first)/terms, 0.)
    return lower, upper


def gaussian_envelope_cap(cfg: Config) -> float:
    epsilon = cfg.mission_risk / cfg.epochs_max
    return cfg.radius_m**2 / float(ndtri(1-epsilon/2))**2


class ExtendedRadio:
    """Extra criteria evaluated on precisely the original finite dictionary."""
    def __init__(self, table: RadioTable):
        self.table = table
        self.parts = contributions(table)
        self.erased = erasure_sets(table.cfg.anchors, table.cfg.erasure_budget)
        self.surviving = [table.total_info - self.parts[:, erased].sum(axis=1)
                          for erased in self.erased]
        self.cache: dict[tuple, int | None] = {}
        self.sdw_status = "not_run"
        self.sdw_power = None
        gamma = table.sinr[np.arange(table.cfg.anchors)[None, :], table.codebook]
        # 32-bit control word + 16-bit CRC; control duration is explicit.
        control_symbols = max(1,int(round(table.cfg.pilot_control_s*table.cfg.bandwidth_hz)))
        probabilities = .02+.98*bler_normal_approx(gamma, control_symbols, 48)
        probabilities = np.where(table.codebook > 0, probabilities, 0.)
        self.packet_budget_tail = poisson_binomial_over_budget(probabilities, table.cfg.erasure_budget)

    def choose(self, prior: np.ndarray, criterion: str) -> int | None:
        table, cfg = self.table, self.table.cfg
        key = (criterion, *np.round(prior.ravel(), 12))
        if key in self.cache:
            return self.cache[key]
        precision = inv2(prior)
        valid = table.bits >= cfg.data_min_bits
        if criterion == "E":
            worst = np.full(len(table.codebook), np.inf)
            for info in self.surviving:
                worst = np.minimum(worst, eigmin2(info + precision))
            valid &= worst >= 1/cfg.covariance_cap - 1e-10
        elif criterion == "SPEB":
            worst = np.zeros(len(table.codebook))
            for info in self.surviving:
                worst = np.maximum(worst, trace_inverse2(info + precision))
            # Same isotropic accuracy reference: tr(nu I) = 2 nu.
            valid &= worst <= 2*cfg.covariance_cap + 1e-10
        elif criterion in {"Gaussian", "PHY-Gaussian"}:
            if criterion == "PHY-Gaussian":
                valid &= self.packet_budget_tail <= .05/cfg.epochs_max
            indices = table.sorted_indices[valid[table.sorted_indices]]
            epsilon = cfg.mission_risk/cfg.epochs_max
            # Coarse bounds reject impossible candidates; only ambiguous
            # candidates need the final, conservative 256-term upper bound.
            for begin in range(0, len(indices), 96):
                chunk = indices[begin:begin+96]
                worst_lower = np.zeros(len(chunk))
                posteriors = [inv2(info[chunk] + precision) for info in self.surviving]
                for posterior in posteriors:
                    lower, _ = radial_tail_bounds(posterior, cfg.radius_m, 16)
                    worst_lower = np.maximum(worst_lower, lower)
                survivors = np.flatnonzero(worst_lower <= epsilon)
                if not len(survivors):
                    continue
                worst_upper = np.zeros(len(survivors))
                for posterior in posteriors:
                    _, upper = radial_tail_bounds(posterior[survivors], cfg.radius_m, 256)
                    worst_upper = np.maximum(worst_upper, upper)
                acceptable = np.flatnonzero(worst_upper <= epsilon*(1-1e-10))
                if len(acceptable):
                    index = int(chunk[survivors[acceptable[0]]])
                    self.cache[key] = index
                    return index
            self.cache[key] = None
            return None
        else:
            raise ValueError(criterion)
        indices = table.sorted_indices[valid[table.sorted_indices]]
        index = int(indices[0]) if len(indices) else None
        self.cache[key] = index
        return index

    def sdw_codeword(self) -> int | None:
        """Published linear-EFIM power objective, then disclosed protocol repair.

        No prior information and no variance floor in the published model.
        The actual-floor, erasure-robust SPEB check is mandatory after rounding.
        The repair raises levels only; among those codewords it minimizes the
        same radio cost as ERGO. A solver failure is recorded, not certified.
        """
        if "sdw" in self.cache:
            return self.cache["sdw"]
        table, cfg = self.table, self.table.cfg
        peak = max(cfg.powers_w)
        kappa = (table.variance[:, -1]-cfg.range_floor_m**2)*peak
        atoms = peak/kappa[:, None, None] * table.H[:, :, None] * table.H[:, None, :]
        targets = [atoms.copy() for _ in self.erased]
        for atoms_e, erased in zip(targets, self.erased):
            atoms_e[list(erased)] = 0
        target = 2*cfg.covariance_cap

        def speb(z, atom):
            J = np.einsum("m,mij->ij", z, atom) + 1e-14*np.eye(2)
            if eigmin2(J) <= 0:
                return 1e20, np.zeros(len(z))
            inverse = inv2(J)
            gradient = -np.einsum("ij,mjk,ki->m", inverse, atom, inverse)
            return float(np.trace(inverse)), gradient

        peak_speb = [speb(np.ones(cfg.anchors), atom)[0] for atom in targets]
        if any(value > target for value in peak_speb):
            self.sdw_status = "linear_model_infeasible"
            self.cache["sdw"] = None
            return None
        # Exact 2-D SOCP: ||(J11-J22,2 J12,2/target)|| <= tr(J)-2/target.
        # The conic residual stays smooth at the singular zero-power point.
        def cone(z, atom):
            W = np.trace(atom, axis1=1, axis2=2)
            Z = np.stack((atom[:,0,0]-atom[:,1,1], 2*atom[:,0,1]), axis=1)
            v = z @ Z
            norm_value = np.sqrt(v@v+4/target**2)
            return (z@W-2/target-norm_value)*target/2, (W-Z@v/norm_value)*target/2
        constraints = [dict(type="ineq", fun=lambda z, a=atom: cone(z,a)[0],
                            jac=lambda z, a=atom: cone(z,a)[1]) for atom in targets]
        # Homogeneity gives an inexpensive feasible starting point close to
        # the active boundary. Starting at full power can make SLSQP jump to
        # the singular all-zero EFIM before its line search finds that boundary.
        initial = np.ones(cfg.anchors)*min(1., 1.05*max(peak_speb)/target)
        result = minimize(lambda z: float(z.sum()), initial,
                          jac=lambda z: np.ones_like(z), bounds=[(0., 1.)]*cfg.anchors,
                          constraints=constraints, method="SLSQP",
                          options=dict(ftol=1e-10, maxiter=180))
        if not result.success or any(speb(result.x, a)[0] > target*(1+1e-7) for a in targets):
            self.sdw_status = "numerical_failure"
            self.cache["sdw"] = None
            return None
        self.sdw_power = peak*result.x
        levels = np.searchsorted(cfg.powers_w, np.maximum(self.sdw_power-1e-9, 0))
        mask = np.all(table.codebook >= levels, axis=1) & (table.bits >= cfg.data_min_bits)
        # No prior is credited, in keeping with the published static EFIM.
        for info in self.surviving:
            with np.errstate(divide="ignore", invalid="ignore"):
                speb_values = trace_inverse2(info)
            mask &= (eigmin2(info) > 1e-12) & (speb_values <= target + 1e-10)
        indices = table.sorted_indices[mask[table.sorted_indices]]
        index = int(indices[0]) if len(indices) else None
        self.sdw_status = "repaired_feasible" if index is not None else "protocol_infeasible"
        self.cache["sdw"] = index
        return index

    def erased_for(self, index: int, prior: np.ndarray, gaussian: bool = False):
        if not gaussian:
            return self.table.worst_erasure(index, prior, self.table.cfg.erasure_budget)
        precision = inv2(prior)
        risks = [float(radial_tail_bounds(inv2(precision+info[index]),
                                        self.table.cfg.radius_m, 256)[1])
                 for info in self.surviving]
        return self.erased[int(np.argmax(risks))]


def plan_from_schedules(method: str, route: Route, schedules, tables, cfg: Config) -> Plan:
    cost = sum(tables[n].cost[i] for n, i in zip(route.events, schedules))
    energy = sum(tables[n].energy[i] for n, i in zip(route.events, schedules))
    count = sum(tables[n].count[i] for n, i in zip(route.events, schedules))
    bits = sum(tables[n].bits[i] for n, i in zip(route.events, schedules) if 1 <= n <= cfg.stops)
    objective = (route.flight_j/cfg.energy_scale_j + cfg.time_weight*route.completion_s/cfg.time_scale_s
                 + cfg.radio_weight*cost)
    return Plan(method, route, tuple(schedules), float(objective), float(bits),
                float(count*cfg.pilot_s/(len(schedules)*cfg.frame_s)), float(energy))


def extended_plan(method: str, routes: list[Route], tables: list[RadioTable],
                  extra: list[ExtendedRadio], cfg: Config) -> Plan | None:
    if method in {"ERGO-joint", "Robust-shortest", "Robust-uniform", "Nominal-joint"}:
        return make_plan(method, routes, tables, cfg)
    if cfg.initial_sd_m**2 > cfg.covariance_cap:
        return None
    best = None
    for route in routes:
        schedules = []
        for node, distance in zip(route.events, route.lengths):
            if method == "SDW-power":
                index = extra[node].sdw_codeword()
            else:
                cap = gaussian_envelope_cap(cfg) if method in {"GRE-joint", "PHY-GRE"} else cfg.covariance_cap
                prior = np.eye(2)*(cap + cfg.drift_m2_per_m*distance)
                criterion = "PHY-Gaussian" if method == "PHY-GRE" else "Gaussian" if method == "GRE-joint" else "SPEB"
                index = extra[node].choose(prior, criterion)
            if index is None:
                break
            schedules.append(index)
        if len(schedules) == len(route.events):
            plan = plan_from_schedules(method, route, schedules, tables, cfg)
            if best is None or plan.objective < best.objective:
                best = plan
    return best


def adaptive_plan(base: Plan | None, method: str, extra: list[ExtendedRadio],
                  tables: list[RadioTable], cfg: Config) -> Plan | None:
    """Evaluate causal reserve selection on a pre-admitted route.

    Online decisions depend only on previous received-link indicators, not
    measured residuals. The offline envelope plan guarantees a feasible
    fallback at each epoch under any permitted erasure history. In this test
    the history is covariance-adversarial; no history-wide optimum is claimed.
    """
    if base is None:
        return None
    P = np.eye(2)*cfg.initial_sd_m**2
    schedules = []
    gaussian = method == "Adaptive-GRE"
    for node, length in zip(base.route.events, base.route.lengths):
        P = P + np.eye(2)*cfg.drift_m2_per_m*length
        index = extra[node].choose(P, "Gaussian" if gaussian else "E")
        if index is None:
            raise RuntimeError("Envelope fallback unexpectedly failed.")
        erased = extra[node].erased_for(index, P, gaussian)
        info = tables[node].total_info[index] - extra[node].parts[index, erased].sum(axis=0)
        P = inv2(inv2(P)+info)
        schedules.append(index)
    return plan_from_schedules(method, base.route, schedules, tables, cfg)


def bler_normal_approx(sinr, symbols: int, bits: int):
    """Complex AWGN normal approximation (PPV 2010), conditional on SINR.

    No third-order term. Not measured NR/LTE BLER and not an implemented code.
    """
    gamma = np.asarray(sinr)
    capacity = np.log2(1+gamma)
    dispersion = (1-(1+gamma)**-2) * np.log2(np.e)**2
    denominator = np.sqrt(np.maximum(dispersion/symbols, 1e-30))
    return ndtr((bits/symbols-capacity)/denominator)


def poisson_binomial_over_budget(probabilities: np.ndarray, budget: int):
    """Exact independent packet-count tail by dynamic programming."""
    probabilities = np.asarray(probabilities, dtype=float)
    distribution = np.zeros(probabilities.shape[:-1]+(probabilities.shape[-1]+1,))
    distribution[...,0] = 1.
    for anchor in range(probabilities.shape[-1]):
        p = probabilities[...,anchor,None]
        shifted = np.concatenate((np.zeros_like(distribution[...,:1]),distribution[...,:-1]),axis=-1)
        distribution = distribution*(1-p)+shifted*p
    return np.clip(distribution[...,budget+1:].sum(axis=-1),0,1)


def data_schedule(count, design_sinr, cfg: Config):
    """Common adaptive-rate dictionary, explicit header, ACK and frame overhead."""
    symbols = np.array([320,640,1280,2560,5120])
    residual = np.maximum(cfg.frame_s-np.asarray(count)*cfg.pilot_s-40e-6,0.)
    packets = np.floor(residual[...,None]/(symbols/cfg.bandwidth_hz+10e-6))
    expected = packets*512*(1-bler_normal_approx(design_sinr,symbols,608))
    choice = expected.argmax(axis=-1)
    selected_symbols = symbols[choice]
    selected_packets = np.take_along_axis(packets,choice[...,None],axis=-1)[...,0]
    return selected_symbols, selected_packets, expected.max(axis=-1)


def expected_payload(count, data_sinr, cfg: Config):
    return data_schedule(count,data_sinr,cfg)[2]


def packet_payload_budget(table: RadioTable):
    gamma = np.expm1(np.log(2)*table.rate/table.cfg.bandwidth_hz)
    return expected_payload(table.count,gamma,table.cfg)


def umi_pathloss(d2d, los, h_ut: float = 20., h_bs: float = 10., fc_ghz: float = 3.5):
    """TR 38.901 V18.0.0, Table 7.4.1-1, UMi large-scale loss only."""
    distance = np.asarray(d2d, dtype=float)
    if (np.any(distance < 10) or np.any(distance > 5000)
            or not (1.5 <= h_ut <= 22.5) or h_bs != 10.):
        raise ValueError("UMi applicability: d2D=10..5000 m, hUT=1.5..22.5 m, hBS=10 m.")
    if not (0.5 < fc_ghz < 100):
        raise ValueError("Carrier outside the UMi model frequency range.")
    d3d = np.sqrt(distance**2+(h_bs-h_ut)**2)
    breakpoint = 4*(h_bs-1)*(h_ut-1)*fc_ghz*1e9/3e8
    pl1 = 32.4+21*np.log10(d3d)+20*np.log10(fc_ghz)
    pl2 = (32.4+40*np.log10(d3d)+20*np.log10(fc_ghz)
           -9.5*np.log10(breakpoint**2+(h_bs-h_ut)**2))
    pl_los = np.where(distance <= breakpoint, pl1, pl2)
    pl_nlos = 22.4+35.3*np.log10(d3d)+21.3*np.log10(fc_ghz)-.3*(h_ut-1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def generate_umi_instance(seed: int, cfg: Config) -> Instance:
    if cfg.altitude_m != 10.:
        raise ValueError("UMi test uses a 20-m terminal and 10-m anchors (relative height 10 m).")
    instance = generate_instance(seed, cfg)
    rng = np.random.default_rng(seed + 40_000_000)
    d2d = np.linalg.norm(instance.nodes[:, None]-instance.anchors[None], axis=2)
    # Table 7.4.2-1; distances in this generator are above 18 m.
    los_probability = np.minimum(18/d2d, 1)*(1-np.exp(-d2d/36))+np.exp(-d2d/36)
    los = rng.random(d2d.shape) < los_probability
    pathloss = umi_pathloss(d2d, los)
    # Spatially correlated SF with d_cor=10 m (LOS), 13 m (NLOS), Table 7.5-6.
    distances = np.linalg.norm(instance.nodes[:, None]-instance.nodes[None], axis=2)
    normals = rng.normal(size=d2d.shape)
    shadow = np.empty_like(d2d)
    for anchor in range(cfg.anchors):
        fields = {}
        for state, length, sd in ((True, 10., 4.), (False, 13., 7.82)):
            correlation = np.exp(-distances/length)+1e-12*np.eye(len(instance.nodes))
            fields[state] = sd*(np.linalg.cholesky(correlation) @ normals[:, anchor])
        shadow[:, anchor] = np.where(los[:, anchor], fields[True], fields[False])
    gains = 10**(-(pathloss+shadow)/10)
    # No small-scale fast fading/clusters in this standards subset.
    return replace(instance, gains=gains)


def evaluate(instance: Instance, plan: Plan, tables: list[RadioTable], extra: list[ExtendedRadio],
             cfg: Config, samples: int, loss_mode: str = "worst", packet_floor: float = .02,
             csi_error_db: float = 0., csi_margin_db: float = 0., control_bits: int = 48) -> dict:
    """Vectorized affine KF with exact covariance for each loss history.

    Adaptive schedules are reselected for every realized history in stochastic
    tests, rather than replaying a schedule chosen for a different history.
    Normal draws and physical-node packet uniforms are paired across policies.
    """
    rng = np.random.default_rng(instance.seed+90_000_000)
    errors = cfg.initial_sd_m*rng.normal(size=(samples, 2))
    process = rng.normal(size=(cfg.stops+2, samples, 2))
    measurement = rng.normal(size=(cfg.stops+2, samples, cfg.anchors))
    loss_rng = np.random.default_rng(instance.seed+91_000_000)
    uniforms = loss_rng.random((cfg.stops+2, samples, cfg.anchors))
    common_burst = loss_rng.random((cfg.stops+2, samples))
    max_packets=int(np.ceil(cfg.frame_s/(320/cfg.bandwidth_hz+10e-6)))
    data_uniforms=loss_rng.random((cfg.stops+2,samples,max_packets))
    fade_rng = np.random.default_rng(instance.seed+92_000_000)
    residual_db = fade_rng.normal(0, csi_error_db, (cfg.stops+2, cfg.anchors))-csi_margin_db
    P = np.broadcast_to(np.eye(2)*cfg.initial_sd_m**2, (samples, 2, 2)).copy()
    any_failure = np.zeros(samples, dtype=bool)
    budget_exceeded = np.zeros(samples, dtype=bool)
    total_pilots = total_goodput = total_nees = 0.
    risk_bounds = np.zeros(samples)
    eigen_bounds = np.zeros(samples)
    over_budget_epochs = active_loss_sum = epoch_sum = 0
    unresolved = np.zeros(samples, dtype=bool)
    gaussian = plan.method in {"GRE-joint", "Adaptive-GRE", "PHY-GRE"}
    adaptive = plan.method.startswith("Adaptive")
    for node, distance, planned_index in zip(plan.route.events, plan.route.lengths, plan.schedules):
        Q = cfg.drift_m2_per_m*distance
        errors += np.sqrt(Q)*process[node]
        P += Q*np.eye(2)
        indices = np.full(samples, planned_index, dtype=int)
        if adaptive:
            for prior_values in np.unique(np.round(P.reshape(samples, 4), 12), axis=0):
                mask = np.all(np.round(P.reshape(samples, 4), 12) == prior_values, axis=1)
                index = extra[node].choose(prior_values.reshape(2, 2), "Gaussian" if gaussian else "E")
                if index is None:
                    unresolved[mask] = True
                    # Unavailable reserve is an operational failure, not an
                    # evidence-free risk-certified admission.
                else:
                    indices[mask] = index
        table = tables[node]
        active = table.codebook[indices] > 0
        sinr = table.sinr[np.arange(cfg.anchors)[None, :], table.codebook[indices]]
        sinr *= 10**(-residual_db[node][None, :]/10)
        if loss_mode == "worst":
            losses = np.zeros_like(active)
            for index in np.unique(indices):
                selected = indices == index
                # Static policies use their declared design prior; adaptive
                # policies use the current, history-dependent covariance.
                if adaptive:
                    prior = P[np.flatnonzero(selected)[0]]
                else:
                    cap = gaussian_envelope_cap(cfg) if gaussian else cfg.covariance_cap
                    prior = (cap+Q)*np.eye(2)
                erased = extra[node].erased_for(int(index), prior, gaussian)
                losses[np.ix_(np.flatnonzero(selected), erased)] = True
        else:
            # Control+CRC bits in the reserved control segment; floor is a declared
            # sensitivity parameter, not a calibrated empirical BLER floor.
            control_symbols = max(1,int(round(cfg.pilot_control_s*cfg.bandwidth_hz)))
            probability = packet_floor+(1-packet_floor)*bler_normal_approx(sinr, control_symbols, control_bits)
            losses = uniforms[node] < probability
            if loss_mode == "correlated":
                losses |= common_burst[node, :, None] < .03
        losses &= active
        lost = losses.sum(axis=1)
        budget_exceeded |= lost > cfg.erasure_budget
        over_budget_epochs += int(np.count_nonzero(lost > cfg.erasure_budget))
        active_loss_sum += int(lost.sum())
        epoch_sum += samples
        received = active & ~losses
        levels = table.codebook[indices]
        R_design = table.variance[np.arange(cfg.anchors)[None, :], levels]
        R_true = cfg.range_floor_m**2 + (R_design-cfg.range_floor_m**2)*10**(residual_db[node][None, :]/10)
        # Filter assumes the planned map. csi_error_db>0 intentionally causes
        # variance mismatch; the reported P then loses its actual-error status.
        for mask in np.unique(received, axis=0):
            members = np.all(received == mask, axis=1)
            anchors = np.flatnonzero(mask)
            if not len(anchors):
                continue
            H = table.H[anchors]
            variance = R_design[members][:, anchors]
            prior = P[members]
            info = np.einsum("ni,ij,ik->njk", 1/variance, H, H)
            posterior = inv2(inv2(prior)+info)
            gain = np.einsum("nij,jk,nk->nik", posterior, H.T, 1/variance)
            A = np.eye(2)-np.einsum("nij,jk->nik", gain, H)
            noise = measurement[node][members][:, anchors]*np.sqrt(R_true[members][:, anchors])
            errors[members] = np.einsum("nij,nj->ni", A, errors[members])-np.einsum("nij,nj->ni", gain, noise)
            P[members] = posterior
        squared = np.sum(errors**2, axis=1)
        any_failure |= squared > cfg.radius_m**2
        nees = np.einsum("ni,nij,nj->n", errors, inv2(P), errors)
        total_nees += float(nees.mean())
        _, upper = radial_tail_bounds(P, cfg.radius_m, 256)
        risk_bounds += upper
        max_eig = np.trace(P, axis1=1, axis2=2)-eigmin2(P)
        eigen_bounds += np.exp(-cfg.radius_m**2/(2*max_eig))
        count = active.sum(axis=1)
        total_pilots += float(count.mean()*cfg.pilot_s/cfg.frame_s)
        if 1 <= node <= cfg.stops:
            # Same rate adaptation for every policy; the design chooses its
            # blocklength before unknown CSI mismatch is realized.
            association = int(np.argmax(instance.gains[node]))
            design_gamma = cfg.data_power_w*instance.gains[node,association]/(cfg.noise_w+instance.interference[node])
            data_gamma = design_gamma*10**(-float(residual_db[node,association])/10)
            symbols, packets, _ = data_schedule(count,design_gamma,cfg)
            payload, overhead = 512, 96
            successful = 1-bler_normal_approx(data_gamma, symbols, payload+overhead)
            decoded = ((np.arange(max_packets)[None,:]<packets[:,None])
                       & (data_uniforms[node]<successful[:,None]))
            total_goodput += float(decoded.sum(axis=1).mean()*payload)
    epochs = len(plan.route.events)
    # Missed allocation after an out-of-budget loss is an explicit failure.
    any_failure |= unresolved
    return dict(violation=float(any_failure.mean()), pilot_fraction=total_pilots/epochs,
                goodput_bits=total_goodput, nees=total_nees/epochs,
                radial_union_bound=float(np.minimum(risk_bounds, 1).mean()),
                eigen_union_bound=float(np.minimum(eigen_bounds, 1).mean()),
                budget_exceed_mission=float(budget_exceeded.mean()),
                budget_exceed_epoch=over_budget_epochs/epoch_sum,
                mean_losses=active_loss_sum/epoch_sum, unresolved=float(unresolved.mean()))


def local_eopt_cut(instance: Instance, node: int, length: float, cfg: Config,
                   time_limit: float = 3., max_rounds: int = 30) -> dict:
    """Independent MILP outer approximation with a certified objective gap.

    Missing eigenvector cuts relax the robust PSD constraints, so the MILP
    dual bound remains a lower bound. Every returned incumbent is checked
    against all erasure subsets. A timeout is 'unresolved', never infeasible.
    """
    begin = time.perf_counter()
    M, L = cfg.anchors, len(cfg.powers_w)-1
    powers = np.asarray(cfg.powers_w[1:])
    delta = instance.nodes[node]-instance.anchors
    d3d = np.sqrt(np.sum(delta**2, axis=1)+cfg.altitude_m**2)
    H = delta/d3d[:, None]
    gamma = instance.gains[node, :, None]*powers/(cfg.noise_w+instance.interference[node])
    beta = cfg.bandwidth_hz/np.sqrt(12)
    R = cfg.range_floor_m**2+cfg.ranging_efficiency*299792458.**2/(8*np.pi**2*beta**2*gamma*cfg.bandwidth_hz*(cfg.pilot_s-cfg.pilot_control_s))
    atoms = H[:, None, :, None]*H[:, None, None, :]/R[:, :, None, None]
    required = 1/cfg.covariance_cap-1/(cfg.covariance_cap+cfg.drift_m2_per_m*length)
    rate = cfg.bandwidth_hz*np.log2(1+cfg.data_power_w*instance.gains[node].max()/(cfg.noise_w+instance.interference[node]))
    data_gamma = np.expm1(np.log(2)*rate/cfg.bandwidth_hz)
    counts = np.arange(M+1)
    allowed = counts[expected_payload(counts,data_gamma,cfg)>=cfg.data_min_bits]
    count_max = int(allowed[-1]) if len(allowed) else -1
    sets = erasure_sets(M, cfg.erasure_budget)
    cost = np.tile(1+cfg.pilot_energy_price_per_j*cfg.pilot_s*powers, M)

    def matrices(levels):
        selected = np.zeros((M, 2, 2))
        nonzero = np.flatnonzero(levels)
        selected[nonzero] = atoms[nonzero, levels[nonzero]-1]
        full = selected.sum(axis=0)
        return [full-selected[list(erased)].sum(axis=0) for erased in sets]

    def feasible(levels):
        return np.count_nonzero(levels) <= count_max and all(eigmin2(J) >= required-1e-8 for J in matrices(levels))

    incumbent = np.full(M, L, dtype=int)
    if not feasible(incumbent):
        incumbent = None
    if incumbent is not None:
        # Feasible greedy incumbent, used only as an upper bound.
        for anchor in np.argsort(instance.gains[node]):
            for level in range(L):
                attempt = incumbent.copy(); attempt[anchor] = level
                if feasible(attempt):
                    incumbent = attempt; break
    def value(levels):
        return float(sum(1+cfg.pilot_energy_price_per_j*cfg.pilot_s*cfg.powers_w[x] for x in levels if x))
    upper = value(incumbent) if incumbent is not None else np.inf
    lower = 0.
    rows, lb, ub = [], [], []
    for anchor in range(M):
        row = np.zeros(M*L); row[anchor*L:(anchor+1)*L] = 1
        rows.append(row); lb.append(0.); ub.append(1.)
    rows.append(np.ones(M*L)); lb.append(0.); ub.append(float(count_max))

    def add_cut(erased, vector):
        row = np.einsum("i,mlij,j->ml", vector, atoms, vector)
        row[list(erased)] = 0.
        rows.append(row.ravel()); lb.append(required); ub.append(np.inf)
    for erased in sets:
        for theta in np.arange(4)*np.pi/4:
            add_cut(erased, np.array([np.cos(theta), np.sin(theta)]))
    status, rounds = "unresolved", 0
    if count_max < 0:
        status = "infeasible"
    else:
        for rounds in range(1, max_rounds+1):
            remaining = time_limit-(time.perf_counter()-begin)
            if remaining <= 0:
                break
            result = milp(cost, integrality=np.ones(M*L), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(np.asarray(rows), lb, ub),
                          options=dict(time_limit=max(.01, remaining), mip_rel_gap=1e-8))
            if result.status == 2:
                status = "infeasible"; break
            if getattr(result, "mip_dual_bound", None) is not None:
                lower = max(lower, float(result.mip_dual_bound))
            if result.x is None:
                break
            z = np.rint(result.x).reshape(M, L)
            levels = np.where(z.sum(axis=1)>0, z.argmax(axis=1)+1, 0)
            matrices_candidate = matrices(levels)
            violated = False
            for erased, J in zip(sets, matrices_candidate):
                eigenvalues, vectors = np.linalg.eigh(J)
                if eigenvalues[0] < required-1e-8:
                    violated = True; add_cut(erased, vectors[:, 0])
            if not violated:
                candidate_value = value(levels)
                if candidate_value < upper:
                    upper, incumbent = candidate_value, levels
                if upper-lower <= 1e-6:
                    status = "optimal"; break
    return dict(status=status, lower_bound=lower, upper_bound=upper,
                gap=max(0.,upper-lower)/max(upper, 1) if np.isfinite(upper) else float("nan"),
                levels=incumbent.tolist() if incumbent is not None else None,
                rounds=rounds, milliseconds=1000*(time.perf_counter()-begin))
