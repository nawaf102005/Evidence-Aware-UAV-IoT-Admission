"""Independent checks of the revision-10 certificates and optimizer bounds."""
from itertools import product
from dataclasses import replace
import numpy as np
from scipy.integrate import quad
from memory_certificates import (RadioAction, MemoryOracle, radial_bounds, posterior,
    plan_grid, queue_service, simulate_plan)
from secure_drop_research import Config
import unittest


def toy_action(strength=2., loss=.2, cost=1.):
    masks = np.array(list(product((0, 1), repeat=2)))
    H = np.eye(2)
    information = np.array([np.diag(m)*strength for m in masks])
    probability = np.prod(np.where(masks, 1-loss, loss), axis=1)
    return RadioAction((0, 1), .08, H, np.ones(2)/strength,
                  np.ones(2)*loss, information, probability, masks, cost, .000016, .0002)


def oracle():
    return MemoryOracle([[toy_action(s, .1+.03*j, cost=s)] for j, s in enumerate([2., 3., 1.5])],
                        [.6, .4, .7], np.eye(2)*.2, 2., terms=256)


def test_radial_bracket_against_adaptive_integration_and_singular_limit():
    for eig in ([.05, 2.], [.7, .7], [0., 1.3], [0., 0.]):
        P = np.diag(eig)
        exact = quad(lambda theta: np.exp(-2/(eig[0]+(eig[1]-eig[0])*np.cos(theta)**2))
                     if eig[0]+(eig[1]-eig[0])*np.cos(theta)**2 > 0 else 0,
                     0, np.pi/2, epsabs=1e-12)[0]*2/np.pi
        low, high = radial_bounds(P, 2., 256)
        assert low-1e-12 <= exact <= high+1e-12
        assert high-low <= 1/256+1e-12


def test_nested_memory_sandwich_encloses_full_history_mixture():
    o = oracle(); path = (0, 0, 0)
    state, weights = o.initial[None], np.ones(1)
    exact_sum = 0.
    for t in range(3):
        a = o.catalogs[t][0]
        # An independently written standard information update, PD here.
        prior = state[:, None]+o.process[t]*np.eye(2)
        state = np.linalg.inv(np.linalg.inv(prior)+a.information[None]).reshape(-1, 2, 2)
        weights = (weights[:, None]*a.probabilities).ravel()
        L, U = radial_bounds(state, o.radius, 8192)
        exact_sum += float(weights@((L+U)/2))
    bounds = [o.check(path, w) for w in (1, 2, 3)]
    assert all(L <= exact_sum <= U for L, U in bounds)
    assert bounds[0][1] >= bounds[1][1] >= bounds[2][1]
    assert bounds[0][0] <= bounds[1][0] <= bounds[2][0]
    # The lower sum bounds marginal sums, not the union of epoch events.


def test_one_epoch_window_equals_no_observation_formula():
    o = oracle()
    for t in range(3):
        a = o.catalogs[t][0]
        B = o.initial+sum(o.process[:t+1])*np.eye(2)
        P = np.linalg.inv(np.linalg.inv(B)[None]+a.information)
        expected = a.probabilities@radial_bounds(P, 2., 256)[1]
        assert abs(o.local(t, (0,))[1]-expected) < 1e-11


def test_pruned_histories_still_enclose_exact_window():
    o = oracle(); pruned = MemoryOracle(o.catalogs, o.process, o.initial, 2., 256, max_histories=4)
    for t in range(3):
        lo, hi, omitted = pruned.local(t, (0,)*(t+1))
        elo, ehi, _ = o.local(t, (0,)*(t+1))
        assert lo <= elo+1e-11 and hi >= ehi-1e-11
        if t:
            assert omitted > 0


def test_grid_cost_bracket_against_exhaustive_sequence_search():
    groups = [[toy_action(s, .15, cost=s) for s in (1., 2., 4.)] for _ in range(3)]
    o = MemoryOracle(groups, [.7]*3, np.eye(2)*.3, 2.)
    factors = o.factors(2)
    valid = [np.ones(3, bool)]*3
    alpha = .3
    plans = [(sum(groups[t][a].cost for t, a in enumerate(p)), p)
        for p in product(range(3), repeat=3) if o.check(p, 2)[1] <= alpha]
    exact = min(plans)[0]
    grid = plan_grid(factors, groups, valid, alpha, bins=600)
    assert grid.path is not None and grid.lower_cost <= exact <= grid.cost
    assert grid.bound <= alpha
    # A sufficiently fine grid must recover the exact discrete optimum here.
    assert plan_grid(factors, groups, valid, alpha, bins=60000).cost == exact


def test_zero_covariance_update_and_riccati_order():
    assert np.allclose(posterior(np.zeros((2, 2)), np.eye(2)), 0)
    A = np.array([[.8, .2], [.2, .7]])
    B = A+np.array([[.5, -.1], [-.1, .2]])
    S = np.array([[2., .6], [.6, 1.]])
    assert np.linalg.eigvalsh(posterior(B, S)-posterior(A, S)).min() >= -1e-12


def test_nonideal_ack_and_multiframe_finite_queue():
    cfg = replace(Config(), pilot_control_s=40e-6)
    one = queue_service(4, 1., 1, 24, cfg)
    two = queue_service(4, 1., 2, 24, cfg)
    assert two.attempts > one.attempts and two.failure < one.failure
    assert 0 <= one.expected_packets <= 24 and 0 <= two.expected_packets <= 24
    assert queue_service(4, .03, 2, 24, cfg).success < .999


def test_joseph_actual_covariance_domination_under_variance_inflation():
    rng = np.random.default_rng(45)
    modeled = np.array([[2., .2], [.2, 1.]])
    actual = .8*modeled
    for _ in range(15):
        h = rng.normal(size=2)
        model_prior, true_prior = modeled+.2*np.eye(2), actual+.2*np.eye(2)
        K = model_prior@h/(h@model_prior@h+1.)
        T = np.eye(2)-np.outer(K, h)
        modeled = posterior(model_prior, np.outer(h, h))
        actual = T@true_prior@T.T+.4*np.outer(K, K)
        assert np.linalg.eigvalsh(modeled-actual).min() >= -1e-10


def test_rare_tail_numerical_stability():
    lo, hi = radial_bounds(np.array([np.eye(2)*.001, np.diag([.002, .1])]), 2., 512)
    assert np.all(np.isfinite(hi)) and np.all(hi >= lo) and hi[0] < 1e-10


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in globals().items()
                              if key.startswith('test_') and callable(value))
