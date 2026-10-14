"""Registered precision/reliability and quantile/discretization experiments."""
from datetime import datetime, timezone
from time import perf_counter
from dataclasses import replace
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from .config import ROOT, Config, guarded, write_json, digest
from .scenario_table import ScenarioTable
from .trained_model import TrainedRiskModel, freeze_model_scenarios
from .exact_reference import risk, cvar
from .feasibility_baselines import estimate_classical_risk
from .es_estimation import estimate_risk
from .amplitude import estimate_payload
from .tail_oracle import excess_payload
from .statistics import holm

METHODS = ['mc', 'sobol', 'is', 'iqae', 'grid_biqae']
SOURCE = ROOT/'artifacts/feasibility_trained_v2'
SUITE = ROOT/'artifacts/validation_suite_v3'

def cp(h, n, delta=.05):
    return (0. if h == 0 else float(beta.ppf(delta/2, h, n-h+1)),
            1. if h == n else float(beta.ppf(1-delta/2, h+1, n-h)))

def load_inputs():
    complete = json.loads((SOURCE/'training_complete.json').read_text())
    model_path = SOURCE/'models/trained_risk_model.npz'
    assert digest(model_path) == complete['model_sha256']
    model = TrainedRiskModel.load(model_path)
    with np.load(SOURCE/'generated_scenarios.npz') as archive:
        raw = archive['returns']
    np.testing.assert_array_equal(raw, model.sample(20003, 20260915))
    w = np.array(json.loads((SOURCE/'models/policy_lp.json').read_text())['weights'])
    table = ScenarioTable.load(SOURCE/'frozen_distribution.npz')
    assert table.sha256 == complete['table_sha256']
    assert table.sha256 == freeze_model_scenarios(raw, 16).sha256
    return table, raw, w

def configure():
    return Config(alpha=.975, epsilon=.0002, grid_size=4097, max_rounds=32,
                  shots_per_round=256, max_grover_power=31, record_posterior=False)

def initialize(directory, protocol):
    directory = guarded(directory)
    directory.mkdir(parents=True, exist_ok=False)
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(),
                    data_model_source='artifacts/feasibility_trained_v2',
                    trained_model_sha256=digest(SOURCE/'models/trained_risk_model.npz'),
                    frozen_model_distribution_sha256=digest(SOURCE/'frozen_distribution.npz'),
                    fixed_LP_weights_sha256=digest(SOURCE/'models/policy_lp.json'),
                    backend='analytic_shots', count_A_inverse=True,
                    classical='one sample of the complete return vector; reuse for VaR+ES',
                    correction='saturated classical VaR confidence band uses full known upper support; one unsampled-tail regression added')
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'environment.json', {'python': sys.version,
               'source_hashes': {f.name: digest(f) for f in (ROOT/'src/q3_biqae').glob('*.py')}})
    return directory

def result_row(result, truth):
    var_ci, es_ci = result['ci']['var'], result['ci']['es']
    return {'status': 'ok', 'var': result['var'], 'es': result['es'],
            'var_error_bp': abs(result['var']-truth['var'])*10000,
            'es_error_bp': abs(result['es']-truth['es'])*10000,
            'var_covered': var_ci[0] <= truth['var'] <= var_ci[1],
            'es_covered': es_ci[0] <= truth['es'] <= es_ci[1],
            'var_ci_low': var_ci[0], 'var_ci_high': var_ci[1], 'es_ci_low': es_ci[0], 'es_ci_high': es_ci[1],
            'var_width_bp': (var_ci[1]-var_ci[0])*10000, 'es_width_bp': (es_ci[1]-es_ci[0])*10000,
            'var_resolved': result['var_resolved'], 'samples': result['samples'],
            'A_calls': result['resources']['oracle_A_calls'], 'shots': result['resources']['shots_total'],
            'queries': result['samples']+result['resources']['oracle_A_calls'],
            'runtime_ms': result['runtime_ms'], 'interval_kind': result['interval_kind']}

def estimate(table, weights, alpha, budget, seed, method, config):
    if method in ['mc', 'sobol', 'is']:
        return estimate_classical_risk(table, weights, alpha, budget, seed, method)
    return estimate_risk(table, alpha, budget, seed, method, replace(config, alpha=alpha), weights)

def finalize(directory, started, counts):
    write_json(directory/'status.json', {'status': 'complete', 'runtime_s': perf_counter()-started, **counts})
    write_json(directory/'checksums.json', {str(f.relative_to(directory)): digest(f) for f in directory.rglob('*') if f.is_file() and f.name != 'checksums.json'})

def precision():
    started = perf_counter()
    directory = initialize(SUITE/'exp2_precision', {
        'experiment': 2, 'wall_clock_cap_minutes': 60, 'alphas': [.95, .975],
        'budgets': [4096, 16384, 65536], 'methods': METHODS, 'seeds': list(range(100, 200)),
        'target_bp': [5, 10, 20], 'success': 'both absolute VaR and ES errors within target',
        'empirical_selection': 'smallest preregistered budget with >=95% joint successes; never interpolate',
        'marginal_evidence_selection': 'smallest budget whose two-sided 95% CP lower bound >=95%',
        'multiplicity': 'also compute Bonferroni CP bound over 5 methods x 3 budgets per alpha/target; keep descriptive/marginal/simultaneous conclusions separate',
        'interval_precision': 'also count successes with both returned CI half-widths <= target',
        'query_units': 'classical samples vs expanded A and A-dagger calls; full-table sorting/loading precondition stated separately',
        'stopping_config': configure().to_dict()})
    table, raw, weights = load_inputs()
    rows, logs = [], []
    for alpha in [.95, .975]:
        truth = risk(table.losses(weights), table.probabilities, alpha)
        for budget in [4096, 16384, 65536]:
            for method in METHODS:
                print(f'Exp2 alpha={alpha}, budget={budget}, {method}, 100 seeds', flush=True)
                for seed in range(100, 200):
                    key = dict(alpha=alpha, budget=budget, method=method, seed=seed)
                    try:
                        r = estimate(table, weights, alpha, budget, seed, method, configure())
                        row = {**key, **result_row(r, truth)}
                        assert row['queries'] <= budget
                        logs.append({**key, 'rounds_json': json.dumps(r['logs'], allow_nan=False)})
                    except Exception as exc:
                        row = {**key, 'status': 'failed', 'error': repr(exc), 'var_covered': False, 'es_covered': False}
                    rows.append(row)
                pd.DataFrame(rows).to_csv(directory/'results.csv', index=False)
                if perf_counter()-started > 3600:
                    raise TimeoutError('Experiment 2 reached 60-minute wall-clock cap')
    data = pd.DataFrame(rows)
    pd.DataFrame(logs).to_parquet(directory/'rounds.parquet', compression='zstd', index=False)
    summaries = []
    for (alpha, budget, method), group in data.groupby(['alpha', 'budget', 'method']):
        for target in [5, 10, 20]:
            successes = (group.status=='ok') & (group.var_error_bp<=target) & (group.es_error_bp<=target)
            n, h = len(group), int(successes.sum())
            narrow = successes & (group.var_width_bp<=2*target) & (group.es_width_bp<=2*target)
            low, high = cp(h, n)
            simultaneous_low, _ = cp(h, n, .05/15)
            vc, ec = int(group.var_covered.sum()), int(group.es_covered.sum())
            summaries.append({'alpha': alpha, 'budget': int(budget), 'method': method, 'target_bp': target,
                'trials': n, 'failures': int((group.status!='ok').sum()), 'joint_successes': h,
                'joint_success_rate': h/n, 'success_cp_low': low, 'success_cp_high': high,
                'success_simultaneous_cp_low': simultaneous_low, 'success_and_narrow_ci_rate': narrow.mean(),
                'var_mae_bp': group.var_error_bp.mean(), 'es_mae_bp': group.es_error_bp.mean(),
                'var_width_bp': group.var_width_bp.mean(), 'es_width_bp': group.es_width_bp.mean(),
                'var_coverage': vc/n, 'es_coverage': ec/n,
                'var_coverage_cp_low': cp(vc, n)[0], 'es_coverage_cp_low': cp(ec, n)[0],
                'coverage_under95_p': min(1., min(binomtest(vc,n,.95,alternative='less').pvalue, binomtest(ec,n,.95,alternative='less').pvalue)*2),
                'mean_queries': group.queries.mean(), 'p95_queries': group.queries.quantile(.95),
                'mean_shots': group.shots.mean(), 'mean_runtime_ms': group.runtime_ms.mean()})
    summary = pd.DataFrame(summaries)
    summary['coverage_under95_p_holm'] = holm(np.minimum(summary.coverage_under95_p,1).values)
    summary.to_csv(directory/'summary.csv', index=False)
    matched = []
    for (alpha, target, method), group in summary.groupby(['alpha', 'target_bp', 'method']):
        group = group.sort_values('budget')
        for criterion, mask in {
            'observed_95pct': group.joint_success_rate>=.95,
            'marginal_CP95_lower_ge95pct': group.success_cp_low>=.95,
            'simultaneous_CP95_lower_ge95pct': group.success_simultaneous_cp_low>=.95,
            'observed_success_and_narrow_CI_95pct': group.success_and_narrow_ci_rate>=.95
        }.items():
            eligible = group[mask]
            row = eligible.iloc[0] if len(eligible) else None
            matched.append({'alpha':alpha, 'target_bp':target, 'method':method, 'criterion':criterion,
                            'reached': row is not None, 'min_tested_budget': int(row.budget) if row is not None else None,
                            'mean_queries_at_budget': float(row.mean_queries) if row is not None else None,
                            'successes_at_budget': int(row.joint_successes) if row is not None else None})
    pd.DataFrame(matched).to_csv(directory/'matched_queries.csv', index=False)
    finalize(directory, started, {'trials': len(data), 'failed_trials': int((data.status!='ok').sum())})
    return directory

def decompose_es(estimated_es, exact_objective_at_tau, discrete_es, continuous_es):
    values = {'discretization_bp': (discrete_es-continuous_es)*10000,
              'threshold_bp': (exact_objective_at_tau-discrete_es)*10000,
              'sampling_amplitude_and_clip_bp': (estimated_es-exact_objective_at_tau)*10000,
              'total_vs_continuous_bp': (estimated_es-continuous_es)*10000}
    assert abs(values['discretization_bp']+values['threshold_bp']+values['sampling_amplitude_and_clip_bp']-values['total_vs_continuous_bp']) < 1e-8
    return values

def ablation():
    started = perf_counter()
    directory = initialize(SUITE/'exp3_ablation', {
        'experiment':3, 'wall_clock_cap_minutes':35, 'states':[16,32,64], 'alphas':[.95,.975],
        'methods':METHODS, 'seeds':list(range(100,130)), 'budget':65536,
        'frozen_inputs':'same serialized trained GMM, exactly same 20003 generated return vectors, same LP16 weights',
        'binning':'existing freeze_model_scenarios equal-weight sorting; do not re-optimize weights or bin by evaluation weights',
        'reference':'exact empirical enumeration of same finite model-generated samples, not unknown population or heldout market truth',
        'ES_ablation':'estimated VaR vs exact discrete VaR; classical variants reuse the identical sample counts; quantum variants use identical ES stage budget and seed',
        'decomposition':'signed total = discretization + exact RU objective loss at estimated VaR + amplitude/sampling error including final support clipping',
        'bias_warning':'exact-VaR diagnostic uses ground truth only for posthoc ablation and is not a deployable prediction or fair overall-query competitor',
        'stopping_config': configure().to_dict()})
    _, raw, weights = load_inputs()
    raw_table = ScenarioTable(raw, np.full(len(raw),1/len(raw)))
    rows, logs, truths = [], [], []
    for states in [16,32,64]:
        table = freeze_model_scenarios(raw, states)
        table.save(directory/f'distribution_{states}.npz')
        losses = table.losses(weights)
        for alpha in [.95,.975]:
            truth = risk(losses, table.probabilities, alpha)
            raw_truth = risk(raw_table.losses(weights), raw_table.probabilities, alpha)
            truths.append({'states':states, 'alpha':alpha, 'var':truth['var'], 'es':truth['es'],
                           'continuous_var':raw_truth['var'], 'continuous_es':raw_truth['es'],
                           'var_discretization_bp':(truth['var']-raw_truth['var'])*10000,
                           'es_discretization_bp':(truth['es']-raw_truth['es'])*10000})
            for method in METHODS:
                print(f'Exp3 states={states}, alpha={alpha}, {method}, 30 paired seeds', flush=True)
                for seed in range(100,130):
                    key = dict(states=states, alpha=alpha, method=method, seed=seed)
                    try:
                        result = estimate(table,weights,alpha,65536,seed,method,configure())
                        exact_at_tau = cvar(table,weights,alpha,result['var'])
                        if method in ['mc','sobol','is']:
                            empirical = np.array(result['logs'][0]['empirical_probabilities'])
                            sorted_losses = np.sort(losses, kind='stable')
                            fixed_es = float(truth['var']+empirical@np.maximum(sorted_losses-truth['var'],0)/(1-alpha))
                            fixed_cost = result['samples']
                            fixed_logs = {'same_batch':True}
                        else:
                            bound = float(np.ptp(losses))
                            cfg = configure()
                            var_budget = int(65536*cfg.var_budget_fraction)
                            fixed = estimate_payload(table.probabilities, excess_payload(losses,truth['var'],bound),
                                method,cfg,65536-var_budget,seed+1000003,cfg.failure_probability/2,
                                min(.5,cfg.epsilon*(1-alpha)/max(bound,1e-12)))
                            fixed_es = float(np.clip(truth['var']+bound*fixed.estimate/(1-alpha),losses.min(),losses.max()))
                            fixed_cost = fixed.ledger.oracle_A_calls
                            fixed_logs = fixed.rounds
                        rows.append({**key, 'status':'ok', **decompose_es(result['es'],exact_at_tau,truth['es'],raw_truth['es']),
                            'estimated_var':result['var'], 'true_var':truth['var'], 'var_abs_error_bp':abs(result['var']-truth['var'])*10000,
                            'estimated_tau_es':result['es'], 'true_tau_es':fixed_es,
                            'estimated_tau_es_abs_error_bp':abs(result['es']-truth['es'])*10000,
                            'true_tau_es_abs_error_bp':abs(fixed_es-truth['es'])*10000,
                            'full_queries':result['resources']['oracle_A_calls']+result['samples'],
                            'true_tau_diagnostic_queries':fixed_cost})
                        logs.append({**key,'estimated_tau_rounds':json.dumps(result['logs']), 'true_tau_rounds':json.dumps(fixed_logs)})
                    except Exception as exc:
                        rows.append({**key,'status':'failed','error':repr(exc)})
                pd.DataFrame(rows).to_csv(directory/'results.csv',index=False)
                if perf_counter()-started>2100:
                    raise TimeoutError('Experiment3 reached 35-minute cap')
    data = pd.DataFrame(rows)
    pd.DataFrame(truths).to_csv(directory/'discretization.csv',index=False)
    pd.DataFrame(logs).to_parquet(directory/'rounds.parquet',compression='zstd',index=False)
    good = data[data.status=='ok'].copy()
    good['abs_discretization_bp']=good.discretization_bp.abs()
    good['abs_threshold_bp']=good.threshold_bp.abs()
    good['abs_sampling_amplitude_and_clip_bp']=good.sampling_amplitude_and_clip_bp.abs()
    good['abs_total_vs_continuous_bp']=good.total_vs_continuous_bp.abs()
    columns = ['discretization_bp','threshold_bp','sampling_amplitude_and_clip_bp','total_vs_continuous_bp',
               'abs_discretization_bp','abs_threshold_bp','abs_sampling_amplitude_and_clip_bp','abs_total_vs_continuous_bp',
               'var_abs_error_bp','estimated_tau_es_abs_error_bp','true_tau_es_abs_error_bp','full_queries','true_tau_diagnostic_queries']
    good.groupby(['states','alpha','method'])[columns].mean().to_csv(directory/'summary.csv')
    finalize(directory, started, {'trials':len(data),'failed_trials':int((data.status!='ok').sum()),
                                 'decomposition_max_residual_bp':float(np.max(np.abs(good.discretization_bp+good.threshold_bp+good.sampling_amplitude_and_clip_bp-good.total_vs_continuous_bp)))})
    return directory

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',choices=['precision','ablation'])
    args=parser.parse_args()
    print(precision() if args.experiment=='precision' else ablation(),flush=True)
