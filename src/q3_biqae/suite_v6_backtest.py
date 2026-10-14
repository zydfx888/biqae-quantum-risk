"""Annual expanding-window retraining, validation-only choices and drifted holdings."""
from time import perf_counter
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from .config import ROOT, write_json, digest
from .suite_v6_stability import train, sparse_lp, CFG
from .trained_model import train_mixture, TrainedRiskModel
from .feasibility import empirical_table
from .exact_reference import cvar, risk
from .data_prepare import parse_french

OUT=ROOT/'artifacts/validation_suite_v6/backtest'

def execute_path(data, policies, method, cost_bp):
    wealth=1.;hold=np.zeros(4);rows=[];years=[]
    for year in range(2011,2026):
        target=np.asarray(policies[(year,method)]);pre=hold.copy()
        turnover=float(np.abs(target-pre).sum()/2)
        # Cost is charged on both purchases and sales, per dollar traded.
        factor=1.
        for _ in range(100):factor=1-cost_bp/1e4*np.abs(factor*target-pre).sum()
        trade_dollars_ratio=float(np.abs(factor*target-pre).sum());cost=1-factor
        values=data.loc[str(year)];first=True;yearstart=wealth;fees=wealth*cost
        for date,r in values.iterrows():
            if first:wealth*=factor;hold=target.copy()
            ret=float(hold@r.values);net=(factor if first else 1.)*(1+ret)-1
            wealth*=1+ret;hold=hold*(1+r.values)/(1+ret)
            rows.append({'date':str(date.date()),'year':year,'method':method,'cost_bp_per_traded_dollar':cost_bp,'net_return':net,'gross_return':ret,'nav':wealth,'turnover':turnover if first else 0.,'fee_fraction_pretrade_nav':cost if first else 0.})
            first=False
        years.append({'year':year,'method':method,'cost_bp_per_traded_dollar':cost_bp,'turnover':turnover,'traded_notional_ratio':trade_dollars_ratio,'fee_fraction_pretrade_nav':cost,'fee_dollars_initial_nav':fees,'net_annual_return':wealth/yearstart-1,'pretrade_weights':pre.tolist(),'target_weights':target.tolist(),'end_weights':hold.tolist()})
    return rows,years

def run():
    OUT.mkdir(parents=True,exist_ok=False);start=perf_counter();source=ROOT/'data/raw/5_Industry_Portfolios_daily_CSV.zip'
    write_json(OUT/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'years':list(range(2011,2026)),'assets':['Cnsmr','Manuf','HiTec','Hlth'],'training':'2000 through year-3 inclusive','validation':'previous two complete calendar years','test':'current calendar year only after model and weights saved','gmm_candidates':[13,29,43],'gmm_selection':'highest validation log likelihood','learning_rate_candidates':[.03,.1,.3],'learning_rate_selection':'lowest validation ES of exact raw-gradient candidates, common to every trained policy','scenario_count':20003,'scenario_seed':20260915,'training_seed':600,'config':CFG.to_dict(),'policies':['dynamic_mc','dynamic_grid_biqae','raw_exact','raw_lp','equal'],'rebalance':'first observed trading date annually, using only preceding years; buy and hold with drifting weights between annual rebalances','costs_bp_per_dollar_purchased_or_sold':[0,10,25],'turnover':'half L1 target minus drifted pretrade weights; initial cash entry has 0.5 turnover and full invested notional','cost_accounting':'self-financing wealth factor f=1-c*sum(abs(f*target-pretrade)); fees apply buys and sells, no daily rebalancing','caveat':'retrospective public industry returns, not investable constituent execution; one fixed policy seed per year','raw_sha256':digest(source),'source_sha256':digest(__file__)})
    data=parse_french(source)[['Cnsmr','Manuf','HiTec','Hlth']];policies={};audit=[]
    for year in range(2011,2026):
        print('annual',year,flush=True);dest=OUT/f'year_{year}';dest.mkdir()
        training=data.loc['2000-01-01':f'{year-3}-12-31'];validation=data.loc[f'{year-2}-01-01':f'{year-1}-12-31']
        assert training.index.max()<validation.index.min() and validation.index.max()<pd.Timestamp(f'{year}-01-01')
        candidates=[]
        for seed in [13,29,43]:
            model,history=train_mixture(training.values,seed=seed);model.save(dest/f'gmm_{seed}.npz');write_json(dest/f'gmm_{seed}_history.json',history)
            candidates.append({'seed':seed,'validation_log_density':float(model.log_density(validation.values).mean()),'iterations':len(history)})
        selected=max(candidates,key=lambda c:c['validation_log_density']);model=TrainedRiskModel.load(dest/f"gmm_{selected['seed']}.npz");model.save(dest/'trained_model.npz')
        raw=model.sample(20003,20260915);np.savez_compressed(dest/'generated_scenarios.npz',returns=raw);full=empirical_table(raw);val=empirical_table(validation.values)
        rates=[];refs={}
        for rate in [.03,.1,.3]:
            p=train(raw,'raw','exact',600,rate,dest/f'validation_exact_{rate}.json');refs[rate]=p
            rates.append({'rate':rate,'validation_es':cvar(val,p['weights'],.975)})
        rate=min(rates,key=lambda c:c['validation_es'])['rate'];p=refs[rate]
        write_json(dest/'policy_raw_exact.json',p);policies[(year,'raw_exact')]=p['weights']
        lp=sparse_lp(full);write_json(dest/'policy_raw_lp.json',lp);policies[(year,'raw_lp')]=lp['weights']
        equal={'weights':[.25]*4};write_json(dest/'policy_equal.json',equal);policies[(year,'equal')]=equal['weights']
        for method in ['mc','grid_biqae']:
            p=train(raw,'dynamic64',method,600,rate,dest/f'policy_dynamic_{method}.json');policies[(year,f'dynamic_{method}')]=p['weights']
        selection={'year':year,'train_rows':len(training),'train_first':str(training.index.min().date()),'train_last':str(training.index.max().date()),'validation_rows':len(validation),'validation_first':str(validation.index.min().date()),'validation_last':str(validation.index.max().date()),'test_first':str(data.loc[str(year)].index.min().date()),'gmm_candidates':candidates,'selected_gmm_seed':selected['seed'],'learning_rates':rates,'selected_rate':rate,'test_used_for_selection':False,'model_sha256':digest(dest/'trained_model.npz')}
        write_json(dest/'selection.json',selection);audit.append(selection)
    # Only reloaded, frozen final policies enter business paths.
    for year in range(2011,2026):
        for method in ['dynamic_mc','dynamic_grid_biqae','raw_exact','raw_lp','equal']:
            policies[(year,method)]=json.loads((OUT/f'year_{year}'/f'policy_{method}.json').read_text())['weights']
    rows=[];years=[]
    for method in ['dynamic_mc','dynamic_grid_biqae','raw_exact','raw_lp','equal']:
        for cost in [0,10,25]:
            r,y=execute_path(data,policies,method,cost);rows+=r;years+=y
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'daily_paths.csv',index=False);yf=pd.DataFrame(years);yf.to_csv(OUT/'annual_paths.csv',index=False)
    summary=[]
    for (method,cost),g in frame.groupby(['method','cost_bp_per_traded_dollar']):
        ret=g.net_return.values;nav=g.nav.values;peaks=np.maximum.accumulate(np.r_[1.,nav])[1:];annual=yf[(yf.method==method)&(yf.cost_bp_per_traded_dollar==cost)]
        rr=risk(-ret,np.full(len(ret),1/len(ret)),.975)
        summary.append({'method':method,'cost_bp':cost,'days':len(ret),'total_return':nav[-1]-1,'annualized_return_252':nav[-1]**(252/len(ret))-1,'annualized_volatility_252':float(ret.std(ddof=1)*np.sqrt(252)),'daily_es_975':rr['es'],'max_drawdown':float(np.min(nav/peaks-1)),'mean_annual_turnover_including_initial':float(annual.turnover.mean()),'mean_annual_turnover_excluding_initial':float(annual[annual.year>2011].turnover.mean()),'total_fee_initial_nav_units':float(annual.fee_dollars_initial_nav.sum()),'ending_nav':float(nav[-1])})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);write_json(OUT/'chronology_audit.json',audit)
    write_json(OUT/'completion.json',{'status':'complete','years':15,'fitted_gmm_candidates':45,'trained_policies_including_validation':75,'business_policies':75,'paths':15,'daily_rows':len(frame),'runtime_s':perf_counter()-start,'date_leakage_violations':0,'independent_seed_replication':False})
    write_json(OUT/'checksums.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json'})

if __name__=='__main__':run()
