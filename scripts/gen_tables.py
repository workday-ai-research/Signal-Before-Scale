import os
# Directory holding the analysis outputs (CSV/JSON) that these tables are built from.
RESULTS = os.environ.get("SBS_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))


"""Generate every LaTeX table in the paper directly from the saved analysis artifacts.
No number is typed by hand: each cell is formatted from the CSV/JSON values."""
import pandas as pd, json, numpy as np, os

P = {
 'cell': os.path.join(RESULTS, "cell_metrics.csv"),
 'contr': os.path.join(RESULTS, "contrasts.csv"),
 'ent':  os.path.join(RESULTS, "entropy.csv"),
 'ctl':  os.path.join(RESULTS, "controls.csv"),
 'fits': os.path.join(RESULTS, "fits.json"),
 'dr':   os.path.join(RESULTS, "decision_rule.json"),
}
cell=pd.read_csv(P['cell']); contr=pd.read_csv(P['contr'])
ent=pd.read_csv(P['ent']);   ctl=pd.read_csv(P['ctl'])
fits=json.load(open(P['fits'])); dr=json.load(open(P['dr']))
OUT="paper"

TASKS=['LaMP-1','LaMP-2','LaMP-3','LaMP-5']
MLABEL={'LaMP-1':'accuracy','LaMP-2':'macro-F1','LaMP-3':'MAE','LaMP-5':'ROUGE-1'}
METRIC={'LaMP-1':'accuracy','LaMP-2':'macro_f1','LaMP-3':'mae','LaMP-5':'rouge_1'}
MODELS=['haiku','sonnet','opus']
KS=[0,1,2,4,8,16,32]
f4=lambda x: "--" if x is None or (isinstance(x,float) and np.isnan(x)) else "%.4f"%x
def ci(v,lo,hi): return f"{f4(v)} {{\\scriptsize[{f4(lo)}, {f4(hi)}]}}"
def w(name, s):
    open(os.path.join(OUT,name),"w").write(s.rstrip()+"\n"); print("wrote", name)

gA=cell[cell.grid=='A']
nk=fits['nonparametric_knee']; gf=pd.DataFrame(nk['gain_fractions'])
sub=pd.DataFrame(fits['substitution_haiku_k_vs_opus_k0'])
sv=contr[contr.family=='signal_vs_k0']; cf=contr[contr.family=='compute_at_fixed_k']
did=contr[contr.family=='interaction_DiD']
ex=pd.DataFrame(fits['exchange_rate'])

# ================= T1: signal axis summary =================
L=[r"\begin{tabular}{llrccc}", r"\toprule",
   r"Task & Metric & $n$ & Total gain $k{=}0\!\to\!32$ & Knee $k$ (90\%) & Fraction of gain by $k{=}8$\\",
   r"\midrule"]
for t in TASKS:
    p=nk['model_pooled'][t]; g8=gf[(gf.task==t)&(gf.k==8)].iloc[0]
    L.append(f"{t} & {MLABEL[t]} & {p['n_items']} & {ci(p['total_gain'],p['tg_lo'],p['tg_hi'])} & "
             f"{p['knee']} [{int(p['knee_lo'])}, {int(p['knee_hi'])}] & {ci(g8.gain_frac,g8.gf_lo,g8.gf_hi)}\\\\")
L+= [r"\bottomrule", r"\end{tabular}"]
w("tab_saturation.tex","\n".join(L))

# ================= T2: substitution + resolvable-effect rate =================
L=[r"\begin{tabular}{llcccc}", r"\toprule",
   r"\multirow{2}{*}{Task} & \multirow{2}{*}{Metric} & \multicolumn{2}{c}{Small model $+$ history vs.\ largest model, no history} & "
   r"\multicolumn{2}{c}{Resolvable contrasts}\\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
   r" & & advantage at best $k$ & $p$ & signal & compute\\", r"\midrule"]
for t in TASKS:
    st=sub[sub.task==t]; b=st.loc[st.gain_vs_opus_k0.idxmax()]
    ns=int(sv[sv.task==t].resolved.sum()); nt=len(sv[sv.task==t])
    nc=int(cf[cf.task==t].resolved.sum()); nct=len(cf[cf.task==t])
    L.append(f"{t} & {MLABEL[t]} & {ci(b.gain_vs_opus_k0,b.lo,b.hi)} ($k{{=}}{int(b.k)}$) & {b.p:.4f} & {ns}/{nt} & {nc}/{nct}\\\\")
tot_s=(int(sv.resolved.sum()),len(sv)); tot_c=(int(cf.resolved.sum()),len(cf))
L+=[r"\midrule",
    f"All & & & & \\textbf{{{tot_s[0]}/{tot_s[1]}}} ({100*tot_s[0]/tot_s[1]:.0f}\\%) & "
    f"\\textbf{{{tot_c[0]}/{tot_c[1]}}} ({100*tot_c[0]/tot_c[1]:.0f}\\%)\\\\",
    r"\bottomrule", r"\end{tabular}"]
w("tab_substitution.tex","\n".join(L))

# ================= T3: the cross-term =================
L=[r"\begin{tabular}{lccccc}", r"\toprule",
   r"\multirow{2}{*}{Task} & \multicolumn{2}{c}{DiD at $k{=}32$ (metric units)} & \multicolumn{3}{c}{Exchange rate}\\",
   r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
   r" & Opus $-$ Haiku & Sonnet $-$ Haiku & signal span & compute span ($k{=}0$) & steps needed\\", r"\midrule"]
for t in TASKS:
    a=did[(did.k_hi==32)&(did.model_hi=='opus')&(did.model_lo=='haiku')&(did.task==t)].iloc[0]
    b=did[(did.k_hi==32)&(did.model_hi=='sonnet')&(did.model_lo=='haiku')&(did.task==t)].iloc[0]
    e=ex[(ex.task==t)&(ex.k_max==32)].iloc[0]
    steps = "unreachable" if not np.isfinite(e.steps_needed) else f"{e.steps_needed:.2f}"
    L.append(f"{t} & {ci(a.delta,a.delta_lo,a.delta_hi)} & {ci(b.delta,b.delta_lo,b.delta_hi)} & "
             f"{f4(e.signal_span)} & {f4(e.compute_span_k0)} & {steps}\\\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_crossterm.tex","\n".join(L))

# ================= T4: mechanism / entropy =================
L=[r"\begin{tabular}{lrrcccc}", r"\toprule",
   r"Task & $k$ & samples & fully deterministic & mean norm.\ entropy & max distinct & self-consistency gain\\", r"\midrule"]
for _,r_ in ent.sort_values(['task','k']).iterrows():
    g = r_.sc_gain_mae if r_.task=='LaMP-3' else r_.sc_gain_acc
    sps = str(r_.samples_per_item).strip('[]')
    L.append(f"{r_.task} & {int(r_.k)} & {sps} & {ci(r_.frac_deterministic,r_.lo_frac_deterministic,r_.hi_frac_deterministic)} & "
             f"{ci(r_.mean_H_norm,r_.lo_mean_H_norm,r_.hi_mean_H_norm)} & {int(r_.max_distinct)} & "
             f"{ci(g,r_.lo_self_consistency_gain,r_.hi_self_consistency_gain)}\\\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_mechanism.tex","\n".join(L))

# ================= T5: decision rule =================
vt=pd.DataFrame(dr['validation']['table']); v3=vt[vt.task=='LaMP-3']
L=[r"\begin{tabular}{cccccc}", r"\toprule",
   r"$\tau$ & $k^\star(\tau)$ (median) & upgrade rate & held-out loss & regret vs.\ oracle & regret of matched-budget random\\", r"\midrule"]
for _,r_ in v3.iterrows():
    ks = r"$\infty$" if not np.isfinite(r_.k_star) else f"{r_.k_star:.1f}"
    L.append(f"{r_.tau:.2f} & {ks} & {100*r_.upgrade:.0f}\\% & {f4(r_.loss_rule)} & {f4(r_.regret)} & {f4(r_.regret_random)}\\\\")
L+=[r"\midrule",
    f"always upgrade & --- & 100\\% & {f4(v3.iloc[0].loss_opus)} & {f4(v3.iloc[0].loss_opus-v3.iloc[0].loss_oracle)} & ---\\\\",
    f"never upgrade & --- & 0\\% & {f4(v3.iloc[0].loss_haiku)} & {f4(v3.iloc[0].loss_haiku-v3.iloc[0].loss_oracle)} & ---\\\\",
    f"oracle & --- & --- & {f4(v3.iloc[0].loss_oracle)} & 0.0000 & ---\\\\",
    r"\bottomrule", r"\end{tabular}"]
w("tab_decision.tex","\n".join(L))

# ================= APPENDIX: full grid A per task =================
blocks=[]
for t in TASKS:
    m=METRIC[t]
    L=[r"\begin{tabular}{rcccr}", r"\toprule",
       r"$k$ & Haiku (small) & Sonnet (mid) & Opus (large) & mean eff.\ $k$\\", r"\midrule"]
    for k in KS:
        cells=[]
        for mo in MODELS:
            r_=gA[(gA.task==t)&(gA.model==mo)&(gA.k==k)]
            if len(r_)==0: cells.append("---"); continue
            r_=r_.iloc[0]
            cells.append(f"{ci(r_[m],r_[m+'_lo'],r_[m+'_hi'])} \\tiny{{$n{{=}}{int(r_.n)}$}}")
        ek=gA[(gA.task==t)&(gA.k==k)].mean_effective_k
        L.append(f"{k} & "+" & ".join(cells)+f" & {ek.mean():.2f}\\\\")
    L+=[r"\bottomrule", r"\end{tabular}"]
    blocks.append((t,"\n".join(L)))
    w(f"tab_gridA_{t.replace('-','')}.tex","\n".join(L))

# ================= APPENDIX: paired CV comparison =================
mc=fits['model_comparison_paired_cv']; sf=fits['shared_floor_vs_free_floor']
forms=[('flat_per_model','no signal effect'),('sat_shared','no compute effect'),
       ('sat_floor_only','floor only'),('power_no_floor','power law, no floor'),
       ('exp_per_model','exponential in $k$')]
L=[r"\begin{tabular}{lcccc}", r"\toprule",
   r"Restriction of the full fit & "+" & ".join(TASKS)+r"\\", r"\midrule"]
for key,lab in forms:
    row=[]
    for t in TASKS:
        d=mc[t].get(key)
        row.append("---" if d is None else f"{1e3*d['delta_vs_sat']:+.2f} ({d['t_stat']:+.1f})")
    L.append(f"{lab} & "+" & ".join(row)+r"\\")
L.append(r"\midrule")
row=[f"{1e3*sf[t]['delta']:+.2f} ({sf[t]['t_stat']:+.1f})" for t in TASKS]
L.append(r"one floor shared by all tiers & "+" & ".join(row)+r"\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_cv.tex","\n".join(L))

# ================= APPENDIX: shared floor params =================
lf=fits['lamp3_shared_floor']
L=[r"\begin{tabular}{lccc}", r"\toprule",
   r"Parameter & Haiku (small) & Sonnet (mid) & Opus (large)\\", r"\midrule",
   r"amplitude $a_m$ & "+" & ".join(ci(lf[f'a_{m}']['est'],lf[f'a_{m}']['lo'],lf[f'a_{m}']['hi']) for m in MODELS)+r"\\",
   r"exponent $b_m$ & "+" & ".join(ci(lf[f'b_{m}']['est'],lf[f'b_{m}']['lo'],lf[f'b_{m}']['hi']) for m in MODELS)+r"\\",
   r"\midrule",
   r"shared floor $e_\infty$ & \multicolumn{3}{c}{"+ci(lf['e_inf']['est'],lf['e_inf']['lo'],lf['e_inf']['hi'])+r"}\\",
   r"\bottomrule", r"\end{tabular}"]
w("tab_floor.tex","\n".join(L))

# ================= APPENDIX: controls =================
cot=ctl[ctl.control=='cot_vs_direct'].sort_values(['task','k'])
L=[r"\begin{tabular}{lrrcc}", r"\toprule",
   r"Task & $k$ & $n$ & quality change from chain-of-thought & resolvable\\", r"\midrule"]
for _,r_ in cot.iterrows():
    L.append(f"{r_.task} & {int(r_.k)} & {int(r_.n_paired)} & {ci(r_.gain,r_.gain_lo,r_.gain_hi)} & "
             f"{'yes' if r_.sig=='*' else 'no'}\\\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_cot.tex","\n".join(L))

rec=ctl[ctl.control=='recency_vs_bm25'].sort_values(['task','k'])
L=[r"\begin{tabular}{lrrcc}", r"\toprule",
   r"Task & $k$ & $n$ & quality change from recency instead of BM25 & resolvable\\", r"\midrule"]
for _,r_ in rec.iterrows():
    L.append(f"{r_.task} & {int(r_.k)} & {int(r_.n_paired)} & {ci(r_.delta,r_.delta_lo,r_.delta_hi)} & "
             f"{'yes' if r_.sig=='*' else 'no'}\\\\")
rc=contr[contr.family=='retriever_ceiling']
L.append(r"\midrule")
for _,r_ in rc.iterrows():
    L.append(f"{r_.task} & ceiling & {int(r_.n_paired)} & {ci(r_.delta,r_.delta_lo,r_.delta_hi)} "
             f"\\tiny{{(BM25 {f4(r_.ceiling_bm25)} vs.\\ recency {f4(r_.ceiling_recency)})}} & no\\\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_retriever.tex","\n".join(L))

# ================= APPENDIX: LaMP-3 accuracy vs floor =================
L=[r"\begin{tabular}{rccc}", r"\toprule", r"$k$ & Haiku & Sonnet & Opus\\", r"\midrule"]
for k in KS:
    row=[]
    for mo in MODELS:
        r_=gA[(gA.task=='LaMP-3')&(gA.model==mo)&(gA.k==k)].iloc[0]
        row.append(f4(r_.accuracy))
    L.append(f"{k} & "+" & ".join(row)+r"\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_l3acc.tex","\n".join(L))

# ================= APPENDIX: decision-rule gain fits =================
ffd=dr['fitted_on_full_data']
L=[r"\begin{tabular}{lcccc}", r"\toprule",
   r"Task & $c$ & $d$ & $k^\star(\tau{=}0.02)$ & $k^\star(\tau{=}0.05)$\\", r"\midrule"]
for t in TASKS:
    c,d=ffd[t]['params']
    ks=lambda tau: (c/tau)**(1.0/d)-1.0
    fmt=lambda x: "diverges" if x>1e6 else f"{x:.1f}"
    L.append(f"{t} & {c:.4f} & {d:.4f} & {fmt(ks(0.02))} & {fmt(ks(0.05))}\\\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_gainfit.tex","\n".join(L))

# ================= numbers used in prose, dumped for the record =================
prose={}
for t in TASKS:
    p=nk['model_pooled'][t]; g8=gf[(gf.task==t)&(gf.k==8)].iloc[0]
    st=sub[sub.task==t]; b=st.loc[st.gain_vs_opus_k0.idxmax()]
    prose[t]=dict(total_gain=[p['total_gain'],p['tg_lo'],p['tg_hi']], n_items=p['n_items'],
                  knee=[p['knee'],p['knee_lo'],p['knee_hi']],
                  frac_by_8=[g8.gain_frac,g8.gf_lo,g8.gf_hi],
                  subst=[float(b.gain_vs_opus_k0),float(b.lo),float(b.hi),float(b.p),int(b.k),int(b.n)])
json.dump(prose, open("handoff/prose_numbers.json","w"), indent=1)
print("min frac by k=8 across tasks:", min(prose[t]['frac_by_8'][0] for t in TASKS))
