import os
# Directory holding the analysis outputs (CSV/JSON) that these tables are built from.
RESULTS = os.environ.get("SBS_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))


import pandas as pd, json, numpy as np, os
exec(open("gen_tables.py").read().split("# ================= T1")[0])

# effective_k (transcribed from analysis_report.md s8.2)
EFFK = {0:[("0.00","0"),("0.00","0"),("0.00","0"),("0.00","0")],
        1:[("1.00","0"),("1.00","0"),("1.00","0"),("1.00","0")],
        2:[("2.00","0"),("2.00","0"),("2.00","0"),("2.00","0")],
        4:[("4.00","0"),("4.00","0"),("4.00","0"),("---","---")],
        8:[("8.00","0"),("7.84","8"),("8.00","0"),("8.00","0")],
        16:[("16.00","0"),("13.86","38"),("16.00","0"),("16.00","0")],
        32:[("32.00","0"),("22.11","60"),("32.00","0"),("32.00","0")]}
L=[r"\begin{tabular}{rcccc}", r"\toprule",
   r"$k$ & "+" & ".join(TASKS)+r"\\", r"\midrule"]
for k in KS:
    L.append(f"{k} & "+" & ".join((f"{a} ({b}\\%)" if b!="---" else "---") for a,b in EFFK[k])+r"\\")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_effk.tex","\n".join(L))

# per-stratum gains, exploratory (stratum_gains.csv)
sg=pd.read_csv(os.path.join(RESULTS, "stratum_gains.csv"))
L=[r"\begin{tabular}{llcccc}", r"\toprule",
   r"Stratum & Tier & "+" & ".join(TASKS)+r"\\", r"\midrule"]
for st in ['short','medium','long']:
    for mo in MODELS:
        row=[]
        for t in TASKS:
            r_=sg[(sg.task==t)&(sg.stratum==st)&(sg.model_s==mo)]
            row.append("---" if len(r_)==0 else ci(r_.iloc[0].gain, r_.iloc[0].lo, r_.iloc[0].hi))
        n=sg[(sg.stratum==st)&(sg.model_s==mo)].n.min()
        L.append(f"{st if mo=='haiku' else ''} & {mo} & "+" & ".join(row)+r"\\")
    L.append(r"\addlinespace")
L+=[r"\bottomrule", r"\end{tabular}"]
w("tab_stratum.tex","\n".join(L))

# parse failures across all grids
pf=cell[cell.parse_failures>0][['grid','task','model','k','retriever','variant','n','parse_failures','parse_failure_rate']]
L=[r"\begin{tabular}{llrllrr}", r"\toprule",
   r"Grid & Task & $k$ & retriever & variant & $n$ & parse failures (rate)\\", r"\midrule"]
for _,r_ in pf.iterrows():
    L.append(f"{r_.grid} & {r_.task} & {int(r_.k)} & {r_.retriever} & {r_.variant} & {int(r_.n)} & "
             f"{int(r_.parse_failures)} ({100*r_.parse_failure_rate:.1f}\\%)\\\\")
L+=[r"\midrule", f"\\multicolumn{{6}}{{l}}{{total, all 106 cells}} & {int(cell.parse_failures.sum())} "
   f"({100*cell.parse_failures.sum()/11197:.2f}\\% of 11{{,}}197 usable calls)\\\\",
   r"\bottomrule", r"\end{tabular}"]
w("tab_parsefail.tex","\n".join(L))
print("done")
