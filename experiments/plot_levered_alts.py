#!/usr/bin/env python3
"""Charts: leveraged equity + managed futures vs QQQ."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_levered_alts import DATA, RESULTS_DIR, MF, rebal, stats

m=(1+pd.read_csv(DATA/"levered_stacked.csv",index_col=0,parse_dates=True).pct_change()).resample("ME").apply(lambda x:x.prod()-1)
m=m.assign(MFB=m[MF].mean(axis=1,skipna=True))
win=m.index>="2020-01-01"
books={"QQQ":m["QQQ"].loc[win],"QLD (2x QQQ)":m["QLD"].loc[win],
       "QLD 60% + MF 40%":rebal({"QLD":0.6,"MFB":0.4},m).loc[win],
       "QLD 70% + MF 30%":rebal({"QLD":0.7,"MFB":0.3},m).loc[win]}
cols={"QQQ":"#888","QLD (2x QQQ)":"#aaa","QLD 60% + MF 40%":"#B71C1C","QLD 70% + MF 30%":"#1565C0"}
fig,axes=plt.subplots(1,2,figsize=(14,5.4))
ax=axes[0]
for n,r in books.items():
    eq=(1+r).cumprod(); ax.plot(eq.index,eq.values,color=cols[n],lw=2.4 if "60" in n else 1.7,
        label=f"{n}  (CAGR {stats(r)['cagr']:+.0%}, maxDD {stats(r)['maxdd']:.0%})")
ax.set_yscale("log"); ax.axvspan(pd.Timestamp("2022-01-01"),pd.Timestamp("2022-12-31"),color="orange",alpha=0.13)
ax.set_ylabel("growth of $1 (log)"); ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.3,which="both")
ax.set_title("Leveraged equity + managed futures vs QQQ (2020-26)\n2022 shaded — MF cushioned QLD's crash",fontweight="bold",fontsize=11)
ax=axes[1]
pts={"SPY":m["SPY"],"QQQ":m["QQQ"],"QLD":m["QLD"],"QLD60+MF40":rebal({"QLD":0.6,"MFB":0.4},m),
     "QLD70+MF30":rebal({"QLD":0.7,"MFB":0.3},m),"SSO60+MF40":rebal({"SSO":0.6,"MFB":0.4},m)}
for n,r in pts.items():
    s=stats(r.loc[win]); c="#B71C1C" if "MF" in n else "#444"
    ax.scatter(abs(s['maxdd'])*100,s['cagr']*100,s=90,color=c,zorder=3)
    ax.annotate(n,(abs(s['maxdd'])*100,s['cagr']*100),fontsize=9,xytext=(6,4),textcoords="offset points")
ax.set_xlabel("max drawdown % (lower = better →)"); ax.set_ylabel("CAGR %")
ax.invert_xaxis(); ax.grid(alpha=0.3)
ax.set_title("Return vs drawdown frontier\nMF books (red) sit up-and-left of QQQ",fontweight="bold",fontsize=11)
fig.tight_layout(); out=RESULTS_DIR/"levered_alts_dashboard.png"; fig.savefig(out,dpi=140,bbox_inches="tight"); print("saved",out)
