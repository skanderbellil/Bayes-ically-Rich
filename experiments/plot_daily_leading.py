#!/usr/bin/env python3
"""Charts: daily fast-vol-managed leveraged equity vs QQQ."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
DATA=_bootstrap.ROOT/"datasets"; RES=_bootstrap.ROOT/"results"; ANN=252
px=pd.read_csv(DATA/"levered_stacked.csv",index_col=0,parse_dates=True)
vix=pd.read_csv(DATA/"vix_termstructure.csv",index_col=0,parse_dates=True)
df=px.join(vix,how="inner").sort_index().loc["2011-01-01":]
rets=df.pct_change(); qqq=rets["QQQ"]; qld=rets["QLD"]; rf_d=0.04/ANN
mf=rets[["DBMF","KMLM","CTA"]].mean(axis=1,skipna=True).fillna(rf_d)
rv=qqq.rolling(20).std()*np.sqrt(ANN); on=(rv<rv.rolling(252).median()*1.3)
pos=on.shift(1).fillna(False); sig=pd.Series(np.where(pos,qld,mf),index=qld.index)
sw=pos.ne(pos.shift(1)).fillna(False); sig=(sig-sw*10/1e4)
def SR(r): r=r.dropna(); return (r.mean()*ANN-0.04)/(r.std()*np.sqrt(ANN))
def DD(r): eq=(1+r.dropna()).cumprod(); return (eq/eq.cummax()-1).min()
fig,axes=plt.subplots(2,1,figsize=(13,9),height_ratios=[2,1]); 
ax=axes[0]
for r,n,c,lw in [(qqq,"QQQ buy & hold","#888",1.6),(qld,"QLD (2x QQQ)","#ccc",1.2),
                 (sig,"DAILY fast-vol QLD/MF (the signal)","#B71C1C",2.4)]:
    eq=(1+r.reindex(sig.index).fillna(0)).cumprod()
    ax.plot(eq.index,eq.values,color=c,lw=lw,label=f"{n}  (CAGR {eq.iloc[-1]**(ANN/len(eq))-1:+.0%}, Sharpe {SR(r):.2f}, maxDD {DD(r):.0%})")
ax.set_yscale("log");ax.set_ylabel("growth of $1 (log)");ax.legend(loc="upper left",fontsize=10);ax.grid(alpha=0.3,which="both")
for span in [("2020-02-15","2020-04-15"),("2022-01-01","2022-12-31")]:
    ax.axvspan(pd.Timestamp(span[0]),pd.Timestamp(span[1]),color="orange",alpha=0.12)
ax.set_title("Daily fast-vol de-risking of leveraged equity vs QQQ (2011-2026)\nshaded: COVID-2020 & 2022 — the signal cut exposure as vol spiked",fontweight="bold",fontsize=12)
ax=axes[1]
for r,n,c in [(qqq,"QQQ","#888"),(sig,"fast-vol QLD/MF","#B71C1C")]:
    eq=(1+r.reindex(sig.index).fillna(0)).cumprod();dd=eq/eq.cummax()-1
    ax.fill_between(dd.index,dd.values,0,color=c,alpha=0.35,label=n)
ax.set_ylabel("drawdown");ax.legend(loc="lower left",fontsize=10);ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
ax.set_title("Drawdowns",fontsize=11)
fig.tight_layout();out=RES/"daily_leading_dashboard.png";fig.savefig(out,dpi=140,bbox_inches="tight");print("saved",out)
