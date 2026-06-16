#!/usr/bin/env python3
"""Graph + honest t>3 reckoning for the exploitable-IR claim."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
DATA=_bootstrap.ROOT/"datasets"; RES=_bootstrap.ROOT/"results"; ANN=252
src=open(_bootstrap.ROOT/"experiments"/"run_retail_ir.py").read().split("def main()")[0]
exec(src.replace("import _bootstrap  # noqa: F401",""))
px=pd.read_csv(DATA/"levered_stacked.csv",index_col=0,parse_dates=True).loc["2011-01-01":]
rets=px.pct_change(); spy,qqq=rets["SPY"],rets["QQQ"]
book=build(px,0.25,0.30).rename("book")   # best-IR mix
idx=book.index

def appraisal(r,b):
    df=pd.concat([r,b],axis=1,keys=["r","b"]).dropna()
    X=np.column_stack([np.ones(len(df)),df["b"].values])
    coef,*_=np.linalg.lstsq(X,df["r"].values,rcond=None)
    resid=df["r"].values-X@coef
    se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*resid.var(ddof=2))
    return coef[0]*ANN, coef[0]/se[0], coef[0]*ANN/(resid.std()*np.sqrt(ANN)+1e-12)

fig=plt.figure(figsize=(13,10)); gs=fig.add_gridspec(3,1,hspace=0.32,height_ratios=[2,1,1])
# growth
ax=fig.add_subplot(gs[0])
for r,n,c,lw in [(spy,"SPY","#888",1.6),(qqq,"QQQ","#bbb",1.3),(book,"vol-QLD + 30% MF","#B71C1C",2.4)]:
    eq=(1+r.reindex(idx).fillna(0)).cumprod(); ax.plot(eq.index,eq.values,color=c,lw=lw,label=f"{n} (${eq.iloc[-1]:.1f})")
ax.set_yscale("log");ax.set_ylabel("growth of $1 (log)");ax.legend(loc="upper left");ax.grid(alpha=0.3,which="both")
ax.set_title("Exploitable-IR book vs SPY/QQQ — and the honest significance reckoning",fontweight="bold",fontsize=13)
# rolling 3y appraisal IR
ax=fig.add_subplot(gs[1])
roll=[]
for i in range(756,len(idx)):
    w=slice(idx[i-756],idx[i]); _,_,ir=appraisal(book.loc[w],spy.loc[w]); roll.append((idx[i],ir))
rr=pd.Series(dict(roll))
ax.plot(rr.index,rr.values,color="#1565C0",lw=1.6);ax.axhline(0.5,color="g",ls="--",lw=0.8,label="0.5 'good'")
ax.axhline(0,color="k",lw=0.6);ax.fill_between(rr.index,rr.values,0,where=(rr<0),color="red",alpha=0.15)
ax.set_ylabel("rolling 3y\nappraisal IR vs SPY");ax.legend(fontsize=8);ax.grid(alpha=0.3)
ax.set_title("Rolling 3-year appraisal IR — positive on average but it swings through ZERO",fontsize=10.5)
# alpha t-stat bar vs the t>3 bar
ax=fig.add_subplot(gs[2])
a_full,t_full,_=appraisal(book,spy); a_oos,t_oos,_=appraisal(book.loc["2017":],spy.loc["2017":])
bars=["full 2011-26","OOS 2017-26"]; ts=[t_full,t_oos]
ax.bar(bars,ts,0.5,color=["#90CAF9","#90CAF9"])
ax.axhline(2.0,color="orange",ls="--",lw=1.2,label="t=2 (classical)")
ax.axhline(3.0,color="red",ls="--",lw=1.5,label="t=3 (Harvey / OUR bar)")
for i,t in enumerate(ts): ax.text(i,t+0.05,f"{t:.2f}",ha="center",fontsize=10)
ax.set_ylabel("t(alpha) vs SPY");ax.legend(fontsize=9);ax.grid(alpha=0.3,axis="y");ax.set_ylim(0,3.5)
ax.set_title("Alpha t-stat FAILS our own t>3 bar (and ignores this session's multiple testing)",fontsize=10.5,color="#B71C1C")
fig.tight_layout();out=RES/"retail_ir_dashboard.png";fig.savefig(out,dpi=140,bbox_inches="tight");print("saved",out)
