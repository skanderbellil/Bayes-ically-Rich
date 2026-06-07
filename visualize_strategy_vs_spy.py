#!/usr/bin/env python3
"""Visualize final strategy improvements vs SPY baseline."""

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")

d = pd.read_csv('results/pead/daily_megacap_prices.csv', skiprows=[1,2], index_col=0, parse_dates=True)
d = d.apply(pd.to_numeric, errors='coerce').sort_index(); didx=d.index

sig = pd.read_csv('results/pead/signal_panel.csv', parse_dates=['announcement_date'])
mega = sig[sig['market_cap']=='Mega Cap'].dropna(subset=['sue','ret_t1','ret_t21','ret_t63']).copy()
s = mega[mega['sue']>=mega['sue'].quantile(0.75)].copy()
s['announcement_date']=s['announcement_date'].dt.tz_localize(None)
s['drift_21d']=s['ret_t21']-s['ret_t1']; s['drift_63d']=s['ret_t63']-s['ret_t1']
s['win_63d']=(s['drift_63d']>0).astype(int); s['q']=s['announcement_date'].dt.to_period('Q')
s=s.sort_values('announcement_date')

spy = pd.read_csv("results/pead/spy_quarterly.csv", index_col=0, parse_dates=True)['spy_qret'].sort_index()

def path_for(ticker, adate):
    if ticker not in d.columns: return None
    pos = didx.searchsorted(adate)
    if pos>=len(didx) or pos+64>=len(didx): return None
    if abs((didx[pos]-adate).days) > 5: return None
    px = d[ticker].iloc[pos+1:pos+1+63].values
    if len(px)<63 or np.isnan(px).any(): return None
    return (px/px[0]-1.0)*100.0

covered=[(r,path_for(r['ticker'],r['announcement_date'])) for _,r in s.iterrows()]
covered=[(r,p) for r,p in covered if p is not None]

def model_for(q):
    prior=s[s['q']<q]
    return LogisticRegression(max_iter=1000).fit(prior[['drift_21d']],prior['win_63d']) if len(prior)>=30 else None

def sim(stop=None, trend=False):
    byq={}
    for r,p in covered:
        m=model_for(r['q'])
        if m is None: continue
        prob=m.predict_proba([[r['drift_21d']]])[0,1]
        hold_to=63 if prob>0.30 else 21
        if stop is None: ret=p[hold_to-1]/100
        else:
            seg=p[:hold_to]; hit=np.where(seg<=stop)[0]
            ret=stop/100 if len(hit)>0 else p[hold_to-1]/100
        byq.setdefault(str(r['q']),[]).append(ret)
    qrets=pd.Series({pd.Period(q,'Q').to_timestamp():np.mean(v) for q,v in byq.items()}).sort_index()
    if trend:
        j=pd.DataFrame({'strat':qrets,'spy':spy}).dropna()
        prev_spy=j['spy'].shift(1).fillna(0.01)
        qrets=j['strat'].where(prev_spy>0,0)
    return qrets

base = sim(None, False)
stop20 = sim(-20, False)
combined = sim(-20, True)

# Align all to SPY
j = pd.DataFrame({'Base':base,'Stop-20%':stop20,'Stop+Trend':combined,'SPY':spy}).dropna()

# Metrics
def metrics(x):
    x=np.asarray(x); ann=np.prod(1+x)**(4/len(x))-1; sh=x.mean()/x.std()*2
    c=np.cumprod(1+x); dd=((c-np.maximum.accumulate(c))/np.maximum.accumulate(c)).min()
    return ann,sh,dd

# Build visualizations
fig = plt.figure(figsize=(18,12))
gs = fig.add_gridspec(3,3, hspace=0.35, wspace=0.3)

# 1. Cumulative wealth (log)
ax1 = fig.add_subplot(gs[0,:2])
colors = {'Base':'#1f77b4', 'Stop-20%':'#ff7f0e', 'Stop+Trend':'#2ca02c', 'SPY':'#d62728'}
for col in j.columns:
    w = np.cumprod(1+j[col].values)
    ax1.plot(j.index, w, linewidth=2.5, label=col, color=colors[col], marker='o', markersize=3, alpha=0.8)
ax1.set_yscale('log'); ax1.set_ylabel('Wealth (log scale)', fontweight='bold')
ax1.set_title('Cumulative Growth: Dynamic Hold Variants vs SPY', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, which='both'); ax1.legend(fontsize=11, loc='upper left')

# 2. Performance table
ax2 = fig.add_subplot(gs[0,2])
ax2.axis('off')
table_data = []
for col in j.columns:
    ann, sh, dd = metrics(j[col].values)
    table_data.append([col, f'{ann:+.1%}', f'{sh:.2f}', f'{dd:.0%}'])
table = ax2.table(cellText=table_data, colLabels=['Strategy','Annual','Sharpe','MaxDD'],
                  cellLoc='center', loc='center', colWidths=[0.22]*4)
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,2.2)
for i,col in enumerate(j.columns):
    table[(1+i,0)].set_facecolor(colors[col]); table[(1+i,0)].set_alpha(0.3)
ax2.set_title('Performance Summary', fontweight='bold', fontsize=10)

# 3. Quarterly returns heatmap
ax3 = fig.add_subplot(gs[1,:2])
ret_mat = j.iloc[:30].values*100  # first 30 qtrs
im = ax3.imshow(ret_mat.T, aspect='auto', cmap='RdYlGn', vmin=-15, vmax=25)
ax3.set_yticks(range(4)); ax3.set_yticklabels(j.columns)
ax3.set_xlabel('Quarter (2015-2020)', fontweight='bold')
ax3.set_title('Quarterly Returns Heatmap (First 30 Quarters)', fontweight='bold')
plt.colorbar(im, ax=ax3, label='Return (%)')

# 4. Rolling Sharpe (4-quarter window)
ax4 = fig.add_subplot(gs[1,2])
for col in j.columns:
    rolling_sh = []
    for i in range(len(j)-3):
        ret_window = j[col].iloc[i:i+4].values
        sh = ret_window.mean()/ret_window.std()*2
        rolling_sh.append(sh)
    ax4.plot(j.index[3:], rolling_sh, linewidth=2, label=col, color=colors[col], alpha=0.8)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Sharpe', fontweight='bold')
ax4.set_title('Rolling Sharpe Ratio (4Q)', fontweight='bold')
ax4.grid(True, alpha=0.3); ax4.legend(fontsize=9)

# 5. Distribution of quarterly returns
ax5 = fig.add_subplot(gs[2,0])
for col in j.columns:
    ax5.hist(j[col].values*100, bins=15, alpha=0.5, label=col, color=colors[col])
ax5.set_xlabel('Quarterly Return (%)', fontweight='bold')
ax5.set_ylabel('Frequency', fontweight='bold')
ax5.set_title('Return Distribution', fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3, axis='y')

# 6. Drawdown comparison
ax6 = fig.add_subplot(gs[2,1])
for col in j.columns:
    c = np.cumprod(1+j[col].values)
    dd_path = (c - np.maximum.accumulate(c))/np.maximum.accumulate(c)*100
    ax6.plot(j.index, dd_path, linewidth=2, label=col, color=colors[col], alpha=0.8)
ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax6.set_ylabel('Drawdown (%)', fontweight='bold')
ax6.set_title('Underwater Plot: Peak-to-Trough Drawdown', fontweight='bold')
ax6.grid(True, alpha=0.3); ax6.legend(fontsize=9)

# 7. Excess return (strategy - SPY)
ax7 = fig.add_subplot(gs[2,2])
excess_base = (np.cumprod(1+j['Base'].values) - np.cumprod(1+j['SPY'].values))*100
excess_stop = (np.cumprod(1+j['Stop-20%'].values) - np.cumprod(1+j['SPY'].values))*100
excess_comb = (np.cumprod(1+j['Stop+Trend'].values) - np.cumprod(1+j['SPY'].values))*100
ax7.fill_between(j.index, 0, excess_base, alpha=0.4, label='Base', color='#1f77b4')
ax7.fill_between(j.index, 0, excess_stop, alpha=0.4, label='Stop-20%', color='#ff7f0e')
ax7.fill_between(j.index, 0, excess_comb, alpha=0.4, label='Stop+Trend', color='#2ca02c')
ax7.axhline(0, color='black', linestyle='-', linewidth=1)
ax7.set_ylabel('Cumulative Alpha vs SPY (%)', fontweight='bold')
ax7.set_title('Alpha Accumulation', fontweight='bold')
ax7.grid(True, alpha=0.3); ax7.legend(fontsize=9)

fig.suptitle('Dynamic Hold Strategy Improvements: Full Comparative Analysis vs SPY',
             fontweight='bold', fontsize=14)
fig.savefig('results/pead/strategy_improvements_vs_spy.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/pead/strategy_improvements_vs_spy.png\n")

# Print summary
print("="*85)
print("STRATEGY PERFORMANCE SUMMARY (2015-2026, 45 quarters)")
print("="*85)
print()
for col in j.columns:
    ann, sh, dd = metrics(j[col].values)
    wealth = np.prod(1+j[col].values)
    print(f"{col:15s}: {ann:+6.1%} annual  Sharpe {sh:4.2f}  maxDD {dd:5.0%}  Cumulative {wealth:+6.1%}")
print()
print("vs SPY alpha:")
spy_ann, spy_sh, spy_dd = metrics(j['SPY'].values)
for col in ['Base','Stop-20%','Stop+Trend']:
    ann, sh, dd = metrics(j[col].values)
    print(f"  {col:20s}: {(ann-spy_ann)*100:+5.1f}pp annual, {sh-spy_sh:+4.2f} Sharpe")
print("="*85)
