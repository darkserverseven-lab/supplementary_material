import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9})
runs = []
for f in sorted(glob.glob('results_exp2_*.json')): runs += json.load(open(f))['runs']
ext = json.load(open('results_svm_ext.json'))
effs = sorted({r['effect'] for r in runs})
def stat(vals): v=np.array(vals)*100; return v.mean(), v.std(ddof=1)
series = {'Random Forest': ('Random Forest', 'o', '-'), 'SVM (RBF)': ('SVM, original grid', 's', ':'),
          'Neural Network (MLP)': ('MLP', '^', '--')}
fig, ax = plt.subplots(figsize=(4.8, 3.0))
for k, (lab, mk, ls) in series.items():
    m = [stat([r[k]['test_acc'] for r in runs if r['effect'] == e]) for e in effs]
    ax.errorbar(effs, [a for a, _ in m], yerr=[b for _, b in m], marker=mk, ls=ls, color='black', capsize=2, label=lab, ms=4, lw=1)
m = [stat([r['test_acc'] for r in ext if r['effect'] == e]) for e in effs]
ax.errorbar(effs, [a for a, _ in m], yerr=[b for _, b in m], marker='D', ls='-.', color='0.5', capsize=2, label='SVM, extended grid', ms=4, lw=1)
ax.axhline(39.7, color='0.6', lw=0.8); ax.text(2.05, 36.0, 'majority-class baseline (39.7%)', fontsize=7, color='0.4')
ax.set_xlabel('Marker mean shift δ'); ax.set_ylabel('Test accuracy (%)'); ax.set_xticks(effs)
ax.grid(color='0.92'); ax.legend(fontsize=7, frameon=False, loc='upper left'); fig.tight_layout(); fig.savefig('fig3.png', dpi=300)
