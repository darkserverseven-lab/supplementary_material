import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pipeline import generate_synthetic_expression, SUBTYPES
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9})
X, lab, _ = generate_synthetic_expression()
names = ['Luminal A','Luminal B','HER2-enriched','Basal-like']
M = np.array([X[lab==s,:125].mean(0) for s in SUBTYPES])
fig, ax = plt.subplots(1,2, figsize=(7.2,2.6), gridspec_kw={'width_ratios':[2.2,1]})
im = ax[0].imshow(M, aspect='auto', cmap='viridis')
ax[0].set_yticks(range(4)); ax[0].set_yticklabels(names); ax[0].set_xlabel('Gene index (first 125 of 500)')
ax[0].set_title('(a) Mean expression by subtype'); fig.colorbar(im, ax=ax[0], fraction=0.04)
C = np.corrcoef(X[:, :125], rowvar=False)
im2 = ax[1].imshow(C, cmap='Greys', vmin=0, vmax=1); ax[1].set_title('(b) Gene–gene correlation')
ax[1].set_xlabel('Gene index'); ax[1].set_ylabel('Gene index'); fig.colorbar(im2, ax=ax[1], fraction=0.046)
fig.tight_layout(); fig.savefig('fig1.png', dpi=300)
r = json.load(open('results.json'))['stability']
fig, ax = plt.subplots(figsize=(4.6,2.6))
keys = list(r.keys()); data = [np.array(r[k]['test_acc'])*100 for k in keys]
ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color='black'))
for i, d in enumerate(data): ax.scatter(np.full(len(d), i+1)+np.linspace(-.12,.12,len(d)), d, s=10, color='0.35', zorder=3)
ax.set_xticks(range(1,len(keys)+1)); ax.set_xticklabels(['Random Forest','SVM (RBF)','MLP'])
ax.set_ylabel('Test accuracy (%)'); ax.grid(axis='y', color='0.9'); fig.tight_layout(); fig.savefig('fig2.png', dpi=300)
