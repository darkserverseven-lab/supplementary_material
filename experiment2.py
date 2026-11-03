"""Experiment 2: harder p >> n setting (p=5000, n=300), weaker and partially overlapping marker signal.
Ten independently generated datasets per effect size. Writes results_exp2.json."""
import os, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pipeline import SUBTYPES, make_models

MARKERS2 = {'Luminal_A': range(0, 25), 'Luminal_B': range(15, 40),      # 10 genes shared by Luminal A and B
            'HER2_enriched': range(50, 75), 'Basal_like': range(75, 100)}
N_MARKERS = len(set().union(*[set(r) for r in MARKERS2.values()]))       # 90 unique marker genes

def generate_hard(n_samples=300, n_genes=5000, effect=1.0, seed=0, block=25, rho=0.3, noise=0.05):
    rng = np.random.default_rng(seed)
    labels = rng.choice(SUBTYPES, size=n_samples, p=[0.40, 0.25, 0.15, 0.20])
    # Same covariance as Experiment 1 (I + 0.3 within 25-gene blocks), generated via a shared block factor
    Z = rng.standard_normal((n_samples, n_genes))
    F = rng.standard_normal((n_samples, n_genes // block))
    X = Z + np.sqrt(rho) * np.repeat(F, block, axis=1)
    for i, label in enumerate(labels):
        idx = np.array(MARKERS2[label])
        X[i, idx] += rng.normal(effect, 0.2 * effect, size=len(idx))
    true_labels = labels.copy()
    flip = rng.random(n_samples) < noise
    labels[flip] = rng.choice(SUBTYPES, size=flip.sum())
    enc = {s: i for i, s in enumerate(SUBTYPES)}
    return X, np.array([enc[s] for s in labels]), np.array([enc[s] for s in true_labels])

import sys

if __name__ == '__main__':
    EFFECTS = [float(a) for a in sys.argv[1:]] or [0.5, 0.75, 1.0, 1.5]
    OUT = 'results_exp2.json' if len(sys.argv) == 1 else 'results_exp2_' + '_'.join(sys.argv[1:]) + '.json'
    res = {'design': dict(n_samples=300, n_genes=5000, effects=EFFECTS, reps=10, n_unique_markers=N_MARKERS), 'runs': []}
    for eff in EFFECTS:
        for s in range(1, 11):
            X, y, yt = generate_hard(effect=eff, seed=1000 + s)
            tr, te = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=s)
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            row = {'effect': eff, 'seed': s}
            for name, m in make_models(s).items():
                t0 = time.perf_counter(); m.fit(Xtr, y[tr]); t = time.perf_counter() - t0
                p = m.predict(Xte)
                r = dict(test_acc=accuracy_score(y[te], p), acc_true=accuracy_score(yt[te], p),
                         train_acc=accuracy_score(y[tr], m.predict(Xtr)), f1=f1_score(y[te], p, average='macro'),
                         time_s=t, cm_true=confusion_matrix(yt[te], p, labels=[0, 1, 2, 3]).tolist())
                if name == 'Random Forest':
                    top = np.argsort(m.feature_importances_)[::-1][:N_MARKERS]
                    r['markers_in_top'] = int(np.sum(top < 100) - np.sum((top >= 40) & (top < 50)))
                if name == 'SVM (RBF)': r['best_params'] = m.best_params_
                row[name] = r
            res['runs'].append(row)
            print(eff, s, {k: round(v['test_acc'] * 100, 1) for k, v in row.items() if isinstance(v, dict)}, flush=True)
            json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print('DONE')
