"""Experiment 2 supplement: SVM with an extended hyperparameter grid (incl. linear kernel), same datasets/splits."""
import json, sys, time, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from experiment2 import generate_hard
GRID = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1e3, 1e4, 1e5], 'gamma': [1e-7, 1e-6, 1e-5, 1e-4, 'scale']},
        {'kernel': ['linear'], 'C': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]}]
out = []
for eff in [float(a) for a in sys.argv[1:]]:
    for s in range(1, 11):
        X, y, yt = generate_hard(effect=eff, seed=1000 + s)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=s)
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        g = GridSearchCV(SVC(), GRID, cv=StratifiedKFold(5, shuffle=True, random_state=s), n_jobs=-1).fit(Xtr, y[tr])
        p = g.predict(Xte)
        out.append(dict(effect=eff, seed=s, test_acc=accuracy_score(y[te], p), acc_true=accuracy_score(yt[te], p),
                        train_acc=accuracy_score(y[tr], g.predict(Xtr)), best={k: str(v) for k, v in g.best_params_.items()}))
        print(eff, s, round(out[-1]['test_acc'] * 100, 1), out[-1]['best'], flush=True)
json.dump(out, open('results_svm_ext.json', 'w'), indent=1)
print('DONE')
