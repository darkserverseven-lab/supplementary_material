"""Reproducible pipeline: synthetic breast-cancer subtype expression data, RF / SVM / MLP comparison.
Produces every number reported in the article (results.json)."""
import os, json, time, sys, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
import tensorflow as tf

SUBTYPES = ['Luminal_A', 'Luminal_B', 'HER2_enriched', 'Basal_like']
MARKERS = {'Luminal_A': range(0, 25), 'Luminal_B': range(25, 50),
           'HER2_enriched': range(50, 75), 'Basal_like': range(75, 100)}

def generate_synthetic_expression(n_samples=2000, n_genes=500, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(SUBTYPES, size=n_samples, p=[0.40, 0.25, 0.15, 0.20])
    base_cov = np.eye(n_genes) * 1.0
    for b in range(0, n_genes, 25):
        base_cov[b:min(b + 25, n_genes), b:min(b + 25, n_genes)] += 0.3
    X = rng.multivariate_normal(mean=np.zeros(n_genes), cov=base_cov, size=n_samples)
    for i, label in enumerate(labels):
        for g in MARKERS[label]:
            X[i, g] += rng.normal(2.5, 0.5)
    true_labels = labels.copy()                      # ground truth before label noise
    flip = rng.random(n_samples) < 0.05
    labels[flip] = rng.choice(SUBTYPES, size=flip.sum())
    return X, labels, true_labels

def build_mlp(n_in, seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n_in,)),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

class MLPWrap:
    def __init__(self, seed): self.seed = seed
    def fit(self, X, y):
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        self.model = build_mlp(X.shape[1], self.seed)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = self.model.fit(Xt, yt, validation_data=(Xv, yv), epochs=50, batch_size=32, callbacks=[es], verbose=0)
        self.epochs_run = len(h.history['loss'])
        return self
    def predict_proba(self, X): return self.model.predict(X, verbose=0, batch_size=600)
    def predict(self, X): return self.predict_proba(X).argmax(1)

SVM_GRID = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 1e-4, 1e-3, 1e-2]}

def make_models(seed):
    return {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, max_features='sqrt',
                                                random_state=seed, n_jobs=-1),
        'SVM (RBF)': GridSearchCV(SVC(kernel='rbf', probability=True, random_state=seed), SVM_GRID,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1),
        'Neural Network (MLP)': MLPWrap(seed)}

def run(X, y, ytrue, seed, detailed=False):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    out = {}
    for name, m in make_models(seed).items():
        t0 = time.perf_counter(); m.fit(Xtr, y[tr]); fit_t = time.perf_counter() - t0
        if name == 'SVM (RBF)':
            best = m.best_params_
            final = SVC(kernel='rbf', probability=True, random_state=seed, **best)
            t0 = time.perf_counter(); final.fit(Xtr, y[tr]); final_t = time.perf_counter() - t0
            grid_t, fit_t, m = fit_t, final_t, final
        t0 = time.perf_counter(); proba = m.predict_proba(Xte); pred = m.predict(Xte); inf_t = time.perf_counter() - t0
        tr_acc = accuracy_score(y[tr], m.predict(Xtr)); te_acc = accuracy_score(y[te], pred)
        r = {'train_acc': tr_acc, 'test_acc': te_acc, 'test_acc_vs_true': accuracy_score(ytrue[te], pred)}
        if detailed:
            p, rc, f, s = precision_recall_fscore_support(y[te], pred, average='macro')
            pc, rcc, fc, sc_ = precision_recall_fscore_support(y[te], pred, labels=[0,1,2,3])
            r.update(precision=p, recall=rc, f1=f, auc=roc_auc_score(y[te], proba, multi_class='ovr', average='macro'),
                     confusion=confusion_matrix(y[te], pred, labels=[0,1,2,3]).tolist(),
                     per_class={SUBTYPES[k]: dict(precision=pc[k], recall=rcc[k], f1=fc[k], support=int(sc_[k])) for k in range(4)},
                     train_time_s=fit_t, infer_ms_per_sample=1000 * inf_t / len(te))
            if name == 'SVM (RBF)': r.update(best_params=best, grid_time_s=grid_t)
            if name == 'Random Forest':
                top = np.argsort(m.feature_importances_)[::-1][:100]
                r['markers_in_top100'] = int(np.sum(top < 100))
                r['markers_in_top100_by_subtype'] = {s: int(np.sum((top >= MARKERS[s].start) & (top < MARKERS[s].stop))) for s in SUBTYPES}
            if name == 'Neural Network (MLP)':
                r['epochs_run'] = m.epochs_run
                r['n_params'] = int(m.model.count_params())
        out[name] = r
    return out

if __name__ == '__main__':
    X, lab, true_lab = generate_synthetic_expression()
    enc = {s: i for i, s in enumerate(SUBTYPES)}
    y = np.array([enc[s] for s in lab]); ytrue = np.array([enc[s] for s in true_lab])
    res = {'env': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'tensorflow': tf.__version__,
                   'numpy': np.__version__, 'cpu_count': os.cpu_count()},
           'data': {'class_counts': {s: int((y == enc[s]).sum()) for s in SUBTYPES},
                    'effective_label_noise': float((y != ytrue).mean())}}
    res['main'] = run(X, y, ytrue, seed=42, detailed=True)
    stab = {}
    for s in range(1, 11):
        r = run(X, y, ytrue, seed=s)
        for k, v in r.items(): stab.setdefault(k, []).append(v)
        print('seed', s, {k: round(v['test_acc'] * 100, 1) for k, v in r.items()}, flush=True)
    res['stability'] = {k: {'test_acc': [x['test_acc'] for x in v], 'gap': [x['train_acc'] - x['test_acc'] for x in v]} for k, v in stab.items()}
    json.dump(res, open('results.json', 'w'), indent=1, default=float)
    print(json.dumps(res['main'], indent=1, default=float)[:6000]); print(res['data'])
