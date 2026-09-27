"""Seção 6 do relatório: parâmetros das regras por busca automática (Optuna, TPE), só na validação.

    python full_gcn/search.py --trials 30 --workers 3     # busca (teste só registrado)
    python full_gcn/search.py --report

Objetivo: maior cobertura na validação com acerto >= 90% (abaixo disso, penalidade proporcional).
Parâmetros (os da tabela da Seção 6): K, M, sigma_min, tau_full, m_full, tau_GCN, m_GCN.
tau_GCN numa faixa própria (0,85-0,999): no espaço de 5 dimensões os cossenos entre vizinhos são altos.
Visão GCN = Z.npy de gcn.py (--z para escolher outro).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_data as data  # noqa: E402
import rules  # noqa: E402

HERE = Path(__file__).resolve().parent
STORAGE = f"sqlite:///{HERE / 'runs' / 'optuna.db'}"
TARGET = 0.90
K_MAX = 200
_views = {}


def views(z_path):
    if _views:
        return _views
    x, meta, edges0, img_y, img_split = data.region_graph("lit2")
    node_img = meta[:, 0]
    orig = meta[:, 1] == 0
    nodes = np.flatnonzero(orig)
    e = edges0[orig[edges0[:, 0]] & orig[edges0[:, 1]]]
    adj = {int(v): set() for v in nodes}
    for u, v in e:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    Z = np.load(z_path)
    assert len(Z) == len(x), "Z e o grafo da visão full precisam ter os mesmos vértices"
    for name, vec in (("full", x), ("gcn", Z)):
        nb, sm = rules.knn(vec, nodes, node_img, K_MAX)
        _views[name] = (nb, sm)
    _views.update(nodes=nodes, node_img=node_img, adj=adj, img_y=img_y, img_split=img_split)
    return _views


def objective(trial, z_path):
    V = views(z_path)
    K = trial.suggest_int("K", 20, K_MAX, log=True)
    M = trial.suggest_int("M", 1, 16, log=True)
    sigma_min = trial.suggest_int("sigma_min", 1, 5)
    preds = {}
    for name, tau_rng in (("full", (0.3, 0.95)), ("gcn", (0.85, 0.999))):
        tau = trial.suggest_float(f"tau_{name}", *tau_rng)
        m = trial.suggest_int(f"m_{name}", 1, 30, log=True)
        nb, sm = V[name]
        t0 = time.time()
        pats = rules.mine(V["nodes"], V["node_img"], V["img_split"], V["adj"], nb[:, :K], sm[:, :K], tau, sigma_min, M)
        preds[name], n_rules = rules.vote(pats, V["img_y"], V["img_split"], m)
        trial.set_user_attr(f"{name}_rules", n_rules)
        trial.set_user_attr(f"{name}_seconds", time.time() - t0)
        for s in ("val", "test"):
            a, c = rules.score(preds[name], V["img_y"], V["img_split"], s)
            trial.set_user_attr(f"{name}_{s}_acc", a)
            trial.set_user_attr(f"{name}_{s}_cov", c)
    final = rules.decide(preds["full"], preds["gcn"])
    for s in ("val", "test"):
        a, c = rules.score(final, V["img_y"], V["img_split"], s)
        trial.set_user_attr(f"{s}_acc", a)
        trial.set_user_attr(f"{s}_cov", c)
    acc, cov = trial.user_attrs["val_acc"], trial.user_attrs["val_cov"]
    return cov if acc >= TARGET else cov - 10 * (TARGET - acc)


def study(name):
    return optuna.create_study(study_name=name, storage=STORAGE, direction="maximize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10))


def report(name):
    st = optuna.load_study(study_name=name, storage=STORAGE)
    done = sorted((t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: -t.value)
    print(f"estudo {name}: {len(done)} completos, {sum(t.state.name == 'FAIL' for t in st.trials)} falharam")
    for t in done[:5]:
        u = t.user_attrs
        print(f"  #{t.number:3d} obj {t.value:.3f} | val {u['val_acc']:.1%} acerto {u['val_cov']:.1%} cob | "
              f"teste {u['test_acc']:.1%} acerto {u['test_cov']:.1%} cob | {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in t.params.items()})}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="rules")
    p.add_argument("--z", default=str(HERE / "runs" / "gcn" / "Z.npy"))
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        return report(a.study)
    if a.worker:
        study(a.study).optimize(lambda t: objective(t, a.z), n_trials=1,
                                catch=(rules.TooManyPatterns, MemoryError))
        return
    (HERE / "runs").mkdir(exist_ok=True)
    st = study(a.study)
    if not st.trials:  # começa pela configuração da versão anterior do relatório
        st.enqueue_trial({"K": 85, "M": 3, "sigma_min": 1, "tau_full": 0.6475, "m_full": 2, "tau_gcn": 0.8937, "m_gcn": 1})
    n_done = lambda: sum(t.state.is_finished() for t in optuna.load_study(study_name=a.study, storage=STORAGE).trials)  # noqa: E731
    target = n_done() + a.trials
    import os
    import threading
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    def loop(i):
        time.sleep(15 * i)
        with open(HERE / "runs" / f"{a.study}_worker{i}.log", "a") as log:
            while n_done() < target:
                subprocess.run([sys.executable, __file__, "--worker", "--study", a.study, "--z", a.z],
                               stdout=log, stderr=subprocess.STDOUT, env=env)

    th = [threading.Thread(target=loop, args=(i,)) for i in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    report(a.study)


if __name__ == "__main__":
    main()
