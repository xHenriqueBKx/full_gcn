"""Avalia uma configuração fixa das regras (sem busca) e imprime o resultado da Seção 6.

    python evaluate.py                      # configuração da tabela do relatório
    python evaluate.py --params '{"K": 85, "M": 3, ...}'

Precisa do embedding da GCN (python gcn.py -> runs/gcn/Z.npy).
"""
import argparse
import json
from pathlib import Path

import optuna

import search

REPORT = {"K": 188, "M": 8, "sigma_min": 1, "tau_full": 0.6731877033255295, "m_full": 2,
          "tau_gcn": 0.9420075638292873, "m_gcn": 1}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--params", default=None, help="JSON com K, M, sigma_min, tau_full, m_full, tau_gcn, m_gcn")
    p.add_argument("--z", default=str(Path(__file__).resolve().parent / "runs" / "gcn" / "Z.npy"))
    a = p.parse_args()
    params = json.loads(a.params) if a.params else REPORT
    t = optuna.trial.FixedTrial(params)
    obj = search.objective(t, a.z)
    u = t.user_attrs
    print(f"parâmetros: {json.dumps(params)}")
    for name in ("full", "gcn"):
        print(f"  visão {name:4s}: {u[f'{name}_rules']} regras | val {u[f'{name}_val_acc']:.1%} acerto "
              f"{u[f'{name}_val_cov']:.1%} cob | teste {u[f'{name}_test_acc']:.1%} acerto {u[f'{name}_test_cov']:.1%} cob")
    print(f"decisão (as duas concordam): val {u['val_acc']:.1%} acerto {u['val_cov']:.1%} cobertura | "
          f"teste {u['test_acc']:.1%} acerto {u['test_cov']:.1%} cobertura | objetivo {obj:.3f}")
