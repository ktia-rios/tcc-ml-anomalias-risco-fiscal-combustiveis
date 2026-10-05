"""
06_sensibilidade.py
-------------------
Análise de sensibilidade do parâmetro `contamination` do Isolation Forest.

Avalia a estabilidade do ranking de empresas anômalas para
contamination = 5%, 10%, 15% e 20%, usando os scores contínuos
(if_score) já calculados na etapa de modelagem.

Saídas:
  outputs/tabelas/sensibilidade_contamination.csv
  outputs/tabelas/sensibilidade_resumo.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuração ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "outputs" / "tabelas" / "resultado_completo.csv"
OUT_DIR    = ROOT / "outputs" / "tabelas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CENARIOS   = [0.05, 0.10, 0.15, 0.20]
BASE       = 0.10   # cenário de referência (usado no TCC)
TOP_N      = 50     # empresas prioritárias para análise de overlap


# ── Funções ───────────────────────────────────────────────────────────────────

def classificar_cenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna binária de anomalia para cada cenário."""
    for c in CENARIOS:
        thresh = np.percentile(df["if_score"], c * 100)
        col    = f"anomalia_{int(c * 100)}pct"
        df[col] = (df["if_score"] <= thresh).astype(int)
        log.info(f"  contamination={int(c*100)}%: threshold={thresh:.4f} | "
                 f"n={df[col].sum()} empresas")
    return df


def resumo_sensibilidade(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela-resumo: n anomalias, % amostra, overlap com cenário-base."""
    n_total  = len(df)
    base_col = f"anomalia_{int(BASE * 100)}pct"
    base_set = set(df[df[base_col] == 1].index)
    top_base = set(df.nsmallest(TOP_N, "if_score").index)

    rows = []
    for c in CENARIOS:
        col      = f"anomalia_{int(c * 100)}pct"
        anom_set = set(df[df[col] == 1].index)
        top_c    = set(df.nsmallest(TOP_N, "if_score").index)

        n_anom        = len(anom_set)
        pct_amostra   = n_anom / n_total * 100
        overlap_base  = len(anom_set & base_set) / len(base_set) * 100 if c != BASE else 100.0
        overlap_top50 = len(top_c & top_base) / TOP_N * 100

        rows.append({
            "contamination":        f"{int(c * 100)}%",
            "n_anomalias":          n_anom,
            "pct_amostra":          round(pct_amostra, 1),
            "overlap_base_10pct":   round(overlap_base, 1),
            f"overlap_top{TOP_N}":  round(overlap_top50, 1),
        })

    return pd.DataFrame(rows)


def nucleo_estavel(df: pd.DataFrame) -> pd.DataFrame:
    """Empresas presentes em TODOS os cenários (núcleo estável = top 5%)."""
    col_min = f"anomalia_{int(min(CENARIOS) * 100)}pct"
    nucleo  = df[df[col_min] == 1].copy()
    nucleo["presente_todos_cenarios"] = True
    return nucleo[["id_empresa", "if_score", "if_score_risco",
                   "score_final", "classe_final",
                   "presente_todos_cenarios"]].sort_values("if_score")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Etapa 6 — Análise de Sensibilidade (contamination IF)")
    log.info("=" * 60)

    log.info(f"\nLendo: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    log.info(f"  ✓ {len(df):,} empresas carregadas")

    log.info("\nClassificando cenários...")
    df = classificar_cenarios(df)

    log.info("\nCalculando resumo de sensibilidade...")
    df_resumo = resumo_sensibilidade(df)
    log.info("\n" + df_resumo.to_string(index=False))

    log.info("\nIdentificando núcleo estável...")
    df_nucleo = nucleo_estavel(df)
    log.info(f"  ✓ Núcleo estável: {len(df_nucleo)} empresas "
             f"({len(df_nucleo)/len(df)*100:.1f}% da amostra)")

    # Salvar outputs
    out_resumo = OUT_DIR / "sensibilidade_resumo.csv"
    out_detalhe = OUT_DIR / "sensibilidade_contamination.csv"

    df_resumo.to_csv(out_resumo, index=False)
    log.info(f"\n  ✓ Resumo salvo em: {out_resumo}")

    df_nucleo.to_csv(out_detalhe, index=False)
    log.info(f"  ✓ Núcleo estável salvo em: {out_detalhe}")

    log.info("\n✅ Análise de sensibilidade concluída.")
    return df_resumo, df_nucleo


if __name__ == "__main__":
    main()
