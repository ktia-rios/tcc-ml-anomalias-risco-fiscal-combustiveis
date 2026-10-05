"""
07_robustez_if.py
-----------------
Análise de robustez: compara o ranking do Isolation Forest
com e sem os 13 indicadores binários que compõem o score_risco_v2.

Avalia se os três componentes do score_final representam fontes
independentes de evidência ou se há sobreposição de informação.

Saídas:
  outputs/tabelas/robustez_if_ranking.csv
  outputs/tabelas/robustez_if_resumo.txt
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
FEAT_FILE  = ROOT / "data" / "processed" / "dataset_features_rj.parquet"
RESULT_CSV = ROOT / "outputs" / "tabelas" / "resultado_completo.csv"
OUT_DIR    = ROOT / "outputs" / "tabelas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Os 13 indicadores binários que compõem o score_risco_v2
INDICADORES_SCORE_MANUAL = [
    "inapta", "suspensa",
    "socio_risco_alto", "socio_risco_medio", "socio_risco_baixo",
    "alta_concentracao_pj", "sem_socios",
    "empresa_muito_nova", "nova_inapta", "antiga_inapta",
    "capital_muito_baixo", "porte_incompativel", "simples_suspeito",
]

# Variáveis excluídas do modelo original (scores derivados e target)
EXCLUIR_SEMPRE = ["score_risco", "score_risco_v2", "sancionada", "id_empresa"]


def rodar_if(X: np.ndarray, label: str) -> np.ndarray:
    """Treina Isolation Forest e retorna scores normalizados (0=normal, 1=anômalo)."""
    log.info(f"  Treinando IF [{label}] com {X.shape[1]} variáveis...")
    IF = IsolationForest(n_estimators=200, contamination=0.10,
                         random_state=42, n_jobs=-1)
    IF.fit(X)
    scores_raw = IF.decision_function(X)
    # Inverte: score mais negativo = mais anômalo → normaliza para [0,1]
    score_norm = (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min())
    score_risco = 1 - score_norm
    return score_risco


def main():
    log.info("=" * 60)
    log.info("Etapa 7 — Robustez do Isolation Forest")
    log.info("(com vs. sem indicadores do score manual)")
    log.info("=" * 60)

    # Carregar dados
    log.info(f"\nLendo: {FEAT_FILE}")
    df_feat = pd.read_parquet(FEAT_FILE)
    df_res  = pd.read_csv(RESULT_CSV)
    df = df_feat.merge(df_res[["id_empresa", "if_score_risco", "ranking"]], on="id_empresa")
    log.info(f"  ✓ {len(df):,} empresas carregadas")

    # Selecionar variáveis numéricas (mesmas do modelo original)
    cols_num = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUIR_SEMPRE
        and "cluster" not in c and "outlier" not in c
        and "pca" not in c and "score" not in c
        and "ranking" not in c
    ]

    # Modelo A: COMPLETO (igual ao original)
    X_completo = StandardScaler().fit_transform(df[cols_num].fillna(0))
    score_completo = rodar_if(X_completo, "completo")

    # Modelo B: SEM os 13 indicadores do score manual
    cols_sem_manual = [c for c in cols_num if c not in INDICADORES_SCORE_MANUAL]
    log.info(f"\n  Variáveis removidas (score manual): {len(INDICADORES_SCORE_MANUAL)}")
    log.info(f"  Variáveis restantes: {len(cols_sem_manual)}")
    X_sem_manual = StandardScaler().fit_transform(df[cols_sem_manual].fillna(0))
    score_sem_manual = rodar_if(X_sem_manual, "sem_manual")

    # Rankings
    rank_completo   = pd.Series(score_completo).rank(ascending=False).astype(int)
    rank_sem_manual = pd.Series(score_sem_manual).rank(ascending=False).astype(int)

    # Correlação de Spearman entre os dois rankings
    rho, pval = spearmanr(score_completo, score_sem_manual)
    log.info(f"\n  Correlação de Spearman (scores): ρ = {rho:.4f} (p = {pval:.2e})")

    # Top-50: sobreposição
    top50_completo   = set(np.argsort(score_completo)[-50:])
    top50_sem_manual = set(np.argsort(score_sem_manual)[-50:])
    overlap_top50 = len(top50_completo & top50_sem_manual)
    log.info(f"  Sobreposição Top-50: {overlap_top50}/50 ({overlap_top50*2:.0f}%)")

    # Top-152 (10%): sobreposição
    top152_completo   = set(np.argsort(score_completo)[-152:])
    top152_sem_manual = set(np.argsort(score_sem_manual)[-152:])
    overlap_152 = len(top152_completo & top152_sem_manual)
    log.info(f"  Sobreposição Top-152 (10%): {overlap_152}/152 ({overlap_152/152*100:.1f}%)")

    # Salvar comparação por empresa
    df_out = df[["id_empresa"]].copy()
    df_out["score_if_original"]    = df["if_score_risco"]
    df_out["score_if_completo"]    = score_completo
    df_out["score_if_sem_manual"]  = score_sem_manual
    df_out["rank_completo"]        = rank_completo
    df_out["rank_sem_manual"]      = rank_sem_manual
    df_out["diff_rank"]            = (rank_completo - rank_sem_manual).abs()
    df_out = df_out.sort_values("rank_completo")

    out_csv = OUT_DIR / "robustez_if_ranking.csv"
    df_out.to_csv(out_csv, index=False)
    log.info(f"\n  ✓ Ranking comparativo salvo: {out_csv}")

    # Resumo textual
    resumo = f"""
════════════════════════════════════════════════════════════
  ROBUSTEZ DO ISOLATION FOREST
  Modelo completo vs. sem indicadores do score manual
════════════════════════════════════════════════════════════

  Variáveis modelo completo:    {X_completo.shape[1]}
  Variáveis sem score manual:   {X_sem_manual.shape[1]}
  Indicadores removidos:        {len(INDICADORES_SCORE_MANUAL)}

  Correlação de Spearman (ρ):   {rho:.4f}
  p-valor:                      {pval:.2e}

  Sobreposição Top-50:          {overlap_top50}/50 ({overlap_top50*2:.0f}%)
  Sobreposição Top-152 (10%):   {overlap_152}/152 ({overlap_152/152*100:.1f}%)

  Diferença média de ranking:   {df_out['diff_rank'].mean():.1f} posições
  Diferença mediana de ranking: {df_out['diff_rank'].median():.1f} posições
  Diferença máxima de ranking:  {df_out['diff_rank'].max()} posições

════════════════════════════════════════════════════════════
"""
    print(resumo)

    out_txt = OUT_DIR / "robustez_if_resumo.txt"
    out_txt.write_text(resumo)
    log.info(f"  ✓ Resumo salvo: {out_txt}")
    log.info("\n✅ Análise de robustez concluída.")


if __name__ == "__main__":
    main()
