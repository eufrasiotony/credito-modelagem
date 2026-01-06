# =====================================================
# PROCESSO DE SCORAGEM - MODELO DE CRÉDITO
# =====================================================

import pandas as pd
import scorecardpy as sc
import pickle

# -----------------------------------------------------
# 1. Carregar artefatos do modelo
# -----------------------------------------------------
with open("credit_model.pkl", "rb") as f:
    artefatos = pickle.load(f)

model = artefatos["model"]
bins = artefatos["bins"]
encoders = artefatos["encoders"]
variaveis_modelo = artefatos["variaveis_modelo"]

# -----------------------------------------------------
# 2. Carregar base para score (NOVA BASE)
# -----------------------------------------------------
df_score = pd.read_csv("dados_emprestimos.csv")

# -----------------------------------------------------
# 3. Aplicar encoders categóricos
# -----------------------------------------------------
for col, encoder in encoders.items():
    if col not in df_score.columns:
        raise ValueError(f"Coluna ausente na base de score: {col}")

    # tratar categorias novas
    df_score[col] = df_score[col].map(
        lambda x: x if x in encoder.classes_ else encoder.classes_[0]
    )
    df_score[col] = encoder.transform(df_score[col])

# -----------------------------------------------------
# 4. Remover colunas que NÃO entram no modelo
# -----------------------------------------------------
colunas_remover = ["atraso_30d", "atraso_60d", "atraso_90d", "pago"]
df_score.drop(columns=[c for c in colunas_remover if c in df_score.columns],
              inplace=True)

# -----------------------------------------------------
# 5. Aplicar WOE
# -----------------------------------------------------
df_woe = sc.woebin_ply(df_score, bins)

# garantir ordem correta das variáveis
X = df_woe[variaveis_modelo]

# -----------------------------------------------------
# 6. Gerar PD e SCORE
# -----------------------------------------------------
df_score["pd"] = model.predict_proba(X)[:, 1]

# score simples (0–1000)
df_score["score"] = (1 - df_score["pd"]) * 1000

# -----------------------------------------------------
# 7. Criar 10 grupos de score (decis)
# -----------------------------------------------------
df_score["grupo_score"] = pd.qcut(
    df_score["score"],
    q=10,
    labels=False,
    duplicates="drop"
) + 1

# inverter: 10 = melhor score
df_score["grupo_score"] = 11 - df_score["grupo_score"]

# -----------------------------------------------------
# 8. Output final
# -----------------------------------------------------
df_score.to_csv("resultado_score.csv", index=False)

print("Scoragem finalizada com sucesso!")
print(df_score[["score", "pd", "grupo_score"]].head())
