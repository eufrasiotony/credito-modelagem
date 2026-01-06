import pandas as pd

# -----------------------------------------------------
# 1. Carregar bases
# -----------------------------------------------------
df_dados = pd.read_csv("dados_emprestimos.csv")
df_score = pd.read_csv("resultado_score.csv")

# -----------------------------------------------------
# 2. Criar chave fictícia (ID)
# -----------------------------------------------------
df_dados["id_cliente"] = range(1, len(df_dados) + 1)
df_score["id_cliente"] = range(1, len(df_score) + 1)

# -----------------------------------------------------
# 3. Selecionar colunas necessárias
# -----------------------------------------------------
df_dados_sel = df_dados[["id_cliente", "pago"]]
df_score_sel = df_score[["id_cliente", "grupo_score"]]

# -----------------------------------------------------
# 4. Join das bases
# -----------------------------------------------------
df_join = df_dados_sel.merge(
    df_score_sel,
    on="id_cliente",
    how="inner"
)

# -----------------------------------------------------
# 5. Resultado final
# -----------------------------------------------------
print(df_join.head())

# (opcional) salvar para análise
df_join.to_csv("base_join_score.csv", index=False)
print("Base join salva como 'base_join_score.csv'")