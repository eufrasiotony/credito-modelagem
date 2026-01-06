import pandas as pd
import numpy as np

df_dados_sel = pd.read_csv("dados_emprestimos.csv")
df_score = pd.read_csv("resultado_score.csv")
df_dados_sel = df_dados_sel[["id_cliente", "pago"]]
df_score_sel = df_score[["id_cliente", "grupo_score"]]


# 1 . Join das bases
df_join = df_dados_sel.merge(
    df_score_sel,
    on="id_cliente",
    how="inner"
)

# (opcional) salvar para análise
df_join.to_csv("base_join_score.csv", index=False)
print("Base join salva como 'base_join_score.csv'")

df_ks = pd.read_csv("base_join_score.csv")
df_ks = df_ks[["pago", "grupo_score"]].copy()
df_ks.rename(columns={"pago": "y", "grupo_score": "score"}, inplace=True)

df_ks["grupo"] = pd.qcut(df_ks["score"], 10, duplicates="drop")

ks_table = df_ks.groupby("grupo", observed=True).agg(
    total=("y", "count"),
    bads=("y", "sum")
)

ks_table["goods"] = ks_table["total"] - ks_table["bads"]

ks_table["cum_bads"] = ks_table["bads"].cumsum() / ks_table["bads"].sum()
ks_table["cum_goods"] = ks_table["goods"].cumsum() / ks_table["goods"].sum()

ks_table["ks"] = (ks_table["cum_bads"] - ks_table["cum_goods"]).abs()

ks = ks_table["ks"].max()

print(f"KS: {ks:.2%}")
print(ks_table)
