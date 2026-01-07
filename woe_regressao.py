# =====================================================
# PIPELINE COMPLETO: WOE + REGRESSÃO LOGÍSTICA
# =====================================================

# 1. Importações
import pandas as pd
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------
# 2. Carregar dados
# -----------------------------------------------------
df = pd.read_csv("dados_emprestimos.csv")

# GARANTA: 0 = bom | 1 = mau
df["pago"] = df["pago"].astype(int)

le = LabelEncoder()
# transformar variáveis categóricas em numéricas

df["sexo"] = le.fit_transform(df["sexo"])  # M=1, F=0
df["estado_civil"] = le.fit_transform(df["estado_civil"])  # solteiro=0, casado=1, divorciado=2, viuvo=3
df["escolaridade"] = le.fit_transform(df["escolaridade"])  # fundamental=0, medio=1, superior=2, pos=3
df["regiao"] = le.fit_transform(df["regiao"])  # Norte=0

# remover colunas de atraso

df.drop(columns=["atraso_30d", "atraso_60d", "atraso_90d"], inplace=True)

# -----------------------------------------------------
# 3. Split ANTES do WOE (REGRA DE OURO)
# -----------------------------------------------------
train, test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["pago"]
)

# -----------------------------------------------------
# 4. Binning + WOE (APENAS TREINO)
# -----------------------------------------------------
bins = sc.woebin(
    train,
    y="pago"
)

# -----------------------------------------------------
# 5. Aplicar WOE em TODAS as variáveis
# -----------------------------------------------------
train_woe = sc.woebin_ply(train, bins)
test_woe  = sc.woebin_ply(test, bins)

# -----------------------------------------------------
# 6. Separar X e y (SOMENTE WOE)
# -----------------------------------------------------
X_train = train_woe.drop("pago", axis=1)
y_train = train_woe["pago"]

X_test = test_woe.drop("pago", axis=1)
y_test = test_woe["pago"]

# -----------------------------------------------------
# 7. Modelo de Regressão Logística
# -----------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# 8. Avaliação
# -----------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, y_prob))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 9. Coeficientes do modelo (interpretação)
# -----------------------------------------------------
coeficientes = pd.DataFrame({
    "variavel": X_train.columns,
    "coeficiente": model.coef_[0]
}).sort_values("coeficiente", ascending=False)

print("\nCoeficientes do Modelo:")
print(coeficientes)

# -----------------------------------------------------
# 10. Information Value (IV)
# -----------------------------------------------------

iv_list = []

for var in bins.keys():
    iv_list.append({
        "variavel": var,
        "info_value": bins[var]["total_iv"].iloc[0]
    })

iv = pd.DataFrame(iv_list).sort_values(
    "info_value",
    ascending=False
)

print("\nInformation Value (IV):")
print(iv)


