# =====================================================
# PIPELINE COMPLETO: WOE + REGRESSÃO LOGÍSTICA + PKL
# =====================================================

import pandas as pd
import scorecardpy as sc
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------
# 1. Carregar dados
# -----------------------------------------------------
df = pd.read_csv("dados_emprestimos.csv")

df["pago"] = df["pago"].astype(int)

# -----------------------------------------------------
# 2. Encoders categóricos (UM POR VARIÁVEL)
# -----------------------------------------------------
encoders = {}

for col in ["sexo", "estado_civil", "escolaridade", "regiao"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # guardar encoder

# -----------------------------------------------------
# 3. Remover colunas de atraso
# -----------------------------------------------------
df.drop(columns=["atraso_30d", "atraso_60d", "atraso_90d"], inplace=True)

# -----------------------------------------------------
# 4. Split ANTES do WOE
# -----------------------------------------------------
train, test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["pago"]
)

# -----------------------------------------------------
# 5. Binning + WOE (APENAS TREINO)
# -----------------------------------------------------
bins = sc.woebin(
    train,
    y="pago"
)

# -----------------------------------------------------
# 6. Aplicar WOE
# -----------------------------------------------------
train_woe = sc.woebin_ply(train, bins)
test_woe  = sc.woebin_ply(test, bins)

# -----------------------------------------------------
# 7. Separar X e y
# -----------------------------------------------------
X_train = train_woe.drop("pago", axis=1)
y_train = train_woe["pago"]

X_test = test_woe.drop("pago", axis=1)
y_test = test_woe["pago"]

# -----------------------------------------------------
# 8. Modelo de Regressão Logística
# -----------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

# -----------------------------------------------------
# 9. Avaliação
# -----------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 10. SALVAR ARTEFATOS EM PKL
# -----------------------------------------------------
import pickle
import os

print("\nChegou no momento de salvar o modelo")
print("Diretório atual:", os.getcwd())

artefatos_modelo = {
    "model": model,
    "bins": bins,
    "encoders": encoders,
    "variaveis_modelo": list(X_train.columns)
}

with open("credit_model.pkl", "wb") as f:
    pickle.dump(artefatos_modelo, f)

print("Modelo salvo com sucesso!")



