# 📊 Projeto de Modelagem de Crédito

## 📌 Visão Geral

Este projeto tem como objetivo desenvolver um modelo de crédito capaz de estimar a probabilidade de inadimplência de clientes, auxiliando na tomada de decisão para concessão de crédito. A solução utiliza técnicas de análise exploratória de dados, engenharia de atributos e modelos de machine learning. O projeto contou com a auxilio de dados fictícios 

## 🎯 Objetivos

Analisar o perfil dos clientes

Identificar variáveis relevantes para risco de crédito

Construir e avaliar modelos preditivos

Gerar métricas para suporte à decisão de crédito

---

## 🗂 Estrutura do Projeto

```text

├── data/
│   ├── raw/
│   │   └── dados_emprestimos.csv
│   ├── processed/
│   │   └── resultado_score.csv
│
├── src/
│   ├── cria_dados.py
│   ├── woe_regressao.py
│   ├── modelo.py
│   ├── scoragem.py
│   └── ks.py
│   
│
├── models/
│   └── credit_model.pkl
│
└── README.md
```
---

## 📊 Base de Dados

A base de dados contém informações demográficas, financeiras e comportamentais dos clientes, como:

Idade, sexo, estado_civil, escolaridade, qtde_dependentes, renda_mensal, valor_emprestimo, tempo_emprego, regiao e  variável dependente (pago)

⚠️ Observação: Os dados utilizados são fictícios ou anonimizados, respeitando a LGPD.

---

## 🧪 Metodologia

Análise Exploratória (EDA)

Estatísticas descritivas

Análise de correlação

Distribuição das variáveis

Pré-processamento

Codificação de variáveis categóricas

Modelagem

Regressão Logística

Avaliação

AUC-ROC

KS

Precisão, Recall e F1-score

Matriz de confusão

## 📈 Resultados

O modelo final apresentou:

AUC: 0.5031797124538866

KS: 2.28%

Péssima separação entre bons e maus pagadores. O motivo é aleatoriedade da criação dos dados do projeto.

Os resultados indicam que o modelo não é adequado para apoiar decisões de crédito.

🛠 Tecnologias Utilizadas

Scorecardpy

Pandas

NumPy

Scikit-learn


## 👤 Autor

Tony Eufrasio
Cientista de Dados / Analista de Risco de Crédito