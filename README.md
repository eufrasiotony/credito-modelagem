📊 Projeto de Modelagem de Crédito
📌 Visão Geral

Este projeto tem como objetivo desenvolver um modelo de crédito capaz de estimar a probabilidade de inadimplência de clientes, auxiliando na tomada de decisão para concessão de crédito. A solução utiliza técnicas de análise exploratória de dados, engenharia de atributos e modelos de machine learning. O projeto contou com a auxilio de dados fictícios 

🎯 Objetivos

Analisar o perfil dos clientes

Identificar variáveis relevantes para risco de crédito

Construir e avaliar modelos preditivos

Gerar métricas para suporte à decisão de crédito

🗂 Estrutura do Projeto
├── data/
│   ├── dados_emprestimos.csv/              # Dados Criados
│   ├── resultado_score.csv/        # Dados scorados
│
├── src/
│   ├── cria-dados.py/ criação de base ficticios 
│   ├── Woe + regressão.py / analise explanatória
│   ├── modelo.py / modelo + criação do PKL
│
│
├── models/
│   └── modelo_final.pkl
│
├── calculo de KS/
│   └── Ks.py
│
├── requirements.txt
└── README.md

📊 Base de Dados

A base de dados contém informações demográficas, financeiras e comportamentais dos clientes, como:

idade,sexo,estado_civil,escolaridade,qtde_dependentes,renda_mensal,valor_emprestimo,tempo_emprego,regiao,atraso_30d,atraso_60d,atraso_90d, variável dependente (pago)


⚠️ Observação: Os dados utilizados são fictícios ou anonimizados, respeitando a LGPD.

🧪 Metodologia

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

📈 Resultados

O modelo final apresentou:

AUC: 0.5031797124538866

KS: 2.28%

Péssima separação entre bons e maus pagadores. O motivo é aleatoriedade da criação dos dados do projeto.

Os resultados indicam que o modelo não é adequado para apoiar decisões de crédito.

🛠 Tecnologias Utilizadas

scorecardpy

Pandas

NumPy

Scikit-learn


👤 Autor

Tony Eufrasio
Cientista de Dados / Analista de Risco de Crédito