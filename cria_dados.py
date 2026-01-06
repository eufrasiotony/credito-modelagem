import numpy as np
import pandas as pd

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'idade': np.random.randint(18, 70, n),
    'sexo': np.random.choice(['M', 'F'], n),
    'estado_civil': np.random.choice(
        ['solteiro', 'casado', 'divorciado', 'viuvo'],
        n,
        p=[0.45, 0.40, 0.10, 0.05]
    ),
    'escolaridade': np.random.choice(
        ['fundamental', 'medio', 'superior', 'pos'],
        n,
        p=[0.20, 0.40, 0.30, 0.10]
    ),
    'qtde_dependentes': np.random.poisson(1.5, n).clip(0, 6)
})


df['renda_mensal'] = (
    np.random.lognormal(mean=8.4, sigma=0.5, size=n)
    .clip(1200, 30000)
    .round(2)
)

df['valor_emprestimo'] = (
    np.random.normal(15000, 7000, n)
    .clip(1000, 80000)
    .round(2)
)

df['tempo_emprego'] = np.random.exponential(6, n).clip(0, 40).round(1)

df['regiao'] = np.random.choice(
    ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'],
    n,
    p=[0.08, 0.22, 0.12, 0.38, 0.20]
)

df['atraso_30d'] = np.random.binomial(1, 0.18, n)
df['atraso_60d'] = np.random.binomial(1, 0.10, n)
df['atraso_90d'] = np.random.binomial(1, 0.06, n)


df["id_cliente"] = range(1, len(df) + 1)


df['pago'] = np.where(
    (df['atraso_30d'] > 0) |
    (df['atraso_60d'] > 0) |
    (df['atraso_90d'] > 0),
    0,
    1
)

df.to_csv('dados_emprestimos.csv', index=False)

