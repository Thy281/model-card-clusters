# Model Card Clusters

Projeto de segmentação de clientes de cartão de crédito usando aprendizado não supervisionado. O objetivo é transformar os dados de comportamento dos clientes em grupos interpretáveis, permitindo comparar perfis de uso, tamanho dos segmentos e características médias de cada cluster.

## Objetivos

- Preparar a base de clientes para análise de clustering.
- Comparar configurações de KMeans e clustering hierárquico.
- Registrar experimentos, parâmetros e métricas no MLflow.
- Treinar um modelo final com quatro clusters.
- Gerar visualizações para interpretar a separação e o perfil dos clusters.

## Dados

A base original está em `data/raw/BankChurners_clusterizacao.csv`. O fluxo gera dois arquivos intermediários:

| Arquivo | Descrição |
| --- | --- |
| `data/interim/BankChurners_clean.parquet` | Dados sem linhas duplicadas |
| `data/processed/ABT.parquet` | Base analítica usada no treinamento |

A ABT contém 10.127 registros e é carregada por `src/training/load_date.py`.

O dataset processado também pode ser armazenado no Amazon S3. Quando `DATA_SOURCE=s3`, os scripts de treinamento e avaliação leem a ABT do bucket definido em `S3_BUCKET`, usando `boto3`. O projeto não define um bucket padrão: cada pessoa informa o próprio bucket. As credenciais são obtidas pelo mecanismo padrão da AWS e não devem ser salvas no código ou no Git.

Por padrão, `DATA_SOURCE=local` e a ABT é lida de `data/processed/ABT.parquet`. Para usar o S3, primeiro prepare a ABT localmente e configure o perfil da AWS CLI. Em seguida, defina as variáveis e envie o arquivo:

```bash
export AWS_PROFILE=model-card-s3
export DATA_SOURCE=s3
export S3_BUCKET=seu-bucket-privado
export S3_KEY=data/processed/ABT.parquet
export AWS_REGION=sa-east-1
aws s3 cp data/processed/ABT.parquet s3://$S3_BUCKET/$S3_KEY
```

O nome do perfil, bucket, chave e região podem ser substituídos pelos valores da sua configuração. Para voltar à leitura local, remova `DATA_SOURCE` do ambiente ou defina `DATA_SOURCE=local`.

## Features usadas

O pipeline seleciona as seguintes variáveis:

- Numéricas: `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1` e `Avg_Utilization_Ratio`.
- Transformadas com `log1p`: `Credit_Limit` e `Total_Trans_Amt`.
- Categórica: `Card_Category`.

O pré-processamento aplica `StandardScaler` às variáveis numéricas, `log1p` seguido de padronização às variáveis de escala e `OneHotEncoder` à categoria do cartão.

## Modelo final

O modelo final é um `KMeans` configurado com:

```text
n_clusters = 4
init = k-means++
n_init = 20
random_state = 42
```

O modelo é encapsulado em um pipeline do scikit-learn, com o pré-processamento e o clusterizador na mesma sequência. Isso mantém a transformação aplicada aos dados consistente com a usada durante o treinamento.

## Avaliação

Os experimentos calculam:

- Silhouette Score: avalia coesão e separação dos clusters; valores maiores são melhores.
- Davies-Bouldin Score: compara similaridade entre clusters; valores menores são melhores.
- Calinski-Harabasz Score: mede a razão entre dispersão entre grupos e dentro dos grupos; valores maiores são melhores.

O script `src/evaluation/profile_clusters.py` também imprime o tamanho de cada cluster, as medianas das features numéricas e a distribuição de `Card_Category` por cluster.

## Visualizações do modelo final

O script `src/evaluation/visualize_clusters.py` treina o modelo final e gera:

1. Um gráfico de dispersão em duas dimensões usando PCA, com os clientes coloridos por cluster e os centróides destacados.
2. Um gráfico de barras com a quantidade de clientes em cada cluster.
3. Um heatmap com o perfil médio padronizado das features numéricas por cluster.

Para exibir os gráficos:

```bash
python src/evaluation/visualize_clusters.py
```

### Imagens do modelo

As figuras abaixo são geradas pelo script de visualização e salvas em `reports/model/`.

![Clusters do modelo final e tamanho dos grupos](reports/model/clusters_pca_e_tamanho.png)

![Perfil médio padronizado dos clusters](reports/model/perfil_clusters_heatmap.png)

### Perfis sugeridos para os clusters

A tabela abaixo resume uma interpretação de negócio para cada grupo identificado pelo modelo:

![Perfis sugeridos para os clusters](reports/model/perfis_sugeridos_clusters.png)

## Interpretação inicial e hipóteses de negócio

Os nomes abaixo descrevem padrões observados nos dados e são interpretações preliminares. Os números dos clusters são identificadores atribuídos pelo algoritmo; não representam uma ordem de valor ou importância.

| Cluster | Sinais resumidos | Interpretação possível | Hipótese a investigar | Como validar |
| --- | --- | --- | --- | --- |
| 0 | Uso intenso, limite mais baixo e utilização elevada. | Uso intenso do limite disponível. | Avaliar se uma comunicação sobre organização de gastos ou benefícios adequados ao padrão de uso melhora a experiência. | Comparar engajamento e satisfação entre clientes que receberam a comunicação e um grupo de controle. |
| 1 | Limite alto e alto volume de transações. | Clientes com uso elevado do cartão. | Testar uma ação de relacionamento ou benefícios para incentivar retenção e uso recorrente. | Comparar retenção e volume de transações com um grupo de controle. |
| 2 | Volume baixo e mudança positiva no valor transacionado entre os períodos observados. | Clientes com pouco volume e possível mudança recente no valor de uso. | Investigar se uma ação de ativação ajuda a aumentar o uso recorrente. | Comparar quantidade e valor de transações após a ação com um grupo de controle. |
| 3 | Mais meses de inatividade e menos transações. | Baixo engajamento recente. | Testar uma campanha de reativação. | Comparar a proporção de clientes que voltam a transacionar com um grupo de controle. |

Essas ações são hipóteses, não resultados comprovados pelo modelo. A base não contém resposta a campanhas nem uma variável de resultado que permita afirmar que um cluster cancelará o cartão ou que uma ação aumentará a retenção. Uma avaliação futura deve usar um experimento controlado e medir os resultados depois da ação.

### Imagens da análise exploratória

Alguns resultados da análise exploratória estão disponíveis em `reports/eda/`.

![Correlação entre variáveis numéricas](reports/eda/correlacao_numerica.png)

![Distribuições das variáveis numéricas](reports/eda/distribuicoes_numericas.png)

![Frequências das variáveis categóricas](reports/eda/frequencias_categoricas.png)

## Como executar

### 1. Criar o ambiente

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar os dados

```bash
python src/preprocessing/clean.py
python src/features/build_abt.py
```

### 3. Explorar a base

```bash
python nootbooks/Eda.py
```

Os resultados exploratórios são salvos em `reports/eda/`.

### 4. Iniciar o MLflow

Em um terminal, na raiz do projeto e com o ambiente virtual ativado, inicie o servidor local:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Mantenha esse terminal aberto durante os passos de treinamento. O servidor local usa `mlflow.db` para os metadados e disponibiliza a interface em `http://127.0.0.1:5000`.

### 5. Comparar modelos

Em outro terminal, na raiz do projeto e com o ambiente virtual ativado:

```bash
python src/training/train.py
```

Esse treinamento avalia configurações de KMeans e `AgglomerativeClustering` e registra os resultados no experimento `model-card-clusters` do MLflow.

> A busca avalia 45 configurações. Como calcula métricas sobre os registros da ABT, pode levar tempo e consumir memória.

### 6. Treinar o modelo final

O script final registra o pipeline no MLflow usando o nome `kmeans_final`:

```bash
python src/training/train_final_model.py
```

Esse script espera o MLflow disponível em `http://127.0.0.1:5000/`.

### 7. Inspecionar os clusters

```bash
python src/evaluation/profile_clusters.py
python src/evaluation/visualize_clusters.py
```

O perfil é exibido no terminal. Os gráficos são salvos em `reports/model/` e o servidor MLflow pode ser acessado em `http://127.0.0.1:5000`. Para encerrar o servidor, use `Ctrl+C` no terminal em que ele está rodando.

## Estrutura principal

```text
model-card-clusters/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── nootbooks/
│   └── Eda.py
├── reports/eda/
├── reports/model/
├── src/
│   ├── evaluation/
│   ├── features/
│   ├── preprocessing/
│   └── training/
├── requirements.txt
└── README.md
```

## Tecnologias

- Python 3.11
- pandas e NumPy
- scikit-learn
- Matplotlib e Seaborn
- MLflow
- Parquet via PyArrow

## Limitações

Este projeto realiza segmentação exploratória e não possui uma variável-alvo supervisionada. Os clusters representam padrões estatísticos da base e devem ser interpretados junto com as métricas, os perfis gerados e o contexto de negócio.
