# Segmentação de Clientes com K-Means e DBSCAN

Este projeto de Machine Learning não supervisionado foca na **segmentação de clientes** de um *mall* (shopping center) com base em sua Renda Anual e Pontuação de Gastos. O objetivo é identificar grupos de clientes com perfis de consumo e financeiro semelhantes, utilizando os algoritmos **K-Means** (baseado em centroides) e **DBSCAN** (baseado em densidade).

## ⚙️ Tecnologias Utilizadas

O projeto foi desenvolvido em **Python** e utiliza as seguintes bibliotecas:

* **Pandas**: Manipulação e análise de dados.
* **Seaborn** e **Matplotlib**: Visualização e exploração gráfica dos dados.
* **Scikit-learn (sklearn)**: Implementação dos algoritmos de *Clustering* (K-Means, DBSCAN) e métricas de avaliação.

## 💾 Conjunto de Dados

O script carrega o arquivo `mall.csv`, que contém as seguintes colunas de interesse:

* **`CustomerID`**: (Removida no pré-processamento).
* **`Gender`**: Gênero do cliente.
* **`Age`**: Idade do cliente.
* **`Annual Income (k$)`**: Renda Anual do cliente em milhares de dólares.
* **`Spending Score (1-100)`**: Pontuação de gastos do cliente (normalizada de 1 a 100).

## 🚀 Etapas do Projeto

### 1. Exploração e Análise de Dados

* **Inspeção Inicial**: Verificação da dimensionalidade (`.shape`), das primeiras instâncias (`.head()`) e tipos de dados (`.info()`, `.describe()`).
* **Visualizações Univariadas**: Plotagem de **Histogramas** para visualizar a distribuição individual de todas as variáveis numéricas.
* **Matriz de Correlação**: Geração de um **Mapa de Calor (Heatmap)** com a correlação de Spearman para analisar a relação entre as variáveis.
* **Análise Bivariada**: Utilização do **Pair Plot** (com `hue='Gender'`) para visualizar a relação paritária entre todas as variáveis, diferenciando os dados por gênero.

### 2. Clusterização K-Means

O K-Means foi aplicado nas variáveis **'Annual Income (k$)'** e **'Spending Score (1-100)'** para identificar grupos de consumo e renda.

* **Método do Cotovelo**:
    * Um *loop* de $K=1$ a $K=9$ é executado.
    * O gráfico **'Número de clusters' vs. 'Inércia'** é plotado para determinar o número ideal de clusters (*elbow method*).
* **Modelagem**: O modelo é instanciado e treinado com **$n\_clusters=5$** (o valor ideal inferido pelo método do cotovelo).
* **Visualização do Cluster**: Os clusters formados são plotados em um gráfico de dispersão, onde cada grupo recebe uma cor diferente. Os **Centroides** calculados são adicionados ao gráfico (marcador 'X' preto).

### 3. Clusterização DBSCAN

O algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é utilizado como um método de *clustering* alternativo, focado na densidade dos pontos.

* **Modelagem**: O modelo é instanciado e treinado com os seguintes hiperparâmetros:
    * **$\text{eps}=10$** (raio de vizinhança).
    * **$\text{min\_samples}=7$** (mínimo de pontos para formar uma região densa).
* **Visualização**: Os clusters gerados pelo DBSCAN, incluindo os pontos classificados como ruído (outliers), são plotados em um gráfico de dispersão.

### 4. Avaliação e Comparação dos Modelos

Para comparar a eficácia e a similaridade dos agrupamentos gerados pelos dois modelos, as seguintes métricas são utilizadas:

* **Adjusted Rand Score (ARS)**: Mede a similaridade entre os agrupamentos do K-Means e do DBSCAN (escala de 0 a 1).
* **Silhouette Score (K-Means)**: Mede quão bem representados e separados estão os clusters do K-Means.
* **Silhouette Score (DBSCAN)**: Mede quão bem representados e separados estão os clusters do DBSCAN.

---

A análise desses resultados permite que o negócio tome decisões estratégicas sobre marketing e ofertas personalizadas para cada segmento de clientes.
