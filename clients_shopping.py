import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importando os dados em .csv como um dataframe
dados = pd.read_csv('K-means\mall.csv')
# Verificando a dimensionalidade da base de dados
print(dados.shape)
# Análise das 5 primeiras instâncias 
print(dados.head())
# Topologia dos dados 
print(dados.info())
# Análise de estatística básica
print(dados.describe())

dados.drop(columns='CustomerID', inplace=True)
dados.hist(figsize=(15,15))
plt.show()

corr_spearman = dados.corr('spearman', numeric_only=True)
plt.figure(figsize=(8,6))

# Criar o heatmap
sns.heatmap(
    corr_spearman,      # A matriz de correlação
    annot=True,         # Escrever os valores de correlação nas células
    fmt=".1f",          # Formatar os números para duas casas decimais
    cmap='coolwarm',    # Esquema de cores (cores quentes para > 0, frias para < 0)
    linewidths=.5,       # Adicionar linhas entre as células
)
plt.title('Mapa de Calor da Correlação de Spearman')
plt.show()

# Quantidade de instâncias por valor exclusivo
print(dados['Gender'].value_counts())

sns.pairplot(dados, hue='Gender')
plt.show()

from sklearn.cluster import KMeans
par_ordenado = ['Annual Income (k$)', 'Spending Score (1-100)']

seq = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(dados[par_ordenado])
    seq.append(kmeans.inertia_)

plt.rcParams['figure.figsize'] = (10, 5)
plt.plot(range(1,10), seq, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.show()

# Instanciando um objeto K-Means com o setup
kmeans = KMeans(n_clusters=5, random_state=0)
# Clusterização dos dados segundo atributos descritivos selecionados
kmeans.fit(dados[par_ordenado])
# Armazenamento das coordenadas dos centróides calculados para cada cluster
centroides = kmeans.cluster_centers_
# ID de cada classe para o atributo alvo, a partir dos atributos descritivos selecionados ([0, {...}, n] representando cada grupo)
kmeans_groups = kmeans.predict(dados[par_ordenado])
dados['Grupos'] = kmeans_groups

# plotando os dados identificando com os seus clusters
plt.scatter(dados[['Annual Income (k$)']], dados[['Spending Score (1-100)']], c=kmeans_groups, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')
# plotando os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
plt.show()
# Verificar clusterização resultante para as 10 primeiras instâncias
print(dados.head(10))

from sklearn.cluster import DBSCAN
# Criando a circunferência com o raio (EPS = 10) de aglutinação dos pontos por densidade (minPts = 8)
dbscan = DBSCAN(eps=10, min_samples=7)
dbscan.fit(dados[par_ordenado])
dbscan_groups = dbscan.labels_

plt.scatter(dados[['Annual Income (k$)']],dados[['Spending Score (1-100)']], c=dbscan_groups, alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

from sklearn.metrics import adjusted_rand_score, silhouette_score

print('Índice de similaridade entre modelos (0 até 1):', adjusted_rand_score(kmeans_groups, dbscan_groups))
print('Índice de representatividade dos clusters com a realidade K-means:', silhouette_score(dados[par_ordenado], kmeans_groups))
print('Índice de representatividade dos clusters com a realidade DBSCAN:', silhouette_score(dados[par_ordenado], dbscan_groups))
