import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from questao1 import database

X = database.drop(['CLASS'], axis=1)
y = database['CLASS'].values

#INICIALIZANDO
scaler = StandardScaler()
lb = LabelEncoder()

# T-SNE
normalized = normalize(X)

# Criação de uma instância TSNE: model
model = TSNE(n_components=2)

# Aplicação fit_transform às amostras: tsne_features
tsne_features = model.fit_transform(normalized)

# Selecionando o 0º recurso: xs
xs = tsne_features[:,0]

# Selecionando o 1º recurso: ys
ys = tsne_features[:,1]

# Gráfico de dispersão, colorindo por variedade_números
plt.scatter(xs, ys, c=y)
plt.show()

# PCA
scaled_samples = scaler.fit_transform(X)
class_pca = lb.fit_transform(database['CLASS'])

# Criação de um modelo PCA com 2 componentes
pca = PCA(n_components=2)

# Ajustar a instância do PCA às amostras dimensionadas
pca.fit(scaled_samples)

# Transformação das amostras dimensionadas: pca_features
transformed = pca.transform(scaled_samples)

# Visualização do gráfico de dispersão com dimensão reduzida
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=class_pca)
plt.title('PCA')
plt.show()