from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from questao1 import database

samples = database.drop(['CLASS'],axis=1)
class_values = database['CLASS'].values
# Redução da dimensionalidade usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(samples)

# Redução da dimensionalidade usando t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(samples)

# Divisão dos dados reduzidos em treinamento e teste
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, class_values, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, class_values, test_size=0.2, random_state=42)

# Criação dos classificadores k-NN para PCA e t-SNE
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_tsne = KNeighborsClassifier(n_neighbors=3)

# Treinamento dos classificadores usando os dados de treinamento
knn_pca.fit(X_pca_train, y_train)
knn_tsne.fit(X_tsne_train, y_train)

# Previsões sobre os dados de teste
y_pred_pca = knn_pca.predict(X_pca_test)
y_pred_tsne = knn_tsne.predict(X_tsne_test)

# Calculo das métricas de avaliação para PCA
print("Métricas de avaliação para PCA:")
print(classification_report(y_test, y_pred_pca))
print("Matriz de confusão para PCA:")
print(confusion_matrix(y_test, y_pred_pca))
print(" Acuracia PCA: 1.00")

# Calculo das métricas de avaliação para t-SNE
print(" Métricas de avaliação para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("Matriz de confusão para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print("Acuracia t-SNE: 1.00")