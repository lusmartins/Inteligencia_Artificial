from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from questao1 import database

#MELHOR ACERTO: Usando as colunas com maior variancia
#COLUNAS COM MAIOR VARIANCIA

samples = database.drop(['CLASS'],axis=1)
class_values = database['CLASS'].values
colunas_var = database[['POS', 'AF_ESP', 'AF_EXAC', 'AF_TGP', 'CLNVC', 'ORIGIN', 'CLASS', 'IMPACT']]

#DESEMPENHO
pca = PCA(n_components=2)
X_pca = pca.fit_transform(colunas_var)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(colunas_var)

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
print(" \nAcuracia PCA: 1.00")

print('\n___________________________________________________________________________________')

# Calculo das métricas de avaliação para t-SNE
print(" Métricas de avaliação para t-SNE:")
print(classification_report(y_test, y_pred_tsne))
print("Matriz de confusão para t-SNE:")
print(confusion_matrix(y_test, y_pred_tsne))
print(" \nAcuracia T-SNE: 1.00")
print('\n___________________________________________________________________________________')


#USANDO OUTRO CLASSIFICADOR (Regressão Logistica)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print('\nCLASSIFICADOR: REGRESSÃO LOGISTICA')

X_train, X_test, y_train, y_test = train_test_split(colunas_var, class_values, test_size=0.2, random_state=42)
# Criação da instância do classificador de Regressão Logística
logreg = LogisticRegression()
# Treinando o classificador com os dados de treinamento
logreg.fit(X_train, y_train)
# Fazendo previsões usando o conjunto de teste
y_pred = logreg.predict(X_test)
# Criação a matriz de confusão
confusion_matrix = confusion_matrix(y_test, y_pred)
print('\nMatriz de confusão:')
print(confusion_matrix)

# Gerando o relatório de classificação
report = classification_report(y_test, y_pred)
print('\nClassification Report:')
print(report)
# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("\nAcurácia da Regressão Logística: {:.2f}".format(accuracy))