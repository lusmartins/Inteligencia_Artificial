from questao1 import database
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
import numpy as np

#Utilização do kfold e cross-validation para criar uma regressão linear utilizando os mesmos atributos definidos na questão 3
X = database["price"].values.reshape(-1, 1)
y = database["duration"].values

parametro_lasso = {'alpha': 0.01}
parametro_ridge = {'alpha': 10}

regressor_lasso = Lasso(**parametro_lasso)
regressor_ridge = Ridge(**parametro_ridge)

kf_lasso = KFold(n_splits=6, shuffle=True, random_state=5)
score_lasso=cross_val_score(regressor_lasso, X, y, cv=kf_lasso)
kf_ridge = KFold(n_splits=6, shuffle=True, random_state=5)
score_ridge=cross_val_score(regressor_ridge, X, y, cv=kf_ridge)

print("Score médio para Lasso:", np.mean(score_lasso))
print("Score médio para Ridge:", np.mean(score_ridge))












