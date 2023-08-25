# AVALIAÇÃO 2

## Dataset
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

### Questão 1

```questao1.py```

Verifique qual atributo será o alvo para regressão no seu dataset
e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
Lembre de comprovar via gráfico.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.


### Questão 2

```questao2.py```

Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando somente este atributo mais
relevante, para predição do atributo alvo determinado na questão 1 também. Mostre o gráfico da reta de regressão  em conjunto com a nuvem 
de atributo. 
Determine também os valores: 
RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

### Questão 3

```questao3.py```

Remova os atributos que não são relevantes para o processo de regressão e realize um gridsearch cross-validation para verificar 
qual a melhor parametrização para os regressores de Lasso e Ridge. Print as melhores configurações de cada um mostre também os melhores scores.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

### Questão 4

```questao4.py```

Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos definidos na questão 3.
Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma comparação entre os desempenhos. Escolha o regressor adequado
e informe o motivo da escolha. Discuta sobre as limitações e acertos encontrados.
