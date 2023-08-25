# AVALIAÇÃO 3

## Dataset
https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting

### Questão 1

```questao1.py```

Faça uma análise do dataset utilizando dendograma. Verifique as possibilidades 
de clusterização e aplique o k-medias. Observe os resultados e descreva sua iterpretação
no relatório.
Dica: Observe se há necessidade de normalização dos dados nas colunas ou nas linhas.


### Questão 2

```questao2.py```
Reduza o dataset T-SNE e com PCA para duas dimensões. Plote o gráfico do atributos que as duas técnicas geraram.
De forma subjetiva e visual, qual dos gráficos você avredita que vai possuir um melhor
desempenho em um processo de classificação utilizando os dois atribuitos ?

### Questão 3

```questao3.py```

Utilizando os dados da questão 2, aplique algum método de classificação e gere números
que quantificam o desempenho deste. Compare os números classificando o dataset reduzido pelo PCA e pelo T-SNE.


### Questão 4

```questao4.py```

Utilizando análise de variância do PCA. Reduza a dimensão para realizar uma classificação utilizando somente as colunas de maior variância.
Aplique o mesmo método de classificação testado na questão 3. Gere os mesmos números que analisam o desempenho do classificador e verifique se houve melhoria no resultado.


### Questão 5

```questao5.py```

Você descobriu qual a melhor forma de pré-processar os dados. Assim, utilizando a metodologia que proporcionou o melhor acerto do classficador faça agora uma comparação 
entre classicadores para que você também possa descobrir qual classificador mais adequado. Utilize outra técnica de classificação com os mesmo dados, gere os numeros que quantificam o 
desempenho e faça uma comparação entre estes.
Conclua o relatótório  com auxílio de um fluxogragrama mostrando qual a metodologia completa para classificação dos dados do seu dataset.

