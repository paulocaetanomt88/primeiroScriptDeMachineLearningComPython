<h1>Machine Learning para Análise de Amostras de Vinhos com Python</h1>
<h3>Utilizando a Ferramenta Anaconda Jupyter Notebook</h3>

```python
# Importanto a biblioteca Pandas
import pandas as pd
# carregando o conjunto de dados
arquivo = pd.read_csv('wine_dataset.csv')
```


```python
# selecionando somente as primeiras linhas
arquivo.head()
```




<div>

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>style</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




```python
# substituindo o texto da coluna style por números para serem tratados pelo python
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)
```


```python
# separando as variáveis entre preditora e variáveis alvo
y = arquivo['style']
x = arquivo.drop('style', axis=1) ## axis = 0 -> linha || ou axis = 1 -> coluna
```


```python
from sklearn.model_selection import train_test_split

# criando os conjuntos de dados de treino e teste, pois o modelo vai receber alguns dados para treinar  (aprender) 
# e depois a gente vai testar pra ver se está fazendo certo

# passando os dados para a função train_test_split e especificando a porcentagem 30% para teste e 70% (o restante) para treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)
```


```python
y_teste.shape
```




    (1950,)




```python
# importando algoritmo que cria árvores de decisão
from sklearn.ensemble import ExtraTreesClassifier

# criação do modelo
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)
```




    ExtraTreesClassifier()




```python
# imprimindo resultados

resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)
```

    Acurácia: 0.9953846153846154
    


```python
# selecionando algumas amostras aleatoriamente
```


```python
y_teste[400:403]
```




    4295    1
    3168    1
    5109    1
    Name: style, dtype: int64




```python
x_teste[400:403]
```




<div>

    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4295</th>
      <td>6.5</td>
      <td>0.22</td>
      <td>0.50</td>
      <td>16.4</td>
      <td>0.048</td>
      <td>36.0</td>
      <td>182.0</td>
      <td>0.99904</td>
      <td>3.02</td>
      <td>0.49</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3168</th>
      <td>6.7</td>
      <td>0.30</td>
      <td>0.74</td>
      <td>5.0</td>
      <td>0.038</td>
      <td>35.0</td>
      <td>157.0</td>
      <td>0.99450</td>
      <td>3.21</td>
      <td>0.46</td>
      <td>9.9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5109</th>
      <td>7.0</td>
      <td>0.22</td>
      <td>0.24</td>
      <td>11.0</td>
      <td>0.041</td>
      <td>75.0</td>
      <td>167.0</td>
      <td>0.99508</td>
      <td>2.98</td>
      <td>0.56</td>
      <td>10.5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
previsoes = modelo.predict(x_teste[400:403])
```


```python
previsoes
```




    array([1, 1, 1], dtype=int64)



<h1>Conclusões</h1>

O algoritmo conseguiu prever com exatidão, baseados nas informações fornecidas nas outras colunas, que as três amostras no vetor "array([1, 1, 1])" são do tipo vinho branco.
