###########################
## BIBLIOTECAS IMPORTADAS

# Bibliotecas básicas
import numpy as np
import itertools
import pandas as pd

# Gráficos e tabelas
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
from IPython.display import Image

# Algoritmo para treinar o modelo de arvore de decisao
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Métricas para avaliação de modelo
from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from six import StringIO

# -----------------------------------------------------------

###########################
## IMPORTANDO O DATASET (Importando o csv para o dataframe)
database_train = pd.read_csv("../database/train.csv")

# Verificando as 5 primeiras linhas do database de train
print("\t\t\tPRÉ VISUALIZAÇÃO DA BASE DE DADOS DE TREINO")
print(database_train.head(5),'\n')

# Número de linhas e colunas
num_linhas = database_train.shape[0]
num_colunas = database_train.shape[1]
print(f"Numero de linhas: {num_linhas} e colunas: {num_colunas}\n")

###########################
# VERIFICAR NO DATASET A QUANTIDADE DE REGISTROS PARA CADA CATEGORIA DE VALOR
# price_range - a variável-alvo com valor de:
    # 0 (custo baixo) / 1 (custo médio)
    # 2 (custo alto) / 3 (custo muito alto).
print(f"Total de registros com a categoria de valor de custo baixo: {database_train[database_train['price_range'] == 0].shape[0]}")
print(f"Total de registros com a categoria de valor de custo médio: {database_train[database_train['price_range'] == 1].shape[0]}")
print(f"Total de registros com a categoria de valor de custo alto: {database_train[database_train['price_range'] == 2].shape[0]}")
print(f"Total de registros com a categoria de valor de custo alto: {database_train[database_train['price_range'] == 3].shape[0]}\n")

###########################
# CONVERTER
# É necessário converter o database_train em uma matriz numpy para scikit learn
# Além disso, separar as features de entrada e as labels (price_range)

# Definindo rótulos de saída como price_range
    # Cada categoria de valor é o alvo que queremos prever a partir dos valores das outras colunas
Y = database_train['price_range'].values # Usa values para converter o dataframe do pandas para o array numpy
print(f"Rotulos de saida: {Y}\n")

# Definindo a matriz de features
    # Escolhemos um conjunto de recursos manualmente e os convertemos
X = database_train[['battery_power', 'blue','clock_speed', 'dual_sim', 'fc']].values
print(f"Matriz com as features de entrada: \n{X}\n")

###########################
# ÁRVORE DE DECISÃO

# Criar um algoritmo que será do tipo de arvore de decisão
  # Criando o algoritmo de arvore de decisão
  # DecisionTreeClassifier - Classificação de arvore de decisão
  # Criterios -> Existem outras estratpegias mas usamos a entropia
  # max_depth -> Altura máxima da arvore

# Learn the decision tree
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 5)

# Treina o algoritmo -> Fazer a logica de divisão
model = clf.fit(X, Y)

###########################
# ENTENDENDO O RESULTADO DA ÁRVORE

# Feature mais importante
print(f"Feature mais importante: {model.feature_importances_}")

# Labels para features
features_names = ['battery_power', 'blue', 'clock_speed','dual_sim','fc']
print(f"Features names: {features_names}")

# Nomes das classes - Rótulos
classes_names = model.classes_
print(f"Classes names - Rotulos: {classes_names}")

# MONTAR A IMAGEM DA ÁRVORE
dot_data = StringIO()

#dot_data = tree.export_graphviz(my_tree_one, out_file=None, feature_names=featureNames)
export_graphviz(model, out_file=dot_data, filled=True, feature_names=features_names, class_names=classes_names, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("arvore.png")
Image('arvore.png')