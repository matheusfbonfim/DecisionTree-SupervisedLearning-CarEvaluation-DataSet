###########################
## BIBLIOTECAS IMPORTADAS

# Bibliotecas básicas
import numpy as np
import pandas as pd
import itertools

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

# Importando dataset e nomeando as labels de entrada e saida (colunas)
database = pd.read_csv("../database/data_folder/car.data", names = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "classification"])

# Verificando as 5 primeiras linhas do database
print("\tPRÉ VISUALIZAÇÃO DA BASE DE DADOS\n")
print(database.head(),'\n')

# Número de linhas e colunas
num_linhas = database.shape[0]
num_colunas = database.shape[1]
print(f"Numero de linhas: {num_linhas} e colunas: {num_colunas}\n")


###########################
## VERIFICAR NO DATASET A QUANTIDADE DE REGISTROS PARA CADA CATEGORIA DE VALOR

# classification - a variável-alvo com valor de:
    # unacc (inaceitavel) / acc (aceitavel)
    # good (bom) / vgood (muito bom).
print(f"Total de registros com a classificação de unacc: {database[database['classification'] == 'unacc'].shape[0]}")
print(f"Total de registros com a classificação de acc: {database[database['classification'] == 'acc'].shape[0]}")
print(f"Total de registros com a classificação de good: {database[database['classification'] == 'good'].shape[0]}")
print(f"Total de registros com a classificação de v-good: {database[database['classification'] == 'vgood'].shape[0]}\n")


############################
## ALTERANDO OS RÓTULOS

# -------------------
# Os rótulos da classificação são strings, assim é necessário converter em um formato inteiro para que o algoritmo entenda
    # 1 para unacc, 2 para acc, 3 para good, 4 para vgood

database.classification[database["classification"] == 'unacc'] = 1
database.classification[database["classification"] == 'acc'] = 2
database.classification[database["classification"] == 'good'] = 3
database.classification[database["classification"] == 'vgood'] = 4

# -------------------
# Convertendo as features de entrada de strings para um formato numérico para identificação pelo algoritmo

inputs_features = list(database.columns)    # Features de entrada e rotulo de saida
inputs_features.remove("classification")    # Removendo rotulo de saída

print(f"Features de entrada: {inputs_features}")

# Dicionario para armazenar os uniques de cada feature
uniques_features = {"buying": [],"maintenance": [], "doors": [], "persons": [], "lug_boot": [], "safety": []}

for i in inputs_features:
    # Utilizando factorize -> Variaveis categoricas para variaveis indicadoras
    database[i],  uniques_features[i] = pd.factorize(database[i])

print(f"Uniques: \n {uniques_features}\n")

# -------------------
# Verificando as 5 primeiras linhas do database
print("\tPRÉ VISUALIZAÇÃO DA BASE DE DADOS COM RÓTULOS ATUALIZADOS\n")
print(database.head(),'\n')


############################
## SEPARANDO FEATURES E ROTULOS E CONVERTER PARA MATRIZ NUMPY

# É necessário converter o database em uma matriz numpy para scikit learn
# Além disso, separar as features de entrada e as labels

# -------------------
# Definindo rótulos de saída como classification
    # Cada classificação é o alvo que queremos prever a partir dos valores das outras colunas (entradas)
Y = database['classification'].values # Usa values para converter o dataframe do pandas para o array numpy
print(f"Rotulos de saida: {Y}\n")

# -------------------
# Definindo a matriz de features
    # Escolhemos um conjunto de recursos manualmente e os convertemos
X = database[["buying", "maintenance", "doors", "persons", "lug_boot", "safety"]].values
print(f"Matriz com as features de entrada: \n{X}\n")


############################
## DIVIDIR O DATASET ENTRE TREINO E TESTE

    # Dataset de treino - 75% dos dados
    # Dataset de teste - 25% dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 3)


############################
## ÁRVORE DE DECISÃO

# Criar um algoritmo que será do tipo de arvore de decisão
  # Criando o algoritmo de arvore de decisão
  # DecisionTreeClassifier - Classificação de arvore de decisão
  # Criterios -> Existem outras estratpegias mas usamos a entropia
  # max_depth -> Altura máxima da arvore

# # Learn the decision tree
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 5)

# Treina o algoritmo -> Fazer a logica de divisão
model = clf.fit(X_train, Y_train)

# ###########################
# # ENTENDENDO O RESULTADO DA ÁRVORE
#
# # Feature mais importante
# print(f"Feature mais importante: {model.feature_importances_}")
#
# # Labels para features
# features_names = ['battery_power', 'blue', 'clock_speed','dual_sim','fc']
# print(f"Features names: {features_names}")
#
# # Nomes das classes - Rótulos
# classes_names = ['low_cost', 'medium_cost', 'high_cost', 'very_high_cost']
# print(f"Classes names - Rotulos: {classes_names}")
#
# # MONTAR A IMAGEM DA ÁRVORE
# dot_data = StringIO()
#
# #dot_data = tree.export_graphviz(my_tree_one, out_file=None, feature_names=featureNames)
# export_graphviz(model, out_file=dot_data, filled=True, feature_names=features_names, class_names=classes_names, rounded=True, special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
# graph.write_png("arvore.png")
# Image('arvore.png')