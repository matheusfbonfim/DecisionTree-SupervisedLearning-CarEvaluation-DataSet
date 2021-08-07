###########################
## BIBLIOTECAS IMPORTADAS

# Bibliotecas básicas
import pandas as pd
import os
import numpy as np

# Gráficos e tabelas
import pydotplus

# Algoritmo para treinar o modelo de arvore de decisao
from sklearn import tree
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# -----------------------------------------------------------

###########################
## IMPORTANDO O DATASET (Importando o csv para o dataframe)

# Importando dataset e nomeando as labels de entrada e saida (colunas)
database = pd.read_csv("../database/data_folder/car.data", names = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "classification"])

# Verificando as 5 primeiras linhas do database
print("="*60)
print("\t\t\tPRÉ VISUALIZAÇÃO DA BASE DE DADOS")
print("="*60)
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

    # 0 para unacc, 1 para acc, 2 para vgood,3 para good,
database['classification'], names_classification = pd.factorize(database['classification'])

    # Codigos inteiros equivalentes aos tipos de classificações
print(names_classification) # Classes de saída
print(database['classification'].unique(),'\n')

# -------------------
# Convertendo as features de entrada de strings para um formato numérico para identificação pelo algoritmo

    # buying -> 0: v-high, 1: high, 2: med, 3: low
database['buying'], _ = pd.factorize(database['buying'])

    # maintenance -> 0: v-high, 1: high, 2: med, 3: low
database['maintenance'], _ = pd.factorize(database['maintenance'])

    # doors -> 0: 2, 1: 3, 2: 4, 3: 5 - more
database['doors'], _ = pd.factorize(database['doors'])

    # persons ->  0: 2, 1: 4, 2: more
database['persons'], _ = pd.factorize(database['persons'])

    # lug_boot -> 0: small, 1: med, 2: big
database['lug_boot'], _ = pd.factorize(database['lug_boot'])

    # safety ->  0: low, 1: med, 2: high
database['safety'], _ = pd.factorize(database['safety'])

# -------------------
# Verificando as 5 primeiras linhas do database
print("="*60)
print("\tPRÉ VISUALIZAÇÃO DA BASE DE DADOS COM RÓTULOS ATUALIZADOS")
print("="*60)
print(database.head(),'\n')


############################
## SEPARANDO FEATURES E ROTULOS E CONVERTER PARA MATRIZ NUMPY

# É necessário converter o database em uma matriz numpy para scikit learn
# Além disso, separar as features de entrada e as labels

# -------------------
# Definindo a matriz de features
    # Escolhemos um conjunto de recursos manualmente e os convertemos
columns = ["buying", "maintenance", "doors", "persons", "lug_boot", "safety"]
X = database[list(columns)].values

# -------------------
# Definindo rótulos de saída como classification
    # Cada classificação é o alvo que queremos prever a partir dos valores das outras colunas (entradas)
Y = database["classification"].values


############################
## DIVIDIR O DATASET ENTRE TREINO E TESTE

    # Dataset de treino - 70% dos dados
    # Dataset de teste - 30% dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)


############################
## ÁRVORE DE DECISÃO

# Criar um algoritmo que será do tipo de arvore de decisão
  # Criando o algoritmo de arvore de decisão
  # DecisionTreeClassifier - Classificação de arvore de decisão
  # Criterios -> Existem outras estrategias mas usamos a entropia
  # max_depth -> Altura máxima da arvore

# Learn the decision tree
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Treina o algoritmo -> Fazer a logica de divisão
model = clf.fit(X_train, Y_train)


###########################
# ENTENDENDO O RESULTADO DA ÁRVORE

# Montando a imagem da arvore

# Export as png or pdf
dot_data = tree.export_graphviz(model, out_file=None, feature_names=columns,
                                class_names=names_classification, filled=True,
                                rounded=True, special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png('Tree_Decision_Car_Acceptability.png')
