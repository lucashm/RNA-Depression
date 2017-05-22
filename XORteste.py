#!-*- coding: utf8 -*-
# *********************************************************
# Rede FeedFoward
# Multplayer Perceptron
# Algoritmo Backpropagation
# Função Sigmoide
#
# ********************************************************

# Aprendizagem Supervisionada padrão - SupervisedDataSet
#    Conjunto de Entradas que tenha entradas e alvos.
from pybrain.datasets import SupervisedDataSet

# Atalho BuildNetwork
#    Essa chamada retorna uma rede que tem duas entradas, três oculto e um único
#    neurônio de saída.
#    Na camada oculta, é construída a função Sigmoide por padrão.
from pybrain.tools.shortcuts import buildNetwork
# Para ajustar os parâmetros dos módulos na aprendizagem supervisionada usando
# backpropagation.

from pybrain.supervised import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tests.helpers import gradientCheck

# Cria um conjunto de dados (dataset) para treinamento. São passadas as dimensões
#   dos vetores e de entrada e do objetivo.
ds = SupervisedDataSet(4,1) # Suporta entradas quadmensionais e alvos dimensionais


# Adiciona as amostras que consiste numa entrada e num objetivo.
ds.addSample((0,0,0,0), (0,))
ds.addSample((0,0,0,1), (1,))
ds.addSample((0,0,1,0), (1,))
ds.addSample((0,0,1,1), (0,))
ds.addSample((0,1,0,0), (0,))
ds.addSample((0,1,0,1), (0,))
ds.addSample((0,1,1,0), (1,))
ds.addSample((0,1,1,1), (0,))
ds.addSample((1,0,0,0), (0,))
ds.addSample((1,0,0,1), (0,))
ds.addSample((1,0,1,0), (1,))
ds.addSample((1,0,1,1), (1,))
ds.addSample((1,1,0,0), (0,))
ds.addSample((1,1,0,1), (0,))
ds.addSample((1,1,1,0), (1,))
ds.addSample((1,1,1,1), (1,))

for inpt, target in ds:
    print inpt, target

# Dataset.indim -> é o tamanho da camada de entrada;
# Número (8) -> é a quantidade de camadas intermediárias;
# Dataset.outdim -> é o tamanho da camada de saída;
# Bias -> adaptação por parte da rede neural ao conhecimento à ela fornecido.
# hiddenclass -> Função de ativação
INTERMEDIATE_LAYERS_QUANTITY = 8
net = buildNetwork(ds.indim, INTERMEDIATE_LAYERS_QUANTITY, ds.outdim, bias=True)
counter = 0;

# Verificação de gradiente
print 'gradient check'
gradientCheck(net)

# Os trainers tomam um módulo e um conjunto de dados para treinar o módulo para ajustar os dados no conjunto de dados.
#  learningrate -> taxa de aprendizado
#  momentum -> velocidade de treinamento
#  verbose=True -> indica que deve ser impressas mensagens
                                                                            #Trocar essas linhas para:
trainer = BackpropTrainer(net,ds, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001) # trainer = BackpropTrainer(net,ds)
                                                                                #lrdecay -> lrdecay * learningrate
                                                                                # a cada época
# Treinamos a rede 400 vezes
trainer.trainOnDataset(ds,400)

# Testamos nosso modelo
trainer.testOnData(verbose=True)

# Vamos se melhorou depois de treinar
print "ativando 0,0,0,0", net.activate([0,0,0,0])
print "ativando 0,0,0,1", net.activate([0,0,0,1])
print "ativando 0,0,1,0", net.activate([0,0,1,0])
print "ativando 0,0,1,1", net.activate([0,0,1,1])
print "ativando 0,1,0,0", net.activate([0,1,0,0])
print "ativando 0,1,0,1", net.activate([0,1,0,1])
print "ativando 0,1,1,0", net.activate([0,1,1,0])
print "ativando 0,1,1,1", net.activate([0,1,1,1])
print "ativando 1,0,0,0", net.activate([1,0,0,0])
print "ativando 1,0,0,1", net.activate([1,0,0,1])
print "ativando 1,0,1,0", net.activate([1,0,1,0])
print "ativando 1,0,1,1", net.activate([1,0,1,1])
print "ativando 1,0,1,1", net.activate([1,1,0,0])
print "ativando 1,0,1,1", net.activate([1,1,0,1])
print "ativando 1,0,1,1", net.activate([1,1,1,0])
print "ativando 1,0,1,1", net.activate([1,1,1,1])

EPOCH_VALUE_MAX = 3000
for epoch in range(0, EPOCH_VALUE_MAX):                                        # for epoch in range(0, 10000):
    training = trainer.train();
    counter = counter + 1

    if training < 0.001:
        break

print "Contador de épocas: ", counter



#print "\nWeights: ", net.params
print "\n\n"
x = input('Exaustão física? ')
y= input('Exaustão mental? ')
w= input('Culpabilidade? ')
z= input('Crises de ansiedade? ')

print 'Será?', net.activate([x,y,w,z])
