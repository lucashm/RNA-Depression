#!-*- coding: utf8 -*-
# Aprendizagem Supervisionada padrão - SupervisedDataSet
#   Conjunto de Entradas que tenha entradas e alvos.
from pybrain.datasets import SupervisedDataSet
import time
# Atalho BuildNetwork
#   Essa chamada retorna uma rede que tem duas entradas, três oculto e um único
#   neurônio de saída.
#   Na camada oculta, é construída a função Sigmoide por padrão
from pybrain.tools.shortcuts import buildNetwork
# Para ajustar os parâmetros dos módulos na aprendizagem supervisionada usando
# backpropagation.

from pybrain.supervised import BackpropTrainer
from pybrain.structure import SoftmaxLayer

# Cria um conjunto de dados (dataset) para treinamento. São passadas as dimensões
#   dos vetores e de entrada e do objetivo.
ds = SupervisedDataSet(4,1) # Suporta entradas quadmensionais e alvos dimensionais


# Adiciona as amostras que consiste numa entrada e num objetivo.
ds.addSample((0,0,0,0), (0,))
ds.addSample((0,0,0,1), (0,))
ds.addSample((0,0,1,0), (0,))
ds.addSample((0,0,1,1), (1,))
ds.addSample((0,1,0,0), (0,))
ds.addSample((0,1,0,1), (1,))
ds.addSample((0,1,1,0), (1,))
ds.addSample((0,1,1,1), (1,))
ds.addSample((1,0,0,0), (0,))
ds.addSample((1,0,0,1), (1,))
ds.addSample((1,0,1,0), (1,))
ds.addSample((1,0,1,1), (1,))
ds.addSample((1,1,0,0), (1,))
ds.addSample((1,1,0,1), (1,))
ds.addSample((1,1,1,0), (1,))
ds.addSample((1,1,1,1), (1,))



for inpt, target in ds:
    print inpt, target

# Dataset.indim -> é o tamanho da camada de entrada;
# Número (8) -> é a quantidade de camadas intermediárias;
# Dataset.outdim -> é o tamanho da camada de saída;
# Bias -> adaptação por parte da rede neural ao conhecimento à ela fornecido.
INTERMEDIATE_LAYERS_QUANTITY = 8
net = buildNetwork(ds.indim, INTERMEDIATE_LAYERS_QUANTITY, ds.outdim, bias=True)
counter = 0;

# Os trainers tomam um módulo e um conjunto de dados para treinar o módulo para ajustar os dados no conjunto de dados.
#  learningrate -> taxa de aprendizado
#  momentum -> velocidade de treinamento
# # verbose=True -> indica que deve ser impressas mensagens

trainer = BackpropTrainer(net,ds, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001)
errorList = []                                                                                #lrdecay -> lrdecay * learningrate

EPOCH_VALUE_MAX = 3000
for epoch in range(0, EPOCH_VALUE_MAX):                                        # for epoch in range(0, 10000):
    training = trainer.train();
    errorList.insert(epoch, training)
    if training < 0.001:
        break


print "\n\n"
a = input('Exaustão física ou mental?')
b = input('Culpabilidade?')
c = input('Crises de ansiedade?')
d = input('Insonia ou sono excessivo?')

ativa = net.activate([a,b,c,d])
print 'Ativa?', ativa

if ativa <= 1.2 and ativa >= 0.8:
    entrada_um = 1
else:
    entrada_um = 0

print "Processando o próximo passo, aguarde..."
time.sleep(5.0)
##############FIM DO PRIMEIRO ROUND#####################

ds2 = SupervisedDataSet(4,1) # Suporta entradas quadmensionais e alvos dimensionais


# Adiciona as amostras que consiste numa entrada e num objetivo.
ds2.addSample((0,0,0,0), (0,))
ds2.addSample((0,0,0,1), (0,))
ds2.addSample((0,0,1,0), (0,))
ds2.addSample((0,0,1,1), (1,))
ds2.addSample((0,1,0,0), (0,))
ds2.addSample((0,1,0,1), (1,))
ds2.addSample((0,1,1,0), (1,))
ds2.addSample((0,1,1,1), (1,))
ds2.addSample((1,0,0,0), (0,))
ds2.addSample((1,0,0,1), (1,))
ds2.addSample((1,0,1,0), (1,))
ds2.addSample((1,0,1,1), (1,))
ds2.addSample((1,1,0,0), (1,))
ds2.addSample((1,1,0,1), (1,))
ds2.addSample((1,1,1,0), (1,))
ds2.addSample((1,1,1,1), (1,))

INTERMEDIATE_LAYERS_QUANTITY = 8
net2 = buildNetwork(ds2.indim, INTERMEDIATE_LAYERS_QUANTITY, ds2.outdim, bias=True)

trainer2 = BackpropTrainer(net2,ds2, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001)
errorList2 = []

for epoch2 in range(0, EPOCH_VALUE_MAX):
    training2 = trainer2.train();
    errorList2.insert(epoch2, training2)
    if training2 < 0.001:
        break

e = input('Trauma recente?')
f = input('Pensamentos Suicidas?')
g = input('Alucinações ou Delirios?')

ativa2 = net2.activate([entrada_um,e,f,g])
print 'Ativa2?', ativa2

print "Processando o próximo passo, aguarde..."
time.sleep(5.0)
##################### FIM DO SEGUNDO ROUND ########################

ds3 = SupervisedDataSet(6,1) # Suporta entradas quadmensionais e alvos dimensionais


# Adiciona as amostras que consiste numa entrada e num objetivo.
ds3.addSample((0,0,0,0,0,0), (0,))
ds3.addSample((0,0,0,0,0,1), (0,))
ds3.addSample((0,0,0,0,1,0), (0,))
ds3.addSample((0,0,0,0,1,1), (0,))
ds3.addSample((0,0,0,1,0,0), (0,))
ds3.addSample((0,0,0,1,0,1), (0,))
ds3.addSample((0,0,0,1,1,0), (0,))
ds3.addSample((0,0,0,1,1,1), (0,))
ds3.addSample((0,0,1,0,0,0), (0,))
ds3.addSample((0,0,1,0,0,1), (0,))
ds3.addSample((0,0,1,0,1,0), (0,))
ds3.addSample((0,0,1,0,1,1), (0,))
ds3.addSample((0,0,1,1,0,0), (0,))
ds3.addSample((0,0,1,1,0,1), (0,))
ds3.addSample((0,0,1,1,1,0), (0,))
ds3.addSample((0,0,1,1,1,1), (0,))
ds3.addSample((0,1,0,0,0,0), (0,))
ds3.addSample((0,1,0,0,0,1), (0,))
ds3.addSample((0,1,0,0,1,0), (0,))
ds3.addSample((0,1,0,0,1,1), (0,))
ds3.addSample((0,1,0,1,0,0), (0,))
ds3.addSample((0,1,0,1,0,1), (0,))
ds3.addSample((0,1,0,1,1,0), (0,))
ds3.addSample((0,1,0,1,1,1), (0,))
ds3.addSample((0,1,1,0,0,0), (0,))
ds3.addSample((0,1,1,0,0,1), (0,))
ds3.addSample((0,1,1,0,1,0), (0,))
ds3.addSample((0,1,1,0,1,1), (0,))
ds3.addSample((0,1,1,1,0,0), (0,))
ds3.addSample((0,1,1,1,0,1), (0,))
ds3.addSample((0,1,1,1,1,0), (0,))
ds3.addSample((0,1,1,1,1,1), (0,))
ds3.addSample((1,0,0,0,0,0), (0,))
ds3.addSample((1,0,0,0,0,1), (1,))
ds3.addSample((1,0,0,0,1,0), (1,))
ds3.addSample((1,0,0,0,1,1), (1,))
ds3.addSample((1,0,0,1,0,0), (1,))
ds3.addSample((1,0,0,1,0,1), (1,))
ds3.addSample((1,0,0,1,1,0), (1,))
ds3.addSample((1,0,0,1,1,1), (1,))
ds3.addSample((1,0,1,0,0,0), (1,))
ds3.addSample((1,0,1,0,0,1), (1,))
ds3.addSample((1,0,1,0,1,0), (1,))
ds3.addSample((1,0,1,0,1,1), (1,))
ds3.addSample((1,0,1,1,0,0), (1,))
ds3.addSample((1,0,1,1,0,1), (1,))
ds3.addSample((1,0,1,1,1,0), (1,))
ds3.addSample((1,0,1,1,1,1), (1,))
ds3.addSample((1,1,0,0,0,0), (1,))
ds3.addSample((1,1,0,0,0,1), (1,))
ds3.addSample((1,1,0,0,1,0), (1,))
ds3.addSample((1,1,0,0,1,1), (1,))
ds3.addSample((1,1,0,1,0,0), (1,))
ds3.addSample((1,1,0,1,0,1), (1,))
ds3.addSample((1,1,0,1,1,0), (1,))
ds3.addSample((1,1,0,1,1,1), (1,))
ds3.addSample((1,1,1,0,0,0), (1,))
ds3.addSample((1,1,1,0,0,1), (1,))
ds3.addSample((1,1,1,0,1,0), (1,))
ds3.addSample((1,1,1,0,1,1), (1,))
ds3.addSample((1,1,1,1,0,0), (1,))
ds3.addSample((1,1,1,1,0,1), (1,))
ds3.addSample((1,1,1,1,1,0), (1,))
ds3.addSample((1,1,1,1,1,1), (1,))


INTERMEDIATE_LAYERS_QUANTITY3 = 16
net3 = buildNetwork(ds3.indim, INTERMEDIATE_LAYERS_QUANTITY3, ds3.outdim, bias=True)

trainer3 = BackpropTrainer(net3,ds3, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001)
errorList3 = []

for epoch3 in range(0, EPOCH_VALUE_MAX):
    training3 = trainer3.train();
    errorList3.insert(epoch3, training3)
    if training3 < 0.001:
        break

print net3.activate([1,1,1,1,1,1])
print net3.activate([0,1,1,1,1,1])

#h = input('Variações de humor constantes?')
#i = input('Desinteresse por quaisquer tipos de atividades?')
#j = input('Medo de ser rejeitado?')
#k = input('Paranoias?')
#l = input('Sente-se inutil na maioria das vezes?')
