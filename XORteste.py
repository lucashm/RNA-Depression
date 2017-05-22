#!-*- coding: utf8 -*-
# Aprendizagem Supervisionada padrão - SupervisedDataSet
#   Conjunto de Entradas que tenha entradas e alvos.
from pybrain.datasets import SupervisedDataSet

# Atalho BuildNetwork
#   Essa chamada retorna uma rede que tem duas entradas, três oculto e um único
#   neurônio de saída.
#   Na camada oculta, é construída a função Sigmoide por padrão
from pybrain.tools.shortcuts import buildNetwork
# Para ajustar os parâmetros dos módulos na aprendizagem supervisionada usando
# backpropagation.

from pybrain.supervised import BackpropTrainer
from pybrain.structure import SoftmaxLayer

# Coisas a serem feitas:
# Descobrir quais sao as variaveis importantes
# Ligar elas ao conteudo da professora, pra quando ela perguntar e querer que algo seja mudado
# docs do pyBrain: http://pybrain.org/docs/
# deem uma pesquisada, se eu tiver feito algo errado, ou se tiver um jeito melhor, a vontade


ds = SupervisedDataSet(4,1) # Suporta entradas quadmensionais e alvos dimensionais

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


net = buildNetwork(ds.indim, 8, ds.outdim, bias=True)
counter = 0;
# Os trainers tomam um módulo e um conjunto de dados para treinar o módulo para ajustar os dados no conjunto de dados.
                                                                            #Trocar essas linhas para:
trainer = BackpropTrainer(net,ds, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001) # trainer = BackpropTrainer(net,ds)
                                                                                #lrdecay -> lrdecay * learningrate
                                                                                # a cada época

for epoch in range(0, 3000):                                        # for epoch in range(0, 10000):
    training = trainer.train();
    counter = counter + 1

    if training < 0.001:
        break

print "Contador de epocas: ", counter



#print "\nWeights: ", net.params
print "\n\n"
x = input('Exaustão física?')
y= input('Exaustão mental?')
w= input('Culpabilidade?')
z= input('Crises de ansiedade?')

print 'Será?', net.activate([x,y,w,z])
