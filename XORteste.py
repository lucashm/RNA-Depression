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
INTERMEDIATE_LAYERS_QUANTITY = 8
net = buildNetwork(ds.indim, INTERMEDIATE_LAYERS_QUANTITY, ds.outdim, bias=True)
counter = 0;

# Os trainers tomam um módulo e um conjunto de dados para treinar o módulo para ajustar os dados no conjunto de dados.
#  learningrate -> taxa de aprendizado
#  momentum -> velocidade de treinamento
# # verbose=True -> indica que deve ser impressas mensagens
                                                                            #Trocar essas linhas para:
trainer = BackpropTrainer(net,ds, learningrate=0.001, momentum=0.99, verbose=True, lrdecay=1.0001) # trainer = BackpropTrainer(net,ds)
                                                                                #lrdecay -> lrdecay * learningrate
                                                                                # a cada época
EPOCH_VALUE_MAX = 3000
for epoch in range(0, EPOCH_VALUE_MAX):                                        # for epoch in range(0, 10000):
    training = trainer.train();
    counter = counter + 1

    if training < 0.001:
        break

print "Contador de épocas: ", counter



# print "\nWeights: ", net.params
print "\n\n"

# Round I
# Precisa de no inimo 2 bits pra definir que tem depressão
a = input('Exaustão física ou mental?')
b = input('Culpabilidade?')
c = input('Crises de ansiedade?')
d = input('Insonia ou sono excessivo?')

# Round II
e = input('Trauma recente?')
f = input('Pensamentos Suicidas?')
g = input('Alucinações ou Delirios?')

# Round III
# situacional => e
# distimia => i and j and m
# psicotica => g and k
# atipica => i and j and l
# bipolar => h
# maior => f and i and j and l
h = input('Variações de humor constantes?')
i = input('Desinteresse por quaisquer tipos de atividades?')
j = input('Medo de ser rejeitado?')
k = input('Paranoias?')
l = input('Sente-se inutil na maioria das vezes?')
m = input('Os sintomas se mantem a mais de 1 ano?')



print 'Será?', net.activate([a,b,c,d])
