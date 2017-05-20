from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

# Coisas a serem feitas:
# Descobrir quais sao as variaveis importantes
# Ligar elas ao conteudo da professora, pra quando ela perguntar e querer que algo seja mudado
# docs do pyBrain: http://pybrain.org/docs/
# deem uma pesquisada, se eu tiver feito algo errado, ou se tiver um jeito melhor, a vontade


ds = SupervisedDataSet(2,1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

for inpt, target in ds:
    print inpt, target


net = buildNetwork(ds.indim, 4, ds.outdim, bias=True)
                                                                    #Trocar essas linhas para:
trainer = BackpropTrainer(net,ds, learningrate=0.01, momentum=0.99) # trainer = BackpropTrainer(net,ds)
for epoch in range(0, 3000):                                        # for epoch in range(0, 10000):
    training = trainer.train();
    print(training);
    if training < 0.001:
        break




#print "\nWeights: ", net.params
print "\n\n"

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([2,1])
