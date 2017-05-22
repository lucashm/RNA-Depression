from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

# Coisas a serem feitas:
# Descobrir quais sao as variaveis importantes
# Ligar elas ao conteudo da professora, pra quando ela perguntar e querer que algo seja mudado
# docs do pyBrain: http://pybrain.org/docs/
# deem uma pesquisada, se eu tiver feito algo errado, ou se tiver um jeito melhor, a vontade

n = FeedForwardNetwork()

ds = SupervisedDataSet(5,1)

ds.addSample((0,0,0,0,0), (0,))
ds.addSample((0,0,0,0,1), (1,))
ds.addSample((0,0,0,1,0), (1,))
ds.addSample((0,0,0,1,1), (0,))
ds.addSample((0,0,1,0,0), (0,))
ds.addSample((0,0,1,0,1), (0,))
ds.addSample((0,0,1,1,0), (1,))
ds.addSample((0,0,1,1,1), (0,))
ds.addSample((0,1,0,0,0), (0,))
ds.addSample((0,1,0,0,1), (0,))
ds.addSample((0,1,0,1,0), (1,))
ds.addSample((0,1,0,1,1), (0,))
ds.addSample((0,1,1,0,0), (0,))
ds.addSample((0,1,1,0,1), (0,))
ds.addSample((0,1,1,1,0), (1,))
ds.addSample((0,1,1,1,1), (0,))
ds.addSample((1,0,0,0,0), (0,))
ds.addSample((1,0,0,0,1), (1,))
ds.addSample((1,0,0,1,0), (0,))
ds.addSample((1,0,0,1,1), (0,))
ds.addSample((1,0,1,0,0), (1,))
ds.addSample((1,0,1,0,1), (0,))
ds.addSample((1,0,1,1,0), (0,))
ds.addSample((1,0,1,1,1), (1,))
ds.addSample((1,1,0,0,0), (0,))
ds.addSample((1,1,0,0,1), (0,))
ds.addSample((1,1,0,1,0), (0,))
ds.addSample((1,1,0,1,1), (1,))
ds.addSample((1,1,1,0,0), (0,))
ds.addSample((1,1,1,0,1), (1,))
ds.addSample((1,1,1,1,0), (1,))
ds.addSample((1,1,1,1,1), (1,))

hiddenLayer = SigmoidLayer(3)

n.addInputModule(ds.indim)
n.addModule(hiddenLayer)
n.addOutputModule(ds.outdim)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()

print "in to hidden params: ", in_to_hidden.params

print n.activate([1,1,1,1,1])
