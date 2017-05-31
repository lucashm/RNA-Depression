#!/usr/bin/python2.7
#!home/ann/anaconda2/lib/python2.7
from pybrain.datasets import SupervisedDataSet
import time
import sys
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.structure import SoftmaxLayer



ds = SupervisedDataSet(4,1)

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

INTERMEDIATE_LAYERS_QUANTITY = 8
net = buildNetwork(ds.indim, INTERMEDIATE_LAYERS_QUANTITY, ds.outdim, bias=True)
counter = 0;

trainer = BackpropTrainer(net,ds, learningrate=0.001, momentum=0.99, verbose=False, lrdecay=1.0001)
errorList = []

EPOCH_VALUE_MAX = 3000
for epoch in range(0, EPOCH_VALUE_MAX):
    training = trainer.train();
    errorList.insert(epoch, training)
    if training < 0.001:
        break

ds2 = SupervisedDataSet(4,1) # Suporta entradas quadmensionais e alvos dimensionais

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
ds2.addSample((1,1,0,0), (2,)) #
ds2.addSample((1,1,0,1), (2,)) # -> 2 -> Depressao situacional
ds2.addSample((1,1,1,0), (2,)) #
ds2.addSample((1,1,1,1), (2,)) #

INTERMEDIATE_LAYERS_QUANTITY = 8
net2 = buildNetwork(ds2.indim, INTERMEDIATE_LAYERS_QUANTITY, ds2.outdim, bias=True)

trainer2 = BackpropTrainer(net2,ds2, learningrate=0.001, momentum=0.99, verbose=False, lrdecay=1.0001)
errorList2 = []

for epoch2 in range(0, EPOCH_VALUE_MAX):
    training2 = trainer2.train();
    errorList2.insert(epoch2, training2)
    if training2 < 0.001:
        break


ds3 = SupervisedDataSet(6,1) # Suporta entradas quadmensionais e alvos dimensionais

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
ds3.addSample((0,0,1,1,0,0), (5,))
ds3.addSample((0,0,1,1,0,1), (3,))
ds3.addSample((0,0,1,1,1,0), (5,))
ds3.addSample((0,0,1,1,1,1), (4,))
ds3.addSample((0,1,0,0,0,0), (6,))
ds3.addSample((0,1,0,0,0,1), (6,))
ds3.addSample((0,1,0,0,1,0), (6,))
ds3.addSample((0,1,0,0,1,1), (6,))
ds3.addSample((0,1,0,1,0,0), (6,))
ds3.addSample((0,1,0,1,0,1), (6,))
ds3.addSample((0,1,0,1,1,0), (6,))
ds3.addSample((0,1,0,1,1,1), (6,))
ds3.addSample((0,1,1,0,0,0), (6,))
ds3.addSample((0,1,1,0,0,1), (6,))
ds3.addSample((0,1,1,0,1,0), (6,))
ds3.addSample((0,1,1,0,1,1), (6,))
ds3.addSample((0,1,1,1,0,0), (5,))
ds3.addSample((0,1,1,1,0,1), (3,))
ds3.addSample((0,1,1,1,1,0), (4,))
ds3.addSample((0,1,1,1,1,1), (4,))
ds3.addSample((1,0,0,0,0,0), (0,))
ds3.addSample((1,0,0,0,0,1), (0,))
ds3.addSample((1,0,0,0,1,0), (4,))
ds3.addSample((1,0,0,0,1,1), (4,))
ds3.addSample((1,0,0,1,0,0), (0,))
ds3.addSample((1,0,0,1,0,1), (0,))
ds3.addSample((1,0,0,1,1,0), (4,))
ds3.addSample((1,0,0,1,1,1), (4,))
ds3.addSample((1,0,1,0,0,0), (0,))
ds3.addSample((1,0,1,0,0,1), (0,))
ds3.addSample((1,0,1,0,1,0), (4,))
ds3.addSample((1,0,1,0,1,1), (4,))
ds3.addSample((1,0,1,1,0,0), (7,))
ds3.addSample((1,0,1,1,0,1), (3,))
ds3.addSample((1,0,1,1,1,0), (4,))
ds3.addSample((1,0,1,1,1,1), (4,))
ds3.addSample((1,1,0,0,0,0), (6,))
ds3.addSample((1,1,0,0,0,1), (6,))
ds3.addSample((1,1,0,0,1,0), (4,))
ds3.addSample((1,1,0,0,1,1), (4,))
ds3.addSample((1,1,0,1,0,0), (6,))
ds3.addSample((1,1,0,1,0,1), (6,))
ds3.addSample((1,1,0,1,1,0), (4,))
ds3.addSample((1,1,0,1,1,1), (4,))
ds3.addSample((1,1,1,0,0,0), (6,))
ds3.addSample((1,1,1,0,0,1), (6,))
ds3.addSample((1,1,1,0,1,0), (4,))
ds3.addSample((1,1,1,0,1,1), (4,))
ds3.addSample((1,1,1,1,0,0), (6,))
ds3.addSample((1,1,1,1,0,1), (3,))
ds3.addSample((1,1,1,1,1,0), (4,))
ds3.addSample((1,1,1,1,1,1), (4,))


INTERMEDIATE_LAYERS_QUANTITY3 = 48
net3 = buildNetwork(ds3.indim, INTERMEDIATE_LAYERS_QUANTITY3, ds3.outdim, bias=True)

trainer3 = BackpropTrainer(net3,ds3, learningrate=0.001, momentum=0.99, verbose=False, lrdecay=1.00001)
errorList3 = []

for epoch3 in range(0, EPOCH_VALUE_MAX):
    training3 = trainer3.train();
    errorList3.insert(epoch3, training3)
    if training3 < 0.001:
        break


#######################################################################################
import cgi
import cgitb
cgitb.enable()

a = {0,1,2}
form = cgi.FieldStorage()

questions = []

questions.append(form.getvalue("q1"))
questions.append(form.getvalue("q2"))
questions.append(form.getvalue("q3"))
questions.append(form.getvalue("q4"))
questions.append(form.getvalue("q5"))
questions.append(form.getvalue("q6"))
questions.append(form.getvalue("q7"))
questions.append(form.getvalue("q8"))
questions.append(form.getvalue("q9"))
questions.append(form.getvalue("q10"))
questions.append(form.getvalue("q11"))
questions.append(form.getvalue("q12"))

a = questions[0]
b = questions[1]
c = questions[2]
d = questions[3]
e = questions[4]
f = questions[5]
g = questions[6]
h = questions[7]
i = questions[8]
j = questions[9]
k = questions[10]
l = questions[11]

ativa = net.activate([a,b,c,d])

ativa = net.activate([a,b,c,d])

if ativa <= 1.2 and ativa >= 0.8:
    entrada_um = 1
else:
    entrada_um = 0

ativa2 = net2.activate([entrada_um,e,f,g])

resultado_final = net3.activate([ativa2, h, i, j, k, l])
print_final = 'teste'

if ativa2 >= 1.8 and ativa2 <= 2.2:
    print_final = 'Depressao situacional'
elif resultado_final >= 2.8 and resultado_final <= 3.2:
    print_final = 'Depressao distimica'
elif resultado_final >= 3.8 and resultado_final <= 4.2:
    print_final = 'Depressao psicotica'
elif resultado_final >= 4.8 and resultado_final <= 5.2:
    print_final = 'Depressao atipica'
elif resultado_final >= 5.8 and resultado_final <= 6.2:
    print_final = 'Depressao bipolar'
elif resultado_final >= 6.8 and resultado_final <= 7.2:
    print_final = 'Depressao maior'
else:
    print_final = "Nenhuma depressao encontrada"


print 'Content-type: text/html\r\n\r'
print '<meta charset="utf-8"/>'
print '<html>'
print '<form action="database.py" method="post">' ################################
print '<h1>Respostas: </h1>'
print '<h1>{0}</h1>'.format(questions)
print '<input type="hidden" name="questions" value="{0}">'.format(questions)
print '<h1>Resultado final:</h1>'
print '<h1>{0}</h1>'.format(print_final)
print '<input type="hidden" name="print_final" value="{0}">'.format(print_final)
print '<label>Seu nome: </label>'
print '<input type="text" name="nome">'
print '<input type="submit" value="Inserir no banco de dados">'
print '</form>' ##################################################################
print '</html>'
