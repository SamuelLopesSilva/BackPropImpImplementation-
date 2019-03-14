import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

boston = load_boston()

Xdata = boston['data']
Ydata = boston['target']

Xdata = (Xdata - np.mean(Xdata, axis=0)) / np.std(Xdata, axis=0)

number_features = Xdata.shape[1]
number_hidden_layer = 10

initial_weights = np.random.randn(number_features, number_hidden_layer)
initial_bias = np.zeros(number_hidden_layer)
initial_weights2 = np.random.randn(number_hidden_layer,1)
initial_bias2 = np.zeros(1)

X, y = Input(), Input()
w1, b1 = Input(), Input()
w2, b2 = Input(), Input()

l1 = Linear(X, w1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, w2, b2)
cost = MSE(y,l2)


node_dict = {X: Xdata,
             y: Ydata,
             w1:initial_weights,
             b1: initial_bias,
             w2: initial_weights2,
             b2: initial_bias2}

epochs = 1000

examples = Xdata.shape[0]

batch_size = 11
steps_per_epoch = examples // batch_size

graph = topological_sort(node_dict)

params = [w1, b1, w2, b2]

print("Número total de exemplos = {}".format(examples))

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, y_batch = resample(Xdata, Ydata, n_samples = batch_size)

        X.valor = X_batch
        y.valor = y_batch

        forward_and_backward(graph)

        sgd_update(params)

        loss += graph[-1].valor
    print("Epoch: {}, Custo: {:.3f}".format(i+1, loss/steps_per_epoch))
