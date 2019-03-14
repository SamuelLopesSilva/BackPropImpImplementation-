### Author: SAmuel Lopes

class Neuronio(object):
    def __init__(self, nodes_entrada = []):
        self.nodes_entrada = nodes_entrada
        self.nodes_saida = []
        self.valor = None
        self.gradientes = {}
        for n in nodes_entrada:
            n.nodes_saida.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Neuronio):
    def __init__(self):
        Neuronio.__init__(self)

    def forward(self):
        pass

    def backward(self):
        self.gradientes = {self:0}

        for n in self.nodes_saida:
            grad_cost = n.gradientes[self]
            self.gradientes[self] += grad_cost

class Linear(Neuronio):
    # X = Inputs
    # W = weights
    # b = bias
    def __init__(self, X, W, b):
        Neuronio.__init__(self, [X, W, b])

    def forward(self):
        X = self.nodes_entrada[0].valor
        W = self.nodes_entrada[1].valor
        b = self.nodes_entrada[2].valor
        self.valor = np.dot(X, W) + b

    def backward(self):
        self.gradientes = {n: np.zeros_like(n.valor) for n in self.nodes_entrada}

        for n in self.nodes_saida:
            grad_cost = n.gradientes[self]
            self.gradientes[self.nodes_entrada[0]] += np.dot(grad_cost, self.nodes_entrada[1].valor.T)
            self.gradientes[self.nodes_entrada[1]] += np.dot(self.nodes_entrada[0].valor.T, grad_cost)
            self.gradientes[self.nodes_entrada[2]] += np.sum(grad_cost, axis=0, keepdims = False)

class Sigmoid(Neuronio):
    def __init__(self, node):
        Neuronio.__init__(self,[node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.nodes_entrada[0].valor
        self.valor = self._sigmoid(input_value)

    def backward(self):
        self.gradientes = {n: np.zeros_like(n.valor) for n in self.nodes_entrada}

        for n in self.nodes_saida:
            grad_cost = n.gradientes[self]
            sigmoid = self.valor
            self.gradientes[self.nodes_entrada[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Neuronio):
    def __init__(self, y, a):
        Neuronio.__init__(self, [y,a])

    def forward(self):
        y = self.nodes_entrada[0].valor.reshape(-1, 1)
        a = self.nodes_entrada[1].valor.reshape(-1, 1)

        self.m = self.nodes_entrada[0].valor.shape[0]

        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradientes[self.nodes_entrada[0]] = (2 / self.m) * self.diff
        self.gradientes[self.nodes_entrada[1]] = (-2 / self.m) * self.diff

def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.nodes_saida:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.valor = feed_dict[n]

        L.append(n)
        for m in n.nodes_saida:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_and_backward(graph):
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()

def sgd_update(params, learning_rate = 1e-2):
    for t in params:
        partial = t.gradientes[t]
        t.valor -= learning_rate * partial
