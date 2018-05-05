import random
import numpy as np
import math



# initialize the gene pool
def init(filePath=None): 
    pass

prevGen = None
generation = 0

def getNewBatch(batch_size):
    global prevGen, generation
    generation += 1
    currGen = []
    print("Generation", generation)
    if prevGen:
        ss = sorted(prevGen, key=lambda p: -1*p.fitness)
        currGen = cross_breath(ss[0:int(len(ss)/5)], batch_size)
    else:
        currGen = [Player(DANN([Layer((4, 10)), Layer((10, 1))])) for _ in range(batch_size)]

    prevGen = currGen
    return currGen


class Player(object):
    def __init__(self, dann):
        self.fitness = 0
        self.dann = dann

    def set_fitness(self, fitness):
        self.fitness = fitness

    def play(self, playerInfo):
            # playerInfo = {
            #     'playerY': playery,
            #     'playerVelY': playerVelY,
            #     'playerToPipeX': lowerPipes['x'] - playerx,
            #     'playerToPipeY': lowerPipes['y'] - playery
            # }
        return self.dann.predict(np.array(list(playerInfo.values()))) > 0

def relu(x):
    return np.maximum(x, 0)

### Create a simple ANN
class Layer(object):
    def __init__(self, shape, activ=relu, mat=None, bias=None):
        self.shape = shape
        self.activ = activ

        self.mat = mat if mat is not None else np.random.rand(shape[1], shape[0])*2-1
        self.bias = bias if bias is not None else np.random.rand(shape[1])*2-1

    def __call__(self, x):
        return self.activ(self.mat @ x + self.bias)


class DANN(object):
    """Gogagago"""
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layer(self, i):
        return self.layers[i]
        

def cross_breath(danns, limit):
    newDanns = []
    while len(newDanns) <= limit:
        da = random.choice(danns).dann
        db = random.choice(danns).dann
        newDanns.append(Player(breath_DANN(da,db)))
    return newDanns


def breath_DANN(dann, dbnn):
    newLayers = []
    for la, lb in zip(dann.layers, dbnn.layers):
        newLayers.append(breath_layer(la, lb))
    return DANN(newLayers)

def breath_layer(la, lb, bias=0, mut=0.4):
    mm = (np.random.rand(la.shape[1], la.shape[0])*2-1) > bias
    am = np.copy(la.mat)
    am[mm] = 0
    bm = np.copy(lb.mat)
    bm[mm==False] = 0
    nm = (am + bm) + (np.random.rand(la.shape[1], la.shape[0])*2-1) * mut

    mb = (np.random.rand(la.shape[1])*2-1) > bias
    ab = np.copy(la.bias)
    ab[mb] = 0
    bb = np.copy(lb.bias)
    bb[mb==False] = 0
    nb = (ab + bb) + (np.random.rand(la.shape[1])*2-1) * mut
    return Layer(la.shape, mat=nm, bias=nb)
    
