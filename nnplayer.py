import random
import numpy as np
import math
import ANNGenetic.ann as ann


network = ann.ANN(6)
network.add_layer(ann.Layer(16, activation=np.tanh))
network.add_layer(ann.Layer(1, activation=np.tanh))

FAMILY = None
FIRST_GEN = True
GENERATION = 0

PLAYERS = None


# initialize the gene pool
def init(filePath=None): 
        global FAMILY
        FAMILY = ann.Genetic(200, verbose=False)
        FAMILY.create_family(network)

def getNewBatch(batch_size):
    global FIRST_GEN, FAMILY, GENERATION, PLAYERS
    GENERATION += 1
    print("Generation", GENERATION)

    if not FIRST_GEN:
        fitnesses = [x.fitness for x in PLAYERS]
        print(sum(fitnesses) / len(fitnesses))
        FAMILY.evolve(fitnesses)

    FIRST_GEN = False
    PLAYERS = [Player(x) for x in FAMILY.family]

    return PLAYERS


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
        
        pred = self.dann.prop(np.array(playerInfo))
        return pred > 0

