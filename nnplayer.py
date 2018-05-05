import random
import numpy as np
import math
import ANNGenetic.ann


network = ann.ANN(4)
network.add_layer(ann.Layer(8, activation=np.sigmoid))
network.add_layer(ann.Layer(8, activation=np.sigmoid))
network.add_layer(ann.Layer(1, activation=np.tanh))

FAMILY = None
FIRST_GEN = True
GENERATION = 0

PLAYERS = None


# initialize the gene pool
def init(filePath=None): 
	FAMILY = ann.Genetic(50, verbose=False)
	FAMILY.create_family(network)

def getNewBatch(batch_size):
    global FIRST_GEN, FAMILY, GENERATION, PLAYERS
    GENERATION += 1
    print("Generation", GENERATION)

    if not FIRST_GEN:
	fitnesses = [x.fitness for x in PLAYERS]
	FAMILY.evolve(fitnesses)

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
        return self.dann.prop(np.array(playerInfo)) > 0

