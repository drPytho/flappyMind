import random
import numpy as np
import math
import ANNGenetic.ann as ann
import matplotlib.pyplot as plt


network = ann.ANN(6)
network.add_layer(ann.Layer(16, activation=np.tanh))
network.add_layer(ann.Layer(1, activation=np.tanh))

FAMILY = None
FIRST_GEN = True
GENERATION = 0

PLAYERS = None

MEAN_SCORE = [0]
GEN = [0]



# initialize the gene pool
def init(filePath=None): 
        global FAMILY
        FAMILY = ann.Genetic(1000, verbose=False)
        FAMILY.create_family(network)

def getNewBatch(batch_size):
    global FIRST_GEN, FAMILY, GENERATION, PLAYERS, MEAN_SCORE, GEN

    if not FIRST_GEN:
        fitnesses = [x.fitness for x in PLAYERS]

        GENERATION += 1
        mean = (sum(fitnesses) / len(fitnesses))

        print("Generation", GENERATION)
        
        MEAN_SCORE.append(mean)
        GEN.append(GENERATION)

        plt.plot(GEN, MEAN_SCORE)
        #plt.show()
        plt.pause(0.5)


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

