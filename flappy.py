from itertools import cycle
import random
import sys

import nnplayer as nnp

import pygame
from pygame.locals import *


FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

PLAYERX = int(SCREENWIDTH * 0.2)
DRAW_SPEED = 1

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK, PLAYERX, DRAW_SPEED
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )


        EPOCS = 100000000000
        BATCH_SIZE = 20

        nnp.init() # Optional load file here
        result = None
        ## Train a batch
        # Get batch of networks
        for i in range(EPOCS):
            print("Epoc: #", i)
            players = nnp.getNewBatch(BATCH_SIZE)
            mainGame(players)


        # Get the score and return it to some authority on how to update
        # showGameOverScreen(result)


class Bird(object):
    def __init__(self, player):
        self.alive    =  True
        self.p        =  player
        self.x        =  PLAYERX
        self.y        =  int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

        self.index    =  0 
        self.loopIter =  0
        self.velY     = -9   # player's velocity along Y, default same as playerFlapped
        self.rot      = 45   # player's rotation
        self.velRot   =  3   # angular speed
        self.flapped  =  False # True when player flaps

        # Const isch
        self.w        = IMAGES['player'][0].get_width()
        self.h        = IMAGES['player'][0].get_height()
        self.flapAcc  =  -9   # players speed on flapping
        self.accY     =   1   # players downward accleration
        self.rotThr   =  20   # rotation threshold
        self.maxVelY  =  10   # max vel along Y, max descend speed
        self.minVelY  =  -8   # min vel along Y, max ascend speed

        self.fitness  =   0
        self.score    =   0
    
    # TODO: Fix me
    def update(self, pipes):
        if not self.alive:
            return
        pipesInfront = [pipe for pipe in pipes if not pipe.behind(self.x)]
        nPipe = pipesInfront[0]
        nnPipe = pipesInfront[1]

        playerInfo = [
            self.y, self.velY, 
            nPipe.x - PLAYERX, nPipe.y - self.y,
            nnPipe.x - PLAYERX, nnPipe.y - self.y
        ]
        if self.p.play(playerInfo): # Should jump
            if self.y > -2 * self.h:
                self.velY = self.flapAcc
                self.flapped = True
        
    def update_after(self, pipes):
        # Update player fitness
        if not self.alive:
            return

        self.fitness += 1

        # check for score
        playerMidPos = PLAYERX + self.w/2
        for pipe in pipes:
            if not pipe.behind(playerMidPos-4) and pipe.behind(playerMidPos):
                pipe.scored = True
                self.score += 1
        
        # rotate the player
        if self.rot > -90:
            self.rot -= self.velRot

        # player's movement
        if self.velY < self.maxVelY and not self.flapped:
            self.velY += self.accY

        if self.flapped:
            self.flapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.rot = 45

        self.y += min(self.velY, BASEY - self.y - self.h)

    def draw(self):
        # Player rotation has a threshold
        rot = min(self.rot, self.rotThr)
        playerSurface = pygame.transform.rotate(IMAGES['player'][self.index], rot)
        SCREEN.blit(playerSurface, (PLAYERX, self.y))

    # TODO: Fix me
    def crashed(self, pipes):
        # check for crash here
        crash = self._crashed(pipes)
        if crash[0]:
            self.alive = False
            self.p.set_fitness(self.fitness + 1000*self.score)

    def _crashed(self, pipes):
        ## Did we hit the ground
        if self.y + self.h >= BASEY - 1:
            return [True, True]
        
        for pipe in pipes:
            if pipe.colide(self):
                return [True, False]

        return [False, False]


class Pipe(object):
    def __init__(self, x=SCREENWIDTH+10):
        self.x = x
        #TODO: Fix, cant be used with many birds
        self.scored = False
        self.y = int(BASEY * 0.8) - random.randrange(0, int(BASEY * 0.6))

        self.w = IMAGES['pipe'][0].get_width()
        self.h = IMAGES['pipe'][0].get_height()

    def update(self):
        # TODO: Change to game constant
        pipeVelX = -4
        self.x += pipeVelX

    def is_out(self):
        return self.x < -IMAGES['pipe'][0].get_width()
    
    def behind(self, ox):
        return self.x + IMAGES['pipe'][0].get_width() < ox

    def colide(self, bird):
        birdRect = pygame.Rect(bird.x, bird.y, bird.w, bird.h)

        hiPipeRect = pygame.Rect(self.x, self.y - PIPEGAPSIZE - self.h, self.w, self.h)
        loPipeRect = pygame.Rect(self.x, self.y, self.w, self.h)

        pHitMask = HITMASKS['player'][bird.index]
        hiHitmask = HITMASKS['pipe'][0]
        loHitmask = HITMASKS['pipe'][1]

        uCollide = pixelCollision(birdRect, hiPipeRect, pHitMask, hiHitmask)
        lCollide = pixelCollision(birdRect, loPipeRect, pHitMask, loHitmask)

        if uCollide or lCollide:
            return True

        return False


    def draw(self):
        # Draw pipes
        SCREEN.blit(IMAGES['pipe'][0], (self.x, self.y - PIPEGAPSIZE - self.h))
        SCREEN.blit(IMAGES['pipe'][1], (self.x, self.y))


def mainGame(players):
    global DRAW_SPEED
    basex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    loopIter = 0

    birds = [Bird(p) for p in players]

    # TODO: Change to game constant
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # TODO: Change to game constant
    pipeVelX = -4

    pipes = [
        Pipe(SCREENWIDTH + 200),
        Pipe(int(SCREENWIDTH * 3/2) + 200)
    ]

    drawCount = 0

    while True:
        endRound = True
        for bird in birds:
            # If all birds are dead, end game
            if bird.alive:
                endRound = False
                bird.update(pipes)
                bird.crashed(pipes)
                bird.update_after(pipes)

        if endRound:
            return

        for event in pygame.event.get():
            #TODO: Set useful commands here
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                if event.key == K_1:
                    DRAW_SPEED = 1
                if event.key == K_2:
                    DRAW_SPEED = 2
                if event.key == K_3:
                    DRAW_SPEED = 4
                if event.key == K_4:
                    DRAW_SPEED = 8
                if event.key == K_5:
                    DRAW_SPEED = 12

                if event.key == K_0:
                    DRAW_SPEED = 10000000


        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        for pipe in pipes:
            pipe.update()

        if 0 < pipes[0].x < 5:
            pipes.append(Pipe(SCREENWIDTH + 10))
        
        if pipes[0].is_out():
            pipes.pop(0)

        ### Draw section
        drawCount += 1
        if drawCount % DRAW_SPEED == 0:
            drawCount = 0
            # draw backgroud sprite
            SCREEN.blit(IMAGES['background'], (0,0))

            # Tick pipes and then draw here
            # add new pipe when first pipe is about to touch left of SCREEN
            for pipe in pipes:
                pipe.draw()


            SCREEN.blit(IMAGES['base'], (basex, BASEY))
            # print score so player overlaps the score
            score = max([bird.score for bird in birds])
            showScore(score)

            # Draw birds 
            for bird in birds:
                if bird.alive:
                    bird.draw()
            
            # probably some screen caping
            pygame.display.update()
            FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    PLAYERX = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (PLAYERX,playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask


if __name__ == '__main__':
    main()
