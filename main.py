import logging
import os

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'startIter': 1,
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','checkpoint_2.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def extractIndex(checkpointFile):
    return int(checkpointFile[len('checkpoint_'):-len('.pth.tar')])

def loadCheckpoint():
    checkpointFiles = [file for file in os.listdir(args.checkpoint) if file.startswith('checkpoint') and file.endswith('tar')]
    checkpointFiles.sort(key=checkpointFile, reverse=True)
    if checkpointFiles:
        checkpointFile = checkpointFiles[0]
        args.load_model = True
        args.load_folder_file = (args.checkpoint, checkpointFile)
        args.startIter =  extractIndex(checkpointFile) + 1

def main():
    if os.path.exists(args.checkpoint):
        loadCheckpoint()

    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
