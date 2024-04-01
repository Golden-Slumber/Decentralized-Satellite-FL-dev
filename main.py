import sys
import numpy
from satellite_system import *
from learning_task import *

home_dir = './'
sys.path.append(home_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuroSat')
    parser.add_argument('--intra-plane-iters', type=int, default=1, metavar='N')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--local-iters', type=int, default=2, metavar='N')
    parser.add_argument('--verbose', action='store_true', default=True)

    with open('./Resources/EuroSAT_train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open('./Resources/EuroSAT_test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    print(len(train_set))
    num_planes = 5
    num_satellites = 20
