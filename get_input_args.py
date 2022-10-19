# Imports python modules
import argparse
# Get arguments data_dir, learning rate, hidden units, epochs
def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type=str, help = 'data directory(required)')
    parser.add_argument("--save_dir", type=str, default="./", help="saving directory(optional")
    parser.add_argument("--arch", type=str, default="vgg11", help="options of CNN architecture: vgg11 (default), vgg13")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate: .0003 (default)")
    parser.add_argument("--hidden_units", type=int, nargs=3, default=[4096, 1000, 500], help="hidden units in a list: [4096, 1000, 500]   (default)")
    parser.add_argument("--ep", type=int, default=2, help="number of epoch: 2 (default)")
    parser.add_argument("--gpu", type=str, default="y", help="use GPU? (y/n)")
    parser.add_argument('--dropout', type=float, default=0.2, help = 'set dropout rate: 0.2 (defalut)')
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()