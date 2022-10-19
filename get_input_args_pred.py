# Imports python modules
import argparse
# Get arguments data_dir, learning rate, hidden units, epochs
def get_input_args_pred():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above and additional arguments
    # using add_argument() from ArguementParser method
    parser.add_argument('image_path', type=str, help = 'image path(required):')
    parser.add_argument('model_path', type=str, help = 'model checkpoint path(required):')
    parser.add_argument('--top_k', type=int, default=5, help = 'top K most likely classes (default5):')
    parser.add_argument('--cat_names', type=str, default = './cat_to_name.json', help ='JSON file name for mapping categories to names')
    parser.add_argument("--gpu", type=str, default="y", help="use GPU? (y/n)")
    # you created with this function 
    return parser.parse_args()