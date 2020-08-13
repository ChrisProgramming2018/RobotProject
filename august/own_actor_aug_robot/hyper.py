""" search for the optimal hyperparameter values  to run several in parallel use
    different """
import os
import argparse

def main(args):
    list_of_update_freq = [ 2, 3, 4]
    for freq in list_of_update_freq:
        freq = int(freq)
        lr = freq * 2
        os.system(f'python3 ./main.py\
                --locexp {25.07}\
                --policy_freq {lr}\
                --repeat_opt {freq}')

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Average_TD3')
    main(parser.parse_args())
