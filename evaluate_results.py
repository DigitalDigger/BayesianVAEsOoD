import random
import os
import argparse
from utils.eval import evalIDvsOOD

if __name__ == '__main__':
    # due to several stages of randomness we set the seed for the exact reproducibility of the results
    parser = argparse.ArgumentParser(description="VariationPersistence")
    parser.add_argument("--result_folder", type=str, default=None, required=True)

    args = parser.parse_args()

    if not (os.path.isdir(args.result_folder)):
       print("Result folder %s is not found" % args.result_folder)
    else:
        evalIDvsOOD(args.result_folder)

