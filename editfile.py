import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--verbosity", help="increase output verbosity")
parser.add_argument('--dataset', type=str, default="pokec2", help='One dataset from income, bail, pokec1, pokec2, fair_pokec1, fair_pokec2')
args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")
if args.dataset:
    print("using dataset: ", args.dataset)


