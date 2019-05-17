import numpy as np
import argparse
import sys
sys.path.append("..")

from os import path
from scipy.stats import pearsonr
from skbio.stats.distance import mantel

from utils import get_non_diagonal_entries, create_symm_dist_mat


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sim_mat1_file', help='First similarity matrix file', type=str)
    parser.add_argument(
        '--sim_mat2_file', help='Second similarity matrix file', type=str)

    args = parser.parse_args()
    return args


def calc_corr(sim_mat1, sim_mat2):
    """Calculate correlation between symmetric and non-symmetric matrices."""
    non_symm_corr = pearsonr(get_non_diagonal_entries(sim_mat1),
                             get_non_diagonal_entries(sim_mat2))[0]
    symm_corr = mantel(create_symm_dist_mat(sim_mat1),
                       create_symm_dist_mat(sim_mat2))[0]
    print ("Correlation between non-diagonal entries: %.3f" %non_symm_corr)
    print ("Mantel correlation: %.3f" %symm_corr)


def main():
    args = parse_args()

    # Load similarity matrices
    sim_mat1 = np.load(args.sim_mat1_file)
    sim_mat2 = np.load(args.sim_mat2_file)

    calc_corr(sim_mat1, sim_mat2)

if __name__=='__main__':
    main()
