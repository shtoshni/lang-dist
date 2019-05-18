import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging

from os import path

from utils.data_utils import languages
from utils.utils import get_non_diagonal_entries

logging.basicConfig(format='%(message)s', level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '--sim_mat_file', help='Matrix as input for heatmap', type=str)
    parser.add_argument(
        "--out_dir", help="Output directory",
        default='/home/shtoshni/Research/MUSE/results/plots', type=str)
    parser.add_argument(
        "--cbar", help="Use color bar or not.", default=True,
        action="store_false")

    args = parser.parse_args()
    return args


def generate_heatmap(data_mat, output_file, cbar=True):
    if cbar:
        fig, ax = plt.subplots(figsize=(11,8))
    else:
        fig, ax = plt.subplots(figsize=(9,8))
    epsilon = 2e-2
    vmin = np.amin(data_mat)
    # Calculate the maximum over non-diagonal entries since the diagonals are
    # all 1
    vmax = max(get_non_diagonal_entries(data_mat)) + epsilon

    sns_plot = sns.heatmap(
        data_mat, xticklabels=languages, yticklabels=languages,
        vmin=vmin, vmax=vmax, linecolor='white', linewidths=0.1,
        cbar=cbar, ax=ax)
    sns_plot.figure.savefig(output_file, dpi=300, format='pdf',
                            bbox_inches='tight')
    logging.info("Output written at: %s" %output_file)
    plt.show()


def main():
    args = parse_args()

    if path.isfile(args.sim_mat_file):
        data_mat = np.load(args.sim_mat_file)
        base_file = path.basename(args.sim_mat_file)
        output_file = path.join(args.out_dir,
                                path.splitext(base_file)[0] + ".pdf")
        generate_heatmap(data_mat, output_file, cbar=args.cbar)

if __name__=='__main__':
    main()
