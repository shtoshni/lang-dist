import io
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import data_utils
import logging

from os import path
from logging import getLogger
from collections import OrderedDict
from skbio.tree import TreeNode
from six import StringIO
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

logging.basicConfig(format='%(message)s', level=logging.INFO)


GT_INDO_EUROPEAN_STRING = (
    "(Greek:1, (Romanian:1, (Italian:1, ((Catalan:1, French:1):1,"
    "(Spanish:1, Portuguese:1):1):1):1):1,"
    "((Norwegian:1, (Danish:1, Swedish:1):1):1, (English:1, Dutch:1, German:1):1):1,"
    "((Russian:1, Ukrainian:1):1, ((Czech:1, Slovak:1):1,Polish:1):1,"
    "((Bulgarian:1, Macedonian:1):1, (Slovenian:1, Croatian:1):1):1):1):1;")

GT_INDO_EUROPEAN_TREE = TreeNode.read(StringIO(GT_INDO_EUROPEAN_STRING))

LANG_CODES = data_utils.lang_codes
INDO_EURO_LANG_CODES = data_utils.indo_european
INDO_EURO_LANG_NAMES = [
    data_utils.lang_code_to_language[lang_code] for lang_code in INDO_EURO_LANG_CODES]

# Indo European distance normalizer
RFD_NORMALIZER = 36
LEAF_NORMALIZER = 6832


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '--sim_mat_file', help='Similarity matrix as input', type=str)
    parser.add_argument(
        "--out_dir", help="Output directory",
        default='/home/shtoshni/Research/MUSE/results/plots/muse', type=str)

    args = parser.parse_args()
    return args


def get_linkage_matrix(sim_mat):
    np.fill_diagonal(sim_mat, 1.0)
    dists = squareform(1.0 - sim_mat)
    return linkage(dists, "ward")


def get_sub_mat(full_sim_mat):
    """Get sub-matrix corresponding to a subset of language codes."""
    subset_indices = []
    for lang_code in INDO_EURO_LANG_CODES:
        subset_indices.append(LANG_CODES.index(lang_code))

    sub_mat = np.zeros((len(subset_indices), len(subset_indices)))
    for i, idx1 in enumerate(subset_indices):
        for j, idx2 in enumerate(subset_indices):
            sub_mat[i, j] = full_sim_mat[idx1, idx2]
    return sub_mat


def main_calc_tree_distance(lang_set_mat, dist_metric="rfd"):
    """Calculate Tree Distance."""
    pred_linkage = get_linkage_matrix(lang_set_mat)
    pred_tree = TreeNode.from_linkage_matrix(pred_linkage, INDO_EURO_LANG_NAMES)

    if dist_metric == "rfd":
        tree_dist = pred_tree.compare_rfd(GT_INDO_EUROPEAN_TREE)
    else:
        pred_tree_string_io = StringIO()
        pred_tree.write(pred_tree_string_io)
        pred_tree_string  = pred_tree_string_io.getvalue()

        # Replace distances with 1
        unweighted_tree_string = re.sub(r"\d+\.\d+", "1", pred_tree_string)
        pred_tree = TreeNode.read(StringIO(unweighted_tree_string))

        gt_distances_struct = GT_INDO_EUROPEAN_TREE.tip_tip_distances()
        gt_distances = gt_distances_struct.data
        gt_ids = gt_distances_struct.ids

        pred_distances = pred_tree.tip_tip_distances(endpoints=list(gt_ids)).data
        tree_dist = np.sum((gt_distances - pred_distances)**2)

    return tree_dist, pred_tree


def get_results_from_rand(dist_metric="rfd", N=25000):
    """Normalizers obtained using the max distance from 25K random runs."""
    np.random.seed(10)
    dist_list = []
    for i in range(N):
        rand_mat = np.random.random((len(lang_codes), len(lang_codes)))
        rand_sim_mat = (rand_mat + rand_mat.T) / 2.0
        dist = main_calc_tree_distance(rand_sim_mat, dist_metric=dist_metric)
        dist_list.append(dist)

    return dist_list


def dendrogram_plot(lang_set_mat, fig_file):
    """Make dendrogram plots."""
    linkage_matrix = get_linkage_matrix(lang_set_mat)
    dendrogram(linkage_matrix, labels=INDO_EURO_LANG_NAMES,
               leaf_rotation=30, leaf_font_size=6)
    plt.title("Indo European")
    plt.savefig(fig_file, dpi=300, format='pdf', bbox_inches='tight')
    plt.show()


def main():
    args = parse_args()

    # Check sim_mat_file
    sim_mat_file = args.sim_mat_file
    if not path.isfile(sim_mat_file):
        logging.warning("Not a valid file: %s" %args.sim_mat_file)
        return

    # Get sim_mat
    sim_mat = np.load(sim_mat_file)
    # Make the similarity matrix symmetric
    sim_mat = (sim_mat + sim_mat.T)/2
    # Get indo european subset of similarity matrix
    lang_set_mat = get_sub_mat(sim_mat)

    # Calculate predicted tree and distance
    rfd_dist, pred_tree = main_calc_tree_distance(
        lang_set_mat, dist_metric="rfd")
    leaf_dist, _ = main_calc_tree_distance(lang_set_mat, dist_metric="leaf")

    print ("RFD dist: %d, Normalized-RFD: %.3f"
           %(rfd_dist, rfd_dist/RFD_NORMALIZER))
    print ("Leaf dist: %d, Normalized-leaf distance: %.3f"
           %(leaf_dist, leaf_dist/LEAF_NORMALIZER))

    # Output file prefix
    base_file = path.basename(sim_mat_file)
    file_prefix = path.splitext(base_file)[0]

    # Save predicted tree
    string_io = StringIO()
    pred_tree.write(string_io)
    pred_tree_string = string_io.getvalue().strip()
    output_file = path.join(args.out_dir, file_prefix + "_pred_tree.txt")
    with open(output_file, 'w') as f:
        f.write(pred_tree_string)
    logging.info("Pred tree saved at: %s" %output_file)

    # Save a dendrogram
    fig_file = path.join(args.out_dir, file_prefix + "_dendrogram.pdf")
    dendrogram_plot(lang_set_mat, fig_file)
    logging.info("Dendrogram plot saved at: %s" %fig_file)


if __name__=='__main__':
    main()
