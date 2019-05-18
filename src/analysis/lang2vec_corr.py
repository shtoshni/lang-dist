import argparse
import lang2vec.lang2vec as l2v
import numpy as np

from utils.data_utils import lang_codes_3
from utils.utils import create_symm_dist_mat
from os import path
from sklearn.metrics.pairwise import cosine_similarity
from skbio.stats.distance import mantel

FEAT_LIST = ["geo", "fam", "syntax_knn", "phonology_knn", "inventory_knn",
             "fam+syntax_knn"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '--sim_mat_file', help='Similarity matrix as input', type=str)

    args = parser.parse_args()
    return args


def stack_features(features):
    # Transform features from lan2vec to a matrix
    all_lang_features = [features[lang_code] for lang_code in lang_codes_3]
    return np.vstack(all_lang_features)


def get_feature_dist_dicts():
    distance_dict = {}
    for feature_type in FEAT_LIST:
        distance_dict[feature_type] = l2v.get_features(
            lang_codes_3, feature_type, minimal=False)
        stacked_features = stack_features(distance_dict[feature_type])
        distance_dict[feature_type] = 1 - cosine_similarity(
            stacked_features, stacked_features)

        np.fill_diagonal(distance_dict[feature_type], 0)
    return distance_dict


def calc_dist_mat_corr(sim_mat, distance_dict):
    """Calculate correlation of different distance matrices with similarity matrix."""
    dist_mat = create_symm_dist_mat(sim_mat)

    for feature_type in FEAT_LIST:
        coeff, p_value, n = mantel(dist_mat, distance_dict[feature_type])
        print ("Feature type: %s Coeff: %.3f" %(feature_type, coeff))


def main():
    args = parse_args()

    # Load similarity matrix
    sim_mat = np.load(args.sim_mat_file)

    # Get distances for different features
    distance_dict = get_feature_dist_dicts()

    # Calculate correlation
    calc_dist_mat_corr(sim_mat, distance_dict)


if __name__=='__main__':
    main()
