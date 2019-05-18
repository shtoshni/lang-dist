import numpy as np
import argparse

from skbio.tree import TreeNode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tree1_file', help='First tree file', type=str)
    parser.add_argument(
        '--tree2_file', help='Second tree file', type=str)

    args = parser.parse_args()
    return args


def calc_tree_distance(tree1, tree2):
    """Calculate difference in leaf-pair distances."""
    tree1_struct = tree1.tip_tip_distances()
    tree1_distances = tree1_struct.data
    tree1_ids = tree1_struct.ids

    tree2_distances = tree2.tip_tip_distances(endpoints=list(tree1_ids)).data
    tree_diff = np.sum((tree1_distances - tree2_distances)**2)
    return tree_diff


def main():
    args = parse_args()

    tree1 = TreeNode.read(open(args.tree1_file))
    tree2 = TreeNode.read(open(args.tree2_file))

    tree_dist = calc_tree_distance(tree1, tree2)
    print ("Tree distance: %d" %tree_dist)


if __name__=='__main__':
    main()
