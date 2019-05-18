import argparse
import faiss
import logging
from collections import OrderedDict
from os import path
import numpy as np

from utils.utils import load_all_embeddings
from utils.data_utils import lang_codes, languages

res = faiss.StandardGpuResources()
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '--emb_dir',
        default='/home/shtoshni/Research/MUSE/data/multilingual/',
        help='Embedding directory', type=str)

    parser.add_argument(
        "--vocab_dir",
        default='/home/shtoshni/Research/MUSE/data/google_translate/filtered/muse_unsup',
        help="Vocab directory", type=str)

    parser.add_argument(
        "--out_dir",
        default='/home/shtoshni/Research/MUSE/results/google_translate/muse_unsup',
        help="Output directory", type=str)

    parser.add_argument(
        "--threshold", default=10000,
        help="# of terms used in unsupervised analysis", type=int)

    args = parser.parse_args()
    return args


def calc_sim_mat(lang_to_emb, k=1):
    sim_mat = np.zeros((len(lang_codes), len(lang_codes)))
    for j, tgt in enumerate(lang_codes):
        tgt_index = faiss.IndexFlatIP(300)
        tgt_index = faiss.index_cpu_to_gpu(res, 0, tgt_index)
        tgt_index.add(lang_to_emb[tgt]['embeddings'])
        for i, src in enumerate(lang_codes):
            if i == j:
                sim_mat[i, i] = 0.0
            else:
                D, I = tgt_index.search(lang_to_emb[src]['embeddings'], k)
                sim_mat[i, j] = np.mean(D)

    logging.info("Max similarity: %.3f" %np.amax(sim_mat))
    np.fill_diagonal(sim_mat, 1.0)
    logging.info("Min similarity: %.3f" %np.amin(sim_mat))
    return sim_mat


def main():
    args = parse_args()

    # Load all embeddings
    logging.info("Loading all embeddings.")
    lang_to_emb = load_all_embeddings(
        emb_dir=args.emb_dir, vocab_dir=args.vocab_dir,
        threshold=args.threshold)

    logging.info("Calculate similarity matrix.")
    sim_mat = calc_sim_mat(lang_to_emb)

    logging.info("Save similarity matrix.")
    np.save(path.join(args.out_dir, "unsup_sim_mat.npy"), sim_mat)


if __name__=='__main__':
    main()
