import argparse
import faiss
import logging
from collections import OrderedDict
from os import path
import numpy as np
import torch

from utils import load_all_embeddings
from data_utils import lang_codes, languages

res = faiss.StandardGpuResources()
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
cuda0 = torch.device('cuda:0')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '--emb_dir',
        default='/home/shtoshni/Research/MUSE/data/rcsls/',
        help='Embedding directory', type=str)

    parser.add_argument(
        "--vocab_dir",
        default='/home/shtoshni/Research/MUSE/data/google_translate/filtered/muse_unsup',
        help="Vocab directory", type=str)

    parser.add_argument(
        "--out_dir",
        default='/home/shtoshni/Research/MUSE/results/google_translate/rcsls',
        help="Output directory", type=str)

    parser.add_argument(
        "--threshold", default=10000,
        help="# of terms used in unsupervised analysis", type=int)

    args = parser.parse_args()
    return args


def compute_csls_score(x_src, x_tgt, distance_to_src_words,
                       distance_to_tgt_words):
    """Compute CSLS score
    x_src: B x d
    x_tgt: B' x d
    distance_to_src_words: B'
    distance_to_tgt_words: B
    """
    # sc = np.dot(x_src, x_tgt.T)

    sc = torch.mm(x_src, torch.transpose(x_tgt, 0, 1))
    similarities = 2 * sc

    # Remove avg distance to src words
    similarities -= torch.unsqueeze(
        torch.tensor(distance_to_src_words), dim=0).to(cuda0)

    # Remove avg distance to tgt words
    similarities -= torch.unsqueeze(
        torch.tensor(distance_to_tgt_words), dim=1).to(cuda0)

    nn, _ = torch.max(similarities, dim=1)
    mean_nn = torch.mean(nn).item()
    return mean_nn


def calc_sim_mat(lang_to_emb, cuda_embeddings, faiss_indices, knn=10):
    sim_mat = np.zeros((len(lang_codes), len(lang_codes)))
    for i, src in enumerate(lang_codes):
        src_index = faiss_indices[src]
        for j, tgt in enumerate(lang_codes):
            if i == j:
                sim_mat[i, i] = 0.0
                continue

            tgt_index = faiss_indices[tgt]

            D_src, _ = src_index.search(lang_to_emb[tgt]['embeddings'], knn)
            distance_to_src_words = np.mean(D_src, axis=1)

            D_tgt, _ = tgt_index.search(lang_to_emb[src]['embeddings'], knn)
            distance_to_tgt_words = np.mean(D_tgt, axis=1)

            sim_mat[i, j] = compute_csls_score(
                cuda_embeddings[src],
                cuda_embeddings[tgt],
                distance_to_src_words,
                distance_to_tgt_words
            )

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

    cuda_embeddings = {}
    for lang in lang_codes:
        cuda_embeddings[lang] = torch.tensor(
            lang_to_emb[lang]['embeddings']).to(cuda0)

    faiss_indices = {}
    for lang in lang_codes:
        faiss_indices[lang] = faiss.IndexFlatIP(300)
        faiss_indices[lang] = faiss.index_cpu_to_gpu(
            res, 0, faiss_indices[lang])
        faiss_indices[lang].add(lang_to_emb[lang]['embeddings'])


    logging.info("Calculate similarity matrix.")
    sim_mat = calc_sim_mat(lang_to_emb, cuda_embeddings, faiss_indices)

    logging.info("Save similarity matrix.")
    np.save(path.join(args.out_dir, "unsup_sim_mat.npy"), sim_mat)


if __name__=='__main__':
    main()
