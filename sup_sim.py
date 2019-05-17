import argparse
import faiss
import logging
import numpy as np

from collections import OrderedDict
from os import path

from scipy.spatial.distance import cosine

from utils import load_all_embeddings,load_all_translations
from data_utils import lang_codes, languages

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
        "--dict_dir",
        default='/home/shtoshni/Research/MUSE/data/google_translate/filtered/muse_unsup',
        help="Dictionary (and vocab) directory", type=str)

    parser.add_argument(
        "--out_dir",
        default='/home/shtoshni/Research/MUSE/results/google_translate/muse_unsup',
        help="Output directory", type=str)

    parser.add_argument(
        "--threshold", default=2500,
        help="# of translations used in word translation performance", type=int)

    args = parser.parse_args()
    return args


def calc_sim_mat(lang_to_emb, all_pairs_translation_dict):
    sim_mat = np.zeros((len(lang_codes), len(lang_codes)))
    for i, src_lang in enumerate(lang_codes):
        for j, tgt_lang in enumerate(lang_codes):
            if i == j:
                continue

            translation_pairs = all_pairs_translation_dict[(src_lang, tgt_lang)]
            src_embeddings = lang_to_emb[src_lang]['embeddings']
            src_word2id = lang_to_emb[src_lang]['word2id']

            tgt_embeddings = lang_to_emb[tgt_lang]['embeddings']
            tgt_word2id = lang_to_emb[tgt_lang]['word2id']

            avg_sim = 0
            for word, translation in translation_pairs:
                vec1 = src_embeddings[src_word2id[word]]
                vec2 = tgt_embeddings[tgt_word2id[translation]]

                avg_sim += 1 - cosine(vec1, vec2)

            avg_sim /= len(translation_pairs)
            sim_mat[i, j] = avg_sim

    print ("Max similarity:", np.amax(sim_mat))
    np.fill_diagonal(sim_mat, 1.0)
    print ("Min similarity:", np.amin(sim_mat))
    return sim_mat


def main():
    args = parse_args()

    # Load all embeddings
    logging.info("Loading all embeddings.")
    lang_to_emb = load_all_embeddings(
        emb_dir=args.emb_dir, vocab_dir=args.dict_dir,
        threshold=None)

    logging.info("Loading all dictionaries")
    all_pairs_translation_dict = load_all_translations(
        dict_dir=args.dict_dir, threshold=args.threshold)

    logging.info("Calculate similarity matrix.")
    sup_sim_mat = calc_sim_mat(lang_to_emb, all_pairs_translation_dict)

    logging.info("Save similarity matrix.")
    if not args.threshold:
        prefix = "all"
    else:
        prefix = str(args.threshold)
    np.save(path.join(args.out_dir, "sup_sim_mat_{}.npy".format(prefix)),
            sup_sim_mat)


if __name__=='__main__':
    main()
