import argparse
import faiss
import logging
from collections import OrderedDict
from os import path
import numpy as np
import torch

from utils import load_all_embeddings,load_all_translations
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


def compute_csls_neighbors(x_src, x_tgt, nn_distance_to_src_words):
    """Compute CSLS score
    x_src: B x d
    x_tgt: B' x d
    nn_distance_to_src_words: B'
    """
    bs = 128
    x_tgt = x_tgt.transpose(0, 1).contiguous()
    similarities = []
    for i in range(0, x_src.shape[0], bs):
        batch_sims = x_src[i:i + bs].mm(x_tgt)
        batch_sims.mul_(2)
        similarities.append(batch_sims.cpu())

    similarities = torch.cat(similarities)
    similarities -= torch.unsqueeze(
        torch.tensor(nn_distance_to_src_words), dim=0)
    nn_idx = torch.argmax(similarities, dim=1, keepdim=True)
    return nn_idx.numpy()


def calc_dict_ind_perf(lang_to_emb, cuda_embeddings,
                       all_pairs_translation_dict, k=1):
    dict_ind = np.zeros((len(lang_codes), len(lang_codes)))

    for i, src_lang in enumerate(lang_codes):
        src_word2id = lang_to_emb[src_lang]['word2id']
        src_embeddings = lang_to_emb[src_lang]['embeddings']
        src_index = faiss.IndexFlatIP(300)
        src_index = faiss.index_cpu_to_gpu(res, 0, src_index)
        src_index.add(lang_to_emb[src_lang]['embeddings'])

        for j, tgt_lang in enumerate(lang_codes):
            if i == j:
                continue

            tgt_embeddings = lang_to_emb[tgt_lang]['embeddings']
            tgt_cuda_embeddings = cuda_embeddings[tgt_lang]
            tgt_word2id = lang_to_emb[tgt_lang]['word2id']

            translation_pairs = all_pairs_translation_dict[(src_lang, tgt_lang)]

            num_test = len(translation_pairs)
            corr_tgt_idx = np.zeros(num_test, dtype=np.int32)
            src_lang_idx = np.zeros(num_test, dtype=np.int32)

            for idx, (word, translation) in enumerate(translation_pairs):
                if word in src_word2id:
                    src_lang_idx[idx] = src_word2id[word]
                if translation in tgt_word2id:
                    corr_tgt_idx[idx] = tgt_word2id[translation]

            dict_src_embeddings = cuda_embeddings[src_lang][src_lang_idx]

            D_neigh, I_neigh = src_index.search(tgt_embeddings, 10)
            nn_distance_to_src_words = np.mean(D_neigh, axis=1)
            nn_idx = compute_csls_neighbors(
                dict_src_embeddings, tgt_cuda_embeddings,
                nn_distance_to_src_words)

            corr_tgt_idx = np.expand_dims(corr_tgt_idx, axis=1)
            corr_tgt_idx = np.repeat(corr_tgt_idx, k, axis=1)

            perf = np.sum(corr_tgt_idx == nn_idx)
            norm_perf = perf/num_test

            dict_ind[i, j] = norm_perf

    logging.info("Max perf: %.3f" %np.amax(dict_ind))
    np.fill_diagonal(dict_ind, 1.0)
    logging.info("Min perf: %.3f" %np.amin(dict_ind))

    np.fill_diagonal(dict_ind, 1)
    return dict_ind


def main():
    args = parse_args()

    # Load all embeddings
    logging.info("Loading all embeddings.")
    lang_to_emb = load_all_embeddings(
        emb_dir=args.emb_dir, vocab_dir=None, threshold=None)

    logging.info("Copying embeddings to cuda.")
    cuda_embeddings = {}
    for lang in lang_codes:
        cuda_embeddings[lang] = torch.tensor(
            lang_to_emb[lang]['embeddings']).to(cuda0)

    # Load all dictionaries
    logging.info("Loading all dictionaries")
    all_pairs_translation_dict = load_all_translations(
        dict_dir=args.dict_dir, threshold=args.threshold)

    # Calculate word translation performance
    logging.info("Calculate word translation performance.")
    dict_ind_mat = calc_dict_ind_perf(
        lang_to_emb, cuda_embeddings, all_pairs_translation_dict)

    logging.info("Save similarity matrix.")
    if not args.threshold:
        prefix = "all"
    else:
        prefix = str(args.threshold)
    np.save(path.join(args.out_dir, "dict_ind_mat_{}.npy".format(prefix)),
            dict_ind_mat)


if __name__=='__main__':
    main()
