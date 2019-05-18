import io
import logging
import numpy as np

from collections import OrderedDict
from os import path

from .data_utils import get_emb_file, get_vocab_file
from .data_utils import languages, lang_codes


def load_embeddings(emb_path, vocab, threshold=None):
    """Load embeddings corresponding to entries in vocab."""
    word2id = {}
    vectors = []
    max_norm = 0
    min_norm = 1e9

    if threshold is None:
        threshold = 200000
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                if vocab and (word not in vocab):
                    continue
                vect = np.fromstring(vect, sep=' ', dtype=np.float32)
                # vect_norm = np.linalg.norm(vect)
                # if max_norm < vect_norm:
                #     max_norm = vect_norm
                # if min_norm > vect_norm:
                #     min_norm = vect_norm
                # vect = vect / vect_norm

                word2id[word] = len(word2id)
                vectors.append(vect[None])

            if threshold and len(word2id) >= threshold:
                break

    assert len(word2id) == len(vectors)

    # if not np.isclose(max_norm, 1, atol=1e-3):
    #     logging.info("Max norm: %.3f" %max_norm)
    # if not np.isclose(min_norm, 1, atol=1e-3):
    #     logging.info("Min norm: %.3f" %min_norm)

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return id2word, word2id, embeddings


def load_vocab(vocab_path):
    """Load vocab."""
    vocab = set()
    with io.open(vocab_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as vocab_f:
        for line in vocab_f:
            token = line.strip()
            if token not in vocab:
                vocab.add(token)
    return vocab


def load_all_embeddings(emb_dir, vocab_dir, threshold=None):
    lang_to_emb = OrderedDict()
    for lang_code, language in zip(lang_codes, languages):
        emb_file = get_emb_file(emb_dir, lang_code)
        if vocab_dir:
            vocab_file = get_vocab_file(vocab_dir, lang_code)
            lang_vocab = load_vocab(vocab_file)
        else:
            lang_vocab = None

        id2word, word2id, embeddings = load_embeddings(emb_file, lang_vocab, threshold=threshold)

        if threshold and (len(word2id) != threshold):
            logging.warning("Not enough words for lang: %s" %language)

        lang_to_emb[lang_code] = {"id2word": id2word, "word2id": word2id, "embeddings": embeddings}
        logging.info("Language: %s done, # of terms: %d" %(language, len(id2word)))

    return lang_to_emb


def load_translations(dict_dir, src_lang, tgt_lang, threshold=None):
    translation_file = path.join(dict_dir, src_lang + "_" + tgt_lang + ".txt")
    translation_pairs = []
    with io.open(translation_file, mode="r", encoding="utf-8") as translation_file:
        for line in translation_file:
            word, translation = line.strip().split()
            translation_pairs.append((word, translation))
            if threshold and len(translation_pairs) >= threshold:
                break

    return translation_pairs


def load_all_translations(dict_dir, threshold=None):
    all_pairs_translation_dict = {}
    for src_lang in lang_codes:
        for tgt_lang in lang_codes:
            if src_lang == tgt_lang:
                continue
            translation_pairs = load_translations(
                dict_dir, src_lang, tgt_lang, threshold=threshold)
            all_pairs_translation_dict[(src_lang, tgt_lang)] = translation_pairs

    return all_pairs_translation_dict


def get_non_diagonal_entries(data_mat):
    num_rows, _ = data_mat.shape
    lower_tri_indices = np.tril_indices(num_rows, k=-1)
    upper_tri_indices = np.triu_indices(num_rows, k=1)

    entries = list(data_mat[lower_tri_indices]) + list(data_mat[upper_tri_indices])
    return entries

def create_symm_dist_mat(sim_mat):
    # Make sure the matrix is symmetric
    sim_mat = (sim_mat + sim_mat.T)/2
    # Create the distance matrix
    dist_mat = 1.0 - sim_mat
    # Zero out diagonals
    np.fill_diagonal(dist_mat, 0)

    return dist_mat
