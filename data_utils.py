from os import path
import glob

# lang_codes = [
#     # Romance
#     'fr', 'it', 'pt', 'es', 'ca', 'ro',
#     # Germanic
#     'en', 'da', 'nl', 'de', 'no', 'sv',
#     # Finno-Ugric
#     'et', 'fi', 'hu',
#     # Slavic
#     'hr', 'sl', 'cs', 'sk', 'mk', 'bg', 'pl', 'ru', 'uk',
#     # Hellenic
#     'el',
#     # Random
#     'ar', 'iw', 'id', 'tr'
# ]

lang_codes = [
    # Romance
    'fr', 'it', 'pt', 'es', 'ca', 'ro',
    # Germanic
    'en', 'da', 'nl', 'de', 'no', 'sv',
    # Finno-Ugric
    'et', 'fi', 'hu',
    # Slavic
    'hr', 'sl', 'cs', 'sk', 'mk', 'bg', 'pl', 'ru', 'uk',
    # Hellenic
    'el',
    # Random
    'ar', 'iw', 'id', 'tr'
]

# Just Indo-European langs
indo_european = [
    # Romance
    'fr', 'it', 'pt', 'es', 'ca', 'ro',
    # Germanic
    'en', 'da', 'nl', 'de', 'no', 'sv',
    # Slavic
    'hr', 'sl', 'cs', 'sk', 'mk', 'bg', 'pl', 'ru', 'uk',
    # Hellenic
    'el'
]

lang_code_to_language = {
    'ar': 'Arabic', 'bg': 'Bulgarian', 'ca': 'Catalan', 'hr': 'Croatian',
    'cs':'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
    'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'de': 'German',
    'el': 'Greek', 'iw': 'Hebrew', 'hu': 'Hungarian', 'id': 'Indonesian',
    'it': 'Italian', 'mk': 'Macedonian', 'no': 'Norwegian', 'pl': 'Polish',
    'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk':'Slovak',
    'sl': 'Slovenian', 'es': 'Spanish', 'sv': 'Swedish', 'tr': 'Turkish',
    'uk': 'Ukrainian',
    }

languages = [lang_code_to_language[lang_code] for lang_code in lang_codes]


def get_emb_file(emb_dir, lang_code):
    if lang_code == "iw":
        # Language code for Hebrew is different in Google Translate and
        # FastText embeddings
        lang_code = "he"
    emb_path_pattern = path.join(emb_dir, "*." + lang_code + ".*vec")
    all_emb_paths = glob.glob(emb_path_pattern)
    assert (len(all_emb_paths) == 1)
    return all_emb_paths[0]

def get_vocab_file(vocab_dir, lang_code):
    return path.join(vocab_dir, lang_code + ".txt")
