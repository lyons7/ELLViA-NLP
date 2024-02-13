import pandas as pd
import numpy as np
import sklearn
# We are interested in bigrams and trigrams potentially, so have to format the data correctly first!
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy.lang.xx import MultiLanguage



"""Custom tokenizer for uni and bigrams"""
def custom_tokenizer(nlp):
    """Need a custome tokenizer so we don't break up hyphenated words like 'work-life balance'."""
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)


"""Fx to tokenize and get unigrams"""
def get_unigrams(dataFrame, textCol):  # textLangCol?
    tm_docs = []
    nlp = spacy.load("en_core_web_sm")
    # nlp_de = spacy.load("de_core_news_sm")  # Also doing a pass for German?
    nlp.tokenizer = custom_tokenizer(nlp)  # Maybe we don't want tokenizer that splits on '-'?
    # nlp_de.tokenizer = custom_tokenizer(nlp_de)  # ?
    for doc in nlp.pipe(dataFrame[textCol]):  # nlp.pipe more efficient for larger datasets (see https://spacy.io/usage/processing-pipelines#processing)
        tm_docs.append([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])  # Tokenize based on custom tokenizer
        # tm_docs.append([token.lemma_ for token in doc if not token.is_punct])  # Tokenize based on custom tokenizer -- FOLLOW MONROE, keep stop words for WLO 
    return tm_docs


"""Fx to tokenize and get bigrams"""
def get_bigrams(dataFrame, textCol):
    tm_docs = []
    bigrams = []
    nlp = spacy.load("en_core_web_sm")
    # nlp = MultiLanguage()  # Because we have mixed languages?
    nlp.tokenizer = custom_tokenizer(nlp)  # Maybe we don't want tokenizer that splits on '-'?
    for doc in nlp.pipe(dataFrame[textCol]):  # nlp.pipe more efficient for larger datasets (see https://spacy.io/usage/processing-pipelines#processing)
        tm_docs.append([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])  # Tokenize based on custom tokenizer
    for doc in tm_docs:
        bigrams.append([(doc[i], doc[i+1]) for i in range(0, len(doc)-1)])
    return bigrams