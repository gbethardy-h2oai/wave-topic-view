import time
from typing import List

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from nltk.stem.snowball import SnowballStemmer


def clean_text(text_data: pd.core.series.Series):
    text_data = text_data.replace(r"[^A-Za-z0-9 ]+", "", regex=True)
    return text_data.str.lower()


def remove_stop_words(text_data: pd.core.series.Series, alternate_stop_words: List):
    stop_words = [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "from",
        "subject",
        "re",
        "edu",
        "use",
        "?",
        "&",
        ".",
        "chipotle",
    ]
    stop_words = stop_words + alternate_stop_words
    return text_data.apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )


def stem_words(text_data: pd.core.series.Series):
    text_data = text_data.str.split(" ")
    stemmer = SnowballStemmer("english")
    text_data = text_data.apply(lambda x: [stemmer.stem(word) for word in x])
    text_data = pd.Series([" ".join(map(str, t)) for t in text_data])
    return text_data


def topic_modeling(text_data: pd.core.series.Series, number_of_topics, data_location):
    # Transform series to a list of list
    text_data = text_data.values.tolist()
    text_data = [t.split(" ") for t in text_data]

    # Term Document Frequency
    id2word = Dictionary(text_data)
    corpus = [id2word.doc2bow(t) for t in text_data]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=number_of_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        alpha="auto",
        per_word_topics=True,
    )

    prep_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

    html_file_location = (
        data_location + "pyldavis_html_file_" + time.strftime("%Y%m%d-%H%M%S") + ".html"
    )

    pyLDAvis.save_html(prep_data, html_file_location)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=text_data, dictionary=id2word, coherence="c_v"
    )
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_lda = round(coherence_lda, 4)

    # Compute Perplexity
    perplexity_lda = np.exp(-1.0 * lda_model.log_perplexity(corpus))
    perplexity_lda = round(perplexity_lda, 4)

    # Compute Topics
    topics = lda_model.print_topics()
    topics = pd.DataFrame(topics, columns=["topic", "probabilities"])

    return html_file_location, coherence_lda, perplexity_lda, topics
