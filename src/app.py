import os
import shutil
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import timestring
from h2o_wave import Q, app, handle_on, main, on, ui

from loguru import logger

from .views import (
    display_home,
    display_dataset,
    display_model,
    display_results,
    main_layout,
    ui_loading_card,
)
from .wave_utils import handle_crash

#need to fix datable install issue on the mac
#from .text_preprocessing_transformer import TextPreprocessingTransformer


import nltk
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel
import pandas as pd



LOCAL_TESTING = None
DATA_SAVE_LOCATION = None
DEFAULT_GROUP = "HAC_TRIAL_DEFAULT"
DEACTIVATE_GROUP = "TRIAL ENV DEACTIVATED"

def on_startup() -> None:
    """
    Runs when the app is first launched
    Determines how to set global variable based on where the app is running
    """
    global LOCAL_TESTING
    global DATA_SAVE_LOCATION


    if os.getenv("DB_URL") is not None:  # running in the H2O AI Hybrid Cloud
        LOCAL_TESTING = False
        DATA_SAVE_LOCATION = "/mount-data/"

        if os.path.exists("./data/"):
            for f in os.listdir(
                "./data"
            ):  # move files that came with the app to the ai cloud storage
                if not os.path.isfile(os.path.join("/mount-data", f)):
                    shutil.move(
                        os.path.join("./data", f), os.path.join("/mount-data", f)
                    )

    else:  # running locally
        LOCAL_TESTING = True
        DATA_SAVE_LOCATION = "./data/"
        if not os.path.exists(DATA_SAVE_LOCATION):
            os.makedirs(DATA_SAVE_LOCATION)

    logger.info("Starting app...")
    logger.info(f"Local Testing: {LOCAL_TESTING}")
    logger.info(f"Data Save Location: {DATA_SAVE_LOCATION}")


@app("/", on_startup=on_startup)
async def serve(q: Q):
    logger.info(f"Current query arguments: {q.args}")

    try:
        if not q.client.initialized:
            await initialize_app_for_new_client(q)

        if q.args.nav_home is not None:
            q.client.nav_page = "nav_home"
        if q.args.nav_dataset is not None:
            q.client.nav_page = "nav_dataset"
        if q.args.nav_model is not None:
            q.client.nav_page = "nav_model"
        if q.args.nav_results is not None:
            q.client.nav_page = "nav_results"

        if q.client.nav_page is not None:
            #display_command_card(q, choices, company_choices)
            setup_new_page(q)
            eval(q.client.nav_page)(q)

        if q.args.file_upload is not None:
            await handle_uploaded_data(q)

        await q.page.save()

    except Exception as unknown_exception:
        logger.error(str(unknown_exception))
        await handle_crash(q)


async def handle_uploaded_data(q: Q):
    """Saves a file uploaded by a user from the UI"""
    data_path = DATA_SAVE_LOCATION

    # Download new dataset to data directory
    q.client.working_file_path = await q.site.download(url=q.args.file_upload[0], path=data_path)


async def initialize_app_for_new_client(q: Q):
    logger.info(f"Initializing app for user: {q.auth.username}")

    q.client.app_name = "Topic View"
    q.client.theme = "h2o-dark"
    #q.client.theme = "light"
    q.client.cards = []

    main_layout(q)

    # Show loading dialog while app launches / gets data
    #await ui_loading_card(q)
    await q.page.save()

    q.client.nav_page = "nav_home"

    q.client.initialized = True



def nav_home(q: Q):
    logger.info("Preparing the Home view.")
    display_home(q)

def nav_dataset(q: Q):
    logger.info("Preparing the Dataset view.")
    display_dataset(q, DATA_SAVE_LOCATION)


def nav_model(q: Q):
    logger.info("Preparing the Model view.")
    display_model(q)


def nav_results(q: Q):
    global PYLDA_HTML_FILE

    logger.info("Preparing the Visualize view.")
    # Show not_implemented dialog until development is complete

    # Read the Dataset upload file into a data frame
    text_data = pd.read_csv(q.client.working_file_path, encoding_errors='ignore')
    text_data.rename(columns={q.args.text_column: 'text'}, inplace=True)

    #remove special characters
    text_data = text_data.replace(r'[^A-Za-z0-9 ]+', '', regex=True)

    if q.args.preprocess_checklist and 'StopWordRemoval' in q.args.preprocess_checklist:
        stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                      "you'd",
                      'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
                      'hers',
                      'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                      'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                      'were',
                      'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                      'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                      'about',
                      'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                      'from',
                      'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                      'here',
                      'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                      'other',
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                      'can',
                      'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                      'y',
                      'ain',
                      'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn',
                      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                      'needn',
                      "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                      "won't",
                      'wouldn', "wouldn't", 'from', 'subject', 're', 'edu', 'use',
                      '?', '&', '.', 'chipotle']
        stop_words = stop_words + q.args.stop_words.split(",")
        text_data['text'] = text_data['text'].str.lower().astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    if q.args.preprocess_checklist and 'Stemming' in q.args.preprocess_checklist:
        text_data['text'] = text_data['text'].str.split(' ')
        stemmer = nltk.stem.snowball.SnowballStemmer("english")
        text_data['text'] = text_data['text'].apply(lambda x: [stemmer.stem(word) for word in x])
        text_data['text'] = [' '.join(map(str, t)) for t in text_data['text']]

    if q.args.preprocess_checklist and 'Lemmatization' in q.args.preprocess_checklist:
            _spoof=1

    if q.args.preprocess_checklist:
        if 'Stemming' in q.args.preprocess_checklist:
            text_data['text'] = text_data['text'].astype(str)
            text_data = pd.DataFrame(text_data).text.values.tolist()
            text_data = [t.split(' ') for t in text_data]
        else:
            text_data['text'] = text_data['text'].astype(str)
            text_data = pd.DataFrame(text_data).text.values.tolist()
            text_data = [t.split(' ') for t in text_data]
    else:
        text_data['text'] = text_data['text'].astype(str)
        text_data = text_data.text.values.tolist()
        text_data = [t.split(' ') for t in text_data]


    # #add code for text pre-processing
    # if q.args.preprocess_checklist:
    #     #Use h2o driverless text pre-processor recipe
    #     tpt = TextPreprocessingTransformer()
    #     tpt.remove_stopwords = False
    #     tpt.do_stemming = False
    #     tpt.do_lemmatization = False
    #
    #     #remove stop words
    #     if 'StopWordRemoval' in q.args.preprocess_checklist:
    #         tpt.remove_stopwords = True
    #      #perform stemming
    #     if 'Stemming' in q.args.preprocess_checklist:
    #         tpt.do_stemming = True
    #     #perform lemmatization
    #     if 'Lemmatization' in q.args.preprocess_checklist:
    #         tpt.do_lemmatization = True
    #
    #     text_data_clean = tpt.transform(text_data_dt)
    #
    #     text_data = pd.DataFrame(text_data_clean).text.astype(str).values.tolist()


    #remove words that appear infrequently
    #text_data['text'] = text_data['text'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    #remove punctuation
    #tokens = list(filter(lambda token: token not in string.punctuation, tokens))


    # Build LDA model
    #text_data['text'] = text_data['text'].astype(str)
    #text_data = text_data['text'].values.tolist()
    #text_data = [t.split(' ') for t in text_data]
    id2word = Dictionary(text_data)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in text_data]

    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=q.args.num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         alpha='auto',
                         per_word_topics=True)
    prep_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

    # create and save the html visualization
    PYLDA_HTML_FILE = DATA_SAVE_LOCATION + 'pyldavis_html_file_' + time.strftime("%Y%m%d-%H%M%S") + '.html'
    pyLDAvis.save_html(prep_data, PYLDA_HTML_FILE)

    # Compute metrics
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_lda = round(coherence_lda,4)
    # Compute Perplexity
    perplexity_lda = np.exp(-1. * lda_model.log_perplexity(corpus))
    perplexity_lda = round(perplexity_lda,4)

    metrics_df = pd.DataFrame(
        [
            [str(q.args.num_topics) + ' Topics', coherence_lda, perplexity_lda]
        ],
        columns=[
            "Model", "coherence", "perplexity"
        ],
    )
    metrics_df["coherence"] = metrics_df["coherence"].astype(str)
    metrics_df["perplexity"] = metrics_df["perplexity"].astype(str)
    metrics_df["Model"] = metrics_df["Model"].astype(str)
    params = [
        "coherence",
        "perplexity",
    ]

    topics = lda_model.print_topics()
    topics_df = pd.DataFrame(topics, columns=['topic', 'probabilities'])

    display_results(q, metrics_df, params, topics_df, PYLDA_HTML_FILE)


def nav_visualize(q: Q):
    logger.info("Preparing the Visualize view.")
    display_visualize(q, DATA_SAVE_LOCATION, PYLDA_HTML_FILE)


def setup_new_page(q: Q):
    for c in q.client.cards:
        del q.page[c]

    q.client.cards = []
    q.page["sidebar"].value = q.client.nav_page
