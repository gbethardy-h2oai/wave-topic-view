import os

import pandas as pd
from h2o_wave import Q, app, handle_on, main, on, ui

from .topic_analytics import clean_text, remove_stop_words, stem_words, topic_modeling


@app("/")
async def serve(q: Q):
    print(q.args)
    print(q.client)

    if not q.client.initialized:
        # First time a browser comes to the app
        await init(q)
        q.client.initialized = True

    elif q.args.file_upload:
        # User has clicked the upload data button
        if q.client.file_path is not None:
            os.remove(q.client.file_path)

        # Save the location of the data and the dataframe to use in the app
        q.client.file_path = await q.site.download(
            url=q.args.file_upload[0], path=q.app.data_save_location
        )
        q.client.original_data = pd.read_csv(
            q.client.file_path, encoding_errors="ignore"
        )

        # Show the upload data page again
        await nav_import(q)

    elif q.args.configure_model:
        # User has clicked the Run button after configuring data and model

        # User has clicked the upload data button
        if q.client.html_file_location is not None:
            os.remove(q.client.html_file_location)

        # Save the variables for use throughout the app
        q.client.text_column = q.args.text_column
        q.client.remove_stop_words = (
            True if "StopWordRemoval" in q.args.preprocess_checklist else False
        )
        q.client.stem_words = (
            True if "Stemming" in q.args.preprocess_checklist else False
        )
        q.client.additional_stop_words = q.args.stop_words.split(",")
        q.client.num_topics = q.args.num_topics

        # Prepare text data
        text = q.client.original_data[q.client.text_column]
        text = clean_text(text)

        if q.client.remove_stop_words:
            text = remove_stop_words(text, q.client.additional_stop_words)

        if q.client.stem_words:
            text = stem_words(text)

        (
            q.client.html_file_location,
            q.client.coherence_lda,
            q.client.perplexity_lda,
            q.client.topics,
        ) = topic_modeling(text, q.client.num_topics, q.app.data_save_location)

        await nav_results(q)

    else:
        # Other browser interactions
        await handle_on(q)

    await q.page.save()


async def init(q: Q) -> None:
    q.client.cards = []

    if not q.app.initialized:
        # Upload the home page image to the server once to use for all users
        (q.app.header_image,) = await q.site.upload(["wordcloud_splash.png"])

        # All user data is saved in the save folder
        q.app.data_save_location = "./data/"
        if not os.path.exists(q.app.data_save_location):
            os.makedirs(q.app.data_save_location)

        q.app.initialized = True

    q.page["meta"] = ui.meta_card(
        box="",
        title="H2O Topic View",
        theme="light",
        layouts=[
            ui.layout(
                breakpoint="xs",
                min_height="100vh",
                max_width="1500px",
                zones=[
                    ui.zone("header"),
                    ui.zone(
                        "content",
                        size="1",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone("navigation", size="15%"),
                            ui.zone("main"),
                        ],
                    ),
                    ui.zone(name="footer"),
                ],
            )
        ],
    )
    q.page["header"] = ui.header_card(
        box="header",
        title="Topic View",
        subtitle="Visually explore topic and word distributions",
        image="https://cloud.h2o.ai/logo.svg",
    )
    q.page["navigation"] = ui.nav_card(
        box=ui.box("navigation", size=0),
        items=[
            ui.nav_group(
                label="Workflow Menu",
                items=[
                    ui.nav_item("nav_home", "Home", "Home"),
                    ui.nav_item("nav_import", "Data Upload", "TableGroup"),
                    ui.nav_item("nav_configure", "Configuration", "TestPlan"),
                    ui.nav_item("nav_results", "Results", "Diagnostic"),
                ],
            ),
        ],
    )
    q.page["footer"] = ui.footer_card(
        box="footer", caption="Made with 💛 using [H2O Wave](https://wave.h2o.ai)."
    )

    await nav_home(q)


@on()
async def nav_home(q: Q):
    clear_cards(q)
    q.page["navigation"].value = "nav_home"

    q.page["home"] = ui.form_card(
        box="main",
        items=[
            ui.text_xl("Model your text data:"),
            ui.text("1. Upload a dataset"),
            ui.text("2. Configure and run the model"),
            ui.text("3. View the results"),
            ui.message_bar(
                type="info", text="It may take a few minutes to model your data."
            ),
            ui.text(f'<center><img src="{q.app.header_image}"></center>'),
        ],
    )
    q.client.cards.append("home")


@on()
async def nav_import(q: Q):
    clear_cards(q)
    q.page["navigation"].value = "nav_import"

    # If a dataset has already been uploaded, lets show it on this page
    header = ui.text(" ")
    table = ui.text(" ")
    if q.client.original_data is not None:
        print(q.client.original_data.head())
        header = ui.text_l("Data Sample")
        table = ui.table(
            name="sample_data",
            columns=[
                ui.table_column(col, col) for col in q.client.original_data.columns
            ],
            rows=[
                ui.table_row(
                    str(i),
                    [
                        str(q.client.original_data.loc[i, col])
                        for col in q.client.original_data.columns
                    ],
                )
                for i in range(len(q.client.original_data.head()))
            ],
        )

    q.page["import"] = ui.form_card(
        box="main",
        items=[
            ui.file_upload(
                name="file_upload",
                label="Import a CSV",
                compact=False,
                multiple=False,
                file_extensions=["csv"],
                max_file_size=50,
            ),
            header,
            table,
        ],
    )
    q.client.cards.append("import")


@on()
async def nav_configure(q: Q):
    clear_cards(q)
    q.page["navigation"].value = "nav_configure"

    if q.client.original_data is None:
        # Users cannot configure without a dataset
        no_data(q)
        return

    # If the user has previously configured, we will show those values
    q.page["configure"] = ui.form_card(
        box="main",
        items=[
            ui.dropdown(
                name="text_column",
                label="Text Column",
                required=True,
                choices=[ui.choice(col, col) for col in q.client.original_data.columns],
                value=None if q.client.text_column is None else q.client.text_column,
            ),
            ui.checklist(
                name="preprocess_checklist",
                label="Text Pre-Processing",
                choices=[
                    ui.choice(
                        "StopWordRemoval", "StopWordRemoval", q.client.remove_stop_words
                    ),
                    ui.choice("Stemming", "Stemming", q.client.stem_words),
                    ui.choice("Lemmatization", "Lemmatization", disabled=True),
                ],
            ),
            ui.slider(
                name="num_topics",
                label="Number of Topics",
                min=2,
                max=100,
                step=1,
                value=5 if q.client.num_topics is None else q.client.num_topics,
            ),
            ui.textbox(
                name="stop_words",
                label="Additional Stopwords: add to the default list, separate with a comma",
                value=None
                if q.client.additional_stop_words is None
                else ", ".join(q.client.additional_stop_words),
            ),
            ui.button(name="configure_model", label="Submit", primary=True),
        ],
    )
    q.client.cards.append("configure")


@on()
async def nav_results(q: Q):
    clear_cards(q)
    q.page["navigation"].value = "nav_results"

    if q.client.original_data is None:
        # Users cannot model without a dataset
        no_data(q)
        return
    if q.client.text_column is None:
        # Users cannot model without a text column
        no_config(q)
        return

    q.page["stats_information_card"] = ui.form_card(
        box=ui.box("main", size=0),
        items=[
            ui.stats(
                justify="between",
                items=[
                    ui.stat(
                        label="Topics",
                        value=str(q.client.num_topics),
                        icon="NumberSymbol",
                    ),
                    ui.stat(
                        label="Coherence",
                        value=str(q.client.coherence_lda),
                        icon="Puzzle",
                    ),
                    ui.stat(
                        label="Perplexity",
                        value=str(q.client.perplexity_lda),
                        icon="SunQuestionMark",
                    ),
                ],
            )
        ],
    )
    q.client.cards.append("stats_information_card")

    # This seems like the same content as the HTML of Topics, I'm not sure it's adding anything
    # q.page["topics_table"] = ui.stat_list_card(
    #     box=ui.box('main', size=0),
    #     title="Word Probabilities by Topic",
    #     items=[
    #         ui.stat_list_item(
    #             label=str(topics.loc[i, "topic"]),
    #             caption=topics.loc[i, "probabilities"]
    #         ) for i in range(len(topics))
    #     ]
    # )
    # q.client.cards.append("topics_table")

    with open(q.client.html_file_location) as fp:
        topic_html = fp.read()

    q.page["topics_html"] = ui.frame_card(box="main", title="", content=topic_html)
    q.client.cards.append("topics_html")


def no_data(q: Q):
    q.page["no_data"] = ui.form_card(
        box="main", items=[ui.text("Please upload a dataset!")]
    )
    q.client.cards.append("no_data")


def no_config(q: Q):
    q.page["no_config"] = ui.form_card(
        box="main", items=[ui.text("Please configure your model and dataset!")]
    )
    q.client.cards.append("no_config")


def clear_cards(q: Q):
    # Remove temporal cards from the page
    for card in q.client.cards:
        del q.page[card]

    q.client.cards = []
