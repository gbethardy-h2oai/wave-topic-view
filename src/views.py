import os
import base64
import glob

import pandas as pd
import numpy as np
from h2o_wave import Q, data, ui


def main_layout(q: Q):

    q.page["__meta_card__"] = ui.meta_card(
        box="",
        title=q.client.app_name,
        theme=q.client.theme,
        layouts=[
            ui.layout(
                breakpoint="xs",
                max_width="1250px",
                zones=[
                    ui.zone("header"),
                    ui.zone("commands"),
                    ui.zone("body"),
                    ui.zone(
                        "sidebar_body",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone("sidebar", size="20%"),
                            ui.zone("content", size="80%"),
                        ],
                    ),
                    ui.zone("footer"),
                ],
            )
        ],
    )

    q.page["header_card"] = ui.header_card(
        box="header",
        title="Topic View",
        subtitle="Visually explore topic and word distributions",
        commands=[
            # ui.command(name="color_theme", label="Dark Mode", icon="ClearNight"),
        ],
    )

    q.page["sidebar"] = ui.nav_card(
        box=ui.box("sidebar", size=0),
        items=[
            ui.nav_group(
                label="Workflow Menu",
                items=[
                    ui.nav_item("nav_home", "Home", "Home"),
                    ui.nav_item("nav_dataset", "Dataset Upload", "TableGroup"),
                    ui.nav_item("nav_model", "Topic Model Configuration", "TestPlan"),
                    ui.nav_item("nav_results", "Topic/Word Visual Exploration", "Diagnostic")
                ],
            ),
        ],
    )


def display_home(q: Q):
    q.client.theme = "h2o-dark"
    main_layout(q)

    with open('wordcloud_splash.png', "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    q.page['splash_card'] = ui.form_card(
        box=ui.box(zone="content", width="600px", height="600px"),
        items=[

        ui.text('Follow the workflow menu through:'),
        ui.text('Dataset upload>Topic Model Configuration>Topic/Word Visual Exploration'),
        ui.text('Make sure to work through each item in succession, and to populate required dataset file and text column name'),
        ui.text('Depending on the configuration, it may take a few minutes to generate/load the topic model/visuals.'),
        ui.text(' '),
        ui.text('To clear and start over, click the reload/refresh button on your browser'),
        ui.text(' '),
        ui.image(title='Topic View new',
                path=f'data:image/png;base64,{image_data}',
                type='png'),
    ])

    q.client.cards.append("splash_card")


def display_dataset(q: Q, DATA_SAVE_LOCATION):
    q.client.theme = "h2o-dark"
    main_layout(q)

    #clean up last dataset
    csvList = glob.glob(DATA_SAVE_LOCATION + './*.csv')
    for filePath in csvList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    #ui.button(name='submit', label='Submit', primary=True)
    q.page['file_upload'] = ui.form_card(
        box=ui.box(zone="content", order=2),
        items=[
            ui.file_upload(name='file_upload', label='Select csv Dataset file <50MB to upload (required)', compact=True,
                           multiple=False, file_extensions=['csv'], width="375px", max_file_size=50)

        ]
    )

    fileList = glob.glob(DATA_SAVE_LOCATION + './pylda*.html')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    q.client.cards.append("file_upload")
    q.page.save()



def display_model(q: Q):
    q.client.theme = "h2o-dark"
    main_layout(q)

    #q.args.file_upload
    #q.client.working_file_path
    if not q.args.file_upload:
        q.page["__meta_card__"].dialog = ui.dialog(
            title="Please go back and upload a Dataset!",
            items=[
                ui.progress(
                    label="Please go back and upload a Dataset!",
                    caption="",
                    value=None,
                    visible=True,
                    tooltip="",
                    name="",
                ),
            ],
        )
        q.client.cards.append("__meta_card__")

    else:
        q.page['model_card'] = ui.form_card(
            box=ui.box(zone="content", order=1),
            items=[

            ui.textbox(name='text_column', label='Text Column (required)', icon='Variable', width="200px"),
            ui.checklist(name='preprocess_checklist',
                         label='Text Pre-Processing',
                         choices=[ui.choice(name=x, label=x) for x in ['StopWordRemoval', 'Stemming', 'Lemmatization (not yet implementned']]),
            ui.spinbox(name='num_topics', label='Number of Topics (required)', min=2, max=100, step=1, value=5, width="125px"),
            ui.textbox(name='stop_words', label='Add additional stop words separated by comma'),
            ui.text('Make sure that you have selected a Dataset and valid text column before generating the model.'),
                ui.text(' '),
            ui.text('Depending on the configuration, it may take a few minutes to generate/load the topic model/visuals.'),
        ])

        q.client.cards.append("model_card")


def display_results(q: Q, metricsdf, params, topicsdf, PYLDA_HTML_FILE):
    q.client.theme = "light"
    main_layout(q)

    def make_markdown_row(values):
        return f"| {' | '.join([str(x) for x in values])} |"

    def make_markdown_table(fields, rows):
        return '\n'.join([
            make_markdown_row(fields),
            make_markdown_row('-' * len(fields)),
            '\n'.join([make_markdown_row(row) for row in rows]),
        ])

    if not q.client.working_file_path or not q.args.text_column:
        q.page["__meta_card__"].dialog = ui.dialog(
            title="Please go back and upload a Dataset and enter a text column!",
            items=[
                ui.progress(
                    label="Please go back and upload a Dataset!",
                    caption="",
                    value=None,
                    visible=True,
                    tooltip="",
                    name="",
                ),
            ],
        )
        q.client.cards.append("__meta_card__")
    else:
        q.page["metrics_content_card"] = ui.form_card(
            box="content",
            items=[
                ui.text(make_markdown_table(
                    fields=metricsdf.columns.tolist(),
                    rows=metricsdf.values.tolist(),
                )),
            ],
        )

        q.page["topics_content_card"] = ui.form_card(
            box="content",
            items=[
                ui.text(make_markdown_table(
                    fields=topicsdf.columns.tolist(),
                    rows=topicsdf.values.tolist(),
                )),
            ],
        )

        # Get the data from the file
        with open(PYLDA_HTML_FILE) as fp:
            topic_html = fp.read()

        q.page['visuals_content_card'] = ui.form_card(
            box=ui.box(zone="content", order=1),
            items=[
                ui.frame(content=topic_html, width='1500px', height='1000px')
            ])

        q.client.cards.append("metrics_content_card")
        q.client.cards.append("topics_content_card")
        q.client.cards.append("visuals_content_card")


def display_visualize(q: Q, DATA_SAVE_LOCATION, PYLDA_HTML_FILE):
    #not_implemented_dialog(q)
    q.client.theme = "light"
    main_layout(q)

    # Get the data from the file
    with open(PYLDA_HTML_FILE) as fp:
        topic_html = fp.read()

    q.page['visuals_content_card'] = ui.form_card(
        box=ui.box(zone="content", order=1),
        items=[
        ui.frame(content=topic_html, width='1500px', height='1000px')
        ])
    q.client.cards.append("visuals_content_card")



async def ui_loading_card(q: Q):
    q.page["__meta_card__"].dialog = ui.dialog(
        title="Fetching the Latest Data",
        items=[
            ui.progress(
                label="This should just take a couple of seconds",
                caption="",
                value=None,
                visible=True,
                tooltip="",
                name="",
            ),
        ],
    )
    await q.page.save()
    #q.page["__meta_card__"].dialog = None


def not_implemented_dialog(q: Q):
    q.page["__meta_card__"].dialog = ui.dialog(
        title="Feature Not Implemented!",
        items=[
            ui.text(
                "You clicked on a feature which is not yet implemented, but currently in the app to give "
                "a sense of what is coming!"
            ),
        ],
    )