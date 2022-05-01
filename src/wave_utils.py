import os
import sys
import traceback

import jwt
import pandas as pd
from h2o_wave import Q, ui
from h2o_wave.core import expando_to_dict


def refresh_app(q: Q):
    q.page.drop()
    q.page["__meta_card__"] = ui.meta_card(
        box="",
        title=q.client.app_name,
        theme=q.client.theme,
        layouts=[
            ui.layout(breakpoint="xs", max_width="1250px", zones=[ui.zone("body")])
        ],
    )


async def ui_crash_card(q: Q, app_name, card_name, box):
    refresh_app(q)

    error_msg_items = [
        ui.text_xl("Error!"),
        ui.text_l(
            "Apologies for the inconvenience. "
            f"Please refresh your browser to restart {app_name}. "
        ),
    ]
    error_report_items = [
        ui.text(
            "To report this crash, please send an email to [cloud-feedback@h2o.ai](cloud-feedback@h2o.ai) "
            "with the following information:"
        ),
        ui.separator("Crash Report"),
        ui.text_l(app_name),
    ]

    q_args = [f"{k}: {v}" for k, v in expando_to_dict(q.args).items()]
    q_args_str = "#### q.args\n```\n" + "\n".join(q_args) + "\n```"
    q_args_items = [ui.text_m(q_args_str)]

    q_app = [f"{k}: {v}" for k, v in expando_to_dict(q.app).items()]
    q_app_str = "#### q.app\n```\n" + "\n".join(q_app) + "\n```"
    q_app_items = [ui.text_m(q_app_str)]

    q_user = [f"{k}: {v}" for k, v in expando_to_dict(q.user).items()]
    q_user_str = "#### q.user\n```\n" + "\n".join(q_user) + "\n```"
    q_user_items = [ui.text_m(q_user_str)]

    q_client = [f"{k}: {v}" for k, v in expando_to_dict(q.client).items()]
    q_client_str = "#### q.client\n```\n" + "\n".join(q_client) + "\n```"
    q_client_items = [ui.text_m(q_client_str)]

    type_, value_, traceback_ = sys.exc_info()
    stack_trace = traceback.format_exception(type_, value_, traceback_)
    stack_trace_items = [ui.text("**Stack Trace**")] + [
        ui.text(f"`{x}`") for x in stack_trace
    ]

    error_report_items.extend(
        q_args_items + q_app_items + q_user_items + q_client_items + stack_trace_items
    )

    q.page[card_name] = ui.form_card(
        box=box,
        items=error_msg_items
        + [
            ui.expander(
                name="error_report",
                label="Report this error",
                expanded=False,
                items=error_report_items,
            )
        ],
    )

    await q.page.save()


def ui_table_from_df(df: pd.DataFrame, rows: int, name: str):
    """
    Convert a dataframe into wave ui table format.
    """
    df = df.copy().reset_index(drop=True)
    columns = [ui.table_column(name=str(x), label=str(x)) for x in df.columns.values]
    rows = min(rows, df.shape[0])

    try:
        table = ui.table(
            name=name,
            columns=columns,
            rows=[
                ui.table_row(
                    name=str(i),
                    cells=[str(df[col].values[i]) for col in df.columns.values],
                )
                for i in range(rows)
            ],
        )
    except Exception:
        table = ui.table(
            name=name,
            columns=[ui.table_column("0", "0")],
            rows=[ui.table_row(name="0", cells=[str("No data found")])],
        )

    return table


async def handle_crash(q: Q):

    for c in q.client.cards:
        del q.page[c]

    await ui_crash_card(
        q, app_name=q.client.app_name, card_name="__crash_card__", box="body"
    )

    q.client.cards = ["__crash_card__"]
