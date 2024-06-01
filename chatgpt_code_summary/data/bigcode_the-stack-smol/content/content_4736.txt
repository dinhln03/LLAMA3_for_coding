"""Various functions that interact with Slack, e.g. posting messages."""
import asyncio
import logging
import socket
from pathlib import Path
from typing import Union, Optional

from slack_sdk.errors import SlackApiError

from lsw_slackbot.plots import plot_resource_use
from lsw_slackbot.resources import current_memory_fraction, _get_resource_usage_dataframe
from lsw_slackbot.util import string_time


async def _send_message(client, channel: str, message: str):
    """Sends a message to a channel, with basic logging & error handling."""

    try:
        await client.chat_postMessage(channel=channel, text=message)

    # Handle various different errors, *some* of which are non-critical...
    except SlackApiError as e:
        logging.exception(f"error from slack API when trying to send message: {e.response['error']}")
        print("Encountered SlackApiError when trying to send message (see logs.)")

    except AttributeError:
        logging.exception("suspected issue in Slack API when trying to send message. This bug has occured before!")
        print("Encountered AttributeError when trying to send message (see logs.)")


async def _send_file(client, channel: str, file: Union[Path, str], title):
    """Sends a file to a channel, with basic logging & error handling."""

    if isinstance(file, Path):
        file = str(file.absolute())

    try:
        await client.files_upload(channels=channel, file=file, title=title)

    # Handle various different errors, *some* of which are non-critical...
    except SlackApiError as e:
        logging.exception(f"error from Slack API when trying to upload file: {e.response['error']}")
        print("Encountered SlackApiError when trying to upload file (see logs.)")

    except AttributeError:
        logging.exception("suspected issue in Slack API when trying to upload file. This bug has occured before!")
        print("Encountered AttributeError when trying to upload file (see logs.)")


async def hello_world(client, channel: str):
    """Basic function to post an init message to a channel."""
    # Todo: it would be really cool if hello_world also printed the latest commit message.
    #   This could be done by running the command `git log -1` from Python?
    #   See https://stackoverflow.com/questions/7293008/display-last-git-commit-comment
    logging.info(f"Saying hello world in {channel}!")
    system_name = socket.gethostname()
    await _send_message(
        client, channel, f"Server time & date: {string_time()}\nApp is running on system {system_name}.")


async def send_resource_use_plot(client, channel: str, plot_kwargs: dict, title: Optional[str] = None):
    """Sends a resource usage plot to a given channel."""

    if title is None:
        title = f"Resource usage plot generated at {string_time()}"
    else:
        title = title + f" (plot generated at {string_time()})"

    # Firstly, let's generate a plot
    logging.info("Generating a resource usage plot")
    logging.debug(f"plot kwargs: {plot_kwargs}")
    location_plot = await plot_resource_use(**plot_kwargs)

    # Now, let's try and send it to slack
    logging.info(f"Sending to Slack in channel {channel}")
    await _send_file(client, channel, location_plot, title)


_LAST_MEMORY_FRACTION = 0.0


async def check_memory(client, channel: str, memory_warn_fraction=0.8, sleep_time=3600):
    """Quick function for checking current server memory and sending a warning to a desired channel if it's
    too high."""
    global _LAST_MEMORY_FRACTION  # Sorry for using global variables =(

    current_usage = current_memory_fraction()

    # Only warn if we didn't warn before
    if _LAST_MEMORY_FRACTION < memory_warn_fraction:

        if current_usage > memory_warn_fraction:
            # Firstly, prioritise sending a basic warning
            await _send_message(client, channel, f"WARNING: current memory usage at {current_usage:.2%}!")

            # Next, grab info on currently running threads
            thread_df = await _get_resource_usage_dataframe(measurement_time=1.0)
            thread_df = thread_df.sort_values("memory")

            # ... and format it into something we can send
            message = ["Users with something currently running:"]

            for i, a_row in thread_df.iterrows():
                message.append(f"{a_row.name}: {a_row['cpu_percent']:.2f}% CPU "
                               f"-- {a_row['memory']:.2f} GB"
                               f"-- {a_row['threads']} threads")

            message.append(f"\n(no further warnings will be sent for a sleep period of {sleep_time/60**2:.2f} hour(s))")

            # Send it!
            await _send_message(client, channel, "\n".join(message))

            # Sleep so we don't spam the chat
            await asyncio.sleep(sleep_time)

    _LAST_MEMORY_FRACTION = current_usage
