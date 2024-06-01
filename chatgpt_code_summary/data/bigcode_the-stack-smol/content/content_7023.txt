import functools
from teamiclink.slack.model import GoalContent
from typing import Any, Dict

from slack_bolt import Ack
from slack_bolt.context import BoltContext
from pydantic import ValidationError

CREATE_GOAL_CALLBACK_ID = "create_goal_view_id"
CREATE_GOAL_INPUT = "create_goal_action"
CREATE_GOAL_INPUT_BLOCK = "create_goal_input_block"
CREATE_GOAL = {
    "type": "modal",
    "callback_id": CREATE_GOAL_CALLBACK_ID,
    "title": {"type": "plain_text", "text": "Teamiclink"},
    "submit": {"type": "plain_text", "text": "Submit"},
    "close": {"type": "plain_text", "text": "Cancel"},
    "blocks": [
        {
            "type": "input",
            "block_id": CREATE_GOAL_INPUT_BLOCK,
            "element": {"type": "plain_text_input", "action_id": CREATE_GOAL_INPUT},
            "label": {"type": "plain_text", "text": "Create goal"},
        }
    ],
}


def add_goal_to_payload(func):
    """Adds a goal to payload for 't-goal' key."""

    @functools.wraps(func)
    def wrapper_inject_goal(ack: Ack, payload: Dict[str, Any], context: BoltContext):
        try:
            content = GoalContent(
                content=payload["state"]["values"][CREATE_GOAL_INPUT_BLOCK][
                    CREATE_GOAL_INPUT
                ]["value"]
            ).content
        except ValidationError as error:
            return ack(
                response_action="errors",
                errors={CREATE_GOAL_INPUT_BLOCK: error.errors()[0]["msg"]},
            )
        payload["t-goal"] = content
        return func(ack=ack, payload=payload, context=context)

    return wrapper_inject_goal
