"""Current date tool: return the local date and time."""

import datetime

from pydantic import BaseModel, ConfigDict

from .base import Tool

CURRENT_DATE_DESCRIPTION = (
    "Return the current local date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). "
    "Use it whenever the answer depends on today's date."
)


class CurrentDateArgs(BaseModel):
    """No arguments are needed for this tool."""

    model_config = ConfigDict(extra="forbid")


def current_date(args: CurrentDateArgs) -> str:
    """Return the current local date and time, to the second (matches the description)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


current_date_tool: Tool[CurrentDateArgs] = Tool(
    name="current_date",
    description=CURRENT_DATE_DESCRIPTION,
    args_model=CurrentDateArgs,
    function=current_date,
)
