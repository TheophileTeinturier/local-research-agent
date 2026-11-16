"""Current date tool: return the local date and time.

The time source is injectable (`make_current_date_tool(clock)`), so the evaluation harness
can freeze "now". The module-level `current_date_tool` reads the real local clock.
"""

import datetime
from collections.abc import Callable
from functools import partial

from pydantic import BaseModel, ConfigDict

from .base import Tool

# Returns the current local date and time. Not the loop's monotonic clock (seconds).
DateClock = Callable[[], datetime.datetime]

CURRENT_DATE_DESCRIPTION = (
    "Return the current local date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). "
    "Use it whenever the answer depends on today's date."
)


class CurrentDateArgs(BaseModel):
    """No arguments are needed for this tool."""

    model_config = ConfigDict(extra="forbid")


def current_date(args: CurrentDateArgs, clock: DateClock = datetime.datetime.now) -> str:
    """Return `clock()` to the second (matches the description)."""
    return clock().isoformat(timespec="seconds")


def make_current_date_tool(clock: DateClock = datetime.datetime.now) -> Tool[CurrentDateArgs]:
    """The `current_date` tool reading `clock` (default: the real local time)."""
    return Tool(
        name="current_date",
        description=CURRENT_DATE_DESCRIPTION,
        args_model=CurrentDateArgs,
        function=partial(current_date, clock=clock),
    )


current_date_tool: Tool[CurrentDateArgs] = make_current_date_tool()
