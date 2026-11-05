"""current date tool : return the current date and time"""

from pydantic import BaseModel, ConfigDict
import datetime

from .base import Tool


CURRENT_DATE_DESCRIPTION = (
    "Return the current date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)."
)

class CurrentDateArgs(BaseModel):
    """No arguments are needed for this tool."""
    model_config = ConfigDict(extra="forbid")

def current_date(args: CurrentDateArgs) -> str:
    """Return the current date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)."""
    return datetime.datetime.now().isoformat()


current_date_tool: Tool[CurrentDateArgs] = Tool(
    name="current_date",
    description=CURRENT_DATE_DESCRIPTION,
    args_model=CurrentDateArgs,
    function=current_date,
)