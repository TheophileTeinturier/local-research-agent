"""arXiv tools: `search_arxiv` and `get_paper_details`, backed by `ArxivClient`.

Modules, in reading order: models (capped paper records), errors (typed failures), query
(input validation and arXiv query building, pure), parsing (Atom -> models, pure), client
(HTTP, rate limit, cache), tools (Tool definitions, output formatting, error messages),
testing (offline replay transport).
"""

from .client import ArxivClient
from .errors import ArxivError
from .models import PaperDetails, PaperSummary
from .tools import make_arxiv_tools

__all__ = ["ArxivClient", "ArxivError", "PaperDetails", "PaperSummary", "make_arxiv_tools"]
