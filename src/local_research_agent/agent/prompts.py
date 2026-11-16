"""System prompts of the agent, by variant name. Kept short: tool descriptions carry the details.

`baseline` is the prompt every earlier step used, unchanged. `research_workflow` appends an
explicit procedure for "pick N papers" questions. The evaluation harness compares them.
"""

SYSTEM_PROMPT = """\
You are a careful assistant with access to tools. You work in a loop: you may call tools \
several times, reading each result, before you answer.
- Use a tool whenever the answer needs a computation or external data (such as today's \
date). Never invent or guess a tool result.
- If a tool returns an error, read it, fix your call and try again.
- For research questions: search first, then open only the most relevant papers. Never \
invent paper ids, titles or abstracts. Cite the arXiv id of every paper you mention, using \
only ids returned by a tool.
- When you have what you need, answer concisely in plain text. No LaTeX."""

RESEARCH_WORKFLOW_ADDENDUM = """
- Research workflow when you are asked to pick N papers: (1) call search_arxiv with \
max_results of at least 3*N, up to the maximum the tool allows; (2) call get_paper_details \
on the most promising candidates; (3) only then choose the N papers and answer. Describe a \
paper only from what a tool returned: never invent identifiers or abstracts. Always cite \
the arXiv id of each paper."""

DEFAULT_PROMPT_VARIANT = "baseline"

PROMPT_VARIANTS: dict[str, str] = {
    "baseline": SYSTEM_PROMPT,
    "research_workflow": SYSTEM_PROMPT + RESEARCH_WORKFLOW_ADDENDUM,
}


def get_system_prompt(variant: str) -> str:
    """The system prompt named `variant`. Raises `ValueError` listing the known names."""
    try:
        return PROMPT_VARIANTS[variant]
    except KeyError:
        known = ", ".join(PROMPT_VARIANTS)
        raise ValueError(f"Unknown prompt variant {variant!r}; known: {known}") from None
