"""Compliance tools used by every version of the agent (v0, v1, v2).

A "tool" is just a plain Python function. The LLM never runs code: it only
asks us to call one of these functions with some arguments, and we send the
result back to it. Because the tools are plain functions, we can test them
without any LLM (see tests/test_tools.py).

Design rules for every tool:
1. Arguments are validated with pydantic before doing anything.
2. A tool NEVER raises an exception. On a problem it returns {"error": "..."}.
   The agent sends this error back to the LLM as an observation, so the LLM
   can fix its call and try again (self-correction).
3. Tools return plain dicts (easy to turn into JSON for the LLM).
4. Type hints use Literal for the allowed values: the MCP server (v2) turns
   them into the JSON schema it publishes. Python itself does not enforce them,
   so the pydantic validation inside each tool is still needed.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Knowledge base (loaded once, when the module is imported)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"
REPORTS_DIR = ROOT / "reports"

ARTICLES = json.loads((KNOWLEDGE_DIR / "articles.json").read_text())
PROHIBITED = json.loads((KNOWLEDGE_DIR / "prohibited_practices.json").read_text())
ANNEX_III = json.loads((KNOWLEDGE_DIR / "annex_iii.json").read_text())

RISK_CATEGORIES = ["prohibited", "high_risk", "limited_risk", "minimal_risk"]
RiskCategory = Literal["prohibited", "high_risk", "limited_risk", "minimal_risk"]

DISCLAIMER = (
    "Disclaimer: this is an automated first-pass assessment produced by an "
    "educational prototype. It is not legal advice."
)
MAX_QUESTIONS = 3  # max ask_user calls per assessment

# Citation format required in reports: [AI Act Art. 6(3)] or [GDPR Art. 22]
CITATION_PATTERN = re.compile(r"\[(AI Act|GDPR) Art\. (\d+)[^\]]*\]")
# Any loose mention like "Article 30" or "Art. 30", cited or not
ANY_ARTICLE_PATTERN = re.compile(r"\bArt(?:icle|\.)\s*(\d+)", re.IGNORECASE)

# Which articles to list for each risk category (texts come from articles.json)
OBLIGATIONS = {
    "prohibited": [("ai_act", "5"), ("ai_act", "99")],
    "high_risk": [
        ("ai_act", "9"), ("ai_act", "10"), ("ai_act", "11"), ("ai_act", "12"),
        ("ai_act", "13"), ("ai_act", "14"), ("ai_act", "15"), ("ai_act", "17"),
        ("ai_act", "43"), ("ai_act", "49"), ("ai_act", "26"), ("ai_act", "27"),
        ("ai_act", "86"), ("ai_act", "4"),
    ],
    "limited_risk": [("ai_act", "50"), ("ai_act", "4"), ("ai_act", "95")],
    "minimal_risk": [("ai_act", "4"), ("ai_act", "95")],
}
REGULATION_NAMES = {"ai_act": "AI Act", "gdpr": "GDPR"}


# ---------------------------------------------------------------------------
# Pydantic models: one per tool, they describe and validate the arguments.
# (v0 also turns them into the JSON schemas sent to the LLM.)
# ---------------------------------------------------------------------------
class CheckProhibitedArgs(BaseModel):
    description: str = Field(min_length=10, max_length=4000,
                             description="Description of the AI system")


class CheckHighRiskArgs(BaseModel):
    domain: str = Field(min_length=2, max_length=100,
                        description="Domain keyword, e.g. 'employment', 'credit', 'education', 'biometrics'")


class GetArticleArgs(BaseModel):
    regulation: Literal["ai_act", "gdpr"] = Field(description="'ai_act' or 'gdpr'")
    article: str = Field(description="Article number, e.g. '6' or '22'")

    @field_validator("article")
    @classmethod
    def keep_number_only(cls, value: str) -> str:
        # Refuse several articles at once ("5,6,9"): silently keeping one would mislead the LLM.
        if re.search(r"\d+\s*(,|;|\band\b)\s*\d+", value):
            raise ValueError("ask for ONE article per call, e.g. '22'")
        # Small models often send "Article 22" or "22(1)": keep only "22".
        match = re.search(r"\d+", value)
        if not match:
            raise ValueError("article must contain a number, e.g. '22'")
        return match.group()


class GetObligationsArgs(BaseModel):
    risk_category: RiskCategory = Field(description="One of: " + ", ".join(RISK_CATEGORIES))


class AskUserArgs(BaseModel):
    question: str = Field(min_length=5, max_length=500,
                          description="One short clarifying question for the user")


class SaveReportArgs(BaseModel):
    title: str = Field(min_length=3, max_length=120, description="Short report title")
    risk_category: RiskCategory = Field(description="One of: " + ", ".join(RISK_CATEGORIES))
    content_markdown: str = Field(min_length=50, description="Full report in Markdown")


def _validate(model: type[BaseModel], **kwargs):
    """Return (validated_args, None) or (None, error_dict)."""
    try:
        return model(**kwargs), None
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return None, {"error": f"Invalid arguments: {problems}"}


# ---------------------------------------------------------------------------
# Tool 1 — prohibited practices (Article 5)
# ---------------------------------------------------------------------------
def check_prohibited_practices(description: str) -> dict:
    """Return the Article 5 prohibited practices and their criteria.

    The tool gives the RULES; the LLM compares them with the description.
    """
    args, error = _validate(CheckProhibitedArgs, description=description)
    if error:
        return error
    return {
        "instructions": (
            "Compare the system description with each practice. A practice applies "
            "only if ALL its criteria are met and no exception applies."
        ),
        "practices": PROHIBITED["practices"],
    }


# ---------------------------------------------------------------------------
# Tool 2 — high-risk areas (Article 6 + Annex III)
# ---------------------------------------------------------------------------
def check_high_risk_areas(domain: str) -> dict:
    """Return the Annex III area(s) matching a domain keyword."""
    args, error = _validate(CheckHighRiskArgs, domain=domain)
    if error:
        return error
    query = args.domain.lower().strip().replace("_", " ")

    matches = []
    for area in ANNEX_III["areas"]:
        names = [area["domain"].replace("_", " ")] + area["keywords"]
        # whole-word match, so "hr" does not match inside "three"
        if any(re.search(rf"\b{re.escape(name)}\b", query) for name in names):
            matches.append(area)

    valid_domains = [area["domain"] for area in ANNEX_III["areas"]]
    if not matches:
        return {
            "matches": [],
            "message": f"No Annex III area matches '{args.domain}'. Valid domains: "
                       + ", ".join(valid_domains)
                       + ". If none fits the system, it is probably not high-risk under Annex III.",
            "valid_domains": valid_domains,
        }
    return {
        "matches": matches,
        "decision_rule": (
            "If the system does one of the use cases above, its category is high_risk "
            "(skip the transparency step). The Article 6(3) exception NEVER applies if the "
            "system profiles, ranks or scores people, or makes or replaces a decision about them."
        ),
        "article_6": ARTICLES["ai_act"]["6"]["summary"],
    }


# ---------------------------------------------------------------------------
# Tool 3 — one article summary
# ---------------------------------------------------------------------------
def get_article(regulation: Literal["ai_act", "gdpr"], article: str) -> dict:
    """Return the summary of one article of the AI Act or the GDPR."""
    args, error = _validate(GetArticleArgs, regulation=regulation, article=article)
    if error:
        return error
    articles = ARTICLES[args.regulation]
    if args.article not in articles:
        return {
            "error": f"Article {args.article} of {args.regulation} is not in the knowledge base. "
                     f"Available articles: {', '.join(articles)}"
        }
    return {
        "citation": f"[{REGULATION_NAMES[args.regulation]} Art. {args.article}]",
        **articles[args.article],
    }


# ---------------------------------------------------------------------------
# Tool 4 — obligations for a risk category
# ---------------------------------------------------------------------------
def get_obligations(risk_category: RiskCategory) -> dict:
    """Return the main obligations for a risk category, with article references."""
    args, error = _validate(GetObligationsArgs, risk_category=risk_category)
    if error:
        return error
    obligations = []
    for regulation, number in OBLIGATIONS[args.risk_category]:
        article = ARTICLES[regulation][number]
        obligations.append({
            "citation": f"[{REGULATION_NAMES[regulation]} Art. {number}]",
            "title": article["title"],
            "summary": article["summary"],
        })
    return {"risk_category": args.risk_category, "obligations": obligations}


# ---------------------------------------------------------------------------
# Tool 5 — ask the user a clarifying question
# ---------------------------------------------------------------------------
# Where answers come from. None = the terminal. The eval sets a function that
# answers from the case file; v1 sets a function that pauses the graph.
_answer_provider: Callable[[str], str] | None = None
_questions_asked = 0


def set_answer_provider(provider: Callable[[str], str] | None) -> None:
    global _answer_provider
    _answer_provider = provider


def reset_assessment() -> None:
    """Call at the start of each new assessment (resets the question counter)."""
    global _questions_asked
    _questions_asked = 0


def ask_user(question: str) -> dict:
    """Ask the human a clarifying question and return the answer."""
    global _questions_asked
    args, error = _validate(AskUserArgs, question=question)
    if error:
        return error
    if _questions_asked >= MAX_QUESTIONS:
        return {
            "error": f"Question limit reached ({MAX_QUESTIONS}). Do not ask more questions: "
                     "continue with the information you have and state your assumptions in the report."
        }
    if _answer_provider is None:
        answer = input(f"\n🤖 Agent asks: {args.question}\n👤 Your answer: ")
    else:
        answer = _answer_provider(args.question)
    # count AFTER getting the answer (matters for v1, where the call is replayed on resume)
    _questions_asked += 1
    return {"question": args.question, "answer": answer}


# ---------------------------------------------------------------------------
# Tool 6 — save the report (sensitive action)
# ---------------------------------------------------------------------------
def extract_citations(text: str) -> list[str]:
    """Return the citations found in a text, e.g. ['AI Act Art. 6', 'GDPR Art. 22']."""
    return sorted({f"{reg} Art. {num}" for reg, num in CITATION_PATTERN.findall(text)})


def validate_report(title: str, risk_category: str, content_markdown: str) -> dict | None:
    """Check a report before saving it. Return None if OK, else an error dict.

    The agents call this BEFORE asking the human for approval, so the human
    only reviews reports that already pass the automatic checks.
    """
    args, error = _validate(SaveReportArgs, title=title, risk_category=risk_category,
                            content_markdown=content_markdown)
    if error:
        return error
    content = args.content_markdown

    # The disclaimer must be at the TOP of the report (in the first lines), not hidden at the end.
    if "not legal advice" not in content[:400].lower():
        return {"error": f"The report must START with this exact disclaimer line: '{DISCLAIMER}'"}

    citations = CITATION_PATTERN.findall(content)
    if not citations:
        return {"error": "The report must cite at least one article, using the format "
                         "[AI Act Art. 6] or [GDPR Art. 22]."}

    # Guardrail against invented articles: every cited article must exist in the knowledge base.
    unknown = []
    for reg_name, number in citations:
        regulation = "ai_act" if reg_name == "AI Act" else "gdpr"
        if number not in ARTICLES[regulation]:
            unknown.append(f"{reg_name} Art. {number}")
    # Also check loose mentions like "Article 30" written without the bracket format.
    known_numbers = set(ARTICLES["ai_act"]) | set(ARTICLES["gdpr"])
    unknown += [f"Art. {n}" for n in ANY_ARTICLE_PATTERN.findall(content) if n not in known_numbers]
    if unknown:
        return {"error": f"The report cites articles that are not in the knowledge base: "
                         f"{', '.join(sorted(set(unknown)))}. Remove them or use get_article to check."}
    return None


def save_report(title: str, risk_category: RiskCategory, content_markdown: str) -> dict:
    """Write the report to reports/. Human approval is handled by the agent, not here."""
    error = validate_report(title, risk_category, content_markdown)
    if error:
        return error

    # Least privilege: build the filename ourselves, never use a path from the LLM.
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "report"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (REPORTS_DIR / f"{timestamp}_{risk_category}_{slug}.md").resolve()
    if path.parent != REPORTS_DIR.resolve():  # defence in depth: must stay in reports/
        return {"error": "Refused: the report path is outside the reports/ folder."}

    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(content_markdown)
    return {"status": "saved", "path": str(path.relative_to(ROOT))}


# Registry used by the agents: tool name -> (function, pydantic model of its arguments)
TOOLS = {
    "check_prohibited_practices": (check_prohibited_practices, CheckProhibitedArgs),
    "check_high_risk_areas": (check_high_risk_areas, CheckHighRiskArgs),
    "get_article": (get_article, GetArticleArgs),
    "get_obligations": (get_obligations, GetObligationsArgs),
    "ask_user": (ask_user, AskUserArgs),
    "save_report": (save_report, SaveReportArgs),
}
