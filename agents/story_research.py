import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, CASES_DIR, RESEARCH_MODEL

SYSTEM_PROMPT = """You are a research assistant for a true-crime short-form video series. \
You research real, well-documented cases and produce a structured, factual brief with \
sources, detailed enough that a scriptwriter can turn it into a multi-part (Part 1, 2, 3...) \
narration series without inventing anything. Only use facts you can support with a source — \
do not invent names, dates, quotes, or scene details.

Research process:
1. Use web_search to find several strong sources on the case (aim for a mix of an \
encyclopedic overview like Wikipedia and at least one detailed narrative account, e.g. a \
long-form timeline or true-crime feature article).
2. Use web_fetch to retrieve the full text of the 1-2 richest sources you found — do not \
rely on search snippets alone. Full articles are where specific, vivid, sourced details \
come from.
3. Write the brief from that full material.

Prefer cases that are:
- Closed (conviction, acquittal, or the case is otherwise legally resolved), or old enough \
that ongoing-investigation sensitivities don't apply.
- Already widely documented (Wikipedia, court records, established news coverage), so \
facts can be cross-checked.

Flag the case as sensitive (sensitivity.flag = true) if ANY of the following apply, and \
explain why in sensitivity.reason:
- The investigation or trial is still ongoing / unresolved.
- A living person is named as a suspect or perpetrator without a conviction.
- Victims were minors and the case involves graphic details needing extra care.
- The case is less than 5 years old.

After researching, respond with ONLY a single JSON object (no markdown fences, no other \
text) matching exactly this shape:

{
  "case_id": "short-slug-for-this-case",
  "title": "Human-readable case title",
  "summary": "2-4 sentence factual summary",
  "timeline": [
    {"date": "YYYY-MM-DD or approximate description", "event": "what happened"}
  ],
  "key_details": [
    {"detail": "a specific, vivid, sourced fact, quote, or scene description usable for dramatization", "source_url": "..."}
  ],
  "key_people": [
    {"name": "...", "role": "victim | survivor | suspect | investigator | witness | other"}
  ],
  "sources": [
    {"url": "...", "title": "..."}
  ],
  "sensitivity": {"flag": false, "reason": ""}
}

Requirements for depth:
- "key_people": list EVERY victim and survivor known by name in the sources -- not just the \
famous ones. Downstream, each named person gets a real-photo search so their photo can appear \
on screen when the narration names them; anyone you omit can never be shown. For a case with \
many victims this list is expected to be long (20+ entries is normal). Include investigators, \
key witnesses and defense/prosecution figures named in the sources too.
- "timeline": include not just the headline events but the intermediate steps too \
(investigative leads, near-misses, specific discoveries, court dates) — aim for 15-25 \
entries for a well-documented case. Only give fewer if the case genuinely has that little \
recorded detail.
- "key_details": 8-15 entries. Each one must be a concrete, specific, sourced detail a \
narrator could say on camera — a direct quote, a physical description, a specific number or \
object, a detail of place or atmosphere. Every entry needs a source_url pointing at a URL \
from the sources array. Do not paraphrase into vagueness — keep the specificity that makes \
it usable.

Write plain prose in every text field. Do not include <cite> tags, citation markers, or any \
other markup in the JSON output — citations belong only in the "sources" array.
"""

_CITE_TAG_RE = re.compile(r"</?cite[^>]*>")


def _case_dir(case_id: str):
    d = CASES_DIR / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_json(text: str) -> dict:
    text = _CITE_TAG_RE.sub("", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:500]!r}")
    return json.loads(match.group(0))


def run(case_id: str, db) -> None:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")

    case = db.get_case(case_id)
    topic = case["topic"] if case and case["topic"] else "any well-documented, closed true crime case"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = (
        f"Research a true crime case for the series. Topic/hint: {topic}\n\n"
        "Search the web to confirm facts, dates, and sources before writing the brief. "
        "Then output the JSON object described in your instructions."
    )

    messages = [{"role": "user", "content": user_prompt}]
    tools = [
        {"type": "web_search_20260209", "name": "web_search", "allowed_callers": ["direct"]},
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "allowed_callers": ["direct"],
            "max_content_tokens": 30000,
        },
    ]

    final_text = None
    for _ in range(8):
        response = client.messages.create(
            model=RESEARCH_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        db.log_usage(
            case_id, "story", RESEARCH_MODEL,
            response.usage.input_tokens, response.usage.output_tokens,
        )

        if response.stop_reason == "refusal":
            raise RuntimeError("Model refused the research request")
        if response.stop_reason == "pause_turn":
            continue

        text_blocks = [b.text for b in response.content if b.type == "text"]
        final_text = "\n".join(text_blocks)
        break
    else:
        raise RuntimeError("Research did not complete after multiple resumes (pause_turn)")

    if not final_text:
        raise RuntimeError("Model returned no text content")

    brief = _extract_json(final_text)

    out_path = _case_dir(case_id) / "brief.json"
    out_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")

    sensitivity = brief.get("sensitivity", {})
    flag = "review_required" if sensitivity.get("flag") else "clear"
    db.set_sensitivity_flag(case_id, flag)
    db.update_case_status(case_id, "story_done")

    print(f"  brief written: {out_path}")
    print(f"  sensitivity: {flag}" + (f" ({sensitivity.get('reason')})" if flag == "review_required" else ""))
