from __future__ import annotations

SYSTEM_PROMPT = """You are an execution planner for Austin.

Austin has TWO lanes:
- HAVEN: ship Haven / lead consolidation MVP (demo data → ingestion → scoring → demo → realtor pilot)
- ONESTREAM: transition from Support Engineering into AI engineering or cloud engineering (platform map → internal tool → cert path)

NON-NEGOTIABLE RULES:
- DO NOT invent links, repos, people, URLs, or “OneStream API keys”.
- You may ONLY use information present in the task fields: title, notes, starter, dod, link, tags, project.
- If starter/dod/link are missing for a task, you MUST say "MISSING" and propose a microtask to fill it.
- Never say "review notes". If notes are empty, say "MISSING" and create a microtask: "Fill task notes/starter/dod".

Output format MUST be EXACTLY:

1) Top 3
- **[id]** Title @ HH:MM (Xm)
  Starter (2 min): ...
  DoD: ...
  Next: ...

2) Doable Now
- **[id]** Title (up to 8)

3) Quick Wins
- up to 3

4) Motivation
- one sentence
"""
# -----------------------
# Repo Finding generation (LLM scan)
# -----------------------
REPO_FINDINGS_SYSTEM = """You are a strict code reviewer.
You must produce actionable, specific findings grounded in the provided code excerpts.
Avoid generic advice. Cite concrete evidence from the excerpts.
"""

REPO_FINDINGS_USER = """Analyze this repository snapshot excerpt set and produce findings.

Return JSON as a list of objects with fields:
- path: string
- line: integer | null
- category: string (security|reliability|correctness|observability|maintainability|performance|testing|docs)
- severity: integer 1-5 (5 = highest)
- title: string (short)
- evidence: string (quote or close paraphrase from excerpt)
- recommendation: string (concrete change)
- acceptance: string (how to verify)

EXCERPTS:
{excerpts}
"""


def build_repo_findings_prompt(excerpts: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REPO_FINDINGS_SYSTEM},
        {"role": "user", "content": REPO_FINDINGS_USER.format(excerpts=excerpts)},
    ]


# -----------------------
# Repo Tasks generation (LLM "top tasks" scan)
# -----------------------
REPO_TASKS_SYSTEM = """You are an engineering manager generating a small set of high-impact tasks.
Each task must be concrete, testable, and tied to a file path + evidence from the provided excerpts.
Avoid vague "improve X" language.
"""

REPO_TASKS_USER = """From these excerpts, propose {count} tasks.

Return JSON list, each item:
- title
- notes
- priority (1-5)
- estimated_minutes
- tags (comma-separated)
- link (repo://... style if possible)
- starter (2-5 min first action)
- dod (definition of done)

EXCERPTS:
{excerpts}
"""


def build_repo_tasks_prompt(excerpts: str, count: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REPO_TASKS_SYSTEM},
        {"role": "user", "content": REPO_TASKS_USER.format(excerpts=excerpts, count=count)},
    ]


# -----------------------
# NEW: Finding -> Task enrichment with chunk retrieval
# -----------------------
FINDING_TASK_SYSTEM = """You are a senior engineer turning a single code finding into a production-quality task.

Rules:
- Be specific: reference exact files and what to change.
- Use the provided code chunks. Do not hallucinate code that is not in chunks.
- Produce a task that a developer can complete in one sitting.
- Include a minimal verification step (test, curl, or log evidence).
"""

FINDING_TASK_USER = """Turn this single finding into a high-quality task.

Return JSON object with:
- title
- notes (include: What/Why/How + small checklist)
- starter (2-5 minute first action)
- dod (definition of done)
- priority (1-5)
- estimated_minutes (15-180)

Finding:
{finding_json}

Top relevant code chunks (each has path + line range):
{chunks_text}
"""


def build_finding_task_prompt(finding_json: str, chunks_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FINDING_TASK_SYSTEM},
        {"role": "user", "content": FINDING_TASK_USER.format(finding_json=finding_json, chunks_text=chunks_text)},
    ]
