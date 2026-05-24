"""LLM-driven narrative analysis of a student's course-plan dashboard.

Takes the structured dashboard payload (recommendations + skills + alt careers +
profile) and asks an LLM for a 4-section synthesis: headline / strengths / gaps /
strategy. The LLM is given only summaries (not raw PPR scores) so its output is
readable advice, not a number recap.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE_EN = """You are an academic + career advisor synthesizing a personalized course-recommendation result for a student.

The recommendations come from a Personalized PageRank algorithm over a heterogeneous graph (courses / skills / careers / students). You will NOT mention the algorithm — speak to the student in human terms.

## Student profile
- Major: {major}
- Year: {year_text}
- GPA: {gpa}
- Completed courses: {n_done}
- Currently enrolled: {n_current}

## Target career: {career}
(This career has {n_postings} sampled JD postings.)

## Top {n_recs} recommended courses for next semester
{recs_block}

## Skills the student already has (top from completed courses)
{student_skills}

## Skills this career values most
{career_skills}

## Alternative careers (most similar skill profile)
{alt_careers}

---

Write a synthesis in **{language_name}** with EXACTLY these four parts, returned as a JSON object:

{{
  "headline": "<one decisive sentence capturing the key insight, ~15 words max>",
  "strengths": "<2-3 sentences identifying what the student already has going for them>",
  "gaps": "<2-3 sentences naming the most important gaps and why they matter>",
  "strategy": "<1-2 sentences of concrete actionable next-step advice (e.g., which 2-3 courses to prioritize, or whether they should consider an alternative career)>"
}}

Rules:
- Be SPECIFIC: name actual skills, course codes, or career titles from the data above.
- Be DIRECT and CONFIDENT: no hedging, no marketing tone, no "we suggest" filler.
- DO NOT mention PageRank, PPR, graphs, or algorithms.
- DO NOT use em-dashes or "—" — use periods or commas.
- Return ONLY the JSON object. No markdown fences, no preamble.
"""

PROMPT_TEMPLATE_ZH = """你是一位为学生分析个性化选课推荐结果的学业/职业顾问。

推荐结果来自异构图上的 Personalized PageRank 算法（课程/技能/职业/学生节点）。**不要**向学生提及算法名称，用人话讲。

## 学生画像
- 专业: {major}
- 年级: {year_text}
- GPA: {gpa}
- 已修课程数: {n_done}
- 本学期在修: {n_current}

## 目标职业: {career}
（该职业在数据集中有 {n_postings} 个 JD 样本）

## 下学期推荐的 {n_recs} 门课
{recs_block}

## 学生已具备的核心技能（来自已修课程）
{student_skills}

## 该职业最看重的技能
{career_skills}

## 相似职业（技能向量接近）
{alt_careers}

---

用**{language_name}**写一份四段式分析，以 JSON 对象返回：

{{
  "headline": "<一句话点出关键洞察，约 15 字>",
  "strengths": "<2-3 句指出学生已经具备的优势>",
  "gaps": "<2-3 句指出最关键的缺口及其重要性>",
  "strategy": "<1-2 句具体可执行的下一步建议（例如优先选哪 2-3 门、或考虑某个备选职业）>"
}}

规则:
- 要**具体**: 用上面数据里的真实技能名、课程代码、职业名称。
- 要**直接**: 不绕弯子、不营销腔、不用"我们建议"之类的填词。
- **不要**提 PageRank、PPR、图算法。
- **不要**用破折号（——）。
- 仅返回 JSON 对象。无 markdown 包裹，无前言。
"""


def _format_recs(recs: list[dict]) -> str:
    lines = []
    for i, r in enumerate(recs, 1):
        bridges = [b["skill"] for b in (r.get("bridge_skills") or [])[:4]]
        gaps = r.get("gap_skills") or []
        gap_str = f" gaps: {', '.join(gaps[:3])}" if gaps else ""
        bridge_str = f" covers: {', '.join(bridges)}" if bridges else ""
        lines.append(f"{i}. {r['code']} {r.get('name','')}{bridge_str}{gap_str}")
    return "\n".join(lines)


def _format_skills(skills: list[dict], key_for_value: str = "weight") -> str:
    return ", ".join(f"{s['skill']} ({s[key_for_value]})" for s in skills[:10])


def _format_alt_careers(alts: list[dict]) -> str:
    if not alts:
        return "(none)"
    return "\n".join(
        f"- {c['title']} (similarity {c['similarity']}, {c['shared_skill_count']} shared skills)"
        for c in alts[:5]
    )


def build_insight_prompt(
    profile: Optional[dict],
    career: str,
    n_postings: int,
    recommendations: list[dict],
    student_skills: list[dict],
    career_skills: list[dict],
    alt_careers: list[dict],
    language: str = "en",
) -> str:
    """Compose the LLM prompt from a dashboard-shaped payload."""
    profile = profile or {}
    year = profile.get("year")
    if language == "zh":
        if year and 1 <= year <= 4:
            year_text = f"大{year}"
        elif year == 5:
            year_text = "研究生"
        else:
            year_text = "未知"
        language_name = "中文"
        template = PROMPT_TEMPLATE_ZH
    else:
        if year and 1 <= year <= 4:
            year_text = f"Year {year}"
        elif year == 5:
            year_text = "Graduate"
        else:
            year_text = "Unknown"
        language_name = "English"
        template = PROMPT_TEMPLATE_EN

    return template.format(
        major=profile.get("major") or ("未填" if language == "zh" else "Unknown"),
        year_text=year_text,
        gpa=profile.get("gpa") if profile.get("gpa") is not None else "—",
        n_done=len(profile.get("completed_courses") or []),
        n_current=len(profile.get("current_courses") or []),
        career=career,
        n_postings=n_postings,
        n_recs=len(recommendations),
        recs_block=_format_recs(recommendations),
        student_skills=_format_skills(student_skills) or "(none)",
        career_skills=_format_skills(career_skills) or "(none)",
        alt_careers=_format_alt_careers(alt_careers),
        language_name=language_name,
    )


def parse_insight_response(raw: str) -> dict:
    """Locate and parse the first JSON object in the LLM response.

    Returns {headline, strengths, gaps, strategy} (all strings, blank if missing).
    Returns all-empty dict on parse failure — caller can detect via empty values.
    """
    empty = {"headline": "", "strengths": "", "gaps": "", "strategy": ""}
    if not raw:
        return empty
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return empty
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return empty
    return {
        "headline": str(parsed.get("headline") or "").strip(),
        "strengths": str(parsed.get("strengths") or "").strip(),
        "gaps": str(parsed.get("gaps") or "").strip(),
        "strategy": str(parsed.get("strategy") or "").strip(),
    }


def generate_insight(
    client,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 800,
) -> dict:
    """Send prompt to LLM and parse structured insight. Returns all-empty on failure."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("insight LLM call failed: %s", exc)
        return {"headline": "", "strengths": "", "gaps": "", "strategy": ""}
    raw = ""
    if response.choices and response.choices[0].message:
        raw = response.choices[0].message.content or ""
    return parse_insight_response(raw)
