"""Prompts for LLM-based skill extraction."""

EXTRACTION_PROMPT = """Extract concrete technical skills from the following text.

Rules:
- Include: programming languages, frameworks, tools, methodologies, technical concepts (e.g., "Python", "Machine Learning", "SQL", "TensorFlow", "Statistical Analysis").
- Exclude: soft skills (e.g., "communication", "teamwork"), job titles, company names, locations, vague phrases.
- Use canonical names (e.g., "Python" not "python programming"; "Machine Learning" not "ML").
- Limit: maximum 15 skills per text.

Text:
\"\"\"
{text}
\"\"\"

Return ONLY a valid JSON array of skill strings. No explanation, no markdown fences.
Example output: ["Python", "SQL", "Machine Learning", "Data Visualization"]
"""
