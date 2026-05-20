from openai import AzureOpenAI
import json
import re
import os

_client = AzureOpenAI(
    api_version= os.getenv('API_VERSION'),
    azure_endpoint= os.getenv('ENDPOINT'),#<endpoint>,
    api_key= os.getenv('API_KEY')#<api_key>,
)

SYSTEM_PROMPT = """You are a strict but fair technical recruiter.
You are given a JOB DESCRIPTION and excerpts from a candidate's RESUME.
Compare them and return a JSON object with these exact keys:
{
  "score": <integer 0-5, how well the resume matches the job>,
  "verdict": <"Strong Match" | "Good Match" | "Partial Match" | "Weak Match" | "No Match">,
  "matched_strengths": [<short bullets of skills/experience present in resume that match the JD>],
  "gaps": [<short bullets of JD requirements missing or weak in the resume>],
  "summary": <one short paragraph explaining the score>
}
Rules:
- Base your judgment ONLY on the provided resume excerpts. Do not invent experience.
- Be specific: cite skills, years, technologies, or domains.
- Return ONLY the JSON object. No prose before or after."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {
        "score": None,
        "verdict": "Unknown",
        "matched_strengths": [],
        "gaps": [],
        "summary": text.strip(),
    }


def score_resume_against_job(job_description: str, resume_context: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"JOB DESCRIPTION:\n{job_description}\n\n"
                f"RESUME EXCERPTS:\n{resume_context}"
            ),
        },
    ]

    response = _client.chat.completions.create(
        model="gpt-5-chat",
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )

    content = response.choices[0].message.content
    return _extract_json(content)
