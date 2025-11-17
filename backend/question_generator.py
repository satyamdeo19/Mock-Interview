import requests
import json
import logging
from typing import List, Dict, Any
import re
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class QuestionGenerator:
    def __init__(self):
        # ✅ Gemini 2.5 endpoints
        self.gemini_api_urls = [
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        ]
        self.gemini_api_key = GEMINI_API_KEY

    # -------------------------------------------------------------------------
    # TEMPLATE / FALLBACK GENERATION
    # -------------------------------------------------------------------------
    def generate_questions_free(self, resume_data: Dict[str, Any], mode: str = "technical") -> List[Dict[str, str]]:
        print("⚙️  Using TEMPLATE generator (Gemini unavailable or failed)")
        try:
            skills = resume_data.get("skills", [])
            experience = resume_data.get("experience", [])
            education = resume_data.get("education", [])
            projects = resume_data.get("projects", [])

            if mode == "technical":
                return self._generate_technical_questions(skills, experience, projects)
            return self._generate_hr_questions(resume_data)
        except Exception:
            logger.exception("Error in template generation")
            return self._get_fallback_questions(mode)

    def _generate_technical_questions(self, skills, experience, projects):
        questions = []
        if skills:
            for skill in skills[:3]:
                s = skill.lower()
                if any(x in s for x in ["python", "java", "javascript", "c++", "react", "node"]):
                    questions.append({
                        "question": f"You mentioned {skill}. Can you explain a challenging problem you solved using it?",
                        "ideal_answer": f"Should describe a project, approach, and successful outcome using {skill}."
                    })
                elif any(x in s for x in ["sql", "mongodb", "database"]):
                    questions.append({
                        "question": f"Describe your experience with {skill}. How would you optimize a slow query?",
                        "ideal_answer": "Should discuss indexing, restructuring, and performance monitoring."
                    })
        if projects:
            p = projects[0][:100] + "..." if len(projects[0]) > 100 else projects[0]
            questions.append({
                "question": f"You worked on '{p}'. What was the most technically challenging part?",
                "ideal_answer": "Should discuss the challenge, solution, tools used, and results."
            })
        while len(questions) < 5:
            questions.append({
                "question": "Describe your approach to debugging a complex issue.",
                "ideal_answer": "Should mention structured debugging, logging, and prevention."
            })
        return questions[:5]

    def _generate_hr_questions(self, resume_data):
        name = resume_data.get("name", "candidate")
        skills = resume_data.get("skills", [])
        education = resume_data.get("education", [])
        projects = resume_data.get("projects", [])
        questions = [
            {
                "question": f"Hi {name.split()[0] if name else 'there'}! Tell me about yourself and what drew you to this field.",
                "ideal_answer": "Should summarize experience, motivation, and career goals."
            }
        ]
        if education:
            edu = education[0]
            questions.append({
                "question": f"I see you studied {edu}. How has it prepared you for this role?",
                "ideal_answer": f"Should connect {edu} to real-world applications and skills."
            })
        if skills:
            questions.append({
                "question": f"You have experience with {', '.join(skills[:3])}. Which would you like to deepen and why?",
                "ideal_answer": "Should show self-awareness and growth mindset."
            })
        if projects:
            questions.append({
                "question": "Describe a project you’re proud of. What challenges did you face and how did you overcome them?",
                "ideal_answer": "Should show ownership and problem-solving using STAR format."
            })
        while len(questions) < 5:
            questions.append({
                "question": "Where do you see yourself in 3–5 years?",
                "ideal_answer": "Should show ambition aligned with company’s goals."
            })
        return questions[:5]

    # -------------------------------------------------------------------------
    # GEMINI GENERATION (Primary)
    # -------------------------------------------------------------------------
    def generate_questions_gemini(self, resume_data, mode="technical"):
        print("🚀 Attempting to use GEMINI API for question generation...")
        try:
            print("🔑 GEMINI_API_KEY =", "SET ✅" if self.gemini_api_key else "❌ NOT SET")
            if not self.gemini_api_key or self.gemini_api_key.strip() in ["", "YOUR_GEMINI_API_KEY"]:
                return self.generate_questions_free(resume_data, mode)

            context = self._prepare_resume_context(resume_data)
            if len(context) > 800:  # 🧠 Keep context compact
                context = context[:800] + "..."

            role = resume_data.get("target_role", "software engineer")

            # 🧠 Tight, JSON-only prompt
            base_prompt = f"""
You are an expert interviewer preparing for a {role} role.

Candidate Resume:
{context}

Generate exactly 5 personalized {'technical' if mode == 'technical' else 'HR/behavioral'} interview questions
based on the candidate's skills, projects, and experience.

Each question should test deep understanding or reasoning (not definitions).

Return ONLY a valid JSON array of exactly 5 objects with fields:
"question" and "ideal_answer"

Example:
[
  {{"question": "Explain OOP principles in Python.", "ideal_answer": "Encapsulation, inheritance, etc."}}
]

No explanations or text outside the JSON.
"""

            # ✅ Compact generation config to avoid overthinking
            payload = {
                "contents": [{"parts": [{"text": base_prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 512
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            print("🌐 Trying available Gemini endpoints...")
            res = None
            for api_url in self.gemini_api_urls:
                print(f"➡️  Trying model endpoint: {api_url}")
                try:
                    res = requests.post(
                        f"{api_url}?key={self.gemini_api_key}",
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=40,
                    )
                    print(f"📩 Response Status: {res.status_code}")
                    if res.status_code == 200:
                        print(f"✅ Model {api_url.split('/')[-1].split(':')[0]} responded successfully!")
                        self.gemini_api_url = api_url
                        break
                    else:
                        print(f"⚠️ Model returned {res.status_code}: {res.text[:300]}")
                except Exception as e:
                    print(f"💥 Error calling {api_url}: {str(e)}")

            if not res or res.status_code != 200:
                return self.generate_questions_free(resume_data, mode)

            data = res.json()
            print("🧾 Gemini full response snippet:", json.dumps(data, indent=2)[:800])

            # Retry with Pro if Flash hits token limit
            finish_reason = data.get("candidates", [{}])[0].get("finishReason", "")
            if finish_reason == "MAX_TOKENS" and "flash" in self.gemini_api_url:
                print("⚠️ Flash hit MAX_TOKENS — retrying with Gemini 2.5-Pro …")
                pro_url = self.gemini_api_urls[1]
                res = requests.post(
                    f"{pro_url}?key={self.gemini_api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=40,
                )
                data = res.json()

            # ✅ Extract text safely
            text = ""
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts and isinstance(parts[0], dict):
                        text = parts[0].get("text", "")
                if not text:
                    print("⚠️ No text found, fallback to raw JSON dump")
                    text = json.dumps(data)
            except Exception as e:
                print("⚠️ Parsing error:", str(e))

            text = re.sub(r"```json|```", "", text)
            print("🧠 Gemini raw text snippet:", text[:300], "...")

            parsed = self._parse_questions_from_text(text)
            if not parsed or len(parsed) < 3:
                print("⚠️ Gemini output insufficient → fallback to TEMPLATE.")
                return self.generate_questions_free(resume_data, mode)

            print(f"✅ Gemini successfully generated {len(parsed)} personalized questions!\n")
            return parsed

        except Exception as e:
            print("💥 Unexpected Gemini error:", str(e))
            return self.generate_questions_free(resume_data, mode)

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _prepare_resume_context(self, resume_data):
        parts = []
        name = resume_data.get("name", "The candidate")
        skills = resume_data.get("skills", [])
        exp = resume_data.get("experience", [])
        edu = resume_data.get("education", [])
        projects = resume_data.get("projects", [])
        parts.append(f"{name} is skilled in {', '.join(skills[:10])}.")
        if exp:
            parts.append(f"They have experience in {'. '.join(exp[:2])}.")
        if projects:
            parts.append(f"Notable projects include {', '.join(projects[:2])}.")
        if edu:
            parts.append(f"They studied {', '.join(edu[:2])}.")
        if resume_data.get("raw_text"):
            snippet = resume_data["raw_text"][:600]
            parts.append(f"Resume snippet: {snippet}")
        return " ".join(parts)

    def _parse_questions_from_text(self, text):
        try:
            match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                return [
                    {"question": i["question"].strip(), "ideal_answer": i["ideal_answer"].strip()}
                    for i in arr if "question" in i and "ideal_answer" in i
                ][:5]
            return self._extract_questions_manually(text)
        except Exception:
            print("⚠️ Could not parse Gemini JSON → using manual extraction.")
            return self._extract_questions_manually(text)

    def _extract_questions_manually(self, text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        questions = []
        for line in lines:
            if len(questions) >= 5:
                break
            if line.endswith("?"):
                questions.append({
                    "question": line,
                    "ideal_answer": "Answer should demonstrate understanding and experience."
                })
        return questions[:5]

    def _get_fallback_questions(self, mode):
        if mode == "technical":
            return [
                {"question": "Explain time complexity with an example.", "ideal_answer": "Explain Big O with a real algorithm."},
                {"question": "Difference between REST and GraphQL?", "ideal_answer": "REST uses endpoints; GraphQL allows flexible queries."},
                {"question": "How would you optimize a slow backend API?", "ideal_answer": "Discuss profiling, caching, and DB tuning."},
                {"question": "SQL vs NoSQL?", "ideal_answer": "SQL is relational; NoSQL is flexible and scalable."},
                {"question": "Explain version control importance.", "ideal_answer": "Tracks changes, enables collaboration, rollback."}
            ]
        return [
            {"question": "Tell me about yourself.", "ideal_answer": "Concise summary of experience and motivation."},
            {"question": "What are your strengths?", "ideal_answer": "2–3 strengths with examples."},
            {"question": "Describe a challenging project.", "ideal_answer": "Show teamwork and problem-solving."},
            {"question": "Where do you see yourself in five years?", "ideal_answer": "Show ambition aligned with goals."},
            {"question": "Why do you want to work here?", "ideal_answer": "Align personal goals with company mission."}
        ]


# -------------------------------------------------------------------------
# MAIN WRAPPER FUNCTION
# -------------------------------------------------------------------------
def generate_personalized_questions(
    resume_data: Dict[str, Any], mode: str = "technical"
) -> List[Dict[str, str]]:
    print("\n==============================")
    print("🎯 Generating Personalized Questions")
    print("==============================")
    generator = QuestionGenerator()
    try:
        questions = generator.generate_questions_gemini(resume_data, mode)
        if not questions or len(questions) < 3:
            print("⚠️ Gemini returned weak results → using TEMPLATE fallback.")
            return generator.generate_questions_free(resume_data, mode)
        print("✅ Successfully generated questions!\n")
        return questions
    except Exception as e:
        print("💥 Total failure:", str(e))
        return generator._get_fallback_questions(mode)
