import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def extract_resume_info(text):
    """
    Extract information from resume text
    """
    result = {
        "name": "Not Found",
        "email": "Not Found", 
        "phone": "Not Found",
        "skills": [],
        "education": [],
        "projects": [],
        "experience": [],
        "raw_text": text
    }
    
    try:
        logger.info("Starting resume information extraction")
        
        # Extract name (first line with proper case)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        logger.info(f"Processing {len(lines)} lines of text")
        
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
                result["name"] = line
                logger.info(f"Found name: {line}")
                break
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            result["email"] = email_match.group()
            logger.info(f"Found email: {result['email']}")
        
        # Extract phone number
        phone_patterns = [
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(\+?\d{1,3}[-.\s]?)?\d{10}',
            r'(\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                result["phone"] = phone_match.group()
                logger.info(f"Found phone: {result['phone']}")
                break
        
        # Extract skills
        skills_patterns = [
            r'(?i)(?:skills?|technologies?|technical skills?)[:\-\s]*\n?(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:programming languages?|languages?)[:\-\s]*\n?(.*?)(?=\n\n|\n[A-Z]|$)',
            r'(?i)(?:tools?|software)[:\-\s]*\n?(.*?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in skills_patterns:
            skills_match = re.search(pattern, text, re.DOTALL)
            if skills_match:
                skills_text = skills_match.group(1)
                # Split by common delimiters
                skills = re.split(r'[,\n•·\-\|\;]', skills_text)
                found_skills = [skill.strip() for skill in skills if skill.strip() and len(skill.strip()) > 1][:10]
                if found_skills:
                    result["skills"].extend(found_skills)
                    logger.info(f"Found {len(found_skills)} skills")
                    break
        
        # Remove duplicates from skills
        result["skills"] = list(dict.fromkeys(result["skills"]))
        
        # Extract education
        education_patterns = [
            r'(?i)(bachelor[^,\n]*)',
            r'(?i)(master[^,\n]*)',
            r'(?i)(phd[^,\n]*)',
            r'(?i)(diploma[^,\n]*)',
            r'(?i)([^,\n]*university[^,\n]*)',
            r'(?i)([^,\n]*college[^,\n]*)',
            r'(?i)([^,\n]*institute[^,\n]*)',
            r'(?i)(b\.?tech[^,\n]*)',
            r'(?i)(m\.?tech[^,\n]*)',
            r'(?i)(b\.?sc[^,\n]*)',
            r'(?i)(m\.?sc[^,\n]*)'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 5 and cleaned not in result["education"]:
                    result["education"].append(cleaned)
        
        logger.info(f"Found {len(result['education'])} education entries")
        
        # Extract projects
        project_patterns = [
            r'(?i)projects?\s*[:\-]*\n(.*?)(?=\n\n[A-Z]|\n[A-Z][A-Z]|experience|education|skills|$)',
            r'(?i)personal projects?\s*[:\-]*\n(.*?)(?=\n\n[A-Z]|\n[A-Z][A-Z]|experience|education|skills|$)'
        ]
        
        for pattern in project_patterns:
            project_section = re.search(pattern, text, re.DOTALL)
            if project_section:
                project_text = project_section.group(1)
                # Split by bullet points or line breaks
                projects = re.split(r'\n(?=•|\d\.|\-)', project_text)
                for project in projects:
                    cleaned = re.sub(r'^[•\-\d\.\s]+', '', project).strip()
                    if len(cleaned) > 10:
                        result["projects"].append(cleaned[:200])  # Limit length
                break
        
        logger.info(f"Found {len(result['projects'])} project entries")
        
        # Extract experience
        exp_patterns = [
            r'(?i)(?:experience|work experience|employment|professional experience)\s*[:\-]*\n(.*?)(?=\n\n[A-Z]|\n[A-Z][A-Z]|education|projects|skills|$)',
            r'(?i)(?:career|work history)\s*[:\-]*\n(.*?)(?=\n\n[A-Z]|\n[A-Z][A-Z]|education|projects|skills|$)'
        ]
        
        for pattern in exp_patterns:
            exp_section = re.search(pattern, text, re.DOTALL)
            if exp_section:
                exp_text = exp_section.group(1)
                # Split by common patterns
                experiences = re.split(r'\n(?=[A-Z][^a-z\n]*(?:at|@|\|))', exp_text)
                for exp in experiences:
                    cleaned = exp.strip()
                    if len(cleaned) > 10:
                        result["experience"].append(cleaned[:300])  # Limit length
                break
        
        logger.info(f"Found {len(result['experience'])} experience entries")
        
        logger.info("Resume extraction completed successfully")
    
    except Exception as e:
        logger.error(f"Error in extract_resume_info: {str(e)}")
        # Return basic structure with error info
        result["error"] = str(e)
    
    return result