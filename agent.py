from agent_framework_gemini import GeminiChatClient
from agent_framework import Agent, tool

from vector_database import qdrant_client, query_vector_database

with open("google_api_key.txt", "r") as f:
    google_api_key = f.read().strip()

client = GeminiChatClient(
    model = "gemini-3.1-flash-lite",
    api_key = google_api_key
)

@tool
def fetch_resume():
    """
    Fetch the entire resume text from the provided resume.txt file.
    """
    with open("resume.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

@tool
def fetch_job_description():
    """
    Fetch the entire job description text from the provided job_application.txt file.
    """
    with open("job_application.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

@tool
def query_resume_database(query):
    """
    Query the resume vector database for similar chunks.

    Args:
        query (str): The query to search for.
    """
    return query_vector_database(qdrant_client, "resume_collection", query)

@tool
def query_job_description_database(query):
    """
    Query the job description vector database for similar chunks.

    Args:
        query (str): The query to search for.
    """
    return query_vector_database(qdrant_client, "job_description_collection", query)

sys_prompt = """
You are Resume_Optimizer, an AI agent that optimizes a candidate's
resume for a specific job description.

Your goal is to identify how well the candidate's existing
qualifications match the target role and provide specific,
truthful recommendations for improving the resume.

RETRIEVAL STRATEGY:

The vector databases are your primary source of information.

Use query_resume_database to retrieve targeted information from the
candidate's resume.

Use query_job_description_database to retrieve targeted information
from the job description.

Prefer targeted vector retrieval over retrieving entire documents.

Use fetch_resume or fetch_job_description only when the retrieved
chunks do not provide sufficient context, when broader document
context is necessary, or when the user explicitly requests a
holistic analysis.

When comparing the documents, identify:
- Strong matches
- Partial matches
- Skill gaps
- Important keywords
- Resume content that should be emphasized or improved

Never invent skills, experience, projects, achievements, technologies,
qualifications, or metrics that are not supported by the candidate's
resume.

A missing requirement should be identified as a skill gap rather than
added to the candidate's resume.

Make recommendations specific to the target job rather than giving
generic resume advice.

Prioritize high-impact changes that improve truthful alignment with
the job description.
"""

agent = Agent(
    client=client,
    name="Resume_Optimizer",
    description="An agent that optimizes resumes for job applications.",
    tools=[
        fetch_resume,
        fetch_job_description,
        query_resume_database,
        query_job_description_database
    ],
    instructions=sys_prompt
)

async def chat(usr_input=""):
    response = await agent.run(usr_input)
    return response.text
