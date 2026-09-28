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
    Fetch the resume text from the provided resume.txt file.
    """
    with open("resume.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

@tool
def fetch_job_description():
    """
    Fetch the job description text from the provided job_application.txt file.
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
    instructions="You are a resume optimization agent. Use the provided tool to optimize resumes based on job descriptions."
)

async def chat(usr_input=""):
    response = await agent.run(usr_input)
    return response.text
