from agent_framework_gemini import GeminiChatClient
from agent_framework import Agent, tool

from text_chunker import chunk_text
from vector_database import create_vector_database, query_vector_database, embed_query

with open("google_api_key.txt", "r") as f:
    google_api_key = f.read().strip()

client = GeminiChatClient(
    model = "gemini-3.1-flash-lite",
    api_key = google_api_key
)

@tool
def fetch_resume(resume_id):
    """
    Fetch a resume from a database or API based on the provided resume_id.
    For demonstration purposes, this function returns a static resume text.
    """
    return ""

agent = Agent(
    client=client,
    name="Resume_Optimizer",
    description="An agent that optimizes resumes for job applications.",
    tools=[
        tool(
            name="optimize_resume",
            description="Optimizes a resume based on job description and best practices.",
            func=lambda resume, job_description: f"Optimized Resume for {job_description}: {resume}"
        )
    ],
    instructions="You are a resume optimization agent. Use the provided tool to optimize resumes based on job descriptions."
)

async def chat(usr_input=""):
    response = await agent.chat(usr_input)
    return response
