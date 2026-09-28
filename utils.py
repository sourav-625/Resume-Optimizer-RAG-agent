from vector_database import create_vector_database
from text_chunker import chunk_text

def fetch_resume():
    """
    Fetch the resume text from the provided resume.txt file.
    """
    with open("resume.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def fetch_job_description():
    """
    Fetch the job description text from the provided job_application.txt file.
    """
    with open("job_application.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def create_vector_db_from_resume():
    """
    Create a vector database from the resume text.
    """
    resume_text = fetch_resume()
    chunks = chunk_text(resume_text)
    client, collection_name = create_vector_database(chunks, database_name="resume_collection")
    return f"Vector database created with collection name: {collection_name}"

def create_vector_db_from_job_description():
    """
    Create a vector database from the job description text.
    """
    job_description_text = fetch_job_description()
    chunks = chunk_text(job_description_text)
    client, collection_name = create_vector_database(chunks, database_name="job_description_collection")
    return f"Vector database created with collection name: {collection_name}"