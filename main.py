from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from agent import chat
from utils import (
    create_vector_db_from_resume,
    create_vector_db_from_job_description,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_files(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):
    print("File A:", file_a.filename)
    print("File B:", file_b.filename)

    file_a_content = PdfReader(file_a.file)
    file_b_content = PdfReader(file_b.file)

    print("File A page count:", len(file_a_content.pages))
    print("File B page count:", len(file_b_content.pages))

    file_a_text = ""
    for page in file_a_content.pages:
        file_a_text += page.extract_text() or ""

    file_b_text = ""
    for page in file_b_content.pages:
        file_b_text += page.extract_text() or ""

    with open("resume.txt", "w", encoding="utf-8") as f:
        f.write(file_a_text)

    with open("job_application.txt", "w", encoding="utf-8") as f:
        f.write(file_b_text)

    create_vector_db_from_resume()
    create_vector_db_from_job_description()

    return {
        "response": await chat("Please optimize the resume based on the job description provided."),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
