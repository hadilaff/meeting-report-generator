from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from app.services.report_generator import generate_report

router = APIRouter()

@router.post("/generate-report", response_class=FileResponse)
async def generate_meeting_report(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Generate report
    pdf_file = generate_report(temp_path)
    
    return FileResponse(pdf_file, filename="meeting_report.pdf")
