from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

@router.get("/report")
async def get_report():
    report_path = f"app/api/report/report.pdf"
    
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"Report.pdf"
    )