from fastapi import APIRouter, HTTPException
import os
from typing import List, Optional
from pydantic import BaseModel
import warnings
import json
from langchain.prompts import ChatPromptTemplate
import ast
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from reportlab.platypus import Frame, NextPageTemplate, PageTemplate, BaseDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from datetime import datetime

warnings.filterwarnings("ignore")

router = APIRouter()

# ---------- RAG Setup (Load once) ----------
# ---------- API Schema ----------
from typing import Dict, List, Union
import re

class SymptomAnalysisRequest(BaseModel):
    symptoms: str
    language: Optional[str] = "en"
    detailed: Optional[bool] = False

# Pydantic model: only the diagnoses dictionary
class AnalysisResult(BaseModel):
    diagnoses: dict  # disease -> confidence (0.0 to 1.0)

class DiseaseInfo(BaseModel):
    infos: dict

class DiseaseInput(BaseModel):
    disease_name: str



    


def process_rag_response(rag_response: Union[str, None]) -> AnalysisResult:
    if not rag_response or not rag_response.strip():
        raise ValueError("Empty response from RAG model")

    # Extract JSON array from any extra text
    match = re.search(r'\[\s*\{.*?\}\s*\]', rag_response, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON array found in response")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(data, list) or not data:
        raise ValueError("Parsed JSON is not a valid non-empty list")

    # Build the diagnoses dict
    diagnoses = {
        item.get("disease", "Unknown"): item.get("probability", 0) / 100
        for item in data
    }
    
    print(diagnoses)
    return AnalysisResult(diagnoses=diagnoses)



def extract_dict_from_text(text):
    # Find the code block containing the dictionary
    match = re.search(r"```python\s*({.*?})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"({.*})", text, re.DOTALL)
    if match:
        dict_str = match.group(1)
        try:
            return ast.literal_eval(dict_str)
        except Exception:
            return None
    return None


from fastapi import Request

def analyze_symptoms(request: Request, symptoms: str) -> AnalysisResult:
    try:
        rag_response = request.app.state.diagnosis_chain.run(symptoms)
        return process_rag_response(rag_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom analysis failed: {str(e)}")

    

def get_disease_info(request: Request,disease_name: str) -> DiseaseInfo:
    dict_str = request.app.state.info_chain.run(disease_name)
    disease_info = extract_dict_from_text(dict_str)
    return DiseaseInfo(infos=disease_info)


def generate_pdf_report(disease_info, filename="report.pdf")->None:
    """Generates a scientific article-style PDF report with two-column layout and a logo."""

    # Custom styles
    styles = getSampleStyleSheet()

    # Add custom styles
    styles.add(ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        alignment=1,  # Center
        spaceAfter=20,
        textColor=colors.darkblue
    ))

    styles.add(ParagraphStyle(
        name='AuthorStyle',
        parent=styles['Heading3'],
        fontSize=10,
        leading=12,
        alignment=1,
        spaceAfter=20,
        textColor=colors.darkgrey
    ))

    styles.add(ParagraphStyle(
        name='AbstractStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        backColor=colors.lightgrey,
        borderPadding=5,
        spaceAfter=20
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.darkblue,
        underlineWidth=1,
        underlineColor=colors.darkblue,
        underlineOffset=-5
    ))

    styles.add(ParagraphStyle(
        name='LeftColumn',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        rightIndent=5,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='RightColumn',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        alignment=TA_JUSTIFY,
        leftIndent=5,
        rightIndent=0,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        leftIndent=15,
        bulletIndent=0,
        spaceAfter=6,
        bulletFontName='Symbol',
        bulletFontSize=8
    ))

    styles.add(ParagraphStyle(
        name='Reference',
        parent=styles['Italic'],
        fontSize=8,
        leading=10,
        textColor=colors.darkgrey,
        spaceBefore=20
    ))

    # Create document with two columns
    class TwoColumnDocTemplate(BaseDocTemplate):
        def __init__(self, filename, **kw):
            BaseDocTemplate.__init__(self, filename, **kw)
            # Calculate column widths
            page_width = self.pagesize[0] - 2*self.leftMargin
            col_width = (page_width - 1*cm) / 2  # 1cm gutter

            # First page template with title
            first_page = PageTemplate(id='FirstPage',
                frames=[
                    Frame(self.leftMargin, self.bottomMargin, 
                          col_width, self.height, 
                          id='leftCol'),
                    Frame(self.leftMargin + col_width + 1*cm, 
                          self.bottomMargin, 
                          col_width, self.height, 
                          id='rightCol')
                ])
            self.addPageTemplates(first_page)

            # Other pages template
            other_pages = PageTemplate(id='OtherPages',
                frames=[
                    Frame(self.leftMargin, self.bottomMargin, 
                          col_width, self.height, 
                          id='leftCol2'),
                    Frame(self.leftMargin + col_width + 1*cm, 
                          self.bottomMargin, 
                          col_width, self.height, 
                          id='rightCol2')
                ])
            self.addPageTemplates(other_pages)

    doc = TwoColumnDocTemplate(filename,
                             pagesize=letter,
                             leftMargin=2*cm,
                             rightMargin=2*cm,
                             topMargin=2*cm,
                             bottomMargin=2*cm)

    story = []

    # Add logo at the top (centered)
    try:
        logo = Image('app/api/data/logo_platform.jpg', width=6*cm, height=2*cm)
        logo.hAlign = 'CENTER'
        story.append(logo)
        story.append(Spacer(1, 0.3*cm))
    except Exception as e:
        pass  # If logo not found, skip

    # Title and authors
    title = Paragraph(disease_info.get('Title', 'Medical Condition Report'), styles['TitleStyle'])
    authors = Paragraph("Generated by AIHealthCheck AI Assistant", styles['AuthorStyle'])
    date = Paragraph(datetime.now().strftime("%B %d, %Y"), styles['AuthorStyle'])

    story.append(title)
    story.append(authors)
    story.append(date)
    story.append(NextPageTemplate('OtherPages'))  # Switch to two-column layout

    # Abstract
    abstract_text = f"<b>Abstract</b><br/><br/>{disease_info.get('Overview', 'No overview available.')}"
    abstract = Paragraph(abstract_text, styles['AbstractStyle'])
    story.append(abstract)

    # Function to format content
    def format_content(text, style):
        if not text:
            return ""
        if isinstance(text, list):
            return [Paragraph(f"• {item}", styles['BulletPoint']) for item in text]
        return Paragraph(text, style)

    # Organize content into left and right columns
    left_column_content = [
        ('Symptoms', disease_info.get('Symptoms')),
        ('Causes', disease_info.get('Causes')),
        ('Risk Factors', disease_info.get('Risk factors')),
        ('Complications', disease_info.get('Complications'))
    ]

    right_column_content = [
        ('Diagnosis', disease_info.get('Diagnosis')),
        ('Treatment', disease_info.get('Treatment')),
        ('Prevention', disease_info.get('Prevention')),
        ('When to See a Doctor', disease_info.get('When to see a doctor')),
        ('Lifestyle and Home Remedies', disease_info.get('Lifestyle and home remedies'))
    ]

    # Add left column content
    for section, content in left_column_content:
        if content:
            story.append(Paragraph(section, styles['SectionHeader']))
            formatted = format_content(content, styles['LeftColumn'])
            if isinstance(formatted, list):
                story.extend(formatted)
            else:
                story.append(formatted)

    # Switch to right column
    story.append(NextPageTemplate('OtherPages'))

    # Add right column content
    for section, content in right_column_content:
        if content:
            story.append(Paragraph(section, styles['SectionHeader']))
            formatted = format_content(content, styles['RightColumn'])
            if isinstance(formatted, list):
                story.extend(formatted)
            else:
                story.append(formatted)

    # Add references
    story.append(Paragraph("References", styles['SectionHeader']))
    story.append(Paragraph("1. World Health Organization (WHO) Disease Database", styles['Reference']))
    story.append(Paragraph("2. Mayo Clinic Medical References", styles['Reference']))
    story.append(Paragraph("3. Centers for Disease Control and Prevention (CDC)", styles['Reference']))

    # Build the PDF
    doc.build(story)


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_selected_symptoms(request: Request, body: SymptomAnalysisRequest):
    if not body.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required")
    return analyze_symptoms(request, body.symptoms)


@router.post("/infos", response_model=DiseaseInfo)
async def give_full_infos(request: Request,requests: DiseaseInput):
    if not requests.disease_name:
        raise HTTPException(
            status_code=400,
            detail="At least one symptom is required"
        )
    disease_info=get_disease_info(request,requests.disease_name)
    filename = f"app/api/report/report.pdf"
    generate_pdf_report(disease_info.infos, filename)
    return disease_info

