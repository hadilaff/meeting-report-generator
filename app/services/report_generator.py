import whisper
from pyannote.audio import Pipeline
from transformers import pipeline
from fpdf import FPDF
import torch
import gc
import re
import os
from dotenv import load_dotenv
from groq import Groq
from app.core.config import settings

load_dotenv()
#HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_TOKEN = settings.HF_TOKEN

def generate_report(audio_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Whisper transcription
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)

    # Speaker diarization
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    diarization = diarization_pipeline(audio_path)
    segments = [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in diarization.itertracks(yield_label=True)]

    # Associate transcription with speakers
    def segments_overlap(s1, e1, s2, e2):
        return max(s1, s2) < min(e1, e2)

    structured_transcript = []
    for seg in result["segments"]:
        start, end, text = seg["start"], seg["end"], seg["text"]
        speaker = "Speaker_Unknown"
        for d in segments:
            if segments_overlap(start, end, d["start"], d["end"]):
                speaker = d["speaker"]
                break
        structured_transcript.append(f"{speaker}: {text}")



    #Summarization Method  using prompt
    #groq_api_key= os.getenv("GROQ_API_KEY")
    groq_api_key = settings.GROQ_API_KEY

    llm_client = Groq(api_key=groq_api_key)

    transcript_text = "\n".join(structured_transcript)
    prompt = f"""
    You are a professional meeting summarizer.
    Analyze the transcript and generate a structured summary including:
    1. Overview of the meeting
    2. Objectives
    3. Decisions made
    4. Tasks to do

    Transcript:
    {transcript_text}
    """

    chat_completion = llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional meeting summarizer."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4000
    )

    summary_text = chat_completion.choices[0].message.content

    # 5. Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Meeting Report", ln=True, align="C")

    # Structured transcript
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, "Transcription with Speakers:", ln=True)
    pdf.ln(5)

    for line in structured_transcript:
        pdf.multi_cell(0, 8, line)

    # Groq summary
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary of the Meeting:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, summary_text)

    # Save PDF
    pdf_file = "meeting_report.pdf"
    pdf.output(pdf_file)

    # Cleanup
    del model, diarization_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return pdf_file


#Method2 using MEETING_SUMMARY
"""     # Summarization
    summarizer_device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model="knkarthick/MEETING_SUMMARY",
        device=summarizer_device
        #device=-1
    )
    full_text = " ".join(structured_transcript)
    summary = summarizer(full_text, max_length=300, min_length=100, do_sample=False)


    summary_text=summary[0]['summary_text']
    summary_dict= {
        "Objectives": [],
        "Tasks":[]
    }

    # define Objectives 
    objectives = re.findall(r"(development of .*?project|improve .*?pipeline|optimize .*?pipeline|goal .*?|objective .*?|purpose .*?|aim .*?|target .*?|mission .*?)", summary_text, re.IGNORECASE)
    for obj in objectives:
        summary_dict["Objectives"].append(f"{obj.strip().capitalize()}")


    # define Tasks 
    tasks = re.findall(r"(test .*?|prepare .*?|investigate .*?|analyze .*?|define .*?|create .*?|assign .*?|review .*?|discuss .*?)\.", summary_text, re.IGNORECASE)
    for task in tasks:
        summary_dict["Tasks"].append(f"{task.strip().capitalize()}") 
    
    

    # print results
    print("\n🔹 Structured Sections:\n")
    for section, items in summary_dict.items():
        print(section + ":")
        if items:
            for item in items:
             print(" -", item)
        else:
            print(" - None")  
            
              
 """