import whisper
from pyannote.audio import Pipeline
from transformers import pipeline
from fpdf import FPDF
import torch
import gc
import re
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

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

    # Summarization
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
    
    

    # Pretty print results
    print("\n🔹 Structured Sections:\n")
    for section, items in summary_dict.items():
        print(section + ":")
        if items:
            for item in items:
             print(" -", item)
        else:
            print(" - None")    

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Meeting Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, "Structured Transcript:", ln=True)
    pdf.ln(5)
    for line in structured_transcript:
        pdf.multi_cell(0, 8, line)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, summary[0]['summary_text'])

    # Add structured sections
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Structured Sections:", ln=True)

    for key, value in summary_dict.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"{key}:", ln=True)
        pdf.set_font("Arial", "", 12)
        
        if value:  # join list into string
            for item in value:
                pdf.multi_cell(0, 8, f"- {item}")
        else:
            pdf.multi_cell(0, 8, "Not mentioned")
        
        pdf.ln(3)

        pdf_file = "meeting_report.pdf"
        pdf.output(pdf_file)

    # Clean up memory
    del model, diarization_pipeline, summarizer
    gc.collect()
    torch.cuda.empty_cache()

    return pdf_file
