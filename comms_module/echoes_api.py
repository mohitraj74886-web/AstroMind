from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
import shutil
import os
import uuid

## Access through -> uvicorn comms_module.echoes_api:app --reload --port 8003
# 1. Initialize App
app = FastAPI(
    title="AstroMind Comms Module (Voice Echoes)",
    description="Handles delayed audio transmissions from Earth to deep space.",
    version="2.0.0"
)

# 2. Setup Audio Storage Directory
AUDIO_DIR = "saved_voice_notes"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Mount the directory so the frontend can actually play the files via URL!
app.mount("/audio_files", StaticFiles(directory=AUDIO_DIR), name="audio_files")

# 3. In-Memory Database (Will hold message metadata)
inbox_database = []

# 4. The Audio Transmission Endpoint (Uploads from Earth)
@app.post("/echoes/transmit_audio")
async def transmit_voice_note(
    sender: str = Form(..., description="e.g., Wife (Earth)"),
    recipient: str = Form(..., description="e.g., Commander"),
    distance_millions_km: float = Form(..., description="Current distance in millions of km"),
    audio_file: UploadFile = File(..., description="The .wav or .mp3 voice recording")
):
    try:
        # Step A: Physics Calculation
        distance_km = distance_millions_km * 1_000_000
        speed_of_light_kms = 299792.0
        delay_seconds = distance_km / speed_of_light_kms
        
        # Step B: Calculate Future Arrival Time
        now_utc = datetime.now(timezone.utc)
        arrival_time = now_utc + timedelta(seconds=delay_seconds)
        
        # Step C: Save the Audio File Safely
        file_extension = audio_file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(AUDIO_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
            
        # Step D: Log into Database
        packet = {
            "id": len(inbox_database) + 1,
            "sender": sender,
            "recipient": recipient,
            "file_url": f"http://127.0.0.1:8003/audio_files/{unique_filename}", # URL for the website audio player
            "sent_time_utc": now_utc.isoformat(),
            "arrival_time_utc": arrival_time.isoformat(),
            "status": "IN_TRANSIT"
        }
        
        inbox_database.append(packet)
        
        return {
            "status": "Audio Transmission Launched",
            "eta_seconds": round(delay_seconds, 2),
            "eta_minutes": round(delay_seconds / 60, 2),
            "arrival_time_utc": arrival_time.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. The Astronaut's Inbox Endpoint
@app.get("/echoes/inbox/{astronaut_name}")
async def check_inbox(astronaut_name: str):
    """Filters messages, only showing ones that have 'arrived'."""
    now_utc = datetime.now(timezone.utc)
    
    delivered_messages = []
    in_transit_count = 0
    
    for msg in inbox_database:
        if msg["recipient"].lower() == astronaut_name.lower():
            # Check if the arrival time is in the past
            msg_arrival_time = datetime.fromisoformat(msg["arrival_time_utc"])
            
            if now_utc >= msg_arrival_time:
                msg["status"] = "DELIVERED"
                delivered_messages.append(msg)
            else:
                in_transit_count += 1
                
    return {
        "messages_in_transit": in_transit_count,
        "unread_delivered_count": len(delivered_messages),
        "inbox": delivered_messages
    }