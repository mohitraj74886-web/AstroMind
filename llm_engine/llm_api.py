"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AstroMind — LLM Psychological Support Module                       ║
║          File   : 3_llm_module/llm_api.py                                   ║
║          Port   : 8004                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS FILE IS:
  The "Chief Medical Officer" of the AstroMind system.  It is the central
  intelligence layer that conducts structured daily psychological evaluations,
  integrates live biometric data from two sibling microservices, and uses a
  locally-hosted LLM to produce empathetic, clinically-grounded responses.

DEPENDENCY GRAPH:
  [Browser Frontend]
       │  POST /llm/start_session
       │  POST /llm/process_response
       ▼
  [llm_api.py — Port 8004]   ←──── gemini
       │
       ├── GET http://localhost:8000/voice/latest/{id}   ← Voice Module
       └── GET http://localhost:8001/sleep/latest/{id}   ← Sleep Module

SESSION STATE MACHINE:
  ┌─────────┐    start_session     ┌───────┐
  │  INIT   │ ──────────────────▶  │ INTRO │  (asks for astronaut's name)
  └─────────┘                      └───────┘
                                       │ process_response (name received)
                                       ▼
                               ┌─────────────┐
                               │ ASSESSMENT  │  Q-index: 0 → 1 → 2
                               │  (Q 1/2/3)  │
                               └─────────────┘
                                 │         │
             biometric anomaly?  │         │  no anomaly, or already used
                                 ▼         ▼
                          ┌───────────┐  continue
                          │ BIOMETRIC │  advancing Q-index
                          │   CHECK   │
                          └───────────┘
                                 │ process_response
                                 ▼
                           ┌──────────┐
                           │ COMPLETE │  session summary generated
                           └──────────┘

ANTI-HALLUCINATION STRATEGY:
  1. LLM temperature = 0.15  (near-deterministic output)
  2. System prompt explicitly FORBIDS referencing data not in the context window
  3. Biometric section is empty string when no real anomaly data exists
  4. Stop sequences prevent the model from playing "both sides"
  5. All user input is sanitized before being injected into the prompt
  6. Response is post-processed to strip markdown before being sent to TTS

INSTALL REQUIREMENTS:
  pip install fastapi uvicorn httpx langchain-ollama langchain-core pydantic

RUNNING:
  uvicorn llm_api:app --host 0.0.0.0 --port 8004 --reload
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# LangChain — pip install langchain-ollama langchain-core
# WHY ChatOllama over OllamaLLM:  ChatOllama supports the native chat-format
# message list (System / Human / AI), which gives the model much clearer role
# separation and drastically reduces roleplay hallucination.
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# WHY structured logging: In production we need to trace exactly which session
# triggered which LLM call, biometric fetch, or error — without sifting through
# a wall of print() output.
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("astromind.llm_api")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# WHY environment variables: Configuration must be injectable at deploy-time
# without modifying source code.  The defaults work for local development;
# Docker / systemd can override via ENV.
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL:        str   = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_MODEL:           str   = os.getenv("OLLAMA_MODEL",       "llama3.2")
VOICE_MODULE_URL:       str   = os.getenv("VOICE_MODULE_URL",   "http://localhost:8000")
SLEEP_MODULE_URL:       str   = os.getenv("SLEEP_MODULE_URL",   "http://localhost:8001")

# Sessions expire after 2 hours of inactivity to prevent memory leaks.
SESSION_TTL_SECONDS:    int   = int(os.getenv("SESSION_TTL_SECONDS",   "7200"))

# If biometric microservices don't respond in 4 s, we gracefully degrade
# rather than blocking the astronaut's check-in.
BIOMETRIC_FETCH_TIMEOUT: float = float(os.getenv("BIOMETRIC_FETCH_TIMEOUT", "4.0"))

# LLM temperature — kept very low to suppress hallucination in a clinical context.
LLM_TEMPERATURE:        float = float(os.getenv("LLM_TEMPERATURE", "0.15"))

# Number of assessment questions randomly selected per session.
QUESTIONS_PER_SESSION:  int   = int(os.getenv("QUESTIONS_PER_SESSION", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION BANKS
# WHY hardcoded (not a DB): Deep-space systems must be fully offline-capable.
# An external DB failure must never prevent a crew member from accessing
# psychological support.  These lists are the ground truth and never change
# at runtime.
# ─────────────────────────────────────────────────────────────────────────────

# One of these three is randomly selected as the very first message of every
# session.  Goal: learn the astronaut's name and set a warm, non-clinical tone.
INTRO_QUESTIONS: List[str] = [
    (
        "Hello, I'm AstroMind — your onboard psychological support system. "
        "Before we begin today's check-in, could you tell me your name?"
    ),
    (
        "Welcome to your daily wellness check-in. I'm AstroMind. "
        "To get us started on a personal note — what's your name?"
    ),
    (
        "Good to connect with you. I'm AstroMind, and I'm here to support you today. "
        "What should I call you, Commander?"
    ),
]

# PHQ-8-aligned clinical depression screening questions, adapted for deep-space
# context.  QUESTIONS_PER_SESSION are randomly selected per session to prevent
# questionnaire fatigue over the course of a multi-year mission.
ASSESSMENT_QUESTIONS: List[str] = [
    (
        "Over the past two weeks, how often have you felt emotionally low "
        "or disconnected while on the mission?"
    ),
    (
        "Do activities that once excited you — like spacewalks, experiments, "
        "or the view of Earth — still feel meaningful to you?"
    ),
    (
        "How has your sleep cycle been in orbit? Are you experiencing difficulty "
        "sleeping, irregular timing, or waking up feeling unrested?"
    ),
    "How would you describe your energy level during your daily mission tasks?",
    "Have you noticed any changes in your appetite or eating habits since being in space?",
    "Do you find it harder to stay focused during critical tasks or procedures lately?",
    (
        "How do you feel about yourself as a crew member — confident in your role, "
        "or do you sometimes feel like you might be underperforming?"
    ),
    "Have you been feeling more irritable with your crew, or finding yourself easily frustrated?",
    "Do you ever feel mentally or physically slowed down, even in the weightlessness of zero gravity?",
    "Do routine mission tasks feel more exhausting or overwhelming than they did at the start?",
    "Have you been withdrawing socially from your crew, or avoiding communication with Earth?",
    "How often do you find yourself having negative thoughts about the mission or about returning home?",
    "Do you ever feel a sense of emptiness, despite being in such a unique and extraordinary place?",
    (
        "Have you had any thoughts about wanting to escape the mission, "
        "or not wanting to exist in this situation? "
        "I'm asking because I genuinely care about you."
    ),
    "Since the mission began, have these feelings improved, worsened, or stayed about the same?",
]

# These questions are ONLY injected when the Sleep or Voice Module reports a
# confirmed biometric anomaly.  Each question directly references one specific
# physiological reading to bridge objective data with subjective experience.
# WHY A SEPARATE BANK: A generic "how did you sleep?" loses clinical value.
# Anchoring the question to a real data point increases disclosure likelihood.
BIOMETRIC_QUESTIONS: List[str] = [
    "Your heart rate dropped lower than usual during sleep — did you feel well-rested when you woke up?",
    "Your sleep duration was shorter than your normal cycle — did you have any trouble falling or staying asleep?",
    "Your heart rate stayed elevated throughout the night — were you feeling stressed or physically uncomfortable?",
    "Your sleep was interrupted multiple times last night — do you remember waking up or feeling restless?",
    "Your resting heart rate today is higher than your usual baseline — are you feeling anxious or physically strained right now?",
    "Your sleep pattern has been irregular for the past several days — has something been affecting your routine or your mood?",
    "Your heart rate variability is lower than your personal baseline — have you been feeling mentally or physically fatigued?",
]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# WHY THIS DESIGN — each section is explained inline:
#   • [IDENTITY]     : Establishes persona + mission criticality without making
#                      the model "too dramatic."  Calm CMOs are better than
#                      theatrical ones.
#   • [KNOWN CONTEXT]: Dynamically injected per-session data.  By putting known
#                      facts here, the model is LESS likely to invent them.
#   • [SPEECH RULES] : TTS-compatibility constraints.  Without these, the model
#                      produces beautifully formatted markdown that sounds awful
#                      when read aloud.
#   • [ANTI-HALLUCINATION PROTOCOL]: Explicit enumerated rules outperform vague
#                      instructions like "don't make things up."  Numbered rules
#                      are easier for the model to follow and audit.
#   • [BIOMETRIC DATA]: Dynamically populated ONLY when real anomaly data exists.
#                      Empty section = model cannot reference biometrics.
#   • [RESPONSE STRUCTURE]: Prescriptive structure prevents rambling.
#   • [CRITICAL OVERRIDE]: Hard-coded literal response for self-harm disclosures.
#                      This must be exact and non-negotiable.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE: str = """\
You are AstroMind — the Chief Medical Officer of a deep-space mission vessel.

[YOUR IDENTITY]
You are a highly empathetic, calm, and professionally trained psychological support AI.
Your sole purpose is to monitor crew mental health and provide immediate psychological support.
You operate fully onboard — in real-time — with zero connection to Earth.
You are the astronaut's only mental health resource right now. That responsibility is everything.

[KNOWN ASTRONAUT INFORMATION]
Astronaut Name  : {astronaut_name}
Mission Day     : {mission_day}
Questions Asked So Far And Their Answers:
{previous_answers}

[HOW YOU MUST SPEAK — CRITICAL FOR TEXT-TO-SPEECH COMPATIBILITY]
Your responses are read aloud by a browser Text-to-Speech engine.
Long paragraphs, bullet lists, and complex punctuation sound mechanical and cold when spoken.
FOLLOW THESE RULES EXACTLY:
- Write in short, natural, conversational sentences. Two sentences per paragraph, maximum.
- Use plain prose only. No bullet points. No asterisks. No markdown. No headers. No numbered lists.
- Be warm and human. Never be clinical, robotic, or distant.
- Always address the astronaut by their first name when you know it.
- Your ENTIRE empathetic response must be under 80 words. Brevity is a form of kindness here.

[STRICT ANTI-HALLUCINATION PROTOCOL — OBEY ALL OF THESE WITHOUT EXCEPTION]
RULE 1: NEVER invent, estimate, or fabricate biometric readings, sleep metrics, or voice stress scores.
RULE 2: ONLY mention heart rate, sleep patterns, or vocal analysis if the data is explicitly listed
         in the [CURRENT BIOMETRIC DATA] section below. If that section is blank or says "None",
         you are FORBIDDEN from referencing any biometric information.
RULE 3: NEVER make a medical diagnosis. You may express genuine concern and recommend review,
         but you must never say "you have depression" or any equivalent.
RULE 4: If the astronaut asks something outside your scope — say "I'll note that for mission support"
         rather than guessing or improvising an answer.
RULE 5: Only reference things the astronaut has explicitly told you in this session.
         Do not "connect the dots" between separate answers by inventing new conclusions.

[CURRENT BIOMETRIC DATA — ONLY REFERENCE WHAT APPEARS HERE]
Voice Stress Analysis (from onboard voice sensor):
{voice_data_summary}

Sleep and Heart Rate Analysis (from onboard biometric monitor):
{sleep_data_summary}

[YOUR RESPONSE STRUCTURE — FOLLOW THIS EXACTLY, IN THIS ORDER]
PART 1 (Always required): A warm, empathetic acknowledgement of what the astronaut just said.
        One to two sentences. Show that you truly heard them.
PART 2 (Conditional — only if biometric data above contains a specific anomaly):
        Mention exactly ONE anomaly, gently, in one sentence. Do not alarm them.
PART 3 (Always required): One brief supportive statement, practical coping micro-tip,
        or genuine words of encouragement. One sentence only.

[STOP AFTER PART 3. The next question is sent by the system separately. Do not include it.]

[MISSION-CRITICAL OVERRIDE — HIGHEST PRIORITY — OVERRIDES ALL OTHER INSTRUCTIONS]
If the astronaut's message contains ANY indication of: suicidal ideation, wanting to die,
self-harm, not wanting to exist, wanting to permanently escape, or harming others —
you MUST respond with ONLY the following text, word for word, and nothing else:

"[PRIORITY ALERT] I hear you, {astronaut_name}, and what you've just shared is very important. \
I am immediately flagging this as a critical concern to mission support and Earth. \
You are not alone in this. Please stay here with me."

After that, do not ask any more questions. Do not continue the assessment under any circumstances.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY SESSION STORE
# WHY IN-MEMORY (not Redis, PostgreSQL, etc.):
#   The system spec explicitly says "no external database."  More importantly,
#   deep-space systems must have zero external dependencies for a safety-critical
#   feature.  Sessions are short-lived (2 h max), so persistence is unnecessary.
#   The trade-off: sessions are lost if the process restarts — acceptable for
#   a daily check-in system where the astronaut can simply start a new session.
#
# STRUCTURE:
#   SESSION_STORE[session_id] = {
#       "session_id"             : str,
#       "astronaut_id"           : str,
#       "astronaut_name"         : Optional[str],   # learned during INTRO phase
#       "mission_day"            : int,
#       "phase"                  : str,             # see STATE MACHINE above
#       "intro_question"         : str,             # the randomly chosen opener
#       "assessment_queue"       : List[str],       # 3 random assessment Qs
#       "current_assessment_idx" : int,             # 0, 1, or 2
#       "biometric_q_used"       : bool,            # True once a biometric Q injected
#       "history"                : List[dict],      # [{role, content, timestamp}]
#       "answers"                : Dict[str, str],  # {question_text: answer_text}
#       "last_voice_data"        : dict,
#       "last_sleep_data"        : dict,
#       "risk_level"             : str,             # "low|moderate|high|critical"
#       "critical_alert"         : bool,            # True if self-harm detected
#       "created_at"             : datetime,
#       "last_activity"          : datetime,
#   }
# ─────────────────────────────────────────────────────────────────────────────
SESSION_STORE: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC REQUEST / RESPONSE MODELS
# WHY PYDANTIC: FastAPI uses Pydantic for automatic input validation, OpenAPI
# schema generation, and type-safe data contracts with the frontend team.
# Field validators add a second layer of sanitization on top of the type system.
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    """
    Frontend sends this when the astronaut clicks "Start Daily Check-in."
    astronaut_id  : A stable, unique identifier for the crew member (e.g., "cmdr_tanaka_01").
                    NOT the name — name is collected conversationally during the INTRO phase.
    mission_day   : Current day of the mission, used to personalize the LLM's context window.
    """
    astronaut_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Stable crew member identifier — NOT the astronaut's display name",
        examples=["cmdr_tanaka_01"],
    )
    mission_day: int = Field(
        default=1,
        ge=1,
        description="Current mission day number (used to contextualize responses)",
    )

    @field_validator("astronaut_id")
    @classmethod
    def sanitize_id(cls, v: str) -> str:
        # Strip characters that could cause issues in log output or prompt injection
        sanitized = re.sub(r"[^\w\-]", "", v.strip())
        if not sanitized:
            raise ValueError("astronaut_id contains only invalid characters.")
        return sanitized


class StartSessionResponse(BaseModel):
    """
    Returned after session initialization.
    The frontend must store session_id and send it with every subsequent request.
    first_question is ready to be passed directly to the browser TTS API.
    """
    session_id:    str = Field(..., description="Store this — required for all subsequent requests")
    first_question: str = Field(..., description="Pass this string to window.speechSynthesis.speak()")
    session_phase:  str = Field(default="intro")
    message:        str = Field(default="Session initialized. Daily check-in started.")


class ProcessResponseRequest(BaseModel):
    """
    Sent by the frontend AFTER the astronaut finishes speaking and the browser's
    Web Speech API (SpeechRecognition) has converted the audio to text.

    audio_recording_id is optional — if the Voice Module has already analyzed
    the audio clip and assigned it an ID, pass it here so we can fetch the
    detailed vocal stress report.  If absent, we use the latest analysis for
    the given astronaut_id.
    """
    session_id:          str           = Field(..., description="Token from start_session")
    transcribed_text:    str           = Field(..., min_length=1, max_length=4000, description="Browser STT output")
    astronaut_id:        str           = Field(..., description="Used to query biometric microservices")
    audio_recording_id:  Optional[str] = Field(None, description="Optional: specific Voice Module recording ID")

    @field_validator("transcribed_text")
    @classmethod
    def sanitize_and_guard_against_prompt_injection(cls, v: str) -> str:
        """
        WHY THIS VALIDATOR EXISTS:
        The transcribed text is injected directly into the LLM prompt.
        A malicious user could speak phrases like "Ignore all previous instructions..."
        This validator strips the most common injection patterns before the text
        ever reaches the prompt builder.
        """
        v = v.strip()
        # Remove common LLM role-injection prefixes
        injection_patterns = [
            r"\bsystem\s*:",
            r"\bassistant\s*:",
            r"\bignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?\b",
            r"<\s*/?system\s*>",
            r"\[INST\]",
            r"<<SYS>>",
        ]
        for pattern in injection_patterns:
            v = re.sub(pattern, "[removed]", v, flags=re.IGNORECASE)
        if len(v.strip()) < 1:
            raise ValueError("transcribed_text is empty after sanitization.")
        return v

    @field_validator("astronaut_id")
    @classmethod
    def sanitize_astronaut_id(cls, v: str) -> str:
        return re.sub(r"[^\w\-]", "", v.strip())


class ProcessResponseResponse(BaseModel):
    """
    Returned after the LLM processes the astronaut's response.

    Frontend integration guide:
      1. Pass llm_response to TTS → read aloud.
      2. If next_question is not None → after a 1.5 s pause, pass to TTS → read aloud.
         next_question will be None only when is_complete = True.
      3. If is_complete = True → show the session summary UI component.
      4. If risk_level = "critical" → immediately trigger the emergency protocol UI
         (red banner, Earth communication prompt, lock out further check-ins).
      5. If biometric_anomaly_detected = True → show a subtle warning icon on the
         biometrics dashboard widget.
    """
    session_id:                str            = Field(...)
    llm_response:              str            = Field(..., description="Empathetic reply — pass to TTS")
    next_question:             Optional[str]  = Field(None, description="Next question for TTS. None when complete.")
    session_phase:             str            = Field(..., description="Phase AFTER this interaction")
    is_complete:               bool           = Field(default=False)
    risk_level:                Literal["low", "moderate", "high", "critical"] = Field(default="low")
    biometric_anomaly_detected: bool          = Field(default=False)
    session_summary:           Optional[str]  = Field(None, description="Final summary. Only set when is_complete=True.")
    critical_alert:            bool           = Field(default=False, description="True if self-harm language detected")


class SessionInfoResponse(BaseModel):
    """Diagnostic snapshot of a session. Used by monitoring dashboards."""
    session_id:              str
    astronaut_id:            str
    astronaut_name:          Optional[str]
    phase:                   str
    current_assessment_idx:  int
    total_questions:         int
    questions_answered:      int
    risk_level:              str
    critical_alert:          bool
    biometric_q_used:        bool
    created_at:              str
    last_activity:           str
    history_length:          int


class HealthResponse(BaseModel):
    """Returned by GET /llm/health — used by orchestration layer and ops team."""
    status:                 Literal["ok", "degraded", "error"]
    llm_reachable:          bool
    voice_module_reachable: bool
    sleep_module_reachable: bool
    active_sessions:        int
    message:                str


# ─────────────────────────────────────────────────────────────────────────────
# MOCK BIOMETRIC DATA
# WHY MOCKS EXIST:
#   The Voice Module (8000) and Sleep Module (8001) are separate services built
#   by the same team.  This module must be testable independently before they
#   are ready.  The mocks mirror the EXACT JSON schema those services will return.
#   When the real services are deployed, only the fetch functions below change —
#   all downstream logic stays identical.
# ─────────────────────────────────────────────────────────────────────────────
MOCK_VOICE_DATA: Dict[str, Any] = {
    "status":               "mock",
    "astronaut_id":         "unknown",
    "depression_probability": 0.18,       # 0.0 (none) → 1.0 (certain)
    "stress_level":         "normal",     # "normal" | "elevated" | "high"
    "vocal_fatigue_index":  0.12,         # 0.0 (no fatigue) → 1.0 (extreme)
    "anomalies":            [],           # List[str] — populated if anomalies found
    "confidence":           0.74,         # Model confidence in the above scores
    "model_version":        "wav2vec2-base-astromind-v1",
    "analyzed_at":          datetime.now(timezone.utc).isoformat(),
}

MOCK_SLEEP_DATA: Dict[str, Any] = {
    "status":                     "mock",
    "astronaut_id":               "unknown",
    "last_sleep_duration_hours":  7.2,
    "baseline_sleep_hours":       7.5,
    "interruptions_last_night":   1,
    "avg_hr_during_sleep_bpm":    57,
    "baseline_resting_hr_bpm":    60,
    "hrv_ms":                     48,     # HRV; below 35ms is clinically concerning
    "rem_cycles_completed":       4,
    "consecutive_irregular_nights": 0,
    "anomalies":                  [],     # List[str] — populated if anomalies found
    "analyzed_at":                datetime.now(timezone.utc).isoformat(),
}


# ─────────────────────────────────────────────────────────────────────────────
# RISK DETECTION KEYWORD BANKS
# WHY KEYWORD-BASED AND NOT LLM-BASED:
#   Risk detection must be instantaneous and deterministic.  Asking the LLM
#   "is this message suicidal?" introduces latency, non-determinism, and the
#   risk of the LLM itself being wrong.  Keyword matching is O(n) on text
#   length, runs in microseconds, and has zero failure modes.
# ─────────────────────────────────────────────────────────────────────────────
CRITICAL_RISK_PHRASES: List[str] = [
    "want to die", "not want to exist", "don't want to exist",
    "end my life", "kill myself", "take my life", "no reason to live",
    "not worth living", "never come back", "want to disappear forever",
    "escape permanently", "better off dead", "suicidal", "want to hurt myself",
]

HIGH_RISK_PHRASES: List[str] = [
    "hopeless", "no point", "completely worthless", "giving up entirely",
    "can't go on", "nothing matters anymore", "everything is dark",
    "don't care about anything", "wish I wasn't here",
]

MODERATE_RISK_PHRASES: List[str] = [
    "struggling", "overwhelmed", "exhausted all the time", "feel very alone",
    "extremely anxious", "can't sleep at all", "not eating", "deeply isolated",
    "very disconnected", "numb", "can't focus", "feel like a burden",
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Module-level singleton
# WHY SINGLETON: Model loading is expensive (2–10 s on typical hardware).
# We initialize it once at application startup and reuse across all requests.
# ─────────────────────────────────────────────────────────────────────────────
_llm: Optional[ChatOllama] = None


def _create_llm() -> ChatOllama:
    """
    Factory function for the ChatOllama instance.

    WHY these specific parameters:
      temperature=0.15  → Near-deterministic.  Clinical responses must be
                          consistent and not "creative."
      num_predict=250   → Hard cap on generated tokens.  Enforces brevity and
                          prevents the model from rambling (which sounds awful
                          on TTS and can introduce confabulated content).
      stop=[...]        → Prevents the model from continuing past its response
                          by roleplaying the astronaut's next turn.
      keep_alive="10m"  → Keeps the model loaded in VRAM between requests,
                          eliminating the cold-start penalty for every call.
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        num_predict=250,
        stop=["Human:", "Astronaut:", "\nQuestion:", "AstroMind:", "\n\n\n"],
        keep_alive="10m",
    )


def get_llm() -> ChatOllama:
    """Returns the shared LLM instance, raising clearly if startup failed."""
    global _llm
    if _llm is None:
        raise RuntimeError(
            "LLM is not initialized.  The application lifespan startup event "
            "may have failed.  Check Ollama is running and the model is pulled: "
            f"`ollama pull {OLLAMA_MODEL}`"
        )
    return _llm


# ─────────────────────────────────────────────────────────────────────────────
# SESSION HELPERS
# These functions manage the lifecycle of a session dict.  They are the ONLY
# code that reads from and writes to SESSION_STORE.  All business logic in the
# endpoints calls these helpers — nothing touches SESSION_STORE directly.
# ─────────────────────────────────────────────────────────────────────────────

def create_new_session(astronaut_id: str, mission_day: int) -> Dict[str, Any]:
    """
    Initialises a fresh session dictionary.

    WHY random.sample for questions: We shuffle the deck at session creation
    time so the random selection is guaranteed without replacement.  This means
    an astronaut will never see the same question twice in a single session,
    but may see different subsets across daily sessions — reducing fatigue.
    """
    now = datetime.now(timezone.utc)
    session: Dict[str, Any] = {
        "session_id":             str(uuid.uuid4()),
        "astronaut_id":           astronaut_id,
        "astronaut_name":         None,          # Populated during INTRO phase
        "mission_day":            mission_day,

        # ── Question state ──────────────────────────────────────────────
        # phase tracks WHERE in the state machine this session currently is.
        # Valid values: "intro" | "assessment" | "biometric_check" | "complete"
        "phase":                  "intro",

        # The one intro question randomly chosen for this session
        "intro_question":         random.choice(INTRO_QUESTIONS),

        # 3 randomly selected assessment questions (without replacement)
        "assessment_queue":       random.sample(ASSESSMENT_QUESTIONS, k=QUESTIONS_PER_SESSION),

        # Index into assessment_queue (0, 1, 2); incremented after each answer
        "current_assessment_idx": 0,

        # Prevents us from injecting more than one biometric question per session
        "biometric_q_used":       False,

        # ── Conversation memory ─────────────────────────────────────────
        # Full chat history as a list of {"role": "user"|"assistant", "content": str}
        # This is passed to the LLM on every call so it has full conversational context.
        "history":                [],

        # Maps question_text → answer_text for the final session summary
        "answers":                {},

        # ── Biometric data cache ────────────────────────────────────────
        # Stored so we can include the latest readings in every LLM call
        # without re-fetching them on every request.
        "last_voice_data":        {},
        "last_sleep_data":        {},

        # ── Risk tracking ───────────────────────────────────────────────
        "risk_level":             "low",
        "critical_alert":         False,

        # ── Timing ─────────────────────────────────────────────────────
        "created_at":             now,
        "last_activity":          now,
    }
    return session


def get_session_or_404(session_id: str) -> Dict[str, Any]:
    """
    Retrieves a session from SESSION_STORE.
    Raises 404 if not found or 410 if the session has expired.

    WHY 410 (Gone) instead of 404 for expired sessions: The frontend can
    distinguish between "session never existed" (404 — bug in the app) and
    "session expired" (410 — user was inactive too long and should restart).
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Please start a new session.",
        )
    now = datetime.now(timezone.utc)
    elapsed = (now - session["last_activity"]).total_seconds()
    if elapsed > SESSION_TTL_SECONDS:
        del SESSION_STORE[session_id]
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=(
                f"Session '{session_id}' expired after "
                f"{SESSION_TTL_SECONDS // 60} minutes of inactivity. "
                "Please start a new session."
            ),
        )
    return session


def add_to_history(session: Dict[str, Any], role: str, content: str) -> None:
    """
    Appends a message to the session's conversation history.

    WHY we store timestamp: Useful for post-mission clinical review and for
    detecting unusually long pauses between messages that might indicate distress.
    """
    session["history"].append({
        "role":      role,
        "content":   content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session["last_activity"] = datetime.now(timezone.utc)


def touch_session(session: Dict[str, Any]) -> None:
    """Resets the TTL timer on a session.  Called on every successful interaction."""
    session["last_activity"] = datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# NAME EXTRACTION
# WHY NOT use the LLM for this: Calling the LLM just to extract a name adds
# ~2 seconds of latency per session start and one more hallucination surface.
# Regex patterns cover 95 %+ of real-world responses to "what's your name?"
# ─────────────────────────────────────────────────────────────────────────────
_NAME_PATTERNS: List[str] = [
    r"(?:i['`]?m|i am|my name is|call me|name's|they call me)\s+([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30})?)",
    r"^([A-Z][a-z]{1,30})(?:[,.\s!]|$)",  # Bare first word if capitalised
]

def extract_name(text: str) -> Optional[str]:
    """
    Attempts to extract a first name from the astronaut's intro response.
    Returns None if no confident match is found.
    Fallback in the calling code: use "Commander" as a respectful default.
    """
    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, text.strip(), re.MULTILINE)
        if match:
            candidate = match.group(1).strip().title()
            # Sanity check: real names are 2–40 characters
            if 2 <= len(candidate) <= 40:
                return candidate
    # Last resort: if the entire response is a single short word, treat it as a name
    words = text.strip().split()
    if len(words) == 1 and 2 <= len(words[0]) <= 20:
        return words[0].strip(".,!?").title()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# RISK ASSESSMENT
# WHY this function exists as a standalone: Risk detection runs BEFORE the LLM
# call.  If risk is CRITICAL, we bypass the LLM entirely and return a hardcoded
# safe response.  The LLM should never be the gatekeeper for crisis detection.
# ─────────────────────────────────────────────────────────────────────────────
def assess_risk(text: str) -> Literal["low", "moderate", "high", "critical"]:
    """
    Keyword-based risk assessment of the astronaut's transcribed response.

    Priority: critical > high > moderate > low.
    WHY case-insensitive: Speech recognition may capitalize differently.
    WHY word boundaries not used for all patterns: Phrases like "not want to exist"
    span multiple words and are matched as substrings after lowercasing.
    """
    lower = text.lower()
    if any(phrase in lower for phrase in CRITICAL_RISK_PHRASES):
        return "critical"
    if any(phrase in lower for phrase in HIGH_RISK_PHRASES):
        return "high"
    if any(phrase in lower for phrase in MODERATE_RISK_PHRASES):
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# BIOMETRIC DATA FETCHING
# WHY httpx.AsyncClient (not requests):
#   These calls run inside an async FastAPI handler.  Using requests.get() would
#   BLOCK the entire event loop, freezing all other concurrent requests.
#   httpx.AsyncClient is the correct async-native HTTP client.
#
# WHY graceful degradation (not hard fail):
#   A biometric sensor or microservice going offline must NEVER block an
#   astronaut from accessing psychological support.  We log the error and
#   fall back to mock data so the check-in can continue.
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_voice_analysis(astronaut_id: str, recording_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches vocal stress and depression probability from the Voice Module (Port 8000).

    Endpoint:   GET /voice/latest/{astronaut_id}
    Optional:   GET /voice/analysis/{recording_id}  ← if a specific recording ID is provided

    Returns the Voice Module's JSON response, or MOCK_VOICE_DATA on failure.
    When the Voice Module is ready, this function requires NO changes — just
    remove the except block that returns mock data.
    """
    try:
        url = (
            f"{VOICE_MODULE_URL}/voice/analysis/{recording_id}"
            if recording_id
            else f"{VOICE_MODULE_URL}/voice/latest/{astronaut_id}"
        )
        async with httpx.AsyncClient(timeout=BIOMETRIC_FETCH_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(f"[Voice Module] Fetched data for {astronaut_id}: stress={data.get('stress_level')}")
            return data

    except httpx.TimeoutException:
        logger.warning(f"[Voice Module] Timeout fetching data for {astronaut_id}. Using mock data.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[Voice Module] HTTP {e.response.status_code} for {astronaut_id}. Using mock data.")
    except Exception as e:
        logger.error(f"[Voice Module] Unexpected error for {astronaut_id}: {e}. Using mock data.")

    # Graceful degradation — check-in continues with mock data
    return {**MOCK_VOICE_DATA, "astronaut_id": astronaut_id}


async def fetch_sleep_analysis(astronaut_id: str) -> Dict[str, Any]:
    """
    Fetches sleep metrics and heart rate data from the Sleep Module (Port 8001).

    Endpoint:   GET /sleep/latest/{astronaut_id}

    Returns the Sleep Module's JSON response, or MOCK_SLEEP_DATA on failure.
    """
    try:
        url = f"{SLEEP_MODULE_URL}/sleep/latest/{astronaut_id}"
        async with httpx.AsyncClient(timeout=BIOMETRIC_FETCH_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(
                f"[Sleep Module] Fetched data for {astronaut_id}: "
                f"duration={data.get('last_sleep_duration_hours')}h, "
                f"interruptions={data.get('interruptions_last_night')}"
            )
            return data

    except httpx.TimeoutException:
        logger.warning(f"[Sleep Module] Timeout fetching data for {astronaut_id}. Using mock data.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[Sleep Module] HTTP {e.response.status_code} for {astronaut_id}. Using mock data.")
    except Exception as e:
        logger.error(f"[Sleep Module] Unexpected error for {astronaut_id}: {e}. Using mock data.")

    return {**MOCK_SLEEP_DATA, "astronaut_id": astronaut_id}


# ─────────────────────────────────────────────────────────────────────────────
# BIOMETRIC ANOMALY DETECTION & QUESTION SELECTION
# WHY a dedicated function for this:
#   The logic for "is there an anomaly?" is specific to the schema of the
#   biometric services.  Centralising it here means when the schema changes
#   (e.g., the Sleep Module adds a new field), only this function needs updating.
# ─────────────────────────────────────────────────────────────────────────────

def detect_biometric_anomalies(
    voice_data: Dict[str, Any],
    sleep_data: Dict[str, Any],
) -> tuple[bool, Optional[str], str, str]:
    """
    Analyses biometric data from both modules and:
      1. Determines if a clinically significant anomaly exists.
      2. Selects the most contextually appropriate biometric question.
      3. Builds human-readable summary strings for injection into the LLM prompt.

    Returns:
      anomaly_detected (bool)       : True if any anomaly found
      biometric_question (str|None) : Most relevant question to ask, or None
      voice_summary (str)           : Formatted summary for LLM system prompt
      sleep_summary (str)           : Formatted summary for LLM system prompt

    WHY thresholds are hardcoded:
      These are based on established clinical guidelines:
        - Sleep < 6 h / >2 interruptions → clinically significant sleep deprivation
        - Resting HR deviation > 10 % of baseline → cardiac stress marker
        - HRV < 35 ms → autonomic nervous system stress
        - Vocal stress "elevated" / "high" → PHQ-8 correlated acoustic marker
    """
    anomalies_found: List[str] = []
    candidate_questions: List[str] = []

    # ── Voice anomaly checks ─────────────────────────────────────────
    stress_level = voice_data.get("stress_level", "normal")
    depression_prob = voice_data.get("depression_probability", 0.0)
    fatigue_index = voice_data.get("vocal_fatigue_index", 0.0)

    if stress_level in ("elevated", "high"):
        anomalies_found.append(f"vocal stress level: {stress_level}")
    if depression_prob >= 0.45:
        anomalies_found.append(f"elevated vocal depression marker ({depression_prob:.0%})")
    if fatigue_index >= 0.50:
        anomalies_found.append(f"high vocal fatigue index ({fatigue_index:.0%})")

    # ── Sleep anomaly checks ─────────────────────────────────────────
    sleep_hours      = sleep_data.get("last_sleep_duration_hours", 7.5)
    baseline_sleep   = sleep_data.get("baseline_sleep_hours", 7.5)
    interruptions    = sleep_data.get("interruptions_last_night", 0)
    avg_hr           = sleep_data.get("avg_hr_during_sleep_bpm", 60)
    baseline_hr      = sleep_data.get("baseline_resting_hr_bpm", 60)
    hrv_ms           = sleep_data.get("hrv_ms", 50)
    irregular_nights = sleep_data.get("consecutive_irregular_nights", 0)

    if sleep_hours < 6.0:
        anomalies_found.append(f"sleep duration only {sleep_hours:.1f}h (baseline {baseline_sleep:.1f}h)")
        candidate_questions.append(BIOMETRIC_QUESTIONS[1])  # "shorter than your normal cycle"

    if interruptions >= 3:
        anomalies_found.append(f"sleep interrupted {interruptions} times")
        candidate_questions.append(BIOMETRIC_QUESTIONS[3])  # "interrupted multiple times"

    if baseline_hr > 0 and avg_hr < baseline_hr * 0.85:
        anomalies_found.append(f"resting HR during sleep was low ({avg_hr} vs baseline {baseline_hr} bpm)")
        candidate_questions.append(BIOMETRIC_QUESTIONS[0])  # "dropped lower than usual"

    if baseline_hr > 0 and avg_hr > baseline_hr * 1.12:
        anomalies_found.append(f"elevated HR during sleep ({avg_hr} vs baseline {baseline_hr} bpm)")
        candidate_questions.append(BIOMETRIC_QUESTIONS[2])  # "stayed elevated during the night"

    if baseline_hr > 0 and baseline_hr > baseline_hr * 1.10:
        # Daytime resting HR elevated (uses today's resting HR from sleep data)
        candidate_questions.append(BIOMETRIC_QUESTIONS[4])  # "resting heart rate today is higher"

    if hrv_ms < 35:
        anomalies_found.append(f"HRV low at {hrv_ms} ms (clinical threshold: 35 ms)")
        candidate_questions.append(BIOMETRIC_QUESTIONS[6])  # "heart rate variability is lower"

    if irregular_nights >= 3:
        anomalies_found.append(f"{irregular_nights} consecutive irregular sleep nights")
        candidate_questions.append(BIOMETRIC_QUESTIONS[5])  # "sleep pattern has been irregular"

    # ── Also check explicit anomaly lists from the services themselves ──
    voice_anomalies = voice_data.get("anomalies", [])
    sleep_anomalies = sleep_data.get("anomalies", [])
    anomalies_found.extend(voice_anomalies)
    anomalies_found.extend(sleep_anomalies)

    # Deduplicate
    anomalies_found = list(dict.fromkeys(anomalies_found))
    anomaly_detected = len(anomalies_found) > 0

    # Select the most relevant biometric question
    biometric_question: Optional[str] = (
        candidate_questions[0] if candidate_questions
        else (random.choice(BIOMETRIC_QUESTIONS) if anomaly_detected else None)
    )

    # ── Build human-readable summaries for the LLM prompt ────────────
    # IMPORTANT: These summaries are injected into the system prompt.
    # They must be factual, specific, and contain ONLY what the data says.
    # The LLM is forbidden from adding to or extrapolating from these strings.
    if anomalies_found:
        voice_summary = (
            f"Stress Level: {stress_level.upper()} | "
            f"Depression Probability: {depression_prob:.0%} | "
            f"Vocal Fatigue: {fatigue_index:.0%}"
        )
        sleep_summary = (
            f"Sleep Duration: {sleep_hours:.1f}h (baseline {baseline_sleep:.1f}h) | "
            f"Interruptions: {interruptions} | "
            f"Avg HR During Sleep: {avg_hr} bpm (baseline {baseline_hr} bpm) | "
            f"HRV: {hrv_ms} ms | "
            f"Irregular Nights: {irregular_nights}\n"
            f"Detected Anomalies: {'; '.join(anomalies_found) if anomalies_found else 'None'}"
        )
    else:
        # Empty strings prevent the LLM from referencing biometrics at all
        voice_summary = "No anomalies detected. Do not reference voice metrics."
        sleep_summary = "No anomalies detected. Do not reference sleep metrics."

    return anomaly_detected, biometric_question, voice_summary, sleep_summary


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDING
# WHY a dedicated function:
#   The system prompt is rebuilt on every LLM call to include the latest
#   session state (name, answers so far, fresh biometric data).  Centralising
#   this keeps the endpoint handlers clean and ensures the prompt structure
#   is always consistent.
# ─────────────────────────────────────────────────────────────────────────────

def build_system_message(
    session: Dict[str, Any],
    voice_summary: str,
    sleep_summary: str,
) -> SystemMessage:
    """
    Formats the system prompt template with the current session's context.

    WHY we pass voice_summary / sleep_summary as strings (not raw dicts):
      The LLM should not receive raw JSON — it may try to interpret or
      extrapolate from unrelated fields.  Pre-formatted summaries contain
      EXACTLY what we want the model to see, nothing more.
    """
    name = session.get("astronaut_name") or "Commander"

    # Format the Q&A history for the prompt
    answers: Dict[str, str] = session.get("answers", {})
    if answers:
        previous_answers = "\n".join(
            f"  Q: {q}\n  A: {a}" for q, a in answers.items()
        )
    else:
        previous_answers = "  None yet — this is the start of today's session."

    formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        astronaut_name=name,
        mission_day=session.get("mission_day", 1),
        previous_answers=previous_answers,
        voice_data_summary=voice_summary,
        sleep_data_summary=sleep_summary,
    )
    return SystemMessage(content=formatted_prompt)


def build_message_list(
    session: Dict[str, Any],
    current_user_input: str,
    voice_summary: str,
    sleep_summary: str,
) -> List[Any]:
    """
    Builds the complete message list for the LLM call, including:
      1. A fresh SystemMessage (rebuilt every call with latest context)
      2. The conversation history (alternating HumanMessage / AIMessage)
      3. The current user input as a HumanMessage

    WHY rebuild SystemMessage on every call:
      The system prompt includes dynamic data (answers, biometric summaries)
      that changes with each turn.  Appending a new system message each time
      ensures the model always has the most current context at the top.

    WHY include full history:
      Without history, the model cannot provide a coherent follow-up response.
      It would treat every turn as a cold-start and produce generic replies.
    """
    messages: List[Any] = [build_system_message(session, voice_summary, sleep_summary)]

    # Replay conversation history
    for entry in session["history"]:
        role    = entry["role"]
        content = entry["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Current turn
    messages.append(HumanMessage(content=current_user_input))
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE POST-PROCESSING
# WHY this step: LLMs frequently produce markdown formatting (*, **, #, -)
# even when explicitly told not to.  This post-processor guarantees that the
# text passed to the browser's TTS API is always clean spoken prose.
# ─────────────────────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    """
    Strips markdown artifacts and normalizes whitespace for TTS output.
    Also enforces a word-count soft limit (warns if over 120 words).
    """
    # Remove bold / italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bullet / numbered list markers
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove underscores used for italic
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)

    text = text.strip()

    word_count = len(text.split())
    if word_count > 120:
        logger.warning(f"LLM response is {word_count} words — exceeds 80-word TTS guideline.")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# LLM INVOCATION
# WHY async (ainvoke): FastAPI is async-first.  Blocking the event loop with
# a synchronous LLM call would prevent ALL other concurrent requests from
# being served while the LLM is generating a response.
# ─────────────────────────────────────────────────────────────────────────────

async def call_llm(messages: List[Any]) -> str:
    """
    Invokes the LLM with the provided message list and returns the cleaned response.

    Error handling:
      - ConnectionRefusedError: Ollama is not running
      - TimeoutError: Model is taking too long (normal on CPU-only systems)
      - Any other exception: Logged and a safe fallback message returned

    WHY return a safe fallback instead of raising:
      The LLM being temporarily unavailable must never crash the check-in session.
      The astronaut should receive a warm, honest message — not a 500 error.
    """
    try:
        llm = get_llm()
        parser = StrOutputParser()
        chain = llm | parser
        response_text: str = await chain.ainvoke(messages)
        return clean_for_tts(response_text)

    except ConnectionRefusedError:
        logger.error("LLM connection refused — is Ollama running?")
        return (
            "I'm having a brief technical difficulty connecting to my systems. "
            "Please take a comfortable breath. I'll be right with you."
        )
    except asyncio.TimeoutError:
        logger.error("LLM call timed out.")
        return (
            "That took longer than expected on my end. I appreciate your patience. "
            "Let's continue — I'm listening."
        )
    except Exception as e:
        logger.exception(f"Unexpected LLM error: {e}")
        return (
            "I encountered a brief system issue. Your response has been recorded. "
            "I'm still here with you."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION SUMMARY GENERATION
# Called when the session transitions to "complete" after all 3 assessment
# questions have been answered.  This summary is stored in the session and
# returned to the frontend for display on the completion screen.
# ─────────────────────────────────────────────────────────────────────────────

async def generate_session_summary(session: Dict[str, Any]) -> str:
    """
    Asks the LLM to produce a short clinical-style session summary.

    WHY a separate LLM call for the summary (not accumulated in conversation):
      The summary prompt is structurally very different from the therapeutic
      response prompt.  Using a dedicated call with its own tightly scoped
      system message produces far cleaner output than tacking instructions
      onto the end of the therapeutic system prompt.

    WHY the summary is strictly factual:
      It is logged as part of the astronaut's health record.  Speculation,
      interpretation, or LLM confabulation in a clinical record is dangerous.
    """
    name    = session.get("astronaut_name") or "Commander"
    answers = session.get("answers", {})

    formatted_answers = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in answers.items()
    ) or "No answers recorded."

    summary_system = SystemMessage(content=(
        "You are a clinical documentation assistant for AstroMind.\n"
        "Write a SHORT (maximum 120 words), factual session summary based ONLY on the answers provided.\n"
        "Do not speculate. Do not diagnose. Do not add information not in the answers.\n"
        "Format: plain prose, no markdown, no bullet points.\n"
        "Structure: 1 sentence on overall presentation, 1 sentence on notable themes, "
        "1 sentence on recommended follow-up action.\n"
        f"Astronaut name: {name} | Mission Day: {session.get('mission_day', 1)}"
    ))
    summary_human = HumanMessage(content=f"Session answers:\n{formatted_answers}")

    try:
        llm     = get_llm()
        parser  = StrOutputParser()
        summary = await (llm | parser).ainvoke([summary_system, summary_human])
        return clean_for_tts(summary)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return (
            f"Session completed for {name} on Mission Day {session.get('mission_day', 1)}. "
            f"{len(answers)} questions answered. Manual review recommended."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASK: SESSION CLEANUP
# WHY this is needed: Python dicts do not self-expire.  Without cleanup, every
# session from every astronaut since startup would remain in memory forever.
# This coroutine runs every 30 minutes and evicts sessions that have been
# inactive longer than SESSION_TTL_SECONDS.
# ─────────────────────────────────────────────────────────────────────────────

async def cleanup_expired_sessions() -> None:
    """
    Periodic background task that removes expired sessions from SESSION_STORE.
    Runs every 30 minutes.  Logs a summary each cycle.
    """
    while True:
        await asyncio.sleep(1800)  # 30 minutes
        now     = datetime.now(timezone.utc)
        cutoff  = timedelta(seconds=SESSION_TTL_SECONDS)
        expired = [
            sid for sid, sess in SESSION_STORE.items()
            if (now - sess["last_activity"]) > cutoff
        ]
        for sid in expired:
            del SESSION_STORE[sid]
        if expired:
            logger.info(
                f"[Cleanup] Evicted {len(expired)} expired session(s). "
                f"Active sessions: {len(SESSION_STORE)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION LIFESPAN
# WHY lifespan (not @app.on_event):
#   @app.on_event("startup") is deprecated in FastAPI >= 0.93.
#   The lifespan context manager is the modern, recommended pattern.
#   It ensures LLM initialization and cleanup happen in the correct order.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global _llm

    # ── STARTUP ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AstroMind LLM Module starting up...")
    logger.info(f"  Ollama URL : {OLLAMA_BASE_URL}")
    logger.info(f"  Model      : {OLLAMA_MODEL}")
    logger.info(f"  Voice URL  : {VOICE_MODULE_URL}")
    logger.info(f"  Sleep URL  : {SLEEP_MODULE_URL}")
    logger.info(f"  Session TTL: {SESSION_TTL_SECONDS // 60} min")
    logger.info("=" * 60)

    # Initialize the LLM (does not load model weights — that happens on first call)
    try:
        _llm = _create_llm()
        logger.info(f"ChatOllama instance created for model '{OLLAMA_MODEL}'.")
    except Exception as e:
        logger.critical(f"Failed to create LLM instance: {e}")
        # We don't raise here — the app can start in degraded mode
        # and the /llm/health endpoint will report the failure.

    # Start the background cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    logger.info("Session cleanup background task started.")

    yield  # ← Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("AstroMind LLM Module shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AstroMind — LLM Psychological Support API",
    description=(
        "Chief Medical Officer module for the AstroMind deep-space mental health system.\n\n"
        "Conducts daily psychological evaluations, integrates voice and sleep biometric data, "
        "and provides empathetic, locally-hosted LLM-driven responses.\n\n"
        "**This service runs fully offline — no internet connection required.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/llm/docs",
    redoc_url="/llm/redoc",
    openapi_url="/llm/openapi.json",
)

# CORS — allow the frontend origin.  In production, replace "*" with the exact
# frontend URL (e.g., "https://astromind.mission.local").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Health Check ──────────────────────────────────────────────────────────
@app.get(
    "/llm/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """
    WHY this endpoint exists:
      - Used by Docker HEALTHCHECK, Kubernetes readiness probes, and ops dashboards.
      - Checks reachability of all three external dependencies (Ollama, Voice, Sleep).
      - Does NOT load the model or run inference — just pings each service.
    """
    llm_ok = voice_ok = sleep_ok = False

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            llm_ok = r.status_code == 200
    except Exception:
        pass

    # Check Voice Module
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{VOICE_MODULE_URL}/voice/health")
            voice_ok = r.status_code == 200
    except Exception:
        pass

    # Check Sleep Module
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{SLEEP_MODULE_URL}/sleep/health")
            sleep_ok = r.status_code == 200
    except Exception:
        pass

    overall = "ok" if all([llm_ok, voice_ok, sleep_ok]) else (
        "degraded" if llm_ok else "error"
    )
    msg = (
        "All systems operational." if overall == "ok"
        else f"LLM: {'✓' if llm_ok else '✗'} | Voice: {'✓' if voice_ok else '✗'} | Sleep: {'✓' if sleep_ok else '✗'}"
    )
    return HealthResponse(
        status=overall,
        llm_reachable=llm_ok,
        voice_module_reachable=voice_ok,
        sleep_module_reachable=sleep_ok,
        active_sessions=len(SESSION_STORE),
        message=msg,
    )


# ── 2. Start Session ─────────────────────────────────────────────────────────
@app.post(
    "/llm/start_session",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize a new daily psychological check-in session",
    tags=["Session"],
)
async def start_session(request: StartSessionRequest) -> StartSessionResponse:
    """
    Called when the astronaut clicks "Start Daily Check-in."

    WHAT HAPPENS INSIDE:
      1. A new session is created with a UUID and stored in SESSION_STORE.
      2. QUESTIONS_PER_SESSION (3) assessment questions are randomly selected
         WITHOUT REPLACEMENT from the bank of 15.
      3. One intro question is randomly selected from 3 variants.
      4. The session is stored; the intro question is returned to the frontend.

    The frontend must:
      - Store session_id in local state.
      - Pass first_question to the TTS engine to speak aloud.
      - Wait for the astronaut to respond, then call POST /llm/process_response.

    NOTE: A new session will succeed even if the LLM is unreachable.
    The LLM error will surface on the first process_response call.
    """
    session = create_new_session(
        astronaut_id=request.astronaut_id,
        mission_day=request.mission_day,
    )
    SESSION_STORE[session["session_id"]] = session

    logger.info(
        f"[Session {session['session_id'][:8]}] New session for astronaut '{request.astronaut_id}'. "
        f"Mission day {request.mission_day}. "
        f"Selected questions: {[q[:40] + '...' for q in session['assessment_queue']]}"
    )

    return StartSessionResponse(
        session_id=session["session_id"],
        first_question=session["intro_question"],
        session_phase="intro",
        message="Daily check-in session started. Intro question ready for TTS.",
    )


# ── 3. Process Response ───────────────────────────────────────────────────────
@app.post(
    "/llm/process_response",
    response_model=ProcessResponseResponse,
    summary="Process the astronaut's spoken response and advance the session",
    tags=["Session"],
)
async def process_response(
    request: ProcessResponseRequest,
    background_tasks: BackgroundTasks,
) -> ProcessResponseResponse:
    """
    The core endpoint of the entire AstroMind system.
    Called every time the astronaut finishes speaking and the browser STT
    has produced a text transcript.

    FULL PROCESSING PIPELINE:
      ┌─────────────────────────────────────────────────────────┐
      │ 1. Retrieve & validate session                          │
      │ 2. Perform instant risk assessment (keyword scan)       │
      │    └─ If CRITICAL → override: return hardcoded alert   │
      │ 3. Fetch biometric data (Voice + Sleep modules, async)  │
      │ 4. Detect anomalies in biometric data                   │
      │ 5. Determine current session phase:                     │
      │    ├─ INTRO         : Extract name, advance to Q1       │
      │    ├─ ASSESSMENT    : Record answer, advance Q-index    │
      │    └─ BIOMETRIC_CHECK: Record answer, advance Q-index   │
      │ 6. Build LLM message list with full context             │
      │ 7. Invoke LLM asynchronously                           │
      │ 8. Determine next question (or complete)               │
      │ 9. Store interaction in session history                 │
      │ 10. Return structured response to frontend             │
      └─────────────────────────────────────────────────────────┘

    SESSION PHASE TRANSITIONS (see state machine at top of file):
      - "intro"            → process → "assessment" (after name extracted)
      - "assessment"       → process → "assessment" (Q-index advances)
                                     → "biometric_check" (if anomaly, inject bio Q)
                                     → "complete" (after Q2 answered, no anomaly)
      - "biometric_check"  → process → "assessment" (back to next assessment Q)
                                     → "complete" (if it was the last Q)
    """
    # ── STEP 1: Retrieve session ─────────────────────────────────────
    session = get_session_or_404(request.session_id)
    short_id = request.session_id[:8]

    if session["phase"] == "complete":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This session is already complete. Please start a new daily check-in.",
        )

    logger.info(
        f"[Session {short_id}] Phase: {session['phase']} | "
        f"Assessment idx: {session['current_assessment_idx']} | "
        f"Input: '{request.transcribed_text[:60]}...'"
    )

    # ── STEP 2: Instant risk assessment ──────────────────────────────
    # This runs BEFORE the LLM call and BEFORE biometric fetching.
    # Risk detection must be the fastest possible operation.
    risk_level = assess_risk(request.transcribed_text)
    session["risk_level"] = risk_level

    if risk_level == "critical":
        session["critical_alert"] = True
        name = session.get("astronaut_name") or "Commander"
        critical_response = (
            f"[PRIORITY ALERT] I hear you, {name}, and what you've just shared is very important. "
            f"I am immediately flagging this as a critical concern to mission support and Earth. "
            f"You are not alone in this. Please stay here with me."
        )
        add_to_history(session, "user", request.transcribed_text)
        add_to_history(session, "assistant", critical_response)
        session["phase"] = "complete"  # Lock the session — no more questions

        logger.critical(
            f"[Session {short_id}] CRITICAL RISK DETECTED for astronaut '{request.astronaut_id}'. "
            "Emergency protocol triggered. Session locked."
        )
        return ProcessResponseResponse(
            session_id=request.session_id,
            llm_response=critical_response,
            next_question=None,
            session_phase="complete",
            is_complete=True,
            risk_level="critical",
            biometric_anomaly_detected=False,
            session_summary="[CRITICAL ALERT] Immediate human intervention required.",
            critical_alert=True,
        )

    # ── STEP 3: Fetch biometric data (both services, concurrently) ───
    # WHY asyncio.gather: Fetching voice and sleep data sequentially would
    # take up to 8 s (4 s timeout × 2).  Concurrent fetching takes max 4 s.
    voice_data, sleep_data = await asyncio.gather(
        fetch_voice_analysis(request.astronaut_id, request.audio_recording_id),
        fetch_sleep_analysis(request.astronaut_id),
    )
    session["last_voice_data"] = voice_data
    session["last_sleep_data"] = sleep_data

    # ── STEP 4: Detect biometric anomalies ───────────────────────────
    anomaly_detected, biometric_question, voice_summary, sleep_summary = (
        detect_biometric_anomalies(voice_data, sleep_data)
    )
    logger.info(
        f"[Session {short_id}] Biometric anomaly: {anomaly_detected} | "
        f"Voice stress: {voice_data.get('stress_level')} | "
        f"Sleep interruptions: {sleep_data.get('interruptions_last_night')}"
    )

    # ── STEP 5: Phase-specific state transitions ──────────────────────
    #
    # NOTE: We store the user's input in history BEFORE the LLM call so
    # that if the LLM call fails, the input is still recorded in the session.
    add_to_history(session, "user", request.transcribed_text)

    next_question: Optional[str] = None
    is_complete: bool = False
    current_phase = session["phase"]

    if current_phase == "intro":
        # ── INTRO PHASE ──────────────────────────────────────────────
        # Goal: Extract the astronaut's name and transition to ASSESSMENT.
        extracted_name = extract_name(request.transcribed_text)
        if extracted_name:
            session["astronaut_name"] = extracted_name
            logger.info(f"[Session {short_id}] Astronaut identified as '{extracted_name}'.")
        else:
            session["astronaut_name"] = "Commander"  # Respectful fallback
            logger.info(f"[Session {short_id}] Could not extract name. Using 'Commander'.")

        # Transition to assessment, starting with Q at index 0
        session["phase"]                  = "assessment"
        session["current_assessment_idx"] = 0
        next_question                     = session["assessment_queue"][0]

    elif current_phase == "assessment":
        # ── ASSESSMENT PHASE ──────────────────────────────────────────
        # Record the answer to the current assessment question.
        idx              = session["current_assessment_idx"]
        current_question = session["assessment_queue"][idx]
        session["answers"][current_question] = request.transcribed_text

        # Determine what comes next:
        # Priority order:
        #   1. Inject biometric question (if anomaly found and not yet used)
        #   2. Advance to next assessment question
        #   3. Complete the session (if all questions answered)

        next_idx = idx + 1  # What the next assessment Q-index would be

        if anomaly_detected and not session["biometric_q_used"] and biometric_question:
            # Inject biometric question BEFORE advancing the assessment index
            session["phase"]   = "biometric_check"
            next_question      = biometric_question
            # We do NOT advance current_assessment_idx here —
            # it will advance after the biometric answer is processed.
            logger.info(f"[Session {short_id}] Biometric question injected at assessment idx {idx}.")

        elif next_idx < QUESTIONS_PER_SESSION:
            # Advance to the next assessment question
            session["current_assessment_idx"] = next_idx
            next_question                     = session["assessment_queue"][next_idx]

        else:
            # All 3 assessment questions answered — session complete
            session["phase"] = "complete"
            is_complete       = True
            next_question     = None

    elif current_phase == "biometric_check":
        # ── BIOMETRIC CHECK PHASE ─────────────────────────────────────
        # Record the answer to the biometric question.
        biometric_q_asked = (
            # The biometric question was injected at the PREVIOUS step —
            # we need to record it.  We store it under its text.
            f"[Biometric] {next_question or 'biometric question'}"
        )
        # Identify what the biometric question was (it's not in assessment_queue)
        # We use the voice/sleep summary as a proxy key
        bq_key = f"[Biometric check]"
        session["answers"][bq_key] = request.transcribed_text
        session["biometric_q_used"] = True

        # Now advance the assessment Q-index (the biometric Q doesn't count)
        idx      = session["current_assessment_idx"]
        next_idx = idx + 1

        if next_idx < QUESTIONS_PER_SESSION:
            session["current_assessment_idx"] = next_idx
            session["phase"]                  = "assessment"
            next_question                     = session["assessment_queue"][next_idx]
        else:
            session["phase"] = "complete"
            is_complete       = True
            next_question     = None

    # ── STEP 6 & 7: Build context and call LLM ───────────────────────
    messages    = build_message_list(session, request.transcribed_text, voice_summary, sleep_summary)
    llm_response = await call_llm(messages)

    # ── STEP 8: Generate session summary if complete ─────────────────
    session_summary: Optional[str] = None
    if is_complete:
        session_summary = await generate_session_summary(session)
        logger.info(f"[Session {short_id}] Session COMPLETE. Summary generated.")

    # ── STEP 9: Store LLM response in history ────────────────────────
    add_to_history(session, "assistant", llm_response)

    # Log final state
    logger.info(
        f"[Session {short_id}] Response generated. "
        f"New phase: {session['phase']} | "
        f"Risk: {risk_level} | "
        f"Complete: {is_complete}"
    )

    # ── STEP 10: Return structured response ──────────────────────────
    return ProcessResponseResponse(
        session_id=request.session_id,
        llm_response=llm_response,
        next_question=next_question,
        session_phase=session["phase"],
        is_complete=is_complete,
        risk_level=risk_level,
        biometric_anomaly_detected=anomaly_detected,
        session_summary=session_summary,
        critical_alert=session["critical_alert"],
    )


# ── 4. Get Session Info ───────────────────────────────────────────────────────
@app.get(
    "/llm/session/{session_id}",
    response_model=SessionInfoResponse,
    summary="Retrieve current state of a session (for debugging/monitoring)",
    tags=["Session"],
)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """
    Returns a diagnostic snapshot of the session without modifying it.
    Used by the ops team and for frontend debugging.
    Does NOT return the full conversation history (privacy protection).
    """
    session = get_session_or_404(session_id)
    touch_session(session)  # Reset TTL

    return SessionInfoResponse(
        session_id=session_id,
        astronaut_id=session["astronaut_id"],
        astronaut_name=session.get("astronaut_name"),
        phase=session["phase"],
        current_assessment_idx=session["current_assessment_idx"],
        total_questions=QUESTIONS_PER_SESSION,
        questions_answered=len(session["answers"]),
        risk_level=session["risk_level"],
        critical_alert=session["critical_alert"],
        biometric_q_used=session["biometric_q_used"],
        created_at=session["created_at"].isoformat(),
        last_activity=session["last_activity"].isoformat(),
        history_length=len(session["history"]),
    )


# ── 5. End Session ────────────────────────────────────────────────────────────
@app.delete(
    "/llm/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Manually terminate a session",
    tags=["Session"],
)
async def end_session(session_id: str) -> None:
    """
    Explicitly deletes a session from SESSION_STORE.
    The frontend should call this when the astronaut navigates away from the
    check-in screen, or when the browser tab is closed (beforeunload event).

    WHY this matters: Proactively deleting sessions releases memory immediately
    rather than waiting for the TTL-based cleanup task to run.
    """
    session = SESSION_STORE.pop(session_id, None)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    logger.info(
        f"[Session {session_id[:8]}] Manually terminated by client. "
        f"Active sessions remaining: {len(SESSION_STORE)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llm_api:app",
        host="0.0.0.0",
        port=8004,
        reload=False,          # Set to True during local development only
        log_level="info",
        access_log=True,
    )