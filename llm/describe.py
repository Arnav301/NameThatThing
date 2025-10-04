from __future__ import annotations

from typing import List, Dict, Optional
import os

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Gemini client is optional based on env var presence
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
_gemini_model = None
if _GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_GEMINI_KEY)
        # Lightweight, fast model suitable for short text summaries
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        _gemini_model = None


def llm_available() -> bool:
    return bool(_GEMINI_KEY and _gemini_model)


def summarize_scene(object_counts: Dict[str, int], extra_context: Optional[str] = None) -> str:
    """Return a natural language summary of the scene.

    If Gemini is configured, uses a short prompt; otherwise a heuristic fallback.
    """
    if llm_available():
        objects_line = ", ".join([f"{k} x{v}" for k, v in sorted(object_counts.items())]) or "no notable objects"
        prompt = (
            "You are a concise vision assistant. Given a list of detected objects with counts, "
            "produce a short, camera-like description of the scene in 2-3 sentences. "
            "Avoid hallucinating objects not listed.\n\n"
            f"Objects: {objects_line}\n"
        )
        if extra_context:
            prompt += f"Notes: {extra_context}\n"

        try:
            resp = _gemini_model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                # Fallback in case of SDK shape differences
                text = resp.candidates[0].content.parts[0].text if resp.candidates[0].content.parts else ""
            return (text or "").strip()
        except Exception:
            pass

    # Fallback caption
    if not object_counts:
        return "No prominent objects detected."
    parts = [f"{name} ({cnt})" for name, cnt in sorted(object_counts.items(), key=lambda x: (-x[1], x[0]))]
    base = "Detected: " + ", ".join(parts) + "."
    if extra_context:
        base += f" Notes: {extra_context}."
    return base


def describe_objects_brief(object_counts: Dict[str, int]) -> str:
    """Optionally provide a second compact explanation of what the objects imply."""
    if llm_available():
        prompt = (
            "Given object categories and counts, provide a single-sentence insight about the likely scene, "
            "without introducing new objects.\n\n"
            + "\n".join([f"- {k}: {v}" for k, v in sorted(object_counts.items())])
        )
        try:
            resp = _gemini_model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                text = resp.candidates[0].content.parts[0].text if resp.candidates[0].content.parts else ""
            return (text or "").strip()
        except Exception:
            pass

    # Fallback
    if not object_counts:
        return ""
    return "This scene likely contains the listed items in everyday context."
