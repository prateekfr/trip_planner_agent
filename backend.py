from __future__ import annotations
import os, json, secrets, traceback, re
from typing import Any, Dict, List, Optional
from base64 import urlsafe_b64decode
from fastapi import FastAPI, Request, HTTPException, Header, Query
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request as GReq
from src.gmail_filter import build_query, filter_messages
from pathlib import Path
from src.extract import extract_trip_facts
from src.agent import run_planner, ChatAgent
from datetime import datetime
import gradio as gr

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/oauth/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")

app = FastAPI(title="Trip Planner Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DB: Dict[str, Dict[str, str]] = {}
CHAT_STATE: Dict[str, Dict[str, Any]] = {}  

def _load_client_config() -> dict:
    path = "client_secret_web.json"
    if not os.path.exists(path):
        raise RuntimeError(
            "Missing client_secret_web.json. Create a **Web application** OAuth client in Google Cloud "
            "and download the JSON next to backend.py."
        )
    cfg = json.load(open(path, "r", encoding="utf-8"))
    if "web" not in cfg:
        raise RuntimeError("Invalid client_secret_web.json: expected top-level key 'web' (Web OAuth client).")
    redirects = cfg["web"].get("redirect_uris", [])
    if REDIRECT_URI not in redirects:
        raise RuntimeError(
            f"REDIRECT_URI mismatch.\n"
            f"- .env REDIRECT_URI = {REDIRECT_URI}\n"
            f"- client_secret_web.json redirect_uris = {redirects}\n"
            f"Add the .env REDIRECT_URI to the OAuth client in Google Cloud."
        )
    return cfg

def _flow() -> Flow:
    cfg = _load_client_config()
    fl = Flow.from_client_config(cfg, scopes=SCOPES)
    fl.redirect_uri = REDIRECT_URI
    return fl

def _gmail_service(creds: Credentials):
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

def _creds_from_sid(sid: str) -> Credentials:
    data = SESSION_DB.get(sid)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in.")
    info = json.loads(data["creds_json"])
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(GReq())
        SESSION_DB[sid]["creds_json"] = creds.to_json()
    return creds

def _extract_plain_text_from_payload(part: Dict[str, Any]) -> str:
    text = ""
    mime = part.get("mimeType", "")
    body = part.get("body", {}) or {}
    data = body.get("data")
    if mime == "text/plain" and data:
        try:
            text += urlsafe_b64decode(data).decode(errors="ignore")
        except Exception:
            pass
    for p in part.get("parts", []) or []:
        text += _extract_plain_text_from_payload(p)
    if not text and mime == "text/html" and data:
        try:
            html = urlsafe_b64decode(data).decode(errors="ignore")
            text = re.sub("<[^<]+?>", " ", html)
        except Exception:
            pass
    return text[:8000]

@app.get("/health")
def health():
    try:
        cfg = _load_client_config()
        return {
            "status": "ok",
            "redirect_uri_env": REDIRECT_URI,
            "frontend_url_env": FRONTEND_URL,
            "has_web_block": "web" in cfg,
            "redirects_in_file": cfg["web"].get("redirect_uris", []),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error", "details": str(e)})

@app.get("/oauth/login")
def oauth_login():
    try:
        fl = _flow()
        auth_url, state = fl.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return RedirectResponse(auth_url)
    except Exception as e:
        tb = traceback.format_exc()
        return PlainTextResponse(f"/oauth/login error:\n{e}\n\n{tb}", status_code=500)

@app.get("/oauth/callback")
def oauth_callback(request: Request):
    try:
        fl = _flow()
        fl.fetch_token(authorization_response=str(request.url))
        creds = fl.credentials
        service = _gmail_service(creds)
        profile = service.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress")

        sid = secrets.token_urlsafe(24)
        SESSION_DB[sid] = {"creds_json": creds.to_json(), "email": email}

        target = f"{FRONTEND_URL}?sid={sid}"
        return RedirectResponse(target)
    except Exception as e:
        tb = traceback.format_exc()
        return PlainTextResponse(f"/oauth/callback error:\n{e}\n\n{tb}", status_code=500)

@app.get("/me")
def me(x_session_id: Optional[str] = Header(None)):
    if not x_session_id or x_session_id not in SESSION_DB:
        raise HTTPException(status_code=401, detail="Not signed in.")
    entry = SESSION_DB[x_session_id]
    return {"email": entry["email"]}

@app.get("/gmail/search")
def gmail_search(
    q: str = Query("", description="Gmail search query"),
    max_results: int = Query(20, ge=1, le=50),
    x_session_id: Optional[str] = Header(None),
):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="Missing X-Session-ID header.")
    creds = _creds_from_sid(x_session_id)
    service = _gmail_service(creds)

    if not q.strip():
        q = (
            'subject:(invite OR invitation OR meeting OR itinerary OR booking OR ticket OR travel) '
            'OR (has:attachment filename:(ics OR pdf) AND (meeting OR itinerary OR booking)) '
            'OR ("flight" OR "train" OR "bus" OR "hotel" OR "venue")'
        )
    resp = service.users().messages().list(userId="me", q=q, maxResults=max_results).execute()
    ids = [m["id"] for m in resp.get("messages", [])]
    return {"message_ids": ids}

@app.get("/gmail/read")
def gmail_read(
    message_id: str = Query(...),
    x_session_id: Optional[str] = Header(None),
):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="Missing X-Session-ID header.")
    creds = _creds_from_sid(x_session_id)
    service = _gmail_service(creds)

    m = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in (m.get("payload", {}) or {}).get("headers", [])}
    subject = headers.get("subject", "(no subject)")
    sender  = headers.get("from", "")
    snippet = m.get("snippet", "") or ""
    body    = _extract_plain_text_from_payload(m.get("payload", {}) or {})
    internal = m.get("internalDate")

    return JSONResponse({
        "id": message_id,
        "subject": subject,
        "from": sender,
        "snippet": snippet,
        "body": body,
        "internalDate": internal,
    })

def _api_me(sid: str) -> Dict[str, Any]:
    if sid not in SESSION_DB:
        raise HTTPException(401, "Not signed in.")
    return {"email": SESSION_DB[sid]["email"]}

def _api_gmail_search(sid: str, q: str, max_results: int) -> List[str]:
    creds = _creds_from_sid(sid)
    service = _gmail_service(creds)
    resp = service.users().messages().list(userId="me", q=q, maxResults=max_results).execute()
    return [m["id"] for m in resp.get("messages", [])]

def _api_gmail_read(sid: str, message_id: str) -> Dict[str, Any]:
    r = gmail_read(message_id=message_id, x_session_id=sid)
    return json.loads(r.body.decode("utf-8")) if hasattr(r, "body") else r

def fetch_invites_for_session(
    sid: str,
    *,
    newer_than_days: int = 45,
    require_ics: bool = False,
    extra_and: str = "",
    max_results: int = 50,
) -> list[dict]:
    q = build_query(newer_than_days=newer_than_days, require_ics=require_ics, extra_and=extra_and)
    ids = _api_gmail_search(sid, q, max_results=max_results)
    messages: List[Dict[str, Any]] = []
    for mid in ids:
        try:
            messages.append(_api_gmail_read(sid, mid))
        except Exception:
            continue
    return filter_messages(messages)

@app.get("/invites/fetch")
def invites_fetch(
    newer_than_days: int = 45,
    require_ics: bool = False,
    extra_and: str = "",
    max_results: int = 50,
    x_session_id: str = Header(alias="X-Session-ID"),
):
    invites = fetch_invites_for_session(
        x_session_id,
        newer_than_days=newer_than_days,
        require_ics=require_ics,
        extra_and=extra_and,
        max_results=max_results,
    )
    return {"count": len(invites), "invites": invites}

HELP_TEXT = (
    "Hi, I'm **Soumya**, your trip planner.\n"
    "I can fetch your invites, set your base city, and plan an itinerary for a selected invite.\n\n"
    "Try:\n"
    "• **fetch my invites** — pull recent invitation emails from Gmail\n"
    "• **set my base to Indore** — store your base city for planning\n"
    "• **plan trip for invite 2** — plan the itinerary for invitation #2\n"
)

_chat_agent = ChatAgent()

def _format_invite_preview(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No invitations found."
    lines = []
    for i, m in enumerate(items[:10], 1):
        subj = (m.get("subject") or "(no subject)").strip()
        snip = (m.get("snippet") or "").strip()
        reasons = ", ".join(m.get("filter_reasons", []))
        lines.append(f"**{i}.** {subj}\n\n> {snip[:180]}{'…' if len(snip)>180 else ''}\n_Reasons_: {reasons}\n")
    return "\n".join(lines)

def _plan_for_message(user_email: str, base_city: Optional[str], msg: Dict[str, Any]) -> Dict[str, Any]:
    facts = extract_trip_facts(user_email=user_email, base_city=base_city, emails=[msg])
    return run_planner(facts[0])

CSS = """
/* Limit overall width and center the app */
.gradio-container {
  max-width: min(980px, 94vw) !important;  /* 👈 shrink width */
  margin: 0 auto !important;
  padding: 12px 10px !important;
}

/* Optional: slightly smaller chat area minimum height */
#bigbox .wrap { min-height: 420px; }

/* Reduce headline top/bottom gap if you like */
.gr-markdown h2 { margin: 8px 0 12px !important; }
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS, fill_height=False, analytics_enabled=False) as demo:
    gr.Markdown("## 🧭 Soumya — Trip Planner (Chat)")
    status_md   = gr.Markdown()
    sid_state   = gr.State(value=None)
    email_state = gr.State(value=None)

    def _on_load(request: gr.Request):
        sid = request.query_params.get("sid")
        if not sid:
            return ("Please sign in first (no sid found in URL).", None, None)
        who = _api_me(sid)
        CHAT_STATE.setdefault(sid, {"base_city": None, "invites": [], "history": [], "greeted": False})
        return (f"Signed in as **{who.get('email','unknown')}**.\n\n" + HELP_TEXT,
                sid, who.get("email"))

    demo.load(_on_load, inputs=None, outputs=[status_md, sid_state, email_state])

    def chat_fn(user_msg: str, history, sid, email):
        if not sid or sid not in SESSION_DB:
            return "Please sign in first (no sid captured).", sid, email

        def _fetch_invites():
            return fetch_invites_for_session(
                sid,
                newer_than_days=45,
                require_ics=False,
                extra_and="",
                max_results=50,
            )

        def _extract_trip_facts(user_email: str, base_city: Optional[str], emails: List[Dict[str, Any]]):
            return extract_trip_facts(user_email=user_email, base_city=base_city, emails=emails)

        session = CHAT_STATE.setdefault(sid, {"base_city": None, "invites": [], "history": [], "greeted": False})

        out = _chat_agent.process(
            user_msg=user_msg,
            session=session,
            user_email=email,
            fetch_invites_fn=_fetch_invites,
            plan_fn=lambda tf: run_planner(tf),
            extract_trip_facts_fn=_extract_trip_facts,
        )

        if "invites" in out and isinstance(out["invites"], list):
            session["invites"] = out["invites"]
        try:
            if isinstance(out, dict):
                print("[chat_fn] out keys:", list(out.keys()))
            else:
                print("[chat_fn] out is not a dict:", type(out))

            summary = out.get("summary") if isinstance(out, dict) else None
            if isinstance(summary, dict):
                resp_dir = (BASE_DIR / "data" / "responses")
                resp_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                fname = resp_dir / f"{sid}_{ts}.json"
                fname.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[save] wrote {fname}")
            else:
                print(f"[save] no 'summary' dict to save (type={type(summary)})")
        except Exception as e:
            print("[save] ERROR while writing summary:", repr(e))

        if "invites" in out and out.get("reply", "").lower().startswith("here are your recent invites"):
            preview = _format_invite_preview(out["invites"])
            return out["reply"] + "\n\n" + preview, sid, email

        return out.get("reply", "Done."), sid, email

    gr.ChatInterface(
        fn=chat_fn,
        title="Soumya — Trip Planner",
        description="Sign in, then chat to fetch invites and plan your trip.",
        chatbot=gr.Chatbot(height=300, type="messages", elem_id="bigbox"),
        additional_inputs=[sid_state, email_state],
        additional_outputs=[sid_state, email_state],
        submit_btn="Send",
        stop_btn="Stop",
        autofocus=True,
        cache_examples=False,
    )

gr.mount_gradio_app(app, demo, path="/chat")