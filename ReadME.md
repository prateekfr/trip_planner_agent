# 🧭 Soumya — Trip Planner

**Soumya** is a lightweight trip-planning assistant AI agent that connects to your Gmail, extracts meeting/invite details, and produces decision-ready travel options (flights/trains/buses + hotels).  
Frontend is a minimal **Streamlit** app; backend is **FastAPI** with a **Gradio** chat mounted **Streamlit**. Route planning uses a small LangGraph pipeline and your LLM via **OpenRouter**.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1) Python environment](#1-python-environment)
  - [2) Google OAuth Client (Gmail)](#2-google-oauth-client-gmail)
  - [3) Environment variables](#3-environment-variables)
  - [4) Install dependencies](#4-install-dependencies)
- [Running Locally](#running-locally)
- [How to Use](#how-to-use)
- [Key Files & What They Do](#key-files--what-they-do)
- [APIs & Endpoints](#apis--endpoints)
- [Data & Persistence](#data--persistence)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [License](#license)

---

## Features

- **Google Sign-In (OAuth)** — read-only Gmail scope to fetch invites/meetings/itineraries.
- **Invite detection** — filters your inbox for relevant events (with/without ICS/PDF).
- **Extraction pipeline** — normalizes venue/time/city from email content.
- **Multi-modal routing** — searches **flights, trains, buses** + suggests **hotels** using **MCP servers**.
- **Smart ranking** — highlights **Cheapest**, **Fastest**, and **Value** options.
- **LLM itinerary** — concise output plus a **Suggested Plan** and **Cheapest Plan** block.
- **Polite chat agent** — named **Soumya**; understands “fetch invites”, “setting base location ”, “plan trip for invite N”, small talk, and rejects off-domain questions.

---

## Architecture

- **Frontend (Streamlit)**: minimal landing page with a “Sign in with Google” button; embeds the Gradio chat via iframe once authenticated.
- **Backend (FastAPI)**: OAuth , Gmail read APIs, invite filtering, chat orchestration.
- **Gradio Chat**: mounted at `/chat`; talks to the `ChatAgent` which drives the planner.
- **LangGraph Agent**: deterministic stages → MCP servers providers → rank → LLM summarize.
- **Providers**: thin wrappers that simulate/search/normalize flights, trains, buses, hotels though MCP.

---

## Folder Structure

> Root folder: `trip_planner`

```
trip_planner/
├─ data/
│  ├─ flight/
│  ├─ bus/
│  ├─ train/
│  ├─ hotel/
│  └─ response/             
├─ providers/
│  ├─ client.py
│  ├─ flights_server.py
│  ├─ buses_server.py
│  ├─ trains_server.py
│  └─ hotels_server.py
├─ src/
│  ├─ agent.py
│  ├─ extract.py
│  ├─ gmail_filter.py
│  └─ llm.py
├─ utils/
│  ├─ config.py
│  ├─ indian_cities.py
│  ├─ schemas.py 
│  └─ time.py
├─ venv/                    # (local virtualenv, optional)
├─ .env
├─ backend.py
├─ client_secret_web.json   # Google OAuth Web client JSON
├─ main_app.py              # Streamlit front-end
└─requirements.txt
```

> **Note:** The backend creates and writes to `data/responses/` by default for JSON dumps.  
> If you prefer `data/response/`, update the path in `backend.py` (see [Data & Persistence](#data--persistence)).

---

## Prerequisites

- **Python** 3.11 recommended
- **Google Cloud** project with **OAuth 2.0 Client ID** (type: *Web application*)
- **OpenRouter API key** (for `src/llm.py`)
- Local ports available:  
  - FastAPI backend (default `:8000`)  
  - Streamlit frontend (default `:8501`)

---

## Setup

### 1) Python environment

```bash
# From repository root
conda create -p venv python==3.11 -y
# Windows:
conda activate venv/
```

### 2) Google OAuth Client (Gmail)

1. Go to **Google Cloud Console → APIs & Services → Credentials**.
2. **Create credentials → OAuth client ID → Web application**.
3. Add an authorized redirect URI, e.g.:
   - `http://localhost:8000/oauth/callback`
4. Download the client JSON as **`client_secret_web.json`** into the repo root.
5. Enable **Gmail API** in the same project.
6. If testing with personal Gmail, you can stay in testing mode and add yourself as a test user.

### 3) Environment variables

Create a `.env` in the repo root:

```env
# OpenRouter (LLM via openrouter.ai)
OPENROUTER_API_KEY=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
# Flights (SerpAPI Google Flights engine)
SERPAPI_KEY=e3a02156b80f82fea7ac89783564872f14d8823c48c6bbfc3f56726a09a903e8
# Optional: where to store snapshots
DATA_ROOT=./data

# === App URLs ===
# Where your Streamlit app runs (dev):
FRONTEND_URL=http://localhost:8501
# Where your FastAPI backend runs (dev):
BACKEND_BASE_URL=http://localhost:8000
# Google OAuth redirect URI (must match in GCP OAuth client)
REDIRECT_URI=http://localhost:8000/oauth/callback

HTTP_REFERER=http://localhost
X_TITLE=trip-planner-agent
```

> The redirect URI must match one of the URIs in your `client_secret_web.json`.

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Locally

In 6 terminals:

**A) Backend (FastAPI + Gradio):**
```bash
# from repo root, venv activated
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

**B) Frontend (Streamlit):**
```bash
# from repo root, venv activated
streamlit run main_app.py --server.port 8501
```
**C) MCP Servers :**
```
python -m providers.bus_server
python -m providers.train_server
python -m providers.flight_server
python -m providers.hotel_server
```


Then open **http://127.0.0.1:8501** and click **Sign in with Google**.

---

## How to Use

In the chat (Soumya):

- `fetch my invites` → pulls recent invitations from Gmail
- `set my base to Indore` (or *“I am currently in Pune”*) → sets your origin city
- `plan trip for invite 2` → plans against invite #2 (must fetch invites first)

The agent will:
1. Parse the invite (destination, meeting time, venue text).
2. Search flights/trains/buses + hotels.
3. Rank **Cheapest / Fastest / Value**.
4. Return a compact itinerary and append:
   - **Soumya’s Suggested Plan** (Value route + a good-rated, affordable hotel)
   - **Cheapest Plan** (Cheapest route + cheapest hotel)

---

## Key Files & What They Do

- **`backend.py`**
  - OAuth (`/oauth/login`, `/oauth/callback`)
  - Gmail search/read endpoints
  - Invite pipeline (`/invites/fetch`)
  - Gradio chat mounted at `/chat`
  - Saves itinerary summaries to `data/responses/`

- **`main_app.py`**
  - Streamlit landing page
  - Google sign-in button
  - Embeds Gradio chat iframe when `sid` is present

- **`src/agent.py`**
  - LangGraph pipeline: constraints → Agent MCP tools → ranking → LLM itinerary
  - ChatAgent: intent parsing, small talk, guardrails, planner orchestration

- **`src/gmail_filter.py`**
  - Builds Gmail query and filters raw Gmail messages to likely invites

- **`src/extract.py`**
  - Extracts structured trip facts (destination city, times, venue text)

- **`providers/client.py`** & **`providers/*_server.py`**
  - Client for flight/train/bus/hotel search MCP servers of LLM tool calls

- **`utils/*`**
  - Helpers: config, city data, time utilities, schemas

---

## APIs & Endpoints

- **OAuth**
  - `GET /oauth/login` → redirect to Google
  - `GET /oauth/callback` → sets session ID, redirects to `FRONTEND_URL?sid=...`

- **Session/User**
  - `GET /me` (Header: `X-Session-ID`) → returns `{ email }`

- **Gmail**
  - `GET /gmail/search?q=...` (Header: `X-Session-ID`)
  - `GET /gmail/read?message_id=...` (Header: `X-Session-ID`)

- **Invites**
  - `GET /invites/fetch` (Header: `X-Session-ID`)  
    Query params: `newer_than_days`, `require_ics`, `extra_and`, `max_results`

- **Chat**
  - Mounted Gradio UI at **`/chat`** (expects `?sid=...` in the iframe URL)

---

## Data & Persistence

- **Responses (itinerary summaries)**  
  The backend writes structured JSON files to:

```
trip_planner/data/responses/<sid>_YYYYMMDD_HHMMSS.json
```

If you prefer the singular `data/response/`:
- Update the save path in `backend.py` inside `chat_fn`:
  ```python
  resp_dir = BASE_DIR / "data" / "response"
  ```
- Ensure the directory exists (code already creates it if missing).

---

## Customization

- **Rename the agent**: change `SOUMYA_*` constants in `src/agent.py`.
- **LLM model**: set `DEFAULT_MODEL` or edit `src/llm.py` (uses `OPENROUTER_API_KEY`).
- **UI width/height**: tweak CSS in `backend.py` (`CSS = """..."""`) and iframe height in `main_app.py`.
- **City → IATA map**: extend `_IATA` in `src/agent.py` or reference `utils/indian_cities.py`.
- **Routing logic**: adjust `_score_value`, thresholds in `_pick_cost_effective`, or itinerary prompts.

---

## Troubleshooting

**No JSON files saved**
- Make sure your interaction reached the **PLAN_TRIP** path (only that path creates a `summary`).
- Check server logs:
  - You should see `[chat_fn] out keys: ['reply', 'summary']` and `[save] wrote ...`.
- Verify `.env` URLs: `BACKEND_BASE_URL`, `FRONTEND_URL`, `REDIRECT_URI`.
- Confirm `client_secret_web.json` redirect URIs include your `REDIRECT_URI`.
- Ensure `data/` is writeable.

**“Missing client_secret_web.json” error**
- Download your Web OAuth client JSON to repo root.
- Ensure it has a top-level `"web"` key.

**OAuth redirect mismatch**
- Update the OAuth client’s **Authorized redirect URIs** to exactly match your `REDIRECT_URI`.

**It keeps asking for base city**
- Say: `set my base to Indore` (or `I am currently in Pune`).
- The agent validates against a known city/IATA map.

**Out-of-domain questions (e.g., “who is Virat Kohli?”)**
- Soumya is scoped to trip planning and will respond with a guardrail message.

---

## Security Notes

- Gmail scope is **read-only**.
- Do **not** commit `client_secret_web.json` or `.env` to version control.
- Sessions are kept in memory (`SESSION_DB`) for local dev only — replace with a persistent store for production.
- Consider rate limiting and proper CORS in production.

---

## Future Improvements

- Support for cafes or restraunts for food and uber/ola for cabs.
- Support to address colliding meetings
- Currently supports indian cities only, scope for international
- Support for a finance server for currency exchange
- Make the user also make request for custom tripfacts besides the emails extracted

---