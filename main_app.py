# import os, requests
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv(override=False)
# BACKEND = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

# st.set_page_config(page_title="Trip Planner", layout="wide")

# qp = st.query_params
# if "sid" not in st.session_state and "sid" in qp:
#     st.session_state.sid = qp["sid"]

# GLASS_CSS = """
# <style>
# .center-wrap {display:flex; align-items:center; justify-content:center; height:70vh;}
# .glass-btn {
#   display:inline-block;
#   padding:18px 28px;          /* was ~14px 22px — bump up */
#   border-radius:18px;         /* slightly rounder */
#   min-width: 280px;           /* wider tap target */
#   text-align:center;          /* center the label */
#   backdrop-filter: blur(10px);
#   -webkit-backdrop-filter: blur(10px);
#   background: rgba(255,255,255,0.12);
#   border: 1px solid rgba(255,255,255,0.25);
#   box-shadow: 0 10px 30px rgba(0,0,0,0.15);
#   text-decoration:none;
#   font-weight:700;
#   font-size:18px;             /* bigger text */
# }
# .glass-btn:hover {background: rgba(255,255,255,0.20);}

# /* Flashy gradient title (responsive) */
# .app-heading {
#   font-size: clamp(56px, 8vw, 104px);  /* scales with viewport */
#   line-height: 1.05;
#   font-weight: 900;
#   letter-spacing: .5px;
#   margin: 0;
#   text-align: center;

#   /* gradient text */
#   background: linear-gradient(90deg, #6ee7f9, #8b5cf6, #f472b6);
#   -webkit-background-clip: text;
#   background-clip: text;
#   color: transparent;

#   /* optional subtle glow for contrast */
#   text-shadow: 0 0 12px rgba(139, 92, 246, 0.25);
# }
# </style>
# """

# if not st.session_state.get("sid"):
#     st.markdown(GLASS_CSS, unsafe_allow_html=True)
#     st.markdown(
#         f"""
#         <div class="center-wrap" style="flex-direction:column; gap:28px;">
#           <h1 class="app-heading">Trip Planner</h1>
#           <a class="glass-btn" href="{BACKEND}/oauth/login">🔐 Sign in with Google</a>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# else:
#     chat_url = f"{BACKEND}/chat?sid={st.session_state.sid}"
#     st.markdown(
#         f'<iframe src="{chat_url}" style="width:100%; height:75vh; border:none; border-radius:16px;"></iframe>',
#         unsafe_allow_html=True,
#     )



import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=False)
BACKEND = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Trip Planner", layout="wide")

# --- Global CSS: remove Streamlit chrome and scrolling, reclaim all space ---
st.markdown("""
<style>
/* Remove default Streamlit padding and header/footer */
.block-container { padding: 0 !important; }
header, footer { visibility: hidden; }

/* Make the outer app take full height and prevent double scrollbars */
html, body, .stApp { height: 100%; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Read sid from query string (if provided)
# Use the stable API across Streamlit versions:
try:
    qp = st.query_params  # new API (>=1.32)
except Exception:
    qp = st.experimental_get_query_params()  # fallback

if "sid" not in st.session_state and "sid" in qp:
    # qp may be a list-like in older API; handle both cases
    sid_val = qp["sid"][0] if isinstance(qp.get("sid"), (list, tuple)) else qp["sid"]
    st.session_state.sid = sid_val

GLASS_CSS = """
<style>
.center-wrap {display:flex; align-items:center; justify-content:center; height:100vh;}
.glass-btn {
  display:inline-block;
  padding:18px 28px;
  border-radius:18px;
  min-width: 280px;
  text-align:center;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.25);
  box-shadow: 0 10px 30px rgba(0,0,0,0.15);
  text-decoration:none;
  font-weight:700;
  font-size:18px;
}
.glass-btn:hover {background: rgba(255,255,255,0.20);}

/* Flashy gradient title (responsive) */
.app-heading {
  font-size: clamp(56px, 8vw, 104px);
  line-height: 1.05;
  font-weight: 900;
  letter-spacing: .5px;
  margin: 0;
  text-align: center;
  background: linear-gradient(90deg, #6ee7f9, #8b5cf6, #f472b6);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 0 12px rgba(139, 92, 246, 0.25);
}
</style>
"""

if not st.session_state.get("sid"):
    st.markdown(GLASS_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="center-wrap" style="flex-direction:column; gap:28px;">
          <h1 class="app-heading">Trip Planner</h1>
          <a class="glass-btn" href="{BACKEND}/oauth/login">🔐 Sign in with Google</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    chat_url = f"{BACKEND}/chat?sid={st.session_state.sid}"
    # Optional: if your backend supports embed mode to trim inner paddings:
    # chat_url += "&embed=true"

    # Render the chat as a full-viewport iframe directly in the page (no extra component iframe)
    st.markdown(f"""
    <style>
      .fullpage-iframe {{
        position: fixed;
        inset: 0;               /* left:0; top:0; right:0; bottom:0 */
        width: 100%;
        height: 100vh;
        border: 0;
      }}
      @supports (height: 100dvh) {{
        .fullpage-iframe {{ height: 100dvh; }}
      }}
    </style>
    <iframe class="fullpage-iframe" src="{chat_url}" allow="clipboard-write *"></iframe>
    """, unsafe_allow_html=True)
