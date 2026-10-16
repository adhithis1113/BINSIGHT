import base64, glob, io, json, math, os, random, time
from datetime import datetime

from auth import (
    authenticate,
    create_citizen_account,
)

from streamlit_js_eval import get_geolocation

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="BINSIGHT", page_icon="♻️", layout="wide")

# ---------------- PASTEL GLOW HOVER EFFECT ----------------
def inject_pastel_hover():
    st.markdown(
        """
        <style>
        /* Glowing border for dashboard layout cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(190, 170, 255, 0.25);
            border-radius: 22px;
            transition: border-color 0.3s ease,
                        box-shadow 0.3s ease,
                        transform 0.3s ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(220, 190, 255, 1);
            box-shadow:
                0 0 8px rgba(220, 190, 255, 0.65),
                0 0 20px rgba(190, 160, 255, 0.4),
                0 0 32px rgba(160, 190, 255, 0.2);
            transform: translateY(-3px);
        }

        /* Glowing hover effect for buttons */
        div[data-testid="stButton"] > button {
            border-radius: 12px;
            transition: border-color 0.25s ease,
                        box-shadow 0.25s ease;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #d8b4fe;
            box-shadow:
                0 0 8px rgba(216, 180, 254, 0.6),
                0 0 18px rgba(196, 181, 253, 0.35);
        }

        @media (prefers-reduced-motion: reduce) {
            div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stButton"] > button {
                transition: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- SETTINGS (edit these) ----------------
APP_NAME = "BINSIGHT"
TEAM_NAME = "Your Team Name"
HELPLINE = "+91 90000 00000"  # sample number, replace with a real one
HELP_EMAIL = "help@binsight.example"  # sample email, replace with a real one
FILL_PER_ITEM = 5
SCALE_MODE = "auto"  # how the model expects pixel values: "auto", "minus1_1", "zero_1" or "raw"
WASTE_TYPES = ["plastic", "paper", "organic", "metal", "glass", "cardboard", "e-waste", "medicine", "other"]
WASTE_COLORS = ["#3B82F6", "#FACC15", "#22C55E", "#A855F7", "#06B6D4", "#A16207", "#F97316", "#EF4444", "#9CA3AF"]
RECYCLABLE = ("plastic", "paper", "metal", "glass", "cardboard")
PICKUP_THRESHOLD = 70
TRUCK_L_PER_KM, CO2_KG_PER_L, ROAD_FACTOR = 0.5, 2.68, 1.3
DEPOT_COORD = (12.9716, 77.5946)
COMPLAINT_CSV, COMPLAINT_DIR = "complaints.csv", "complaints"
CATEGORIES = ["Overflowing bin", "Litter on street", "Illegal dumping", "Missed collection", "Other"]

DISPOSAL_TIPS = {
    "plastic": "Rinse it and put it in the dry-waste/recycling bin. Avoid single-use plastic.",
    "paper": "Keep it dry and clean, then put it in the dry-waste/recycling bin.",
    "organic": "Put it in the wet-waste bin. It can be composted into manure.",
    "metal": "Put it in the dry-waste bin or give it to a scrap collector. Metal is highly recyclable.",
    "glass": "Wrap broken glass safely before binning it, or give clean bottles to a recycler.",
    "cardboard": "Flatten it, keep it dry, and put it with dry waste or give it to a recycler.",
    "e-waste": "Never put it in the normal bin. Hand it to an authorised e-waste collection point.",
    "medicine": "Do not flush it or put it in the wet-waste bin. Return expired medicines to a pharmacy take-back or an authorised biomedical-waste collection point.",
    "other": "If it cannot be recycled or composted, put it in the general (reject) waste bin.",
}

# Rename zones (wards) ONLY here.
ZONE_COORDS = {
    "Zone 1 - Indiranagar": (12.9784, 77.6408),
    "Zone 2 - Koramangala": (12.9352, 77.6245),
    "Zone 3 - Whitefield": (12.9698, 77.7500),
    "Zone 4 - Jayanagar": (12.9250, 77.5938),
    "Zone 5 - HSR Layout": (12.9116, 77.6389),
    "Zone 6 - Malleshwaram": (13.0035, 77.5710),
}

MODEL_CANDIDATES = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

ABOUT = ("BINSIGHT is a smart waste management web app. A photo of waste is classified (plastic, paper, organic, metal) by a "
         "computer vision model. Each detection updates a live digital twin of city wards, which predicts overflow days ahead. "
         "Generative AI turns the data into advice, and the app plans an optimized pickup route.")
FAQ = {
    "What is BINSIGHT?": ABOUT,
    "How does waste detection work?": "Upload or capture a photo on the Detect Waste screen. Our trained model labels it, you can correct it, then add it to a ward.",
    "How do I report a dirty area?": "Open the Complaint Desk, fill in your details and the address, attach photos or videos as proof, and submit. You get a complaint ID to track it.",
    "Is the bin data real?": "This is a prototype. Fill levels are simulated; real IoT sensors in bins would feed the same numbers.",
    "How is the pickup route chosen?": "Only wards at 70% or more are visited, always going to the nearest unvisited ward next, starting from the depot.",
    "Is my personal data safe?": "Your details are used only to follow up on your complaint and are never shown on the public page.",
}

# icons on the home screen: (key, icon, title, description)
SCREENS = [
    ("overview", "📊", "Overview", "City status at a glance"),
    ("detect", "📷", "Detect Waste", "Photo to waste type"),
    ("map", "🗺️", "City Map", "Wards on a 2D or 3D map"),
    ("zones", "🏙️", "Ward Status", "Cards, gauges, waste mix"),
    ("alerts", "🔔", "Live Alerts", "Overflow and activity feed"),
    ("route", "🚛", "Route Planner", "Optimized pickup route"),
    ("impact", "🌱", "Impact", "Distance, CO₂, recycling"),
    ("history", "📈", "History", "Fill-level trends"),
    ("advice", "🤖", "AI Advice", "Generative AI tips"),
    ("chat", "🗣️", "Talk to the Twin", "Ask in Kannada or English"),
    ("vision", "🔬", "Gemini Vision", "Compare AI with our model"),
    ("report", "📄", "Report", "Summary for officials"),
    ("transparency", "🔍", "Public View", "What citizens can see"),
    ("complaint", "📝", "Complaint Desk", "Report an issue with proof"),
    ("help", "🛟", "Help Desk", "Ask about BINSIGHT"),
    ("offline", "📴", "Offline Mode", "Queue and sync demo"),
    ("official_complaints", "📋", "Manage Citizen Complaints",
     "Review and manage citizen-submitted complaints"),
]

   
PASTELS = ["#a78bfa", "#f9a8d4", "#6ee7b7", "#7dd3fc", "#fcd34d", "#fca5a5", "#c4b5fd", "#86efac"]
# The 3 big icons on the dashboard: (key, icon, title, description, sub-feature keys)
GROUPS = [
    ("monitor", "👁️", "Monitor", "Detect waste and watch every ward live", ["overview", "detect", "map", "zones", "alerts"]),
    ("operate", "🚛", "Operate", "Plan pickups, measure impact, report", ["route", "impact", "history", "report", "offline"]),
    ("ai", "🤖", "AI Assistant", "Chat, voice questions, and smart waste recommendations", ["chat"]),
]
# Set a feature to False to hide it everywhere (tile, sidebar switch) without deleting code.
FEATURES = {"vision": True, "offline": True}
GROUPS = [(g[0], g[1], g[2], g[3], [k for k in g[4] if FEATURES.get(k, True)]) for g in GROUPS]
PARENT = {k: g[0] for g in GROUPS for k in g[4]}  # sub-feature -> its group
_TILE_KEYS = [k for k, *_ in SCREENS] + [f"g_{g[0]}" for g in GROUPS]
TILE_CSS = "".join(f".st-key-tile_{k}:hover{{box-shadow:0 0 26px {PASTELS[i % len(PASTELS)]}; transform:translateY(-4px);}}"
                   for i, k in enumerate(_TILE_KEYS))

st.markdown(
    """
<style>
.hero {padding:44px 40px; border-radius:24px; margin:10px 0 18px 0; text-align:center;
    background:linear-gradient(120deg,rgba(196,181,253,.22),rgba(251,207,232,.18),rgba(167,243,208,.22));
    border:1px solid rgba(139,92,246,.25);}
.hero-title {font-size:4rem; font-weight:900; letter-spacing:.18em;
    background:linear-gradient(90deg,#8b5cf6,#ec4899,#10b981,#0ea5e9,#8b5cf6); background-size:300% 100%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:gm 8s linear infinite;}
.hero-sub {opacity:.9; margin:10px auto 0 auto; font-size:1.15rem; max-width:760px;}
@keyframes gm {0%{background-position:0% 50%} 100%{background-position:300% 50%}}
.stButton > button, .stDownloadButton > button {border-radius:14px; border:1px solid #a78bfa; transition:all .25s;}
.stButton > button:hover, .stDownloadButton > button:hover {box-shadow:0 0 12px #a78bfa, 0 0 26px #f9a8d4; border-color:#f9a8d4; transform:translateY(-2px);}
[data-testid="stMetric"] {background:rgba(139,92,246,.07); border:1px solid rgba(139,92,246,.22); border-radius:16px; padding:16px; transition:all .25s;}
[data-testid="stMetric"]:hover {box-shadow:0 0 22px #7dd3fc; transform:translateY(-3px);}
[class*="st-key-tile_"] {border-radius:18px; transition:all .25s;}
.tile-icon {font-size:2.7rem; line-height:1.1;} .tile-title {font-weight:700; font-size:1.05rem; margin-top:4px;}
.tile-desc {font-size:.85rem; opacity:.75; min-height:36px; margin-bottom:8px;}
.step, .zcard {background:rgba(139,92,246,.07); border:1px solid rgba(139,92,246,.22); border-radius:16px; padding:16px; height:100%; transition:all .25s;}
.step:hover, .zcard:hover {box-shadow:0 0 22px #6ee7b7; transform:translateY(-3px);}
.zcard {margin-bottom:12px;} .zname {font-weight:700;} .zfill {font-size:1.9rem; font-weight:800;} .zmeta {font-size:.82rem; opacity:.75; margin-top:6px;}
.bar {height:8px; background:rgba(128,128,128,.25); border-radius:6px; margin-top:6px;} .bar > div {height:100%; border-radius:6px;}
.legend span {margin-right:18px; font-size:.9rem;} .dot {display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px;}
</style>
""" + f"<style>{TILE_CSS}</style>",
    unsafe_allow_html=True,
)


def status(f):
    return "🔴 Overflow risk" if f >= 90 else "🟠 Getting full" if f >= 70 else "🟢 OK"


def status_color(f):
    return "#FF3B3B" if f >= 90 else "#FFA500" if f >= 70 else "#00B050"


def status_rgb(f):
    return [255, 59, 59, 210] if f >= 90 else [255, 165, 0, 210] if f >= 70 else [0, 176, 80, 210]


def plain_status(f):
    return "✅ Bins have space" if f < 70 else "🟡 Collection due soon" if f < 90 else "🔴 Collection needed urgently"


# ---------------- TWIN STORE, EVENTS, HISTORY, OFFLINE QUEUE ----------------
def create_initial_twin():
    return pd.DataFrame({
        "Zone": list(ZONE_COORDS.keys()),
        "Fill %": [30, 55, 80, 20, 65, 45],
        "Daily growth %": [8, 12, 10, 6, 9, 7],
        **{w: [0] * 6 for w in WASTE_TYPES},
        "Last updated": ["-"] * 6,
    })


def log_event(icon, text):
    st.session_state.events = (st.session_state.events + [(datetime.now().strftime("%H:%M:%S"), icon, text)])[-30:]


def record_history():
    row = {"Reading": len(st.session_state.history) + 1}
    for _, r in st.session_state.bins.iterrows():
        row[r["Zone"]] = int(r["Fill %"])
    st.session_state.history = (st.session_state.history + [row])[-200:]


def reset_twin():
    st.session_state.bins = create_initial_twin()
    st.session_state.history, st.session_state.events, st.session_state.queue = [], [], []
    record_history()


def go_to(screen):
    st.session_state.screen = screen


if "bins" not in st.session_state:
    reset_twin()
if "history" not in st.session_state:
    st.session_state.history = []
    record_history()
for k, v in {"events": [], "queue": [], "last_tick": 0.0, "screen": "landing"}.items():
    st.session_state.setdefault(k, v)
for w in WASTE_TYPES:
    if w not in st.session_state.bins.columns:
        st.session_state.bins[w] = 0

# ---------------- VISION CLASSIFIER (runs on the device, no internet needed) ----------------

# ---------------- VISION CLASSIFIER ----------------
MODEL_PATH = "keras_model_new.h5"
LABELS_PATH = "labels_new.txt"

MODEL_AVAILABLE = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(LABELS_PATH)
)

@st.cache_resource
@st.cache_resource
def load_my_model():
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(
           MODEL_PATH,
           compile=False
    )
    except Exception as e:
        st.error(f"Model loading failed: {type(e).__name__}: {e}")
        raise
    
    with open(LABELS_PATH, "r") as f:
        labels = []
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)

            # Supports labels such as "0 cardboard"
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].lower())
            else:
                labels.append(line.lower())

    output_shape = model.output_shape
    n_out = output_shape[-1]

    if n_out != len(labels):
        raise ValueError(
            f"Model predicts {n_out} classes, "
            f"but {LABELS_PATH} contains {len(labels)} labels."
        )

    return model, labels, (224, 224)


@st.cache_data(show_spinner=False)
def classify_bytes(data):
    model, labels, size = load_my_model()

    img = ImageOps.fit(
        Image.open(io.BytesIO(data)).convert("RGB"),
        (size[1], size[0]),
        Image.Resampling.LANCZOS,
    )

    arr = np.asarray(img, dtype=np.float32)
    batch = arr[np.newaxis, ...]

    pred = model.predict(batch, verbose=0)[0]

    best = int(np.argmax(pred))
    return labels[best], float(pred[best])




# ---------------- UPDATE LOGIC / SIMULATION / IOT / OFFLINE ----------------
def set_fill(idx, new):
    bins = st.session_state.bins
    old, zone = int(bins.loc[idx, "Fill %"]), bins.loc[idx, "Zone"]
    new = max(0, min(100, int(new)))
    bins.loc[idx, "Fill %"] = new
    bins.loc[idx, "Last updated"] = datetime.now().strftime("%H:%M:%S")
    if old < 90 <= new:
        log_event("🔴", f"{zone} crossed 90% (overflow risk)")
    elif old < 70 <= new:
        log_event("🟠", f"{zone} is getting full ({new}%)")


def add_waste_to_zone(zone, waste_type):
    bins = st.session_state.bins
    idx = bins.index[bins["Zone"] == zone][0]
    if waste_type not in bins.columns:
        bins[waste_type] = 0
    bins.loc[idx, waste_type] += 1
    set_fill(idx, int(bins.loc[idx, "Fill %"]) + FILL_PER_ITEM)
    log_event("➕", f"{waste_type} detected and added to {zone}")
    record_history()


def sync_queue():
    n = len(st.session_state.queue)
    for z, lab, _ in st.session_state.queue:
        add_waste_to_zone(z, lab)
    st.session_state.queue = []
    log_event("🔄", f"Back online: synced {n} saved detection(s)")
    return n


def simulate(bins, days):
    future = bins.copy()
    future["Fill %"] = (future["Fill %"] + days * future["Daily growth %"]).clip(upper=100)
    return future


def sensor_tick():
    bins = st.session_state.bins
    for i in bins.index:
        set_fill(i, int(bins.loc[i, "Fill %"]) + random.randint(0, 3))
    record_history()


def collect_zones(zones):
    bins = st.session_state.bins
    for z in zones:
        set_fill(bins.index[bins["Zone"] == z][0], 5)
        log_event("🚛", f"{z} collected")
    record_history()


# ---------------- ROUTE + IMPACT ----------------
def haversine_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def route_length(order):
    pts = [DEPOT_COORD] + [ZONE_COORDS[z] for z in order] + [DEPOT_COORD]
    return sum(haversine_km(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) * ROAD_FACTOR


def nearest_neighbor_order(zones):
    remaining, cur, order = list(zones), DEPOT_COORD, []
    while remaining:
        nxt = min(remaining, key=lambda z: haversine_km(cur, ZONE_COORDS[z]))
        order.append(nxt)
        cur = ZONE_COORDS[nxt]
        remaining.remove(nxt)
    return order


def compute_impact(view):
    needs = view[view["Fill %"] >= PICKUP_THRESHOLD]["Zone"].tolist()
    order = nearest_neighbor_order(needs)
    naive_km = route_length(list(view["Zone"]))
    opt_km = route_length(order) if order else 0.0
    saved = max(0.0, naive_km - opt_km)
    bins = st.session_state.bins
    counts = {w: int(bins[w].sum()) for w in WASTE_TYPES if w in bins.columns}
    total = sum(counts.values())
    rec = sum(counts.get(w, 0) for w in RECYCLABLE)
    return {"order": order, "naive_km": naive_km, "opt_km": opt_km, "km_saved": saved,
            "skipped": len(view) - len(needs), "co2": saved * TRUCK_L_PER_KM * CO2_KG_PER_L,
            "rate": (rec / total * 100) if total else None, "items": total, "counts": counts}


# ---------------- COMPLAINTS ----------------
def load_complaints():
    return pd.read_csv(COMPLAINT_CSV, dtype=str).fillna("") if os.path.exists(COMPLAINT_CSV) else pd.DataFrame()


def save_complaint(name, phone, email, address, zone, category, desc, files):
    cid = f"BIN-{datetime.now():%y%m%d}-{random.randint(1000, 9999)}"
    folder = os.path.join(COMPLAINT_DIR, cid)
    os.makedirs(folder, exist_ok=True)
    names = []
    for i, f in enumerate(files or []):
        fn = f"{i + 1}_{os.path.basename(f.name)}"
        with open(os.path.join(folder, fn), "wb") as out:
            out.write(f.getvalue())
        names.append(fn)
    row = {"ID": cid, "Time": datetime.now().strftime("%d %b %Y %H:%M"), "Name": name, "Phone": phone, "Email": email,
           "Address": address, "Zone": zone, "Category": category, "Description": desc, "Files": "; ".join(names), "Status": "Received"}
    pd.concat([load_complaints(), pd.DataFrame([row])], ignore_index=True).to_csv(COMPLAINT_CSV, index=False)
    return cid


# ---------------- ADVICE / GEMINI / REPORT ----------------
def twin_text(view):
    return view[["Zone", "Fill %", "Status"] + WASTE_TYPES].to_string(index=False)

def report_ward_status(view):
    lines = [
        "### Ward Status Overview",
        "",
        "| Ward | Fill Level | Status | Waste Detected |",
        "|---|---:|---|---|",
    ]

    for _, row in view.iterrows():
        zone = row["Zone"]
        fill = int(row["Fill %"])
        ward_status = str(row["Status"])

        waste_counts = []
        for waste_type in WASTE_TYPES:
            if waste_type in view.columns:
                count = int(row[waste_type])
                if count > 0:
                    waste_counts.append(f"{waste_type}: {count}")

        waste_summary = ", ".join(waste_counts) if waste_counts else "No detections"

        lines.append(
            f"| {zone} | {fill}% | {ward_status} | {waste_summary} |"
        )

    return "\n".join(lines)


def ask_gemini(key, prompt, image=None, audio=None):
    if st.session_state.get("offline"):
        raise RuntimeError("Offline mode is on, so cloud AI is unavailable.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    parts = []
    if image is not None:
        parts.append(image)
    if audio is not None:
        parts.append(types.Part.from_bytes(data=audio, mime_type="audio/wav"))
    parts.append(prompt)
    contents = parts if len(parts) > 1 else prompt
    last = None
    for name in MODEL_CANDIDATES:
        try:
            return client.models.generate_content(model=name, contents=contents).text, name
        except Exception as e:
            last = e
    raise last


def speak_button(text, lang):
    components.html(
        f"""<button id="b" style="padding:8px 16px;border-radius:12px;border:1px solid #a78bfa;background:#ede9fe;color:#3b0764;cursor:pointer">🔊 Listen</button>
<script>const t={json.dumps(text)};document.getElementById('b').onclick=()=>{{speechSynthesis.cancel();
const u=new SpeechSynthesisUtterance(t);u.lang='{lang}';speechSynthesis.speak(u);}};</script>""", height=52)
def rule_based_advice(view, days):
    ranked = view.sort_values("Fill %", ascending=False)

    urgent = ranked[ranked["Fill %"] >= 90]
    soon = ranked[
        (ranked["Fill %"] >= 70) &
        (ranked["Fill %"] < 90)
    ]

    when = "now" if days == 0 else f"in {days} day(s)"
    lines = []

    if not urgent.empty:
        lines.append(
            f"**1. Collection priority:** Send trucks first to "
            f"{', '.join(urgent['Zone'].astype(str))}. "
            f"They are at overflow risk {when}."
        )
    elif not soon.empty:
        lines.append(
            f"**1. Collection priority:** Schedule the next pickup for "
            f"{', '.join(soon['Zone'].astype(str))}."
        )
    else:
        lines.append(
            "**1. Collection priority:** All zones are below 70%. "
            "No urgent pickup needed."
        )

    waste_columns = [w for w in WASTE_TYPES if w in view.columns]
    totals = view[waste_columns].sum() if waste_columns else None

    if totals is not None and not totals.empty and totals.sum() > 0:
        top = totals.idxmax()
        lines.append(
            f"**2. Segregation:** Most detected waste is **{top}** "
            f"({int(totals[top])} items). Consider adding dedicated "
            f"{top} bins."
        )
    else:
        lines.append(
            "**2. Segregation:** No waste detected yet. "
            "Add some photos to the twin first."
        )

    lines.append(
        f"**3. Route efficiency:** Visit the fullest zones first: "
        f"{' → '.join(ranked['Zone'].head(4).astype(str))}."
    )

    return "\n\n".join(lines)

def build_report(view, days, m):
    rate = f"{m['rate']:.0f}%" if m["rate"] is not None else "n/a (no items detected yet)"
    stops = " → ".join(m["order"]) if m["order"] else "No pickups needed"
    return f"""# BINSIGHT Report
Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}  |  Simulated days ahead: {days}

## Ward status
{report_ward_status(view)}

## Recommendations
{rule_based_advice(view, days)}

## Optimized route (nearest-neighbor, from depot)
{stops}
Route length: {m['opt_km']:.1f} km vs {m['naive_km']:.1f} km for visiting every ward.

## Estimated impact (assumptions in the app settings)
- Ward visits saved: {m['skipped']}
- Distance saved: {m['km_saved']:.1f} km
- CO2 avoided: {m['co2']:.1f} kg
- Recycling rate of detected waste: {rate}
"""


# ---------------- UI HELPERS ----------------
def zone_card(r):
    f, c = int(r["Fill %"]), status_color(int(r["Fill %"]))
    return (f'<div class="zcard" style="border-left:6px solid {c}"><div class="zname">{r["Zone"]}</div>'
            f'<div class="zfill" style="color:{c}">{f}%</div><div class="bar"><div style="width:{f}%;background:{c}"></div></div>'
            f'<div class="zmeta">{status(f)} · updated {r["Last updated"]}</div></div>')


LEGEND = ('<div class="legend"><span><i class="dot" style="background:#00B050"></i>Below 70%</span>'
          '<span><i class="dot" style="background:#FFA500"></i>70-89%</span>'
          '<span><i class="dot" style="background:#FF3B3B"></i>90%+ overflow risk</span></div>')


def draw_map(view, three_d):
    df = view.copy()
    df["lat"] = df["Zone"].map(lambda z: ZONE_COORDS[z][0])
    df["lon"] = df["Zone"].map(lambda z: ZONE_COORDS[z][1])
    df["Fill"] = df["Fill %"].astype(int)
    df["color"] = df["Fill"].apply(status_rgb)
    df["radius"] = 800 + df["Fill"] * 20
    df["short"] = df["Zone"].str.split(" - ").str[-1]
    cols = ["Zone", "short", "Fill", "Status", "lat", "lon", "color", "radius"]
    try:
        import pydeck as pdk

        if three_d:
            layers = [pdk.Layer("ColumnLayer", data=df[cols], get_position="[lon, lat]", get_elevation="Fill",
                                elevation_scale=80, radius=600, get_fill_color="color", extruded=True, pickable=True)]
            vs = pdk.ViewState(latitude=12.96, longitude=77.64, zoom=10.3, pitch=55)
        else:
            layers = [pdk.Layer("ScatterplotLayer", data=df[cols], get_position="[lon, lat]", get_radius="radius",
                                get_fill_color="color", pickable=True),
                      pdk.Layer("TextLayer", data=df[cols], get_position="[lon, lat]", get_text="short", get_size=15,
                                get_color=[45, 40, 70], get_alignment_baseline="'bottom'")]
            vs = pdk.ViewState(latitude=12.96, longitude=77.64, zoom=10.3)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=vs,
                                 tooltip={"html": "<b>{Zone}</b><br/>Fill: {Fill}%<br/>{Status}"}))
    except Exception:
        st.map(df[["lat", "lon"]])



def page_header(key):
    info = next((s for s in SCREENS if s[0] == key), None)

    if info is None:
        st.error(f"Page configuration not found for: {key}")
        if st.button("← Back to home", key=f"missing_page_back_{key}"):
            st.session_state.screen = "landing"
            st.rerun()
        return

    
    
    st.button(
        "← Back to Dashboard",
        key=f"back_{key}",
        on_click=go_to,
        args=("home",)
    )

    st.header(f"{info[1]} {info[2]}")
    st.caption(info[3])



   

def get_secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# ---------------- SIDEBAR (hidden on the landing page) ----------------
days, api_key, live = 0, "", False
if st.session_state.screen != "landing":
    with st.sidebar:
        st.header("⚙️ Controls")
        days = st.slider("🔮 Simulate days ahead", 0, 5, 0)
        api_key = get_secret_key()
        if FEATURES["offline"]:
            st.toggle("📴 Simulate network drop", key="offline")
        live = st.toggle("📡 Live sensor feed (mock IoT)")
        st.caption("Turn the live feed OFF while you upload photos.")
        if st.button("Reset city twin"):
            reset_twin()
            st.rerun()
    if not st.session_state.get("offline") and st.session_state.queue:
        n = sync_queue()
        st.session_state.msg = f"Back online: synced {n} saved detection(s)"

view = simulate(st.session_state.bins, days)
view["Status"] = view["Fill %"].apply(status)
at_risk = view[view["Fill %"] >= 90]
impact = compute_impact(view)
if "msg" in st.session_state:
    st.toast(st.session_state.pop("msg"), icon="✅")


# ======================= LANDING =======================
LANDING_CSS = """
<style>
.lp-hero {text-align:center; padding:28px 12px 8px 12px;}
.lp-pill {display:inline-block; padding:6px 16px; border-radius:999px; font-size:.9rem; font-weight:600;
    border:1px solid rgba(139,92,246,.35); background:rgba(139,92,246,.10); animation:lpFade .7s ease both;}
.lp-title {font-size:clamp(3rem,9vw,6rem); font-weight:900; letter-spacing:.14em; line-height:1.1; margin:14px 0 4px 0;
    background:linear-gradient(90deg,#8b5cf6,#ec4899,#10b981,#0ea5e9,#8b5cf6); background-size:300% 100%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:lpGrad 8s linear infinite;}
.lp-sub {font-size:clamp(1.3rem,3.2vw,2.1rem); font-weight:700;}
.lp-w {display:inline-block; opacity:0; animation:lpFade .6s ease forwards;}
.lp-desc {max-width:640px; margin:14px auto 6px auto; text-align:center; font-size:1.1rem; opacity:.85; animation:lpFade .8s ease .8s both;}
.st-key-get_started {margin-top:10px;}
.st-key-get_started button {padding:.75rem 2.4rem; font-size:1.05rem; font-weight:700; color:#fff; border:none;
    background:linear-gradient(90deg,#8b5cf6,#ec4899);}
.lp-marquee {overflow:hidden; margin-top:22px; padding:34px 0;
    -webkit-mask-image:linear-gradient(to bottom,transparent,black 20%,black 80%,transparent);
    mask-image:linear-gradient(to bottom,transparent,black 20%,black 80%,transparent);}
.lp-track {display:flex; gap:18px; width:max-content; animation:lpSlide 45s linear infinite;}
.lp-track:hover {animation-play-state:paused;}
.lp-card {position:relative; flex:0 0 auto; height:270px; aspect-ratio:3/4; border-radius:22px; overflow:hidden;
    box-shadow:0 8px 24px rgba(0,0,0,.18); display:flex; flex-direction:column; align-items:center; justify-content:center;}
.lp-card:nth-child(odd) {transform:rotate(-2deg);} .lp-card:nth-child(even) {transform:rotate(5deg);}
.lp-card img {width:100%; height:100%; object-fit:cover;}
.lp-emoji {font-size:3.4rem;} .lp-cap {margin-top:8px; font-weight:700; color:#3b2a5a;}
@keyframes lpSlide {from{transform:translateX(-50%)} to{transform:translateX(0)}}
@keyframes lpFade {from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none}}
@keyframes lpGrad {0%{background-position:0% 50%} 100%{background-position:300% 50%}}
</style>
"""

# Shown until you add your own photos to the "assets" folder
MARQUEE_FALLBACK = [("🌳", "Green Bengaluru", "#bbf7d0", "#86efac"), ("🏙️", "Clean streets", "#bae6fd", "#7dd3fc"),
                    ("♻️", "Recycle right", "#ddd6fe", "#c4b5fd"), ("🚛", "On-time pickups", "#fde68a", "#fcd34d"),
                    ("🌿", "Fresh air", "#a7f3d0", "#6ee7b7"), ("🏞️", "Clean lakes", "#fbcfe8", "#f9a8d4")]


def list_asset_files(folder="assets"):
    return sorted(p for ext in ("jpg", "jpeg", "png", "webp") for p in glob.glob(os.path.join(folder, f"*.{ext}")))


@st.cache_data(show_spinner=False)
def load_marquee_images(signature):
    out = []
    for p, _ in signature[:12]:
        try:
            img = ImageOps.fit(Image.open(p).convert("RGB"), (390, 520), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        except Exception:
            continue
    return out


def marquee_html():
    files = list_asset_files()
    imgs = load_marquee_images(tuple((p, os.path.getmtime(p)) for p in files))
    if imgs:
        cards = [f'<div class="lp-card"><img src="{src}" alt="Clean Bengaluru"/></div>' for src in imgs]
    else:
        cards = [f'<div class="lp-card" style="background:linear-gradient(160deg,{a},{b})"><div class="lp-emoji">{e}</div><div class="lp-cap">{t}</div></div>'
                 for e, t, a, b in MARQUEE_FALLBACK]
    base = cards * math.ceil(8 / len(cards))
    return '<div class="lp-marquee"><div class="lp-track">' + "".join(base * 2) + "</div></div>"


def fade_words(text, start=0.0):
    return "".join(f'<span class="lp-w" style="animation-delay:{start + i * 0.1:.1f}s">{w}&nbsp;</span>' for i, w in enumerate(text.split()))


def page_landing():
    menu, _ = st.columns([1, 6])
    with menu.popover("☰ Menu"):
        st.button("🛟 Help Desk", key="m_help", on_click=go_to, args=("help",))
        st.button("📝 Complaint Desk", key="m_complaint", on_click=go_to, args=("complaint",))
        st.button("🔍 Public View", key="m_public", on_click=go_to, args=("transparency",))
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="lp-hero"><div class="lp-pill">🌿 Clean City · Clean Bengaluru</div>'
        f'<div class="lp-title">{APP_NAME}</div>'
        f'<div class="lp-sub">{fade_words("See waste before it overflows", 0.3)}</div>'
        '<div class="lp-desc">BINSIGHT spots waste from a photo, mirrors every ward in a live digital twin, '
        'and plans the next pickup with AI, so Bengaluru stays clean.</div></div>', unsafe_allow_html=True)
   
    _, mid, _ = st.columns([2, 1, 2])

    for kw in (
        {"width": "stretch"},
        {"use_container_width": True},
        {}
    ):
        try:
            mid.button(
                "✨ Enter the Twin",
                key="get_started",
                on_click=go_to,
                args=("choose_login",),
                **kw
            )
            break
        except TypeError:
            continue

    st.markdown(marquee_html(), unsafe_allow_html=True)



# ======================= HOME (icon dashboard) =======================





def page_home():
    user = st.session_state.get("user", {})
    role = user.get("role", "citizen")

    st.button(
        "← Back to home",
        key="back_intro",
        on_click=go_to,
        args=("landing",),
    )

    st.title("What would you like to do?")
    st.caption("Choose a service to get started.")

    # Features available to each role
    citizen_features = {
        "detect", "map", "advice", "chat", "offline",
    }

    official_features = {
        "overview", "detect", "map", "zones", "alerts",
        "route", "impact", "history", "advice", "chat",
        "vision", "report", "transparency", "offline",
    }

    allowed_features = (
        official_features if role == "official"
        else citizen_features
    )

    # Official-only metrics
    if role == "official":
        k = st.columns(4)
        k[0].metric("Wards monitored", len(view))
        k[1].metric("Overflow-risk wards", len(at_risk))
        k[2].metric("Items detected", impact["items"])
        k[3].metric(
            "Distance saved",
            f"{impact['km_saved']:.1f} km",
        )

    # Citizen dashboard
    if role != "official":
        citizen_cards = [
            (
                "♻️",
                "Detect Waste",
                "Identify and classify waste.",
                "detect",
            ),
            (
                "🗺️",
                "City Map",
                "Explore waste locations on the map.",
                "map",
            ),
           
(
    "🤖",
    "AI Assistant",
    "Chat, ask questions by voice, and get smart waste-management recommendations.",
    "chat",
),

        ]

        for start in range(0, len(citizen_cards), 2):
            row_cards = citizen_cards[start:start + 2]
            cols = st.columns(2, gap="medium")

            for col, (icon, title, description, screen) in zip(
                cols, row_cards
            ):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="padding: 8px; min-height: 145px;">
                                <div style="
                                    font-size: 3rem;
                                    margin-bottom: 12px;
                                ">{icon}</div>
                                <div style="
                                    font-size: 1.35rem;
                                    font-weight: 700;
                                    margin-bottom: 8px;
                                ">{title}</div>
                                <div style="
                                    color: #b9b9c2;
                                    font-size: 1rem;
                                ">{description}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.button(
                            "Open",
                            key=f"citizen_{screen}",
                            use_container_width=True,
                            on_click=go_to,
                            args=(screen,),
                        )

    # Official dashboard
    else:
        visible_groups = []

        for group in GROUPS:
            gk, icon, title, desc, keys = group

            visible_keys = [
                key for key in keys
                if key in allowed_features and key != "offline"
            ]

            if visible_keys:
                visible_groups.append(
                    (gk, icon, title, desc, visible_keys)
                )

        for start in range(0, len(visible_groups), 3):
            cols = st.columns(3, gap="medium")

            for col, (gk, icon, title, desc, keys) in zip(
                cols, visible_groups[start:start + 3]
            ):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="tile-icon">{icon}</div>'
                            f'<div class="tile-title">{title}</div>'
                            f'<div class="tile-desc">{desc}</div>'
                            f'<div class="tile-desc">'
                            f'{len(keys)} features</div>',
                            unsafe_allow_html=True,
                        )

                        st.button(
                            "Open",
                            key=f"open_g_{gk}",
                            use_container_width=True,
                            on_click=go_to,
                            args=(f"cat_{gk}",),
                        )

        st.divider()
        st.subheader("Government Portal")

        st.button(
            "Manage Citizen Complaints",
            key="home_official_complaints",
            use_container_width=True,
            on_click=go_to,
            args=("official_complaints",),
        )

    # Quick Access for both roles
    st.divider()
    st.subheader("Quick Access")
    st.caption("Useful tools available whenever you need them.")

    st.button(
        "📴 Offline Mode",
        key="dashboard_offline",
        use_container_width=True,
        on_click=go_to,
        args=("offline",),
    )




# ======================= FEATURE SCREENS =======================
def page_overview():
    page_header("overview")
    k = st.columns(4)
    k[0].metric("Wards monitored", len(view))
    k[1].metric("Overflow-risk wards", len(at_risk))
    k[2].metric("Average fill", f"{view['Fill %'].mean():.0f}%")
    k[3].metric("Items detected", impact["items"])
    if len(at_risk) > 0:
        st.error(f"⚠️ Overflow risk {'now' if days == 0 else f'in {days} day(s)'}: {', '.join(at_risk['Zone'])}")
    else:
        st.success("No ward is at overflow risk.")
    ev = st.session_state.events[::-1][:5]
    st.markdown("**Latest alerts**\n\n" + ("\n\n".join(f"`{t}` {i} {x}" for t, i, x in ev) if ev else "No alerts yet."))



def page_detect():
    page_header("detect")

    off = st.session_state.get("offline")
    
    # Request the user's location from their browser
   # Demo mode: allow manual ward selection without accessing GPS
    location = None
    st.caption("📍 Demo mode: choose any ward manually.")

    if off:
        st.warning(
            "📴 Offline: detection still works on this device. "
            "Results are saved and will sync when you are back online."
        )

    if not MODEL_AVAILABLE:
        st.warning(
            "Trained model or labels not found. "
            "Running in manual mode: choose the waste type yourself."
        )

    mode = st.radio(
        "Input method",
        ["Upload photo", "Use camera"],
        horizontal=True
    )

    file = (
        st.file_uploader("Upload a waste photo", type=["jpg", "jpeg", "png"])
        if mode == "Upload photo"
        else st.camera_input("Take a photo of the waste")
    )

    if not file:
        return

    img = Image.open(file)
    model_pred, label = None, None

    left, right = st.columns([1, 2])
    left.image(img, width=300)

    if MODEL_AVAILABLE:
        with right:
            with st.spinner("Classifying..."):
                label, conf = classify_bytes(file.getvalue())

            model_pred = label

            if conf < 0.80:
                st.warning(
                    f"Model is unsure: {label} ({conf:.0%}). "
                    "Please confirm below."
                )
            else:
                st.success(
                    f"Detected: **{label}** ({conf:.0%} confident)"
                )

            options = list(WASTE_TYPES)
            if label not in options:
                options.append(label)

            label = st.selectbox(
                "Correct the waste type if the model is wrong",
                options,
                index=options.index(label),
                key=f"fix_{model_pred}_{conf:.4f}"
            )
    else:
        label = right.selectbox("Waste type (manual)", WASTE_TYPES)

    st.session_state.last_img = file.getvalue()
    st.session_state.last_pred = model_pred

    if label in DISPOSAL_TIPS:
        right.info(f"♻️ Tip: {DISPOSAL_TIPS[label]}")

    zone = right.selectbox(
        "Which ward is this waste from?",
        st.session_state.bins["Zone"]
    )

    if right.button("➕ Add to twin"):
        if off:
            st.session_state.queue.append(
                (zone, label, datetime.now().strftime("%H:%M:%S"))
            )
            st.session_state.msg = (
                f"Saved on device "
                f"({len(st.session_state.queue)} waiting to sync)"
            )
        else:
            add_waste_to_zone(zone, label)
            st.session_state.msg = (
                f"Added {label} to {zone}: fill +{FILL_PER_ITEM}%"
            )

        st.rerun()



def page_map():
    page_header("map")
    draw_map(view, st.toggle("3D view (column height = fill level)"))
    st.markdown(LEGEND, unsafe_allow_html=True)
    st.caption("Hover over a ward for details. Use the sidebar slider to see the predicted future.")



def page_zones():
    page_header("zones")

    cols = st.columns(3)
    for i, (_, r) in enumerate(view.iterrows()):
        cols[i % 3].markdown(zone_card(r), unsafe_allow_html=True)

    st.subheader("🍩 Waste mix")
    if impact["items"] == 0:
        st.caption("No items detected yet.")
    elif HAS_PLOTLY:
        nz = {k: v for k, v in impact["counts"].items() if v > 0}
        fig = go.Figure(
            go.Pie(
                labels=list(nz),
                values=list(nz.values()),
                hole=0.55,
                marker={
                    "colors": [
                        WASTE_COLORS[WASTE_TYPES.index(k)]
                        for k in nz
                    ]
                },
            )
        )
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)
    else:
        st.bar_chart(pd.Series(impact["counts"]))

    with st.expander("Raw data table"):
        st.dataframe(
            view[["Zone", "Fill %", "Status"] + WASTE_TYPES + ["Last updated"]],
            hide_index=True,
        )

def page_alerts():
    page_header("alerts")
    ev = st.session_state.events[::-1]
    st.markdown("\n\n".join(f"`{t}` {i} {x}" for t, i, x in ev) if ev else "No alerts yet. Add waste or turn on the live sensor feed.")


def page_route():
    page_header("route")
    st.caption(f"Visits only wards at {PICKUP_THRESHOLD}%+ fill, always going to the nearest unvisited ward, starting at the depot.")
    order = impact["order"]
    if not order:
        st.success("No ward needs a pickup right now.")
        return
    rows, prev = [], DEPOT_COORD
    for i, z in enumerate(order, 1):
        leg = haversine_km(prev, ZONE_COORDS[z]) * ROAD_FACTOR
        rows.append({"Stop": i, "Ward": z, "Fill %": int(view.loc[view["Zone"] == z, "Fill %"].iloc[0]), "Leg (km)": round(leg, 1)})
        prev = ZONE_COORDS[z]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.metric("Total route (incl. return to depot)", f"{impact['opt_km']:.1f} km")
    try:
        import pydeck as pdk

        pts = [DEPOT_COORD] + [ZONE_COORDS[z] for z in order] + [DEPOT_COORD]
        st.pydeck_chart(pdk.Deck(layers=[
            pdk.Layer("PathLayer", data=[{"path": [[lo, la] for la, lo in pts]}], get_path="path", get_width=60,
                      width_min_pixels=3, get_color=[0, 200, 150]),
            pdk.Layer("ScatterplotLayer", data=[{"lon": lo, "lat": la} for la, lo in pts], get_position="[lon, lat]",
                      get_radius=400, get_fill_color=[255, 80, 80])],
            initial_view_state=pdk.ViewState(latitude=12.96, longitude=77.64, zoom=10.3)))
    except Exception:
        st.caption("Map drawing unavailable; the ordered list above is the route.")
    if st.button("✅ Simulate collection (empties these wards)"):
        collect_zones(order)
        st.session_state.msg = "Route collected: wards reset to 5%"
        st.rerun()


def page_impact():
    page_header("impact")
    st.caption("Estimates, compared with a truck visiting every ward every day. Assumptions are at the top of app.py.")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Ward visits saved", impact["skipped"])
    i2.metric("Distance saved", f"{impact['km_saved']:.1f} km")
    i3.metric("CO₂ avoided", f"{impact['co2']:.1f} kg")
    i4.metric("Recycling rate", f"{impact['rate']:.0f}%" if impact["rate"] is not None else "n/a")
    st.markdown(f"""**How we calculate it**
- Distance saved = every-ward route − optimized route (straight-line × {ROAD_FACTOR} road factor)
- CO₂ avoided = km saved × {TRUCK_L_PER_KM} L/km × {CO2_KG_PER_L} kg CO₂ per litre of diesel
- Recycling rate = recyclable items (plastic, paper, metal, glass, cardboard) ÷ all detected items ({impact['items']} so far)""")


def page_history():
    page_header("history")
    hist = pd.DataFrame(st.session_state.history).set_index("Reading")
    if len(hist) < 2:
        st.info("Add a detection or turn on the live sensor feed to build history.")
    else:
        st.line_chart(hist)


def page_advice():
    page_header("advice")
    if not st.button("🤖 Get AI advice"):
        return
    if not api_key:
        st.info("No API key entered, so this is demo-mode advice (rule-based).")
        st.write(rule_based_advice(view, days))
        return
    try:
        prompt = (f"You are a smart-city waste management advisor for Bengaluru.\nDigital twin state, simulated {days} day(s) ahead:\n\n"
                  f"{twin_text(view)}\n\nGive exactly 3 short, specific recommendations, each mentioning real ward names:\n"
                  "1. Collection priority  2. Segregation / recycling improvement  3. Route efficiency")
        with st.spinner("Gemini is thinking..."):
            text, used = ask_gemini(api_key, prompt)
        st.caption(f"Model used: {used}")
        st.write(text)
    except Exception as e:
        st.error(f"AI unavailable: {e}")
        st.info("Showing demo-mode advice instead:")
        st.write(rule_based_advice(view, days))




def page_chat():
    page_header("chat")

    audio = (
        st.audio_input("🎤 Speak your question")
        if hasattr(st, "audio_input")
        else None
    )

    q = st.text_input(
        "…or type it",
        placeholder="Ask about waste, recycling, or a ward"
    )

    if not (st.button("Ask") and (q.strip() or audio)):
        return

    top = view.sort_values("Fill %", ascending=False).iloc[0]

    if not api_key or st.session_state.get("offline"):
        if st.session_state.get("offline"):
            st.info("Offline mode is ON. Cloud AI is unavailable.")
        else:
            st.info("Add your Gemini API key to use the full AI assistant.")

        if audio:
            st.warning("Voice questions need Gemini to be enabled.")
            return

        ans = (
            f"{top['Zone']} is the fullest ward at "
            f"{int(top['Fill %'])}% ({top['Status']}). "
            "Prioritise collection there. For recycling advice, "
            "enable Gemini to get a detailed answer."
        )

    else:
        try:
            prompt = (
                "You are BINSIGHT AI, a helpful generative AI assistant "
                "for smart-city waste management in Bengaluru.\n\n"
                "Answer only in English. Do not translate responses "
                "into other languages or use multilingual output.\n\n"
                "Answer questions about waste management, segregation, "
                "recycling, reuse, composting, and waste reduction. Give "
                "practical suggestions for homes, colleges, and communities. "
                "Help with overflowing bins, collection priorities, and "
                "route efficiency when relevant.\n\n"
                "Use the digital-twin data for ward-specific claims. "
                "The data may be simulated; never present simulated results "
                "as verified real-world events. Never invent statistics. "
                "For e-waste and hazardous waste, recommend authorised "
                "collection or disposal services. General recycling "
                "questions can be answered even if the data does not "
                "contain the answer. Keep responses friendly and actionable.\n\n"
                f"Digital-twin data (simulated {days} day(s) ahead):\n"
                f"{twin_text(view)}\n\n"
                f"{'The question is in the attached audio.' if audio else 'User question: ' + q.strip()}"
            )

            with st.spinner("Thinking..."):
                ans, used = ask_gemini(
                    api_key,
                    prompt,
                    audio=audio.getvalue() if audio else None
                )

            st.caption(f"Model used: {used}")

        except Exception as e:
            st.error(f"AI unavailable: {e}")
            return

    st.success(ans)
    speak_button(ans, "en-IN")

def page_vision():
    page_header("vision")
    data = st.session_state.get("last_img")
    if data is None:
        st.info("Upload or capture a photo on the Detect Waste screen first.")
    elif not api_key:
        st.info("Paste a Gemini API key in the sidebar to use this.")
    elif st.button("Ask Gemini about this photo"):
        try:
            with st.spinner("Gemini is looking..."):
                text, used = ask_gemini(api_key, "Describe the main waste item in this photo in one line, classify it as one of: " + ", ".join(WASTE_TYPES) + ", and give one disposal tip. Max 3 short lines.", image=Image.open(io.BytesIO(data)))
            c_a, c_b = st.columns(2)
            c_a.info(f"**Our trained model:** {st.session_state.get('last_pred') or 'manual mode'}")
            c_b.success(f"**Gemini ({used}):**\n\n{text}")
        except Exception as e:
            st.error(f"AI unavailable: {e}")



def build_docx(view, days, m):
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

    doc = Document()

    # Page layout
    section = doc.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)

    # Default font
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9)

    title = doc.add_heading("BINSIGHT Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} "
        f"| Simulated days ahead: {days}"
    )
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Ward status table
    doc.add_heading("Ward Status", level=1)

    headers = ["Ward", "Fill Level", "Status", "Last Updated"]
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Shading Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    widths = [Inches(2.0), Inches(1.0), Inches(1.35), Inches(2.0)]

    for cell, header, width in zip(t.rows[0].cells, headers, widths):
        cell.text = header
        cell.width = width
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    for _, r in view.iterrows():
        cells = t.add_row().cells
        values = [
            str(r["Zone"]),
            f"{int(r['Fill %'])}%",
            str(r["Status"]).split(" ", 1)[-1],
            str(r["Last updated"]),
        ]

        for cell, value, width in zip(cells, values, widths):
            cell.text = value
            cell.width = width
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    # Waste breakdown table
    used = [
        w for w in WASTE_TYPES
        if w in view.columns and view[w].sum() > 0
    ]

    if used:
        doc.add_heading("Waste Detected per Ward", level=1)

        t2 = doc.add_table(rows=1, cols=1 + len(used))
        t2.style = "Light Shading Accent 1"
        t2.alignment = WD_TABLE_ALIGNMENT.CENTER

        headers = ["Ward"] + used
        for cell, header in zip(t2.rows[0].cells, headers):
            cell.text = header
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

        for _, r in view.iterrows():
            cells = t2.add_row().cells
            cells[0].text = str(r["Zone"])

            for cell, w in zip(cells[1:], used):
                cell.text = str(int(r[w]))

            for cell in cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    # Recommendations
    doc.add_heading("Recommendations", level=1)
    for line in rule_based_advice(view, days).split("\n\n"):
        doc.add_paragraph(
            line.replace("**", ""),
            style="List Bullet"
        )

    # Route
    doc.add_heading("Optimized Route", level=1)
    doc.add_paragraph(
        " → ".join(m["order"])
        if m["order"] else "No pickups needed"
    )
    doc.add_paragraph(
        f"Route length: {m['opt_km']:.1f} km "
        f"vs {m['naive_km']:.1f} km for visiting every ward."
    )

    # Impact
    doc.add_heading("Estimated Impact", level=1)

    rate = (
        f"{m['rate']:.0f}%"
        if m["rate"] is not None
        else "N/A (no items detected yet)"
    )

    for line in (
        f"Ward visits saved: {m['skipped']}",
        f"Distance saved: {m['km_saved']:.1f} km",
        f"CO2 avoided: {m['co2']:.1f} kg",
        f"Recycling rate of detected waste: {rate}",
    ):
        doc.add_paragraph(line, style="List Bullet")

    doc.add_paragraph(
        "Estimates use simulated sensor data and the assumptions "
        "listed in the app settings."
    )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()



def page_report():
    page_header("report")
    report = build_report(view, days, impact)
    try:
        st.download_button("📄 Download report (Word)", data=build_docx(view, days, impact), file_name="binsight_report.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except ImportError:
        st.info("Run `pip install python-docx` (with the venv active) to download a Word file. Using the text version for now.")
        st.download_button("📄 Download report (text)", data=report, file_name="binsight_report.md", mime="text/markdown")
    st.markdown(report)


def page_transparency():
    page_header("transparency")
    st.caption("What citizens see: simple, live ward status. No personal data is shown here.")
    comp = load_complaints()
    bins = st.session_state.bins
    cols = st.columns(3)
    for i, (_, r) in enumerate(bins.iterrows()):
        f = int(r["Fill %"])
        open_n = int((comp["Zone"] == r["Zone"]).sum()) if len(comp) else 0
        c = status_color(f)
        cols[i % 3].markdown(
            f'<div class="zcard" style="border-left:6px solid {c}"><div class="zname">{r["Zone"]}</div>'
            f'<div class="zfill" style="color:{c}">{100 - f}/100</div><div class="zmeta">Cleanliness score</div>'
            f'<div style="margin-top:8px">{plain_status(f)}</div><div class="zmeta">Open complaints: {open_n} · updated {r["Last updated"]}</div></div>',
            unsafe_allow_html=True)
    st.button("📝 Report a problem", key="pub_report", on_click=go_to, args=("complaint",))


def page_complaint():
    page_header("complaint")
    st.caption("Report an uncleanliness issue. Add photos or videos as proof.")
    with st.form("complaint_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full name *")
        phone = c2.text_input("Phone number *")
        email = st.text_input("Email (optional)")
        address = st.text_area("Address of the place with the problem *", placeholder="Street, landmark, area, pincode")
        z, cat = st.columns(2)
        zone = z.selectbox("Ward", list(ZONE_COORDS.keys()))
        category = cat.selectbox("Type of issue", CATEGORIES)
        desc = st.text_area("Describe the issue *")
        files = st.file_uploader("Photos or videos as proof", type=["jpg", "jpeg", "png", "mp4", "mov", "webm"], accept_multiple_files=True)
        consent = st.checkbox("I agree that BINSIGHT may store my details only to follow up on this complaint. *")
        sent = st.form_submit_button("Submit complaint")
    if sent:
        digits = "".join(ch for ch in phone if ch.isdigit())
        if not (name.strip() and address.strip() and desc.strip() and len(digits) >= 10 and consent):
            st.error("Please fill every field marked *, enter a valid phone number, and tick the consent box.")
        else:
            cid = save_complaint(name.strip(), phone.strip(), email.strip(), address.strip(), zone, category, desc.strip(), files)
            log_event("📝", f"New complaint in {zone}: {category}")
            st.success(f"Complaint submitted. Your complaint ID is **{cid}**. Keep it to track the status.")
    st.subheader("Track a complaint")
    tid = st.text_input("Enter your complaint ID", placeholder="BIN-261011-1234")
    if tid:
        comp = load_complaints()
        hit = comp[comp["ID"] == tid.strip()] if len(comp) else comp
        if len(hit):
            r = hit.iloc[0]
            st.info(f"**{r['ID']}** · {r['Category']} · {r['Zone']} · submitted {r['Time']} · Status: **{r['Status']}**")
        else:
            st.warning("No complaint found with that ID.")
    st.caption(f"Helpline: {HELPLINE}  ·  Email: {HELP_EMAIL}")


def faq_answer(q):
    words = set(q.lower().split())
    best = max(FAQ, key=lambda k: len(words & set((k + " " + FAQ[k]).lower().split())))
    return FAQ[best]


def page_help():
    page_header("help")
    for q, a in FAQ.items():
        with st.expander(q):
            st.write(a)
    q = st.text_input("Ask us anything about BINSIGHT")
    if st.button("Ask help desk") and q:
        if api_key and not st.session_state.get("offline"):
            try:
                with st.spinner("Thinking..."):
                    ans, _ = ask_gemini(api_key, f"You are the BINSIGHT help desk. Facts: {ABOUT}\nFAQ: {FAQ}\nQuestion: {q}\nAnswer briefly and politely using only these facts.")
                st.success(ans)
                return
            except Exception:
                pass
        st.success(faq_answer(q))
    st.info(f"📞 Helpline: {HELPLINE}   ·   ✉️ Email: {HELP_EMAIL}")


def page_offline():
    page_header("offline")
    off = st.session_state.get("offline")
    st.markdown("**Status:** " + ("📴 Offline (simulated)" if off else "📶 Online"))
    st.write("The vision model runs **on the device**, so waste is still classified when the network drops. "
             "Detections are saved in a queue and sync to the twin when the connection returns. "
             "Cloud AI (Gemini) pauses while offline and falls back to simple rule-based answers.")
    st.caption("Try it: switch on 'Simulate network drop' in the sidebar, add detections on the Detect screen, then switch it off.")
    q = st.session_state.queue
    st.metric("Detections waiting to sync", len(q))
    if q:
        st.dataframe(pd.DataFrame(q, columns=["Ward", "Waste", "Saved at"]), hide_index=True)
        if not off and st.button("🔄 Sync now"):
            st.session_state.msg = f"Synced {sync_queue()} detection(s)"
            st.rerun()

def page_official_complaints():
    """Official dashboard for reviewing submitted complaints."""

    page_header("official_complaints")

    st.title("Government Official — Complaint Dashboard")
    st.caption("Review submitted complaints and update their status.")

    complaints = load_complaints()

    if complaints.empty:
        st.info("No complaints have been submitted yet.")
        return

    # Ensure expected columns exist before displaying the dashboard.
    required_columns = [
        "ID", "Time", "Name", "Phone", "Email",
        "Address", "Zone", "Category", "Description",
        "Files", "Status"
    ]

    for column in required_columns:
        if column not in complaints.columns:
            complaints[column] = ""

    st.metric("Total complaints", len(complaints))

    status_filter = st.selectbox(
        "Filter by status",
        ["All", "Received", "In Progress", "Resolved"]
    )

    filtered = complaints.copy()

    if status_filter != "All":
        filtered = filtered[
            filtered["Status"].str.strip().str.lower()
            == status_filter.lower()
        ]

    display_columns = [
        "ID", "Time", "Zone", "Category", "Description", "Status"
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("View complaint details")

    complaint_ids = filtered["ID"].tolist()

    if not complaint_ids:
        st.info("No complaints match this filter.")
        return

    selected_id = st.selectbox(
        "Select a complaint ID",
        complaint_ids
    )

    selected = complaints[complaints["ID"] == selected_id].iloc[0]

    st.write("**Submitted by:**", selected["Name"])
    st.write("**Phone:**", selected["Phone"])
    st.write("**Email:**", selected["Email"])
    st.write("**Address:**", selected["Address"])
    st.write("**Ward:**", selected["Zone"])
    st.write("**Issue:**", selected["Category"])
    st.write("**Description:**", selected["Description"])
    st.write("**Attachments:**", selected["Files"] or "None")
    st.write("**Current status:**", selected["Status"] or "Received")

    status_options = ["Received", "In Progress", "Resolved"]
    current_status = (
        selected["Status"]
        if selected["Status"] in status_options
        else "Received"
    )

    with st.form("official_update_complaint"):
        new_status = st.selectbox(
            "Update status",
            status_options,
            index=status_options.index(current_status)
        )

        update_clicked = st.form_submit_button("Save status")

    if update_clicked:
        all_complaints = load_complaints()

        if not all_complaints.empty and "ID" in all_complaints.columns:
            match = all_complaints["ID"] == selected_id

            if match.any():
                all_complaints.loc[match, "Status"] = new_status
                all_complaints.to_csv(COMPLAINT_CSV, index=False)
                st.success("Complaint status updated.")
                st.rerun()
            else:
                st.error("The selected complaint could not be found.")
        else:
            st.error("Complaint data is unavailable.")


PAGE_FUNCS = {"overview": page_overview, "detect": page_detect, "map": page_map, "zones": page_zones,
              "alerts": page_alerts, "route": page_route, "impact": page_impact, "history": page_history,
              "advice": page_advice, "chat": page_chat, "vision": page_vision, "report": page_report,
              "transparency": page_transparency, "complaint": page_complaint, "help": page_help, "offline": page_offline}




def page_auth():
    """Display the BINSIGHT citizen login and registration page."""

    st.title("Welcome to BINSIGHT")
    st.write("Sign in to continue, or create a citizen account.")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        st.subheader("Sign in")

        with st.form("binsight_login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_clicked = st.form_submit_button(
                "Login", use_container_width=True
            )

        if login_clicked:
            if not email.strip() or not password:
                st.error("Please enter your email and password.")
            else:
                user = authenticate(email, password)

                if user and user.get("role") == "citizen":
                    st.session_state.user = user
                    st.session_state.screen = "home"
                    st.success("Login successful!")
                    st.rerun()
                elif user:
                    st.error(
                        "Please use the Official Login option "
                        "for your account."
                    )
                else:
                    st.error("Invalid email or password.")

    with signup_tab:
        st.subheader("Create a citizen account")

        with st.form("binsight_signup_form"):
            name = st.text_input("Full name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input(
                "Password (minimum 8 characters)",
                type="password", key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password", key="signup_confirm_password"
            )

            signup_clicked = st.form_submit_button(
                "Create Account", use_container_width=True
            )

        if signup_clicked:
            if signup_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                success, message = create_citizen_account(
                    name, signup_email, signup_password
                )
                if success:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)

    st.divider()

    if st.button(
        "← Back to Login Choices",
        key="back_to_choices",
        use_container_width=True
    ):
        st.session_state.screen = "choose_login"
        st.rerun()

def page_access_choice():
    st.title("Welcome to BINSIGHT")
    st.write("Choose how you want to enter the Waste Management Twin.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏛️ Official Login")
        st.write("For authorized government officials.")
        if st.button("Official Login", key="official_choice",
                     use_container_width=True):
            st.session_state.screen = "official_auth"
            st.rerun()

    with col2:
        st.subheader("👤 Citizen Login")
        st.write("Submit and follow up on complaints.")
        if st.button("Citizen Login", key="citizen_choice",
                     use_container_width=True):
            st.session_state.screen = "auth"
            st.rerun()

    

    if st.button("← Back to landing page"):
        st.session_state.screen = "home"
        st.rerun()

def page_official_auth():
    st.title("🏛️ Official Login")
    st.caption("For authorized BINSIGHT officials")

    with st.form("official_login_form"):
        email = st.text_input("Official email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

    if submitted:
        user = authenticate(email, password)

        if user and user.get("role") == "official":
            st.session_state.user = user
            st.session_state.screen = "home"
            st.rerun()
        else:
            st.error("Invalid official email or password.")

    with st.popover("🔑 View Demo Credentials"):
        st.markdown("### 🏛️ Official Demo Account")
        st.code(
            "Email: official.demo@binsight.test\n"
            "Password: OfficialDemo123"
        )
        st.caption(
            "For hackathon demonstration only."
        )

    if st.button("← Back to Login Choices", key="official_back"):
         st.session_state.screen = "choose_login"
         st.rerun()



def page_category(gk):
    # Find the requested category
    g = next((group for group in GROUPS if group[0] == gk), None)

    if g is None:
        st.error("Category not found.")
        st.button(
            "← Back to dashboard",
            on_click=go_to,
            args=("home",),
        )
        return

    user = st.session_state.get("user", {})
    role = user.get("role", "citizen")

    citizen_features = {
        "detect", "map", "transparency", "help",
        "complaint", "advice", "chat", "offline",
    }

    official_features = {
        "overview", "zones", "detect", "alerts", "map",
        "route", "impact", "history", "advice", "chat",
        "vision", "report", "transparency", "complaint",
        "help", "offline", "official_complaints",
    }

    allowed_features = (
        official_features if role == "official"
        else citizen_features
    )

    visible_screens = [
        screen_item for screen_item in SCREENS
        if screen_item[0] in g[4]
        and screen_item[0] in allowed_features
        and screen_item[0] != "offline"
    ]

    st.button(
        "← Back to dashboard",
        key=f"back_cat_{gk}",
        on_click=go_to,
        args=("home",),
    )

    st.title(f"{g[1]} {g[2]}")
    st.caption(g[3])

    if not visible_screens:
        st.info("No features available in this category.")
        return

    cols = st.columns(3, gap="medium")

    for i, (key, icon, title, desc) in enumerate(visible_screens):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="tile-icon">{icon}</div>'
                    f'<div class="tile-title">{title}</div>'
                    f'<div class="tile-desc">{desc}</div>',
                    unsafe_allow_html=True,
                )

                st.button(
                    "Open",
                    key=f"open_{gk}_{key}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(key,),
                )

# ---------------- PAGE ROUTING ----------------
inject_pastel_hover()
screen = st.session_state.screen
user = st.session_state.get("user")
role = user.get("role") if user else None

if screen == "landing":
    page_landing()

elif screen == "choose_login":
    page_access_choice()

elif screen == "official_auth":
    page_official_auth()

elif screen == "auth":
    page_auth()

elif not user:
    st.session_state.screen = "auth"
    st.rerun()

elif screen == "home":
    page_home()

elif screen.startswith("cat_"):
    page_category(screen[4:])

elif screen == "official_complaints":
    if role != "official":
        st.error("This page is available to government officials only.")
        st.session_state.screen = "home"
        st.rerun()
    else:
        page_official_complaints()

elif screen in PAGE_FUNCS:
    PAGE_FUNCS[screen]()

else:
    st.session_state.screen = "home"
    st.rerun()



# ---------------- MOCK IOT LIVE FEED (runs last) ----------------
if live:
    @st.fragment(run_every=4)
    def live_feed():
        now = time.time()

        if now - st.session_state.last_tick >= 3.5:
            st.session_state.last_tick = now
            sensor_tick()

        st.caption(
            "📡 Live sensor feed is ON: "
            "bins update every few seconds."
        )

    live_feed()