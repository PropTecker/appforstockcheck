# app.py — BNG Optimiser (Standalone), v9.14
# Changes in v9.14:
# - Generalized Orchard stacking: added ADJACENT (SRM 4/3) tier support
# - Implemented dynamic "Other" component selection (cheapest eligible area habitat ≤ Medium distinctiveness)
# - Updated pairing mix: ADJACENT uses 1.00 Orchard + 1/3 Other (75%/25% split); FAR uses 0.50 Orchard + 0.50 Other
# - Enhanced split_paired_rows to handle non-50/50 splits correctly
# - Pricing: Adjacent = (1.00*orchard + (1/3)*other) / (4/3); Far = 0.5*orchard + 0.5*other
#
# Changes in v9.13:
# - Added "Start New Quote" button with comprehensive reset functionality
# - Implemented automatic map refresh after optimization completes
# - Ensured financial readout persists across map interactions
# - Enhanced user experience with clear visual feedback
#
# Changes in v9.12:
# - Fixed map disappearing on optimise
# - Improved UI responsiveness
# - Better state management
# - Fixed flickering issues
# - Enhanced error handling

import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
try:
    from streamlit_folium import folium_static
except Exception:
    folium_static = None
import folium

# ================= Config / constants =================
ADMIN_FEE_GBP = 500.0
SINGLE_BANK_SOFT_PCT = 0.01
MAP_CATCHMENT_ALPHA = 0.03
UA = {"User-Agent": "WildCapital-Optimiser/1.0 (+contact@example.com)"}
LEDGER_AREA = "area"
LEDGER_HEDGE = "hedgerow"
LEDGER_WATER = "watercourse"

NET_GAIN_WATERCOURSE_LABEL = "Net Gain (Watercourses)"  # new
POSTCODES_IO = "https://api.postcodes.io/postcodes/"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NCA_URL = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
           "National_Character_Areas_England/FeatureServer/0")
LPA_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
           "Local_Authority_Districts_December_2024_Boundaries_UK_BFC/FeatureServer/0")

# Optional solver
try:
    import pulp
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

# ================= Page Setup =================
st.set_page_config(page_title="BNG Optimiser (Standalone)", page_icon="🧭", layout="wide")
st.markdown("<h2>BNG Optimiser — Standalone</h2>", unsafe_allow_html=True)

# ================= Initialize Session State =================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "auth_ok": False,
        "map_version": 0,
        "target_lpa_name": "",
        "target_nca_name": "",
        "lpa_neighbors": [],
        "nca_neighbors": [],
        "lpa_neighbors_norm": [],
        "nca_neighbors_norm": [],
        "target_lat": None,
        "target_lon": None,
        "lpa_geojson": None,
        "nca_geojson": None,
        "last_alloc_df": None,
        "bank_geo_cache": {},
        "bank_catchment_geo": {},
        "demand_rows": [{"id": 1, "habitat_name": "", "units": 0.0}],
        "_next_row_id": 2,
        "optimization_complete": False,
        "manual_hedgerow_rows": [],
        "manual_watercourse_rows": [],
        "_next_manual_hedgerow_id": 1,
        "_next_manual_watercourse_id": 1,
        "email_client_name": "INSERT NAME",
        "email_ref_number": "BNG00XXX",
        "email_location": "INSERT LOCATION",
        "postcode_input": "",
        "address_input": "",
        "needs_map_refresh": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_quote():
    """Reset all quote-related session state to start a new quote"""
    try:
        # First, delete all widget-bound keys for existing demand rows
        # This must happen BEFORE resetting demand_rows to clear the widget state
        if "demand_rows" in st.session_state:
            for row in st.session_state["demand_rows"]:
                row_id = row.get("id")
                # Delete habitat selectbox key
                hab_key = f"hab_{row_id}"
                if hab_key in st.session_state:
                    del st.session_state[hab_key]
                # Delete units number_input key
                units_key = f"units_{row_id}"
                if units_key in st.session_state:
                    del st.session_state[units_key]
        
        # Now reset demand_rows data
        st.session_state["demand_rows"] = [{"id": 1, "habitat_name": "", "units": 0.0}]
        st.session_state["_next_row_id"] = 2
        st.session_state["target_lpa_name"] = ""
        st.session_state["target_nca_name"] = ""
        st.session_state["lpa_neighbors"] = []
        st.session_state["nca_neighbors"] = []
        st.session_state["lpa_neighbors_norm"] = []
        st.session_state["nca_neighbors_norm"] = []
        st.session_state["target_lat"] = None
        st.session_state["target_lon"] = None
        st.session_state["lpa_geojson"] = None
        st.session_state["nca_geojson"] = None
        st.session_state["last_alloc_df"] = None
        st.session_state["bank_geo_cache"] = {}
        st.session_state["bank_catchment_geo"] = {}
        st.session_state["optimization_complete"] = False
        st.session_state["manual_hedgerow_rows"] = []
        st.session_state["manual_watercourse_rows"] = []
        st.session_state["_next_manual_hedgerow_id"] = 1
        st.session_state["_next_manual_watercourse_id"] = 1
        st.session_state["email_client_name"] = "INSERT NAME"
        st.session_state["email_ref_number"] = "BNG00XXX"
        st.session_state["email_location"] = "INSERT LOCATION"
        st.session_state["map_version"] = st.session_state.get("map_version", 0) + 1
        # Clear location input fields by deleting them (widget-bound variables)
        if "postcode_input" in st.session_state:
            del st.session_state["postcode_input"]
        if "address_input" in st.session_state:
            del st.session_state["address_input"]
        # Clear summary dataframes
        st.session_state["site_hab_totals"] = None
        st.session_state["by_bank"] = None
        st.session_state["by_hab"] = None
        st.session_state["summary_df"] = None
        st.session_state["total_cost"] = None
        st.session_state["contract_size"] = None
    except Exception as e:
        st.error(f"Error resetting quote: {e}")
        # Re-initialize session state as fallback
        init_session_state()

init_session_state()

# ================= Safe strings =================
def sstr(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return ""
    return str(x).strip()

def norm_name(s: str) -> str:
    t = sstr(s).lower()
    t = re.sub(r'\b(city of|royal borough of|metropolitan borough of)\b', '', t)
    t = re.sub(r'\b(council|borough|district|county|unitary authority|unitary|city)\b', '', t)
    t = t.replace("&", "and")
    t = re.sub(r'[^a-z0-9]+', '', t)
    return t

def is_hedgerow(name: str) -> bool:
    name_str = sstr(name)
    # Check if it's the hedgerow net gain label
    if name_str == "Net Gain (Hedgerows)":
        return True
    
    # Check UmbrellaType column if backend is loaded
    try:
        if backend and "HabitatCatalog" in backend:
            catalog = backend["HabitatCatalog"]
            if "UmbrellaType" in catalog.columns:
                match = catalog[catalog["habitat_name"].astype(str).str.strip() == name_str]
                if not match.empty:
                    umbrella_type = sstr(match.iloc[0]["UmbrellaType"]).lower()
                    return umbrella_type == "hedgerow"
    except Exception:
        pass
    
    # Fallback to text matching
    return "hedgerow" in name_str.lower()

def is_watercourse(name: str) -> bool:
    name_str = sstr(name)
    
    # Check UmbrellaType column if backend is loaded
    try:
        if backend and "HabitatCatalog" in backend:
            catalog = backend["HabitatCatalog"]
            if "UmbrellaType" in catalog.columns:
                match = catalog[catalog["habitat_name"].astype(str).str.strip() == name_str]
                if not match.empty:
                    umbrella_type = sstr(match.iloc[0]["UmbrellaType"]).lower()
                    return umbrella_type == "watercourse"
    except Exception:
        pass
    
    # Fallback to text matching
    name_lower = name_str.lower()
    return "watercourse" in name_lower or "water" in name_lower

def get_hedgerow_habitats(catalog_df: pd.DataFrame) -> List[str]:
    """Get list of hedgerow habitats from catalog using UmbrellaType column"""
    if "UmbrellaType" in catalog_df.columns:
        # Use the UmbrellaType column to filter
        hedgerow_df = catalog_df[catalog_df["UmbrellaType"].astype(str).str.strip().str.lower() == "hedgerow"]
        return sorted([sstr(x) for x in hedgerow_df["habitat_name"].dropna().unique().tolist()])
    else:
        # Fallback to text matching if column doesn't exist
        all_habitats = [sstr(x) for x in catalog_df["habitat_name"].dropna().unique().tolist()]
        return sorted([h for h in all_habitats if is_hedgerow(h)])

def get_watercourse_habitats(catalog_df: pd.DataFrame) -> List[str]:
    """Get list of watercourse habitats from catalog using UmbrellaType column"""
    if "UmbrellaType" in catalog_df.columns:
        # Use the UmbrellaType column to filter
        watercourse_df = catalog_df[catalog_df["UmbrellaType"].astype(str).str.strip().str.lower() == "watercourse"]
        return sorted([sstr(x) for x in watercourse_df["habitat_name"].dropna().unique().tolist()])
    else:
        # Fallback to text matching if column doesn't exist
        all_habitats = [sstr(x) for x in catalog_df["habitat_name"].dropna().unique().tolist()]
        return sorted([h for h in all_habitats if is_watercourse(h)])

# ================= Login =================
DEFAULT_USER = "WC0323"
DEFAULT_PASS = "Wimbourne"

def require_login():
    if st.session_state.auth_ok:
        with st.sidebar:
            if st.button("Log out", key="logout_btn"):
                # Clear session state on logout
                for key in list(st.session_state.keys()):
                    if key != "auth_ok":
                        del st.session_state[key]
                st.session_state.auth_ok = False
                st.rerun()
        return
    
    st.markdown("## 🔐 Sign in")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    
    if ok:
        valid_u = st.secrets.get("auth", {}).get("username", DEFAULT_USER)
        valid_p = st.secrets.get("auth", {}).get("password", DEFAULT_PASS)
        if u == valid_u and p == valid_p:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Invalid credentials")
            st.stop()
    st.stop()

require_login()

# ================= HTTP helpers =================
def http_get(url, params=None, headers=None, timeout=25):
    try:
        r = requests.get(url, params=params or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout connecting to {url}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Connection error to {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} error for {url}")
    except Exception as e:
        raise RuntimeError(f"HTTP error for {url}: {e}")

def http_post(url, data=None, headers=None, timeout=25):
    try:
        r = requests.post(url, data=data or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout connecting to {url}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Connection error to {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} error for {url}")
    except Exception as e:
        raise RuntimeError(f"HTTP POST error for {url}: {e}")

def safe_json(r: requests.Response) -> Dict[str, Any]:
    try:
        return r.json()
    except Exception:
        preview = (r.text or "")[:300]
        raise RuntimeError(f"Invalid JSON from {r.url} (status {r.status_code}). Starts: {preview}")

# ================= Geo helpers =================
def esri_polygon_to_geojson(geom: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not geom or "rings" not in geom:
        return None
    rings = geom.get("rings") or []
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}

def add_geojson_layer(fmap, geojson: Dict[str, Any], name: str, color: str, weight: int, fill_opacity: float = 0.05, show=True):
    if not geojson:
        return
    try:
        folium.GeoJson(
            geojson,
            name=name,
            show=show,
            style_function=lambda x: {"color": color, "fillOpacity": fill_opacity, "weight": weight},
            tooltip=name
        ).add_to(fmap)
    except Exception:
        pass

# --- Ledger helpers ---
def get_umbrella_for(hab_name: str, catalog: pd.DataFrame) -> str:
    """Return 'hedgerow' | 'watercourse' | 'area' for a habitat name using UmbrellaType; 
       also handles special Net Gain labels."""
    h = sstr(hab_name)
    if not h:
        return LEDGER_AREA
    if h == NET_GAIN_HEDGEROW_LABEL:
        return LEDGER_HEDGE
    if h == NET_GAIN_WATERCOURSE_LABEL:
        return LEDGER_WATER
    # Lookup in catalog
    m = catalog[catalog["habitat_name"].astype(str).str.strip() == h]
    umb = sstr(m.iloc[0]["UmbrellaType"]) if not m.empty and "UmbrellaType" in m.columns else ""
    umb = umb.lower()
    if umb == LEDGER_HEDGE:
        return LEDGER_HEDGE
    if umb == LEDGER_WATER:
        return LEDGER_WATER
    return LEDGER_AREA


# ================= Geocoding / lookups =================
def get_postcode_info(pc: str) -> Tuple[float, float, str]:
    pc_clean = sstr(pc).replace(" ", "").upper()
    r = http_get(POSTCODES_IO + pc_clean)
    js = safe_json(r)
    if js.get("status") != 200 or not js.get("result"):
        raise RuntimeError(f"Postcode lookup failed for '{pc}'.")
    data = js["result"]
    return float(data["latitude"]), float(data["longitude"]), sstr(data.get("admin_district") or data.get("admin_county"))

def geocode_address(addr: str) -> Tuple[float, float]:
    r = http_get(NOMINATIM_SEARCH, params={"q": sstr(addr), "format": "jsonv2", "limit": 1, "addressdetails": 0})
    js = safe_json(r)
    if isinstance(js, list) and js:
        lat, lon = js[0]["lat"], js[0]["lon"]
        return float(lat), float(lon)
    r = http_get("https://photon.komoot.io/api/", params={"q": sstr(addr), "limit": 1})
    js = safe_json(r)
    feats = js.get("features") or []
    if feats:
        lon, lat = feats[0]["geometry"]["coordinates"]
        return float(lat), float(lon)
    raise RuntimeError("Address geocoding failed.")

def arcgis_point_query(layer_url: str, lat: float, lon: float, out_fields: str) -> Dict[str, Any]:
    geometry_dict = {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}
    params = {
        "f": "json", "where": "1=1",
        "geometry": json.dumps(geometry_dict), "geometryType": "esriGeometryPoint",
        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields or "*", "returnGeometry": "true", "outSR": 4326
    }
    r = http_get(f"{layer_url}/query", params=params)
    js = safe_json(r)
    feats = js.get("features") or []
    return feats[0] if feats else {}

def layer_intersect_names(layer_url: str, polygon_geom: Dict[str, Any], name_field: str) -> List[str]:
    if not polygon_geom:
        return []
    data = {
        "f": "json", "where": "1=1",
        "geometry": json.dumps(polygon_geom), "geometryType": "esriGeometryPolygon",
        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": name_field, "returnGeometry": "false", "outSR": 4326,
        "geometryPrecision": 5,
    }
    r = http_post(f"{layer_url}/query", data=data)
    js = safe_json(r)
    names = [sstr((f.get("attributes") or {}).get(name_field)) for f in js.get("features", [])]
    return sorted({n for n in names if n})

def get_lpa_nca_for_point(lat: float, lon: float) -> Tuple[str, str]:
    lpa = sstr((arcgis_point_query(LPA_URL, lat, lon, "LAD24NM").get("attributes") or {}).get("LAD24NM"))
    nca = sstr((arcgis_point_query(NCA_URL, lat, lon, "NCA_Name").get("attributes") or {}).get("NCA_Name"))
    return lpa, nca

def get_catchment_geo_for_point(lat: float, lon: float) -> Tuple[str, Optional[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
    lpa_feat = arcgis_point_query(LPA_URL, lat, lon, "LAD24NM")
    nca_feat = arcgis_point_query(NCA_URL, lat, lon, "NCA_Name")
    lpa_name = sstr((lpa_feat.get("attributes") or {}).get("LAD24NM"))
    nca_name = sstr((nca_feat.get("attributes") or {}).get("NCA_Name"))
    lpa_gj = esri_polygon_to_geojson(lpa_feat.get("geometry"))
    nca_gj = esri_polygon_to_geojson(nca_feat.get("geometry"))
    return lpa_name, lpa_gj, nca_name, nca_gj

# ================= Tiering =================
def tier_for_bank(bank_lpa: str, bank_nca: str,
                  t_lpa: str, t_nca: str,
                  lpa_neigh: List[str], nca_neigh: List[str],
                  lpa_neigh_norm: Optional[List[str]] = None,
                  nca_neigh_norm: Optional[List[str]] = None) -> str:
    b_lpa = norm_name(bank_lpa)
    b_nca = norm_name(bank_nca)
    t_lpa_n = norm_name(t_lpa)
    t_nca_n = norm_name(t_nca)
    if lpa_neigh_norm is None:
        lpa_neigh_norm = [norm_name(x) for x in (lpa_neigh or [])]
    if nca_neigh_norm is None:
        nca_neigh_norm = [norm_name(x) for x in (nca_neigh or [])]
    
    # Evaluate LPA axis independently
    lpa_same = b_lpa and t_lpa_n and b_lpa == t_lpa_n
    lpa_neighbour = b_lpa and b_lpa in lpa_neigh_norm
    
    # Evaluate NCA axis independently  
    nca_same = b_nca and t_nca_n and b_nca == t_nca_n
    nca_neighbour = b_nca and b_nca in nca_neigh_norm
    
    # Return best (closest) category across both axes
    if lpa_same or nca_same:
        return "local"  # Local > Adjacent > Far
    elif lpa_neighbour or nca_neighbour:
        return "adjacent"  # Adjacent > Far
    else:
        return "far"

def select_contract_size(total_units: float, present: List[str]) -> str:
    tiers = set([sstr(x).lower() for x in present])
    if "fractional" in tiers and total_units < 0.1: return "fractional"
    if "small" in tiers and total_units < 2.5: return "small"
    if "medium" in tiers and total_units < 15: return "medium"
    for t in ["large", "medium", "small", "fractional"]:
        if t in tiers: return t
    return sstr(next(iter(present), "small")).lower()

# ================= Sidebar: backend =================
with st.sidebar:
    st.subheader("Backend")
    uploaded = st.file_uploader("Upload backend workbook (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("Or use an example backend in ./data", icon="ℹ️")
    use_example = st.checkbox("Use example backend from ./data",
                              value=bool(Path("data/HabitatBackend_WITH_STOCK.xlsx").exists()))
    quotes_hold_policy = st.selectbox(
        "Quotes policy for stock availability",
        ["Ignore quotes (default)", "Quotes hold 100%", "Quotes hold 50%"],
        index=0,
        help="How to treat 'quoted' units when computing quantity_available."
    )

@st.cache_data
def load_backend(xls_bytes) -> Dict[str, pd.DataFrame]:
    """Load backend with caching to prevent reprocessing"""
    x = pd.ExcelFile(BytesIO(xls_bytes))
    backend = {
        "Banks": pd.read_excel(x, "Banks"),
        "Pricing": pd.read_excel(x, "Pricing"),
        "HabitatCatalog": pd.read_excel(x, "HabitatCatalog"),
        "Stock": pd.read_excel(x, "Stock"),
        "DistinctivenessLevels": pd.read_excel(x, "DistinctivenessLevels"),
        "SRM": pd.read_excel(x, "SRM"),
        "TradingRules": pd.read_excel(x, "TradingRules") if "TradingRules" in x.sheet_names else pd.DataFrame(),
    }
    return backend

backend = None
if uploaded:
    backend = load_backend(uploaded.getvalue())
elif use_example:
    ex = Path("data/HabitatBackend_WITH_STOCK.xlsx")
    if ex.exists():
        with ex.open("rb") as f:
            backend = load_backend(f.read())

if backend is None:
    st.warning("Upload your backend workbook to continue.", icon="⚠️")
    st.stop()

# ================= BANK_KEY normalisation =================
def make_bank_key_col(df: pd.DataFrame, banks_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    has_df_name = "bank_name" in out.columns and out["bank_name"].astype(str).str.strip().ne("").any()
    if not has_df_name:
        if "bank_id" in out.columns and "bank_id" in banks_df.columns and "bank_name" in banks_df.columns:
            m = banks_df[["bank_id","bank_name"]].drop_duplicates()
            out = out.merge(m, on="bank_id", how="left")
    if "bank_name" in out.columns:
        out["BANK_KEY"] = out["bank_name"].where(out["bank_name"].astype(str).str.strip().ne(""), out.get("bank_id"))
    else:
        out["BANK_KEY"] = out.get("bank_id")
    out["BANK_KEY"] = out["BANK_KEY"].map(sstr)
    return out

# Apply quotes policy if present
if {"available_excl_quotes", "quoted"}.issubset(backend["Stock"].columns):
    s = backend["Stock"].copy()
    s["available_excl_quotes"] = pd.to_numeric(s["available_excl_quotes"], errors="coerce").fillna(0)
    s["quoted"] = pd.to_numeric(s["quoted"], errors="coerce").fillna(0)

    if quotes_hold_policy == "Ignore quotes (default)":
        s["quantity_available"] = s["available_excl_quotes"]
    elif quotes_hold_policy == "Quotes hold 100%":
        s["quantity_available"] = (s["available_excl_quotes"] - s["quoted"]).clip(lower=0)
    else:
        s["quantity_available"] = (s["available_excl_quotes"] - 0.5 * s["quoted"]).clip(lower=0)

    backend["Stock"] = s

# Enrich Banks geography
def bank_row_to_latlon(row: pd.Series) -> Optional[Tuple[float,float,str]]:
    if "lat" in row and "lon" in row:
        try:
            lat = float(row["lat"]); lon = float(row["lon"])
            if np.isfinite(lat) and np.isfinite(lon):
                return lat, lon, f"ll:{lat:.6f},{lon:.6f}"
        except Exception:
            pass
    if "postcode" in row and sstr(row["postcode"]):
        try:
            lat, lon, _ = get_postcode_info(sstr(row["postcode"]))
            return lat, lon, f"pc:{sstr(row['postcode']).upper().replace(' ','')}"
        except Exception:
            pass
    if "address" in row and sstr(row["address"]):
        try:
            lat, lon = geocode_address(sstr(row["address"]))
            return lat, lon, f"addr:{sstr(row['address']).lower()}"
        except Exception:
            pass
    return None

def enrich_banks_geography(banks_df: pd.DataFrame) -> pd.DataFrame:
    df = banks_df.copy()
    if "lpa_name" not in df.columns: df["lpa_name"] = ""
    if "nca_name" not in df.columns: df["nca_name"] = ""
    cache = st.session_state["bank_geo_cache"]
    needs = df[(df["lpa_name"].map(sstr) == "") | (df["nca_name"].map(sstr) == "")]
    prog = None
    if len(needs) > 0:
        prog = st.sidebar.progress(0.0, text="Resolving bank LPA/NCA…")
    rows, updated, total = [], 0, len(df)
    for _, row in df.iterrows():
        lpa_now = sstr(row.get("lpa_name"))
        nca_now = sstr(row.get("nca_name"))
        if lpa_now and nca_now:
            rows.append(row)
        else:
            loc = bank_row_to_latlon(row)
            if not loc:
                rows.append(row)
            else:
                lat, lon, key = loc
                if key in cache:
                    lpa, nca = cache[key]
                else:
                    lpa, nca = get_lpa_nca_for_point(lat, lon)
                    cache[key] = (lpa, nca)
                    time.sleep(0.15)
                if not lpa_now: row["lpa_name"] = lpa
                if not nca_now: row["nca_name"] = nca
                updated += 1
                rows.append(row)
        if prog is not None:
            done = (len(rows) / max(total, 1))
            prog.progress(done, text=f"Resolving bank LPA/NCA… ({int(done*100)}%)")
    if prog is not None:
        prog.empty()
        if updated:
            st.sidebar.success(f"Updated {updated} bank(s) with LPA/NCA")
    return pd.DataFrame(rows)

backend["Banks"] = enrich_banks_geography(backend["Banks"])
backend["Banks"] = make_bank_key_col(backend["Banks"], backend["Banks"])

# Validate minimal columns
for sheet, cols in {
    "Pricing": ["bank_id","habitat_name","contract_size","tier"],
    "Stock": ["bank_id","habitat_name","stock_id","quantity_available"],
    "HabitatCatalog": ["habitat_name","broader_type","distinctiveness_name"],
}.items():
    missing = [c for c in cols if c not in backend[sheet].columns]
    if missing:
        st.error(f"{sheet} is missing required columns: {missing}")
        st.stop()

# Normalise Pricing; drop Hedgerow
def normalise_pricing(pr_df: pd.DataFrame) -> pd.DataFrame:
    df = pr_df.copy()
    price_cols = [c for c in df.columns if c.strip().lower() in ("price","unit price","unit_price","unitprice")]
    if not price_cols:
        st.error("Pricing sheet must contain a 'Price' column (or 'Unit Price').")
        st.stop()
    df["price"] = pd.to_numeric(df[price_cols[0]], errors="coerce")
    df["tier"] = df["tier"].astype(str).str.strip().str.lower()
    df["contract_size"] = df["contract_size"].astype(str).str.strip().str.lower()
    df["bank_id"] = df["bank_id"].astype(str).str.strip()
    df = make_bank_key_col(df, backend["Banks"])
    if "broader_type" not in df.columns: df["broader_type"] = ""
    if "distinctiveness_name" not in df.columns: df["distinctiveness_name"] = ""
    df["habitat_name"] = df["habitat_name"].astype(str).str.strip()
    # NOTE: Do NOT filter out hedgerows here! They need to be available for hedgerow optimization
    return df

# NOTE: Do NOT filter hedgerows from backend globally - they're needed for hedgerow optimization
backend["Pricing"] = normalise_pricing(backend["Pricing"])

# Distinctiveness mapping
dist_levels_map = {
    sstr(r["distinctiveness_name"]): float(r["level_value"])
    for _, r in backend["DistinctivenessLevels"].iterrows()
}
dist_levels_map.update({k.lower(): v for k, v in list(dist_levels_map.items())})

# Check if we need to refresh the map after optimization (after backend is loaded)
if st.session_state.get("needs_map_refresh", False):
    st.session_state["needs_map_refresh"] = False
    st.rerun()

# ================= Locate UI =================
with st.container():
    st.subheader("1) Locate target site")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        postcode = st.text_input("Postcode (quicker)", key="postcode_input")
    with c2:
        address = st.text_input("Address (if no postcode)", key="address_input")
    with c3:
        run_locate = st.button("Locate", key="locate_btn")

def find_site(postcode: str, address: str):
    if sstr(postcode):
        lat, lon, _ = get_postcode_info(postcode)
    elif sstr(address):
        lat, lon = geocode_address(address)
    else:
        raise RuntimeError("Enter a postcode or an address.")
    lpa_feat = arcgis_point_query(LPA_URL, lat, lon, "LAD24NM")
    nca_feat = arcgis_point_query(NCA_URL, lat, lon, "NCA_Name")
    t_lpa = sstr((lpa_feat.get("attributes") or {}).get("LAD24NM"))
    t_nca = sstr((nca_feat.get("attributes") or {}).get("NCA_Name"))
    lpa_geom_esri = lpa_feat.get("geometry")
    nca_geom_esri = nca_feat.get("geometry")
    lpa_gj = esri_polygon_to_geojson(lpa_geom_esri)
    nca_gj = esri_polygon_to_geojson(nca_geom_esri)
    lpa_nei = [n for n in layer_intersect_names(LPA_URL, lpa_geom_esri, "LAD24NM") if n != t_lpa]
    nca_nei = [n for n in layer_intersect_names(NCA_URL, nca_geom_esri, "NCA_Name") if n != t_nca]
    lpa_nei_norm = [norm_name(n) for n in lpa_nei]
    nca_nei_norm = [norm_name(n) for n in nca_nei]
    
    # Update session state - FIXED VERSION
    st.session_state["target_lpa_name"] = t_lpa
    st.session_state["target_nca_name"] = t_nca
    st.session_state["lpa_neighbors"] = lpa_nei
    st.session_state["nca_neighbors"] = nca_nei
    st.session_state["lpa_neighbors_norm"] = lpa_nei_norm
    st.session_state["nca_neighbors_norm"] = nca_nei_norm
    st.session_state["target_lat"] = lat
    st.session_state["target_lon"] = lon
    st.session_state["lpa_geojson"] = lpa_gj
    st.session_state["nca_geojson"] = nca_gj
    # Clear any previous optimization results when locating new site
    if "last_alloc_df" in st.session_state:
        st.session_state["last_alloc_df"] = None
    st.session_state["optimization_complete"] = False
        
    return t_lpa, t_nca

if run_locate:
    try:
        tl, tn = find_site(postcode, address)
        # Force map refresh on new location
        st.session_state["map_version"] = st.session_state.get("map_version", 0) + 1
        st.success(f"Found LPA: **{tl}** | NCA: **{tn}**")
        st.rerun()
    except Exception as e:
        st.error(f"Location error: {e}")

# Show persistent location banner
if st.session_state["target_lpa_name"] or st.session_state["target_nca_name"]:
    st.success(
        f"LPA: **{st.session_state['target_lpa_name'] or '—'}** | "
        f"NCA: **{st.session_state['target_nca_name'] or '—'}**"
    )

# ================= Map functions (CORRECTED STYLING) =================
# ================= Map functions (CORRECTED STYLING) =================
def build_base_map():
    lat = st.session_state.get("target_lat", None)
    lon = st.session_state.get("target_lon", None)
    lpa_gj = st.session_state.get("lpa_geojson", None)
    nca_gj = st.session_state.get("nca_geojson", None)
    t_lpa = st.session_state.get("target_lpa_name", "")
    t_nca = st.session_state.get("target_nca_name", "")

    if lat is None or lon is None:
        fmap = folium.Map(location=[54.5, -2.5], zoom_start=5, control_scale=True)
    else:
        fmap = folium.Map(location=[lat, lon], zoom_start=10, control_scale=True)
        
        # Target LPA - Bright Red border
        if lpa_gj:
            folium.GeoJson(
                lpa_gj,
                name=f"🎯 Target LPA: {t_lpa}" if t_lpa else "Target LPA",
                style_function=lambda x: {
                    "fillColor": "red", 
                    "color": "red",  # Bright red border
                    "weight": 3, 
                    "fillOpacity": 0.05,  # Very light fill
                    "opacity": 1.0  # Solid border
                },
                tooltip=f"Target LPA: {t_lpa}" if t_lpa else "Target LPA"
            ).add_to(fmap)
        
        # Target NCA - Bright Orange border
        if nca_gj:
            folium.GeoJson(
                nca_gj,
                name=f"🎯 Target NCA: {t_nca}" if t_nca else "Target NCA",
                style_function=lambda x: {
                    "fillColor": "orange", 
                    "color": "orange",  # Bright orange border
                    "weight": 3, 
                    "fillOpacity": 0.05,  # Very light fill
                    "opacity": 1.0  # Solid border
                },
                tooltip=f"Target NCA: {t_nca}" if t_nca else "Target NCA"
            ).add_to(fmap)
        
        # Add target site marker
        folium.CircleMarker(
            [lat, lon], 
            radius=8, 
            color="red", 
            fill=True, 
            fillColor="red",
            popup="🎯 Target Site",
            tooltip="Target Site"
        ).add_to(fmap)

    return fmap

def build_results_map(alloc_df: pd.DataFrame):
    # Start with base map (includes target LPA/NCA)
    fmap = build_base_map()
    lat0 = st.session_state.get("target_lat", None)
    lon0 = st.session_state.get("target_lon", None)

    if alloc_df.empty:
        return fmap

    # Get bank coordinates
    bank_coords: Dict[str, Tuple[float,float]] = {}
    banks_df = backend["Banks"].copy()
    for _, b in banks_df.iterrows():
        bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
        loc = bank_row_to_latlon(b)
        if loc:
            bank_coords[bkey] = (loc[0], loc[1])

    # Process each selected bank
    bank_groups = alloc_df.groupby(["BANK_KEY","bank_name"], dropna=False)
    
    for idx, ((bkey, bname), g) in enumerate(bank_groups):
        try:
            # Get bank coordinates
            latlon = bank_coords.get(sstr(bkey))
            if not latlon:
                continue
                
            lat_b, lon_b = latlon
            
            # Ensure we have catchment data
            cache_key = sstr(bkey)
            if cache_key not in st.session_state["bank_catchment_geo"]:
                continue  # Skip if no catchment data

            bgeo = st.session_state["bank_catchment_geo"][cache_key]
            bank_display_name = sstr(bname) or sstr(bkey)
            
            # Add COMBINED bank boundary (LPA + NCA as one dotted green border)
            # First add the LPA
            if bgeo.get("lpa_gj"):
                folium.GeoJson(
                    bgeo["lpa_gj"],
                    name=f"🏢 {bank_display_name} - Catchment Area",
                    style_function=lambda x: {
                        "fillColor": "green", 
                        "color": "green",  # Green border
                        "weight": 2, 
                        "fillOpacity": 0.1,  # Light green fill
                        "opacity": 0.8,
                        "dashArray": "5,5"  # Dotted border
                    },
                    tooltip=f"Bank: {bank_display_name} - LPA: {sstr(bgeo.get('lpa_name', 'Unknown'))}"
                ).add_to(fmap)
            
            # Then add the NCA with same styling to create unified appearance
            if bgeo.get("nca_gj"):
                folium.GeoJson(
                    bgeo["nca_gj"],
                    name=f"🌿 {bank_display_name} - Extended Catchment",
                    style_function=lambda x: {
                        "fillColor": "green", 
                        "color": "green",  # Green border
                        "weight": 2, 
                        "fillOpacity": 0.05,  # Very light green fill for NCA
                        "opacity": 0.8,
                        "dashArray": "5,5"  # Dotted border
                    },
                    tooltip=f"Bank: {bank_display_name} - NCA: {sstr(bgeo.get('nca_name', 'Unknown'))}"
                ).add_to(fmap)

            # Create detailed popup for bank marker
            habitat_details = []
            for _, r in g.sort_values('units_supplied', ascending=False).head(6).iterrows():
                habitat_details.append(
                    f"• {sstr(r['supply_habitat'])}: {float(r['units_supplied']):.2f} units ({sstr(r['tier'])})"
                )
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 300px;">
                <h4 style="margin: 0 0 10px 0; color: green;">🏢 {bank_display_name}</h4>
                <p><strong>📍 LPA:</strong> {sstr(bgeo.get('lpa_name', 'Unknown'))}</p>
                <p><strong>🌿 NCA:</strong> {sstr(bgeo.get('nca_name', 'Unknown'))}</p>
                <p><strong>📊 Total Units:</strong> {g['units_supplied'].sum():.2f}</p>
                <p><strong>💰 Total Cost:</strong> £{g['cost'].sum():,.0f}</p>
                <p><strong>🌱 Habitats:</strong></p>
                <ul style="margin: 5px 0; padding-left: 15px;">
                    {''.join([f'<li>{detail}</li>' for detail in habitat_details])}
                </ul>
            </div>
            """
            
            # Add bank marker - green to match catchment
            folium.Marker(
                [lat_b, lon_b],
                icon=folium.Icon(color="green", icon="building", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=350),
                tooltip=f"🏢 {bank_display_name} - Click for details"
            ).add_to(fmap)

            # Add supply route - green to match bank theme
            if lat0 is not None and lon0 is not None:
                folium.PolyLine(
                    locations=[[lat0, lon0], [lat_b, lon_b]],
                    weight=3, 
                    opacity=0.7, 
                    color="green",  # Green route line
                    dash_array="8,4",
                    tooltip=f"Supply route: Target → {bank_display_name}"
                ).add_to(fmap)

        except Exception as e:
            st.warning(f"Failed to add bank {sstr(bname) or sstr(bkey)} to map: {e}")

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap

# ================= Map Container (FIXED VERSION) =================
with st.container():
    st.markdown("### Map")
    
    # Determine map state
    has_location = st.session_state.get("target_lat") is not None
    has_results = (isinstance(st.session_state.get("last_alloc_df"), pd.DataFrame) and 
                  not st.session_state.get("last_alloc_df").empty and 
                  st.session_state.get("optimization_complete", False))
    
    # Show status and debug if needed
    if has_results:
        num_banks = st.session_state["last_alloc_df"]["BANK_KEY"].nunique()
        banks_list = st.session_state["last_alloc_df"]["BANK_KEY"].unique().tolist()
        st.caption(f"📍 Optimization results: {num_banks} selected bank(s)")
        
        # Show catchment status
        loaded_catchments = len(st.session_state.get("bank_catchment_geo", {}))
        st.info(f"🗺️ Banks: {', '.join(banks_list)} | Catchments loaded: {loaded_catchments}")
        
    elif has_location:
        st.caption("📍 Showing target location with LPA/NCA boundaries")
    else:
        st.caption("📍 UK overview - use 'Locate' to center on your target site")
    
    # Manual refresh option (always available)
    if st.button("🔄 Refresh Map", help="Reload map with fresh data"):
        if has_results:
            # Clear and reload catchment data
            st.session_state["bank_catchment_geo"] = {}
            with st.spinner("Reloading bank catchments..."):
                alloc_df = st.session_state["last_alloc_df"]
                # Same loading logic as above
                selected_banks = alloc_df["BANK_KEY"].unique()
                bank_coords = {}
                banks_df = backend["Banks"].copy()
                for _, b in banks_df.iterrows():
                    bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
                    loc = bank_row_to_latlon(b)
                    if loc:
                        bank_coords[bkey] = (loc[0], loc[1])
                
                for bkey in selected_banks:
                    if sstr(bkey) in bank_coords:
                        try:
                            lat_b, lon_b = bank_coords[sstr(bkey)]
                            b_lpa_name, b_lpa_gj, b_nca_name, b_nca_gj = get_catchment_geo_for_point(lat_b, lon_b)
                            st.session_state["bank_catchment_geo"][sstr(bkey)] = {
                                "lpa_name": b_lpa_name, "lpa_gj": b_lpa_gj,
                                "nca_name": b_nca_name, "nca_gj": b_nca_gj,
                            }
                        except Exception as e:
                            st.warning(f"Failed to load {bkey}: {e}")
        st.rerun()
    
    # Build and render map
    try:
        if has_results:
            current_map = build_results_map(st.session_state["last_alloc_df"])
        else:
            current_map = build_base_map()
        
        # Use a simple, stable key
        map_key = "bng_stable_map"
        
        # Render with folium_static for maximum stability
        if folium_static:
            folium_static(current_map, width=None, height=520)
        else:
            st_folium(current_map, height=520, use_container_width=True, key=map_key)

    except Exception as e:
        st.error(f"Map rendering failed: {e}")

# ================= Demand =================
st.subheader("2) Demand (units required)")
NET_GAIN_LABEL = "Net Gain (Low-equivalent)"
NET_GAIN_HEDGEROW_LABEL = "Net Gain (Hedgerows)"

HAB_CHOICES = sorted(
    [sstr(x) for x in backend["HabitatCatalog"]["habitat_name"].dropna().unique().tolist()] + [NET_GAIN_LABEL]
) + [NET_GAIN_HEDGEROW_LABEL, NET_GAIN_WATERCOURSE_LABEL]  # add watercourses NG

with st.container(border=True):
    st.markdown("**Add habitats one by one** (type to search the catalog):")
    to_delete = []
    for idx, row in enumerate(st.session_state.demand_rows):
        c1, c2, c3 = st.columns([0.62, 0.28, 0.10])
        with c1:
            # Don't autopopulate - use None as default index if habitat is empty
            default_idx = None
            if row["habitat_name"] and row["habitat_name"] in HAB_CHOICES:
                default_idx = HAB_CHOICES.index(row["habitat_name"])
            
            st.session_state.demand_rows[idx]["habitat_name"] = st.selectbox(
                "Habitat", HAB_CHOICES,
                index=default_idx,
                key=f"hab_{row['id']}",
                help="Start typing to filter",
            )
        with c2:
            st.session_state.demand_rows[idx]["units"] = st.number_input(
                "Units", min_value=0.0, step=0.01, value=float(row.get("units", 0.0)), key=f"units_{row['id']}"
            )
        with c3:
            if st.button("🗑️", key=f"del_{row['id']}", help="Remove this row"):
                to_delete.append(row["id"])
    
    if to_delete:
        st.session_state.demand_rows = [r for r in st.session_state.demand_rows if r["id"] not in to_delete]
        st.rerun()

    cc1, cc2, cc3, cc4, cc5 = st.columns([0.22, 0.22, 0.22, 0.22, 0.12])
    with cc1:
        if st.button("➕ Add habitat", key="add_hab_btn"):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": "", "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc2:
        if st.button("➕ Net Gain (Low-equivalent)", key="add_ng_btn",
                     help="Adds a 'Net Gain' line. Trades like Low distinctiveness (can source from any area habitat)."):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": NET_GAIN_LABEL, "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc3:
        if st.button("➕ Net Gain (Hedgerows)", key="add_ng_hedge_btn",
                     help="Adds a 'Net Gain (Hedgerows)' line. Can be fulfilled using any hedgerow habitat credit."):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": NET_GAIN_HEDGEROW_LABEL, "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc4:
        if st.button("➕ Net Gain (Watercourses)", key="add_ng_water_btn",
                     help="Adds a 'Net Gain (Watercourses)' line. Can be fulfilled using any watercourse habitat credit."):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": NET_GAIN_WATERCOURSE_LABEL, "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc5:
        if st.button("🧹 Clear all", key="clear_all_btn"):
            # Reset existing rows to empty state (preserves row count & IDs)
            for row in st.session_state.demand_rows:
                row["habitat_name"] = ""
                row["units"] = 0.0
            st.rerun()


total_units = sum([float(r.get("units", 0.0) or 0.0) for r in st.session_state.demand_rows])
st.metric("Total units", f"{total_units:.2f}")

demand_df = pd.DataFrame(
    [{"habitat_name": sstr(r["habitat_name"]), "units_required": float(r.get("units", 0.0) or 0.0)}
     for r in st.session_state.demand_rows if sstr(r["habitat_name"]) and float(r.get("units", 0.0) or 0.0) > 0]
)

# Display demand (hedgerow units are now supported)
if not demand_df.empty:
    st.dataframe(demand_df, use_container_width=True, hide_index=True)
    
    # Show info if hedgerow units are included
    hedgerow_units = [h for h in demand_df["habitat_name"] if is_hedgerow(h)]
    if hedgerow_units:
        st.info(f"ℹ️ Hedgerow units detected: {', '.join(sorted(set(hedgerow_units)))}. These will be optimized using hedgerow-specific trading rules.")
else:
    st.info("Add at least one habitat and units to continue.", icon="ℹ️")

# ================= Legality =================

def enforce_catalog_rules_official(demand_row, supply_row, dist_levels_map_local, explicit_rule: bool) -> bool:
    if explicit_rule:
        return True
    dh = sstr(demand_row.get("habitat_name"))
    if dh == NET_GAIN_LABEL:
        return True
    sh = sstr(supply_row.get("habitat_name"))
    d_group = sstr(demand_row.get("broader_type"))
    s_group = sstr(supply_row.get("broader_type"))
    d_dist_name = sstr(demand_row.get("distinctiveness_name"))
    s_dist_name = sstr(supply_row.get("distinctiveness_name"))
    d_key = d_dist_name.lower()
    d_val = dist_levels_map_local.get(d_dist_name, dist_levels_map_local.get(d_key, -1e9))
    s_val = dist_levels_map_local.get(s_dist_name, dist_levels_map_local.get(s_dist_name.lower(), -1e-9))

    if d_key == "low" or dh == NET_GAIN_LABEL:
        return True
    if d_key == "medium":
        same_group = (d_group and s_group and d_group == s_group)
        higher_distinctiveness = (s_val > d_val)
        return bool(same_group or higher_distinctiveness)
    return sh == dh  # High / Very High exactly like-for-like

def enforce_hedgerow_rules(demand_row, supply_row, dist_levels_map_local) -> bool:
    """
    Hedgerow trading rules:
    - Very High: Same habitat required
    - High: Like for like or better
    - Medium: Same distinctiveness or better habitat required
    - Low: Same distinctiveness or better habitat required
    - Very Low: Same distinctiveness or better habitat required
    - Net Gain: Can be covered using anything
    """
    dh = sstr(demand_row.get("habitat_name"))
    sh = sstr(supply_row.get("habitat_name"))
    d_dist_name = sstr(demand_row.get("distinctiveness_name"))
    s_dist_name = sstr(supply_row.get("distinctiveness_name"))
    d_key = d_dist_name.lower()
    d_val = dist_levels_map_local.get(d_dist_name, dist_levels_map_local.get(d_key, -1e9))
    s_val = dist_levels_map_local.get(s_dist_name, dist_levels_map_local.get(s_dist_name.lower(), -1e-9))
    
    # Net Gain (both regular and hedgerow) can be covered by anything
    if dh == NET_GAIN_LABEL or dh == "Net Gain (Hedgerows)":
        return True
    
    # Very High - Same habitat required
    if d_key in ["very high", "v.high"]:
        return sh == dh
    
    # High - Like for like or better (same habitat or higher distinctiveness)
    if d_key == "high":
        return (sh == dh) or (s_val > d_val)
    
    # Medium, Low, Very Low - Same distinctiveness or better
    if d_key in ["medium", "low", "very low", "v.low"]:
        return s_val >= d_val
    
    # Default: allow it
    return True

def enforce_watercourse_rules(demand_row, supply_row, dist_levels_map_local) -> bool:
    """
    Watercourse trading rules (mirrors hedgerow approach until you specify otherwise):
    - Very High: Same habitat required
    - High: Like for like or better (same habitat or higher distinctiveness)
    - Medium, Low, Very Low: Same distinctiveness or better
    - Net Gain (Watercourses): Anything within watercourse ledger
    """
    dh = sstr(demand_row.get("habitat_name"))
    sh = sstr(supply_row.get("habitat_name"))
    d_dist_name = sstr(demand_row.get("distinctiveness_name"))
    s_dist_name = sstr(supply_row.get("distinctiveness_name"))
    d_key = d_dist_name.lower()
    d_val = dist_levels_map_local.get(d_dist_name, dist_levels_map_local.get(d_key, -1e9))
    s_val = dist_levels_map_local.get(s_dist_name, dist_levels_map_local.get(s_dist_name.lower(), -1e-9))

    if dh == NET_GAIN_WATERCOURSE_LABEL:
        return True
    if d_key in ["very high", "v.high"]:
        return sh == dh
    if d_key == "high":
        return (sh == dh) or (s_val > d_val)
    if d_key in ["medium", "low", "very low", "v.low"]:
        return s_val >= d_val
    return True

# ================= Options builder =================
def select_size_for_demand(demand_df: pd.DataFrame, pricing_df: pd.DataFrame) -> str:
    present = pricing_df["contract_size"].drop_duplicates().tolist()
    total = float(demand_df["units_required"].sum())
    return select_contract_size(total, present)

def prepare_options(demand_df: pd.DataFrame,
                    chosen_size: str,
                    target_lpa: str, target_nca: str,
                    lpa_neigh: List[str], nca_neigh: List[str],
                    lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[List[dict], Dict[str, float], Dict[str, str]]:
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()
    Trading = backend.get("TradingRules", pd.DataFrame())

    for df, cols in [
        (Banks, ["bank_id","bank_name","BANK_KEY","lpa_name","nca_name","lat","lon","postcode","address"]),
        (Catalog, ["habitat_name","broader_type","distinctiveness_name"]),
        (Stock, ["habitat_name","stock_id","bank_id","quantity_available","bank_name","BANK_KEY"]),
        (Pricing, ["habitat_name","contract_size","tier","bank_id","BANK_KEY","price","broader_type","distinctiveness_name","bank_name"]),
        (Trading, ["demand_habitat","allowed_supply_habitat","min_distinctiveness_name","companion_habitat"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(sstr)

    Stock = make_bank_key_col(Stock, Banks)

    stock_full = Stock.merge(
        Banks[["bank_id","bank_name","lpa_name","nca_name"]],
        on="bank_id", how="left"
    ).merge(Catalog, on="habitat_name", how="left")
    stock_full = stock_full[~stock_full["habitat_name"].map(is_hedgerow)].copy()

    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()

    pc_join = pricing_cs.merge(
        Catalog[["habitat_name","broader_type","distinctiveness_name"]],
        on="habitat_name", how="left", suffixes=("", "_cat")
    )
    pc_join["broader_type_eff"] = np.where(pc_join["broader_type"].astype(str).str.len()>0,
                                           pc_join["broader_type"], pc_join["broader_type_cat"])
    pc_join["distinctiveness_name_eff"] = np.where(pc_join["distinctiveness_name"].astype(str).str.len()>0,
                                                   pc_join["distinctiveness_name"], pc_join["distinctiveness_name_cat"])
    for c in ["broader_type_eff", "distinctiveness_name_eff", "tier", "bank_id", "habitat_name", "BANK_KEY", "bank_name"]:
        if c in pc_join.columns:
            pc_join[c] = pc_join[c].map(sstr)
    pricing_enriched = pc_join[~pc_join["habitat_name"].map(is_hedgerow)].copy()

    def dval(name: Optional[str]) -> float:
        key = sstr(name)
        return dist_levels_map.get(key, dist_levels_map.get(key.lower(), -1e9))

    def find_price_for_supply(bank_key: str,
                              supply_habitat: str,
                              tier: str,
                              demand_broader: str,
                              demand_dist: str) -> Optional[Tuple[float, str, str]]:
        # Exact row first
        pr_exact = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                                    (pricing_enriched["tier"] == tier) &
                                    (pricing_enriched["habitat_name"] == supply_habitat)]
        if not pr_exact.empty:
            r = pr_exact.sort_values("price").iloc[0]
            return float(r["price"]), "exact", sstr(r["habitat_name"])

        d_key = sstr(demand_dist).lower()

        # Low / Net Gain — cheapest per bank/tier as proxy if exact not present
        if d_key == "low":
            grp = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                                   (pricing_enriched["tier"] == tier)]
            if not grp.empty:
                r = grp.sort_values("price").iloc[0]
                return float(r["price"]), "any-low-proxy", sstr(r["habitat_name"])
            return None

        if d_key == "medium":
            d_num = dval(demand_dist)
            grp = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                                   (pricing_enriched["tier"] == tier)]
            grp = grp[(grp["broader_type_eff"].astype(str).str.len() > 0) &
                      (grp["distinctiveness_name_eff"].astype(str).str.len() > 0)]
            if grp.empty:
                return None
            grp_same = grp[grp["broader_type_eff"].map(sstr) == sstr(demand_broader)].copy()
            if not grp_same.empty:
                grp_same["_dval"] = grp_same["distinctiveness_name_eff"].map(dval)
                grp_same = grp_same[grp_same["_dval"] >= d_num]
                if not grp_same.empty:
                    r = grp_same.sort_values("price").iloc[0]
                    return float(r["price"]), "group-proxy", sstr(r["habitat_name"])
            grp_any_higher = grp.assign(_dval=grp["distinctiveness_name_eff"].map(dval))
            grp_any_higher = grp_any_higher[grp_any_higher["_dval"] > d_num]
            if not grp_any_higher.empty:
                r = grp_any_higher.sort_values("price").iloc[0]
                return float(r["price"]), "group-proxy", sstr(r["habitat_name"])
            return None

        return None  # High/Very High: exact only

    def find_catalog_name(substr: str) -> Optional[str]:
        m = Catalog[Catalog["habitat_name"].str.contains(substr, case=False, na=False)]
        return sstr(m["habitat_name"].iloc[0]) if not m.empty else None

    ORCHARD_NAME = find_catalog_name("Traditional Orchard")
    SCRUB_NAME = find_catalog_name("Mixed Scrub") or find_catalog_name("scrub") or find_catalog_name("bramble")

    options: List[dict] = []
    stock_caps: Dict[str, float] = {}
    stock_bankkey: Dict[str, str] = {}
    for _, s in Stock.iterrows():
        stock_caps[sstr(s["stock_id"])] = float(s.get("quantity_available", 0) or 0.0)
        stock_bankkey[sstr(s["stock_id"])] = sstr(s.get("BANK_KEY") or s.get("bank_id"))

    for di, drow in demand_df.iterrows():
        dem_hab = sstr(drow["habitat_name"])
        
        # Skip hedgerow demand in area habitat options (hedgerows handled separately)
        if is_hedgerow(dem_hab):
            continue

        if dem_hab == NET_GAIN_LABEL:
            d_broader = ""
            d_dist = "Low"
        else:
            dcat = Catalog[Catalog["habitat_name"] == dem_hab]
            d_broader = sstr(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
            d_dist = sstr(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""

        # Candidate stock by legality
        cand_parts = []
        explicit = False
        if "TradingRules" in backend and not backend["TradingRules"].empty and dem_hab in set(backend["TradingRules"]["demand_habitat"].astype(str)):
            explicit = True
            for _, rule in backend["TradingRules"][backend["TradingRules"]["demand_habitat"] == dem_hab].iterrows():
                sh = sstr(rule["allowed_supply_habitat"])
                if is_hedgerow(sh):
                    continue
                s_min = sstr(rule.get("min_distinctiveness_name"))
                df_s = stock_full[stock_full["habitat_name"] == sh].copy()
                if s_min:
                    df_s = df_s[df_s["distinctiveness_name"].map(lambda x: dist_levels_map.get(sstr(x), -1e9)) >=
                                dist_levels_map.get(sstr(s_min), -1e9)]
                if not df_s.empty: cand_parts.append(df_s)

        if not cand_parts:
            key = sstr(d_dist).lower()
            if key == "low" or dem_hab == NET_GAIN_LABEL:
                df_s = stock_full.copy()
            elif key == "medium":
                same_group = stock_full["broader_type"].fillna("").astype(str).map(sstr).eq(d_broader)
                higher_dist = stock_full["distinctiveness_name"].map(lambda x: dist_levels_map.get(sstr(x), -1e9)) > \
                              dist_levels_map.get(sstr(d_dist), -1e9)
                df_s = stock_full[same_group | higher_dist].copy()
            else:
                df_s = stock_full[stock_full["habitat_name"] == dem_hab].copy()
            if not df_s.empty: cand_parts.append(df_s)

        if not cand_parts:
            continue

        candidates = pd.concat(cand_parts, ignore_index=True)
        candidates = candidates[~candidates["habitat_name"].map(is_hedgerow)].copy()

        # Single-habitat options
        for _, srow in candidates.iterrows():
            if not enforce_catalog_rules_official(
                pd.Series({"habitat_name": dem_hab, "broader_type": d_broader, "distinctiveness_name": d_dist}),
                srow, dist_levels_map, explicit_rule=explicit
            ):
                continue
            tier = tier_for_bank(
                srow.get("lpa_name",""), srow.get("nca_name",""),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
            )
            bank_key = sstr(srow.get("BANK_KEY") or srow.get("bank_name") or srow.get("bank_id"))
            price_info = find_price_for_supply(
                bank_key=bank_key,
                supply_habitat=srow["habitat_name"],
                tier=tier,
                demand_broader=d_broader,
                demand_dist=d_dist,
            )
            if not price_info:
                continue

            unit_price, price_source, price_hab_used = price_info
            cap = float(srow.get("quantity_available", 0) or 0.0)
            if cap <= 0:
                continue
            options.append({
                "type": "normal",
                "demand_idx": di,
                "demand_habitat": dem_hab,
                "BANK_KEY": bank_key,
                "bank_name": sstr(srow.get("bank_name")),
                "bank_id": sstr(srow.get("bank_id")),
                "supply_habitat": srow["habitat_name"],
                "tier": tier,
                "proximity": tier,
                "unit_price": float(unit_price),
                "stock_use": {sstr(srow["stock_id"]): 1.0},
                "price_source": price_source,
                "price_habitat": price_hab_used,
            })

        # Paired Orchard+Other for MEDIUM — ADJACENT and FAR
        if sstr(d_dist).lower() == "medium" and ORCHARD_NAME:
            banks_keys = stock_full["BANK_KEY"].dropna().unique().tolist()
            for bk in banks_keys:
                orch_rows = stock_full[(stock_full["BANK_KEY"] == bk) & (stock_full["habitat_name"] == ORCHARD_NAME)]
                if orch_rows.empty:
                    continue
                
                # Get "Other" candidates: area habitats with distinctiveness <= Medium, positive stock
                other_candidates = stock_full[
                    (stock_full["BANK_KEY"] == bk) &
                    (stock_full["habitat_name"] != ORCHARD_NAME) &
                    (~stock_full["habitat_name"].map(is_hedgerow)) &
                    (stock_full["distinctiveness_name"].map(lambda x: dval(x) <= dval("Medium"))) &
                    (stock_full["quantity_available"].astype(float) > 0)
                ].copy()
                
                if other_candidates.empty:
                    continue
                
                # Process each Orchard stock entry
                for _, o in orch_rows.iterrows():
                    cap_o = float(o.get("quantity_available", 0) or 0.0)
                    if cap_o <= 0:
                        continue
                    
                    # For each tier (adjacent and far), find the best "Other" component
                    for target_tier in ["adjacent", "far"]:
                        # Find "Other" candidates at this tier with valid prices
                        tier_other_candidates = []
                        for _, other_row in other_candidates.iterrows():
                            tier_test = tier_for_bank(
                                sstr(other_row.get("lpa_name")), sstr(other_row.get("nca_name")),
                                target_lpa, target_nca, lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
                            )
                            if tier_test != target_tier:
                                continue
                            
                            # Check if we can price this "Other" component
                            pi_other = find_price_for_supply(bk, other_row["habitat_name"], target_tier, d_broader, d_dist)
                            if not pi_other:
                                continue
                            
                            tier_other_candidates.append({
                                "row": other_row,
                                "price": float(pi_other[0]),
                                "price_info": pi_other,
                                "cap": float(other_row.get("quantity_available", 0) or 0.0)
                            })
                        
                        if not tier_other_candidates:
                            continue
                        
                        # Sort by price (ascending), then by available stock (descending) for tie-breaking
                        tier_other_candidates.sort(key=lambda x: (x["price"], -x["cap"]))
                        best_other = tier_other_candidates[0]
                        
                        # Get Orchard price at this tier
                        pi_o = find_price_for_supply(bk, ORCHARD_NAME, target_tier, d_broader, d_dist)
                        if not pi_o:
                            continue
                        
                        price_o = float(pi_o[0])
                        price_other = best_other["price"]
                        other_row = best_other["row"]
                        pi_other = best_other["price_info"]
                        
                        # Calculate blended price and stock_use based on tier
                        if target_tier == "adjacent":
                            # Adjacent: 1.00 Orchard + 1/3 Other per 1.00 demand unit (75%/25% split)
                            blended_price = (1.00 * price_o + (1/3) * price_other) / (4/3)
                            stock_use_orchard = 1.00
                            stock_use_other = 1/3
                        else:  # far
                            # Far: 0.50 Orchard + 0.50 Other per 1.00 demand unit
                            blended_price = 0.5 * price_o + 0.5 * price_other
                            stock_use_orchard = 0.50
                            stock_use_other = 0.50
                        
                        options.append({
                            "type": "paired",
                            "demand_idx": di,
                            "demand_habitat": dem_hab,
                            "BANK_KEY": bk,
                            "bank_name": sstr(o.get("bank_name")),
                            "bank_id": sstr(o.get("bank_id")),
                            "supply_habitat": f"{ORCHARD_NAME} + {sstr(other_row['habitat_name'])}",
                            "tier": target_tier,
                            "proximity": target_tier,
                            "unit_price": blended_price,
                            "stock_use": {sstr(o["stock_id"]): stock_use_orchard, sstr(other_row["stock_id"]): stock_use_other},
                            "price_source": "group-proxy",
                            "price_habitat": f"{pi_o[2]} + {pi_other[2]}",
                            "paired_parts": [
                                {"habitat": ORCHARD_NAME, "unit_price": price_o, "stock_use": stock_use_orchard},
                                {"habitat": sstr(other_row["habitat_name"]), "unit_price": price_other, "stock_use": stock_use_other},
                            ],
                        })

    return options, stock_caps, stock_bankkey

def prepare_hedgerow_options(demand_df: pd.DataFrame,
                              chosen_size: str,
                              target_lpa: str, target_nca: str,
                              lpa_neigh: List[str], nca_neigh: List[str],
                              lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[List[dict], Dict[str, float], Dict[str, str]]:
    """Prepare hedgerow unit options using specific hedgerow trading rules"""
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()
    
    for df, cols in [
        (Banks, ["bank_id","bank_name","BANK_KEY","lpa_name","nca_name"]),
        (Catalog, ["habitat_name","broader_type","distinctiveness_name"]),
        (Stock, ["habitat_name","stock_id","bank_id","quantity_available"]),
        (Pricing, ["habitat_name","contract_size","tier","bank_id","BANK_KEY","price"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(sstr)
    
    Stock = make_bank_key_col(Stock, Banks)
    
    # Filter for ONLY hedgerow habitats
    # Ensure bank_name is available from Banks
    banks_cols = ["bank_id"]
    for col in ["bank_name", "lpa_name", "nca_name"]:
        if col in Banks.columns:
            banks_cols.append(col)
    
    stock_full = Stock.merge(
        Banks[banks_cols].drop_duplicates(),
        on="bank_id", how="left"
    ).merge(Catalog, on="habitat_name", how="left")
    
    # Ensure bank_name exists (fallback to BANK_KEY if not present)
    if "bank_name" not in stock_full.columns:
        if "BANK_KEY" in stock_full.columns:
            stock_full["bank_name"] = stock_full["BANK_KEY"]
        else:
            stock_full["bank_name"] = stock_full["bank_id"]
    
    stock_full = stock_full[stock_full["habitat_name"].map(is_hedgerow)].copy()
    
    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()
    pricing_enriched = pricing_cs.merge(
        Catalog[["habitat_name","broader_type","distinctiveness_name"]],
        on="habitat_name", how="left"
    )
    pricing_enriched = pricing_enriched[pricing_enriched["habitat_name"].map(is_hedgerow)].copy()
    
    options = []
    stock_caps = {}
    stock_bankkey = {}
    
    for demand_idx, demand_row in demand_df.iterrows():
        dem_hab = sstr(demand_row.get("habitat_name"))
        
        # Skip non-hedgerow demand (but include hedgerow net gain)
        if not is_hedgerow(dem_hab):
            continue
        
        demand_units = float(demand_row.get("units_required", 0.0))
        if demand_units <= 0:
            continue
        
        # Get demand distinctiveness
        if dem_hab == "Net Gain (Hedgerows)":
            demand_dist = "Low"  # Hedgerow Net Gain trades like Low for hedgerows
            demand_broader = ""
        else:
            cat_match = Catalog[Catalog["habitat_name"] == dem_hab]
            if cat_match.empty:
                continue
            demand_dist = sstr(cat_match.iloc[0]["distinctiveness_name"])
            demand_broader = sstr(cat_match.iloc[0]["broader_type"])
        
        demand_cat_row = pd.Series({
            "habitat_name": dem_hab,
            "distinctiveness_name": demand_dist,
            "broader_type": demand_broader
        })
        
        # Find all eligible supply habitats
        for _, supply_row in stock_full.iterrows():
            supply_hab = sstr(supply_row["habitat_name"])
            
            # Check hedgerow trading rules
            if not enforce_hedgerow_rules(demand_cat_row, supply_row, dist_levels_map):
                continue
            
            bank_key = sstr(supply_row["BANK_KEY"])
            stock_id = sstr(supply_row["stock_id"])
            qty_avail = float(supply_row.get("quantity_available", 0.0))
            
            if qty_avail <= 0:
                continue
            
            tier = tier_for_bank(
                sstr(supply_row.get("lpa_name")), sstr(supply_row.get("nca_name")),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
            )
            
            # Find price
            pr_match = pricing_enriched[
                (pricing_enriched["BANK_KEY"] == bank_key) &
                (pricing_enriched["tier"] == tier) &
                (pricing_enriched["habitat_name"] == supply_hab)
            ]
            
            if pr_match.empty:
                # Try to find any price for this bank/tier as fallback
                pr_match = pricing_enriched[
                    (pricing_enriched["BANK_KEY"] == bank_key) &
                    (pricing_enriched["tier"] == tier)
                ]
                if pr_match.empty:
                    continue
                price = float(pr_match.iloc[0]["price"])
            else:
                price = float(pr_match.iloc[0]["price"])
            
            options.append({
                "demand_idx": demand_idx,
                "demand_habitat": dem_hab,
                "supply_habitat": supply_hab,
                "bank_id": sstr(supply_row["bank_id"]),
                "bank_name": sstr(supply_row["bank_name"]),
                "BANK_KEY": bank_key,
                "stock_id": stock_id,
                "tier": tier,
                "unit_price": price,
                "cost_per_unit": price,
                "stock_use": {stock_id: 1.0},
                "type": "normal",          # <-- add this
                "proximity": tier   
            })
            
            stock_caps[stock_id] = qty_avail
            stock_bankkey[stock_id] = bank_key
    
    return options, stock_caps, stock_bankkey

# --- Watercourse options builder (ledger-scoped) ---
def prepare_watercourse_options(demand_df: pd.DataFrame,
                                chosen_size: str,
                                target_lpa: str, target_nca: str,
                                lpa_neigh: List[str], nca_neigh: List[str],
                                lpa_neigh_norm: List[str], nca_neigh_norm: List[str]
                                ) -> Tuple[List[dict], Dict[str, float], Dict[str, str]]:
    """Build candidate options for watercourse ledger using UmbrellaType='watercourse'."""
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()

    # Normalise strings
    for df, cols in [
        (Banks,   ["bank_id","bank_name","BANK_KEY","lpa_name","nca_name"]),
        (Catalog, ["habitat_name","broader_type","distinctiveness_name","UmbrellaType"]),
        (Stock,   ["habitat_name","stock_id","bank_id","quantity_available","BANK_KEY"]),
        (Pricing, ["habitat_name","contract_size","tier","bank_id","BANK_KEY","price"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(sstr)

    # Ensure BANK_KEY exists on Stock
    Stock = make_bank_key_col(Stock, Banks)

    # Keep only watercourse habitats by UmbrellaType
    wc_catalog = Catalog[Catalog["UmbrellaType"].astype(str).str.lower() == "watercourse"]
    wc_habs = set(wc_catalog["habitat_name"].astype(str))

    stock_full = (
        Stock[Stock["habitat_name"].isin(wc_habs)]
        .merge(Banks[["bank_id","bank_name","lpa_name","nca_name"]].drop_duplicates(),
               on="bank_id", how="left")
        .merge(Catalog[["habitat_name","broader_type","distinctiveness_name","UmbrellaType"]],
               on="habitat_name", how="left")
    )

    pricing_enriched = (
        Pricing[(Pricing["contract_size"] == chosen_size) & (Pricing["habitat_name"].isin(wc_habs))]
        .merge(Catalog[["habitat_name","broader_type","distinctiveness_name","UmbrellaType"]],
               on="habitat_name", how="left")
    )

    options: List[dict] = []
    stock_caps: Dict[str, float] = {}
    stock_bankkey: Dict[str, str] = {}

    for demand_idx, demand_row in demand_df.iterrows():
        dem_hab = sstr(demand_row.get("habitat_name"))

        # Only handle watercourse demands (including NG watercourses)
        # Uses UmbrellaType to decide ledger
        if "UmbrellaType" in Catalog.columns:
            if dem_hab != NET_GAIN_WATERCOURSE_LABEL:
                m = Catalog[Catalog["habitat_name"].astype(str).str.strip() == dem_hab]
                umb = sstr(m.iloc[0]["UmbrellaType"]).lower() if not m.empty else ""
                if umb != "watercourse":
                    continue
        else:
            # Fallback: text heuristic (not ideal, but keeps behavior if column is missing)
            if dem_hab != NET_GAIN_WATERCOURSE_LABEL and not is_watercourse(dem_hab):
                continue

        demand_units = float(demand_row.get("units_required", 0.0))
        if demand_units <= 0:
            continue

        if dem_hab == NET_GAIN_WATERCOURSE_LABEL:
            demand_dist = "Low"     # NG trades like Low within this ledger
            demand_broader = ""
        else:
            cat_match = Catalog[Catalog["habitat_name"] == dem_hab]
            if cat_match.empty:
                continue
            demand_dist = sstr(cat_match.iloc[0]["distinctiveness_name"])
            demand_broader = sstr(cat_match.iloc[0]["broader_type"])

        demand_cat_row = pd.Series({
            "habitat_name": dem_hab,
            "distinctiveness_name": demand_dist,
            "broader_type": demand_broader
        })

        for _, supply_row in stock_full.iterrows():
            supply_hab = sstr(supply_row["habitat_name"])
            if supply_hab not in wc_habs:
                continue

            # Ledger-specific rule check
            if not enforce_watercourse_rules(demand_cat_row, supply_row, dist_levels_map):
                continue

            bank_key = sstr(supply_row["BANK_KEY"])
            stock_id = sstr(supply_row["stock_id"])
            qty_avail = float(supply_row.get("quantity_available", 0.0))
            if qty_avail <= 0:
                continue

            tier = tier_for_bank(
                sstr(supply_row.get("lpa_name")), sstr(supply_row.get("nca_name")),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
            )

            # Find exact price, else fallback to any watercourse price in same bank/tier
            pr_match = pricing_enriched[
                (pricing_enriched["BANK_KEY"] == bank_key) &
                (pricing_enriched["tier"] == tier) &
                (pricing_enriched["habitat_name"] == supply_hab)
            ]
            if pr_match.empty:
                pr_match = pricing_enriched[
                    (pricing_enriched["BANK_KEY"] == bank_key) &
                    (pricing_enriched["tier"] == tier)
                ]
                if pr_match.empty:
                    continue
                price = float(pr_match.iloc[0]["price"])
            else:
                price = float(pr_match.iloc[0]["price"])

            options.append({
                "demand_idx": demand_idx,
                "demand_habitat": dem_hab,
                "supply_habitat": supply_hab,
                "bank_id": sstr(supply_row["bank_id"]),
                "bank_name": sstr(supply_row["bank_name"]),
                "BANK_KEY": bank_key,
                "stock_id": stock_id,
                "tier": tier,
                "unit_price": price,
                "cost_per_unit": price,
                "stock_use": {stock_id: 1.0},
                "type": "normal",
                "proximity": tier,
            })

            stock_caps[stock_id] = qty_avail
            stock_bankkey[stock_id] = bank_key

    return options, stock_caps, stock_bankkey


# ================= Optimiser =================
def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str],
             lpa_neigh_norm: List[str], nca_neigh_norm: List[str]
             ) -> Tuple[pd.DataFrame, float, str]:
    # Pick contract size from total demand (unchanged)
    chosen_size = select_size_for_demand(demand_df, backend["Pricing"])

    # ---- Build options per ledger ----
    # 1) Area (non-hedgerow, non-watercourse)
    options_area, caps_area, bk_area = prepare_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )

    # 2) Hedgerow
    options_hedge, caps_hedge, bk_hedge = prepare_hedgerow_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )

    # 3) Watercourse
    options_water, caps_water, bk_water = prepare_watercourse_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )

    # ---- Combine ledgers into one joint solve ----
    options: List[dict] = []
    options.extend(options_area)
    options.extend(options_hedge)
    options.extend(options_water)

    stock_caps: Dict[str, float] = {}
    stock_caps.update(caps_area)
    stock_caps.update(caps_hedge)
    stock_caps.update(caps_water)

    stock_bankkey: Dict[str, str] = {}
    stock_bankkey.update(bk_area)
    stock_bankkey.update(bk_hedge)
    stock_bankkey.update(bk_water)

    if not options:
        raise RuntimeError("No feasible options. Check prices/stock/rules or location tiers.")

    # ---- Map options to each demand row ----
    idx_by_dem: Dict[int, List[int]] = {}
    dem_need: Dict[int, float] = {}
    for di, drow in demand_df.iterrows():
        idx_by_dem[di] = []
        dem_need[di] = float(drow["units_required"])

    for i, opt in enumerate(options):
        idx_by_dem[opt["demand_idx"]].append(i)

    bad = [di for di, idxs in idx_by_dem.items() if len(idxs) == 0]
    if bad:
        names = [sstr(demand_df.iloc[di]["habitat_name"]) for di in bad]
        raise RuntimeError("No legal options for: " + ", ".join(names))

    bank_keys = sorted({opt["BANK_KEY"] for opt in options})

    try:
        import pulp

        def build_problem(minimise_banks: bool = False, cost_cap: Optional[float] = None):
            prob = pulp.LpProblem("BNG_MinCost_OneOptionPerDemand", pulp.LpMinimize)
            x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(len(options))]
            z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(len(options))]
            y = {b: pulp.LpVariable(f"y_{norm_name(b)}", lowBound=0, upBound=1, cat="Binary") for b in bank_keys}

            # tiny tie-break towards higher-capacity banks
            bank_capacity_total: Dict[str, float] = {b: 0.0 for b in bank_keys}
            for sid, cap in stock_caps.items():
                bkey = stock_bankkey.get(sid, "")
                if bkey in bank_capacity_total:
                    bank_capacity_total[bkey] += float(cap or 0.0)

            if minimise_banks:
                obj = pulp.lpSum([y[b] for b in bank_keys])
                eps = 1e-9
                obj += eps * pulp.lpSum([options[i]["unit_price"] * x[i] for i in range(len(options))])
                obj += -eps * pulp.lpSum([bank_capacity_total[b] * y[b] for b in bank_keys])
            else:
                obj = pulp.lpSum([options[i]["unit_price"] * x[i] for i in range(len(options))])
                eps = 1e-9
                obj += -eps * pulp.lpSum([bank_capacity_total[b] * y[b] for b in bank_keys])
            prob += obj

            # Hard limit: <= 2 banks
            prob += pulp.lpSum([y[b] for b in bank_keys]) <= 2

            # Link option selection to bank usage
            for i, opt in enumerate(options):
                prob += z[i] <= y[opt["BANK_KEY"]]

            # Exactly one option per demand; meet its units; bind x to z
            for di, idxs in idx_by_dem.items():
                need = dem_need[di]
                prob += pulp.lpSum([z[i] for i in idxs]) == 1
                prob += pulp.lpSum([x[i] for i in idxs]) == need
                for i in idxs:
                    prob += x[i] <= need * z[i]

            # Stock capacity constraints
            use_map: Dict[str, List[Tuple[int, float]]] = {}
            for i, opt in enumerate(options):
                for sid, coef in opt["stock_use"].items():
                    use_map.setdefault(sid, []).append((i, float(coef)))
            for sid, pairs in use_map.items():
                cap = float(stock_caps.get(sid, 0.0))
                prob += pulp.lpSum([coef * x[i] for (i, coef) in pairs]) <= cap

            # Optional cost cap (for stage B)
            if cost_cap is not None:
                prob += pulp.lpSum([options[i]["unit_price"] * x[i] for i in range(len(options))]) <= cost_cap + 1e-9

            return prob, x, z, y

        # Stage A: min cost
        probA, xA, zA, yA = build_problem(minimise_banks=False, cost_cap=None)
        probA.solve(pulp.PULP_CBC_CMD(msg=False))
        statusA = pulp.LpStatus[probA.status]
        if statusA not in ("Optimal", "Feasible"):
            raise RuntimeError("Optimiser infeasible.")
        best_cost = pulp.value(pulp.lpSum([options[i]["unit_price"] * xA[i] for i in range(len(options))])) or 0.0

        def extract(xvars, zvars):
            rows, total_cost = [], 0.0
            for i in range(len(options)):
                qty = xvars[i].value() or 0.0
                sel = zvars[i].value() or 0.0
                if sel >= 0.5 and qty > 0:
                    opt = options[i]
                    row = {
                        "demand_habitat": opt["demand_habitat"],
                        "BANK_KEY": opt["BANK_KEY"],
                        "bank_name": opt.get("bank_name",""),
                        "bank_id": opt.get("bank_id",""),
                        "supply_habitat": opt["supply_habitat"],
                        "allocation_type": opt.get("type", "normal"),
                        "tier": opt["tier"],
                        "units_supplied": qty,
                        "unit_price": opt["unit_price"],
                        "cost": qty * opt["unit_price"],
                        "price_source": opt.get("price_source",""),
                        "price_habitat": opt.get("price_habitat",""),
                    }
                    if opt.get("type") == "paired" and "paired_parts" in opt:
                        row["paired_parts"] = json.dumps(opt["paired_parts"])
                    rows.append(row)
                    total_cost += qty * opt["unit_price"]
            return pd.DataFrame(rows), float(total_cost)

        allocA, costA = extract(xA, zA)

        # Stage B: minimise #banks under +1% cost cap
        probB, xB, zB, yB = build_problem(minimise_banks=True, cost_cap=(1.0 + SINGLE_BANK_SOFT_PCT) * best_cost)
        probB.solve(pulp.PULP_CBC_CMD(msg=False))
        statusB = pulp.LpStatus[probB.status]

        if statusB in ("Optimal", "Feasible"):
            allocB, costB = extract(xB, zB)

            def bank_count(df):
                return df["BANK_KEY"].nunique() if not df.empty else 0

            if bank_count(allocB) < bank_count(allocA):
                # Stage C: re-min cost with chosen banks fixed
                chosen_banks = list(allocB["BANK_KEY"].unique())
                probC, xC, zC, yC = build_problem(minimise_banks=False, cost_cap=None)
                for b in bank_keys:
                    if b not in chosen_banks:
                        probC += yC[b] == 0
                probC.solve(pulp.PULP_CBC_CMD(msg=False))
                statusC = pulp.LpStatus[probC.status]
                if statusC in ("Optimal", "Feasible"):
                    allocC, costC = extract(xC, zC)
                    return allocC, costC, chosen_size
                return allocB, costB, chosen_size

        return allocA, costA, chosen_size

    except Exception:
        # ---- Greedy fallback (unchanged) ----
        caps = stock_caps.copy()
        used_banks: List[str] = []

        def bank_ok(b):
            cand = set(used_banks); cand.add(b)
            return len(cand) <= 2

        rows = []
        total_cost = 0.0

        for di, drow in demand_df.iterrows():
            need = float(drow["units_required"])
            cand_idx = sorted([i for i in range(len(options)) if options[i]["demand_idx"] == di],
                              key=lambda i: options[i]["unit_price"])

            best_i = None
            best_cost = float('inf')
            for i in cand_idx:
                opt = options[i]
                bkey = opt["BANK_KEY"]
                if not bank_ok(bkey):
                    continue
                ok = True
                for sid, coef in opt["stock_use"].items():
                    req = coef * need
                    if caps.get(sid, 0.0) + 1e-9 < req:
                        ok = False
                        break
                if not ok:
                    continue
                this_cost = need * opt["unit_price"]
                if this_cost < best_cost - 1e-9:
                    best_cost = this_cost
                    best_i = i

            if best_i is None:
                name = sstr(drow["habitat_name"])
                raise RuntimeError(
                    f"Greedy fallback infeasible for '{name}' (no single option covers need within caps and bank limit)."
                )

            opt = options[best_i]
            bkey = opt["BANK_KEY"]
            for sid, coef in opt["stock_use"].items():
                caps[sid] = caps.get(sid, 0.0) - coef * need
            if bkey not in used_banks:
                used_banks.append(bkey)

            row = {
                "demand_habitat": opt["demand_habitat"],
                "BANK_KEY": opt["BANK_KEY"],
                "bank_name": opt.get("bank_name",""),
                "bank_id": opt.get("bank_id",""),
                "supply_habitat": opt["supply_habitat"],
                "allocation_type": opt.get("type", "normal"),
                "tier": opt["tier"],
                "units_supplied": need,
                "unit_price": opt["unit_price"],
                "cost": need * opt["unit_price"],
                "price_source": opt.get("price_source",""),
                "price_habitat": opt.get("price_habitat",""),
            }
            if opt.get("type") == "paired" and "paired_parts" in opt:
                row["paired_parts"] = json.dumps(opt["paired_parts"])
            rows.append(row)
            total_cost += need * opt["unit_price"]

        return pd.DataFrame(rows), float(total_cost), chosen_size

# ================= Run optimiser UI =================
st.subheader("3) Run optimiser")
left, middle, right = st.columns([1,1,1])
with left:
    run = st.button("Optimise now", type="primary", disabled=demand_df.empty, key="optimise_btn")
with middle:
    if st.button("🔄 Start New Quote", key="start_new_quote_btn", help="Clear all inputs and start fresh"):
        reset_quote()
with right:
    if st.session_state["target_lpa_name"] or st.session_state["target_nca_name"]:
        st.caption(f"LPA: {st.session_state['target_lpa_name'] or '—'} | NCA: {st.session_state['target_nca_name'] or '—'} | "
                   f"LPA neigh: {len(st.session_state['lpa_neighbors'])} | NCA neigh: {len(st.session_state['nca_neighbors'])}")
    else:
        st.caption("Tip: run 'Locate' first for precise tiers (else assumes 'far').")

# ================= Diagnostics =================
with st.expander("🔎 Diagnostics", expanded=False):
    try:
        if demand_df.empty:
            st.info("Add some demand rows above to see diagnostics.", icon="ℹ️")
        else:
            dd = demand_df.copy()
            present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
            total_units_d = float(dd["units_required"].sum())
            chosen_size_d = select_contract_size(total_units_d, present_sizes)
            st.write(f"**Chosen contract size:** `{chosen_size_d}` (present sizes: {present_sizes}, total demand: {total_units_d})")
            st.write(f"**Target LPA:** {st.session_state['target_lpa_name'] or '—'}  |  **Target NCA:** {st.session_state['target_nca_name'] or '—'}")
            st.write(f"**# LPA neighbours:** {len(st.session_state['lpa_neighbors'])}  | **# NCA neighbours:** {len(st.session_state['nca_neighbors'])}")

            s = backend["Stock"].copy()
            s["quantity_available"] = pd.to_numeric(s["quantity_available"], errors="coerce").fillna(0)
            st.write("**Stock sanity**")
            st.write(f"Non-zero stock rows: **{(s['quantity_available']>0).sum()}** | "
                     f"Total available units: **{s['quantity_available'].sum():.2f}**")

            options_preview, _, _ = prepare_options(
                dd, chosen_size_d,
                sstr(st.session_state["target_lpa_name"]), sstr(st.session_state["target_nca_name"]),
                [sstr(n) for n in st.session_state["lpa_neighbors"]], [sstr(n) for n in st.session_state["nca_neighbors"]],
                st.session_state["lpa_neighbors_norm"], st.session_state["nca_neighbors_norm"]
            )
            if not options_preview:
                st.error("No candidate options (check prices/stock/rules).")
            else:
                cand_df = pd.DataFrame(options_preview).rename(columns={"type": "allocation_type"})
                st.write("**Candidate options (by type & tier):**")
                grouped = (
                    cand_df.groupby(["demand_habitat","allocation_type","tier"], as_index=False)
                           .agg(options=("tier","count"),
                                min_price=("unit_price","min"),
                                max_price=("unit_price","max"))
                           .sort_values(["demand_habitat","allocation_type","tier"])
                )
                st.dataframe(grouped, use_container_width=True, hide_index=True)
                if "price_source" in cand_df.columns:
                    st.caption("Note: `price_source='group-proxy'` or `any-low-proxy` indicate proxy pricing rules.")

                st.markdown("**Cheapest candidates per demand (top 5 by unit price)**")
                for dem in dd["habitat_name"].unique():
                    sub = cand_df[cand_df["demand_habitat"] == dem].copy()
                    if sub.empty:
                        continue
                    sub = sub.sort_values("unit_price").head(5)
                    sub = sub[["bank_name","BANK_KEY","proximity","allocation_type","supply_habitat","unit_price","price_source","price_habitat"]]
                    st.write(f"**{dem}**")
                    st.dataframe(sub, use_container_width=True, hide_index=True)

    except Exception as de:
        st.error(f"Diagnostics error: {de}")

# ================= Price readout =================
def _build_pricing_enriched_for_size(chosen_size: str) -> pd.DataFrame:
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Banks   = backend["Banks"].copy()

    pr = Pricing[Pricing["contract_size"] == chosen_size].copy()
    if "bank_name" not in pr.columns and "bank_id" in pr.columns and "bank_name" in Banks.columns:
        pr = pr.merge(Banks[["bank_id","bank_name"]].drop_duplicates(), on="bank_id", how="left")

    pc_join = pr.merge(
        Catalog[["habitat_name", "broader_type", "distinctiveness_name"]],
        on="habitat_name", how="left", suffixes=("", "_cat")
    )
    pc_join["broader_type_eff"] = np.where(pc_join["broader_type"].astype(str).str.len()>0,
                                           pc_join["broader_type"], pc_join["broader_type_cat"])
    pc_join["distinctiveness_name_eff"] = np.where(pc_join["distinctiveness_name"].astype(str).str.len()>0,
                                                   pc_join["distinctiveness_name"], pc_join["distinctiveness_name_cat"])
    for c in ["broader_type_eff","distinctiveness_name_eff","tier","bank_id","habitat_name","BANK_KEY","bank_name"]:
        if c in pc_join.columns:
            pc_join[c] = pc_join[c].astype(str).str.strip()

    cols = [
        "BANK_KEY", "bank_name", "bank_id", "contract_size", "tier",
        "habitat_name", "price", "broader_type_eff", "distinctiveness_name_eff"
    ]
    for c in cols:
        if c not in pc_join.columns:
            pc_join[c] = ""
    pc_join["price"] = pd.to_numeric(pc_join["price"], errors="coerce")
    pc_join = pc_join[cols].sort_values(["BANK_KEY","tier","habitat_name","price"], kind="stable")
    return pc_join

with st.expander("🧾 Price readout (normalised view the optimiser uses)", expanded=False):
    try:
        present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
        total_units = float(demand_df["units_required"].sum()) if not demand_df.empty else 0.0
        chosen_size = select_contract_size(total_units, present_sizes)

        st.write(f"**Chosen contract size:** `{chosen_size}` (present sizes: {present_sizes})")

        prn = _build_pricing_enriched_for_size(chosen_size)

        if prn.empty:
            st.error("No pricing rows found for the chosen contract size.")
        else:
            st.markdown("**Full normalised price table (this size)**")
            st.dataframe(prn, use_container_width=True, hide_index=True)

            st.markdown("**Summary by bank & tier**")
            summ = (prn.groupby(["BANK_KEY","bank_name","tier"], as_index=False)
                        .agg(rows=("price","count"),
                             min_price=("price","min"),
                             p25=("price", lambda s: s.quantile(0.25)),
                             median=("price","median"),
                             p75=("price", lambda s: s.quantile(0.75)),
                             max_price=("price","max"))
                        .sort_values(["tier","min_price","BANK_KEY"]))
            st.dataframe(summ, use_container_width=True, hide_index=True)

            if not demand_df.empty:
                want = sorted(set(demand_df["habitat_name"]) - {NET_GAIN_LABEL})
                if want:
                    st.markdown("**Only demanded habitats (exact names)**")
                    prn_dem = prn[prn["habitat_name"].isin(want)].copy()
                    if prn_dem.empty:
                        st.warning("No exact price rows for the demanded habitat names at this size.")
                    else:
                        st.dataframe(prn_dem.sort_values(["habitat_name","tier","price"]),
                                     use_container_width=True, hide_index=True)

            # Quick scrub check
            try:
                mask_scrub = prn["habitat_name"].str.contains("scrub", case=False, na=False) | \
                             prn["habitat_name"].str.contains("bramble", case=False, na=False)
                prn_scrub = prn[mask_scrub]
                if not prn_scrub.empty:
                    st.markdown("**Scrub pricing quick check**")
                    st.dataframe(
                        prn_scrub.sort_values(["tier","price","BANK_KEY","habitat_name"]),
                        use_container_width=True, hide_index=True
                    )
            except Exception:
                pass

            csv_bytes = prn.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download pricing (normalised, this size) CSV",
                data=csv_bytes,
                file_name=f"pricing_normalised_{chosen_size}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Price readout error: {e}")

# ================= Pricing completeness =================
with st.expander("💷 Pricing completeness (this contract size)", expanded=False):
    try:
        if demand_df.empty:
            st.info("Add demand rows to see pricing completeness.")
        else:
            present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
            total_units_pc = float(demand_df["units_required"].sum())
            chosen_size_pc = select_contract_size(total_units_pc, present_sizes)

            pr = backend["Pricing"].copy()
            pr = pr[pr["contract_size"] == chosen_size_pc]
            needed = pd.MultiIndex.from_product(
                [
                    backend["Stock"]["bank_id"].dropna().unique(),
                    demand_df["habitat_name"].unique(),
                    ["local","adjacent","far"],
                ],
                names=["bank_id","habitat_name","tier"]
            ).to_frame(index=False)

            merged = needed.merge(
                pr[["bank_id","habitat_name","tier","price"]],
                on=["bank_id","habitat_name","tier"],
                how="left",
                indicator=True
            )

            missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
            if missing.empty:
                st.success(f"All exact pricing rows exist for size `{chosen_size_pc}` across the demanded habitats.")
            else:
                st.warning("Some exact pricing rows are missing — that's fine if those rows are untradeable or Medium/Low use proxies.")
                st.dataframe(
                    missing.sort_values(["habitat_name","bank_id","tier"]),
                    use_container_width=True, hide_index=True
                )
    except Exception as e:
        st.error(f"Pricing completeness error: {e}")

# ================= Helpers for downloads =================
def df_to_csv_bytes(df):
    buf = BytesIO()
    buf.write(df.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf


# ================= Run optimiser & compute results =================
# ================= Run optimiser & compute results =================
if run:
    try:
        if demand_df.empty:
            st.error("Add at least one demand row before optimising.")
            st.stop()

        # Auto-locate if user typed address/postcode but forgot Locate
        if not st.session_state["target_lpa_name"] or not st.session_state["target_nca_name"]:
            if sstr(postcode) or sstr(address):
                try:
                    find_site(postcode, address)
                except Exception as e:
                    st.warning(f"Auto-locate failed: {e}. Proceeding with 'far' tiers only.")

        # Validate against catalog — allow special Net Gain labels
        cat_names_run = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        unknown = [h for h in demand_df["habitat_name"] if h not in cat_names_run and h not in [NET_GAIN_LABEL, NET_GAIN_HEDGEROW_LABEL]]
        if unknown:
            st.error(f"These demand habitats aren't in the catalog: {unknown}")
            st.stop()

        # Use session state values
        target_lpa = st.session_state["target_lpa_name"]
        target_nca = st.session_state["target_nca_name"]
        lpa_neighbors = st.session_state["lpa_neighbors"]
        nca_neighbors = st.session_state["nca_neighbors"]
        lpa_neighbors_norm = st.session_state["lpa_neighbors_norm"]
        nca_neighbors_norm = st.session_state["nca_neighbors_norm"]

        # Run optimization
        with st.spinner("Running optimization..."):
            alloc_df, total_cost, size = optimise(
                demand_df,
                target_lpa, target_nca,
                [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
                lpa_neighbors_norm, nca_neighbors_norm
            )

        # IMPORTANT: Load catchment data BEFORE setting optimization_complete
        selected_banks = alloc_df["BANK_KEY"].unique()
        
        with st.spinner(f"Loading catchment areas for {len(selected_banks)} selected bank(s)..."):
            # Get bank coordinates first
            bank_coords: Dict[str, Tuple[float,float]] = {}
            banks_df = backend["Banks"].copy()
            for _, b in banks_df.iterrows():
                bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
                loc = bank_row_to_latlon(b)
                if loc:
                    bank_coords[bkey] = (loc[0], loc[1])
            
            # Load catchment data for each selected bank
            catchments_loaded = []
            catchments_failed = []
            
            for bkey in selected_banks:
                cache_key = sstr(bkey)
                if sstr(bkey) in bank_coords:
                    try:
                        lat_b, lon_b = bank_coords[sstr(bkey)]
                        
                        # Always reload for fresh optimization (don't use cache)
                        b_lpa_name, b_lpa_gj, b_nca_name, b_nca_gj = get_catchment_geo_for_point(lat_b, lon_b)
                        st.session_state["bank_catchment_geo"][cache_key] = {
                            "lpa_name": b_lpa_name, "lpa_gj": b_lpa_gj,
                            "nca_name": b_nca_name, "nca_gj": b_nca_gj,
                        }
                        catchments_loaded.append(cache_key)
                        
                        # Small delay to avoid overwhelming APIs
                        time.sleep(0.1)
                        
                    except Exception as e:
                        st.warning(f"Could not load catchment for bank {bkey}: {e}")
                        catchments_failed.append(bkey)
                else:
                    st.warning(f"No coordinates found for bank {bkey}")
                    catchments_failed.append(bkey)
        
        # NOW save results and set completion flag
        st.session_state["last_alloc_df"] = alloc_df.copy()
        st.session_state["optimization_complete"] = True
        
        # Show what we loaded
        if catchments_loaded:
            st.success(f"✅ Loaded catchment data for {len(catchments_loaded)} bank(s): {', '.join(catchments_loaded)}")
        if catchments_failed:
            st.warning(f"⚠️ Failed to load catchment data for: {', '.join(catchments_failed)}")

        total_with_admin = total_cost + ADMIN_FEE_GBP
        st.success(
            f"Optimisation complete. Contract size = **{size}**. "
            f"Subtotal (units): **£{total_cost:,.0f}**  |  Admin fee: **£{ADMIN_FEE_GBP:,.0f}**  |  "
            f"Grand total: **£{total_with_admin:,.0f}**"
        )

        # ========== PROCESS RESULTS FOR PERSISTENCE (NO INLINE DISPLAY) ==========
        # Calculate summary data and save to session state - displayed in persistent section below
        
        MULT = {"local": 1.0, "adjacent": 4/3, "far": 2.0}

        def split_paired_rows(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty: return df
            rows = []
            for _, r in df.iterrows():
                if sstr(r.get("allocation_type")) != "paired":
                    rows.append(r.to_dict())
                    continue

                # Extract paired parts (each has its own unit price and stock_use)
                parts = []
                try:
                    parts = json.loads(sstr(r.get("paired_parts")))
                except Exception:
                    parts = []

                sh = sstr(r.get("supply_habitat"))
                name_parts = [p.strip() for p in sh.split("+")] if sh else []

                units_total = float(r.get("units_supplied", 0.0) or 0.0)
                tier = sstr(r.get("tier", "")).lower()
                srm = MULT.get(tier, 1.0)

                if len(parts) == 2:
                    # Calculate total stock_use to normalize
                    total_stock_use = sum(float(part.get("stock_use", 0.5)) for part in parts)
                    
                    for idx, part in enumerate(parts):
                        rr = r.to_dict()
                        rr["supply_habitat"] = sstr(part.get("habitat") or (name_parts[idx] if idx < len(name_parts) else f"Part {idx+1}"))
                        
                        # Use stock_use ratio if available, otherwise default to 0.5
                        stock_use_ratio = float(part.get("stock_use", 0.5))
                        
                        # Calculate units supplied: normalize by total stock_use, then divide by SRM
                        # This gives the actual units delivered to customer for this component
                        rr["units_supplied"] = units_total * stock_use_ratio / srm
                        rr["unit_price"] = float(part.get("unit_price", rr.get("unit_price", 0.0)))
                        rr["cost"] = rr["units_supplied"] * rr["unit_price"]
                        rows.append(rr)
                else:
                    # Fallback: split cost/units evenly (50/50)
                    units_each = 0.5 * units_total
                    if len(name_parts) == 2:
                        for part_name in name_parts:
                            rr = r.to_dict()
                            rr["supply_habitat"] = part_name
                            rr["units_supplied"] = units_each
                            rr["cost"] = float(r.get("cost", 0.0) or 0.0) * 0.5
                            rows.append(rr)
                    else:
                        rows.append(r.to_dict())
            return pd.DataFrame(rows)

        expanded_alloc = split_paired_rows(alloc_df.copy())
        expanded_alloc["proximity"] = expanded_alloc.get("tier", "").map(sstr)
        expanded_alloc["effective_units"] = expanded_alloc.apply(
            lambda r: float(r["units_supplied"]) * MULT.get(sstr(r["proximity"]).lower(), 1.0), axis=1
        )

        site_hab_totals = (expanded_alloc.groupby(["BANK_KEY","bank_name","supply_habitat","tier"], as_index=False)
                           .agg(units_supplied=("units_supplied","sum"),
                                effective_units=("effective_units","sum"),
                                cost=("cost","sum"))
                           .sort_values(["bank_name","supply_habitat","tier"]))

        site_hab_totals["avg_unit_price"] = site_hab_totals["cost"] / site_hab_totals["units_supplied"].replace(0, np.nan)
        site_hab_totals["avg_effective_unit_price"] = site_hab_totals["cost"] / site_hab_totals["effective_units"].replace(0, np.nan)

        site_hab_totals = site_hab_totals[[
            "BANK_KEY","bank_name","supply_habitat","tier",
            "units_supplied","effective_units","avg_unit_price","avg_effective_unit_price","cost"
        ]]

        # Calculate by bank summary
        by_bank = alloc_df.groupby(["BANK_KEY","bank_name","bank_id"], as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            cost=("cost","sum")
        ).sort_values("cost", ascending=False)

        # Calculate by habitat summary
        by_hab = alloc_df.groupby("supply_habitat", as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            cost=("cost","sum")
        )

        # Create order summary
        summary_df = pd.DataFrame([
            {"Item": "Subtotal (units)", "Amount £": round(total_cost, 2)},
            {"Item": "Admin fee",        "Amount £": round(ADMIN_FEE_GBP, 2)},
            {"Item": "Grand total",      "Amount £": round(total_with_admin, 2)},
        ])
        
        # Save summary data to session state for persistence
        st.session_state["site_hab_totals"] = site_hab_totals.copy()
        st.session_state["by_bank"] = by_bank.copy()
        st.session_state["by_hab"] = by_hab.copy()
        st.session_state["summary_df"] = summary_df.copy()
        st.session_state["total_cost"] = total_cost
        st.session_state["contract_size"] = size
        
        # Trigger immediate map refresh
        st.session_state["needs_map_refresh"] = True
        st.rerun()

    except Exception as e:
        st.error(f"Optimiser error: {e}")

# ================= Email Report Generation =================
# ================= Email Report Generation (EXACT TEMPLATE MATCH) =================


# ================= Fixed Email Report Generation Function =================
def generate_client_report_table_fixed(alloc_df: pd.DataFrame, demand_df: pd.DataFrame, total_cost: float, admin_fee: float, 
                                       client_name: str, ref_number: str, location: str,
                                       manual_hedgerow_rows: List[dict] = None,
                                       manual_watercourse_rows: List[dict] = None) -> Tuple[pd.DataFrame, str]:
    """Generate the client-facing report table and email body matching exact template with improved styling"""
    
    if manual_hedgerow_rows is None:
        manual_hedgerow_rows = []
    if manual_watercourse_rows is None:
        manual_watercourse_rows = []
    
    # Separate by habitat types
    area_habitats = []
    hedgerow_habitats = []
    watercourse_habitats = []
    
    # Process each demand
    for _, demand_row in demand_df.iterrows():
        demand_habitat = demand_row["habitat_name"]
        demand_units = demand_row["units_required"]
        
        # Find corresponding allocation(s)
        matching_allocs = alloc_df[alloc_df["demand_habitat"] == demand_habitat]
        
        if matching_allocs.empty:
            continue
            
        for _, alloc_row in matching_allocs.iterrows():
            # Determine demand distinctiveness
            if demand_habitat == NET_GAIN_LABEL:
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any"
            elif demand_habitat == "Net Gain (Hedgerows)":
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any (Hedgerows)"
            else:
                # Look up from catalog
                cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == demand_habitat]
                if not cat_match.empty:
                    demand_distinctiveness = cat_match["distinctiveness_name"].iloc[0]
                    demand_habitat_display = demand_habitat
                else:
                    demand_distinctiveness = "Medium"  # Default
                    demand_habitat_display = demand_habitat
            
            # Supply info
            supply_habitat = alloc_row["supply_habitat"]
            supply_units = alloc_row["units_supplied"]
            unit_price = alloc_row["unit_price"]
            offset_cost = alloc_row["cost"]
            
            # Determine supply distinctiveness
            supply_cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == supply_habitat]
            if not supply_cat_match.empty:
                supply_distinctiveness = supply_cat_match["distinctiveness_name"].iloc[0]
            else:
                supply_distinctiveness = "Medium"  # Default
            
            row_data = {
                "Distinctiveness": demand_distinctiveness,
                "Habitats Lost": demand_habitat_display,
                "# Units": f"{demand_units:.2f}",
                "Distinctiveness_Supply": supply_distinctiveness,
                "Habitats Supplied": supply_habitat,
                "# Units_Supply": f"{supply_units:.2f}",
                "Price Per Unit": f"£{unit_price:,.0f}",
                "Offset Cost": f"£{offset_cost:,.0f}"
            }
            
            # Categorize by habitat type
            if demand_habitat == "Net Gain (Hedgerows)" or "hedgerow" in demand_habitat.lower() or "hedgerow" in supply_habitat.lower():
                hedgerow_habitats.append(row_data)
            elif "watercourse" in demand_habitat.lower() or "water" in supply_habitat.lower():
                watercourse_habitats.append(row_data)
            else:
                area_habitats.append(row_data)
    
    # Process manual hedgerow entries
    manual_hedgerow_cost = 0.0
    for row in manual_hedgerow_rows:
        habitat_lost = sstr(row.get("habitat_lost", ""))
        habitat_name = sstr(row.get("habitat_name", ""))
        units = float(row.get("units", 0.0) or 0.0)
        price_per_unit = float(row.get("price_per_unit", 0.0) or 0.0)
        
        if habitat_name and units > 0:
            offset_cost = units * price_per_unit
            manual_hedgerow_cost += offset_cost
            
            # Determine distinctiveness for lost habitat
            if habitat_lost == NET_GAIN_LABEL:
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any"
            elif habitat_lost == "Net Gain (Hedgerows)":
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any (Hedgerows)"
            else:
                cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == habitat_lost]
                if not cat_match.empty:
                    demand_distinctiveness = cat_match["distinctiveness_name"].iloc[0]
                    demand_habitat_display = habitat_lost
                else:
                    demand_distinctiveness = "Medium"
                    demand_habitat_display = habitat_lost if habitat_lost else "Not specified"
            
            # Determine distinctiveness for supplied habitat
            if habitat_name == NET_GAIN_LABEL:
                supply_distinctiveness = "10% Net Gain"
                supply_habitat_display = "Any"
            elif habitat_name == "Net Gain (Hedgerows)":
                supply_distinctiveness = "10% Net Gain"
                supply_habitat_display = "Any (Hedgerows)"
            else:
                cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == habitat_name]
                if not cat_match.empty:
                    supply_distinctiveness = cat_match["distinctiveness_name"].iloc[0]
                    supply_habitat_display = habitat_name
                else:
                    supply_distinctiveness = "Medium"
                    supply_habitat_display = habitat_name
            
            row_data = {
                "Distinctiveness": demand_distinctiveness,
                "Habitats Lost": demand_habitat_display,
                "# Units": f"{units:.2f}",
                "Distinctiveness_Supply": supply_distinctiveness,
                "Habitats Supplied": supply_habitat_display,
                "# Units_Supply": f"{units:.2f}",
                "Price Per Unit": f"£{price_per_unit:,.0f}",
                "Offset Cost": f"£{offset_cost:,.0f}"
            }
            hedgerow_habitats.append(row_data)
    
    # Process manual watercourse entries
    manual_watercourse_cost = 0.0
    for row in manual_watercourse_rows:
        habitat_lost = sstr(row.get("habitat_lost", ""))
        habitat_name = sstr(row.get("habitat_name", ""))
        units = float(row.get("units", 0.0) or 0.0)
        price_per_unit = float(row.get("price_per_unit", 0.0) or 0.0)
        
        if habitat_name and units > 0:
            offset_cost = units * price_per_unit
            manual_watercourse_cost += offset_cost
            
            # Determine distinctiveness for lost habitat
            if habitat_lost == NET_GAIN_LABEL:
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any"
            elif habitat_lost == "Net Gain (Hedgerows)":
                demand_distinctiveness = "10% Net Gain"
                demand_habitat_display = "Any (Hedgerows)"
            else:
                cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == habitat_lost]
                if not cat_match.empty:
                    demand_distinctiveness = cat_match["distinctiveness_name"].iloc[0]
                    demand_habitat_display = habitat_lost
                else:
                    demand_distinctiveness = "Medium"
                    demand_habitat_display = habitat_lost if habitat_lost else "Not specified"
            
            # Determine distinctiveness for supplied habitat
            if habitat_name == NET_GAIN_LABEL:
                supply_distinctiveness = "10% Net Gain"
                supply_habitat_display = "Any"
            elif habitat_name == "Net Gain (Hedgerows)":
                supply_distinctiveness = "10% Net Gain"
                supply_habitat_display = "Any (Hedgerows)"
            else:
                cat_match = backend["HabitatCatalog"][backend["HabitatCatalog"]["habitat_name"] == habitat_name]
                if not cat_match.empty:
                    supply_distinctiveness = cat_match["distinctiveness_name"].iloc[0]
                    supply_habitat_display = habitat_name
                else:
                    supply_distinctiveness = "Medium"
                    supply_habitat_display = habitat_name
            
            row_data = {
                "Distinctiveness": demand_distinctiveness,
                "Habitats Lost": demand_habitat_display,
                "# Units": f"{units:.2f}",
                "Distinctiveness_Supply": supply_distinctiveness,
                "Habitats Supplied": supply_habitat_display,
                "# Units_Supply": f"{units:.2f}",
                "Price Per Unit": f"£{price_per_unit:,.0f}",
                "Offset Cost": f"£{offset_cost:,.0f}"
            }
            watercourse_habitats.append(row_data)
    
    # Update total cost to include manual entries
    total_cost_with_manual = total_cost + manual_hedgerow_cost + manual_watercourse_cost
    total_with_admin = total_cost_with_manual + admin_fee
    
    # Bundle Low + 10% Net Gain rows together for each habitat type
    def bundle_low_and_net_gain(habitats_list):
        """Bundle Low distinctiveness and 10% Net Gain rows together"""
        bundled = []
        low_rows = {}
        net_gain_rows = {}
        other_rows = []
        
        # Separate rows by distinctiveness
        for row in habitats_list:
            dist = row["Distinctiveness"]
            supply_dist = row["Distinctiveness_Supply"]
            
            # Check if this is a Low or Net Gain row
            if dist == "Low" or dist == "10% Net Gain":
                # Group by supply habitat for bundling
                supply_hab = row["Habitats Supplied"]
                if dist == "Low":
                    if supply_hab not in low_rows:
                        low_rows[supply_hab] = []
                    low_rows[supply_hab].append(row)
                else:  # 10% Net Gain
                    if supply_hab not in net_gain_rows:
                        net_gain_rows[supply_hab] = []
                    net_gain_rows[supply_hab].append(row)
            else:
                other_rows.append(row)
        
        # Bundle Low + Net Gain rows for same supply habitat
        all_supply_habitats = set(list(low_rows.keys()) + list(net_gain_rows.keys()))
        for supply_hab in sorted(all_supply_habitats):
            low_list = low_rows.get(supply_hab, [])
            ng_list = net_gain_rows.get(supply_hab, [])
            
            if low_list and ng_list:
                # Bundle them together
                total_units = sum(float(r["# Units"].replace(",", "")) for r in low_list + ng_list)
                total_supply_units = sum(float(r["# Units_Supply"].replace(",", "")) for r in low_list + ng_list)
                total_cost = sum(float(r["Offset Cost"].replace("£", "").replace(",", "")) for r in low_list + ng_list)
                
                # Use weighted average for price per unit
                avg_price = total_cost / total_supply_units if total_supply_units > 0 else 0
                
                bundled_row = {
                    "Distinctiveness": "Low + 10% Net Gain",
                    "Habitats Lost": low_list[0]["Habitats Lost"] if low_list else ng_list[0]["Habitats Lost"],
                    "# Units": f"{total_units:.2f}",
                    "Distinctiveness_Supply": low_list[0]["Distinctiveness_Supply"] if low_list else ng_list[0]["Distinctiveness_Supply"],
                    "Habitats Supplied": supply_hab,
                    "# Units_Supply": f"{total_supply_units:.2f}",
                    "Price Per Unit": f"£{avg_price:,.0f}",
                    "Offset Cost": f"£{total_cost:,.0f}"
                }
                bundled.append(bundled_row)
            elif low_list:
                # Only Low rows, add them as is
                bundled.extend(low_list)
            elif ng_list:
                # Only Net Gain rows, add them as is
                bundled.extend(ng_list)
        
        # Add other rows
        bundled.extend(other_rows)
        return bundled
    
    # Apply bundling to each habitat type
    area_habitats = bundle_low_and_net_gain(area_habitats)
    hedgerow_habitats = bundle_low_and_net_gain(hedgerow_habitats)
    watercourse_habitats = bundle_low_and_net_gain(watercourse_habitats)
    
    # Sort habitats by distinctiveness priority (High > Medium > Low + Net Gain > Very Low)
    def sort_by_distinctiveness(habitats_list):
        """Sort habitat rows by distinctiveness priority"""
        distinctiveness_order = {
            "Very High": 0,
            "V.High": 0,
            "High": 1,
            "Medium": 2,
            "Low + 10% Net Gain": 3,
            "Low": 4,
            "10% Net Gain": 5,
            "Very Low": 6,
            "V.Low": 6
        }
        
        def get_sort_key(row):
            dist = row.get("Distinctiveness", "")
            return distinctiveness_order.get(dist, 99)  # Unknown distinctiveness goes to end
        
        return sorted(habitats_list, key=get_sort_key)
    
    # Apply sorting to each habitat type
    area_habitats = sort_by_distinctiveness(area_habitats)
    hedgerow_habitats = sort_by_distinctiveness(hedgerow_habitats)
    watercourse_habitats = sort_by_distinctiveness(watercourse_habitats)
    
    # Build HTML table with improved styling (30% narrower, better colors)
    html_table = """
    <table border="1" style="border-collapse: collapse; width: 70%; margin: 0 auto; font-family: Arial, sans-serif; font-size: 11px;">
        <thead>
            <tr>
                <th colspan="3" style="text-align: center; padding: 8px; border: 1px solid #000; font-weight: bold; background-color: #F8C237; color: #000;">Development Impact</th>
                <th colspan="5" style="text-align: center; padding: 8px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;">Mitigation Supplied from Wild Capital</th>
            </tr>
            <tr>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #F8C237; color: #000;">Distinctiveness</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #F8C237; color: #000;">Habitats Lost</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #F8C237; color: #000;"># Units</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;">Distinctiveness</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;">Habitats Supplied</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;"># Units</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;">Price Per Unit</th>
                <th style="padding: 6px; border: 1px solid #000; font-weight: bold; background-color: #2A514A; color: #FFFFFF;">Offset Cost</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Add Area Habitats section with light green background
    if area_habitats:
        html_table += """
            <tr style="background-color: #D9F2D0;">
                <td colspan="8" style="padding: 6px; border: 1px solid #000; font-weight: bold; color: #000;">Area Habitats</td>
            </tr>
        """
        for habitat in area_habitats:
            html_table += f"""
            <tr>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Lost"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Supplied"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Price Per Unit"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Offset Cost"]}</td>
            </tr>
            """
    
    # Add Hedgerow Habitats section with light green background
    if hedgerow_habitats:
        html_table += """
            <tr style="background-color: #D9F2D0;">
                <td colspan="8" style="padding: 6px; border: 1px solid #000; font-weight: bold; color: #000;">Hedgerow Habitats</td>
            </tr>
        """
        for habitat in hedgerow_habitats:
            html_table += f"""
            <tr>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Lost"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Supplied"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Price Per Unit"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Offset Cost"]}</td>
            </tr>
            """
    
    # Add Watercourse Habitats section with light green background
    if watercourse_habitats:
        html_table += """
            <tr style="background-color: #D9F2D0;">
                <td colspan="8" style="padding: 6px; border: 1px solid #000; font-weight: bold; color: #000;">Watercourse Habitats</td>
            </tr>
        """
        for habitat in watercourse_habitats:
            html_table += f"""
            <tr>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Lost"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Distinctiveness_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000;">{habitat["Habitats Supplied"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["# Units_Supply"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Price Per Unit"]}</td>
                <td style="padding: 6px; border: 1px solid #000; text-align: right;">{habitat["Offset Cost"]}</td>
            </tr>
            """
    

    
    # Calculate total units including manual entries
    total_demand_units = demand_df['units_required'].sum()
    total_supply_units = alloc_df['units_supplied'].sum()
    
    # Add manual units
    for row in manual_hedgerow_rows:
        units = float(row.get("units", 0.0) or 0.0)
        if units > 0:
            total_demand_units += units
            total_supply_units += units
    
    for row in manual_watercourse_rows:
        units = float(row.get("units", 0.0) or 0.0)
        if units > 0:
            total_demand_units += units
            total_supply_units += units
    
    # Add Planning Discharge Pack and Total
    html_table += f"""
        <tr>
            <td colspan="7" style="padding: 6px; border: 1px solid #000; text-align: right; font-weight: bold;">Planning Discharge Pack</td>
            <td style="padding: 6px; border: 1px solid #000; text-align: right;">£{admin_fee:,.0f}</td>
        </tr>
        <tr style="background-color: #f0f0f0; font-weight: bold;">
            <td style="padding: 6px; border: 1px solid #000;">Total</td>
            <td style="padding: 6px; border: 1px solid #000;"></td>
            <td style="padding: 6px; border: 1px solid #000; text-align: right;">{total_demand_units:.2f}</td>
            <td style="padding: 6px; border: 1px solid #000;"></td>
            <td style="padding: 6px; border: 1px solid #000;"></td>
            <td style="padding: 6px; border: 1px solid #000; text-align: right;">{total_supply_units:.2f}</td>
            <td style="padding: 6px; border: 1px solid #000;"></td>
            <td style="padding: 6px; border: 1px solid #000; text-align: right;">£{total_with_admin:,.0f}</td>
        </tr>
    </tbody>
    </table>
    """
    
    # Determine next steps based on amount (programmatic ending)
    if total_with_admin < 10000:
        next_steps = """<strong>Next Steps</strong>
<br><br>
BNG is a pre-commencement, not a pre-planning, condition.
<br><br>
To accept the quote, let us know—we'll request some basic details before sending the Allocation Agreement. The price is fixed for 30 days, but unit availability is only guaranteed once the agreement is signed.
<br><br>
Once you sign the agreement, pay the settlement fee and provide us with your metric and decision notice we will allocate the units to you.
<br><br>
If you have any questions, please reply to this email or call 01962 436574."""
    else:
        next_steps = """<strong>Next Steps</strong>
<br><br>
BNG is a pre-commencement, not a pre-planning, condition.
<br><br>
To accept the quote, let us know—we'll request some basic details before sending the Allocation Agreement. The price is fixed for 30 days, but unit availability is only guaranteed once the agreement is signed.
<br><br>
We offer two contract options:
<br><br>
1. <strong>Buy It Now:</strong> Pay in full on signing; units allocated immediately.<br>
2. <strong>Reservation & Purchase:</strong> Pay a reservation fee to hold units for up to 6 months, with the option to draw them down anytime in that period.
<br><br>
If you have any questions, please reply to this email or call 01962 436574."""
    
    # Generate full email body matching exact template
    email_body = f"""
<div style="font-family: Arial, sans-serif; font-size: 12px; line-height: 1.4;">

<strong>Dear {client_name}</strong>
<br><br>
<strong>Our Ref: {ref_number}</strong>
<br><br>
Arbtech has advised us that you need Biodiversity Net Gain units for your development in {location}, and we're here to help you discharge your BNG condition.
<br><br>
Thank you for enquiring about BNG Units for your development in {location}
<br><br>
<strong>About Us</strong>
<br><br>
Wild Capital is a national supplier of BNG Units and environmental mitigation credits (Nutrient Neutrality, SANG), backed by institutional finance. We create and manage a large portfolio of nature recovery projects, owning the freehold to all mitigation land for the highest integrity and long-term assurance.
<br><br>
Our key advantages:
<br><br>
1. <strong>Permanent Nature Recovery:</strong> We dedicate all land to conservation in perpetuity, not just for the 30-year minimum.<br>
2. <strong>Independently Managed Endowment:</strong> Long-term management funds are fully insured and overseen by independent asset managers.<br>
3. <strong>Independent Governance:</strong> Leading third-party ecologists and contractors oversee all monitoring and habitat management, ensuring objectivity.<br>
4. <strong>Full Ownership and Responsibility:</strong> We hold the freehold and assume complete responsibility for all delivery and management - no ambiguity.
<br><br>
<strong>Your Quote - £{total_with_admin:,.0f} + VAT</strong>
<br><br>
See a detailed breakdown of the pricing below. I've attached a PDF outlining the BNG offset and condition discharge process. If you have any questions, please let us know—we're here to help.
<br><br>

{html_table}

<br><br>
Prices exclude VAT. Any legal costs for contract amendments will be charged to the client and must be paid before allocation.
<br><br>
{next_steps}

</div>
    """
    
    # Create simplified dataframe for display
    all_habitats = area_habitats + hedgerow_habitats + watercourse_habitats
    report_df = pd.DataFrame(all_habitats) if all_habitats else pd.DataFrame()
    
    return report_df, email_body

# ========== PERSISTENT ALLOCATION DETAILS ==========
# This section persists across reruns because it's outside the "if run:" block
if st.session_state.get("optimization_complete", False) and st.session_state.get("last_alloc_df") is not None:
    st.markdown("---")
    st.markdown("### 📊 Optimization Results")
    
    # Show summary at top
    if st.session_state.get("contract_size") and st.session_state.get("total_cost") is not None:
        total_cost = st.session_state["total_cost"]
        total_with_admin = total_cost + ADMIN_FEE_GBP
        st.success(
            f"Contract size = **{st.session_state['contract_size']}**. "
            f"Subtotal (units): **£{total_cost:,.0f}**  |  Admin fee: **£{ADMIN_FEE_GBP:,.0f}**  |  "
            f"Grand total: **£{total_with_admin:,.0f}**"
        )
    
    # Show allocation detail in expander
    with st.expander("📋 Allocation detail", expanded=True):
        alloc_df = st.session_state["last_alloc_df"]
        st.dataframe(alloc_df, use_container_width=True)
        if "price_source" in alloc_df.columns:
            st.caption("Note: `price_source='group-proxy'` or `any-low-proxy` indicate proxy pricing rules.")
    
    # Show Site/Habitat totals in expander
    if st.session_state.get("site_hab_totals") is not None:
        with st.expander("📊 Site/Habitat totals (effective units)", expanded=True):
            st.dataframe(st.session_state["site_hab_totals"], use_container_width=True, hide_index=True)
    
    # Show By bank in expander
    if st.session_state.get("by_bank") is not None:
        with st.expander("🏢 By bank", expanded=False):
            st.dataframe(st.session_state["by_bank"], use_container_width=True)
    
    # Show By habitat in expander
    if st.session_state.get("by_hab") is not None:
        with st.expander("🌿 By habitat (supply)", expanded=False):
            st.dataframe(st.session_state["by_hab"], use_container_width=True)
    
    # Show Order summary in expander
    if st.session_state.get("summary_df") is not None:
        with st.expander("💰 Order summary (with admin fee)", expanded=True):
            st.dataframe(st.session_state["summary_df"], hide_index=True, use_container_width=True)

# ========== MANUAL HEDGEROW/WATERCOURSE ENTRIES (PERSISTENT) ==========
# This section persists across reruns because it's outside the "if run:" block
if st.session_state.get("optimization_complete", False):
    st.markdown("---")
    st.markdown("#### ➕ Manual Additions (Hedgerow & Watercourse)")
    st.info("Add additional hedgerow or watercourse units to your quote. These will be included in the final client report.")
    
    # Get available habitats
    hedgerow_choices = get_hedgerow_habitats(backend["HabitatCatalog"])
    watercourse_choices = get_watercourse_habitats(backend["HabitatCatalog"])
    
    # Hedgerow Section
    with st.container(border=True):
        st.markdown("**🌿 Manual Hedgerow Units**")
        
        # Add Net Gain option to hedgerow choices
        hedgerow_choices_with_ng = hedgerow_choices + [NET_GAIN_LABEL] if hedgerow_choices else [NET_GAIN_LABEL]
        
        to_delete_hedgerow = []
        for idx, row in enumerate(st.session_state.manual_hedgerow_rows):
            c1, c2, c3, c4, c5 = st.columns([0.30, 0.30, 0.15, 0.15, 0.10])
            with c1:
                if hedgerow_choices_with_ng:
                    default_idx = None
                    if row.get("habitat_lost") and row["habitat_lost"] in hedgerow_choices_with_ng:
                        default_idx = hedgerow_choices_with_ng.index(row["habitat_lost"])
                    st.session_state.manual_hedgerow_rows[idx]["habitat_lost"] = st.selectbox(
                        "Habitat Lost", hedgerow_choices_with_ng,
                        index=default_idx,
                        key=f"manual_hedge_lost_{row['id']}",
                        help="Select hedgerow habitat lost"
                    )
                else:
                    st.warning("No hedgerow habitats available in catalog")
            with c2:
                if hedgerow_choices_with_ng:
                    default_idx = None
                    if row.get("habitat_name") and row["habitat_name"] in hedgerow_choices_with_ng:
                        default_idx = hedgerow_choices_with_ng.index(row["habitat_name"])
                    st.session_state.manual_hedgerow_rows[idx]["habitat_name"] = st.selectbox(
                        "Habitat to Mitigate", hedgerow_choices_with_ng,
                        index=default_idx,
                        key=f"manual_hedge_hab_{row['id']}",
                        help="Select hedgerow habitat to mitigate"
                    )
                else:
                    st.warning("No hedgerow habitats available")
            with c3:
                st.session_state.manual_hedgerow_rows[idx]["units"] = st.number_input(
                    "Units", min_value=0.0, step=0.01, value=float(row.get("units", 0.0)), 
                    key=f"manual_hedge_units_{row['id']}"
                )
            with c4:
                st.session_state.manual_hedgerow_rows[idx]["price_per_unit"] = st.number_input(
                    "Price/Unit (£)", min_value=0.0, step=1.0, value=float(row.get("price_per_unit", 0.0)),
                    key=f"manual_hedge_price_{row['id']}"
                )
            with c5:
                if st.button("🗑️", key=f"del_manual_hedge_{row['id']}", help="Remove this row"):
                    to_delete_hedgerow.append(row["id"])
        
        if to_delete_hedgerow:
            st.session_state.manual_hedgerow_rows = [r for r in st.session_state.manual_hedgerow_rows if r["id"] not in to_delete_hedgerow]
            st.rerun()
        
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            if st.button("➕ Add Hedgerow Entry", key="add_manual_hedge_btn"):
                st.session_state.manual_hedgerow_rows.append({
                    "id": st.session_state._next_manual_hedgerow_id,
                    "habitat_lost": "",
                    "habitat_name": "",
                    "units": 0.0,
                    "price_per_unit": 0.0
                })
                st.session_state._next_manual_hedgerow_id += 1
                st.rerun()
        with col2:
            if st.button("🧹 Clear Hedgerow", key="clear_manual_hedge_btn"):
                st.session_state.manual_hedgerow_rows = []
                st.rerun()
    
    # Watercourse Section
    with st.container(border=True):
        st.markdown("**💧 Manual Watercourse Units**")
        
        # Add Net Gain option to watercourse choices
        watercourse_choices_with_ng = watercourse_choices + [NET_GAIN_LABEL] if watercourse_choices else [NET_GAIN_LABEL]
        
        to_delete_watercourse = []
        for idx, row in enumerate(st.session_state.manual_watercourse_rows):
            c1, c2, c3, c4, c5 = st.columns([0.30, 0.30, 0.15, 0.15, 0.10])
            with c1:
                if watercourse_choices_with_ng:
                    default_idx = None
                    if row.get("habitat_lost") and row["habitat_lost"] in watercourse_choices_with_ng:
                        default_idx = watercourse_choices_with_ng.index(row["habitat_lost"])
                    st.session_state.manual_watercourse_rows[idx]["habitat_lost"] = st.selectbox(
                        "Habitat Lost", watercourse_choices_with_ng,
                        index=default_idx,
                        key=f"manual_water_lost_{row['id']}",
                        help="Select watercourse habitat lost"
                    )
                else:
                    st.warning("No watercourse habitats available in catalog")
            with c2:
                if watercourse_choices_with_ng:
                    default_idx = None
                    if row.get("habitat_name") and row["habitat_name"] in watercourse_choices_with_ng:
                        default_idx = watercourse_choices_with_ng.index(row["habitat_name"])
                    st.session_state.manual_watercourse_rows[idx]["habitat_name"] = st.selectbox(
                        "Habitat to Mitigate", watercourse_choices_with_ng,
                        index=default_idx,
                        key=f"manual_water_hab_{row['id']}",
                        help="Select watercourse habitat to mitigate"
                    )
                else:
                    st.warning("No watercourse habitats available")
            with c3:
                st.session_state.manual_watercourse_rows[idx]["units"] = st.number_input(
                    "Units", min_value=0.0, step=0.01, value=float(row.get("units", 0.0)),
                    key=f"manual_water_units_{row['id']}"
                )
            with c4:
                st.session_state.manual_watercourse_rows[idx]["price_per_unit"] = st.number_input(
                    "Price/Unit (£)", min_value=0.0, step=1.0, value=float(row.get("price_per_unit", 0.0)),
                    key=f"manual_water_price_{row['id']}"
                )
            with c5:
                if st.button("🗑️", key=f"del_manual_water_{row['id']}", help="Remove this row"):
                    to_delete_watercourse.append(row["id"])
        
        if to_delete_watercourse:
            st.session_state.manual_watercourse_rows = [r for r in st.session_state.manual_watercourse_rows if r["id"] not in to_delete_watercourse]
            st.rerun()
        
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            if st.button("➕ Add Watercourse Entry", key="add_manual_water_btn"):
                st.session_state.manual_watercourse_rows.append({
                    "id": st.session_state._next_manual_watercourse_id,
                    "habitat_lost": "",
                    "habitat_name": "",
                    "units": 0.0,
                    "price_per_unit": 0.0
                })
                st.session_state._next_manual_watercourse_id += 1
                st.rerun()
        with col2:
            if st.button("🧹 Clear Watercourse", key="clear_manual_water_btn"):
                st.session_state.manual_watercourse_rows = []
                st.rerun()

# Add this to your optimization results section (after the downloads):
if (st.session_state.get("optimization_complete", False) and 
    isinstance(st.session_state.get("last_alloc_df"), pd.DataFrame) and 
    not st.session_state["last_alloc_df"].empty):
    
    # Get data from session state
    session_alloc_df = st.session_state["last_alloc_df"]
    
    # Reconstruct demand_df from session state
    session_demand_df = pd.DataFrame(
        [{"habitat_name": sstr(r["habitat_name"]), "units_required": float(r.get("units", 0.0) or 0.0)}
         for r in st.session_state.demand_rows if sstr(r["habitat_name"]) and float(r.get("units", 0.0) or 0.0) > 0]
    )
    
    # Calculate total cost from session data
    session_total_cost = session_alloc_df["cost"].sum()
    
    st.markdown("---")
    st.markdown("#### 📧 Client Report Generation")
    
    # Initialize email inputs in session state (only if not exists)
    if "email_client_name" not in st.session_state:
        st.session_state.email_client_name = "INSERT NAME"
    if "email_ref_number" not in st.session_state:
        st.session_state.email_ref_number = "BNG00XXX"
    if "email_location" not in st.session_state:
        st.session_state.email_location = "INSERT LOCATION"
    
    with st.expander("Generate Client Email Report", expanded=True):  # Force it to stay expanded
        st.markdown("**Generate a client-facing report table and email:**")
        
        # ========== FIXED FORM WITH PERSISTENCE ==========
        with st.form("client_email_form", clear_on_submit=False):
            st.markdown("**📝 Email Details:**")
            col_input1, col_input2, col_input3 = st.columns([1, 1, 1])
            
            with col_input1:
                form_client_name = st.text_input(
                    "Client Name", 
                    value=st.session_state.email_client_name,
                    key="form_client_name"
                )
            
            with col_input2:
                form_ref_number = st.text_input(
                    "Reference Number", 
                    value=st.session_state.email_ref_number,
                    key="form_ref_number"
                )
            
            with col_input3:
                form_location = st.text_input(
                    "Development Location", 
                    value=st.session_state.email_location,
                    key="form_location"
                )
            
            # Form submit button
            form_submitted = st.form_submit_button("Update Email Details")
        
        # Handle form submission OUTSIDE the form but INSIDE the expander
        if form_submitted:
            st.session_state.email_client_name = form_client_name
            st.session_state.email_ref_number = form_ref_number
            st.session_state.email_location = form_location
            st.success("Email details updated!")
            # Don't call st.rerun() - let it naturally update
        
        # Use the session state values for generating the report
        client_name = st.session_state.email_client_name
        ref_number = st.session_state.email_ref_number
        location = st.session_state.email_location    
        
        # Generate the report using session data and input values
        client_table, email_html = generate_client_report_table_fixed(
            session_alloc_df, session_demand_df, session_total_cost, ADMIN_FEE_GBP,
            client_name, ref_number, location,
            st.session_state.manual_hedgerow_rows,
            st.session_state.manual_watercourse_rows
        )
        
        # Display the table
        st.markdown("**Client Report Table:**")
        
        # Format for display (clean up column names)
        if not client_table.empty:
            display_table = client_table.copy()
            display_table = display_table.rename(columns={
                "Distinctiveness_Supply": "Supply Distinctiveness",
                "# Units_Supply": "Supply Units"
            })
            
            # Remove empty development impact columns for display
            cols_to_show = ["Distinctiveness", "Habitats Lost", "# Units", 
                           "Supply Distinctiveness", "Habitats Supplied", "Supply Units", 
                           "Price Per Unit", "Offset Cost"]
            
            st.dataframe(display_table[cols_to_show], use_container_width=True, hide_index=True)
        
        # Email generation
        st.markdown("**📧 Email Generation:**")
        
        # Create .eml file content
        import base64
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        
        subject = f"RE: BNG Units for site at {location} - {ref_number}"
        total_with_admin = session_total_cost + ADMIN_FEE_GBP
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = 'quotes@wildcapital.com'  # Replace with your actual email
        msg['To'] = ''  # Will be filled by user
        
        # Create text version for email clients that don't support HTML
        text_body = f"""Dear {client_name}

Our Ref: {ref_number}

Arbtech has advised us that you need Biodiversity Net Gain units for your development in {location}, and we're here to help you discharge your BNG condition.

Thank you for enquiring about BNG Units for your development in {location}

About Us

Wild Capital is a national supplier of BNG Units and environmental mitigation credits (Nutrient Neutrality, SANG), backed by institutional finance.

Your Quote - £{total_with_admin:,.0f} + VAT

[Please view the HTML version of this email for the detailed pricing breakdown table]

Total Units Required: {session_demand_df['units_required'].sum():.2f}
Total Units Supplied: {session_alloc_df['units_supplied'].sum():.2f}
Total Cost: £{total_with_admin:,.0f} + VAT

Next Steps
BNG is a pre-commencement, not a pre-planning, condition.

To accept the quote, let us know—we'll request some basic details before sending the Allocation Agreement. The price is fixed for 30 days, but unit availability is only guaranteed once the agreement is signed.

If you have any questions, please reply to this email or call 01962 436574.

Best regards,
Wild Capital Team"""
        
        # Attach text and HTML versions
        text_part = MIMEText(text_body, 'plain')
        html_part = MIMEText(email_html, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Convert to string
        eml_content = msg.as_string()
        
        # Download button for .eml file
        st.download_button(
            "📧 Download Email (.eml)",
            data=eml_content,
            file_name=f"BNG_Quote_{ref_number}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.eml",
            mime="message/rfc822",
            help="Download as .eml file - double-click to open in your email client with full HTML formatting"
        )

# Debug section (temporary - can remove later)
if st.checkbox("Show detailed debug info", value=False):
    st.subheader("Debug Information")
    st.write("**Session State Map-Related:**")
    debug_keys = ["target_lat", "target_lon", "target_lpa_name", "target_nca_name", 
                  "map_version", "optimization_complete"]
    for key in debug_keys:
        st.write(f"- {key}: {st.session_state.get(key, 'NOT SET')}")
    
    st.write("**Last Allocation DF:**")
    if "last_alloc_df" in st.session_state:
        if st.session_state["last_alloc_df"] is not None:
            st.write(f"Shape: {st.session_state['last_alloc_df'].shape}")
            st.write("Columns:", list(st.session_state["last_alloc_df"].columns))
        else:
            st.write("None")
    else:
        st.write("Not in session state")
    
    st.write("**Import Status:**")
    st.write(f"- folium imported: {folium is not None}")
    st.write(f"- st_folium imported: {st_folium is not None}")
    st.write(f"- folium_static available: {folium_static is not None}")







































































































































































