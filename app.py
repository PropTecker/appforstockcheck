# app.py — BNG Optimiser (Standalone), v9.7
# Policies:
# - Always minimise total cost; no proximity preference.
# - ≤ 2 sites overall (BANK_KEY).
# - Prefer a single site if total cost ≤ 1.01 * absolute cheapest.
# - One option per demand line, except FAR Medium 50/50 Orchard+Scrub (single option).
# - Low/Net Gain: use exact price rows only (no proxies).
# - Medium: Local/Adjacent single-habitat; FAR may use 50/50 Orchard+Scrub.
# - High/Very High: strict like-for-like.
# - Hedgerow excluded.
# - Tie-break on equal cost: prefer sites with larger total stock.
# - Robust result map.

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
import folium

# ========= Page =========
st.set_page_config(page_title="BNG Optimiser (Standalone)", page_icon="🧭", layout="wide")
st.markdown("<h2>BNG Optimiser — Standalone</h2>", unsafe_allow_html=True)

# ========= Safe strings =========
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
    return "hedgerow" in sstr(name).lower()

# ========= Login =========
DEFAULT_USER = "WC0323"
DEFAULT_PASS = "Wimbourne"

def require_login():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        with st.sidebar:
            if st.button("Log out", key="logout_btn"):
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

# ========= Endpoints =========
UA = {"User-Agent": "WildCapital-Optimiser/1.0 (+contact@example.com)"}  # set a real contact email
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

# ========= HTTP helpers =========
def http_get(url, params=None, headers=None, timeout=25):
    try:
        r = requests.get(url, params=params or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise RuntimeError(f"HTTP error for {url}: {e}")

def http_post(url, data=None, headers=None, timeout=25):
    try:
        r = requests.post(url, data=data or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise RuntimeError(f"HTTP POST error for {url}: {e}")

def safe_json(r: requests.Response) -> Dict[str, Any]:
    try:
        return r.json()
    except Exception:
        preview = (r.text or "")[:300]
        raise RuntimeError(f"Invalid JSON from {r.url} (status {r.status_code}). Starts: {preview}")

# ========= Geo helpers =========
def esri_polygon_to_geojson(geom: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not geom or "rings" not in geom:
        return None
    rings = geom.get("rings") or []
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}

def add_geojson_layer(fmap, geojson: Dict[str, Any], name: str, color: str, weight: int, fill_opacity: float = 0.05):
    if not geojson:
        return
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda x: {"color": color, "fillOpacity": fill_opacity, "weight": weight},
        tooltip=name
    ).add_to(fmap)

# ========= Geocoding / lookups =========
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

# ========= Tiering =========
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
    if b_lpa and t_lpa_n and b_lpa == t_lpa_n:
        return "local"
    if b_nca and t_nca_n and b_nca == t_nca_n:
        return "local"
    if b_lpa and b_lpa in lpa_neigh_norm:
        return "adjacent"
    if b_nca and b_nca in nca_neigh_norm:
        return "adjacent"
    return "far"

def select_contract_size(total_units: float, present: List[str]) -> str:
    tiers = set([sstr(x).lower() for x in present])
    if "fractional" in tiers and total_units < 0.1: return "fractional"
    if "small" in tiers and total_units < 2.5: return "small"
    if "medium" in tiers and total_units < 15: return "medium"
    for t in ["large", "medium", "small", "fractional"]:
        if t in tiers: return t
    return sstr(next(iter(present), "small")).lower()

# ========= Sidebar: backend =========
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

def load_backend(xls_file) -> Dict[str, pd.DataFrame]:
    x = pd.ExcelFile(xls_file)
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
    backend = load_backend(uploaded)
elif use_example:
    ex = Path("data/HabitatBackend_WITH_STOCK.xlsx")
    if ex.exists():
        backend = load_backend(ex.open("rb"))

if backend is None:
    st.warning("Upload your backend workbook to continue.", icon="⚠️")
    st.stop()

# ========= BANK_KEY normalisation helpers =========
def normalise_banks_keys(banks_df: pd.DataFrame) -> pd.DataFrame:
    """Keep/derive a stable BANK_KEY per site."""
    df = banks_df.copy()
    for c in ("bank_id", "bank_name", "BANK_KEY"):
        if c in df.columns:
            df[c] = df[c].map(sstr)
    if "BANK_KEY" not in df.columns or df["BANK_KEY"].map(sstr).eq("").all():
        if "bank_name" in df.columns and df["bank_name"].ne("").any():
            df["BANK_KEY"] = df["bank_name"].map(sstr)
        elif "bank_id" in df.columns:
            df["BANK_KEY"] = df["bank_id"].map(sstr)
        else:
            df["BANK_KEY"] = ""
    else:
        df["BANK_KEY"] = df["BANK_KEY"].map(sstr)
    return df

def attach_bank_key(df: pd.DataFrame, banks_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has BANK_KEY by merging from Banks on bank_id and filling as needed."""
    out = df.copy()
    for c in ("bank_id", "bank_name", "BANK_KEY"):
        if c in out.columns:
            out[c] = out[c].map(sstr)
    if "bank_id" in out.columns and "bank_id" in banks_df.columns and "BANK_KEY" in banks_df.columns:
        out = out.merge(
            banks_df[["bank_id","BANK_KEY"]].drop_duplicates(),
            on="bank_id", how="left", suffixes=("", "_from_banks")
        )
        if "BANK_KEY_from_banks" in out.columns:
            out["BANK_KEY"] = np.where(
                out["BANK_KEY_from_banks"].map(sstr).ne(""),
                out["BANK_KEY_from_banks"], out.get("BANK_KEY","")
            )
            out = out.drop(columns=["BANK_KEY_from_banks"])
    if "BANK_KEY" not in out.columns:
        out["BANK_KEY"] = ""
    mask_blank = out["BANK_KEY"].map(sstr).eq("")
    if "bank_name" in out.columns:
        out.loc[mask_blank, "BANK_KEY"] = out.loc[mask_blank, "bank_name"].map(sstr)
        mask_blank = out["BANK_KEY"].map(sstr).eq("")
    if "bank_id" in out.columns:
        out.loc[mask_blank, "BANK_KEY"] = out.loc[mask_blank, "bank_id"].map(sstr)
    out["BANK_KEY"] = out["BANK_KEY"].map(sstr)
    return out

# ========= Quotes policy (if WITH_STOCK columns exist) =========
if "available_excl_quotes" in backend["Stock"].columns and "quoted" in backend["Stock"].columns:
    s = backend["Stock"].copy()
    if quotes_hold_policy == "Ignore quotes (default)":
        s["quantity_available"] = s["available_excl_quotes"]
    elif quotes_hold_policy == "Quotes hold 100%":
        s["quantity_available"] = (s["available_excl_quotes"] - s["quoted"]).clip(lower=0)
    else:
        s["quantity_available"] = (s["available_excl_quotes"] - 0.5 * s["quoted"]).clip(lower=0)
    backend["Stock"] = s

# ========= Enrich Banks geography & keys =========
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
    cache = st.session_state.setdefault("bank_geo_cache", {})
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
backend["Banks"] = normalise_banks_keys(backend["Banks"])

# ========= Validate minimal columns =========
for sheet, cols in {
    "Pricing": ["bank_id","habitat_name","contract_size","tier"],
    "Stock": ["bank_id","habitat_name","stock_id","quantity_available"],
    "HabitatCatalog": ["habitat_name","broader_type","distinctiveness_name"],
}.items():
    missing = [c for c in cols if c not in backend[sheet].columns]
    if missing:
        st.error(f"{sheet} is missing required columns: {missing}")
        st.stop()

# ========= Normalise Pricing; drop Hedgerow =========
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
    if "broader_type" not in df.columns: df["broader_type"] = ""
    if "distinctiveness_name" not in df.columns: df["distinctiveness_name"] = ""
    df["habitat_name"] = df["habitat_name"].astype(str).str.strip()
    df = df[~df["habitat_name"].map(is_hedgerow)].copy()
    return df

# Apply hedgerow filter to Stock; normalise pricing; attach BANK_KEY from Banks
backend["Stock"] = backend["Stock"][~backend["Stock"]["habitat_name"].map(is_hedgerow)].copy()
backend["Pricing"] = normalise_pricing(backend["Pricing"])
backend["Pricing"] = attach_bank_key(backend["Pricing"], backend["Banks"])
backend["Stock"]   = attach_bank_key(backend["Stock"],   backend["Banks"])

# ========= Distinctiveness mapping =========
dist_levels_map = {
    sstr(r["distinctiveness_name"]): float(r["level_value"])
    for _, r in backend["DistinctivenessLevels"].iterrows()
}
dist_levels_map.update({k.lower(): v for k, v in list(dist_levels_map.items())})

# ========= Locate UI =========
with st.container():
    st.subheader("1) Locate target site")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        postcode = st.text_input("Postcode (quicker)", value="")
    with c2:
        address = st.text_input("Address (if no postcode)", value="")
    with c3:
        run_locate = st.button("Locate", key="locate_btn")

# Restore session
target_lpa_name = st.session_state.get("target_lpa_name", "")
target_nca_name = st.session_state.get("target_nca_name", "")
lpa_neighbors = st.session_state.get("lpa_neighbors", [])
nca_neighbors = st.session_state.get("nca_neighbors", [])
lpa_neighbors_norm = st.session_state.get("lpa_neighbors_norm", [])
nca_neighbors_norm = st.session_state.get("nca_neighbors_norm", [])
target_lat = st.session_state.get("target_lat", None)
target_lon = st.session_state.get("target_lon", None)
lpa_geojson = st.session_state.get("lpa_geojson", None)
nca_geojson = st.session_state.get("nca_geojson", None)

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
    return t_lpa, t_nca, lpa_nei, nca_nei, lpa_nei_norm, nca_nei_norm, lat, lon, lpa_gj, nca_gj

if run_locate:
    try:
        (target_lpa_name, target_nca_name,
         lpa_neighbors, nca_neighbors,
         lpa_neighbors_norm, nca_neighbors_norm,
         target_lat, target_lon, lpa_geojson, nca_geojson) = find_site(postcode, address)
        st.success(f"Found LPA: **{target_lpa_name}** | NCA: **{target_nca_name}**")
    except Exception as e:
        st.error(f"Location error: {e}")

def render_base_map():
    lat0 = st.session_state.get("target_lat", target_lat)
    lon0 = st.session_state.get("target_lon", target_lon)
    if lat0 is None or lon0 is None:
        fmap = folium.Map(location=[54.5, -2.5], zoom_start=5, control_scale=True)
        folium.LayerControl(collapsed=True).add_to(fmap)
        return fmap, None, None
    fmap = folium.Map(location=[lat0, lon0], zoom_start=11, control_scale=True)
    add_geojson_layer(fmap, st.session_state.get("lpa_geojson", lpa_geojson),
                      f"LPA: {st.session_state.get('target_lpa_name', target_lpa_name) or 'LPA'}",
                      color="red", weight=2, fill_opacity=0.05)
    add_geojson_layer(fmap, st.session_state.get("nca_geojson", nca_geojson),
                      f"NCA: {st.session_state.get('target_nca_name', target_nca_name) or 'NCA'}",
                      color="yellow", weight=3, fill_opacity=0.05)
    folium.CircleMarker([lat0, lon0], radius=6, color="red", fill=True, tooltip="Target site").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap, lat0, lon0

if (target_lat is not None) and (target_lon is not None):
    st.markdown("### Map")
    fmap_preview, _, _ = render_base_map()
    st_folium(fmap_preview, height=420, returned_objects=[], use_container_width=True, key="basemap")

# ========= Demand =========
st.subheader("2) Demand (units required)")
NET_GAIN_LABEL = "Net Gain (Low-equivalent)"

def init_demand_state():
    if "demand_rows" not in st.session_state:
        st.session_state.demand_rows = [{"id": 1, "habitat_name": "", "units": 0.0}]
        st.session_state._next_row_id = 2

init_demand_state()

HAB_CHOICES = sorted(
    [sstr(x) for x in backend["HabitatCatalog"]["habitat_name"].dropna().unique().tolist()] + [NET_GAIN_LABEL]
)

with st.container(border=True):
    st.markdown("**Add habitats one by one** (type to search the catalog):")
    to_delete = []
    for idx, row in enumerate(st.session_state.demand_rows):
        c1, c2, c3 = st.columns([0.62, 0.28, 0.10])
        with c1:
            st.session_state.demand_rows[idx]["habitat_name"] = st.selectbox(
                "Habitat", HAB_CHOICES,
                index=(HAB_CHOICES.index(row["habitat_name"]) if row["habitat_name"] in HAB_CHOICES else 0),
                key=f"hab_{row['id']}",
                help="Start typing to filter",
            )
        with c2:
            st.session_state.demand_rows[idx]["units"] = st.number_input(
                "Units", min_value=0.0, step=0.01, value=float(row.get("units", 0.0)), key=f"units_{row['id']}"
            )
        with c3:
            if st.button("🗑️", key=f"del_{row['id']}", help="Remove this row"):
                to_delete.append(row["id"]])
    if to_delete:
        st.session_state.demand_rows = [r for r in st.session_state.demand_rows if r["id"] not in to_delete]
        st.rerun()

    cc1, cc2, cc3 = st.columns([0.33, 0.33, 0.34])
    with cc1:
        if st.button("➕ Add habitat", key="add_hab_btn"):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": HAB_CHOICES[0] if HAB_CHOICES else "", "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc2:
        if st.button("➕ Net Gain (Low-equivalent)", key="add_ng_btn",
                     help="Adds a 'Net Gain' line (trades like Low distinctiveness)."):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": NET_GAIN_LABEL, "units": 0.0}
            )
            st.session_state._next_row_id += 1
            st.rerun()
    with cc3:
        if st.button("🧹 Clear all", key="clear_all_btn"):
            init_demand_state()
            st.rerun()

total_units = sum([float(r.get("units", 0.0) or 0.0) for r in st.session_state.demand_rows])
st.metric("Total units", f"{total_units:.2f}")

demand_df = pd.DataFrame(
    [{"habitat_name": sstr(r["habitat_name"]), "units_required": float(r.get("units", 0.0) or 0.0)}
     for r in st.session_state.demand_rows if sstr(r["habitat_name"]) and float(r.get("units", 0.0) or 0.0) > 0]
)

# 🚫 Block Hedgerow
if not demand_df.empty:
    banned = [h for h in demand_df["habitat_name"] if is_hedgerow(h)]
    if banned:
        st.error("Hedgerow units cannot be traded in this optimiser. Remove: " + ", ".join(sorted(set(banned))))
        st.stop()

if not demand_df.empty:
    st.dataframe(demand_df, use_container_width=True, hide_index=True)
else:
    st.info("Add at least one habitat and units to continue.", icon="ℹ️")

# ========= Official rules (legality only) =========
def enforce_catalog_rules_official(demand_row, supply_row, dist_levels_map_local, explicit_rule: bool) -> bool:
    if explicit_rule:
        return True
    dh = sstr(demand_row.get("habitat_name"))
    if dh == NET_GAIN_LABEL:
        return True  # Net Gain behaves like Low legally
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
    return sh == dh  # High/Very High like-for-like

# ========= Options builder (strict pricing) =========
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

    # Enrich stock with bank geo + catalog
    stock_full = Stock.merge(
        Banks[["bank_id","bank_name","lpa_name","nca_name","BANK_KEY"]],
        on="bank_id", how="left"
    ).merge(Catalog, on="habitat_name", how="left")
    stock_full = stock_full[~stock_full["habitat_name"].map(is_hedgerow)].copy()

    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()

    Catalog_cols = ["habitat_name", "broader_type", "distinctiveness_name"]
    pc_join = pricing_cs.merge(
        Catalog[Catalog_cols], on="habitat_name", how="left", suffixes=("", "_cat")
    )
    pc_join["broader_type_eff"] = np.where(pc_join["broader_type"].astype(str).str.len()>0,
                                           pc_join["broader_type"], pc_join["broader_type_cat"])
    pc_join["distinctiveness_name_eff"] = np.where(pc_join["distinctiveness_name"].astype(str).str.len()>0,
                                                   pc_join["distinctiveness_name"], pc_join["distinctiveness_name_cat"])
    for c in ["broader_type_eff","distinctiveness_name_eff","tier","bank_id","habitat_name","BANK_KEY","bank_name"]:
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
        pr_exact = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                                    (pricing_enriched["tier"] == tier) &
                                    (pricing_enriched["habitat_name"] == supply_habitat)]
        if not pr_exact.empty:
            r = pr_exact.sort_values("price").iloc[0]
            return float(r["price"]), "exact", sstr(r["habitat_name"])
        d_key = sstr(demand_dist).lower()
        if d_key == "low":
            return None  # strict: no proxy for Low
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
        return None  # High/VHigh exact only

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
        if dem_hab == NET_GAIN_LABEL:
            d_broader = ""
            d_dist = "Low"
        else:
            dcat = Catalog[Catalog["habitat_name"] == dem_hab]
            d_broader = sstr(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
            d_dist = sstr(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""

        cand_parts = []
        explicit = False
        Trading = backend.get("TradingRules", pd.DataFrame())
        if not Trading.empty and dem_hab in set(Trading["demand_habitat"].astype(str)):
            explicit = True
            for _, rule in Trading[Trading["demand_habitat"] == dem_hab].iterrows():
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

        # Paired 50/50 Orchard+Scrub for FAR Medium
        if sstr(d_dist).lower() == "medium":
            ORCHARD_NAME = ORCHARD_NAME  # keep naming local
            SCRUB_NAME = SCRUB_NAME
            if ORCHARD_NAME and SCRUB_NAME:
                banks_keys = stock_full["BANK_KEY"].dropna().unique().tolist()
                for bk in banks_keys:
                    orch_rows = stock_full[(stock_full["BANK_KEY"] == bk) & (stock_full["habitat_name"] == ORCHARD_NAME)]
                    scrub_rows = stock_full[(stock_full["BANK_KEY"] == bk) & (
                        (stock_full["habitat_name"] == SCRUB_NAME) |
                        (stock_full["habitat_name"].str.contains("scrub", case=False, na=False)) |
                        (stock_full["habitat_name"].str.contains("bramble", case=False, na=False))
                    )]
                    if orch_rows.empty or scrub_rows.empty:
                        continue
                    for _, o in orch_rows.iterrows():
                        for _, s2 in scrub_rows.iterrows():
                            tier_b = tier_for_bank(
                                sstr(s2.get("lpa_name")), sstr(s2.get("nca_name")),
                                target_lpa, target_nca, lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
                            )
                            if tier_b != "far":
                                continue
                            pi_o = find_price_for_supply(bk, ORCHARD_NAME, tier_b, d_broader, d_dist)
                            pi_s = find_price_for_supply(bk, s2["habitat_name"], tier_b, d_broader, d_dist)
                            if not pi_o or not pi_s:
                                continue
                            cap_o = float(o.get("quantity_available", 0) or 0.0)
                            cap_s = float(s2.get("quantity_available", 0) or 0.0)
                            if cap_o <= 0 or cap_s <= 0:
                                continue
                            price = 0.5 * float(pi_o[0]) + 0.5 * float(pi_s[0])
                            options.append({
                                "type": "paired",
                                "demand_idx": di,
                                "demand_habitat": dem_hab,
                                "BANK_KEY": bk,
                                "bank_name": sstr(o.get("bank_name")),
                                "bank_id": sstr(o.get("bank_id")),
                                "supply_habitat": f"{ORCHARD_NAME} + {sstr(s2['habitat_name'])}",
                                "tier": tier_b,
                                "proximity": tier_b,
                                "unit_price": price,
                                "stock_use": {sstr(o["stock_id"]): 0.5, sstr(s2["stock_id"]): 0.5},
                                "price_source": "group-proxy",
                                "price_habitat": f"{pi_o[2]} + {pi_s[2]}",
                            })

    return options, stock_caps, stock_bankkey

# ========= Optimiser =========
def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str],
             lpa_neigh_norm: List[str], nca_neigh_norm: List[str],
             single_site_slack: float = 0.01) -> Tuple[pd.DataFrame, float, str]:
    chosen_size = select_size_for_demand(demand_df, backend["Pricing"])
    options, stock_caps, stock_bankkey = prepare_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )
    if not options:
        raise RuntimeError("No feasible options. Check prices/stock/rules or location tiers.")

    # Index by demand
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

    def solve_with_bank_limit(max_sites: int):
        import pulp
        prob = pulp.LpProblem(f"BNG_MinCost_SitesLe{max_sites}", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(len(options))]
        z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(len(options))]
        y = {b: pulp.LpVariable(f"y_{norm_name(b)}", lowBound=0, upBound=1, cat="Binary") for b in bank_keys}

        # total bank capacity for tie-break
        bank_capacity_total: Dict[str, float] = {b: 0.0 for b in bank_keys}
        for sid, cap in stock_caps.items():
            bkey = stock_bankkey.get(sid, "")
            if bkey in bank_capacity_total:
                bank_capacity_total[bkey] += float(cap or 0.0)

        cost_term = pulp.lpSum([options[i]["unit_price"] * x[i] for i in range(len(options))])
        eps = 1e-9
        tie_break = -eps * pulp.lpSum([bank_capacity_total[b] * y[b] for b in bank_keys])
        prob += cost_term + tie_break

        prob += pulp.lpSum([y[b] for b in bank_keys]) <= max_sites

        for i, opt in enumerate(options):
            prob += z[i] <= y[opt["BANK_KEY"]]

        for di, idxs in idx_by_dem.items():
            need = dem_need[di]
            prob += pulp.lpSum([z[i] for i in idxs]) == 1
            prob += pulp.lpSum([x[i] for i in idxs]) == need
            for i in idxs:
                prob += x[i] <= need * z[i]

        use_map: Dict[str, List[Tuple[int, float]]] = {}
        for i, opt in enumerate(options):
            for sid, coef in opt["stock_use"].items():
                use_map.setdefault(sid, []).append((i, float(coef)))
        for sid, pairs in use_map.items():
            cap = float(stock_caps.get(sid, 0.0))
            prob += pulp.lpSum([coef * x[i] for (i, coef) in pairs]) <= cap

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal", "Feasible"):
            return None, None

        rows, total_cost = [], 0.0
        for i in range(len(options)):
            qty = x[i].value() or 0.0
            sel = z[i].value() or 0.0
            if sel >= 0.5 and qty > 0:
                opt = options[i]
                rows.append({
                    "demand_habitat": opt["demand_habitat"],
                    "BANK_KEY": opt["BANK_KEY"],
                    "bank_name": opt.get("bank_name",""),
                    "bank_id": opt.get("bank_id",""),
                    "supply_habitat": opt["supply_habitat"],
                    "allocation_type": opt["type"],
                    "tier": opt["tier"],
                    "units_supplied": qty,
                    "unit_price": opt["unit_price"],
                    "cost": qty * opt["unit_price"],
                    "price_source": opt.get("price_source",""),
                    "price_habitat": opt.get("price_habitat",""),
                })
                total_cost += qty * opt["unit_price"]
        return (pd.DataFrame(rows), float(total_cost))

    # Solve for absolute cheapest with ≤2 sites
    if not _HAS_PULP:
        raise RuntimeError("MILP solver (PuLP) not available in this environment.")
    sol2, cost2 = solve_with_bank_limit(max_sites=2)
    if sol2 is None:
        raise RuntimeError("Optimiser infeasible with ≤2 sites.")

    # Try single-site solution, prefer if within 1%
    sol1, cost1 = solve_with_bank_limit(max_sites=1)
    chosen_df, chosen_cost = sol2, cost2
    if sol1 is not None and cost1 <= (1.0 + single_site_slack) * cost2 + 1e-6:
        chosen_df, chosen_cost = sol1, cost1

    return chosen_df, chosen_cost, chosen_size

# ========= Run optimiser UI =========
st.subheader("3) Run optimiser")
left, right = st.columns([1,1])
with left:
    run = st.button("Optimise now", type="primary", disabled=demand_df.empty, key="optimise_btn")
with right:
    if target_lpa_name or target_nca_name:
        st.caption(f"LPA: {target_lpa_name or '—'} | NCA: {target_nca_name or '—'} | "
                   f"LPA neigh: {len(lpa_neighbors)} | NCA neigh: {len(nca_neighbors)}")
    else:
        st.caption("Tip: run ‘Locate’ first for precise tiers (else assumes ‘far’).")

# ========= Diagnostics =========
with st.expander("🔎 Diagnostics", expanded=False):
    try:
        if demand_df.empty:
            st.info("Add some demand rows above to see diagnostics.", icon="ℹ️")
        else:
            dd = demand_df.copy()
            present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
            total_units = float(dd["units_required"].sum())
            chosen_size = select_contract_size(total_units, present_sizes)
            st.write(f"**Chosen contract size:** `{chosen_size}` (present sizes: {present_sizes}, total demand: {total_units})")
            st.write(f"**Target LPA:** {target_lpa_name or '—'}  |  **Target NCA:** {target_nca_name or '—'}")
            st.write(f"**# LPA neighbours:** {len(lpa_neighbors)}  | **# NCA neighbours:** {len(nca_neighbors)}")

            # Stock sanity
            try:
                s = backend["Stock"].copy()
                s["quantity_available"] = pd.to_numeric(s["quantity_available"], errors="coerce").fillna(0)
                st.write("**Stock sanity**")
                st.write(f"Non-zero stock rows: **{(s['quantity_available']>0).sum()}** | "
                         f"Total available units: **{s['quantity_available'].sum():.2f}**")
            except Exception as _e:
                st.warning(f"Stock sanity check failed: {_e}")

            options_preview, _, _ = prepare_options(
                dd, chosen_size,
                sstr(target_lpa_name), sstr(target_nca_name),
                [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
                lpa_neighbors_norm, nca_neighbors_norm
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
                    st.caption("Note: `price_source='group-proxy'` is Medium-only (group/distinctiveness proxy).")

                try:
                    st.markdown("**Cheapest candidates per demand (top 5)**")
                    for dem in dd["habitat_name"].unique():
                        sub = cand_df[cand_df["demand_habitat"] == dem].copy()
                        if sub.empty:
                            continue
                        sub = sub.sort_values("unit_price").head(5)
                        sub = sub[["bank_name","BANK_KEY","proximity","allocation_type","supply_habitat","unit_price","price_source","price_habitat"]]
                        st.write(f"**{dem}**")
                        st.dataframe(sub, use_container_width=True, hide_index=True)
                except Exception:
                    pass

    except Exception as de:
        st.error(f"Diagnostics error: {de}")

# ========= Price readout (what the optimiser sees) =========
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

            st.markdown("**Summary by site & tier**")
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

# ========= Pricing completeness =========
with st.expander("💷 Pricing completeness (this contract size)", expanded=False):
    try:
        if demand_df.empty:
            st.info("Add demand rows to see pricing completeness.")
        else:
            present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
            total_units = float(demand_df["units_required"].sum())
            chosen_size = select_contract_size(total_units, present_sizes)

            pr = backend["Pricing"].copy()
            pr = pr[pr["contract_size"] == chosen_size]
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
                st.success(f"All exact pricing rows exist for size `{chosen_size}` across the demanded habitats.")
            else:
                st.warning("Some exact pricing rows are missing — that’s fine if those rows are untradeable or Medium uses proxies.")
                st.dataframe(
                    missing.sort_values(["habitat_name","bank_id","tier"]),
                    use_container_width=True, hide_index=True
                )
    except Exception as e:
        st.error(f"Pricing completeness error: {e}")

# --- Run optimiser ---
if run:
    try:
        if demand_df.empty:
            st.error("Add at least one demand row before optimising.")
            st.stop()

        # Auto-locate if user filled address/postcode but forgot Locate
        if not sstr(target_lpa_name) or not sstr(target_nca_name):
            if sstr(postcode) or sstr(address):
                try:
                    (_t_lpa, _t_nca, _lpaN, _ncaN, _lpaNn, _ncaNn,
                     _lat, _lon, _lpa_gj, _nca_gj) = find_site(postcode, address)
                except Exception as e:
                    st.warning(f"Auto-locate failed: {e}. Proceeding with 'far' tiers only.")

        # Validate against catalog—but allow the special Net Gain label
        cat_names_run = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        unknown = [h for h in demand_df["habitat_name"] if h not in cat_names_run and h != NET_GAIN_LABEL]
        if unknown:
            st.error(f"These demand habitats aren’t in the catalog: {unknown}")
            st.stop()

        target_lpa = sstr(st.session_state.get("target_lpa_name", target_lpa_name))
        target_nca = sstr(st.session_state.get("target_nca_name", target_nca_name))
        lpa_neighbors = st.session_state.get("lpa_neighbors", lpa_neighbors)
        nca_neighbors = st.session_state.get("nca_neighbors", nca_neighbors)
        lpa_neighbors_norm = st.session_state.get("lpa_neighbors_norm", lpa_neighbors_norm)
        nca_neighbors_norm = st.session_state.get("nca_neighbors_norm", nca_neighbors_norm)

        alloc_df, total_cost, size = optimise(
            demand_df,
            target_lpa, target_nca,
            [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
            lpa_neighbors_norm, nca_neighbors_norm,
            single_site_slack=0.01  # ← 1% preference for single site
        )

        st.success(f"Optimisation complete. Contract size = **{size}**. Total cost: **£{total_cost:,.0f}**")

        st.markdown("#### Allocation detail")
        st.dataframe(alloc_df, use_container_width=True)
        if "price_source" in alloc_df.columns:
            st.caption("Note: `price_source='group-proxy'` rows are Medium-only (via group/distinctiveness).")

        st.markdown("#### By site (BANK_KEY)")
        by_bank = alloc_df.groupby(["BANK_KEY","bank_name","bank_id"], as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            cost=("cost","sum")
        ).sort_values("cost", ascending=False)
        st.dataframe(by_bank, use_container_width=True)

        st.markdown("#### By habitat (supply)")
        by_hab = alloc_df.groupby("supply_habitat", as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            cost=("cost","sum")
        )
        st.dataframe(by_hab, use_container_width=True)

        # ------- Map overlay: chosen sites and links -------
        fmap, lat0, lon0 = render_base_map()

        # Site coordinates via Banks (geocode if needed)
        bank_coords: Dict[str, Tuple[float,float]] = {}
        banks_df = backend["Banks"].copy()
        for _, b in banks_df.iterrows():
            bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
            loc = bank_row_to_latlon(b)
            if loc:
                bank_coords[bkey] = (loc[0], loc[1])

        if 'alloc_df' in locals() and not alloc_df.empty:
            grouped = alloc_df.groupby(["BANK_KEY","bank_name"], dropna=False)
            for (bkey, bname), g in grouped:
                try:
                    latlon = bank_coords.get(sstr(bkey))
                    if not latlon:
                        continue
                    lat_b, lon_b = latlon

                    # Draw site catchments
                    bank_catch_cache = st.session_state.setdefault("bank_catchment_geo", {})
                    cache_key = sstr(bkey)
                    if cache_key not in bank_catch_cache:
                        try:
                            b_lpa_name, b_lpa_gj, b_nca_name, b_nca_gj = get_catchment_geo_for_point(lat_b, lon_b)
                            bank_catch_cache[cache_key] = {
                                "lpa_name": b_lpa_name, "lpa_gj": b_lpa_gj,
                                "nca_name": b_nca_name, "nca_gj": b_nca_gj,
                            }
                        except Exception:
                            bank_catch_cache[cache_key] = {"lpa_name": "", "lpa_gj": None, "nca_name": "", "nca_gj": None}

                    bgeo = bank_catch_cache[cache_key]
                    add_geojson_layer(
                        fmap, bgeo.get("lpa_gj"),
                        name=f"{sstr(bname) or sstr(bkey)} — Site LPA: {sstr(bgeo.get('lpa_name')) or 'Unknown'}",
                        color="green", weight=2, fill_opacity=0.03
                    )
                    add_geojson_layer(
                        fmap, bgeo.get("nca_gj"),
                        name=f"{sstr(bname) or sstr(bkey)} — Site NCA: {sstr(bgeo.get('nca_name')) or 'Unknown'}",
                        color="blue", weight=3, fill_opacity=0.03
                    )

                    # Marker
                    popup_lines = []
                    total_units_b = g["units_supplied"].sum()
                    total_cost_b = g["cost"].sum()
                    popup_lines.append(f"<b>Site:</b> {sstr(bname) or sstr(bkey)}")
                    popup_lines.append(f"<b>Total units:</b> {total_units_b:.3f}")
                    popup_lines.append(f"<b>Total cost:</b> £{total_cost_b:,.0f}")
                    popup_lines.append("<b>Breakdown:</b>")
                    for _, r in g.sort_values("units_supplied", ascending=False).head(6).iterrows():
                        popup_lines.append(f"- {sstr(r['supply_habitat'])} — {float(r['units_supplied']):.3f} ({sstr(r['tier'])})")

                    folium.Marker(
                        [lat_b, lon_b],
                        icon=folium.Icon(color="green", icon="leaf"),
                        popup=folium.Popup("<br>".join(popup_lines), max_width=420)
                    ).add_to(fmap)

                    # Route line only if target coords exist
                    if (lat0 is not None) and (lon0 is not None):
                        folium.PolyLine(
                            locations=[[lat0, lon0], [lat_b, lon_b]],
                            weight=2, opacity=0.8, dash_array="6,6", color="blue",
                            tooltip=f"Supply route: target → {sstr(bname) or sstr(bkey)}"
                        ).add_to(fmap)
                except Exception as map_e:
                    st.caption(f"Skipped map overlay for site {sstr(bname) or sstr(bkey)}: {map_e}")

        st.markdown("### Map (with selected supply)")
        st_folium(fmap, height=520, returned_objects=[], use_container_width=True, key="resultmap")

        # downloads
        def df_to_csv_bytes(df):
            buf = BytesIO()
            buf.write(df.to_csv(index=False).encode("utf-8"))
            buf.seek(0)
            return buf

        st.download_button("Download allocation (CSV)", data=df_to_csv_bytes(alloc_df),
                           file_name="allocation.csv", mime="text/csv")
        st.download_button("Download by site (CSV)", data=df_to_csv_bytes(by_bank),
                           file_name="allocation_by_site.csv", mime="text/csv")
        st.download_button("Download by habitat (CSV)", data=df_to_csv_bytes(by_hab),
                           file_name="allocation_by_habitat.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Optimiser error: {e}")


















