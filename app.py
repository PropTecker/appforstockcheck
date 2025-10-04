# app.py — BNG Optimiser (Standalone), v10
# Policy hard-coding (per user spec 2025-10-04):
# - Minimise total cost (primary). No FAR-first stage.
# - ≤ 2 banks across the entire quote (hard).
# - Each demand line must be filled by exactly one option (no cross-bank/habitat splits).
# - Low: any legal (non-hedgerow); no proxy pricing; choose cheapest; tie -> most stock.
# - Medium:
#     - Local/Adjacent: like-for-like only (no pair).
#     - FAR: allow Orchard+Scrub 50/50 pair (same bank). If cheaper, pick it; else cheapest single.
#     - If pair unavailable, still pick cheapest legal single.
# - High/Very High: like-for-like only.
# - Prices: only actual Excel rows; no group/distinctiveness proxy anywhere.
# - Min line size: 0.01
# - Diagnostics: cheapest-by-proximity per line; "why not chosen" flags for cheaper options.
# - Map: guarded + bank catchment perimeters.

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

# ========= Helpers =========
def sstr(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return ""
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
UA = {"User-Agent": "WildCapital-Optimiser/1.0 (+contact@example.com)"}  # put a real contact
POSTCODES_IO = "https://api.postcodes.io/postcodes/"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NCA_URL = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
           "National_Character_Areas_England/FeatureServer/0")
LPA_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
           "Local_Authority_Districts_December_2024_Boundaries_UK_BFC/FeatureServer/0")

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

# ========= Geo =========
def esri_polygon_to_geojson(geom: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not geom or "rings" not in geom: return None
    rings = geom.get("rings") or []
    if not rings: return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}

def add_geojson_layer(fmap, geojson: Dict[str, Any], name: str, color: str, weight: int, fill_opacity: float = 0.05):
    if not geojson: return
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda x: {"color": color, "fillOpacity": fill_opacity, "weight": weight},
        tooltip=name
    ).add_to(fmap)

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
    if not polygon_geom: return []
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
    b_lpa = norm_name(bank_lpa); b_nca = norm_name(bank_nca)
    t_lpa_n = norm_name(t_lpa);  t_nca_n = norm_name(t_nca)
    if lpa_neigh_norm is None: lpa_neigh_norm = [norm_name(x) for x in (lpa_neigh or [])]
    if nca_neigh_norm is None: nca_neigh_norm = [norm_name(x) for x in (nca_neigh or [])]
    if b_lpa and t_lpa_n and b_lpa == t_lpa_n: return "local"
    if b_nca and t_nca_n and b_nca == t_nca_n: return "local"
    if b_lpa and b_lpa in lpa_neigh_norm: return "adjacent"
    if b_nca and b_nca in nca_neigh_norm: return "adjacent"
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

# ========= Normalise BANK_KEY everywhere =========
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

# ========= Quotes policy =========
if "available_excl_quotes" in backend["Stock"].columns and "quoted" in backend["Stock"].columns:
    s = backend["Stock"].copy()
    if quotes_hold_policy == "Ignore quotes (default)":
        s["quantity_available"] = s["available_excl_quotes"]
    elif quotes_hold_policy == "Quotes hold 100%":
        s["quantity_available"] = (s["available_excl_quotes"] - s["quoted"]).clip(lower=0)
    else:
        s["quantity_available"] = (s["available_excl_quotes"] - 0.5 * s["quoted"]).clip(lower=0)
    backend["Stock"] = s

# ========= Enrich Banks geography =========
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
        lpa_now = sstr(row.get("lpa_name")); nca_now = sstr(row.get("nca_name"))
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

    # accept any of these header spellings for the price column
    price_cols = [c for c in df.columns if c.strip().lower() in ("price","unit price","unit_price","unitprice")]
    if not price_cols:
        st.error("Pricing sheet must contain a 'Price' column (or 'Unit Price').")
        st.stop()

    df["price"] = pd.to_numeric(df[price_cols[0]], errors="coerce")

    # ✅ use .str.lower() (Series string accessor), not .lower()
    df["tier"] = df["tier"].astype(str).str.strip().str.lower()
    df["contract_size"] = df["contract_size"].astype(str).str.strip().str.lower()

    df["bank_id"] = df["bank_id"].astype(str).str.strip()
    df = make_bank_key_col(df, backend["Banks"])

    if "broader_type" not in df.columns:
        df["broader_type"] = ""
    if "distinctiveness_name" not in df.columns:
        df["distinctiveness_name"] = ""

    df["habitat_name"] = df["habitat_name"].astype(str).str.strip()

    # 🚫 never price Hedgerow
    df = df[~df["habitat_name"].map(is_hedgerow)].copy()
    return df


backend["Stock"] = backend["Stock"][~backend["Stock"]["habitat_name"].map(is_hedgerow)].copy()
backend["Pricing"] = normalise_pricing(backend["Pricing"])

# ========= Distinctiveness mapping (for rules only; NOT proxy pricing) =========
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
        return float(js[0]["lat"]), float(js[0]["lon"])
    r = http_get("https://photon.komoot.io/api/", params={"q": sstr(addr), "limit": 1})
    js = safe_json(r)
    feats = js.get("features") or []
    if feats:
        lon, lat = feats[0]["geometry"]["coordinates"]
        return float(lat), float(lon)
    raise RuntimeError("Address geocoding failed.")

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
    if target_lat is None or target_lon is None:
        fmap = folium.Map(location=[54.5, -2.5], zoom_start=5, control_scale=True)
        folium.LayerControl(collapsed=True).add_to(fmap)
        return fmap
    fmap = folium.Map(location=[target_lat, target_lon], zoom_start=11, control_scale=True)
    add_geojson_layer(fmap, lpa_geojson, f"LPA: {target_lpa_name}" if target_lpa_name else "LPA",
                      color="red", weight=2, fill_opacity=0.05)
    add_geojson_layer(fmap, nca_geojson, f"NCA: {target_nca_name}" if target_nca_name else "NCA",
                      color="yellow", weight=3, fill_opacity=0.05)
    folium.CircleMarker([target_lat, target_lon], radius=6, color="red", fill=True, tooltip="Target site").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap

if (target_lat is not None) and (target_lon is not None):
    st.markdown("### Map")
    st_folium(render_base_map(), height=420, returned_objects=[], use_container_width=True)

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
                to_delete.append(row["id"])
    if to_delete:
        st.session_state.demand_rows = [r for r in st.session_state.demand_rows if r["id"] not in to_delete]
        st.rerun()

    c1, c2, c3 = st.columns([0.33,0.33,0.34])
    with c1:
        if st.button("➕ Add habitat", key="add_hab_btn"):
            st.session_state.demand_rows.append({"id": st.session_state._next_row_id,
                                                 "habitat_name": HAB_CHOICES[0] if HAB_CHOICES else "",
                                                 "units": 0.0})
            st.session_state._next_row_id += 1
            st.rerun()
    with c2:
        if st.button("➕ Net Gain (Low-equivalent)", key="add_ng_btn"):
            st.session_state.demand_rows.append({"id": st.session_state._next_row_id,
                                                 "habitat_name": NET_GAIN_LABEL, "units": 0.0})
            st.session_state._next_row_id += 1
            st.rerun()
    with c3:
        if st.button("🧹 Clear all", key="clear_all_btn"):
            init_demand_state(); st.rerun()

total_units = sum([float(r.get("units", 0.0) or 0.0) for r in st.session_state.demand_rows])
st.metric("Total units", f"{total_units:.2f}")

demand_df = pd.DataFrame(
    [{"habitat_name": sstr(r["habitat_name"]), "units_required": float(r.get("units", 0.0) or 0.0)}
     for r in st.session_state.demand_rows if sstr(r["habitat_name"]) and float(r.get("units", 0.0) or 0.0) > 0]
)

# Block hedgerow in demand
if not demand_df.empty:
    banned = [h for h in demand_df["habitat_name"] if is_hedgerow(h)]
    if banned:
        st.error("Hedgerow units cannot be traded in this optimiser. Remove: " + ", ".join(sorted(set(banned))))
        st.stop()

if not demand_df.empty:
    st.dataframe(demand_df, use_container_width=True, hide_index=True)
else:
    st.info("Add at least one habitat and units to continue.", icon="ℹ️")

# ========= Rules (NO proxy pricing) =========
def enforce_catalog_rules(demand_hab: str, supply_hab: str, proximity: str,
                          broader_map: Dict[str, str], distinct_map: Dict[str, str]) -> bool:
    # Net Gain behaves as Low demand
    if demand_hab == NET_GAIN_LABEL:
        return True
    d_group = sstr(broader_map.get(demand_hab))
    s_group = sstr(broader_map.get(supply_hab))
    d_dist_name = sstr(distinct_map.get(demand_hab))
    # Low: any legal (non-hedgerow)
    if d_dist_name.lower() == "low":
        return True
    # Medium:
    if d_dist_name.lower() == "medium":
        # Local/Adjacent -> like-for-like only
        if proximity in ("local","adjacent"):
            return supply_hab == demand_hab
        # FAR -> single habitat allowed (any that satisfies Medium rule when priced)
        return True
    # High / Very High -> strict like-for-like
    if d_dist_name.lower() in ("high","very high","very_high","very-high"):
        return supply_hab == demand_hab
    # default conservative
    return supply_hab == demand_hab

# ========= Option builder (exact prices only) =========
def select_size_for_demand(demand_df: pd.DataFrame, pricing_df: pd.DataFrame) -> str:
    present = pricing_df["contract_size"].drop_duplicates().tolist()
    total = float(demand_df["units_required"].sum())
    return select_contract_size(total, present)

def prepare_options(demand_df: pd.DataFrame,
                    chosen_size: str,
                    target_lpa: str, target_nca: str,
                    lpa_neigh: List[str], nca_neigh: List[str],
                    lpa_neigh_norm: List[str], nca_neigh_norm: List[str]):
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()

    # maps
    broader_map = {sstr(r["habitat_name"]): sstr(r["broader_type"]) for _, r in Catalog.iterrows()}
    distinct_map = {sstr(r["habitat_name"]): sstr(r["distinctiveness_name"]) for _, r in Catalog.iterrows()}

    # Pre-normalise key cols
    for df, cols in [
        (Banks, ["bank_id","bank_name","BANK_KEY","lpa_name","nca_name","lat","lon","postcode","address"]),
        (Catalog, ["habitat_name","broader_type","distinctiveness_name"]),
        (Stock, ["habitat_name","stock_id","bank_id","quantity_available","bank_name","BANK_KEY"]),
        (Pricing, ["habitat_name","contract_size","tier","bank_id","BANK_KEY","price","bank_name"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(sstr)
    Stock = make_bank_key_col(Stock, Banks)

    # Filter & attach bank names and geography to stock
    stock_full = Stock.merge(Banks[["bank_id","bank_name","lpa_name","nca_name"]], on="bank_id", how="left")
    stock_full = stock_full[~stock_full["habitat_name"].map(is_hedgerow)].copy()
    stock_full["quantity_available"] = pd.to_numeric(stock_full["quantity_available"], errors="coerce").fillna(0.0)

    # Pricing only chosen size; NO proxy usage later
    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()
    pricing_cs["price"] = pd.to_numeric(pricing_cs["price"], errors="coerce")

    # Build exact price lookup: (BANK_KEY, tier, habitat) -> price
    price_idx: Dict[Tuple[str,str,str], float] = {}
    for _, r in pricing_cs.iterrows():
        if pd.isna(r["price"]): continue
        price_idx[(sstr(r["BANK_KEY"]), sstr(r["tier"]), sstr(r["habitat_name"]))] = float(r["price"])

    # Stock caps per stock_id and which bank they belong to
    stock_caps: Dict[str, float] = {}
    stock_bankkey: Dict[str, str] = {}
    for _, srow in stock_full.iterrows():
        sid = sstr(srow["stock_id"])
        stock_caps[sid] = float(srow.get("quantity_available", 0.0) or 0.0)
        stock_bankkey[sid] = sstr(srow.get("BANK_KEY") or srow.get("bank_id"))

    # Helper to compute tier
    def tier_of_srow(srow):
        return tier_for_bank(
            srow.get("lpa_name",""), srow.get("nca_name",""),
            target_lpa, target_nca,
            lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
        )

    # Find orchard/scrub names from catalog
    def find_one(substr: str) -> Optional[str]:
        m = Catalog[Catalog["habitat_name"].str.contains(substr, case=False, na=False)]
        return sstr(m["habitat_name"].iloc[0]) if not m.empty else None
    ORCHARD_NAME = find_one("traditional orchard")
    SCRUB_NAME = find_one("mixed scrub") or find_one("scrub") or find_one("bramble")

    # Build candidate options for each demand line
    # Each option records: consumes certain stock_id(s) with coefficients (1.0 or 0.5)
    # and has a unit_price strictly from exact pricing rows.
    options: List[dict] = []
    for di, drow in demand_df.iterrows():
        d_hab = sstr(drow["habitat_name"])
        d_units = float(drow["units_required"])
        if d_units < 0.01:
            continue  # too small per policy minimum

        # All stock rows are candidates, we’ll filter by rules and pricing existence
        for _, srow in stock_full.iterrows():
            s_hab = sstr(srow["habitat_name"])
            if is_hedgerow(s_hab):  # belt & braces
                continue
            bank_key = sstr(srow.get("BANK_KEY") or srow.get("bank_id"))
            tier = tier_of_srow(srow)

            # Rules (no proxy): demand vs supply feasibility
            if not enforce_catalog_rules(d_hab, s_hab, tier, broader_map, distinct_map):
                continue

            # Price must exist EXACTLY for this (bank_key, tier, supply_hab)
            key = (bank_key, tier, s_hab)
            if key not in price_idx:
                continue
            unit_price = float(price_idx[key])

            # Candidate NORMAL option
            options.append({
                "option_id": f"norm::{di}::{srow['stock_id']}",
                "type": "normal",
                "demand_idx": di,
                "demand_habitat": d_hab,
                "BANK_KEY": bank_key,
                "bank_name": sstr(srow.get("bank_name")),
                "bank_id": sstr(srow.get("bank_id")),
                "supply_habitat": s_hab,
                "tier": tier,
                "proximity": tier,
                "unit_price": unit_price,
                "stock_use": {sstr(srow["stock_id"]): 1.0},
                "pair_components": None,  # for diagnostics
            })

        # FAR Medium: paired Orchard+Scrub (same bank), 50/50
        d_dist = sstr(distinct_map.get(d_hab)).lower() if d_hab != NET_GAIN_LABEL else "low"
        if d_dist == "medium" and ORCHARD_NAME and SCRUB_NAME:
            # find all banks that have both orchard and scrub stock rows
            sf = stock_full.copy()
            banks_with_orch = sf[sf["habitat_name"] == ORCHARD_NAME]["BANK_KEY"].unique().tolist()
            banks_with_scru = sf[sf["habitat_name"].str.contains("scrub|bramble", case=False, na=False)]["BANK_KEY"].unique().tolist()
            banks_both = sorted(set(banks_with_orch).intersection(banks_with_scru))
            for bk in banks_both:
                orch_rows = sf[(sf["BANK_KEY"] == bk) & (sf["habitat_name"] == ORCHARD_NAME)].copy()
                scrub_rows = sf[(sf["BANK_KEY"] == bk) &
                                (sf["habitat_name"].str.contains("scrub|bramble", case=False, na=False))].copy()
                if orch_rows.empty or scrub_rows.empty:
                    continue
                # Tier must be FAR (policy)
                # We’ll use the geometry from any row; tiers should match within same bank for our point
                any_row = orch_rows.iloc[0]
                tier_b = tier_of_srow(any_row)
                if tier_b != "far":
                    continue
                # Need exact price rows for BOTH orchard and scrub at (bk, 'far')
                key_o = (bk, "far", ORCHARD_NAME)
                # We don't force scrub to be a particular name; use each scrub row's exact name/price
                for _, o in orch_rows.iterrows():
                    for _, s2 in scrub_rows.iterrows():
                        key_s = (bk, "far", sstr(s2["habitat_name"]))
                        if key_o not in price_idx or key_s not in price_idx:
                            continue
                        unit_price = 0.5 * float(price_idx[key_o]) + 0.5 * float(price_idx[key_s])
                        options.append({
                            "option_id": f"pair::{di}::{o['stock_id']}::{s2['stock_id']}",
                            "type": "paired",
                            "demand_idx": di,
                            "demand_habitat": d_hab,
                            "BANK_KEY": bk,
                            "bank_name": sstr(o.get("bank_name")),
                            "bank_id": sstr(o.get("bank_id")),
                            "supply_habitat": f"{ORCHARD_NAME} + {sstr(s2['habitat_name'])}",
                            "tier": "far",
                            "proximity": "far",
                            "unit_price": unit_price,
                            "stock_use": {sstr(o["stock_id"]): 0.5, sstr(s2["stock_id"]): 0.5},
                            "pair_components": (ORCHARD_NAME, sstr(s2["habitat_name"])),
                        })

    return options, stock_caps, stock_bankkey

# ========= Optimiser (pure cost; ≤2 banks; 1 option per line) =========
def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str],
             lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[pd.DataFrame, float, str, pd.DataFrame]:
    chosen_size = select_size_for_demand(demand_df, backend["Pricing"])
    options, stock_caps, stock_bankkey = prepare_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )
    if not options:
        raise RuntimeError("No feasible options. Check prices/stock, rules, or tiers. (Exact pricing only.)")

    # Index options by demand line
    idx_by_dem: Dict[int, List[int]] = {}
    for i, opt in enumerate(options):
        idx_by_dem.setdefault(opt["demand_idx"], []).append(i)

    # Feasibility: each demand line must have at least one option with enough stock to cover its units (considering coefficients)
    infeasible_lines = []
    for di, drow in demand_df.iterrows():
        units = float(drow["units_required"])
        ok_any = False
        for i in idx_by_dem.get(di, []):
            opt = options[i]
            # compute maximum allocatable via available stock for this option
            max_take = float('inf')
            for sid, coef in opt["stock_use"].items():
                cap = float(stock_caps.get(sid, 0.0))
                if coef <= 0: continue
                max_take = min(max_take, cap / coef)
            if max_take + 1e-9 >= units:
                ok_any = True
                break
        if not ok_any:
            infeasible_lines.append((di, sstr(drow["habitat_name"]), units))
    if infeasible_lines:
        msg = "; ".join([f"{hab} (need {u:.2f})" for _, hab, u in infeasible_lines])
        raise RuntimeError("Insufficient stock for at least one demand line (consider other banks/types): " + msg)

    # MILP with pulp
    try:
        import pulp

        n = len(options)
        bank_keys = sorted({opt["BANK_KEY"] for opt in options})

        # Decision: z_i ∈ {0,1} option selection (exactly one per demand line)
        prob = pulp.LpProblem("BNG_MinCost_2Banks_OnePerLine", pulp.LpMinimize)
        z = [pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]
        y = {b: pulp.LpVariable(f"y_{norm_name(b)}", lowBound=0, upBound=1, cat="Binary") for b in bank_keys}

        # Units per line (constants)
        demand_units = {di: float(demand_df.loc[di, "units_required"]) for di in demand_df.index}

        # Cost term
        cost = pulp.lpSum([options[i]["unit_price"] * demand_units[options[i]["demand_idx"]] * z[i] for i in range(n)])

        # Tie-break (tiny): prefer options from banks with more total residual stock (sum of caps of that bank)
        bank_total_cap = {b: 0.0 for b in bank_keys}
        for sid, cap in stock_caps.items():
            b = stock_bankkey.get(sid, "")
            if b in bank_total_cap:
                bank_total_cap[b] += cap
        eps = 1e-6
        tie_pref = -eps * pulp.lpSum([bank_total_cap[options[i]["BANK_KEY"]] * z[i] for i in range(n)])

        prob += cost + tie_pref

        # Each demand line: pick exactly one option
        for di in demand_df.index:
            prob += pulp.lpSum([z[i] for i in idx_by_dem.get(di, [])]) == 1

        # Bank usage linking and ≤ 2 banks total
        # If any option from bank b is chosen, y_b = 1
        # Also ensure bank-count limit
        for i in range(n):
            b = options[i]["BANK_KEY"]
            prob += z[i] <= y[b]
        prob += pulp.lpSum([y[b] for b in bank_keys]) <= 2

        # Stock capacity: for each stock_id, sum over chosen options of (coef * units_of_that_line) ≤ cap
        # Note: each chosen option for demand line di consumes coef * demand_units[di]
        use_map: Dict[str, List[int]] = {}
        for i, opt in enumerate(options):
            for sid in opt["stock_use"].keys():
                use_map.setdefault(sid, []).append(i)

        for sid, idxs in use_map.items():
            cap = float(stock_caps.get(sid, 0.0))
            prob += pulp.lpSum([options[i]["stock_use"][sid] * demand_units[options[i]["demand_idx"]] * z[i] for i in idxs]) <= cap + 1e-12

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        st.session_state["milp_status"] = pulp.LpStatus[prob.status]
        if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            raise RuntimeError(f"Optimiser infeasible ({pulp.LpStatus[prob.status]}).")

        # Extract solution
        chosen = [i for i in range(n) if z[i].value() and z[i].value() > 0.5]
        rows, total_cost = [], 0.0
        for i in chosen:
            opt = options[i]
            qty = float(demand_units[opt["demand_idx"]])
            rows.append({
                "demand_habitat": opt["demand_habitat"],
                "BANK_KEY": opt["BANK_KEY"],
                "bank_name": sstr(opt.get("bank_name","")),
                "bank_id": sstr(opt.get("bank_id","")),
                "supply_habitat": opt["supply_habitat"],
                "allocation_type": opt["type"],
                "tier": opt["tier"],
                "units_supplied": qty,
                "unit_price": opt["unit_price"],
                "cost": qty * opt["unit_price"],
            })
            total_cost += qty * opt["unit_price"]

        alloc_df = pd.DataFrame(rows).sort_values(["bank_name","demand_habitat"]).reset_index(drop=True)

        # Diagnostics: cheapest-by-proximity and skipped-cheaper reasons
        diag_rows = []
        # Compute the set of banks used (to derive "exceeds 2 banks" reason later if needed)
        used_banks = sorted(alloc_df["BANK_KEY"].unique().tolist())

        for di, drow in demand_df.iterrows():
            d_hab = sstr(drow["habitat_name"])
            d_units = float(drow["units_required"])
            opts_i = [options[i] for i in idx_by_dem.get(di, [])]

            # Cheapest by proximity
            for prox in ["local","adjacent","far"]:
                prox_opts = [o for o in opts_i if o["proximity"] == prox]
                # Only feasible options that could cover the line with current caps
                feas = []
                for o in prox_opts:
                    max_take = float('inf')
                    for sid, coef in o["stock_use"].items():
                        cap = float(stock_caps.get(sid, 0.0))
                        # Deduct usage by already-chosen options on OTHER lines? (strict diagnostic uses original caps)
                        # We keep original caps here to show what the optimiser saw pre-selection.
                        if coef > 0:
                            max_take = min(max_take, cap / coef)
                    if max_take + 1e-9 >= d_units:
                        feas.append(o)
                if feas:
                    best = sorted(feas, key=lambda o: (o["unit_price"], -sum(stock_caps.get(sid,0.0) for sid in o["stock_use"].keys())))[0]
                    diag_rows.append({
                        "demand_idx": di, "demand_habitat": d_hab, "proximity": prox,
                        "bank_name": sstr(best["bank_name"]), "BANK_KEY": best["BANK_KEY"],
                        "supply_habitat": best["supply_habitat"], "allocation_type": best["type"],
                        "unit_price": best["unit_price"]
                    })

        diag_df = pd.DataFrame(diag_rows).sort_values(["demand_idx","proximity","unit_price"])

        # Explain why cheaper options not chosen
        explain_rows = []
        # Recompute per-line chosen option map
        chosen_by_line = {}
        for _, r in alloc_df.iterrows():
            # map demand line by name & units; we can map by sequence order (index) instead
            chosen_by_line.setdefault(r["demand_habitat"], []).append(r)

        for di, drow in demand_df.iterrows():
            d_hab = sstr(drow["habitat_name"])
            d_units = float(drow["units_required"])
            chosen_row = alloc_df[alloc_df["demand_habitat"] == d_hab].head(1)
            if chosen_row.empty:
                continue
            chosen_cost = float(chosen_row["unit_price"].iloc[0])
            chosen_bank = sstr(chosen_row["BANK_KEY"].iloc[0])

            # Any strictly cheaper feasible option that was skipped?
            for o in [options[i] for i in idx_by_dem.get(di, [])]:
                # Must be able to cover full line
                ok_cap = True
                for sid, coef in o["stock_use"].items():
                    if coef <= 0: continue
                    if float(stock_caps.get(sid,0.0)) + 1e-9 < coef * d_units:
                        ok_cap = False; break
                if not ok_cap:
                    explain_rows.append({
                        "demand_habitat": d_hab, "skipped_bank": o["BANK_KEY"], "reason": "Insufficient stock for full line",
                        "skipped_unit_price": o["unit_price"]
                    })
                    continue
                # Bank-count if we switched: would set of banks exceed 2?
                hypothetical_banks = set(used_banks)
                hypothetical_banks.discard(chosen_bank)
                hypothetical_banks.add(o["BANK_KEY"])
                if len(hypothetical_banks) > 2:
                    explain_rows.append({
                        "demand_habitat": d_hab, "skipped_bank": o["BANK_KEY"], "reason": "Would exceed 2-bank limit",
                        "skipped_unit_price": o["unit_price"]
                    })
                    continue
                # If strictly cheaper but not chosen: flag that it was dominated by current global optimum (because another line's choice constrained bank-count)
                if o["unit_price"] + 1e-9 < chosen_cost:
                    explain_rows.append({
                        "demand_habitat": d_hab, "skipped_bank": o["BANK_KEY"], "reason": "Cheaper individually, but global 2-bank optimum chose differently",
                        "skipped_unit_price": o["unit_price"]
                    })

        explain_df = pd.DataFrame(explain_rows).drop_duplicates()

        return alloc_df, float(total_cost), chosen_size, (diag_df, explain_df)

    except Exception as e:
        raise RuntimeError(f"Optimiser error (MILP): {e}")

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
with st.expander("🔎 Diagnostics (candidate options & stock)", expanded=False):
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

            opts, _, _ = prepare_options(
                dd, chosen_size,
                sstr(target_lpa_name), sstr(target_nca_name),
                [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
                lpa_neighbors_norm, nca_neighbors_norm
            )
            if not opts:
                st.error("No candidate options (exact pricing only).")
            else:
                cand_df = pd.DataFrame(opts).rename(columns={"type": "allocation_type"})
                st.write("**Candidate options (by type & tier):**")
                grouped = (
                    cand_df.groupby(["demand_habitat","allocation_type","tier"], as_index=False)
                           .agg(options=("tier","count"),
                                min_price=("unit_price","min"),
                                max_price=("unit_price","max"))
                           .sort_values(["demand_habitat","allocation_type","tier"])
                )
                st.dataframe(grouped, use_container_width=True, hide_index=True)

                st.caption("Note: prices shown here are **exact Excel rows** only (no proxies).")
    except Exception as de:
        st.error(f"Diagnostics error: {de}")

# ========= Price readout (exact rows only) =========
def _pricing_for_size(chosen_size: str) -> pd.DataFrame:
    pr = backend["Pricing"].copy()
    pr = pr[(pr["contract_size"] == chosen_size)]
    pr = pr[~pr["habitat_name"].map(is_hedgerow)].copy()
    pr["price"] = pd.to_numeric(pr["price"], errors="coerce")
    pr = pr.merge(backend["Banks"][["bank_id","bank_name"]].drop_duplicates(), on="bank_id", how="left")
    # Show only rows that are actually priceable (exact rows), which is all of them here
    cols = ["BANK_KEY","bank_name","bank_id","contract_size","tier","habitat_name","price"]
    for c in cols:
        if c not in pr.columns: pr[c] = ""
    return pr[cols].sort_values(["BANK_KEY","tier","habitat_name","price"], kind="stable")

with st.expander("🧾 Price readout (exact rows used by optimiser)", expanded=False):
    try:
        present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
        total_units = float(demand_df["units_required"].sum()) if not demand_df.empty else 0.0
        chosen_size = select_contract_size(total_units, present_sizes)
        st.write(f"**Chosen contract size:** `{chosen_size}` (present sizes: {present_sizes})")
        prn = _pricing_for_size(chosen_size)
        if prn.empty:
            st.error("No pricing rows found for the chosen contract size.")
        else:
            st.dataframe(prn, use_container_width=True, hide_index=True)
            csv_bytes = prn.to_csv(index=False).encode("utf-8")
            st.download_button("Download pricing (exact rows, this size) CSV",
                               data=csv_bytes, file_name=f"pricing_exact_{chosen_size}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Price readout error: {e}")

# ========= Run optimiser =========
if run:
    try:
        if demand_df.empty:
            st.error("Add at least one demand row before optimising.")
            st.stop()

        # Validate demand names — allow special Net Gain label
        cat_names_run = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        unknown = [h for h in demand_df["habitat_name"] if h not in cat_names_run and h != NET_GAIN_LABEL]
        if unknown:
            st.error(f"These demand habitats aren’t in the catalog: {unknown}")
            st.stop()

        # Auto-locate if user typed but forgot 'Locate'
        if not sstr(target_lpa_name) or not sstr(target_nca_name):
            if sstr(postcode) or sstr(address):
                try:
                    (_t_lpa, _t_nca, _lpaN, _ncaN, _lpaNn, _ncaNn,
                     _lat, _lon, _lpa_gj, _nca_gj) = find_site(postcode, address)
                except Exception as e:
                    st.warning(f"Auto-locate failed: {e}. Proceeding with 'far' tiers only.")

        target_lpa = sstr(st.session_state.get("target_lpa_name", target_lpa_name))
        target_nca = sstr(st.session_state.get("target_nca_name", target_nca_name))
        lpa_neighbors = st.session_state.get("lpa_neighbors", lpa_neighbors)
        nca_neighbors = st.session_state.get("nca_neighbors", nca_neighbors)
        lpa_neighbors_norm = st.session_state.get("lpa_neighbors_norm", lpa_neighbors_norm)
        nca_neighbors_norm = st.session_state.get("nca_neighbors_norm", nca_neighbors_norm)
        target_lat = st.session_state.get("target_lat", target_lat)
        target_lon = st.session_state.get("target_lon", target_lon)
        lpa_geojson = st.session_state.get("lpa_geojson", lpa_geojson)
        nca_geojson = st.session_state.get("nca_geojson", nca_geojson)

        alloc_df, total_cost, size, diag = optimise(
            demand_df,
            target_lpa, target_nca,
            [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
            lpa_neighbors_norm, nca_neighbors_norm
        )
        diag_df, explain_df = diag

        st.success(f"Optimisation complete. Contract size = **{size}**. Total cost: **£{total_cost:,.0f}**")

        st.markdown("#### Allocation detail")
        st.dataframe(alloc_df, use_container_width=True)

        st.markdown("#### By bank")
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

        # ------- Diagnostics: cheapest-by-proximity; skipped-cheaper reasons -------
        with st.expander("🧭 Cheapest-by-proximity (per demand line)", expanded=True):
            if diag_df is not None and not diag_df.empty:
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
            else:
                st.info("No proximity candidates were feasible with full line coverage.")

        with st.expander("❓ Why cheaper options weren’t chosen", expanded=True):
            if explain_df is not None and not explain_df.empty:
                st.dataframe(explain_df.sort_values(["demand_habitat","skipped_unit_price"]),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No cheaper feasible options were skipped under the ≤2-bank constraint.")

        # ------- Map overlay: chosen banks and links -------
        if (target_lat is not None) and (target_lon is not None):
            fmap = render_base_map()

            # Bank coordinates (geocode if needed)
            bank_coords: Dict[str, Tuple[float,float]] = {}
            for _, b in backend["Banks"].iterrows():
                bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
                loc = bank_row_to_latlon(b)
                if loc: bank_coords[bkey] = (loc[0], loc[1])

            if not alloc_df.empty:
                grouped = alloc_df.groupby(["BANK_KEY","bank_name"], dropna=False)
                for (bkey, bname), g in grouped:
                    try:
                        latlon = bank_coords.get(sstr(bkey))
                        if not latlon: continue
                        lat_b, lon_b = latlon

                        # Draw the bank's catchments (LPA/NCA) as perimeters
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
                            name=f"{sstr(bname) or sstr(bkey)} — Bank LPA: {sstr(bgeo.get('lpa_name')) or 'Unknown'}",
                            color="green", weight=2, fill_opacity=0.03
                        )
                        add_geojson_layer(
                            fmap, bgeo.get("nca_gj"),
                            name=f"{sstr(bname) or sstr(bkey)} — Bank NCA: {sstr(bgeo.get('nca_name')) or 'Unknown'}",
                            color="blue", weight=3, fill_opacity=0.03
                        )

                        # Marker + route
                        popup_lines = []
                        total_units_b = g["units_supplied"].sum()
                        total_cost_b = g["cost"].sum()
                        popup_lines.append(f"<b>Bank:</b> {sstr(bname) or sstr(bkey)}")
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
                        folium.PolyLine(
                            locations=[[target_lat, target_lon], [lat_b, lon_b]],
                            weight=2, opacity=0.8, dash_array="6,6", color="blue",
                            tooltip=f"Supply route: target → {sstr(bname) or sstr(bkey)}"
                        ).add_to(fmap)
                    except Exception as map_e:
                        st.caption(f"Skipped map overlay for bank {sstr(bname) or sstr(bkey)}: {map_e}")

            st.markdown("### Map (with selected supply)")
            st_folium(fmap, height=520, returned_objects=[], use_container_width=True)

        # Downloads
        def df_to_csv_bytes(df):
            buf = BytesIO(); buf.write(df.to_csv(index=False).encode("utf-8")); buf.seek(0); return buf

        st.download_button("Download allocation (CSV)", data=df_to_csv_bytes(alloc_df),
                           file_name="allocation.csv", mime="text/csv")
        st.download_button("Download by bank (CSV)", data=df_to_csv_bytes(by_bank),
                           file_name="allocation_by_bank.csv", mime="text/csv")
        st.download_button("Download by habitat (CSV)", data=df_to_csv_bytes(by_hab),
                           file_name="allocation_by_habitat.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Optimiser error: {e}")



















