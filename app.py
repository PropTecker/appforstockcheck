# app.py — BNG Optimiser (Standalone), v8
# New in v8:
# - Pricing column normalisation: reads price from 'Price' / 'price' / 'Unit Price' etc. and coerces to float.
# - Bank identity: use BANK KEY = bank_name (if present) else bank_id.
#   This ensures distinct banks under the same bank_id are treated separately in pricing, stock, and optimisation.
# - Keeps v7 rules: Medium => same group ≥ Medium else ANY group with higher distinctiveness; Low => any; High/Very High => like-for-like.
# - Everything else unchanged: auto-locate, ≤2 banks, local/adjacent-first, Net Gain line, diagnostics, proximity audit, map overlay.

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

# ========= Login =========
DEFAULT_USER = "WC0323"
DEFAULT_PASS = "Wimbourne"

def require_login():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        with st.sidebar:
            if st.button("Log out"):
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

# ========= Normalise BANK KEY (bank_name preferred) =========
# Add BANK_KEY to Banks, Pricing, Stock for consistent identity
def make_bank_key_col(df: pd.DataFrame, banks_df: pd.DataFrame, bank_id_col: str = "bank_id") -> pd.DataFrame:
    out = df.copy()
    # Attach bank_name from Banks (if not already present)
    cols_to_merge = ["bank_id"]
    if "bank_name" in banks_df.columns:
        cols_to_merge.append("bank_name")
    m = banks_df[cols_to_merge].drop_duplicates()
    out = out.merge(m, on="bank_id", how="left")
    # BANK_KEY: prefer bank_name if present, else bank_id
    out["BANK_KEY"] = out["bank_name"].where(out["bank_name"].astype(str).str.strip().ne(""), out["bank_id"])
    out["BANK_KEY"] = out["BANK_KEY"].map(sstr)
    return out

backend["Banks"]["BANK_KEY"] = backend["Banks"]["bank_name"].where(
    backend["Banks"].get("bank_name", pd.Series(dtype=object)).astype(str).str.strip().ne(""),
    backend["Banks"]["bank_id"]
).map(sstr)

# ========= Apply quotes policy if WITH_STOCK columns exist =========
if "available_excl_quotes" in backend["Stock"].columns and "quoted" in backend["Stock"].columns:
    s = backend["Stock"].copy()
    if quotes_hold_policy == "Ignore quotes (default)":
        s["quantity_available"] = s["available_excl_quotes"]
    elif quotes_hold_policy == "Quotes hold 100%":
        s["quantity_available"] = (s["available_excl_quotes"] - s["quoted"]).clip(lower=0)
    else:  # 50%
        s["quantity_available"] = (s["available_excl_quotes"] - 0.5 * s["quoted"]).clip(lower=0)
    backend["Stock"] = s

# ========= Enrich Banks dynamically (LPA/NCA) =========
def bank_row_to_latlon(row: pd.Series) -> Optional[Tuple[float,float,str]]:
    # 1) lat/lon
    if "lat" in row and "lon" in row:
        try:
            lat = float(row["lat"]); lon = float(row["lon"])
            if np.isfinite(lat) and np.isfinite(lon):
                return lat, lon, f"ll:{lat:.6f},{lon:.6f}"
        except Exception:
            pass
    # 2) postcode
    if "postcode" in row and sstr(row["postcode"]):
        try:
            lat, lon, _ = get_postcode_info(sstr(row["postcode"]))
            return lat, lon, f"pc:{sstr(row['postcode']).upper().replace(' ','')}"
        except Exception:
            pass
    # 3) address
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
    cache = st.session_state.setdefault("bank_geo_cache", {})  # key -> (lpa,nca)
    needs = df[(df["lpa_name"].map(sstr) == "") | (df["nca_name"].map(sstr) == "")]
    prog = None
    if len(needs) > 0:
        prog = st.sidebar.progress(0.0, text="Resolving bank LPA/NCA…")
    rows, updated, total = [], 0, len(df)
    for i, row in df.iterrows():
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
                    time.sleep(0.15)  # courteous
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

# ========= Normalise Pricing columns (price, size, tier) =========
def normalise_pricing(pr_df: pd.DataFrame) -> pd.DataFrame:
    df = pr_df.copy()
    # Standardise common column names/cases
    # Create 'price' from possible variants
    price_cols = [c for c in df.columns if c.strip().lower() in ("price","unit price","unit_price","unitprice")]
    if not price_cols:
        st.error("Pricing sheet must contain a 'Price' column (or 'Unit Price').")
        st.stop()
    df["price"] = pd.to_numeric(df[price_cols[0]], errors="coerce")
    # Normalise tier & size to lowercase strings
    df["tier"] = df["tier"].astype(str).str.strip().str.lower()
    df["contract_size"] = df["contract_size"].astype(str).str.strip().str.lower()
    # Ensure bank_id string and attach BANK_KEY
    df["bank_id"] = df["bank_id"].astype(str).str.strip()
    df = make_bank_key_col(df, backend["Banks"])
    # Keep group/distinctiveness if present (may be blank)
    if "broader_type" not in df.columns:
        df["broader_type"] = ""
    if "distinctiveness_name" not in df.columns:
        df["distinctiveness_name"] = ""
    # Habitat name as string
    df["habitat_name"] = df["habitat_name"].astype(str).str.strip()
    return df

backend["Pricing"] = normalise_pricing(backend["Pricing"])

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
        run_locate = st.button("Locate")

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
    # Persist
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

# Map (base layers)
def render_base_map():
    fmap = folium.Map(location=[target_lat, target_lon], zoom_start=11, control_scale=True)
    add_geojson_layer(fmap, lpa_geojson, f"LPA: {target_lpa_name}" if target_lpa_name else "LPA", color="red", weight=2, fill_opacity=0.05)
    add_geojson_layer(fmap, nca_geojson, f"NCA: {target_nca_name}" if target_nca_name else "NCA", color="yellow", weight=3, fill_opacity=0.05)
    folium.CircleMarker([target_lat, target_lon], radius=6, color="red", fill=True, tooltip="Target site").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap

if (target_lat is not None) and (target_lon is not None):
    st.markdown("### Map")
    st_folium(render_base_map(), height=420, returned_objects=[], use_container_width=True)

# ========= Demand builder =========
st.subheader("2) Demand (units required)")

NET_GAIN_LABEL = "Net Gain (Low-equivalent)"    # special demand that trades like Low

def init_demand_state():
    if "demand_rows" not in st.session_state:
        st.session_state.demand_rows = [{"id": 1, "habitat_name": "", "units": 0.0}]
        st.session_state._next_row_id = 2

init_demand_state()

# choices from catalog plus the special Net Gain label
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
    cc1, cc2, cc3 = st.columns([0.33, 0.33, 0.34])
    with cc1:
        if st.button("➕ Add habitat"):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": HAB_CHOICES[0] if HAB_CHOICES else "", "units": 0.0}
            )
            st.session_state._next_row_id += 1
    with cc2:
        if st.button("➕ Net Gain (Low-equivalent)", help="Adds a 'Net Gain' line. Trades like Low distinctiveness (can source from any habitat)."):
            st.session_state.demand_rows.append(
                {"id": st.session_state._next_row_id, "habitat_name": NET_GAIN_LABEL, "units": 0.0}
            )
            st.session_state._next_row_id += 1
    with cc3:
        if st.button("🧹 Clear all"):
            init_demand_state(); st.rerun()

total_units = sum([float(r.get("units", 0.0) or 0.0) for r in st.session_state.demand_rows])
st.metric("Total units", f"{total_units:.2f}")

demand_df = pd.DataFrame(
    [{"habitat_name": sstr(r["habitat_name"]), "units_required": float(r.get("units", 0.0) or 0.0)}
     for r in st.session_state.demand_rows if sstr(r["habitat_name"]) and float(r.get("units", 0.0) or 0.0) > 0]
)

if not demand_df.empty:
    st.dataframe(demand_df, use_container_width=True, hide_index=True)
else:
    st.info("Add at least one habitat and units to continue.", icon="ℹ️")

# ========= Official rules =========
def enforce_catalog_rules_official(demand_row, supply_row, dist_levels_map_local, explicit_rule: bool) -> bool:
    if explicit_rule:  # TradingRules override
        return True
    dh = sstr(demand_row.get("habitat_name"))
    if dh == NET_GAIN_LABEL:
        return True  # Net Gain behaves like Low (anything)
    sh = sstr(supply_row.get("habitat_name"))
    d_group = sstr(demand_row.get("broader_type"))
    s_group = sstr(supply_row.get("broader_type"))
    d_dist_name = sstr(demand_row.get("distinctiveness_name"))
    s_dist_name = sstr(supply_row.get("distinctiveness_name"))
    d_key = d_dist_name.lower()
    d_val = dist_levels_map_local.get(d_dist_name, dist_levels_map_local.get(d_key, -1e9))
    s_val = dist_levels_map_local.get(s_dist_name, dist_levels_map_local.get(s_dist_name.lower(), -1e-9))
    if d_key == "low":
        return True
    if d_key == "medium":
        same_group = (d_group and s_group and d_group == s_group)
        higher_distinctiveness = (s_val > d_val)
        return bool(same_group or higher_distinctiveness)
    if d_key in ("high", "very high", "very_high", "very-high"):
        return sh == dh
    # Unknown → behave like Medium
    same_group = (d_group and s_group and d_group == s_group)
    higher_distinctiveness = (s_val > d_val)
    return bool(same_group or higher_distinctiveness)

# ========= Options builder (with proxy pricing; no SRM math) =========
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
        (Stock, ["habitat_name","stock_id","bank_id","quantity_available"]),
        (Pricing, ["habitat_name","contract_size","tier","bank_id","BANK_KEY","price","broader_type","distinctiveness_name"]),
        (Trading, ["demand_habitat","allowed_supply_habitat","min_distinctiveness_name","companion_habitat"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].map(sstr)

    # Attach BANK_KEY to Stock
    Stock = make_bank_key_col(Stock, Banks)

    stock_full = Stock.merge(Banks[["bank_id","BANK_KEY","bank_name","lpa_name","nca_name"]], on="bank_id", how="left") \
                      .merge(Catalog, on="habitat_name", how="left")

    # Contract size filter (lowercase)
    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()

    # ---- Build pricing_enriched that always has group & distinctiveness ----
    Catalog_cols = ["habitat_name", "broader_type", "distinctiveness_name"]
    pc_join = pricing_cs.merge(
        Catalog[Catalog_cols], on="habitat_name", how="left", suffixes=("", "_cat")
    )
    pc_join["broader_type_eff"] = pc_join["broader_type"].where(
        pc_join["broader_type"].astype(str).str.len() > 0, pc_join["broader_type_cat"]
    )
    pc_join["distinctiveness_name_eff"] = pc_join["distinctiveness_name"].where(
        pc_join["distinctiveness_name"].astype(str).str.len() > 0,
        pc_join["distinctiveness_name_cat"]
    )
    for c in ["broader_type_eff", "distinctiveness_name_eff", "tier", "bank_id", "habitat_name", "BANK_KEY"]:
        if c in pc_join.columns:
            pc_join[c] = pc_join[c].map(sstr)
    pricing_enriched = pc_join  # use from here on

    # Trading index (still supported)
    trade_idx = {}
    if not Trading.empty:
        for _, r in Trading.iterrows():
            trade_idx.setdefault(sstr(r["demand_habitat"]), []).append({
                "supply_habitat": sstr(r["allowed_supply_habitat"]),
                "min_distinctiveness_name": sstr(r.get("min_distinctiveness_name")),
            })

    # helper to find names for pairing
    def find_catalog_name(substr: str) -> Optional[str]:
        m = Catalog[Catalog["habitat_name"].str.contains(substr, case=False, na=False)]
        return sstr(m["habitat_name"].iloc[0]) if not m.empty else None

    ORCHARD_NAME = find_catalog_name("Traditional Orchard")
    SCRUB_NAME = find_catalog_name("Mixed Scrub")
    if not SCRUB_NAME:
        SCRUB_NAME = find_catalog_name("scrub")
    if not SCRUB_NAME:
        SCRUB_NAME = find_catalog_name("bramble")

    # Determine once per site: any local/adjacent bank?
    _adj_key = f"adj_flag::{norm_name(target_lpa)}::{norm_name(target_nca)}"
    if _adj_key not in st.session_state:
        any_local_adj = False
        for _, b in Banks.iterrows():
            t = tier_for_bank(
                sstr(b.get("lpa_name")), sstr(b.get("nca_name")),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
            )
            if t in ("local", "adjacent"):
                any_local_adj = True
                break
        st.session_state[_adj_key] = any_local_adj
    site_has_local_adjacent = st.session_state[_adj_key]

    options: List[dict] = []
    stock_caps: Dict[str, float] = {}
    stock_bankkey: Dict[str, str] = {}
    for _, s in Stock.iterrows():
        stock_caps[sstr(s["stock_id"])] = float(s.get("quantity_available", 0) or 0.0)
        stock_bankkey[sstr(s["stock_id"])] = sstr(s.get("BANK_KEY") or s.get("bank_id"))

    def dval(name: Optional[str]) -> float:
        key = sstr(name)
        return dist_levels_map.get(key, dist_levels_map.get(key.lower(), -1e9))

    def find_price_for_supply(bank_key: str,
                              supply_habitat: str,
                              tier: str,
                              demand_broader: str,
                              demand_dist: str) -> Optional[Tuple[float, str, str]]:
        """
        Returns (price, source_kind, price_habitat) or None.
        source_kind: "exact" (exact habitat row) | "group-proxy" (group/distinctiveness row)
        price_habitat: the habitat_name used to price (may be blank or different when proxying)
        Uses: pricing_enriched, dist_levels_map, BANK_KEY.
        """
        # 0) Exact habitat price row (BANK_KEY + tier + size)
        pr_exact = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                                    (pricing_enriched["tier"] == tier) &
                                    (pricing_enriched["habitat_name"] == supply_habitat)]
        if not pr_exact.empty:
            r = pr_exact.sort_values("price").iloc[0]
            return float(r["price"]), "exact", sstr(r["habitat_name"])

        d_key = sstr(demand_dist).lower()
        d_num = dval(demand_dist)

        # All rows at this BANK_KEY + tier + size that have effective group & distinctiveness
        grp = pricing_enriched[(pricing_enriched["BANK_KEY"] == bank_key) &
                               (pricing_enriched["tier"] == tier)]
        grp = grp[(grp["broader_type_eff"].astype(str).str.len() > 0) &
                  (grp["distinctiveness_name_eff"].astype(str).str.len() > 0)]
        if grp.empty:
            return None

        if d_key == "low":
            # Low → any priced row (cheapest)
            r = grp.sort_values("price").iloc[0]
            return float(r["price"]), "group-proxy", sstr(r["habitat_name"])

        if d_key == "medium":
            # 2a) Preferred: same group with distinctiveness >= Medium
            grp_same = grp[grp["broader_type_eff"].map(sstr) == sstr(demand_broader)].copy()
            if not grp_same.empty:
                grp_same["_dval"] = grp_same["distinctiveness_name_eff"].map(dval)
                grp_same = grp_same[grp_same["_dval"] >= d_num]
                if not grp_same.empty:
                    r = grp_same.sort_values("price").iloc[0]
                    return float(r["price"]), "group-proxy", sstr(r["habitat_name"])

            # 2b) ANY group with distinctiveness strictly higher than Medium
            grp_any_higher = grp.assign(_dval=grp["distinctiveness_name_eff"].map(dval))
            grp_any_higher = grp_any_higher[grp_any_higher["_dval"] > d_num]
            if not grp_any_higher.empty:
                r = grp_any_higher.sort_values("price").iloc[0]
                return float(r["price"]), "group-proxy", sstr(r["habitat_name"])

            return None  # no compliant proxy at this bank/tier/size

        # High / Very High → like-for-like only (must have exact price row)
        return None

    for di, drow in demand_df.iterrows():
        dem_hab = sstr(drow["habitat_name"])
        required = float(drow["units_required"])

        # Special Net Gain behaves like Low distinctiveness
        if dem_hab == NET_GAIN_LABEL:
            d_broader = ""
            d_dist = "Low"
        else:
            dcat = Catalog[Catalog["habitat_name"] == dem_hab]
            d_broader = sstr(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
            d_dist = sstr(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""

        # Candidate stock via explicit Trading or official rules
        cand_parts = []
        # Join for convenience
        explicit_list = []
        if not Trading.empty and dem_hab in set(Trading["demand_habitat"].astype(str)):
            for _, rule in Trading[Trading["demand_habitat"] == dem_hab].iterrows():
                sh = sstr(rule["allowed_supply_habitat"])
                s_min = sstr(rule.get("min_distinctiveness_name"))
                df_s = stock_full[stock_full["habitat_name"] == sh].copy()
                if s_min:
                    df_s = df_s[df_s["distinctiveness_name"].map(lambda x: dval(x)) >= dval(s_min)]
                if not df_s.empty: cand_parts.append(df_s)
            explicit_list = cand_parts[:]

        if not cand_parts:
            key = d_dist.lower()
            if key == "low" or dem_hab == NET_GAIN_LABEL:
                df_s = stock_full.copy()
            elif key == "medium":
                same_group = stock_full["broader_type"].fillna("").astype(str).map(sstr).eq(d_broader)
                higher_dist = stock_full["distinctiveness_name"].map(lambda x: dval(x)) > dval(d_dist)
                df_s = stock_full[same_group | higher_dist].copy()
            else:
                df_s = stock_full[stock_full["habitat_name"] == dem_hab].copy()
            if not df_s.empty: cand_parts.append(df_s)

        if not cand_parts:
            continue

        candidates = pd.concat(cand_parts, ignore_index=True)

        # Normal (direct) options — with proxy pricing if needed
        for _, s in candidates.iterrows():
            explicit = bool(explicit_list)
            if not enforce_catalog_rules_official(
                pd.Series({"habitat_name": dem_hab, "broader_type": d_broader, "distinctiveness_name": d_dist}),
                s, dist_levels_map, explicit_rule=explicit
            ):
                continue
            tier = tier_for_bank(
                s.get("lpa_name",""), s.get("nca_name",""),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
            )

            bank_key = sstr(s.get("BANK_KEY") or s.get("bank_name") or s.get("bank_id"))

            price_info = find_price_for_supply(
                bank_key=bank_key,
                supply_habitat=s["habitat_name"],
                tier=tier,
                demand_broader=d_broader,
                demand_dist=d_dist,
            )
            if not price_info:
                continue
            unit_price, price_source, price_hab_used = price_info

            cap = float(s.get("quantity_available", 0) or 0.0)
            if cap <= 0:
                continue
            options.append({
                "type": "normal",
                "demand_idx": di,
                "demand_habitat": dem_hab,
                "BANK_KEY": bank_key,
                "bank_name": sstr(s.get("bank_name")),
                "bank_id": sstr(s.get("bank_id")),
                "supply_habitat": s["habitat_name"],
                "tier": tier,
                "proximity": tier,  # local/adjacent/far
                "unit_price": float(unit_price),
                "stock_use": {sstr(s["stock_id"]): 1.0},
                "price_source": price_source,
                "price_habitat": price_hab_used,
            })

        # Medium fallback: paired Orchard + Scrub only when site is far from all banks
        if d_dist.lower() == "medium" and not site_has_local_adjacent and ORCHARD_NAME and SCRUB_NAME:
            banks = stock_full["BANK_KEY"].dropna().unique().tolist()
            for bk in banks:
                orch_rows = stock_full[(stock_full["BANK_KEY"] == bk) & (stock_full["habitat_name"] == ORCHARD_NAME)]
                scrub_rows = stock_full[(stock_full["BANK_KEY"] == bk) & (
                    (stock_full["habitat_name"] == SCRUB_NAME) |
                    (stock_full["habitat_name"].str.contains("scrub", case=False, na=False)) |
                    (stock_full["habitat_name"].str.contains("bramble", case=False, na=False))
                )]
                if orch_rows.empty or scrub_rows.empty:
                    continue
                for _, o in orch_rows.iterrows():
                    for _, s in scrub_rows.iterrows():
                        tier_b = tier_for_bank(
                            sstr(s.get("lpa_name")), sstr(s.get("nca_name")),
                            target_lpa, target_nca, lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
                        )
                        # Price each half via demand-based proxy finder
                        pi_o = find_price_for_supply(bk, ORCHARD_NAME, tier_b, d_broader, d_dist)
                        pi_s = find_price_for_supply(bk, s["habitat_name"], tier_b, d_broader, d_dist)
                        if not pi_o or not pi_s:
                            continue
                        cap_o = float(o.get("quantity_available", 0) or 0.0)
                        cap_s = float(s.get("quantity_available", 0) or 0.0)
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
                            "supply_habitat": f"{ORCHARD_NAME} + {sstr(s['habitat_name'])}",
                            "tier": tier_b,
                            "proximity": tier_b,
                            "unit_price": price,
                            "stock_use": {sstr(o["stock_id"]): 0.5, sstr(s["stock_id"]): 0.5},
                            "price_source": "group-proxy",
                            "price_habitat": f"{pi_o[2]} + {pi_s[2]}",
                        })

    return options, stock_caps, stock_bankkey

# ========= Optimiser =========
def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str],
             lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[pd.DataFrame, float, str]:
    chosen_size = select_size_for_demand(demand_df, backend["Pricing"])
    options, stock_caps, stock_bankkey = prepare_options(
        demand_df, chosen_size, target_lpa, target_nca,
        lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm
    )
    if not options:
        raise RuntimeError("No feasible options. Check prices/stock/rules or location tiers.")

    # index options by demand
    idx_by_dem: Dict[int, List[int]] = {}
    for i, opt in enumerate(options):
        idx_by_dem.setdefault(opt["demand_idx"], []).append(i)

    # ---- MILP (prefer 1 bank; ≤2 max); prioritise local/adjacent over far; prefer direct over paired on ties
    if _HAS_PULP:
        def solve_with_bank_limit(max_banks: int) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
            prob = pulp.LpProblem("BNG_Allocation_MILP", pulp.LpMinimize)
            x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(len(options))]
            bank_keys = sorted({opt["BANK_KEY"] for opt in options})
            y = {b: pulp.LpVariable(f"y_{norm_name(b)}", lowBound=0, upBound=1, cat="Binary") for b in bank_keys}

            # objective: BIG * FAR usage + cost + tiny penalties (fewer banks, prefer direct)
            total_units_req = float(demand_df["units_required"].sum())
            max_price = max([o["unit_price"] for o in options]) if options else 1.0
            BIG = max(1.0, max_price) * max(1.0, total_units_req) * 1000.0

            far_usage = pulp.lpSum([x[i] for i in range(len(options)) if options[i]["proximity"] == "far"])
            cost_term = pulp.lpSum([options[i]["unit_price"] * x[i] for i in range(len(options))])
            eps_bank   = 1e-4
            eps_paired = 1e-6
            bank_pen  = eps_bank * pulp.lpSum([y[b] for b in bank_keys])
            pair_pen  = eps_paired * pulp.lpSum([x[i] for i in range(len(options)) if options[i]["type"] == "paired"])

            prob += BIG * far_usage + cost_term + bank_pen + pair_pen

            # link x to bank y with big-M per BANK_KEY
            M_bank: Dict[str, float] = {b: 0.0 for b in bank_keys}
            for sid, cap in stock_caps.items():
                bkey = stock_bankkey.get(sid, "")
                if bkey in M_bank:
                    M_bank[bkey] += cap
            for i, opt in enumerate(options):
                prob += x[i] <= M_bank[opt["BANK_KEY"]] * y[opt["BANK_KEY"]], f"link_{i}"

            # bank limit
            prob += pulp.lpSum([y[b] for b in bank_keys]) <= max_banks, "bank_limit"

            # demand constraints
            for di, drow in demand_df.iterrows():
                prob += pulp.lpSum([x[i] for i in idx_by_dem.get(di, [])]) >= float(drow["units_required"]), f"demand_{di}"

            # stock caps
            use_map: Dict[str, List[Tuple[int,float]]] = {}
            for i, opt in enumerate(options):
                for sid, coef in opt["stock_use"].items():
                    use_map.setdefault(sid, []).append((i, float(coef)))
            for sid, pairs in use_map.items():
                cap = float(stock_caps.get(sid, 0.0))
                if cap <= 0:
                    continue
                prob += pulp.lpSum([coef * x[i] for (i, coef) in pairs]) <= cap, f"cap_{sid}"

            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            status = pulp.LpStatus[prob.status]
            if status not in ("Optimal", "Feasible"):
                return None, None

            rows, total_cost = [], 0.0
            for i, var in enumerate(x):
                qty = var.value() or 0.0
                if qty <= 1e-9: continue
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

        def _is_bad(solution_tuple):
            df, cost = solution_tuple
            if df is None: return True
            if isinstance(df, pd.DataFrame) and df.empty: return True
            return False

        res = solve_with_bank_limit(1)
        if _is_bad(res):
            res = solve_with_bank_limit(2)
            if _is_bad(res):
                raise RuntimeError("Infeasible even with two banks.")
        alloc_df, total_cost = res
        return alloc_df, total_cost, chosen_size

    # ---- Greedy fallback (prefer 1 bank; else best 2). Prioritise non-far first; paired deprioritised on ties
    from collections import defaultdict

    options_by_bank: Dict[str, List[int]] = defaultdict(list)
    for i, opt in enumerate(options):
        options_by_bank[opt["BANK_KEY"]].append(i)

    def greedy_for_banks(allowed_banks: List[str]) -> Tuple[bool, pd.DataFrame, float]:
        caps = stock_caps.copy()
        rows = []

        for di, drow in demand_df.iterrows():
            need = float(drow["units_required"])
            cand = [i for i, opt in enumerate(options) if opt["demand_idx"] == di and opt["BANK_KEY"] in allowed_banks]
            # sort by proximity first (local/adjacent before far), then price, then prefer direct on ties
            cand.sort(key=lambda i: (
                1 if options[i]["proximity"] == "far" else 0,
                options[i]["unit_price"],
                1 if options[i]["type"] == "paired" else 0
            ))
            for i in cand:
                if need <= 1e-9: break
                opt = options[i]
                # max take limited by all stocks in this option
                max_take = float('inf')
                for sid, coef in opt["stock_use"].items():
                    if coef <= 0: continue
                    max_take = min(max_take, caps.get(sid, 0.0) / coef if coef > 0 else float('inf'))
                if max_take <= 1e-9:
                    continue
                take = min(max_take, need)
                for sid, coef in opt["stock_use"].items():
                    caps[sid] = caps.get(sid, 0.0) - coef * take
                rows.append({
                    "demand_habitat": opt["demand_habitat"],
                    "BANK_KEY": opt["BANK_KEY"],
                    "bank_name": opt.get("bank_name",""),
                    "bank_id": opt.get("bank_id",""),
                    "supply_habitat": opt["supply_habitat"],
                    "allocation_type": opt["type"],
                    "tier": opt["tier"],
                    "units_supplied": take,
                    "unit_price": opt["unit_price"],
                    "cost": take * opt["unit_price"],
                    "price_source": opt.get("price_source",""),
                    "price_habitat": opt.get("price_habitat",""),
                })
                need -= take
            if need > 1e-6:
                return False, pd.DataFrame(), 0.0
        df = pd.DataFrame(rows)
        return True, df, float(df["cost"].sum()) if not df.empty else 0.0

    # Rank banks: any with non-far options first (by their cheapest non-far price), then far-only banks (by cheapest price)
    banks_with_nonfar = []
    banks_far_only = []
    cheapest_nonfar = {}
    cheapest_any = {}

    for b, idxs in options_by_bank.items():
        prices_nonfar = [options[i]["unit_price"] for i in idxs if options[i]["proximity"] != "far"]
        prices_all    = [options[i]["unit_price"] for i in idxs]
        if prices_nonfar:
            banks_with_nonfar.append(b)
            cheapest_nonfar[b] = min(prices_nonfar)
        if prices_all:
            cheapest_any[b] = min(prices_all)
        if not prices_nonfar and prices_all:
            banks_far_only.append(b)

    banks_with_nonfar.sort(key=lambda bb: cheapest_nonfar[bb])
    banks_far_only.sort(key=lambda bb: cheapest_any[bb])
    bank_order_pref = banks_with_nonfar + banks_far_only

    # try 1 bank (prefer non-far capable)
    for b in bank_order_pref:
        ok, df, cost = greedy_for_banks([b])
        if ok:
            return df, cost, chosen_size

    # try best pairs (keep ≤2 banks)
    n = len(bank_order_pref)
    best_df, best_cost = None, float("inf")
    for i in range(n):
        for j in range(i+1, n):
            ok, df, cost = greedy_for_banks([bank_order_pref[i], bank_order_pref[j]])
            if ok and cost < best_cost:
                best_df, best_cost = df, cost
    if best_df is None:
        raise RuntimeError("Infeasible even with two banks in greedy fallback.")
    return best_df, best_cost, chosen_size

# ========= Run optimiser UI =========
st.subheader("3) Run optimiser")
left, right = st.columns([1,1])
with left:
    run = st.button("Optimise now", type="primary", disabled=demand_df.empty)
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
                    st.caption("Note: `price_source='group-proxy'` means we priced using group/distinctiveness rows.")
    except Exception as de:
        st.error(f"Diagnostics error: {de}")

# ========= Proximity Audit =========
with st.expander("🧭 Proximity audit (why a local/adjacent option wasn’t chosen)", expanded=False):
    try:
        if demand_df.empty:
            st.info("Add demand rows to see proximity audit.")
        else:
            dd = demand_df.copy()
            present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
            total_units = float(dd["units_required"].sum())
            chosen_size = select_contract_size(total_units, present_sizes)

            opts, _, _ = prepare_options(
                dd, chosen_size,
                sstr(target_lpa_name), sstr(target_nca_name),
                [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
                lpa_neighbors_norm, nca_neighbors_norm
            )
            if not opts:
                st.warning("No options to audit.")
            else:
                o = pd.DataFrame(opts)
                for dem in dd["habitat_name"].unique():
                    sub = o[o["demand_habitat"] == dem].copy()
                    if sub.empty:
                        st.write(f"**{dem}** — no candidates (check catalog/trading rules).")
                        continue
                    sub["approx_cap"] = sub["stock_use"].apply(lambda d: sum(d.values()) if isinstance(d, dict) else 0.0)
                    g = sub.groupby(["BANK_KEY","bank_name","proximity"], as_index=False).agg(
                        min_price=("unit_price","min"),
                        options=("unit_price","count"),
                        approx_cap=("approx_cap","sum")
                    ).sort_values(["proximity","min_price"])
                    st.write(f"**{dem}** — candidates by bank & proximity")
                    st.dataframe(g, use_container_width=True, hide_index=True)

                st.markdown("**Sanity checks**")
                banks_missing_geo = backend["Banks"][
                    (backend["Banks"]["lpa_name"].map(sstr) == "") | (backend["Banks"]["nca_name"].map(sstr) == "")
                ][["BANK_KEY","bank_name","lpa_name","nca_name"]]
                if not banks_missing_geo.empty:
                    st.warning("Some banks have no LPA/NCA (they will always look FAR):")
                    st.dataframe(banks_missing_geo, use_container_width=True, hide_index=True)

                pr = backend["Pricing"].copy()
                pr = pr[pr["contract_size"] == chosen_size]
                needed = pd.MultiIndex.from_product(
                    [backend["Stock"]["bank_id"].dropna().unique(),
                     demand_df["habitat_name"].unique(),
                     ["local","adjacent","far"]], names=["bank_id","habitat_name","tier"]
                ).to_frame(index=False)
                merged = needed.merge(pr[["bank_id","habitat_name","tier","price"]],
                                      on=["bank_id","habitat_name","tier"], how="left", indicator=True)
                missing_adj = merged[(merged["tier"] == "adjacent") & (merged["_merge"] == "left_only")]
                if not missing_adj.empty:
                    st.warning("Missing ADJACENT exact pricing rows for some bank+habitat — group-proxy may still price Medium/Low where trading rules allow.")
                    st.dataframe(missing_adj[["bank_id","habitat_name","tier"]], use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Audit error: {e}")

# ========= Pricing completeness (this contract size) =========
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
                st.warning("Some exact pricing rows are missing — that’s fine if they’re not tradeable, "
                           "or if Medium/Low can use group-proxy rules. Those exact rows will be ignored if untradeable.")
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

        # Auto-locate if user forgot to press Locate
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
        target_lat = st.session_state.get("target_lat", target_lat)
        target_lon = st.session_state.get("target_lon", target_lon)
        lpa_geojson = st.session_state.get("lpa_geojson", lpa_geojson)
        nca_geojson = st.session_state.get("nca_geojson", nca_geojson)

        alloc_df, total_cost, size = optimise(
            demand_df,
            target_lpa, target_nca,
            [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
            lpa_neighbors_norm, nca_neighbors_norm
        )

        st.success(f"Optimisation complete. Contract size = **{size}**. Total cost: **£{total_cost:,.0f}**")

        st.markdown("#### Allocation detail")
        st.dataframe(alloc_df, use_container_width=True)
        if "price_source" in alloc_df.columns:
            st.caption("Note: rows with `price_source='group-proxy'` were priced using group/distinctiveness rows in the Pricing sheet.")

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

        # ------- Map overlay: chosen banks and links -------
        if (target_lat is not None) and (target_lon is not None):
            fmap = render_base_map()

            # Build BANK_KEY -> (lat,lon) using Banks
            bank_coords: Dict[str, Tuple[float,float]] = {}
            banks_df = backend["Banks"].copy()

            # Try direct lat/lon; else geocode via postcode/address
            for _, b in banks_df.iterrows():
                bkey = sstr(b.get("BANK_KEY") or b.get("bank_name") or b.get("bank_id"))
                loc = bank_row_to_latlon(b)
                if loc:
                    bank_coords[bkey] = (loc[0], loc[1])

            # Summaries per BANK_KEY for popup
            if not alloc_df.empty:
                grouped = alloc_df.groupby(["BANK_KEY","bank_name"], dropna=False)
                for (bkey, bname), g in grouped:
                    latlon = bank_coords.get(sstr(bkey))
                    if not latlon:
                        continue
                    lat_b, lon_b = latlon
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

                    # dashed line target -> bank
                    folium.PolyLine(
                        locations=[[target_lat, target_lon], [lat_b, lon_b]],
                        weight=2, opacity=0.8, dash_array="6,6", color="blue",
                        tooltip=f"Supply route: target → {sstr(bname) or sstr(bkey)}"
                    ).add_to(fmap)

            st.markdown("### Map (with selected supply)")
            st_folium(fmap, height=520, returned_objects=[], use_container_width=True)

        # downloads
        def df_to_csv_bytes(df):
            buf = BytesIO()
            buf.write(df.to_csv(index=False).encode("utf-8"))
            buf.seek(0)
            return buf

        st.download_button("Download allocation (CSV)", data=df_to_csv_bytes(alloc_df),
                           file_name="allocation.csv", mime="text/csv")
        st.download_button("Download by bank (CSV)", data=df_to_csv_bytes(by_bank),
                           file_name="allocation_by_bank.csv", mime="text/csv")
        st.download_button("Download by habitat (CSV)", data=df_to_csv_bytes(by_hab),
                           file_name="allocation_by_habitat.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Optimiser error: {e}")














