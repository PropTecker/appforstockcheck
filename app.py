# app.py — BNG Optimiser (Standalone)
# - login wall (secrets-first fallback to WC0323 / Wimbourne)
# - dynamic Banks LPA/NCA enrichment (lat/lon -> postcode -> address)
# - POST helper to avoid 414
# - instant map on Locate (no need to click Optimise)
# - normalised tiering (local/adjacent/far)
# - official trading rules + broadened candidates for Low/Medium
# - LP (PuLP) with greedy fallback
# - diagnostics + tier breakdown

import json
import re
import time
from io import StringIO, BytesIO
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
st.caption("Upload backend workbook, locate target site, and optimise supply with SRM, official distinctiveness trading rules, and TradingRules.")

# ========= Safe strings =========
def sstr(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
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
POSTCODES_IO_REVERSE = "https://api.postcodes.io/postcodes"
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
    tiers = set(present)
    if "fractional" in tiers and total_units < 0.1: return "fractional"
    if "small" in tiers and total_units < 2.5: return "small"
    if "medium" in tiers and total_units < 15: return "medium"
    for t in ["large", "medium", "small", "fractional"]:
        if t in tiers: return t
    return present[0] if present else "small"

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

# Apply quotes policy if WITH_STOCK columns exist
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
    "Pricing": ["bank_id","habitat_name","contract_size","tier","price"],
    "Stock": ["bank_id","habitat_name","stock_id","quantity_available"],
    "HabitatCatalog": ["habitat_name","broader_type","distinctiveness_name"],
}.items():
    missing = [c for c in cols if c not in backend[sheet].columns]
    if missing:
        st.error(f"{sheet} is missing required columns: {missing}")
        st.stop()

# Distinctiveness mapping
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

# Map
if (target_lat is not None) and (target_lon is not None):
    fmap = folium.Map(location=[target_lat, target_lon], zoom_start=11, control_scale=True)
    add_geojson_layer(fmap, lpa_geojson, f"LPA: {target_lpa_name}" if target_lpa_name else "LPA", color="red", weight=2, fill_opacity=0.05)
    add_geojson_layer(fmap, nca_geojson, f"NCA: {target_nca_name}" if target_nca_name else "NCA", color="yellow", weight=3, fill_opacity=0.05)
    folium.CircleMarker([target_lat, target_lon], radius=5, color="red", fill=True, tooltip="Target").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    st.markdown("### Map")
    st_folium(fmap, height=420, returned_objects=[], use_container_width=True)

# ========= Demand =========
st.subheader("2) Demand (units required)")
default_demand = "Individual trees - Rural tree,0.08\nGrassland - Other neutral grassland,0.3"
demand_csv = st.text_area("CSV: habitat_name,units_required", value=default_demand, height=120)

HAB_ALIAS = {
    "Urban tree": "Individual trees - Urban tree",
    "Urban trees": "Individual trees - Urban tree",
    "Tree - Urban": "Individual trees - Urban tree",
}
cat_names = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
lower_map = {sstr(x).lower(): sstr(x) for x in cat_names}
def normalise_hab(name: str) -> str:
    n = sstr(name)
    if n in HAB_ALIAS: return HAB_ALIAS[n]
    return lower_map.get(n.lower(), n)

# ========= Official rules =========
def enforce_catalog_rules_official(demand_row, supply_row, dist_levels_map_local, explicit_rule: bool) -> bool:
    if explicit_rule:  # TradingRules override
        return True
    dh = sstr(demand_row.get("habitat_name"))
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

# ========= Options builder =========
def prepare_options(demand_df: pd.DataFrame,
                    chosen_size: str,
                    target_lpa: str, target_nca: str,
                    lpa_neigh: List[str], nca_neigh: List[str],
                    lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[List[dict], Dict[str, float]]:
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()
    SRM = backend["SRM"].copy()
    Trading = backend.get("TradingRules", pd.DataFrame())

    # Clean string columns
    for df, cols in [
        (Banks, ["lpa_name","nca_name"]),
        (Catalog, ["habitat_name","broader_type","distinctiveness_name"]),
        (Stock, ["habitat_name","stock_id"]),
        (Pricing, ["habitat_name","contract_size","tier"]),
        (Trading, ["demand_habitat","allowed_supply_habitat","min_distinctiveness_name","companion_habitat"])
    ]:
        if not df.empty:
            for c in cols:
                if c in df.columns: df[c] = df[c].map(sstr)

    srm_map = {sstr(r["tier"]): float(r["multiplier"]) for _, r in SRM.iterrows()}
    stock_full = Stock.merge(Banks[["bank_id","lpa_name","nca_name"]], on="bank_id", how="left") \
                      .merge(Catalog, on="habitat_name", how="left")
    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()

    # Trading index
    trade_idx = {}
    if not Trading.empty:
        for _, r in Trading.iterrows():
            trade_idx.setdefault(sstr(r["demand_habitat"]), []).append({
                "supply_habitat": sstr(r["allowed_supply_habitat"]),
                "min_distinctiveness_name": sstr(r.get("min_distinctiveness_name")),
                "companion_habitat": sstr(r.get("companion_habitat")),
                "companion_ratio": float(r.get("companion_ratio", 0) or 0.0),
            })

    options, remaining = [], {}
    def dval(name: Optional[str]) -> float:
        key = sstr(name)
        return dist_levels_map.get(key, dist_levels_map.get(key.lower(), -1e9))

    for di, drow in demand_df.iterrows():
        dem_hab = sstr(drow["habitat_name"])
        dcat = Catalog[Catalog["habitat_name"] == dem_hab]
        d_broader = sstr(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
        d_dist = sstr(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""
        drow = drow.copy()
        drow["broader_type"] = d_broader
        drow["distinctiveness_name"] = d_dist

        cand_parts = []
        # 1) explicit rules
        for rule in trade_idx.get(dem_hab, []):
            sh = rule["supply_habitat"]
            s_min = rule["min_distinctiveness_name"]
            df_s = stock_full[stock_full["habitat_name"] == sh].copy()
            if s_min:
                df_s = df_s[df_s["distinctiveness_name"].map(lambda x: dval(x)) >= dval(s_min)]
            df_s["companion_habitat"] = rule["companion_habitat"]
            df_s["companion_ratio"] = rule["companion_ratio"]
            if not df_s.empty:
                cand_parts.append(df_s)
        # 2) implicit (official rules)
        if not cand_parts:
            d_key = d_dist.lower()
            if d_key == "low":
                df_s = stock_full.copy()
            elif d_key == "medium":
                same_group = stock_full["broader_type"].fillna("").astype(str).map(sstr).eq(d_broader)
                higher_dist = stock_full["distinctiveness_name"].map(lambda x: dval(x)) > dval(d_dist)
                df_s = stock_full[same_group | higher_dist].copy()
            else:
                df_s = stock_full[stock_full["habitat_name"] == dem_hab].copy()
            df_s["companion_habitat"] = ""
            df_s["companion_ratio"] = 0.0
            if not df_s.empty:
                cand_parts.append(df_s)

        if not cand_parts:
            continue

        candidates = pd.concat(cand_parts, ignore_index=True)

        for _, s in candidates.iterrows():
            tier = tier_for_bank(
                s.get("lpa_name",""), s.get("nca_name",""),
                target_lpa, target_nca,
                lpa_neigh, nca_neigh,
                lpa_neigh_norm, nca_neigh_norm
            )
            price_row = pricing_cs[
                (pricing_cs["bank_id"] == s["bank_id"]) &
                (pricing_cs["habitat_name"] == s["habitat_name"]) &
                (pricing_cs["tier"] == tier)
            ]
            if price_row.empty:
                continue
            explicit = (dem_hab in trade_idx)
            if not enforce_catalog_rules_official(
                pd.Series({"habitat_name": dem_hab, "broader_type": d_broader, "distinctiveness_name": d_dist}),
                s,
                dist_levels_map,
                explicit_rule=explicit
            ):
                continue
            unit_price = float(price_row["price"].iloc[0])
            cap = float(s.get("quantity_available", 0) or 0.0)
            opt = {
                "demand_idx": di,
                "demand_habitat": dem_hab,
                "bank_id": s["bank_id"],
                "stock_id": sstr(s["stock_id"]),
                "supply_habitat": s["habitat_name"],
                "tier": tier,
                "unit_price": unit_price,
                "srm_mult": float(srm_map.get(tier, 0.5)),
                "stock_cap": cap,
                "companion_habitat": sstr(s.get("companion_habitat")),
                "companion_ratio": float(s.get("companion_ratio",0.0) or 0.0),
            }
            options.append(opt)
            remaining[opt["stock_id"]] = max(remaining.get(opt["stock_id"], 0.0), cap)

    return options, remaining

# ========= Optimiser =========
def select_size_for_demand(demand_df: pd.DataFrame, pricing_df: pd.DataFrame) -> str:
    present = pricing_df["contract_size"].drop_duplicates().tolist()
    total = float(demand_df["units_required"].sum())
    return select_contract_size(total, present)

def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str],
             lpa_neigh_norm: List[str], nca_neigh_norm: List[str]) -> Tuple[pd.DataFrame, float, str]:
    chosen_size = select_size_for_demand(demand_df, backend["Pricing"])
    options, remaining = prepare_options(demand_df, chosen_size, target_lpa, target_nca,
                                         lpa_neigh, nca_neigh, lpa_neigh_norm, nca_neigh_norm)
    if not options:
        raise RuntimeError("No feasible options. Check names, rules, prices, or stock availability.")

    if _HAS_PULP:
        prob = pulp.LpProblem("BNG_Allocation", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(len(options))]
        prob += pulp.lpSum([opt["unit_price"] * x[i] for i, opt in enumerate(options)])

        # Demand constraints (primary + companions)
        for di, drow in demand_df.iterrows():
            base = pulp.lpSum([x[i] * options[i]["srm_mult"] for i, _opt in enumerate(options) if _opt["demand_idx"] == di])
            comp = pulp.lpSum([
                x[j] * options[j]["companion_ratio"] * options[j]["srm_mult"]
                for j, _opt in enumerate(options)
                if _opt["companion_habitat"] == drow["habitat_name"]
            ])
            prob += (base + comp >= float(drow["units_required"])), f"demand_{di}"

        # Stock caps
        from collections import defaultdict
        caps = defaultdict(list)
        for i, opt in enumerate(options):
            caps[opt["stock_id"]].append(i)
        for stock_id, idxs in caps.items():
            cap_val = options[idxs[0]]["stock_cap"]
            prob += pulp.lpSum([x[i] for i in idxs]) <= cap_val, f"cap_{stock_id}"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status_str = pulp.LpStatus[prob.status]
        if status_str not in ("Optimal", "Feasible"):
            # Greedy fallback
            rows, total_cost = [], 0.0
            from collections import defaultdict
            remain = defaultdict(float)
            for opt in options:
                remain[opt["stock_id"]] = max(remain[opt["stock_id"]], opt["stock_cap"])
            for di, drow in demand_df.iterrows():
                rem_need = float(drow["units_required"])
                cand = [i for i, opt in enumerate(options) if opt["demand_idx"] == di]
                cand.sort(key=lambda i: options[i]["unit_price"] / max(options[i]["srm_mult"], 1e-9))
                for i in cand:
                    if rem_need <= 1e-9: break
                    opt = options[i]
                    cap = remain.get(opt["stock_id"], 0.0)
                    if cap <= 1e-9: continue
                    need = rem_need / max(opt["srm_mult"], 1e-9)
                    take = min(cap, need)
                    if take <= 1e-9: continue
                    remain[opt["stock_id"]] -= take
                    eff = take * opt["srm_mult"]
                    rem_need -= eff
                    cost = take * opt["unit_price"]
                    rows.append({
                        "demand_habitat": opt["demand_habitat"],
                        "bank_id": opt["bank_id"],
                        "stock_id": opt["stock_id"],
                        "supply_habitat": opt["supply_habitat"],
                        "tier": opt["tier"],
                        "units_supplied": take,
                        "effective_units": eff,
                        "unit_price": opt["unit_price"],
                        "cost": cost
                    })
                if rem_need > 1e-6:
                    raise RuntimeError(f"Optimiser status: {status_str} (and greedy short by {rem_need:.3f})")
            total_cost = float(sum(r["cost"] for r in rows))
            return pd.DataFrame(rows), total_cost, chosen_size

        # Extract LP solution
        rows, total_cost = [], 0.0
        for i, var in enumerate(x):
            qty = var.value() or 0.0
            if qty <= 1e-9: continue
            opt = options[i]
            rows.append({
                "demand_habitat": opt["demand_habitat"],
                "bank_id": opt["bank_id"],
                "stock_id": opt["stock_id"],
                "supply_habitat": opt["supply_habitat"],
                "tier": opt["tier"],
                "units_supplied": qty,
                "effective_units": qty * opt["srm_mult"],
                "unit_price": opt["unit_price"],
                "cost": qty * opt["unit_price"]
            })
            total_cost += qty * opt["unit_price"]
        return pd.DataFrame(rows), float(total_cost), chosen_size

    # Greedy (no PuLP)
    rows, total_cost = [], 0.0
    for di, drow in demand_df.iterrows():
        rem = float(drow["units_required"])
        cand = [i for i, opt in enumerate(options) if opt["demand_idx"] == di]
        cand.sort(key=lambda i: options[i]["unit_price"] / max(options[i]["srm_mult"], 1e-9))
        for i in cand:
            if rem <= 1e-9: break
            opt = options[i]
            cap = remaining.get(opt["stock_id"], 0.0)
            if cap <= 1e-9: continue
            need = rem / max(opt["srm_mult"], 1e-9)
            take = min(cap, need)
            if take <= 1e-9: continue
            remaining[opt["stock_id"]] -= take
            eff = take * opt["srm_mult"]
            rem -= eff
            cost = take * opt["unit_price"]
            rows.append({
                "demand_habitat": opt["demand_habitat"],
                "bank_id": opt["bank_id"],
                "stock_id": opt["stock_id"],
                "supply_habitat": opt["supply_habitat"],
                "tier": opt["tier"],
                "units_supplied": take,
                "effective_units": eff,
                "unit_price": opt["unit_price"],
                "cost": cost
            })
        if rem > 1e-6:
            raise RuntimeError(f"Greedy could not satisfy demand for {drow['habitat_name']} (short by {rem:.2f}).")
    total_cost = float(sum(r["cost"] for r in rows))
    return pd.DataFrame(rows), total_cost, chosen_size

# ========= Run optimiser UI =========
st.subheader("3) Run optimiser")
relaxed = st.checkbox("Relax official rules if explicit TradingRules exist (debug only)", value=False)

cL, cR = st.columns([1,1])
with cL:
    run = st.button("Optimise now", type="primary")
with cR:
    if target_lpa_name or target_nca_name:
        st.caption(f"LPA: {target_lpa_name or '—'} | NCA: {target_nca_name or '—'} | "
                   f"LPA neigh: {len(lpa_neighbors)} | NCA neigh: {len(nca_neighbors)}")
    else:
        st.caption("Tip: run ‘Locate’ first for precise tiers (else assumes ‘far’).")

# ========= Diagnostics =========
with st.expander("🔎 Diagnostics (why it might be infeasible)", expanded=False):
    try:
        dd = pd.read_csv(StringIO(demand_csv), header=None, names=["habitat_name","units_required"])
        dd["habitat_name"] = dd["habitat_name"].map(sstr).map(normalise_hab)
        dd["units_required"] = dd["units_required"].astype(float)

        present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
        total_units = float(dd["units_required"].sum())
        size_preview = select_contract_size(total_units, present_sizes)
        st.write(f"**Chosen contract size:** `{size_preview}` (present sizes: {present_sizes}, total demand: {total_units})")
        st.write(f"**Target LPA:** {target_lpa_name or '—'}  |  **Target NCA:** {target_nca_name or '—'}")
        st.write(f"**# LPA neighbours:** {len(lpa_neighbors)}  | **# NCA neighbours:** {len(nca_neighbors)}")

        st.write("**Demand (after aliases):**")
        st.dataframe(dd, use_container_width=True)

        st.subheader("Pricing coverage")
        cat_names_diag = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        missing_in_catalog = [h for h in dd["habitat_name"] if h not in cat_names_diag]
        if missing_in_catalog:
            st.error(f"Not in HabitatCatalog (fix names or add aliases): {missing_in_catalog}")

        for hab in dd["habitat_name"].unique():
            dfp = backend["Pricing"][(backend["Pricing"]["habitat_name"] == hab) &
                                     (backend["Pricing"]["contract_size"] == size_preview)]
            st.write(f"- **{hab}**: {len(dfp)} price rows for `{size_preview}`")
            if dfp.empty:
                st.warning("→ No pricing rows for this habitat/size; optimiser will have zero options.")
            else:
                st.dataframe(dfp[["bank_id","tier","price"]].sort_values(["bank_id","tier"]),
                             use_container_width=True, hide_index=True)

        st.subheader("Stock snapshot (quantity_available)")
        df_stock = backend["Stock"].merge(backend["HabitatCatalog"], on="habitat_name", how="left")
        df_need = df_stock[df_stock["habitat_name"].isin(dd["habitat_name"])]
        if df_need.empty:
            st.warning("No stock rows match your demand habitat names.")
        else:
            st.dataframe(df_need[["bank_id","habitat_name","quantity_available"]]
                         .sort_values(["habitat_name","quantity_available"], ascending=[True, False]),
                         use_container_width=True, hide_index=True)

        # Candidate options preview with tiers
        st.subheader("Candidate options & capacity by habitat")
        Banks = backend["Banks"].copy()
        Pricing = backend["Pricing"].copy()
        Catalog = backend["HabitatCatalog"].copy()
        Stock = backend["Stock"].copy()
        SRM = backend["SRM"].copy()
        Trading = backend.get("TradingRules", pd.DataFrame())

        for df, cols in [
            (Banks, ["lpa_name","nca_name"]),
            (Catalog, ["habitat_name","broader_type","distinctiveness_name"]),
            (Stock, ["habitat_name"]),
            (Pricing, ["habitat_name","contract_size","tier"]),
            (Trading, ["demand_habitat","allowed_supply_habitat","min_distinctiveness_name","companion_habitat"])
        ]:
            if not df.empty:
                for c in cols:
                    if c in df.columns: df[c] = df[c].map(sstr)

        srm_map = {sstr(r["tier"]): float(r["multiplier"]) for _, r in backend["SRM"].iterrows()}
        D = backend["DistinctivenessLevels"]
        dmap = {sstr(r["distinctiveness_name"]): float(r["level_value"]) for _, r in D.iterrows()}
        dmap.update({k.lower(): v for k, v in list(dmap.items())})

        stock_full = Stock.merge(Banks[["bank_id","lpa_name","nca_name"]], on="bank_id", how="left") \
                          .merge(Catalog, on="habitat_name", how="left")
        pricing_cs = Pricing[Pricing["contract_size"] == size_preview].copy()

        trade_idx = {}
        if not Trading.empty:
            for _, r in Trading.iterrows():
                trade_idx.setdefault(sstr(r["demand_habitat"]), []).append({
                    "supply_habitat": sstr(r["allowed_supply_habitat"]),
                    "min_distinctiveness_name": sstr(r.get("min_distinctiveness_name")),
                    "companion_habitat": sstr(r.get("companion_habitat")),
                    "companion_ratio": float(r.get("companion_ratio",0) or 0.0),
                })

        def dval(x):
            key = sstr(x)
            return dmap.get(key, dmap.get(key.lower(), -1e9))

        for _, row in dd.iterrows():
            dem = sstr(row["habitat_name"]); req = float(row["units_required"])
            st.markdown(f"**Demand: {dem} → {req} units**")
            dcat = Catalog[Catalog["habitat_name"] == dem]
            d_broader = sstr(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
            d_dist = sstr(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""
            if dcat.empty:
                st.error("✖ Not in catalog — add to HabitatCatalog or fix alias.")
                continue
            pol = "Low→anything" if d_dist.lower()=="low" else ("Medium→same group OR higher distinctiveness" if d_dist.lower()=="medium" else "High/Very High→like-for-like")
            st.caption(f"Trading policy for '{dem}': {pol}")

            parts, reasons = [], []
            if dem in trade_idx:
                for rule in trade_idx[dem]:
                    sh = rule["supply_habitat"]; s_min = rule["min_distinctiveness_name"]
                    df_s = stock_full[stock_full["habitat_name"] == sh].copy()
                    if s_min:
                        before = len(df_s)
                        df_s = df_s[df_s["distinctiveness_name"].map(dval) >= dval(s_min)]
                        if len(df_s) < before:
                            reasons.append(f"- Filtered {before-len(df_s)} rows below min distinctiveness {s_min}.")
                    df_s["companion_habitat"] = rule["companion_habitat"]; df_s["companion_ratio"] = rule["companion_ratio"]
                    if not df_s.empty: parts.append(df_s)
                if not parts:
                    st.error("✖ TradingRules present but produced no stock rows.")
                    continue
            else:
                d_key = d_dist.lower()
                if d_key == "low":
                    df_s = stock_full.copy()
                elif d_key == "medium":
                    same_group = stock_full["broader_type"].fillna("").astype(str).map(sstr).eq(d_broader)
                    higher_dist = stock_full["distinctiveness_name"].map(dval) > dval(d_dist)
                    df_s = stock_full[same_group | higher_dist].copy()
                else:
                    df_s = stock_full[stock_full["habitat_name"] == dem].copy()
                df_s["companion_habitat"] = ""; df_s["companion_ratio"] = 0.0
                if df_s.empty:
                    st.error("✖ No candidates from official rules.")
                    continue
                parts.append(df_s)

            cand = pd.concat(parts, ignore_index=True)
            explicit = (dem in trade_idx)
            before = len(cand)
            mask = cand.apply(lambda srow: enforce_catalog_rules_official(
                    pd.Series({"habitat_name": dem, "broader_type": d_broader, "distinctiveness_name": d_dist}),
                    srow, dist_levels_map, explicit_rule=explicit
                ), axis=1)
            cand = cand[mask]
            if len(cand) < before:
                reasons.append(f"- Official rules filtered {before-len(cand)} rows.")
            if cand.empty:
                st.error("✖ No candidates after official rules.")
                if reasons: st.caption("\n".join(reasons))
                continue

            rows2, removed_no_price = [], 0
            for _, srow in cand.iterrows():
                tier = tier_for_bank(
                    srow.get("lpa_name",""), srow.get("nca_name",""),
                    target_lpa_name or "", target_nca_name or "",
                    lpa_neighbors, nca_neighbors,
                    lpa_neighbors_norm, nca_neighbors_norm
                )
                pr = pricing_cs[(pricing_cs["bank_id"] == srow["bank_id"]) &
                                (pricing_cs["habitat_name"] == srow["habitat_name"]) &
                                (pricing_cs["tier"] == tier)]
                if pr.empty:
                    removed_no_price += 1
                    continue
                rows2.append({
                    "bank_id": srow["bank_id"],
                    "supply_habitat": srow["habitat_name"],
                    "tier": tier,
                    "unit_price": float(pr["price"].iloc[0]),
                    "srm_mult": float(srm_map.get(tier, 0.5)),
                    "stock_cap": float(srow.get("quantity_available",0) or 0.0),
                    "companion_habitat": sstr(srow.get("companion_habitat")),
                    "companion_ratio": float(srow.get("companion_ratio",0) or 0.0),
                })
            if removed_no_price:
                reasons.append(f"- Dropped {removed_no_price} rows with no price for chosen size + tier.")
            cand2 = pd.DataFrame(rows2)
            if cand2.empty:
                st.error("✖ All candidates dropped (pricing/tier mismatch).")
                if reasons: st.caption("\n".join(reasons))
                continue
            before = len(cand2); cand2 = cand2[cand2["stock_cap"] > 0]
            if len(cand2) < before:
                reasons.append(f"- Removed {before-len(cand2)} rows with zero stock.")
            if cand2.empty:
                st.error("✖ No candidates have positive stock.")
                if reasons: st.caption("\n".join(reasons))
                continue

            cand2["eff_from_primary"] = cand2["stock_cap"] * cand2["srm_mult"]
            eff_cap = cand2["eff_from_primary"].sum()

            st.write("**Tier breakdown of candidates kept:**")
            st.dataframe(
                cand2.groupby("tier", as_index=False).agg(
                    options=("tier","count"),
                    eff_capacity=("eff_from_primary","sum")
                ),
                use_container_width=True, hide_index=True
            )
            st.dataframe(cand2.sort_values(["unit_price","tier","bank_id"]),
                         use_container_width=True, hide_index=True)

            if reasons: st.caption("\n".join(reasons))
            if eff_cap + 1e-9 < req:
                st.error(f"✖ Insufficient effective capacity: need {req}, best-case {eff_cap:.3f}")
            else:
                st.success(f"✔ Effective capacity OK (need {req}, best-case {eff_cap:.3f})")
    except Exception as de:
        st.error(f"Diagnostics error: {de}")

# --- Run optimiser ---
if run:
    try:
        demand_df = pd.read_csv(StringIO(demand_csv), header=None, names=["habitat_name","units_required"])
        demand_df["habitat_name"] = demand_df["habitat_name"].map(sstr).map(normalise_hab)
        demand_df["units_required"] = demand_df["units_required"].astype(float)

        cat_names_run = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        unknown = [h for h in demand_df["habitat_name"] if h not in cat_names_run]
        if unknown:
            st.error(f"These demand habitats aren’t in the catalog: {unknown}")
            st.stop()

        target_lpa = sstr(target_lpa_name)
        target_nca = sstr(target_nca_name)

        alloc_df, total_cost, size = optimise(
            demand_df,
            target_lpa, target_nca,
            [sstr(n) for n in lpa_neighbors], [sstr(n) for n in nca_neighbors],
            lpa_neighbors_norm, nca_neighbors_norm
        )

        st.success(f"Optimisation complete. Contract size = **{size}**. Total cost: **£{total_cost:,.0f}**")

        st.markdown("#### Allocation detail")
        st.dataframe(alloc_df, use_container_width=True)

        st.markdown("#### By bank")
        by_bank = alloc_df.groupby("bank_id", as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            effective_units=("effective_units","sum"),
            cost=("cost","sum")
        )
        st.dataframe(by_bank, use_container_width=True)

        st.markdown("#### By habitat")
        by_hab = alloc_df.groupby("supply_habitat", as_index=False).agg(
            units_supplied=("units_supplied","sum"),
            effective_units=("effective_units","sum"),
            cost=("cost","sum")
        )
        st.dataframe(by_hab, use_container_width=True)

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







