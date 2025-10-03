# app.py — BNG Optimiser (Standalone) with login + hardened requests

import json
import math
from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium

# =========================
# Page config
# =========================
st.set_page_config(page_title="BNG Optimiser (Standalone)", page_icon="🧭", layout="wide")
st.markdown("<h2>BNG Optimiser — Standalone</h2>", unsafe_allow_html=True)
st.caption("Upload backend workbook, locate target site, and optimise supply with SRM, distinctiveness and TradingRules.")

# =========================
# Login wall (secrets first, else fallback)
# =========================
DEFAULT_USER = "WC0323"
DEFAULT_PASS = "Wimbourne"

def require_login():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        # optional logout
        with st.sidebar:
            if st.button("Log out"):
                st.session_state.auth_ok = False
                st.rerun()
        return

    st.markdown("## 🔐 Sign in")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")

    if submit:
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

# =========================
# Constants & endpoints
# =========================
UA = {"User-Agent": "WildCapital-Optimiser/1.0 (+contact@example.com)"}  # set a real contact email

POSTCODES_IO = "https://api.postcodes.io/postcodes/"
POSTCODES_IO_REVERSE = "https://api.postcodes.io/postcodes"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

# ArcGIS FeatureServers
NCA_URL = (
    "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
    "National_Character_Areas_England/FeatureServer/0"
)
LPA_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2024_Boundaries_UK_BFC/FeatureServer/0"
)

# Try PuLP (optional)
try:
    import pulp
    _HAS_PULP = True
except Exception:
    _HAS_PULP = False

# =========================
# Hardened HTTP helpers
# =========================
def http_get(url, params=None, headers=None, timeout=25):
    try:
        r = requests.get(url, params=params or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise RuntimeError(f"HTTP error for {url}: {e}")

def safe_json(r: requests.Response) -> Dict[str, Any]:
    try:
        return r.json()
    except Exception:
        text_preview = (r.text or "")[:300]
        raise RuntimeError(
            f"Invalid JSON from {r.url} (status {r.status_code}). "
            f"Response starts with: {text_preview}"
        )

def http_post(url, data=None, headers=None, timeout=25):
    try:
        r = requests.post(url, data=data or {}, headers=headers or UA, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise RuntimeError(f"HTTP POST error for {url}: {e}")


# =========================
# Geocoding / lookups
# =========================
def get_postcode_info(pc: str) -> Tuple[float, float, str]:
    pc_clean = pc.replace(" ", "").upper()
    r = http_get(POSTCODES_IO + pc_clean)
    js = safe_json(r)
    if js.get("status") != 200 or not js.get("result"):
        raise RuntimeError(f"Postcode lookup failed for '{pc}'.")
    data = js["result"]
    return float(data["latitude"]), float(data["longitude"]), (data.get("admin_district") or data.get("admin_county") or "")

def reverse_postcode(lat: float, lon: float) -> Optional[str]:
    r = http_get(POSTCODES_IO_REVERSE, params={"lon": lon, "lat": lat, "limit": 1})
    js = safe_json(r)
    res = js.get("result") or []
    return (res[0] or {}).get("postcode") if res else None

def geocode_address(addr: str) -> Tuple[float, float]:
    # Nominatim
    r = http_get(NOMINATIM_SEARCH, params={"q": addr, "format": "jsonv2", "limit": 1, "addressdetails": 0})
    js = safe_json(r)
    if isinstance(js, list) and js:
        lat, lon = js[0]["lat"], js[0]["lon"]
        return float(lat), float(lon)
    # Photon fallback
    r = http_get("https://photon.komoot.io/api/", params={"q": addr, "limit": 1})
    js = safe_json(r)
    feats = js.get("features") or []
    if feats:
        lon, lat = feats[0]["geometry"]["coordinates"]
        return float(lat), float(lon)
    raise RuntimeError("Address geocoding failed.")

def arcgis_point_query(layer_url: str, lat: float, lon: float, out_fields: str) -> Dict[str, Any]:
    geometry_dict = {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}
    params = {
        "f": "json",
        "where": "1=1",
        "geometry": json.dumps(geometry_dict),
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields or "*",
        "returnGeometry": "true",
        "outSR": 4326
    }
    r = http_get(f"{layer_url}/query", params=params)
    js = safe_json(r)
    feats = js.get("features") or []
    return feats[0] if feats else {}

def layer_intersect_names(layer_url: str, polygon_geom: Dict[str, Any], name_field: str) -> List[str]:
    """
    Query ArcGIS with POST to avoid 414 (URI Too Large) when sending polygon geometries.
    We also use geometryPrecision to reduce payload size. Returns all features that INTERSECT
    the given polygon (this includes the polygon itself; we filter the caller’s own name later).
    """
    if not polygon_geom:
        return []

    # ArcGIS accepts either rings (polygon) or an envelope. We keep the polygon for correctness,
    # but reduce coordinate precision to shrink payload size.
    data = {
        "f": "json",
        "where": "1=1",
        "geometry": json.dumps(polygon_geom),           # full polygon, but…
        "geometryType": "esriGeometryPolygon",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": name_field,
        "returnGeometry": "false",
        "outSR": 4326,

        # Light server-side simplification: trims coordinate precision to reduce bytes.
        # (This does NOT generalize topology; it just shortens numbers.)
        "geometryPrecision": 5,   # ~1e-5 deg ≈ 1 m-ish
        # Optional generalization for very complex polygons:
        # "maxAllowableOffset": 0.0001,  # ~11 m in WGS84 (tune if needed)
    }

    r = http_post(f"{layer_url}/query", data=data)
    js = safe_json(r)
    names = [ (f.get("attributes") or {}).get(name_field) for f in js.get("features", []) ]
    return sorted({n for n in names if n})


def tier_for_bank(bank_lpa: str, bank_nca: str, t_lpa: str, t_nca: str, lpa_neigh: List[str], nca_neigh: List[str]) -> str:
    if bank_lpa == t_lpa or bank_nca == t_nca:
        return "local"
    if (bank_lpa in lpa_neigh) or (bank_nca in nca_neigh):
        return "adjacent"
    return "far"

def select_contract_size(total_units: float, present: List[str]) -> str:
    tiers = set(present)
    if "fractional" in tiers and total_units < 0.1: return "fractional"
    if "small" in tiers and total_units < 2.5: return "small"
    if "medium" in tiers and total_units < 15: return "medium"
    for t in ["large", "medium", "small", "fractional"]:
        if t in tiers: return t
    return present[0]

# =========================
# Sidebar: Backend + policy
# =========================
with st.sidebar:
    st.subheader("Backend")
    uploaded = st.file_uploader("Upload backend workbook (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.info("Or use an example backend in ./data", icon="ℹ️")
    use_example = st.checkbox("Use example backend from ./data", value=bool(Path("data/HabitatBackend_WITH_STOCK.xlsx").exists()))
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

# Apply quotes policy if WITH_STOCK fields exist
if "available_excl_quotes" in backend["Stock"].columns and "quoted" in backend["Stock"].columns:
    s = backend["Stock"].copy()
    if quotes_hold_policy == "Ignore quotes (default)":
        s["quantity_available"] = s["available_excl_quotes"]
    elif quotes_hold_policy == "Quotes hold 100%":
        s["quantity_available"] = (s["available_excl_quotes"] - s["quoted"]).clip(lower=0)
    else:  # 50%
        s["quantity_available"] = (s["available_excl_quotes"] - 0.5 * s["quoted"]).clip(lower=0)
    backend["Stock"] = s

# Validate minimal columns
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
    str(r["distinctiveness_name"]).strip(): float(r["level_value"])
    for _, r in backend["DistinctivenessLevels"].iterrows()
}

# =========================
# Target site input
# =========================
with st.container():
    st.subheader("1) Locate target site")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        postcode = st.text_input("Postcode (quicker)", value="")
    with colB:
        address = st.text_input("Address (if no postcode)", value="")
    with colC:
        run_locate = st.button("Locate")

lat = lon = None
lpa_feat = nca_feat = None
target_lpa_name = target_nca_name = ""
lpa_neighbors = nca_neighbors = []

if run_locate:
    try:
        if postcode.strip():
            lat, lon, _ = get_postcode_info(postcode.strip())
        elif address.strip():
            lat, lon = geocode_address(address.strip())
        else:
            st.error("Enter a postcode or an address.")
            st.stop()

        # Fetch LPA/NCA at point
        lpa_feat = arcgis_point_query(LPA_URL, lat, lon, "LAD24NM")
        nca_feat = arcgis_point_query(NCA_URL, lat, lon, "NCA_Name")
        target_lpa_name = (lpa_feat.get("attributes") or {}).get("LAD24NM", "")
        target_nca_name = (nca_feat.get("attributes") or {}).get("NCA_Name", "")
        st.success(f"Found LPA: **{target_lpa_name}** | NCA: **{target_nca_name}**")

        # Neighbours
        lpa_neighbors = [n for n in layer_intersect_names(LPA_URL, lpa_feat.get("geometry"), "LAD24NM") if n != target_lpa_name]
        nca_neighbors = [n for n in layer_intersect_names(NCA_URL, nca_feat.get("geometry"), "NCA_Name") if n != target_nca_name]

        # Map
        fmap = folium.Map(location=[lat, lon], zoom_start=11, control_scale=True)
        folium.CircleMarker([lat, lon], radius=5, color="red", fill=True, tooltip="Target").add_to(fmap)
        st_folium(fmap, height=380, returned_objects=[], use_container_width=True)

    except Exception as e:
        st.error(f"Location error: {e}")

# =========================
# Demand input
# =========================
st.subheader("2) Demand (units required)")
default_demand = "Individual trees - Urban tree,8\nGrassland - Other neutral grassland,30"
demand_csv = st.text_area("CSV: habitat_name,units_required", value=default_demand, height=120)

# =========================
# Optimiser helpers
# =========================
def enforce_catalog_rules(demand_row, supply_row) -> bool:
    # same broader_type AND supply distinctiveness >= demand distinctiveness
    dbt = str(demand_row.get("broader_type","")).strip()
    sbt = str(supply_row.get("broader_type","")).strip()
    dd = dist_levels_map.get(str(demand_row.get("distinctiveness_name","")).strip(), None)
    sd = dist_levels_map.get(str(supply_row.get("distinctiveness_name","")).strip(), None)
    if dbt and sbt and dbt != sbt: return False
    if (dd is not None) and (sd is not None) and (sd < dd): return False
    return True

def prepare_options(demand_df: pd.DataFrame,
                    chosen_size: str,
                    target_lpa: str, target_nca: str,
                    lpa_neigh: List[str], nca_neigh: List[str]) -> Tuple[List[dict], Dict[str, float]]:
    Banks = backend["Banks"].copy()
    Pricing = backend["Pricing"].copy()
    Catalog = backend["HabitatCatalog"].copy()
    Stock = backend["Stock"].copy()
    SRM = backend["SRM"].copy()
    Trading = backend.get("TradingRules", pd.DataFrame())

    srm_map = {r["tier"]: float(r["multiplier"]) for _, r in SRM.iterrows()}

    # Merge stock with bank context + catalog
    stock_full = Stock.merge(Banks[["bank_id","lpa_name","nca_name"]], on="bank_id", how="left") \
                      .merge(Catalog, on="habitat_name", how="left")

    pricing_cs = Pricing[Pricing["contract_size"] == chosen_size].copy()

    # Trading index
    trade_idx = {}
    if not Trading.empty:
        for _, r in Trading.iterrows():
            trade_idx.setdefault(str(r["demand_habitat"]).strip(), []).append({
                "supply_habitat": str(r["allowed_supply_habitat"]).strip(),
                "min_distinctiveness_name": str(r.get("min_distinctiveness_name","")).strip() or None,
                "companion_habitat": str(r.get("companion_habitat","")).strip() or None,
                "companion_ratio": float(r.get("companion_ratio",0) or 0.0),
            })

    options = []
    remaining = {}

    def dval(name: Optional[str]) -> float:
        return dist_levels_map.get((name or "").strip(), -1e9)

    for di, drow in demand_df.iterrows():
        dem_hab = str(drow["habitat_name"]).strip()
        dcat = Catalog[Catalog["habitat_name"] == dem_hab]
        d_broader = str(dcat["broader_type"].iloc[0]) if not dcat.empty else ""
        d_dist = str(dcat["distinctiveness_name"].iloc[0]) if not dcat.empty else ""
        drow = drow.copy()
        drow["broader_type"] = d_broader
        drow["distinctiveness_name"] = d_dist

        cand_parts = []

        # 1) Explicit TradingRules for this demand
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

        # 2) If no explicit rule, allow like-for-like subject to catalog guardrails
        if not cand_parts:
            df_s = stock_full[stock_full["habitat_name"] == dem_hab].copy()
            df_s["companion_habitat"] = ""
            df_s["companion_ratio"] = 0.0
            if not df_s.empty:
                cand_parts.append(df_s)

        if not cand_parts:
            continue

        candidates = pd.concat(cand_parts, ignore_index=True)

        # Attach tier & price
        for _, s in candidates.iterrows():
            tier = tier_for_bank(s.get("lpa_name",""), s.get("nca_name",""),
                                 target_lpa, target_nca, lpa_neigh, nca_neigh)
            price_row = pricing_cs[
                (pricing_cs["bank_id"] == s["bank_id"]) &
                (pricing_cs["habitat_name"] == s["habitat_name"]) &
                (pricing_cs["tier"] == tier)
            ]
            if price_row.empty:
                continue

            # If came from explicit TradingRules, skip catalog guardrail; else enforce it
            explicit = dem_hab in trade_idx
            if not explicit and not enforce_catalog_rules(
                pd.Series({"habitat_name": dem_hab, "broader_type": d_broader, "distinctiveness_name": d_dist}), s
            ):
                continue

            unit_price = float(price_row["price"].iloc[0])
            options.append({
                "demand_idx": di,
                "demand_habitat": dem_hab,
                "bank_id": s["bank_id"],
                "stock_id": s["stock_id"],
                "supply_habitat": s["habitat_name"],
                "tier": tier,
                "unit_price": unit_price,
                "srm_mult": float(srm_map.get(tier, 0.5)),
                "stock_cap": float(s["quantity_available"]),
                "companion_habitat": s.get("companion_habitat","") or "",
                "companion_ratio": float(s.get("companion_ratio",0.0) or 0.0),
            })
            remaining[options[-1]["stock_id"]] = max(remaining.get(options[-1]["stock_id"], 0.0), float(s["quantity_available"]))

    return options, remaining

def optimise(demand_df: pd.DataFrame,
             target_lpa: str, target_nca: str,
             lpa_neigh: List[str], nca_neigh: List[str]) -> Tuple[pd.DataFrame, float, str]:
    present_sizes = backend["Pricing"]["contract_size"].drop_duplicates().tolist()
    total_units = float(demand_df["units_required"].sum())
    chosen_size = select_contract_size(total_units, present_sizes)

    options, remaining = prepare_options(demand_df, chosen_size, target_lpa, target_nca, lpa_neigh, nca_neigh)
    if not options:
        raise RuntimeError("No feasible options. Check naming, rules, prices, or stock availability.")

    # Optimise: LP first
    if _HAS_PULP:
        prob = pulp.LpProblem("BNG_Allocation", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(len(options))]
        prob += pulp.lpSum([opt["unit_price"] * x[i] for i, opt in enumerate(options)])

        # Demand constraints (effective units incl. companions)
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
        if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            raise RuntimeError(f"Optimiser status: {pulp.LpStatus[prob.status]}")

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

    # Greedy fallback
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
            total_cost += cost
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
    return pd.DataFrame(rows), float(total_cost), chosen_size

# =========================
# Run UI
# =========================
st.subheader("3) Run optimiser")
left, right = st.columns([1,1])
with left:
    run = st.button("Optimise now", type="primary", disabled=False)
with right:
    st.caption("Tip: run ‘Locate’ first to load LPA/NCA & neighbours for precise tiering.")

if run:
    try:
        demand_df = pd.read_csv(StringIO(demand_csv), header=None, names=["habitat_name","units_required"])
        demand_df["habitat_name"] = demand_df["habitat_name"].astype(str).str.strip()
        demand_df["units_required"] = demand_df["units_required"].astype(float)

        # Guard: demand habitats exist in catalog
        cat_names = set(backend["HabitatCatalog"]["habitat_name"].astype(str))
        unknown = [h for h in demand_df["habitat_name"] if h not in cat_names]
        if unknown:
            st.warning(f"These demand habitats aren’t in the catalog: {unknown}")

        # If user didn’t run Locate, we still allow optimisation using 'far' tier everywhere
        if not target_lpa_name and not target_nca_name:
            lpa_neighbors, nca_neighbors = [], []
            target_lpa, target_nca = "", ""
        else:
            target_lpa, target_nca = target_lpa_name or "", target_nca_name or ""

        alloc_df, total_cost, size = optimise(
            demand_df,
            target_lpa,
            target_nca,
            lpa_neighbors, nca_neighbors
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

        # Downloads
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

