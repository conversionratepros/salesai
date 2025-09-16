# frankiedash.py
from flask import Flask, render_template, jsonify, request, redirect
import sys
import os
from datetime import datetime, timedelta, date
from google.cloud import bigquery
from google.oauth2 import service_account
from flask import g, request, redirect, url_for
from types import FunctionType

HELPER_NAMES = {"date_from", "date_to", "url_with_dates"}  # legacy names to guard
DATE_FMT = "%Y-%m-%d"
RUN_DIAGNOSTICS = False  # set True only when you need to debug

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the BigQuery merchandise analyzer
from frankmerch_bigquery import MerchandiseAnalyzerBQ

# Import the BigQuery merchandise analyzer safely
try:
    from frankmerch_bigquery import MerchandiseAnalyzerBQ
except ImportError as e:
    MerchandiseAnalyzerBQ = None
    print(f"❌ Could not import frankmerch_bigquery.MerchandiseAnalyzerBQ: {e}")

app = Flask(__name__)
# ---- Simple in-memory cache (TTL) -------------------------------------------
# Usage: data = get_cached("key", lambda: expensive_fn(), ttl_secs=300, refresh=False)
from datetime import datetime, timedelta

CACHE: dict = {}

def get_cached(key: str, builder, ttl_secs: int = 300, refresh: bool = False):
    """
    key: unique cache key (include route + from/to dates)
    builder: zero-arg function that returns the value to cache
    ttl_secs: time-to-live in seconds
    refresh: if True, bypass cache and rebuild
    """
    now = datetime.utcnow()
    if not refresh:
        entry = CACHE.get(key)
        if entry and now < entry["expires"]:
            return entry["value"]

    value = builder()
    CACHE[key] = {"value": value, "expires": now + timedelta(seconds=ttl_secs)}
    return value

# optional helper if you ever want to nuke everything from a REPL
def clear_cache():
    CACHE.clear()
# -----------------------------------------------------------------------------
# Configuration
PROJECT_ID = 'carrol-boyes-ga4'  # Your GCP project
GA4_PROPERTY_ID = '313412063'  # Your GA4 property ID
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _default_dates():
    end = datetime.today().date()
    start = end - timedelta(days=29)  # last 30 days inclusive
    return start.strftime(DATE_FMT), end.strftime(DATE_FMT)

@app.before_request
def inject_global_date_range():
    q_from = request.args.get("from")
    q_to   = request.args.get("to")

    if not q_from or not q_to:
        df, dt = _default_dates()
    else:
        # validate & normalize
        try:
            _f = datetime.strptime(q_from, DATE_FMT).date()
            _t = datetime.strptime(q_to,   DATE_FMT).date()
            if _t < _f:  # swap if user inverted
                _f, _t = _t, _f
            df, dt = _f.strftime(DATE_FMT), _t.strftime(DATE_FMT)
        except ValueError:
            df, dt = _default_dates()

    g.date_from = df
    g.date_to   = dt

@app.context_processor
def add_date_helpers():
    """Provide helpers under a single namespace to avoid name collisions in templates."""
    class _Helpers:
        def date_from(self):
            return g.get("date_from")
        def date_to(self):
            return g.get("date_to")
        def url(self, endpoint, **kwargs):
            kwargs.setdefault("from", g.get("date_from"))
            kwargs.setdefault("to",   g.get("date_to"))
            return url_for(endpoint, **kwargs)

    return {"helpers": _Helpers()}

# Try to import AI recommendations module
try:
    from ai_recommendations import FrankieAIRecommendations
    AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI Recommendations module not found: {e}")
    FrankieAIRecommendations = None
    AI_AVAILABLE = False

# Initialize AI Recommender
ai_recommender = None
if AI_AVAILABLE:
    try:
        ai_recommender = FrankieAIRecommendations()
        print("✓ AI Recommendations module initialized")
    except ValueError as e:
        print(f"⚠️ AI Recommendations not available: {e}")
        print("Set OPENAI_API_KEY environment variable to enable AI features")
    except Exception as e:
        print(f"⚠️ Error initializing AI module: {e}")

def get_bigquery_client():
    """
    Initialize BigQuery client with clear, actionable errors.
    Priority for credentials path:
      1) BQ_CREDENTIALS_PATH env var (absolute path recommended)
      2) GOOGLE_APPLICATION_CREDENTIALS env var
      3) ./bigquery-credentials.json next to this file
    """
    try:
        # 1) Pick a credentials path
        cand_paths = []

        # env-first
        if os.getenv("BQ_CREDENTIALS_PATH"):
            cand_paths.append(os.getenv("BQ_CREDENTIALS_PATH"))
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            cand_paths.append(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

        # fallback to local file in repo
        repo_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bigquery-credentials.json")
        cand_paths.append(repo_local)

        cred_path = None
        for p in cand_paths:
            if p and os.path.isfile(p):
                cred_path = p
                break

        if not cred_path:
            raise FileNotFoundError(
                "No credentials file found. Set BQ_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS, "
                "or place bigquery-credentials.json next to frankiedash.py."
            )

        # 2) Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )

        # 3) Create client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

        # 4) Print who we are (super helpful for debugging)
        sa_email = credentials.service_account_email if hasattr(credentials, "service_account_email") else "Unknown"
        print(f"✓ BigQuery auth using service account: {sa_email}")
        print(f"✓ Project: {PROJECT_ID}")

        # 5) Ping BigQuery so we fail fast with a real error
        client.query("SELECT 1").result()
        print("✓ BigQuery connectivity: OK")

        return client

    except Exception as e:
        print("❌ BigQuery client initialization failed.")
        print(f"   Reason: {e.__class__.__name__}: {e}")
        import traceback; traceback.print_exc()
        raise Exception(f"BigQuery client initialization failed: {e}")
    
def resolve_date_range(preset=None, date_from=None, date_to=None, tz_offset_hours=2):
    """
    Returns (from_str, to_str, prev_from_str, prev_to_str) in 'YYYY-MM-DD'.
    If preset is 'yesterday'|'last7'|'last30', it overrides date_from/date_to.
    'Previous period' is the same length immediately before the current window.
    """
    # Today in UTC; GA4 export timestamps are UTC. Adjust if you want local day boundaries:
    today = datetime.utcnow() + timedelta(hours=tz_offset_hours)

    if preset == "yesterday":
        end = (today - timedelta(days=1)).date()
        start = end
    elif preset == "last7":
        end = (today - timedelta(days=1)).date()
        start = end - timedelta(days=6)
    elif preset == "last30":
        end = (today - timedelta(days=1)).date()
        start = end - timedelta(days=29)
    else:
        # custom range (strings 'YYYY-MM-DD')
        if not date_from or not date_to:
            # default to last 30
            end = (today - timedelta(days=1)).date()
            start = end - timedelta(days=29)
        else:
            start = datetime.strptime(date_from, "%Y-%m-%d").date()
            end   = datetime.strptime(date_to,   "%Y-%m-%d").date()

    length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)

    f = start.strftime("%Y-%m-%d")
    t = end.strftime("%Y-%m-%d")
    pf = prev_start.strftime("%Y-%m-%d")
    pt = prev_end.strftime("%Y-%m-%d")
    return f, t, pf, pt

def get_channel_metrics_from_bigquery(date_from, date_to):
    """Build real Channel Performance data: traffic mix + users & ARPU per channel."""
    if not PROJECT_ID or not GA4_PROPERTY_ID:
        raise ValueError("PROJECT_ID and GA4_PROPERTY_ID must be set")

    client = get_bigquery_client()

    # Parse dates for GA4 table suffix
    start_suffix = datetime.strptime(date_from, '%Y-%m-%d').strftime('%Y%m%d')
    end_suffix   = datetime.strptime(date_to,   '%Y-%m-%d').strftime('%Y%m%d')

    table = f"`{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`"

    # We compute:
    # - assigned_channel: using your mapping
    # - users per channel (distinct user_pseudo_id)
    # - revenue per channel from event_params (common GA4 keys)
    # - ARPU = revenue / users
    query = f"""
    WITH base AS (
      SELECT
        user_pseudo_id,
        event_name,
        -- Map to "Default Channel Group" style buckets
        CASE
          WHEN traffic_source.source = '(direct)' OR traffic_source.source IS NULL OR traffic_source.source = 'None'
            THEN 'Direct'
          WHEN REGEXP_CONTAINS(traffic_source.source, 'cross-network')
            THEN 'Cross-network'
          WHEN traffic_source.medium IN ('cpc','ppc','paidsearch')
               AND traffic_source.source IN ('google','bing','yahoo','baidu','duckduckgo')
            THEN 'Paid Search'
          WHEN traffic_source.medium = 'organic'
            THEN 'Organic Search'
          WHEN (LOWER(traffic_source.source) LIKE '%meta%' OR LOWER(traffic_source.source) LIKE '%facebook%' OR LOWER(traffic_source.source) LIKE '%instagram%' OR LOWER(traffic_source.source) IN ('ig','fb'))
               AND (LOWER(traffic_source.medium) LIKE '%paid%' OR LOWER(traffic_source.medium) LIKE '%ad%' OR LOWER(traffic_source.medium) LIKE '%cpc%' OR LOWER(traffic_source.medium) LIKE '%ppc%' OR LOWER(traffic_source.medium) LIKE '%instagram%' OR LOWER(traffic_source.medium) LIKE '%meta%' OR LOWER(traffic_source.medium) LIKE '%carousel%' OR LOWER(traffic_source.medium) LIKE '%whatsapp%')
            THEN 'Paid Social'
          WHEN (LOWER(traffic_source.source) LIKE '%facebook%' OR LOWER(traffic_source.source) LIKE '%instagram%' OR LOWER(traffic_source.source) LIKE '%twitter%' OR LOWER(traffic_source.source) LIKE '%linkedin%' OR LOWER(traffic_source.source) LIKE '%pinterest%' OR LOWER(traffic_source.source) LIKE '%tiktok%' OR LOWER(traffic_source.source) IN ('meta','socialorganic','social_organic'))
               AND (LOWER(traffic_source.medium) IN ('social','socialorganic','social-network','social-media','sm','social organic','meta') OR traffic_source.medium IS NULL)
            THEN 'Organic Social'
          WHEN LOWER(traffic_source.medium) = 'email' OR LOWER(traffic_source.source) LIKE '%newsletter%' OR LOWER(traffic_source.source) = 'netcore'
            THEN 'Email'
          WHEN traffic_source.medium = 'affiliate'
            THEN 'Affiliates'
          WHEN traffic_source.medium = 'referral'
            THEN 'Referral'
          WHEN traffic_source.medium IN ('display','cpm','banner')
            THEN 'Display'
          WHEN traffic_source.source LIKE '%shopping%' AND traffic_source.medium = 'organic'
            THEN 'Organic Shopping'
          WHEN traffic_source.source LIKE '%shopping%' AND traffic_source.medium IN ('cpc','ppc','paid')
            THEN 'Paid Shopping'
          WHEN traffic_source.source = 'Data Not Available' OR traffic_source.medium = 'Data Not Available'
            THEN 'Unassigned'
          ELSE 'Unassigned'
        END AS assigned_channel,

        -- Pull purchase revenue from common GA4 keys (value/value_in_usd/purchase_revenue_in_usd)
        CASE
          WHEN event_name = 'purchase' THEN COALESCE(
            (
              SELECT MAX(
                     CAST(
                       COALESCE(ep.value.double_value,
                                CAST(ep.value.int_value AS FLOAT64),
                                SAFE_CAST(ep.value.string_value AS FLOAT64)) AS FLOAT64
                     )
                   )
              FROM UNNEST(event_params) ep
              WHERE ep.key IN ('value','value_in_usd','purchase_revenue_in_usd')
            ),
            0.0
          )
          ELSE 0.0
        END AS revenue_value
      FROM {table}
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
    ),

    by_channel AS (
      SELECT
        assigned_channel,
        COUNT(DISTINCT user_pseudo_id) AS users,
        SUM(revenue_value)             AS total_revenue
      FROM base
      GROUP BY assigned_channel
    ),

    nonzero AS (
      SELECT
        assigned_channel,
        users,
        total_revenue,
        ROUND(SAFE_DIVIDE(total_revenue, NULLIF(users,0)), 2) AS arpu
      FROM by_channel
      WHERE users > 0
    ),

    ranked AS (
      SELECT
        *,
        SUM(users) OVER() AS all_users,
        ROW_NUMBER() OVER (ORDER BY users DESC) AS rn
      FROM nonzero
    )

    SELECT
      assigned_channel,
      users,
      total_revenue,
      arpu,
      ROUND( (users / NULLIF(all_users,0)) * 100, 1) AS traffic_share
    FROM ranked
    ORDER BY users DESC;
    """

    try:
        df = client.query(query).to_dataframe(create_bqstorage_client=False)

        if df.empty:
            raise Exception("No channel data found in BigQuery for the specified date range")

        # ---------- Build traffic_mix cards ----------
        # Performance class: relative to median ARPU
        arpu_series = df["arpu"].fillna(0)
        median_arpu = float(arpu_series.median()) if not arpu_series.empty else 0
        hi_cut = median_arpu * 1.25  # >125% of median -> high
        lo_cut = median_arpu * 0.75  # <75%  of median -> low

        def perf_class(arpu):
            if arpu is None: return "medium-performer"
            if arpu > hi_cut: return "high-performer"
            if arpu < lo_cut: return "low-performer"
            return "medium-performer"

        traffic_mix = []
        total_users = int(df["users"].sum())
        for _, row in df.iterrows():
            traffic_mix.append({
                "name": row["assigned_channel"] or "Unassigned",
                "users": f"{int(row['users']):,}",
                "traffic_share": float(row["traffic_share"] or 0.0),
                "arpu": float(row["arpu"] or 0.0),
                "performance_class": perf_class(row["arpu"])
            })

        # ---------- Build chart data ----------
        labels  = [str(x) for x in df["assigned_channel"].tolist()]
        traffic = [int(x)  for x in df["users"].tolist()]
        arpu    = [float(x) for x in df["arpu"].tolist()]

        channel_performance = {
            "labels": labels,
            "traffic": traffic,
            "arpu": arpu
        }

        # ---------- Optional insights / summary ----------
        # (These can replace your hard-coded text in the template.)
        avg_arpu = float(df["arpu"].mean()) if not df["arpu"].empty else 0.0
        top_row = df.sort_values("total_revenue", ascending=False).iloc[0]
        top_name = str(top_row["assigned_channel"])
        top_rev_share = (float(top_row["total_revenue"]) / float(df["total_revenue"].sum())) * 100 if df["total_revenue"].sum() else 0.0

        insights = [
            f"{top_name} leads by revenue with ~{top_rev_share:.0f}% share.",
            f"Median ARPU ≈ R{median_arpu:.0f}; average ARPU ≈ R{avg_arpu:.0f}.",
            f"Total active channels: {len(df)}; total users: {total_users:,}."
        ]

        summary = {
            "total_channels": int(len(df)),
            "avg_arpu": round(avg_arpu, 0),
            "top_channel": top_name,
            "top_channel_rev_share": round(top_rev_share, 0)
        }

        return {
            "traffic_mix": traffic_mix,
            "channel_performance": channel_performance,
            "insights": insights,
            "summary": summary
        }

    except Exception as e:
        print(f"Channel performance query failed: {e}")
        raise Exception(f"Failed to fetch channel metrics from BigQuery: {e}")

def get_device_metrics_from_bigquery(date_from, date_to, min_traffic_pct=0.1):
    """
    Fetch device + screen resolution performance from GA4 BigQuery export.
    Filters out devices whose traffic share is below `min_traffic_pct` (default 0.1%).
    Returns a dict compatible with your device_analysis.html template.
    """
    client = get_bigquery_client()

    start_suffix = datetime.strptime(date_from, '%Y-%m-%d').strftime('%Y%m%d')
    end_suffix   = datetime.strptime(date_to,   '%Y-%m-%d').strftime('%Y%m%d')

    query = f"""
    -- 1) Base rows with normalized revenue + engagement for device & resolution
    WITH base AS (
      SELECT
        device.category AS device_category,
        (SELECT ep.value.string_value FROM UNNEST(event_params) ep WHERE ep.key = 'screen_resolution') AS screen_resolution_raw,
        SAFE_CAST((SELECT ep.value.int_value FROM UNNEST(event_params) ep WHERE ep.key = 'screen_width')  AS INT64) AS _w,
        SAFE_CAST((SELECT ep.value.int_value FROM UNNEST(event_params) ep WHERE ep.key = 'screen_height') AS INT64) AS _h,
        user_pseudo_id,
        event_name,
        CASE
          WHEN event_name = 'purchase' THEN COALESCE(
            (
              SELECT MAX(
                CAST(
                  COALESCE(ep.value.double_value,
                           CAST(ep.value.int_value AS FLOAT64),
                           SAFE_CAST(ep.value.string_value AS FLOAT64)) AS FLOAT64
                )
              )
              FROM UNNEST(event_params) ep
              WHERE ep.key IN ('value','value_in_usd','purchase_revenue_in_usd')
            ),
            0.0
          )
          ELSE 0.0
        END AS revenue_value,
        CASE
          WHEN event_name = 'user_engagement' THEN SAFE_DIVIDE(
            (SELECT ep.value.int_value FROM UNNEST(event_params) ep WHERE ep.key = 'engagement_time_msec'),
            1000.0
          )
          ELSE NULL
        END AS engagement_seconds
      FROM `{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
        AND device.category IS NOT NULL
    ),

    base_norm AS (
      SELECT
        device_category,
        user_pseudo_id,
        event_name,
        revenue_value,
        engagement_seconds,
        COALESCE(
          SAFE_CAST(REGEXP_EXTRACT(screen_resolution_raw, r'^(\\d{{2,4}})') AS INT64),
          _w
        ) AS screen_width,
        CASE
          WHEN screen_resolution_raw IS NOT NULL AND screen_resolution_raw != '' THEN screen_resolution_raw
          WHEN _w IS NOT NULL AND _h IS NOT NULL THEN CONCAT(CAST(_w AS STRING), 'x', CAST(_h AS STRING))
          ELSE '(not set)'
        END AS screen_resolution
      FROM base
    ),

    -- Device aggregation
    device_agg AS (
      SELECT
        LOWER(device_category) AS device_category_lc,
        COUNT(DISTINCT user_pseudo_id) AS users,
        COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_pseudo_id END) AS purchasers,
        SUM(revenue_value) AS total_revenue,
        AVG(engagement_seconds) AS avg_engagement_seconds
      FROM base_norm
      GROUP BY LOWER(device_category)
    ),

    device_metrics AS (
      SELECT
        device_category_lc AS device_category,
        users,
        purchasers,
        SAFE_MULTIPLY(SAFE_DIVIDE(purchasers, NULLIF(users,0)), 100) AS conversion_rate,
        ROUND(SAFE_DIVIDE(total_revenue, NULLIF(users,0)), 2) AS arpu,
        avg_engagement_seconds
      FROM device_agg
    ),

    device_with_share AS (
      SELECT
        *,
        ROUND((users / NULLIF(SUM(users) OVER(),0)) * 100, 3) AS traffic_share -- 3dp; we'll threshold in Python
      FROM device_metrics
    ),

    -- Make width bands
    bands AS (
      SELECT
        *,
        CASE
          WHEN screen_width IS NULL THEN '(not set)'
          WHEN screen_width < 360 THEN '<360 px'
          WHEN screen_width BETWEEN 360 AND 399 THEN '360–399 px'
          WHEN screen_width BETWEEN 400 AND 479 THEN '400–479 px'
          WHEN screen_width BETWEEN 480 AND 599 THEN '480–599 px'
          WHEN screen_width BETWEEN 600 AND 767 THEN '600–767 px'
          WHEN screen_width BETWEEN 768 AND 1023 THEN '768–1023 px'
          WHEN screen_width BETWEEN 1024 AND 1279 THEN '1024–1279 px'
          WHEN screen_width BETWEEN 1280 AND 1439 THEN '1280–1439 px'
          WHEN screen_width BETWEEN 1440 AND 1599 THEN '1440–1599 px'
          ELSE '1600+ px'
        END AS resolution_band
      FROM base_norm
    ),

    res_agg AS (
      SELECT
        resolution_band,
        LOWER(device_category) AS device_type_lc,
        COUNT(DISTINCT user_pseudo_id) AS users,
        COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_pseudo_id END) AS purchasers,
        SUM(revenue_value) AS total_revenue
      FROM bands
      GROUP BY resolution_band, LOWER(device_category)
    ),

    res_metrics AS (
      SELECT
        resolution_band,
        device_type_lc,
        users,
        SAFE_MULTIPLY(SAFE_DIVIDE(purchasers, NULLIF(users,0)), 100) AS conversion_rate,
        ROUND(SAFE_DIVIDE(total_revenue, NULLIF(users,0)), 2) AS arpu
      FROM res_agg
      WHERE users > 0
    ),

    res_ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER(ORDER BY users DESC) AS rn,
        SUM(users) OVER() AS all_users
      FROM res_metrics
    ),

    -- Conversion funnel by device (views -> atc -> purchase)
    views AS (
      SELECT LOWER(device_category) AS dc, COUNT(DISTINCT user_pseudo_id) AS view_users
      FROM base_norm
      WHERE event_name = 'view_item'
      GROUP BY LOWER(device_category)
    ),
    atc AS (
      SELECT LOWER(device_category) AS dc, COUNT(DISTINCT user_pseudo_id) AS atc_users
      FROM base_norm
      WHERE event_name = 'add_to_cart'
      GROUP BY LOWER(device_category)
    ),
    buyers AS (
      SELECT LOWER(device_category) AS dc, COUNT(DISTINCT user_pseudo_id) AS purchase_users
      FROM base_norm
      WHERE event_name = 'purchase'
      GROUP BY LOWER(device_category)
    ),

    funnel_rates AS (
      SELECT
        COALESCE(v.dc, a.dc, b.dc) AS device_category,
        COALESCE(v.view_users, 0)   AS view_users,
        ROUND(SAFE_MULTIPLY(SAFE_DIVIDE(COALESCE(a.atc_users,0), NULLIF(v.view_users,0)), 100), 2) AS atc_rate,
        ROUND(SAFE_MULTIPLY(SAFE_DIVIDE(COALESCE(b.purchase_users,0), NULLIF(v.view_users,0)), 100), 2) AS purchase_rate
      FROM views v
      FULL OUTER JOIN atc a USING (dc)
      FULL OUTER JOIN buyers b USING (dc)
    )

    SELECT
      -- device cards
      (SELECT ARRAY_AGG(STRUCT(
          device_category AS device_category,
          users AS users,
          traffic_share AS traffic_share,
          arpu AS arpu,
          conversion_rate AS conversion_rate,
          avg_engagement_seconds AS avg_engagement_seconds
        ) ORDER BY users DESC)
       FROM device_with_share) AS device_cards,

      -- top bands (not raw resolutions)
      (SELECT ARRAY_AGG(STRUCT(
          resolution_band AS resolution_band,
          device_type_lc AS device_type_lc,
          ROUND((users / NULLIF(all_users,0)) * 100, 3) AS traffic_share,
          arpu AS arpu,
          ROUND(conversion_rate, 2) AS conversion_rate
        ) ORDER BY users DESC LIMIT 50)
       FROM res_ranked) AS top_resolutions,

      -- funnel rows per device
    (SELECT ARRAY_AGG(STRUCT(
        device_category AS device_category,
        view_users     AS view_users,
        atc_rate       AS atc_rate,
        purchase_rate  AS purchase_rate
    )) FROM funnel_rates) AS funnel_rows
    ;
    """

    try:
        # Pull data
        df = client.query(query).to_dataframe(create_bqstorage_client=False)
        if df.empty:
            raise Exception("No device data found for the specified date range")

        row = df.iloc[0]

        # --- Normalize arrays from the single-row SELECT ---
        def _to_list(val):
            if val is None:
                return []
            if hasattr(val, "tolist"):
                return val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val]

        device_cards = _to_list(row.get("device_cards"))
        res_rows     = _to_list(row.get("top_resolutions"))
        funnel_rows  = _to_list(row.get("funnel_rows"))

        # --- Build device categories (cards) ---
        categories = []
        device_icons = {'desktop': '💻', 'mobile': '📱', 'tablet': '📲', 'tv': '📺'}
        for card in device_cards:
            # card can be dict or tuple
            dc    = (card.get("device_category")       if isinstance(card, dict) else card[0]) or "(not set)"
            users =  card.get("users")                 if isinstance(card, dict) else card[1]
            share =  card.get("traffic_share")         if isinstance(card, dict) else card[2]
            arpu  =  card.get("arpu")                  if isinstance(card, dict) else card[3]
            conv  =  card.get("conversion_rate")       if isinstance(card, dict) else card[4]
            eng_s =  card.get("avg_engagement_seconds")if isinstance(card, dict) else card[5]

            dc_lc = str(dc).lower()
            icon  = device_icons.get(dc_lc, '📱')

            categories.append({
                'name': str(dc).title(),
                'icon': icon,
                'traffic': round(float(share or 0.0), 2),
                'arpu': round(float(arpu or 0.0), 2),
                'conversion': float(conv or 0.0),  # percent number
                'engagement': f"{int(float(eng_s or 0.0) // 60)}m {int(float(eng_s or 0.0) % 60)}s"
            })

        # --- FILTER: drop devices with < min_traffic_pct share ---
        categories = [c for c in categories if c['traffic'] >= float(min_traffic_pct)]

        # Build a set of remaining device names (lowercase) to filter funnel + resolutions
        kept_devices_lc = {c['name'].lower() for c in categories}

        # --- Build funnel dict expected by your chart: [100, atc%, purchase%] per device ---
        funnel = {'desktop': [], 'mobile': [], 'tablet': []}
        for f in funnel_rows:
            dc = (f.get("device_category") if isinstance(f, dict) else f[0]) or ""
            atc_rate      = f.get("atc_rate")      if isinstance(f, dict) else f[2]
            purchase_rate = f.get("purchase_rate") if isinstance(f, dict) else f[3]

            dc_key = str(dc).lower()
            # Only keep funnels for devices we kept
            if dc_key in kept_devices_lc:
                arr = [100.0, float(atc_rate or 0.0), float(purchase_rate or 0.0)]
                if   dc_key.startswith('desk'): funnel['desktop'] = arr
                elif dc_key.startswith('mob'):  funnel['mobile']  = arr
                elif dc_key.startswith('tab'):  funnel['tablet']  = arr
                # ignore tv/other

        # --- Build resolutions (bands) and filter to kept devices & min share ---
        resolutions = []
        for r in res_rows:
            band  = (r.get("resolution_band") if isinstance(r, dict) else r[0]) or "(not set)"
            dtype = (r.get("device_type_lc")  if isinstance(r, dict) else r[1]) or "(not set)"
            share =  r.get("traffic_share")   if isinstance(r, dict) else r[2]
            arpu  =  r.get("arpu")            if isinstance(r, dict) else r[3]
            cvr   =  r.get("conversion_rate") if isinstance(r, dict) else r[4]

            dtype_title = str(dtype).title()
            # Filter: only show resolutions for devices we kept and with enough share
            if str(dtype).lower() in kept_devices_lc and float(share or 0.0) >= float(min_traffic_pct):
                label = band if band != "(not set)" else f"{dtype_title} (band n/a)"
                resolutions.append({
                    'resolution': label,
                    'device_type': dtype_title,
                    'traffic_share': round(float(share or 0.0), 1),
                    'arpu': round(float(arpu or 0.0), 2),
                    'conversion': round(float(cvr or 0.0), 1),
                })

        # Sort outputs nicely
        categories.sort(key=lambda x: x['traffic'], reverse=True)
        resolutions.sort(key=lambda x: x['traffic_share'], reverse=True)

        return {
            'categories': categories,
            'funnel': funnel,
            'resolutions': resolutions
        }

    except Exception as e:
        print(f"Error fetching device data: {e}")
        raise Exception(f"Failed to fetch device metrics from BigQuery: {e}")

def get_cro_series_from_bq(date_from: str, date_to: str):
    """
    Returns a dict with daily desktop purchase CVR (%).
    CVR is computed as (unique purchasers who viewed item) / (unique viewers) * 100.
    """
    client = get_bigquery_client()

    start_suffix = datetime.strptime(date_from, "%Y-%m-%d").strftime("%Y%m%d")
    end_suffix   = datetime.strptime(date_to,   "%Y-%m-%d").strftime("%Y%m%d")

    table = f"`{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`"

    query = f"""
    WITH base AS (
      SELECT
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS d,
        LOWER(device.category) AS dc,
        user_pseudo_id,
        event_name
      FROM {table}
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
        AND event_name IN ('view_item','purchase')
        AND device.category IS NOT NULL
    ),
    daily AS (
      SELECT
        d,
        dc,
        COUNT(DISTINCT IF(event_name = 'view_item',  user_pseudo_id, NULL)) AS view_users,
        COUNT(DISTINCT IF(event_name = 'purchase',   user_pseudo_id, NULL)) AS purchase_users
      FROM base
      GROUP BY d, dc
    )
    SELECT
      d,
      view_users,
      purchase_users
    FROM daily
    WHERE dc = 'desktop'
    ORDER BY d
    """

    df = client.query(query).to_dataframe()
    labels, values = [], []

    if not df.empty:
        df["d"] = df["d"].astype(str)
        df["view_users"] = df["view_users"].fillna(0)
        df["purchase_users"] = df["purchase_users"].fillna(0)
        df["cvr_pct"] = (df["purchase_users"] / df["view_users"]).replace([float("inf")], 0).fillna(0) * 100.0

        labels = df["d"].tolist()
        # round to 2dp for chart
        values = [round(float(v), 2) for v in df["cvr_pct"].tolist()]

    return {
        "desktop": {
            "daily": {
                "labels": labels,      # ['2025-09-08', ...]
                "values": values       # [2.13, 2.48, ...] in PERCENT
            }
        }
    }

def get_cro_mobile_series_from_bq(date_from: str, date_to: str):
    """
    Returns a dict with daily mobile purchase CVR (%).
    CVR = unique purchasers / unique viewers * 100.
    """
    client = get_bigquery_client()

    start_suffix = datetime.strptime(date_from, "%Y-%m-%d").strftime("%Y%m%d")
    end_suffix   = datetime.strptime(date_to,   "%Y-%m-%d").strftime("%Y%m%d")

    table = f"`{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`"

    query = f"""
    WITH base AS (
      SELECT
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS d,
        LOWER(device.category) AS dc,
        user_pseudo_id,
        event_name
      FROM {table}
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
        AND event_name IN ('view_item','purchase')
        AND device.category IS NOT NULL
    ),
    daily AS (
      SELECT
        d,
        dc,
        COUNT(DISTINCT IF(event_name = 'view_item',  user_pseudo_id, NULL)) AS view_users,
        COUNT(DISTINCT IF(event_name = 'purchase',   user_pseudo_id, NULL)) AS purchase_users
      FROM base
      GROUP BY d, dc
    )
    SELECT d, view_users, purchase_users
    FROM daily
    WHERE dc = 'mobile'
    ORDER BY d
    """

    df = client.query(query).to_dataframe()
    labels, values = [], []

    if not df.empty:
        df["d"] = df["d"].astype(str)
        df["view_users"] = df["view_users"].fillna(0)
        df["purchase_users"] = df["purchase_users"].fillna(0)
        df["cvr_pct"] = (df["purchase_users"] / df["view_users"]).replace([float("inf")], 0).fillna(0) * 100.0
        labels = df["d"].tolist()
        values = [round(float(v), 2) for v in df["cvr_pct"].tolist()]

    return {
        "mobile": {
            "daily": {
                "labels": labels,
                "values": values
            }
        }
    }

def get_ais_metrics_from_bigquery(date_from, date_to):
    """Calculate Audience Intent Score (AIS) from GA4 BigQuery export with clean syntax."""

    if not date_from or not date_to:
        raise ValueError(f"Invalid date parameters: from={date_from}, to={date_to}")

    client = get_bigquery_client()
    if not client:
        raise Exception("BigQuery client not initialized for AIS metrics")

    start_suffix = datetime.strptime(date_from, '%Y-%m-%d').strftime('%Y%m%d')
    end_suffix   = datetime.strptime(date_to,   '%Y-%m-%d').strftime('%Y%m%d')

    table = f"`{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`"

    # Clean, compact AIS query:
    query = f"""
    -- 1) Base rows: extract channel, user_type, revenue, engagement, page_view, session_id
    WITH base AS (
      SELECT
        user_pseudo_id,

        CASE
          WHEN traffic_source.source = '(direct)' OR traffic_source.source IS NULL OR traffic_source.source = 'None' THEN 'Direct'
          WHEN REGEXP_CONTAINS(traffic_source.source, r'cross-network') THEN 'Cross-network'
          WHEN traffic_source.medium IN ('cpc','ppc','paidsearch')
               AND traffic_source.source IN ('google','bing','yahoo','baidu','duckduckgo') THEN 'Paid Search'
          WHEN traffic_source.medium = 'organic' THEN 'Organic Search'
          WHEN (LOWER(traffic_source.source) LIKE '%meta%'
             OR  LOWER(traffic_source.source) LIKE '%facebook%'
             OR  LOWER(traffic_source.source) LIKE '%instagram%'
             OR  LOWER(traffic_source.source) = 'ig'
             OR  LOWER(traffic_source.source) = 'fb')
             AND (LOWER(traffic_source.medium) LIKE '%paid%'
               OR LOWER(traffic_source.medium) LIKE '%ad%'
               OR LOWER(traffic_source.medium) LIKE '%cpc%'
               OR LOWER(traffic_source.medium) LIKE '%ppc%'
               OR LOWER(traffic_source.medium) LIKE '%instagram%'
               OR LOWER(traffic_source.medium) LIKE '%meta%'
               OR LOWER(traffic_source.medium) LIKE '%carousel%'
               OR LOWER(traffic_source.medium) LIKE '%whatsapp%') THEN 'Paid Social'
          WHEN (LOWER(traffic_source.source) LIKE '%facebook%'
             OR  LOWER(traffic_source.source) LIKE '%instagram%'
             OR  LOWER(traffic_source.source) LIKE '%twitter%'
             OR  LOWER(traffic_source.source) LIKE '%linkedin%'
             OR  LOWER(traffic_source.source) LIKE '%pinterest%'
             OR  LOWER(traffic_source.source) LIKE '%tiktok%'
             OR  LOWER(traffic_source.source) = 'meta'
             OR  LOWER(traffic_source.source) = 'socialorganic'
             OR  LOWER(traffic_source.source) = 'social_organic')
             AND (LOWER(traffic_source.medium) IN ('social','socialorganic','social-network','social-media','sm','social organic','meta')
               OR traffic_source.medium IS NULL) THEN 'Organic Social'
          WHEN LOWER(traffic_source.medium) = 'email'
             OR LOWER(traffic_source.source) LIKE '%newsletter%'
             OR LOWER(traffic_source.source) = 'netcore' THEN 'Email'
          WHEN traffic_source.medium = 'affiliate' THEN 'Affiliates'
          WHEN traffic_source.medium = 'referral' THEN 'Referral'
          WHEN traffic_source.medium IN ('display','cpm','banner') THEN 'Display'
          WHEN traffic_source.source LIKE '%shopping%' AND traffic_source.medium = 'organic' THEN 'Organic Shopping'
          WHEN traffic_source.source LIKE '%shopping%' AND traffic_source.medium IN ('cpc','ppc','paid') THEN 'Paid Shopping'
          WHEN traffic_source.source = 'Data Not Available'
            OR traffic_source.medium = 'Data Not Available' THEN 'Unassigned'
          ELSE 'Unassigned'
        END AS channel,

        CASE
          WHEN (SELECT p.value.int_value FROM UNNEST(event_params) p WHERE p.key = 'ga_session_number') = 1
            THEN 'new'
          ELSE 'returning'
        END AS user_type,

        event_name,

        -- revenue per event (GA4 'value' param, common pattern)
        COALESCE(
          (SELECT CAST(p.value.double_value AS FLOAT64)
           FROM UNNEST(event_params) p
           WHERE p.key = 'value'),
          0.0
        ) AS revenue_value,

        -- engagement seconds (from engagement_time_msec when present)
        CASE
          WHEN event_name = 'user_engagement' THEN SAFE_DIVIDE(
            (SELECT p.value.int_value FROM UNNEST(event_params) p WHERE p.key = 'engagement_time_msec'),
            1000.0
          )
          ELSE NULL
        END AS engagement_seconds,

        -- page_view flag for pages/session
        CASE WHEN event_name = 'page_view' THEN 1 ELSE 0 END AS page_view,

        -- session id for proper sessions
        (SELECT p.value.int_value FROM UNNEST(event_params) p WHERE p.key = 'ga_session_id') AS session_id
      FROM {table}
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
    ),

    -- 2) Aggregate per channel
    channel_agg AS (
      SELECT
        channel,

        COUNT(DISTINCT user_pseudo_id) AS total_users,

        COUNT(DISTINCT CASE WHEN user_type = 'new' THEN user_pseudo_id END) AS new_users,
        COUNT(DISTINCT CASE WHEN user_type = 'returning' THEN user_pseudo_id END) AS returning_users,

        COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_pseudo_id END) AS purchasers,
        COUNT(DISTINCT CASE WHEN event_name = 'purchase' AND user_type = 'new' THEN user_pseudo_id END) AS new_purchasers,
        COUNT(DISTINCT CASE WHEN event_name = 'purchase' AND user_type = 'returning' THEN user_pseudo_id END) AS returning_purchasers,

        -- purchase event count (not unique users)
        COUNTIF(event_name = 'purchase') AS purchase_events,

        -- revenue from purchase events
        SUM(CASE WHEN event_name = 'purchase' THEN revenue_value ELSE 0 END) AS total_revenue,

        -- avg engagement where we have a value
        AVG(engagement_seconds) AS avg_engagement_seconds,

        -- total page_views and session count
        SUM(page_view) AS total_page_views,
        COUNT(DISTINCT CONCAT(CAST(user_pseudo_id AS STRING), '-', CAST(session_id AS STRING))) AS sessions
      FROM base
      GROUP BY channel
    ),

    -- 3) overall totals for mixes
    overall AS (
      SELECT
        SUM(total_users) AS all_users,
        SUM(new_users) AS all_new_users,
        SUM(returning_users) AS all_returning_users,
        SUM(purchasers) AS all_purchasers,
        SUM(new_purchasers) AS all_new_purchasers,
        SUM(returning_purchasers) AS all_returning_purchasers,
        SUM(total_revenue) AS all_revenue,
        AVG(avg_engagement_seconds) AS overall_avg_engagement,
        SUM(total_page_views) AS all_page_views,
        SUM(sessions) AS all_sessions
      FROM channel_agg
    )

    SELECT
      ca.channel,
      ca.total_users,
      ca.new_users,
      ca.returning_users,
      ca.purchasers,
      ca.new_purchasers,
      ca.returning_purchasers,
      ca.purchase_events,
      ca.total_revenue,
      ca.avg_engagement_seconds,
      ca.total_page_views,
      ca.sessions,
      -- derived rates
      SAFE_DIVIDE(ca.purchasers, ca.total_users) * 100 AS conversion_rate,
      SAFE_DIVIDE(ca.new_purchasers, ca.new_users) * 100 AS new_user_cvr,
      SAFE_DIVIDE(ca.returning_purchasers, ca.returning_users) * 100 AS returning_user_cvr,
      SAFE_DIVIDE(ca.total_revenue, ca.total_users) AS arpu,
      SAFE_DIVIDE(ca.total_page_views, ca.sessions) AS pages_per_session,

      -- pull overall mixes once for easy consumption
      ov.all_users,
      ov.all_new_users,
      ov.all_returning_users,
      ov.all_purchasers,
      ov.all_new_purchasers,
      ov.all_returning_purchasers,
      ov.all_revenue,
      ov.overall_avg_engagement,
      ov.all_page_views,
      ov.all_sessions
    FROM channel_agg ca
    CROSS JOIN overall ov
    WHERE ca.total_users > 0
    ORDER BY ca.total_users DESC
    ;
    """

    try:
        df = client.query(query).to_dataframe()

        if df.empty:
            raise Exception("AIS query returned no rows for the given date range")

        # Overall values (use first row after ordering by users desc)
        overall = df.iloc[0]

        # --- Overall metrics across the selected window (not weekly) ---
        sql_more = """
        SELECT
        SUM(users)    AS users,
        SUM(purchasers) AS purchasers,
        SUM(revenue) AS revenue,
        SUM(page_views) AS page_views,
        SUM(sessions)   AS sessions,
        -- weighted avg engagement by users across day×channel rows
        SUM(avg_engagement_seconds * users) / NULLIF(SUM(users), 0) AS w_engagement_seconds
        FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
        WHERE event_date BETWEEN @from AND @to
        """
        df_more = client.query(
            sql_more,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("from", "DATE", date_from),
                    bigquery.ScalarQueryParameter("to",   "DATE", date_to),
                ]
            )
        ).to_dataframe().iloc[0]

        total_users    = float(df_more.get("users") or 0)
        total_purch    = float(df_more.get("purchasers") or 0)
        total_revenue  = float(df_more.get("revenue") or 0)
        total_page_views = float(df_more.get("page_views") or 0)
        total_sessions   = float(df_more.get("sessions") or 0)
        w_eng_secs       = float(df_more.get("w_engagement_seconds") or 0)

        overall_cvr      = (total_purch / total_users * 100.0) if total_users > 0 else 0.0
        revenue_per_user = (total_revenue / total_users) if total_users > 0 else 0.0
        pages_per_sess   = (total_page_views / total_sessions) if total_sessions > 0 else 0.0

        def fmt_time(seconds):
            s = int(seconds or 0)
            return f"{s//60}m {s%60:02d}s"

        engagement_metrics = [
            {"label": "Avg. Engagement Time", "value": fmt_time(w_eng_secs)},
            {"label": "Pages per Session",    "value": f"{pages_per_sess:.2f}"},
            {"label": "Overall Bounce Rate",  "value": "—"},   # not wired yet
            {"label": "New vs Returning Mix", "value": "—"},   # needs schema upgrade
            {"label": "Returning User Multiplier", "value": "—"},
        ]

        performance_metrics = [
            {"label": "Purchase Conversion Rate",    "value": f"{overall_cvr:.2f}%"},
            {"label": "New User Purchase CVR",       "value": "—"},
            {"label": "Returning User Purchase CVR", "value": "—"},
            {"label": "Revenue per User",            "value": f"R{revenue_per_user:.2f}"},
            {"label": "Baseline Channel",            "value": "N/A"},
        ]

        # Build channel list and AIS score
        baseline_cvr = float(df['conversion_rate'].max()) if not df['conversion_rate'].isnull().all() else 0.0
        channels = []
        overall_score = 0.0

        for _, row in df.iterrows():
            ch_cvr = float(row['conversion_rate'] or 0.0)
            intent_score = min(100.0, (ch_cvr / baseline_cvr) * 100.0) if baseline_cvr > 0 else 0.0
            traffic_share = float(row['total_users'] / overall['all_users'] * 100.0) if overall['all_users'] else 0.0
            impact = (intent_score * traffic_share) / 100.0
            overall_score += impact

            if intent_score >= 80:
                intent_level, intent_label = 'high', 'High Intent'
            elif intent_score >= 50:
                intent_level, intent_label = 'medium', 'Medium Intent'
            else:
                intent_level, intent_label = 'low', 'Low'

            channels.append({
                'name': str(row['channel']),
                'score': round(intent_score, 1),
                'intent_level': intent_level,
                'intent_label': intent_label,
                'traffic_share': round(traffic_share, 1),
                'impact': round(impact, 1)
            })

        channels = sorted(channels, key=lambda x: x['score'], reverse=True)

        if overall_score >= 75:
            quality_label = 'Excellent'
        elif overall_score >= 60:
            quality_label = 'Good'
        elif overall_score >= 40:
            quality_label = 'Moderate'
        else:
            quality_label = 'Needs Improvement'

        time_series = {
            'labels': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'scores': [overall_score * 0.9, overall_score * 0.95, overall_score * 1.02, overall_score]
        }

        strategies = []
        if channels:
            strategies.append({
                'label': 'TOP PERFORMER',
                'title': channels[0]['name'],
                'description': f'Your best channel with {channels[0]["score"]:.0f} intent score.',
                'intent_class': 'high-intent'
            })

        return {
            'score': round(overall_score, 1),
            'quality_label': quality_label,
            'interpretation': f'{channels[0]["name"]} ({channels[0]["traffic_share"]:.1f}% of users) contributes the most to your score.' if channels else 'No channels found.',
            'channels': channels[:8],
            'time_series': time_series,
            'engagement_metrics': engagement_metrics,
            'performance_metrics': performance_metrics,
            'user_segments': [
                {
                    'channel': str(row['channel']),
                    'new_users': f"{int(row['new_users']):,}",
                    'returning': f"{int(row['returning_users']):,}",
                    'new_cvr': f"{float(row['new_user_cvr'] or 0.0):.2f}",
                    'ret_cvr': f"{float(row['returning_user_cvr'] or 0.0):.2f}",
                    'new_mix': int((float(row['new_users']) / float(row['total_users']) * 100.0) if row['total_users'] else 0.0),
                    'ret_mix': int((float(row['returning_users']) / float(row['total_users']) * 100.0) if row['total_users'] else 0.0)
                }
                for _, row in df.iterrows()
            ],
            'strategies': strategies
        }

    except Exception as e:
        print(f"❌ Error calculating AIS: {e}")
        import traceback; traceback.print_exc()
        raise

# Store the analyzer globally for AI recommendations
current_analyzer = None

def get_merchandise_data(date_from=None, date_to=None):
    """Fetch and analyze merchandise data from BigQuery via MerchandiseAnalyzerBQ."""
    if MerchandiseAnalyzerBQ is None:
        raise ImportError(
            "frankmerch_bigquery.py not found or failed to import. "
            "Ensure the file exists and is importable from this directory."
        )

    print("Loading merchandise data from BigQuery...")

    analyzer = MerchandiseAnalyzerBQ(
        project_id=PROJECT_ID,
        credentials_path='bigquery-credentials.json'
    )

    success = analyzer.load_from_bigquery(
        date_from=date_from,
        date_to=date_to,
        ga4_property_id=GA4_PROPERTY_ID,
        min_views=150
    )
    if not success:
        raise Exception("Failed to load merchandise data from BigQuery")

    analyzer.calculate_metrics()
    analyzer.segment_products()
    analyzer.identify_opportunities()

    insights, recommendations, products = analyzer.prepare_dashboard_data()

        # ---- SANITIZE: convert numpy/pandas objects to plain Python so Jinja is happy ----
    import numpy as np

    def _coerce_scalar(x):
        # numpy scalars -> python
        if isinstance(x, np.generic):
            return x.item()
        # pandas Series with a single value -> that value
        try:
            import pandas as pd
            if isinstance(x, pd.Series):
                if len(x) == 1:
                    return _coerce_scalar(x.iloc[0])
                else:
                    return [_coerce_scalar(v) for v in x.tolist()]
        except Exception:
            pass
        return x

    def _coerce_value(v):
        if isinstance(v, dict):
            return {k: _coerce_value(w) for k, w in v.items()}
        elif isinstance(v, (list, tuple)):
            return [ _coerce_value(w) for w in v ]
        else:
            return _coerce_scalar(v)

    insights        = _coerce_value(insights)
    recommendations = _coerce_value(recommendations)
    products        = _coerce_value(products)

    # Helpful trace so you can see if anything odd survived
    try:
        print("🧼 sanitize check:",
              "insights types ->", {k: type(v).__name__ for k,v in (insights.items() if isinstance(insights, dict) else [])})
        if isinstance(products, list) and products:
            print("🧼 product sample types ->", {k: type(v).__name__ for k,v in products[0].items()})
    except Exception:
        pass
    # ------------------------------------------------------------------------------

    # Ensure keys exist so the template never crashes
    for key in ['homepage_heroes', 'email_features', 'promotion_needed', 'increase_exposure']:
        recommendations.setdefault(key, [])
        # --- Normalize everything to plain Python types (no pandas objects) ---
    import numpy as np
    import pandas as pd

    def _to_scalar(x):
        """Convert numpy/pandas scalars to plain Python scalars."""
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
        return x

    def _deep_clean(obj):
        """Recursively convert pandas/numpy objects to JSON-safe Python types."""
        if isinstance(obj, dict):
            return {str(k): _deep_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_deep_clean(v) for v in obj]
        if hasattr(obj, "to_dict") and not isinstance(obj, pd.Series):
            # DataFrame-like: convert to records
            try:
                return obj.to_dict(orient="records")
            except Exception:
                pass
        if isinstance(obj, pd.Series):
            # Make Series into a plain dict
            return {str(k): _deep_clean(_to_scalar(v)) for k, v in obj.to_dict().items()}
        if isinstance(obj, (np.generic, pd.Timestamp)):
            return _to_scalar(obj)
        return obj

    insights         = _deep_clean(insights)
    products         = _deep_clean(products)
    recommendations  = _deep_clean(recommendations)

    # --- Debug: log a compact type summary so we see what's going to Jinja ---
    def _type_of(x):
        try:
            return type(x).__name__
        except Exception:
            return "<?>"

    print("🧪 merchandise payload types:",
          "insights:", _type_of(insights),
          "products:", _type_of(products),
          "recs:", _type_of(recommendations))

    # Helpful spot checks (remove later if noisy)
    for k in ("total_revenue", "total_views", "total_products", "avg_conversion"):
        if isinstance(insights, dict) and k in insights:
            print(f"  - insights[{k}] ->", insights[k], "(", _type_of(insights[k]), ")")
    return insights, products, recommendations, True, analyzer

def debug_bigquery_structure():
    """Debug function to check BigQuery table structure"""
    client = get_bigquery_client()
    
    # Check if tables exist
    dataset_id = f"analytics_{GA4_PROPERTY_ID}"
    
    try:
        # List tables in the dataset
        tables = list(client.list_tables(f"{PROJECT_ID}.{dataset_id}"))
        print(f"✓ Found {len(tables)} tables in dataset {dataset_id}")
        
        # Get a recent table to check structure
        if tables:
            # Get the most recent events table
            recent_table = max([t for t in tables if t.table_id.startswith('events_')], 
                             key=lambda x: x.table_id)
            
            print(f"📊 Checking structure of: {recent_table.table_id}")
            
            # Get table schema
            table = client.get_table(recent_table.reference)
            
            print("\n🔍 Table Schema:")
            for field in table.schema:
                print(f"  - {field.name}: {field.field_type}")
                if field.name == 'ecommerce' and hasattr(field, 'fields'):
                    for subfield in field.fields:
                        print(f"    - ecommerce.{subfield.name}: {subfield.field_type}")
                        if subfield.name == 'items' and hasattr(subfield, 'fields'):
                            for item_field in subfield.fields:
                                print(f"      - ecommerce.items.{item_field.name}: {item_field.field_type}")
            
            # Test a simple query
            test_query = f"""
            SELECT 
                event_name,
                COUNT(*) as event_count
            FROM `{PROJECT_ID}.{dataset_id}.{recent_table.table_id}`
            WHERE event_name IN ('view_item', 'add_to_cart', 'purchase', 'session_start')
            GROUP BY event_name
            ORDER BY event_count DESC
            LIMIT 10
            """
            
            print(f"\n🧪 Testing simple query...")
            query_job = client.query(test_query)
            results = query_job.to_dataframe()
            
            if not results.empty:
                print("✓ Query successful! Event counts:")
                for _, row in results.iterrows():
                    print(f"  - {row['event_name']}: {row['event_count']:,}")
            else:
                raise Exception("Query returned no results")
                
        else:
            raise Exception("No events tables found in dataset")
            
    except Exception as e:
        print(f"❌ Error checking BigQuery structure: {e}")
        raise Exception(f"BigQuery structure check failed: {e}")

def test_merchandise_query():
    """Test the merchandise query with a small sample"""
    client = get_bigquery_client()

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    test_query = f"""
    WITH base AS (
      SELECT
        event_name,
        -- items are nested under ecommerce.items in GA4 export
        it.item_name AS item_name,
        user_pseudo_id,

        -- revenue per event
        COALESCE(
          (SELECT CAST(p.value.double_value AS FLOAT64)
           FROM UNNEST(event_params) p
           WHERE p.key = 'value'),
          0.0
        ) AS revenue_value
      FROM `{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_{yesterday}`
      LEFT JOIN UNNEST(ecommerce.items) AS it
      WHERE event_name IN ('view_item', 'add_to_cart', 'purchase')
        AND it.item_name IS NOT NULL
        AND it.item_name != '(not set)'
    )

    SELECT
      item_name,
      event_name,
      COUNT(DISTINCT user_pseudo_id) AS users,
      SUM(CASE WHEN event_name = 'purchase' THEN revenue_value ELSE 0 END) AS total_revenue
    FROM base
    GROUP BY item_name, event_name
    ORDER BY users DESC
    LIMIT 20
    """

    try:
        print(f"🧪 Testing merchandise query for {yesterday}...")
        results = client.query(test_query).to_dataframe()
        if not results.empty:
            print("✓ Merchandise query successful!")
            for _, row in results.iterrows():
                revenue_info = f" (Revenue: {row['total_revenue']:.2f})" if row['total_revenue'] > 0 else ""
                print(f"  - {row['item_name']}: {row['event_name']} ({row['users']}){revenue_info}")
        else:
            raise Exception("No merchandise data found")
    except Exception as e:
        print(f"❌ Merchandise query failed: {e}")
        raise Exception(f"Merchandise query test failed: {e}")

def audit_bigquery_data():
    """Audit what data is actually available"""
    client = get_bigquery_client()
    
    dataset_id = f"analytics_{GA4_PROPERTY_ID}"
    yesterday = "20250910"  # Use the date we know has data
    
    # Test 1: What events exist
    events_query = f"""
    SELECT 
        event_name,
        COUNT(*) as count
    FROM `{PROJECT_ID}.{dataset_id}.events_{yesterday}`
    GROUP BY event_name
    ORDER BY count DESC
    LIMIT 20
    """
    
    print("=== EVENTS AVAILABLE ===")
    try:
        events_df = client.query(events_query).to_dataframe()
        if events_df.empty:
            raise Exception("No events found in BigQuery")
        for _, row in events_df.iterrows():
            print(f"{row['event_name']}: {row['count']:,}")
    except Exception as e:
        print(f"Error: {e}")
        raise Exception(f"Events audit failed: {e}")
    
    # Test 2: Check ecommerce structure
    ecommerce_query = f"""
    SELECT 
        event_name,
        COUNT(*) as total_events,
        COUNT(ecommerce) as events_with_ecommerce,
        COUNT(items) as events_with_items
    FROM `{PROJECT_ID}.{dataset_id}.events_{yesterday}`
    WHERE event_name IN ('purchase', 'add_to_cart', 'view_item')
    GROUP BY event_name
    """
    
    print("\n=== ECOMMERCE DATA CHECK ===")
    try:
        ecom_df = client.query(ecommerce_query).to_dataframe()
        if ecom_df.empty:
            raise Exception("No ecommerce events found")
        for _, row in ecom_df.iterrows():
            print(f"{row['event_name']}: {row['total_events']} total, {row['events_with_ecommerce']} with ecommerce, {row['events_with_items']} with items")
    except Exception as e:
        print(f"Error: {e}")
        raise Exception(f"Ecommerce audit failed: {e}")
    
    # Test 3: Sample actual data structure
    sample_query = f"""
    SELECT 
        event_name,
        ecommerce,
        items,
        device,
        traffic_source
    FROM `{PROJECT_ID}.{dataset_id}.events_{yesterday}`
    WHERE event_name = 'purchase'
    LIMIT 3
    """
    
    print("\n=== SAMPLE DATA STRUCTURE ===")
    try:
        sample_df = client.query(sample_query).to_dataframe()
        if sample_df.empty:
            raise Exception("No sample purchase events found")
        for idx, row in sample_df.iterrows():
            print(f"Sample {idx + 1}:")
            print(f"  Event: {row['event_name']}")
            print(f"  Ecommerce: {str(row['ecommerce'])[:100]}...")
            print(f"  Items: {str(row['items'])[:100]}...")
            print(f"  Device: {row['device']}")
            print("---")
    except Exception as e:
        print(f"Error: {e}")
        raise Exception(f"Sample data audit failed: {e}")

def get_channel_metrics_from_daily(date_from, date_to):
    print(f"DEBUG: Starting function with dates {date_from} to {date_to}")

    """
    Read daily channel KPIs from carrol-boyes-ga4.frankie_derived.daily_channel_kpis
    and shape them for the Channel Performance dashboard.
    """
    client = get_bigquery_client()

    from datetime import datetime
    from google.cloud import bigquery

        # 1) Convert UI strings -> DATE  (keep the try/except *only* around strptime)
    try:
        d_from = datetime.strptime(date_from, "%Y-%m-%d").date()
        d_to   = datetime.strptime(date_to,   "%Y-%m-%d").date()
    except Exception as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Details: {e}")

    # 2) Parameterized query (use a distinct name to avoid any accidental shadowing)
    query_sql = """
    SELECT
      assigned_channel AS channel,
      SUM(users)       AS users,
      SUM(purchasers)  AS purchasers,
      SUM(revenue)     AS revenue
    FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
    WHERE event_date BETWEEN @from AND @to
    GROUP BY assigned_channel
    HAVING users > 0
    ORDER BY users DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("from", "DATE", d_from),
            bigquery.ScalarQueryParameter("to",   "DATE", d_to),
        ]
    )

    df = client.query(query_sql, job_config=job_config).to_dataframe()

    if df.empty:
        raise Exception("No channel data in daily_channel_kpis for this date range")

    # 3) Derived metrics
    import numpy as np
    df["arpu"] = np.where(df["users"] > 0, df["revenue"] / df["users"], 0.0)

    total_users   = float(df["users"].sum())
    total_revenue = float(df["revenue"].sum())

    df["traffic_share"] = np.where(total_users > 0, (df["users"] / total_users) * 100.0, 0.0)

    # Simple border-colour classes by ARPU terciles
    q_low  = float(df["arpu"].quantile(0.33))
    q_high = float(df["arpu"].quantile(0.66))
    def perf_class(x):
        if x >= q_high: return "high-performer"
        if x >= q_low:  return "medium-performer"
        return "low-performer"
    df["performance_class"] = df["arpu"].apply(perf_class)

    # 4) Cards (traffic mix grid)
    traffic_mix = []
    for _, r in df.iterrows():
        traffic_mix.append({
            "name": str(r["channel"]),
            "users": f'{int(r["users"]):,}',
            "traffic_share": round(float(r["traffic_share"]), 1),
            "arpu": round(float(r["arpu"]), 2),
            "performance_class": r["performance_class"]
        })

    # 5) Chart data (aligned arrays)
    labels  = [str(x) for x in df["channel"].tolist()]
    traffic = [int(x) for x in df["users"].tolist()]
    arpu    = [round(float(x), 2) for x in df["arpu"].tolist()]

    # 6) Summary + insights
    top_row = df.loc[df["revenue"].idxmax()]
    top_channel = str(top_row["channel"])
    top_share   = round((float(top_row["revenue"]) / total_revenue) * 100.0, 1) if total_revenue > 0 else 0.0

    summary = {
        "total_channels": int(df.shape[0]),
        "avg_arpu": round(float(total_revenue / total_users) if total_users else 0.0, 2),
        "top_channel": top_channel,
        "top_channel_rev_share": top_share
    }

    # A couple of friendly sentences for the purple insights box
    hi_arpu_row = df.loc[df["arpu"].idxmax()]
    insights = [
        f"Top channel by revenue is {top_channel} ({top_share}%)",
        f"Highest ARPU is {hi_arpu_row['channel']} (R{round(float(hi_arpu_row['arpu']),2)})",
    ]

    return {
        "summary": summary,
        "insights": insights,
        "traffic_mix": traffic_mix,
        "channel_performance": {
            "labels": labels,
            "traffic": traffic,
            "arpu": arpu
        }
    }

def debug_items_structure():
    """Check the items field structure"""
    client = get_bigquery_client()
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # Check items structure
    items_query = f"""
    SELECT 
        items,
        event_name,
        ecommerce
    FROM `{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_{yesterday}`
    WHERE event_name IN ('view_item', 'add_to_cart', 'purchase')
    AND items IS NOT NULL
    LIMIT 5
    """
    
    try:
        print(f"🔍 Checking items structure...")
        query_job = client.query(items_query)
        results = query_job.to_dataframe()
        
        if not results.empty:
            print("✓ Found items data!")
            print("Sample items structure:")
            for idx, row in results.iterrows():
                print(f"Event: {row['event_name']}")
                print(f"Items: {row['items']}")
                print(f"Ecommerce: {row['ecommerce']}")
                print("-" * 30)
                if idx >= 2:  # Only show first 3
                    break
        else:
            raise Exception("No items data found")
            
    except Exception as e:
        print(f"❌ Items structure check failed: {e}")
        raise Exception(f"Items structure check failed: {e}")

@app.route('/')
def index():
    """Redirect root to merchandise dashboard with current date range"""
    df, dt = g.date_from, g.date_to
    return redirect(url_for('merchandise_dashboard', **{'from': df, 'to': dt}))

@app.route('/merchandise')
def merchandise_dashboard():
    """Merchandise Analysis Dashboard route"""
    global current_analyzer

    df, dt = g.date_from, g.date_to

    try:
        # --- caching for merchandise data ---
        refresh = request.args.get("refresh") in ("1", "true", "yes")  # ?refresh=1 to bypass
        cache_key = f"merch:{df}:{dt}"

        def _build():
            return get_merchandise_data(df, dt)

        insights, products, recommendations, success, current_analyzer = get_cached(
            cache_key, _build, ttl_secs=600, refresh=refresh  # 10-minute TTL
        )

        # ---- ROUTE GUARD: quick shape/type checks before handing to Jinja ----
        def _t(x): 
            try: return type(x).__name__
            except: return "<?>"

        print("🔎 merchandise_dashboard guard:")
        print("  insights type:", _t(insights), "products type:", _t(products), "recs type:", _t(recommendations))

        # 1) insights must be a dict with specific keys used in the template
        expected_insight_keys = {
            "total_revenue", "total_views", "total_products", "avg_conversion",
            "carted_not_purchased", "not_carted", "hidden_gems", "views_threshold"
        }
        if isinstance(insights, dict):
            missing = sorted([k for k in expected_insight_keys if k not in insights])
            if missing:
                print("  ⚠️ insights missing keys:", missing)
        else:
            print("  ❗ insights is not a dict")

        # 2) products must be a list of dicts with required fields
        required_product_fields = {
            "item_name","items_viewed","items_added_to_cart","items_purchased","item_revenue","segment"
        }
        if isinstance(products, list):
            print("  products count:", len(products))
            if products:
                sample = products[0]
                print("  sample product keys:", list(sample.keys()))
                missing_pf = sorted(list(required_product_fields - set(sample.keys())))
                if missing_pf:
                    print("  ⚠️ product sample missing fields:", missing_pf)
        else:
            print("  ❗ products is not a list")

        # 3) print a couple of scalars that are interpolated in headers
        if isinstance(insights, dict):
            for k in ("total_revenue","total_views","total_products","avg_conversion","views_threshold"):
                if k in insights:
                    print(f"  insights[{k}] ->", insights[k], "(", _t(insights[k]), ")")

        # ----------------------------------------------------------------------

        # Try rendering; if Jinja errors, catch and show a focused error page with context
        try:
            return render_template(
                'dashboard.html',
                insights=insights,
                products=products,
                recommendations=recommendations,
            )
        except Exception as jinja_err:
            import traceback
            print("❌ Jinja render failed:")
            traceback.print_exc()
            # Simple focused error response that includes the types we just checked
            return (
                "<h1>Data Loading Failed</h1>"
                f"<p>Error while rendering template: {jinja_err.__class__.__name__}: {jinja_err}</p>"
                "<h3>Context snapshot</h3>"
                f"<pre>"
                f"insights type: {_t(insights)}\n"
                f"products type: {_t(products)} (count: {len(products) if isinstance(products, list) else 'n/a'})\n"
                f"recs type: {_t(recommendations)}\n"
                f"</pre>"
                "<p>See server console for full stack trace and missing keys/fields.</p>",
                500
            )
        return render_template(
            'dashboard.html',
            insights=insights,
            products=products,
            recommendations=recommendations,
        )
    except Exception as e:
        print(f"❌ Merchandise dashboard failed: {e}")
        import traceback; traceback.print_exc()
        return (
            "<h1>Data Loading Failed</h1>"
            f"<p>Error: {str(e)}</p><p>Check console for details.</p>", 
            500
        )

@app.route('/channel-performance')
def channel_performance():
    """Channel Performance Dashboard"""
    global current_analyzer

    df, dt = g.date_from, g.date_to  # from @before_request

    try:
        # --- caching for channel data ---
        refresh = request.args.get("refresh") in ("1", "true", "yes")
        cache_key = f"channel:{df}:{dt}"

        def _build():
            return get_channel_metrics_from_daily(df, dt)

        channel_data = get_cached(cache_key, _build, ttl_secs=600, refresh=refresh)
        # --------------------------------

        return render_template(
            'channel_performance.html',
            channel_data=channel_data,
        )
    except Exception as e:
        print(f"❌ Channel performance failed: {e}")
        return (
            "<h1>Channel Data Loading Failed</h1>"
            f"<p>Error: {str(e)}</p>",
            500
        )

@app.route('/device-analysis')

def device_analysis():
    """Device Analysis Dashboard"""
    global current_analyzer

    df, dt = g.date_from, g.date_to

    try:
        # --- caching for device data ---
        refresh = request.args.get("refresh") in ("1", "true", "yes")
        cache_key = f"device:{df}:{dt}"

        def _build():
            return get_device_metrics_from_bigquery(df, dt)

        device_data = get_cached(cache_key, _build, ttl_secs=900, refresh=refresh)
        # --------------------------------

        # === Route-level filtering (only if not done in get_device_metrics...) ===
        kept = {
            d['name'].lower()
            for d in device_data.get('categories', [])
            if d.get('traffic', 0) >= 0.1
        }
        device_data['categories'] = [
            d for d in device_data.get('categories', [])
            if d['name'].lower() in kept
        ]

        # keep funnel keys that correspond to kept devices
        device_data['funnel'] = {
            k: v for k, v in device_data.get('funnel', {}).items()
            if k in {'desktop', 'mobile', 'tablet'} and k in {n.split()[0].lower() for n in kept}
        }

        # keep resolutions rows only for kept device types
        device_data['resolutions'] = [
            r for r in device_data.get('resolutions', [])
            if r.get('device_type', '').lower() in kept and r.get('traffic_share', 0) >= 0.1
        ]
        # ========================================================================

        return render_template(
            'device_analysis.html',
            device_data=device_data,
        )

    except Exception as e:
        print(f"❌ Device analysis failed: {e}")
        return (
            "<h1>Device Data Loading Failed</h1>"
            f"<p>Error: {str(e)}</p>",
            500
        )

def get_ais_metrics_from_daily(date_from, date_to):
    """
    Compute Audience Intent Score using the pre-aggregated daily table
    (frankie_derived.daily_channel_kpis), and return:
      - overall score + quality label
      - weekly time series (true weeks, Monday-start)
      - engagement/performance metrics (incl. Revenue per User)
      - top channels with intent scores
    """
    client = get_bigquery_client()
    from datetime import datetime
    from google.cloud import bigquery
    import pandas as pd

    # --- 1) Parse dates ---
    d_from = datetime.strptime(date_from, "%Y-%m-%d").date()
    d_to   = datetime.strptime(date_to,   "%Y-%m-%d").date()

    # --- 2) Window totals for performance/engagement panel ---
    sql_totals = """
    SELECT
      SUM(users)      AS users,
      SUM(purchasers) AS purchasers,
      SUM(revenue)    AS revenue,
      SUM(page_views) AS page_views,
      SUM(sessions)   AS sessions,
      SAFE_DIVIDE(SUM(avg_engagement_seconds * users), NULLIF(SUM(users), 0)) AS w_engagement_seconds
    FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
    WHERE event_date BETWEEN @from AND @to
    """
    totals = client.query(
        sql_totals,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("from", "DATE", d_from),
            bigquery.ScalarQueryParameter("to",   "DATE", d_to),
        ])
    ).to_dataframe().iloc[0]

    total_users   = float(totals.get("users") or 0)
    total_purch   = float(totals.get("purchasers") or 0)
    total_revenue = float(totals.get("revenue") or 0)
    page_views    = float(totals.get("page_views") or 0)
    sessions      = float(totals.get("sessions") or 0)
    w_eng_secs    = float(totals.get("w_engagement_seconds") or 0)

    overall_cvr       = (total_purch / total_users * 100.0) if total_users > 0 else 0.0
    revenue_per_user  = (total_revenue / total_users)       if total_users > 0 else 0.0
    pages_per_session = (page_views / sessions)             if sessions     > 0 else 0.0

    def fmt_time(seconds: float) -> str:
        s = int(seconds or 0)
        return f"{s//60}m {s%60:02d}s"

    # --- Engagement panel (keep only what we trust) ---
    engagement_metrics = [
        {"label": "Avg. Engagement Time", "value": fmt_time(w_eng_secs)},
        {"label": "Pages per Session",    "value": f"{pages_per_session:.2f}"},
    ]

    # --- 3) Channel aggregates for current window (users, purchasers, revenue) ---
    sql_channels = """
    SELECT
      assigned_channel AS channel,
      SUM(users)       AS users,
      SUM(purchasers)  AS purchasers,
      SUM(revenue)     AS revenue
    FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
    WHERE event_date BETWEEN @from AND @to
    GROUP BY assigned_channel
    HAVING users > 0
    """
    df_ch = client.query(
        sql_channels,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("from", "DATE", d_from),
            bigquery.ScalarQueryParameter("to",   "DATE", d_to),
        ])
    ).to_dataframe()

    if df_ch.empty:
        # Keep the route resilient
        return {
            "score": 0.0,
            "quality_label": "Needs Improvement",
            "interpretation": "No channels in range.",
            "channels": [],
            "time_series": {"labels": [], "scores": []},
            "engagement_metrics": engagement_metrics,
            "performance_metrics": [
                {"label": "Purchase Conversion Rate",    "value": f"{overall_cvr:.2f}%"},
                {"label": "New User Purchase CVR",       "value": "—"},
                {"label": "Returning User Purchase CVR", "value": "—"},
                {"label": "Revenue per User",            "value": f"R{revenue_per_user:.2f}"},
                {"label": "Baseline Channel",            "value": "N/A"},
            ],
            "user_segments": [],
            "strategies": []
        }

    # Derived per-channel metrics
    df_ch["cvr"] = (df_ch["purchasers"] / df_ch["users"] * 100.0).fillna(0.0)
    total_users_window = float(df_ch["users"].sum()) or 0.0
    df_ch["traffic_share"] = (df_ch["users"] / total_users_window * 100.0) if total_users_window > 0 else 0.0

    baseline_cvr = float(df_ch["cvr"].max()) if (df_ch["cvr"] > 0).any() else 0.0
    baseline_channel = (df_ch.loc[df_ch["cvr"].idxmax(), "channel"]
                        if (df_ch["cvr"] > 0).any() else "N/A")

    def intent_score(cvr, base):
        if base <= 0: return 0.0
        return min(100.0, (float(cvr) / float(base)) * 100.0)

    df_ch["intent_score"] = df_ch["cvr"].apply(lambda x: intent_score(x, baseline_cvr))
    df_ch["impact"] = df_ch["intent_score"] * (df_ch["traffic_share"] / 100.0)

    channels = []
    for _, r in df_ch.sort_values("intent_score", ascending=False).iterrows():
        score = float(r["intent_score"])
        level = "high" if score >= 80 else ("medium" if score >= 50 else "low")
        label = "High Intent" if level == "high" else ("Medium Intent" if level == "medium" else "Low")
        channels.append({
            "name": str(r["channel"]),
            "score": round(score, 1),
            "intent_level": level,
            "intent_label": label,
            "traffic_share": round(float(r["traffic_share"]), 1),
            "impact": round(float(r["impact"]), 1),
        })

    overall_score = float(df_ch["impact"].sum())
    quality_label = ("Excellent" if overall_score >= 75 else
                     "Good"      if overall_score >= 60 else
                     "Moderate"  if overall_score >= 40 else
                     "Needs Improvement")

    # --- Performance panel (keep only CVR + RPU + Baseline) ---
    performance_metrics = [
        {"label": "Purchase Conversion Rate", "value": f"{overall_cvr:.2f}%"},
        {"label": "Revenue per User",         "value": f"R{revenue_per_user:.2f}"},
        {"label": "Baseline Channel",         "value": str(baseline_channel)},
    ]

    # --- 4) True weekly AIS series (Monday-start) ---
    # Compute per-week per-channel CVR, baseline CVR per week, then AIS = sum(intent_score * user share)
    sql_weeks = """
    WITH base AS (
    SELECT
        DATE_TRUNC(event_date, WEEK(MONDAY)) AS week_start,
        assigned_channel AS channel,
        SUM(users)                              AS users,
        SUM(IFNULL(purchasers, 0))              AS purchasers
    FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
    WHERE event_date BETWEEN @from AND @to
    GROUP BY week_start, assigned_channel
    HAVING users > 0
    ),
    rates AS (
    SELECT
        week_start, channel, users,
        SAFE_MULTIPLY(SAFE_DIVIDE(purchasers, NULLIF(users, 0)), 100.0) AS cvr
    FROM base
    ),
    wk AS (
    SELECT
        week_start,
        IFNULL(MAX(cvr), 0.0) AS base_cvr,
        SUM(users)            AS all_users
    FROM rates
    GROUP BY week_start
    ),
    scored AS (
    SELECT
        r.week_start,
        r.channel,
        r.users,
        w.all_users,
        LEAST(
        100.0,
        SAFE_DIVIDE(IFNULL(r.cvr, 0.0), NULLIF(w.base_cvr, 0.0)) * 100.0
        ) AS intent_score
    FROM rates r
    JOIN wk w USING (week_start)
    ),
    weighted AS (
    SELECT
        week_start,
        SAFE_MULTIPLY(intent_score, SAFE_DIVIDE(users, NULLIF(all_users, 0))) AS contrib
    FROM scored
    )
    SELECT
    week_start,
    IFNULL(SUM(contrib), 0.0) AS ais
    FROM weighted
    GROUP BY week_start
    ORDER BY week_start
    """
    df_wk = client.query(
        sql_weeks,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("from", "DATE", d_from),
            bigquery.ScalarQueryParameter("to",   "DATE", d_to),
        ])
    ).to_dataframe()

    if df_wk.empty:
        time_series = {"labels": [], "scores": []}
    else:
        df_wk["week_start"] = pd.to_datetime(df_wk["week_start"])
        df_wk["ais"] = df_wk["ais"].fillna(0.0)
        labels = [d.strftime("%b %d") for d in df_wk["week_start"].tolist()]  # e.g., "Sep 02"
        scores = [round(float(x), 1) for x in df_wk["ais"].tolist()]
        time_series = {"labels": labels, "scores": scores}

    strategies = []
    if channels:
        strategies.append({
            "label": "TOP PERFORMER",
            "title": channels[0]["name"],
            "description": f'Your best channel with {channels[0]["score"]:.0f} intent score.',
            "intent_class": "high-intent"
        })

    return {
        "score": round(overall_score, 1),
        "quality_label": quality_label,
        "interpretation": (
            f'{channels[0]["name"]} ({channels[0]["traffic_share"]:.1f}% of users) contributes the most to your score.'
            if channels else "No channels in range."
        ),
        "channels": channels[:8],
        "time_series": time_series,
        "engagement_metrics": engagement_metrics,
        "performance_metrics": performance_metrics,
        "user_segments": [],  # hide the table via template guard
        "strategies": strategies
    }

# --- AI eCommerce Manager (Overwatch) ---------------------------------------

def _last7():
    # Reuse your date helper so all dashboards line up with "last 7 days"
    f, t, _, _ = resolve_date_range(preset="last7")
    return f, t

def get_cro_timeseries(date_from: str, date_to: str):
    """
    Returns daily time-series for:
      - purchase_cvr (% of users who purchased)
      - desktop_purchase_cvr
      - mobile_purchase_cvr
      - add_to_cart_rate (% of users who added to cart)
      - begin_checkout_rate (% of users who began checkout)

    Uses the fast daily rollup for total users/purchasers and a focused events_* scan
    (only for add_to_cart / begin_checkout and device splits) bounded to the given dates.
    """
    client = get_bigquery_client()
    from google.cloud import bigquery
    import pandas as pd
    from datetime import datetime

    # --- 1) Parse date strings & table suffixes ---
    d_from = datetime.strptime(date_from, "%Y-%m-%d").date()
    d_to   = datetime.strptime(date_to,   "%Y-%m-%d").date()
    start_suffix = d_from.strftime("%Y%m%d")
    end_suffix   = d_to.strftime("%Y%m%d")

    # --- 2) Sitewide users & purchasers (fast, from daily rollup) ---
    sql_users = """
    SELECT
      event_date,
      SUM(users)      AS users,
      SUM(purchasers) AS purchasers
    FROM `carrol-boyes-ga4.frankie_derived.daily_channel_kpis`
    WHERE event_date BETWEEN @from AND @to
    GROUP BY event_date
    ORDER BY event_date
    """
    df_users = client.query(
        sql_users,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("from", "DATE", d_from),
            bigquery.ScalarQueryParameter("to",   "DATE", d_to),
        ])
    ).to_dataframe()

    if df_users.empty:
        # Keep caller resilient
        return {
            "labels": [],
            "purchase_cvr": [],
            "desktop_purchase_cvr": [],
            "mobile_purchase_cvr": [],
            "add_to_cart_rate": [],
            "begin_checkout_rate": [],
        }

    # --- 3) Per-day counts for ATC + Begin Checkout (focused scan) ---
    sql_stage = f"""
    WITH base AS (
      SELECT
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS event_date,
        LOWER(device.category) AS device_category,
        user_pseudo_id,
        event_name
      FROM `{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
        AND event_name IN ('add_to_cart','begin_checkout','purchase')
    )
    SELECT
      event_date,
      COUNT(DISTINCT IF(event_name='add_to_cart',     user_pseudo_id, NULL)) AS atc_users,
      COUNT(DISTINCT IF(event_name='begin_checkout',  user_pseudo_id, NULL)) AS bc_users
    FROM base
    GROUP BY event_date
    ORDER BY event_date
    """
    df_stage = client.query(sql_stage).to_dataframe()

    # --- 4) Device split for purchase CVR (desktop, mobile) ---
    sql_device = f"""
    WITH base AS (
      SELECT
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS event_date,
        LOWER(device.category) AS device_category,
        user_pseudo_id,
        event_name
      FROM `{PROJECT_ID}.analytics_{GA4_PROPERTY_ID}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
    ),
    by_dev AS (
      SELECT
        event_date,
        device_category,
        COUNT(DISTINCT user_pseudo_id) AS users,
        COUNT(DISTINCT IF(event_name='purchase', user_pseudo_id, NULL)) AS purchasers
      FROM base
      WHERE device_category IN ('desktop','mobile')
      GROUP BY event_date, device_category
    )
    SELECT * FROM by_dev
    """
    df_dev = client.query(sql_device).to_dataframe()

    # --- 5) Assemble time-series (align by date) ---
    # Ensure we have a full date index from df_users
    df = df_users.copy()
    df["purchase_cvr"] = (df["purchasers"] / df["users"] * 100.0).fillna(0.0)

    # Merge stage rates
    if not df_stage.empty:
        df = df.merge(df_stage, on="event_date", how="left")
        df["atc_users"] = df["atc_users"].fillna(0)
        df["bc_users"]  = df["bc_users"].fillna(0)
        df["add_to_cart_rate"]    = (df["atc_users"] / df["users"] * 100.0).fillna(0.0)
        df["begin_checkout_rate"] = (df["bc_users"]  / df["users"] * 100.0).fillna(0.0)
    else:
        df["add_to_cart_rate"] = 0.0
        df["begin_checkout_rate"] = 0.0

    # Device CVRs
    desktop = df_dev[df_dev["device_category"]=="desktop"].rename(
        columns={"users":"desktop_users","purchasers":"desktop_purch"})
    mobile  = df_dev[df_dev["device_category"]=="mobile"].rename(
        columns={"users":"mobile_users","purchasers":"mobile_purch"})

    df = df.merge(desktop[["event_date","desktop_users","desktop_purch"]], on="event_date", how="left")
    df = df.merge(mobile[["event_date","mobile_users","mobile_purch"]],   on="event_date", how="left")

    df["desktop_purchase_cvr"] = (df["desktop_purch"] / df["desktop_users"] * 100.0).fillna(0.0)
    df["mobile_purchase_cvr"]  = (df["mobile_purch"]  / df["mobile_users"]  * 100.0).fillna(0.0)

    # Prepare outputs
    labels = [d.strftime("%b %d") for d in pd.to_datetime(df["event_date"]).tolist()]
    return {
        "labels": labels,
        "purchase_cvr":        [round(float(x), 2) for x in df["purchase_cvr"].tolist()],
        "desktop_purchase_cvr":[round(float(x), 2) for x in df["desktop_purchase_cvr"].tolist()],
        "mobile_purchase_cvr": [round(float(x), 2) for x in df["mobile_purchase_cvr"].tolist()],
        "add_to_cart_rate":    [round(float(x), 2) for x in df["add_to_cart_rate"].tolist()],
        "begin_checkout_rate": [round(float(x), 2) for x in df["begin_checkout_rate"].tolist()],
    }

def build_overwatch_brief(date_from, date_to):
    """
    Pull concise, model-friendly stats from each dashboard’s data layer.
    Keep it small and deterministic (no raw tables) to control token use.
    """
    brief = {"window": {"from": date_from, "to": date_to}, "panels": {}}

    # Channel Performance (daily rollup = fastest)
    try:
        ch = get_channel_metrics_from_daily(date_from, date_to)
        # Summarize for the model
        brief["panels"]["channels"] = {
            "summary": ch.get("summary", {}),
            "insights": ch.get("insights", []),
            "top_channels": [
                {"name": n, "users": u, "arpu": a, "share": s}
                for n, u, a, s in zip(
                    ch["channel_performance"]["labels"],
                    ch["channel_performance"]["traffic"],
                    ch["channel_performance"]["arpu"],
                    [tm["traffic_share"] for tm in ch["traffic_mix"]],
                )
            ][:8],
        }
    except Exception as e:
        brief["panels"]["channels_error"] = str(e)

    # Device Analysis (direct BQ, but already optimized)
    try:
        dev = get_device_metrics_from_bigquery(date_from, date_to)
        brief["panels"]["devices"] = {
            "categories": dev.get("categories", []),
            "funnel": dev.get("funnel", {}),
            "resolutions_top": dev.get("resolutions", [])[:10],
        }
    except Exception as e:
        brief["panels"]["devices_error"] = str(e)

    # Merchandise Analysis (uses your MerchandiseAnalyzerBQ)
    try:
        insights, products, recs, _ok, _an = get_merchandise_data(date_from, date_to)
        # Keep the payload tight
        brief["panels"]["merchandise"] = {
            "kpis": {
                "total_revenue": insights.get("total_revenue"),
                "total_views": insights.get("total_views"),
                "total_products": insights.get("total_products"),
                "avg_conversion": insights.get("avg_conversion"),
            },
            "need_attention": [
                {"name": p["item_name"], "views": p["items_viewed"], "lost": p.get("lost_revenue", 0)}
                for p in (insights.get("carted_not_purchased") or [])[:8]
            ],
            "hidden_gems": [
                {"name": p["item_name"], "views": p["items_viewed"], "rev": p["item_revenue"]}
                for p in (insights.get("hidden_gems") or [])[:8]
            ],
            "top_products": [
                {"name": p["item_name"], "views": p["items_viewed"], "rev": p["item_revenue"]}
                for p in sorted(products, key=lambda r: r.get("item_revenue", 0), reverse=True)[:10]
            ],
        }
    except Exception as e:
        brief["panels"]["merchandise_error"] = str(e)

    return brief

def _ai_call_overwatch(brief_dict):
    """
    Cached wrapper: returns a cached Overwatch summary for the given date window.
    The fresh logic lives in `_ai_call_overwatch_fresh`.
    """
    cache_key = f"ai_overwatch:{brief_dict['window']['from']}:{brief_dict['window']['to']}"

    def _builder():
        # Only runs when cache is cold or expired
        return _ai_call_overwatch_fresh(brief_dict)

    return get_cached(cache_key, _builder, ttl_secs=900)

def build_cro_brief_for_ai(date_from: str, date_to: str) -> dict:
    """
    Make a compact, model-friendly brief for CRO.
    Uses your get_cro_timeseries() so it matches the on-screen charts.
    """
    ts = get_cro_timeseries(date_from, date_to)  # already cached via /cro
    labels = ts.get("labels", [])
    def _stats(vals):
        try:
            xs = [float(v) for v in (vals or []) if v is not None]
            if not xs: return {}
            return {
                "first": xs[0], "last": xs[-1],
                "min": min(xs), "max": max(xs),
                "avg": sum(xs)/len(xs)
            }
        except Exception:
            return {}

    brief = {
        "window": {"from": date_from, "to": date_to},
        "series": {
            "purchase_cvr": ts.get("purchase_cvr", []),
            "desktop_purchase_cvr": ts.get("desktop_purchase_cvr", []),
            "mobile_purchase_cvr": ts.get("mobile_purchase_cvr", []),
            "add_to_cart_rate": ts.get("add_to_cart_rate", []),
            "begin_checkout_rate": ts.get("begin_checkout_rate", []),
            "labels": labels,
        },
        "stats": {
            "purchase_cvr": _stats(ts.get("purchase_cvr")),
            "desktop_purchase_cvr": _stats(ts.get("desktop_purchase_cvr")),
            "mobile_purchase_cvr": _stats(ts.get("mobile_purchase_cvr")),
            "add_to_cart_rate": _stats(ts.get("add_to_cart_rate")),
            "begin_checkout_rate": _stats(ts.get("begin_checkout_rate")),
        }
    }
    return brief

def _ai_cro_summary_fresh(brief: dict) -> str:
    """
    Return HTML only (no markdown). 3 sections: Overview, Insights, Recommendations.
    Uses ZAR wording where money is referenced, and labels rates clearly.
    """
    import os, json
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BEARER") or ""
    # Local fallback if no key:
    if not api_key:
        s = brief.get("stats", {})
        p = s.get("purchase_cvr", {}) or {}
        d = s.get("desktop_purchase_cvr", {}) or {}
        m = s.get("mobile_purchase_cvr", {}) or {}
        atc = s.get("add_to_cart_rate", {}) or {}
        bc  = s.get("begin_checkout_rate", {}) or {}
        def f(x): 
            return f"{x:.2f}" if isinstance(x, (int,float)) else "—"
        return (
            "<h3>Overview</h3><ul>"
            f"<li>Site purchase rate ended at {f(p.get('last'))}% "
            f"({('+' if (p.get('last',0)-p.get('first',0))>=0 else '−')}{f(abs((p.get('last',0)-p.get('first',0))) )} pts vs start).</li>"
            f"<li>Desktop: {f(d.get('last'))}% · Mobile: {f(m.get('last'))}%.</li>"
            "</ul>"
            "<h3>Insights</h3><ul>"
            f"<li>Add-to-Cart latest: {f(atc.get('last'))}% · Begin-Checkout latest: {f(bc.get('last'))}%.</li>"
            f"<li>Range (site purchase rate): {f(p.get('min'))}%–{f(p.get('max'))}% (avg {f(p.get('avg'))}%).</li>"
            "</ul>"
            "<h3>Recommendations</h3><ul>"
            "<li>Investigate steps between Add-to-Cart and Purchase where drop-off is largest (shipping fees, payment fails, validation friction).</li>"
            "<li>Target mobile first if gap to desktop persists; prioritise PDP load, variant clarity, and checkout field minimisation.</li>"
            "<li>Run a price/offer A/B for top traffic sources; report impact on site purchase rate and ATC→Checkout progression.</li>"
            "</ul>"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "You are a senior CRO analyst. Output HTML only (no Markdown, no code fences). "
            "Write three sections with <h3> headings and <ul><li> bullets:\n"
            "<h3>Overview</h3>\n<ul>...</ul>\n"
            "<h3>Insights</h3>\n<ul>...</ul>\n"
            "<h3>Recommendations</h3>\n<ul>...</ul>\n"
            "Use precise wording: 'site purchase rate' for site-wide purchasers÷users; "
            "'device purchase rate' when split by device. Use percentages with 1–2 dp. "
            "If money is mentioned, call it ZAR (R). Keep ~130–170 words total. "
            "Avoid sweeping claims; reference context like change vs window start, min/max, or device diffs."
        )
        user = (
            "Here is the CRO timeseries brief for the selected window. "
            "Summarise movements and produce 3–6 practical recommendations that a PM can try next. "
            "Keep it crisp.\n\n"
            f"{json.dumps(brief, ensure_ascii=False)}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.25,
        )
        html = (resp.choices[0].message.content or "").strip()
        return html or "<h3>Overview</h3><ul><li>No AI content returned.</li></ul><h3>Insights</h3><ul></ul><h3>Recommendations</h3><ul></ul>"
    except Exception as e:
        return (
            "<h3>Overview</h3><ul>"
            f"<li>AI call failed: {e}</li>"
            "</ul><h3>Insights</h3><ul><li>Using local data on page only.</li></ul>"
            "<h3>Recommendations</h3><ul><li>Retry after configuring OPENAI_API_KEY.</li></ul>"
        )

def ai_cro_summary_cached(brief: dict, refresh: bool=False) -> str:
    key = f"ai_cro:{brief['window']['from']}:{brief['window']['to']}"
    return get_cached(key, lambda: _ai_cro_summary_fresh(brief), ttl_secs=600, refresh=refresh)

@app.route('/api/ask-frankie/cro')
def api_ask_frankie_cro():
    df, dt = g.date_from, g.date_to
    refresh = request.args.get("refresh") in ("1","true","yes")
    try:
        brief = build_cro_brief_for_ai(df, dt)
        html = ai_cro_summary_cached(brief, refresh=refresh)
        return jsonify({"ok": True, "html": html, "window": brief["window"]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

def _ai_call_overwatch_fresh(brief_dict):
    """
    Build the Overwatch weekly wrap. Uses OpenAI if configured; otherwise returns a local fallback.
    Keep this function pure: it returns a plain string.
    """
    import json, os

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BEARER") or ""
    if not api_key:
        # Fallback summary — no external call
        return (
            "Overwatch Insights (local summary)\n\n"
            "• What’s happening — Summaries compiled from your last 7 days across Channel, Device, and Merchandise.\n"
            "• What to be aware of — Check ‘Needs Attention’ items in Merchandise; review devices with weak cart→purchase.\n"
            "• Alerts — Missing API key for AI analysis. Set OPENAI_API_KEY to enable full AI.\n"
            "• Wins — See ‘Hidden Gems’ and top ARPU channels to amplify."
        )

    # ---- Real OpenAI call ----
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "You are an eCommerce actuary, merchandising and marketing expert. Output HTML only (no Markdown, no code fences). "
            "Structure exactly as four sections with <h3> headings and <ul><li> bullets:\n"
            "<h3>What's happening</h3>\n<ul>...</ul>\n"
            "<h3>What to be aware of</h3>\n<ul>...</ul>\n"
            "<h3>Alerts</h3>\n<ul>...</ul>\n"
            "<h3>Wins</h3>\n<ul>...</ul>\n"
            "Write ~150–180 words total. Use South African Rand (ZAR) for all money. "
            "Always label rates precisely to avoid confusion:\n"
            "- Use 'site purchase rate' ONLY for site-wide purchasers÷users.\n"
            "- Use 'product purchase rate' ONLY for per-product purchases÷views.\n"
            "If you mention a rate, say which it is (site purchase rate vs product purchase rate). "
            "Avoid sweeping claims; include brief context (e.g., compare to another channel/device or previous value in the brief). "
            "Do not include tables, code, or emojis."
        )
        user = (
            "Use this 7-day brief from Channels, Devices, and Merchandise. "
            "Only include materially relevant changes or outliers.\n\n"
            f"{json.dumps(brief_dict, ensure_ascii=False)}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Defensive fallback if API returns empty
        return text or "Overwatch Insights: AI returned no content."
    except Exception as e:
        # Fail soft
        return (
            "Overwatch Insights (degraded)\n\n"
            f"AI call failed: {e}\n"
            "• Check OPENAI_API_KEY and network access.\n"
            "• Meanwhile, review Merchandise ‘Needs Attention’, device funnels, and top ARPU channels."
        )

@app.post('/api/ask-frankie')
def api_ask_frankie():
    """
    Body: { kind: 'summary'|'deep', page: 'cro'|'merch'|'channel'|'device'|..., window:{from,to}, payload:{...} }
    Returns: { success: bool, html: "<h3>...</h3>..." }
    """
    import os, json
    data = request.get_json(force=True) or {}
    kind   = data.get("kind", "summary")
    page   = data.get("page", "generic")
    window = data.get("window", {})
    payload= data.get("payload", {})

    # Defensive clamp: keep payload small
    def shrink_series(d, n=60):
        # trim any large arrays in place to latest n points
        if isinstance(d, dict):
            for k,v in list(d.items()):
                if isinstance(v, list) and len(v) > n:
                    d[k] = v[-n:]
                elif isinstance(v, dict):
                    shrink_series(v, n)
        return d

    payload = shrink_series(payload)

    # Build targeted instructions based on page
    page_hints = {
        "cro": "Focus on site purchase rate, desktop vs mobile CVR, add-to-cart and begin-checkout rates. Call out inflection points and likely causes.",
        "channel": "Focus on traffic share, ARPU and site purchase rate by channel. Flag channels to scale or trim.",
        "device": "Focus on device mix, resolution bands, and device-level funnels (ATC→Checkout→Purchase).",
        "merch": "Focus on product purchase rate vs views, hidden gems, and carted-not-purchased items with remedies.",
        "generic": "Summarize the most material movements and risks."
    }.get(page, "Summarize key movements and actions.")

    depth = "Give a crisp summary with 3–5 bullets per section." if kind=="summary" else \
            "Provide a deeper diagnosis with brief hypotheses and 3–6 action recommendations."

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_BEARER")
    if not api_key:
        # graceful fallback
        html = f"""
        <h3>Overview</h3>
        <ul><li>AI key not configured. Showing local summary if available.</li></ul>
        """
        return jsonify({"success": True, "html": html})

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system = (
            "You are an eCommerce CRO analyst. Respond with **HTML only** (no Markdown). "
            "Use clear <h3> section headings and <ul><li> bullets. "
            "Currency is South African Rand (ZAR). "
            "Be explicit about metric types: 'site purchase rate' vs 'product purchase rate'. "
            "Avoid vague claims; give compact context (vs prior day/period, device/channel, etc.). "
            "No tables or code."
        )
        user = {
            "window": window,
            "page": page,
            "instructions": page_hints,
            "depth": depth,
            "data": payload
        }

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": system},
                {"role":"user","content": json.dumps(user, ensure_ascii=False)}
            ],
            temperature=0.3,
        )
        html = (resp.choices[0].message.content or "").strip()
        if not html:
            html = "<h3>Overview</h3><ul><li>No insights returned.</li></ul>"
        return jsonify({"success": True, "html": html})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ai-manager')
def ai_manager():
    """
    Renders the AI eCommerce Manager page.
    Always uses last 7 days for the Overwatch block (as requested).
    """
    # Always lock the AI block to last 7
    df7, dt7 = _last7()

    # Build brief and (optionally) call AI
    brief = build_overwatch_brief(df7, dt7)
    overwatch_text = _ai_call_overwatch(brief)

    # Keep sidebar date picker live with current global dates
    df, dt = g.date_from, g.date_to

    return render_template(
        "ai_manager.html",
        overwatch_text=overwatch_text,
        brief_window={"from": df7, "to": dt7},
        date_from=df,
        date_to=dt,
    )
# ---------------------------------------------------------------------------

@app.route('/audience-intent')
def audience_intent():
    """Audience Intent Dashboard using daily rollups -> true weekly series"""
    df, dt = g.date_from, g.date_to
    try:
        # --- caching for AIS data ---
        refresh = request.args.get("refresh") in ("1", "true", "yes")
        cache_key = f"ais:{df}:{dt}"

        def _build():
            return get_ais_metrics_from_daily(df, dt)

        ais_data = get_cached(cache_key, _build, ttl_secs=900, refresh=refresh)
        # --------------------------------
        return render_template('audience_intent.html',
                               ais_data=ais_data,
                               date_from=df,
                               date_to=dt)
    except Exception as e:
        print(f"❌ Audience intent failed: {e}")
        return f"<h1>AIS Data Loading Failed</h1><p>Error: {str(e)}</p>", 500
    
@app.route('/cro')
def cro_dashboard():
    # respect the sidebar picker
    df, dt = g.date_from, g.date_to
    cro_data = get_cro_series_from_bq(df, dt)
    mob = get_cro_mobile_series_from_bq(df, dt)
    cro_data.update(mob)  # now cro_data has both .desktop and .mobile

    # simple TTL cache (10 min)
    refresh = request.args.get("refresh") in ("1","true","yes")
    cache_key = f"cro:{df}:{dt}"

    def _build():
        return get_cro_timeseries(df, dt)

    ts = get_cached(cache_key, _build, ttl_secs=600, refresh=refresh)

    return render_template(
        'cro_dashboard.html',
        cro_data=cro_data,
        ts=ts,               # time-series dict
        date_from=df,
        date_to=dt
    )

@app.route('/ai-manager-old')
def ai_manager_old():
    """
    AI eCommerce Manager (Overwatch Insights)
    Step 1: render the page with a placeholder block.
    In Step 2 we'll call OpenAI to generate the weekly wrap-up.
    """
    # Compute a default last-7-days window (independent of sidebar picker)
    # We’ll still pass the global picker values so the nav stays consistent.
    from datetime import datetime, timedelta
    today_local = datetime.utcnow() + timedelta(hours=2)  # adjust if you prefer
    end_date = (today_local - timedelta(days=1)).date()   # yesterday
    start_date = (end_date - timedelta(days=6))           # 7-day window

    last7_from = start_date.strftime("%Y-%m-%d")
    last7_to   = end_date.strftime("%Y-%m-%d")

    # No AI yet — just render a clean shell with placeholders
    return render_template(
        'ai_manager.html',
        date_from=g.date_from,
        date_to=g.date_to,
        last7_from=last7_from,
        last7_to=last7_to,
        overwatch_text=None  # will be filled in Step 2
    )

@app.route('/api/update-dashboard', methods=['POST'])
def update_dashboard_api():
    """API endpoint for merchandise dashboard date range updates"""
    data = request.json
    date_from = data.get('from')
    date_to = data.get('to')
    
    try:
        insights, products, recommendations, success, _ = get_merchandise_data(date_from, date_to)
        
        return jsonify({
            'success': success,
            'insights': insights,
            'products': products,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"❌ API update failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FRANKIE AI MERCHANDISE ANALYTICS DASHBOARD")
    print("="*60)
    print(f"📊 Connected to BigQuery project: {PROJECT_ID}")
    print(f"📁 GA4 Property: {GA4_PROPERTY_ID}")

    print("\n🔧 Running BigQuery diagnostics...")
    try:
        if RUN_DIAGNOSTICS:
            debug_bigquery_structure()
            print("\n" + "-"*50)
            debug_items_structure()
            print("\n" + "-"*50)
            test_merchandise_query()
            print("\n" + "="*60)
            print("\n" + "="*60)
            print("BIGQUERY DATA AUDIT")
            print("="*60)
            audit_bigquery_data()

        # Run data audit
        print("\n" + "="*60)
        print("BIGQUERY DATA AUDIT")
        print("="*60)
        audit_bigquery_data()
        
        # Check if OpenAI is configured
        if ai_recommender:
            print("🤖 AI Recommendations: ENABLED")
        else:
            print("🤖 AI Recommendations: DISABLED (set OPENAI_API_KEY to enable)")
        
        print("\nStarting Flask server...")
        print("Navigate to: http://localhost:8080")
        print("\nPress CTRL+C to stop the server\n")
        print("-"*60)
        
        # Run the Flask app
        app.run(debug=True, host='127.0.0.1', port=8080)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("The application cannot start because BigQuery data is not accessible.")
        print("Please check your credentials and data availability.")
        sys.exit(1)