import uuid
import math
import time
import requests
import psycopg2
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
from playwright.sync_api import sync_playwright
import re
import urllib.parse
import requests
import xml.etree.ElementTree as ET
import execjs


# -----------------------------
# JavaScript code as a raw string
# -----------------------------
JS_CODE = r"""
function radiansToDegrees(rad) {
    return rad / (Math.PI / 180);
}

function z192LL(x19, y19) {
    var x = x19 / Math.pow(2, 19);
    var y = y19 / Math.pow(2, 19);
    var originX = 128.0;
    var originY = 128.0;
    var pixelsPerLon = 256.0 / 360.0;
    var pixelsPerLonRadian = 256.0 / (2 * Math.PI);
    var lon = (x - originX) / pixelsPerLon;
    var latRadians = (y - originY) / (-pixelsPerLonRadian);
    var lat = radiansToDegrees(2 * Math.atan(Math.exp(latRadians)) - Math.PI / 2);
    return { m_lon: lon, m_lat: lat };
}

function parsePolyGeometry(geom) {
    var g = geom.split(",");
    var color = g[0].length > 0 ? g[0] : "#FFFF00";
    var lineWidth = g[1].length > 0 ? parseInt(g[1]) : 3;
    var ptCount = (g.length - 2) / 2;
    var xpts = [];
    var ypts = [];
    xpts[0] = parseInt(g[2]);
    ypts[0] = parseInt(g[3]);
    for (var i = 1; i < ptCount; i++) {
        xpts[i] = xpts[i - 1] + parseInt(g[2 + i * 2]);
        ypts[i] = ypts[i - 1] + parseInt(g[2 + i * 2 + 1]);
    }
    var coordinates = [];
    for (var i = 0; i < ptCount; i++) {
        var ll = z192LL(xpts[i], ypts[i]);
        coordinates.push([ll.m_lon, ll.m_lat]);
    }
    if (coordinates.length > 0 &&
        (coordinates[0][0] !== coordinates[coordinates.length - 1][0] ||
         coordinates[0][1] !== coordinates[coordinates.length - 1][1])) {
        coordinates.push(coordinates[0]);
    }
    return { color: color, lineWidth: lineWidth, coordinates: coordinates };
}

function toGeoJSON(polyResult, zoning_name) {
    return {
        type: "Feature",
        properties: {
            color: polyResult.color,
            lineWidth: polyResult.lineWidth,
            zoning_name: zoning_name
        },
        geometry: {
            type: "Polygon",
            coordinates: [polyResult.coordinates]
        }
    };
}

// Manual trim function for Windows JScript
function jsTrim(s) {
    return s.replace(/^\s+|\s+$/g, '');
}

// Parse all polys with zoning_name
function parseAllPolysWithZoning(polyTags, zoning_name) {
    var features = [];
    for (var i = 0; i < polyTags.length; i++) {
        var tag = polyTags[i];
        var geom = jsTrim(tag.replace(/<\/?poly>/g, ""));
        if (!geom) continue;
        var polyResult = parsePolyGeometry(geom);
        var feature = toGeoJSON(polyResult, zoning_name);
        features.push(feature);
    }
    return { type: "FeatureCollection", features: features };
}
"""
# Compile JS
ctx = execjs.compile(JS_CODE)

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track job progress
progress_status = {}
running_jobs = {}

# -----------------------------
# DB Helpers
# -----------------------------


def sanitize_column_name(col: str) -> str:
    """
    Clean a column name so it’s safe for PostgreSQL.
    Keeps only letters, numbers, and underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '_', col.strip().lower())


# -----------------------------
# Failed tracking helpers (per table)
# -----------------------------
def get_failed_file(table_name):
    return os.path.join(EXPORT_DIR, f"{table_name}_failed_features.json")


def load_failed(table_name):
    failed_file = get_failed_file(table_name)
    if os.path.exists(failed_file):
        with open(failed_file, "r") as f:
            return json.load(f)
    return []


def save_failed(table_name, failed_list):
    failed_file = get_failed_file(table_name)
    with open(failed_file, "w") as f:
        json.dump(failed_list, f)


def remove_failed_file(table_name):
    failed_file = get_failed_file(table_name)
    if os.path.exists(failed_file):
        os.remove(failed_file)

def parse_info_cdata(info_html):
    info = {}
    if not info_html.strip():
        return info
    soup = BeautifulSoup(info_html, "html.parser")
    for row in soup.find_all("tr"):
        key_tag = row.find("td", class_="ftrfld")
        val_tag = row.find("td", class_="ftrval")
        if key_tag and val_tag:
            key = key_tag.get_text(strip=True)
            val = val_tag.get_text(strip=True)
            info[key] = val
    return info

from threading import Lock

def fetch_feature(f_id, base_url, session, headers=None):
    """Fetch and parse a feature by ID (F).
    Returns:
        - (poly_tags, data_dict, True) for successful fetches (even if no polygons)
        - (None, None, False) for actual failures (network errors, parsing errors, etc.)
    """
    if headers is None:
        headers = {}

    url = base_url.format(f_id)

    try:
        resp = session.get(url, headers=headers, timeout=25)

        # Basic HTTP and content validation
        if resp is None or resp.status_code != 200:
            print(f"❌ Failed to fetch F={f_id} (HTTP {resp.status_code if resp else 'No Response'})")
            return None, None, False

        if not resp.text:
            print(f"⚠️ Empty response for F={f_id}")
            return None, None, False

        # Check if this looks like a valid XML response
        if "<![CDATA[" not in resp.text:
            print(f"⚠️ Invalid content for F={f_id} - no CDATA found")
            return None, None, False

        xml_text = resp.text.strip()
        if not xml_text:
            print(f"⚠️ No XML text for F={f_id}")
            return None, None, False

        # Parse XML safely
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as pe:
            print(f"⚠️ XML parse error for F={f_id}: {pe}")
            return None, None, False

        # Extract info section
        info_tag = root.find("info")
        info_html = info_tag.text if info_tag is not None and info_tag.text else ""
        data_dict = parse_info_cdata(info_html)

        # Extract all <poly> elements
        poly_tags = [poly.text for poly in root.findall("poly") if poly.text]

        # SUCCESS: Return data even if no polygons (empty list is valid)
        return poly_tags, data_dict, True

    except requests.exceptions.Timeout:
        print(f"⏰ Timeout fetching F={f_id}")
        return None, None, False
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error fetching F={f_id}")
        return None, None, False
    except requests.exceptions.RequestException as re:
        print(f"🌐 Network error fetching F={f_id}: {re}")
        return None, None, False
    except Exception as e:
        print(f"⚠️ Unexpected error fetching F={f_id}: {e}")
        return None, None, False


def run_job(job_id, base_url, start_f, max_f, table_name):
    """Run the job with multithreading and rich GeoJSON output."""
    start = time.time()
    all_features = []
    all_features_lock = Lock()
    os.makedirs(EXPORT_DIR, exist_ok=True)

    failed_list = load_failed(table_name)
    all_f_values = list(range(start_f, max_f + 1))
    if failed_list:
        print(f"🔁 Retrying previously failed features: {failed_list}")
        all_f_values = failed_list

    current_failed = []

    # Initialize progress counters
    progress_status[job_id]["total"] = len(all_f_values)
    progress_status[job_id]["completed"] = 0
    progress_status[job_id]["failed"] = 0

    # Setup persistent connection pool
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def process_f(f_val):
        if progress_status[job_id]["stop"]:
            return f"⏹️ Stopped before F={f_val}"

        poly_tags, data_dict, success = fetch_feature(f_val, base_url, session)

        if not success:
            # Actual failure (network error, parsing error, etc.)
            with all_features_lock:
                current_failed.append(f_val)
                progress_status[job_id]["failed"] += 1
            return f"❌ Failed F={f_val} (network/parsing error)"

        # SUCCESS: We got a valid response, even if no polygons
        if not poly_tags:
            # Valid response, but no polygons - this is a SUCCESS case
            print(f"ℹ️ No polygons for F={f_val} (valid response)")
            feature = {
                "type": "Feature",
                "properties": data_dict,
                "geometry": None
            }
            with all_features_lock:
                all_features.append(feature)
                progress_status[job_id]["completed"] += 1
            return f"✅ Processed F={f_val} (no polygons)"

        # SUCCESS: We have polygons to process
        features_local = []
        for poly_text in poly_tags:
            try:
                poly_result = ctx.call("parsePolyGeometry", poly_text)
                feature = {
                    "type": "Feature",
                    "properties": data_dict,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [poly_result["coordinates"]],
                    },
                }
                features_local.append(feature)
            except Exception as e:
                print(f"⚠️ Error parsing polygon for F={f_val}: {e}")
                # Continue with other polygons even if one fails

        with all_features_lock:
            all_features.extend(features_local)
            progress_status[job_id]["completed"] += 1

        polygon_count = len(features_local)
        return f"✅ Processed F={f_val} ({polygon_count} polygons)"

    # Run parallel fetches
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(process_f, f): f for f in all_f_values}
        for future in as_completed(futures):
            msg = future.result()
            progress_status[job_id]["logs"].append(msg)
            # Print to console for debugging
            print(msg)

    # Retry failed ones if needed
    retry_count = 0
    while current_failed and retry_count < 3:  # Limit retries to 3
        retry_count += 1
        to_retry = current_failed.copy()
        current_failed.clear()
        retry_msg = f"🔁 Retry #{retry_count} for {len(to_retry)} failed features"
        progress_status[job_id]["logs"].append(retry_msg)
        print(retry_msg)

        # Update total for retry batch
        progress_status[job_id]["total"] += len(to_retry)

        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = {executor.submit(process_f, f): f for f in to_retry}
            for future in as_completed(futures):
                msg = future.result()
                progress_status[job_id]["logs"].append(msg)
                print(msg)

    # Save final failed list if any remain
    if current_failed:
        save_failed(table_name, current_failed)
        failed_msg = f"💾 Saved {len(current_failed)} failed features for retry"
        progress_status[job_id]["logs"].append(failed_msg)
        print(failed_msg)
    else:
        remove_failed_file(table_name)

    # Export to /exports folder
    output_path = os.path.join(EXPORT_DIR, f"{table_name}.geojson")
    geojson_fc = {"type": "FeatureCollection", "features": all_features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_fc, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    completion_msg = f"🎉 Job completed in {elapsed:.2f}s. Processed {len(all_features)} features. Saved to {output_path}"
    progress_status[job_id]["done"] = True
    progress_status[job_id]["logs"].append(completion_msg)
    print(completion_msg)

    running_jobs.pop(job_id, None)





# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home_page():
    with open("frontend_new.html", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Extract Layers Logic
# -----------------------------
def get_layer_links_from_xml(xml_text):
    links = []
    try:
        root = ET.fromstring(xml_text)
        info_elem = root.find("info")
        if info_elem is not None and info_elem.text:
            cdata = info_elem.text
            matches = re.findall(r"fetchOverlay\('([^']+)'\)", cdata)
            links.extend(matches)
    except Exception as e:
        print("Error parsing XML:", e)
    return links

def normalize_table_name(base_url, name):
    # Get base domain without https:// and .wthgis.com
    domain = urllib.parse.urlparse(base_url).netloc.split('.')[0].lower()
    table_name = name.lower().replace(" ", "_")
    return f"{domain}_{table_name}"


def extract_layers_from_url(base_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Changed to True for better compatibility
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()
        layerlist_xml = None

        def handle_response(response):
            nonlocal layerlist_xml
            if "index.ashx?action=getLayerList" in response.url:
                try:
                    layerlist_xml = response.text()
                    print(f"✅ Found layer list XML: {len(layerlist_xml)} characters")
                except Exception as e:
                    print(f"❌ Error reading layer list response: {e}")

        page.on("response", handle_response)

        try:
            # Increased timeout and added more wait options
            print(f"🌐 Navigating to: {base_url}")
            page.goto(base_url, wait_until="domcontentloaded", timeout=60000)

            # Wait a bit for potential JavaScript to load
            page.wait_for_timeout(5000)

            # Try different selectors for the disclaimer
            disclaimer_selectors = [
                "input#disclaimerAccept",
                "#disclaimerAccept",
                "input[type='checkbox'][id*='disclaimer']",
                ".disclaimer-accept",
                "input[name*='disclaimer']"
            ]

            disclaimer_found = False
            for selector in disclaimer_selectors:
                if page.query_selector(selector):
                    print(f"✅ Found disclaimer with selector: {selector}")
                    page.click(selector)
                    page.wait_for_timeout(2000)
                    disclaimer_found = True
                    break

            if not disclaimer_found:
                print("ℹ️ No disclaimer found, continuing...")

            # Try different selectors for the browse button
            browse_selectors = [
                "td#btnIndex img[title='Browse map content']",
                "#btnIndex",
                "img[title*='Browse']",
                "img[alt*='Browse']",
                "td[onclick*='browse']",
                ".browse-button",
                "button[title*='Browse']"
            ]

            browse_clicked = False
            for selector in browse_selectors:
                if page.query_selector(selector):
                    print(f"✅ Found browse button with selector: {selector}")
                    page.click(selector)
                    page.wait_for_timeout(5000)
                    browse_clicked = True
                    break

            if not browse_clicked:
                print("❌ Could not find browse button, trying fallback...")
                # Fallback: try to trigger the layer list directly
                layer_list_url = f"{base_url.rstrip('/')}/index.ashx?action=getLayerList"
                print(f"🔄 Trying direct layer list URL: {layer_list_url}")
                try:
                    response = page.goto(layer_list_url, wait_until="networkidle", timeout=30000)
                    if response and response.status == 200:
                        layerlist_xml = response.text()
                        print("✅ Successfully fetched layer list directly")
                except Exception as e:
                    print(f"❌ Direct fetch failed: {e}")

            # Wait a bit more for any additional responses
            page.wait_for_timeout(3000)

        except Exception as e:
            print(f"❌ Error during page navigation: {e}")
            # Try to get the layer list directly as fallback
            try:
                layer_list_url = f"{base_url.rstrip('/')}/index.ashx?action=getLayerList"
                print(f"🔄 Fallback: trying direct layer list URL: {layer_list_url}")
                response = requests.get(layer_list_url, timeout=30)
                if response.status_code == 200:
                    layerlist_xml = response.text
                    print("✅ Successfully fetched layer list via direct request")
            except Exception as req_e:
                print(f"❌ Direct request also failed: {req_e}")

        finally:
            browser.close()

        if not layerlist_xml:
            print("❌ No layer list XML found")
            return []

        print(f"📄 Layer list XML length: {len(layerlist_xml)}")

        zoning_links = get_layer_links_from_xml(layerlist_xml)
        print(f"🔗 Found {len(zoning_links)} total layer links")

        # zoning_links = [l for l in layer_links if "Zoning" in l or "zoning" in l]
        # print(f"🏘️ Found {len(zoning_links)} zoning-related links")

        layers = []
        for link in zoning_links:
            full_url = urllib.parse.urljoin(base_url, link)
            print(f"🔍 Processing layer: {link}")
            try:
                resp = requests.get(full_url, timeout=30)
                if resp.status_code != 200:
                    print(f"❌ HTTP {resp.status_code} for {link}")
                    continue

                root = ET.fromstring(resp.text)

                # Extract D from URL
                dsid_match = re.search(r"dsid=(\d+)", link)
                D = dsid_match.group(1) if dsid_match else None

                # Extract F from <small> tag
                info_elem = root.find("info")
                F = None
                if info_elem is not None and info_elem.text:
                    match = re.search(r"<small>(.*?)</small>", info_elem.text, re.IGNORECASE)
                    if match:
                        F = match.group(1).strip()

                # Extract table name
                name_match = re.search(r"name=([^&]+)", link)
                raw_name = urllib.parse.unquote(name_match.group(1)) if name_match else "table"
                table_name = normalize_table_name(base_url, raw_name)

                layer_info = {
                    "name": raw_name,
                    "link": link,
                    "D": D,
                    "F": F,
                    "table_name": table_name
                }
                layers.append(layer_info)
                print(f"✅ Added layer: {raw_name} (D={D}, F={F})")

            except Exception as e:
                print(f"❌ Error processing layer {link}: {e}")
                continue

        print(f"🎉 Successfully processed {len(layers)} zoning layers")
        return layers


# -----------------------------
# API Endpoints
# -----------------------------

# Endpoint to extract layers
@app.post("/extract_layers")
def extract_layers(base_url: str = Form(...)):
    try:
        layers = extract_layers_from_url(base_url)
        if not layers:
            return {"error": "No layers found for the given URL."}
        return {"layers": layers}
    except Exception as e:
        return {"error": str(e)}

# Start job endpoint (updated to include layer selection)
@app.post("/start_job")
async def start_job(
    base_url: str = Form(...),
    selected_layer: str = Form(...),
    start_f: int = Form(...),
    max_f: int = Form(...),
    table_name: str = Form(...),
    D_value: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    if not background_tasks:
        return {"error": "BackgroundTasks not provided"}

    # Construct the correct layer URL using D_value
    layer_url = f"{base_url}/tgis/getftr.aspx?D={D_value}&Z=1&F={{}}"

    job_id = str(uuid.uuid4())

    # Initialize progress tracking with proper counters
    progress_status[job_id] = {
        "completed": 0,
        "failed": 0,
        "total": (max_f - start_f + 1),
        "stop": False,
        "logs": [],
        "done": False
    }

    running_jobs[job_id] = {
        "table_name": table_name,
        "start_f": start_f,
        "max_f": max_f
    }

    # Start the job in background
    background_tasks.add_task(run_job, job_id, layer_url, start_f, max_f, table_name)
    print(f'Layer URL: {layer_url}')
    return {"job_id": job_id, "message": f"Job started for table {table_name}"}

@app.post("/stop_job/{job_id}")
async def stop_job(job_id: str):
    if job_id in progress_status:
        progress_status[job_id]["stop"] = True
        return {"status": "stopping"}
    return {"status": "not_found"}

@app.get("/jobs")
def list_jobs():
    return {"running_jobs": running_jobs, "all_jobs": progress_status}


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(progress_status[job_id])
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"WebSocket closed for job {job_id}")


from fastapi.responses import FileResponse

# -----------------------------
# GeoJSON Files Listing & Download
# -----------------------------
@app.get("/geojson_list")
def list_geojson_files():
    files = [f for f in os.listdir(EXPORT_DIR) if f.endswith(".geojson")]
    return {"files": sorted(files)}

@app.get("/download/{filename}")
def download_geojson_file(filename: str):
    filepath = os.path.join(EXPORT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="application/geo+json", filename=filename)
    return {"error": "File not found"}


#uvicorn app:app --reload
