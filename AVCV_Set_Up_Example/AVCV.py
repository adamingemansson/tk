"""Interactive imaging viewer.

Features:
    - Multi-channel time series (optionally z-stacks) with synchronous display.
    - Unified image processing: percentile normalization, background subtraction,
        pure additive RGB color mapping and optional per-channel x-offset.
    - Tracking mode: navigate IDs, plot normalized fluorescence intensity (FI) traces
        with optional comparison / coverage segmentation.
    - Detection mode: display detection points for selected sources, click-to-center.
    - Dynamic channel activation and per-channel patch views for quick inspection.
    - Keyboard + mouse interaction: zoom, pan (W/A/S/D + drag), frame & z navigation.

Core pipeline is centralized in process_fluorescence_image to guarantee visual consistency.
The rest of the code focuses on data orchestration, UI events and caching.
"""

# === Imports & configuration ===
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from matplotlib.widgets import Slider, TextBox, CheckButtons, Button, RadioButtons
import ast
from functools import lru_cache
import Comparison as cp

# FOR THE COMMON USER - ONLY THE FOLLOWING PART NEEDS TO BE ADAPTED
#-------------------------------------------------------------------------------------

ROOT_DIR = "/Users/adamingemansson/presentation_250812/"

# File paths to Base Dataset or Comparison file
BASE_FILES = [
    {"file": "CME_tracks.csv", "name": "CME"},
    {"file": "Dino_tracks.csv", "name": "Dino"},
]

# Configuration for the datasets to compare. IGNORE if comparison is not needed
SEC_FILES = [
    {"file": "Dino.csv", "name": "Dino"},
    {"file": "CME.csv", "name": "CME"}
]

# Detection configuration
DETECTION_CONFIG = [
    {"name": "CME", "file": "detections_CME.csv", "color": "blue"},
    {"name": "Dino", "file": "detections_Dino.csv", "color": "red"},
]

# Channel/Volume configuration
CHANNEL_CONFIG = [
    {"name": "Channel 1", "path": "Channel_1_folder", "color": "white"},  # Choose from: "red", "green", "blue", "white", "magenta", "cyan", "yellow", "orange"
    {"name": "Channel 2", "path": "Channel_2_folder", "color": "green"}
]

# Channel offset in pixels (x-direction)
CHANNEL_OFFSET = 0  # IGNORE if not needed

#-------------------------------------------------------------------------------------

# Configure Matplotlib keymap for saving
matplotlib.rcParams["keymap.save"] = ["ctrl+s"]

# Constants for movement and zoom
WASD_STEP = 1.0
DRAG_SENSITIVITY = 0.5
ZOOM_FACTOR = 1.1
COMPARISON_THRESHOLD = 3.5

# === Initialize State and Load Data ===

def setup_folder_structure_for_path(channel_path):
    """Ensure expected timepoint folder structure for a channel.

    If flat images exist directly inside the channel directory they are moved into
    numbered subfolders (001, 002, ...). For each timepoint a max projection
    file is created if missing (works for single 2D images and multi-page stacks)."""
    items = os.listdir(channel_path)
    allowed_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    tif_files = [f for f in items if f.lower().endswith(allowed_exts)]
    subdirs = [d for d in items if os.path.isdir(os.path.join(channel_path, d))]
    # If there are .tif files but no subfolders - create subfolders
    if tif_files and not subdirs:
        print(f"Creating subfolders for {len(tif_files)} .tif files in {channel_path}...")
        for i, tif_file in enumerate(sorted(tif_files)):
            folder_name = f"{i+1:03d}"
            folder_path = os.path.join(channel_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            old_path = os.path.join(channel_path, tif_file)
            new_path = os.path.join(folder_path, tif_file)
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
            create_maxproj_if_needed(folder_path, tif_file)
    # Also check existing folders for missing max projection files
    elif subdirs:
        print(f"Checking {len(subdirs)} existing subfolders in {channel_path} for missing maxproj files...")
        for subdir in subdirs:
            folder_path = os.path.join(channel_path, subdir)
            tif_file = find_tif_file(folder_path)
            if tif_file:
                create_maxproj_if_needed(folder_path, tif_file)

def create_maxproj_if_needed(folder_path, tif_filename):
    """Create a _maxproj.tif sibling if it does not already exist.

    Single 2D image: copied as-is.
    Multi-page TIFF: maximum projection over pages.
    Other formats (png/jpg): read and saved out as TIFF for consistency."""
    base_name = os.path.splitext(tif_filename)[0]
    maxproj_filename = f"{base_name}_maxproj.tif"
    maxproj_path = os.path.join(folder_path, maxproj_filename)
    # If max projection already exists, skip
    if os.path.exists(maxproj_path):
        return
    # Create max projection from the volume
    volume_path = os.path.join(folder_path, tif_filename)
    try:
        print(f"Creating maxproj: {maxproj_filename}")
        ext = os.path.splitext(tif_filename)[1].lower()
        if ext in (".tif", ".tiff"):
            with tifffile.TiffFile(volume_path) as tif:
                if len(tif.pages) == 1:
                    # Single page TIFF -> copy as maxproj
                    img = tif.pages[0].asarray()
                    tifffile.imwrite(maxproj_path, img)
                else:
                    # Max projection across z-pages
                    volume = np.array([page.asarray() for page in tif.pages])
                    maxproj = np.max(volume, axis=0)
                    tifffile.imwrite(maxproj_path, maxproj)
        else:
            # Non-TIFF (png/jpg): read and write as TIFF maxproj
            img = plt.imread(volume_path)
            tifffile.imwrite(maxproj_path, img)
        print(f"Created: {maxproj_filename}")
    except Exception as e:
        print(f"Error creating maxproj for {tif_filename}: {e}")

def find_tif_file(folder_path):
    """Return first non-maxproj image (tif/tiff/png/jpg/jpeg) or None."""
    try:
        allowed_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_exts) and "maxproj" not in f.lower()]
        return files[0] if files else None
    except Exception:
        return None

def find_maxproj_file(folder_path):
    """Return max projection filename in folder or None."""
    try:
        files = os.listdir(folder_path)
        maxproj_files = [f for f in files if f.endswith(".tif") and "maxproj" in f.lower()]
        return maxproj_files[0] if maxproj_files else None
    except Exception:
        return None

class State:
    """Holds all runtime state for the viewer.

    Aggregates:
      - Channel metadata (folders, colors, active flags)
      - Tracking / comparison / coverage DataFrames and derived ID lists
      - Detection datasets and selection state
      - View / zoom / pan / frame / z-plane parameters
      - Caches for image data and UI flags

    Acts as a single source of truth accessed by callbacks and rendering functions.
    """

    def __init__(self):
        """Initializer `__init__`.

        Args:
            self: See usage in code.

        Returns:
            None: Operates via side effects or returns implicit values if noted in code.
        """
        # Constructor logic; sets up state and UI components.
        # Process channels
        # Channel discovery & color assignment: build per-channel source list and assign display colors.
        self.channel_sources = []
        default_colors = ["red", "green", "blue", "white", "magenta", "cyan", "yellow", "orange"]
        color_index = 0
        for config in CHANNEL_CONFIG:
            channel_path = config["path"]
            if not os.path.exists(channel_path):
                print(f"Warning: Channel path does not exist: {channel_path}")
                continue
            setup_folder_structure_for_path(channel_path)
            folders = sorted([f for f in os.listdir(channel_path) if os.path.isdir(os.path.join(channel_path, f))])
            if not folders:
                print(f"Warning: No folders found in {channel_path}")
                continue
            try:
                folder_path = os.path.join(channel_path, folders[0])
                tif_file = find_tif_file(folder_path)
                if tif_file:
                    path = os.path.join(folder_path, tif_file)
                    with tifffile.TiffFile(path) as tif:
                        img = tif.pages[0].asarray()
                        n_channels = img.shape[2] if img.ndim == 3 else 1
                else:
                    n_channels = 1
            except Exception:
                n_channels = 1
            if "color" in config and config["color"]:
                base_color = config["color"]
            else:
                base_color = default_colors[color_index % len(default_colors)]
                color_index += 1
            if n_channels > 1:
                for ch_idx in range(n_channels):
                    if "color" in config and config["color"]:
                        channel_color = config["color"]
                    else:
                        channel_color = base_color if ch_idx == 0 else default_colors[(color_index + ch_idx - 1) % len(default_colors)]
                    self.channel_sources.append({
                        "name": f"{config["name"]}_Ch{ch_idx+1}",
                        "path": channel_path,
                        "folders": folders,
                        "channel_idx": ch_idx,
                        "color": channel_color
                    })
                if "color" not in config or not config["color"]:
                    color_index += n_channels - 1
            else:
                self.channel_sources.append({
                    "name": config["name"],
                    "path": channel_path,
                    "folders": folders,
                    "channel_idx": 0,
                    "color": base_color
                })
            print(f"Added channel(s) from {config["name"]}: {n_channels} channel(s), {len(folders)} timepoints")
        if not self.channel_sources:
            raise RuntimeError("No valid channels found in CHANNEL_CONFIG!")
        self.folders = self.channel_sources[0]["folders"]
        self.first_frame = 1
        self.max_frame = len(self.folders)
        self.n_channels = len(self.channel_sources)
        self.channel_labels = [ch["name"] for ch in self.channel_sources]
        self.cmaps = [ch["color"] for ch in self.channel_sources]
        print(f"Total channels configured: {self.n_channels}")
        for ch in self.channel_sources:
            print(f"  Channel: {ch["name"]} - {ch["color"]} - {len(ch["folders"])} timepoints")
        # Initialize channel on/off states: show first channel by default for safer brightness scaling.
        self.active_channels = {label: (i == 0) for i, label in enumerate(self.channel_labels)}
        # Populate base/secondary dataset and (re)build comparison/coverage (may create CSVs lazily).
        self.base_selection(0)
        try:
        # Try to load tracking/comparison/coverage CSVs; if missing, tracking is disabled gracefully.
            ok, err = self.set_tracking_sources(self.comparison_file, self.coverage_file)
            if not ok and err:
                print(f"Tracking disabled: {err}")
        except Exception as e:
            print(f"Failed to initialize tracking sources: {e}")
        # Detection datasets: collect available detection CSVs and default selections.
        self.detection_data = {}
        self.available_detections = []
        for config in DETECTION_CONFIG:
            try:
                df = pd.read_csv(config["file"])
                self.detection_data[config["name"]] = {"data": df, "color": config["color"]}
                self.available_detections.append(config["name"])
                print(f"Loaded {config["name"]} detections from {config["file"]} - {len(df)} detections")
            except FileNotFoundError:
                print(f"Warning: {config["file"]} not found - {config["name"]} detections not available")
        self.active_detections = {label: False for label in self.available_detections}
        self.selected_detections = {}
        self.box_size = 20.0
        self.current_z_plane = 0
        self.z_tol = 2.0
        self.view_center = [None, None]
        self.last_click_position = [None, None]
        self.center_view = False
        self.current_frame = 1
        # Determine a FIXED global number of z-planes (pages) across channels.
        # We look at the first timepoint folder for every configured channel and take the max
        # number of TIFF pages. Internal indexing will be 0-based (0 .. max_z_planes-1).
        self.max_z_planes = 1
        try:
            for ch in self.channel_sources:
                if not ch["folders"]:
                    continue
                folder_path = os.path.join(ch["path"], ch["folders"][0])
                tif_file = find_tif_file(folder_path)
                if not tif_file:
                    continue
                path = os.path.join(folder_path, tif_file)
                with tifffile.TiffFile(path) as tif:
                    pages = len(tif.pages)
                    if pages > self.max_z_planes:
                        self.max_z_planes = pages
        except Exception:
            pass
        if not self.has_tracking:
            try:
                if self.folders:
                    folder_path = os.path.join(self.channel_sources[0]["path"], self.folders[0])
                    tif_file = find_tif_file(folder_path)
                    if tif_file:
                        path = os.path.join(folder_path, tif_file)
                        with tifffile.TiffFile(path) as tif:
                            img = tif.pages[0].asarray()
                            h, w = img.shape[:2] if img.ndim != 2 else img.shape
                            self.view_center[0] = w // 2
                            self.view_center[1] = h // 2
                            self.last_click_position[0] = w // 2
                            self.last_click_position[1] = h // 2
                    else:
                        self.view_center[0] = 256
                        self.view_center[1] = 256
                        self.last_click_position[0] = 256
                        self.last_click_position[1] = 256
                else:
                    self.view_center[0] = 256
                    self.view_center[1] = 256
                    self.last_click_position[0] = 256
                    self.last_click_position[1] = 256
            except Exception:
                self.view_center[0] = 256
                self.view_center[1] = 256
                self.last_click_position[0] = 256
                self.last_click_position[1] = 256
        # Caches used by LRU wrappers; also used to store precomputed projections if needed.
        self.maxproj_cache = {}
        self.slice_cache = {}
        # View parameters: store zoom & pan extents for main image axis.
        self.main_zoom = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
        self.main_img_obj = None
        self.rect_patch = None
        self.is_panning = False
        self.pan_start = {"x": 0, "y": 0, "xlim": (0, 1), "ylim": (0, 1)}
        self.mouse_press_pos = None
        # Default view flags and playback state.
        self.show_maxproj = True
        self.show_channels = False
        self.show_zoom_maxproj = True
        self.is_playing = False
        self.fps = 5.0
        self.manual_z_override = False
    
    def base_selection(self,idx):
        """Select active base / secondary dataset (radio button) and (re)build comparison / coverage.

        Handles lazy generation of comparison & coverage CSVs if they do not yet exist."""
        global BASE_FILE, SEC_FILE, BASE_NAME, SEC_NAME
        # Resolve active base dataset path/name by index.
        self.base_file = BASE_FILES[idx]["file"]
        self.base_name = BASE_FILES[idx]["name"]
        try:
            self.sec_file = SEC_FILES[idx]["file"]
            self.sec_name = SEC_FILES[idx]["name"]
        except Exception:
            self.sec_file = ""
            self.sec_name = ""
        # Derive comparison/coverage file paths from selected base/secondary names.
        # Compute expected output CSV paths for comparison and per-ID coverage.
        self.comparison_file = os.path.join(ROOT_DIR, f"Comparison_{self.base_name}_vs_{self.sec_name}.csv")
        self.tl_coverage_file = os.path.join(ROOT_DIR, f"ID_Coverage_{self.base_name}_vs_{self.sec_name}.csv")
        # Lazy generation: if comparison CSV is missing, compute or fall back to base file.
        if not os.path.isfile(self.comparison_file):
            if not os.path.isfile(self.sec_file):
                self.comparison_file = self.base_file
            else:
                self.comparison_file = cp.comp(self.base_file, self.base_name, self.sec_file, self.sec_name, COMPARISON_THRESHOLD)
                self.comparison_file.to_csv(os.path.join(ROOT_DIR, f"Comparison_{self.base_name}_vs_{self.sec_name}.csv"), index=False)
        # Accept both path and in-memory DataFrame; normalize to DataFrame.
        if isinstance(self.comparison_file, pd.DataFrame):
            df = self.comparison_file
        else:
            df = pd.read_csv(self.comparison_file)
        if os.path.isfile(self.tl_coverage_file) and df.shape[1] > 8:
            self.coverage_file = self.tl_coverage_file
        elif df.shape[1] > 8:
            self.coverage_file = cp.cov(self.comparison_file, self.base_name, self.sec_name, COMPARISON_THRESHOLD)
            self.coverage_file.to_csv(os.path.join(ROOT_DIR, f"ID_Coverage_{self.base_name}_vs_{self.sec_name}.csv"), index=False)
        else:
            self.coverage_file = None
        self.set_tracking_sources(self.comparison_file, self.coverage_file)

    def set_tracking_sources(self, comparison, coverage=None):
        """Load or switch tracking (comparison) and optional coverage data.

        Parameters
        ----------
        comparison : (str|pd.DataFrame)
            Path or DataFrame containing tracking / comparison columns (ID, coordinates, FI, etc.).
        coverage : (str|pd.DataFrame|None)
            Optional coverage segmentation data.

        Returns
        -------
        (bool, str|None)
            (success_flag, error_message_if_any)."""

        # Helper to load a DataFrame from path or pass-through
        def _to_df(obj):
            if obj is None:
                return None, "None provided"
            if isinstance(obj, str):
                """Helper `_to_df`.

                Args:
                    obj: See usage in code.

                Returns:
                    None: Operates via side effects or returns implicit values if noted in code.
                """
                # See docstring for purpose and flow.
                try:
                    if "os" in globals():
                        if not os.path.isfile(obj):
                            return None, f"File not found: {obj}"
                    return pd.read_csv(obj), None
                except Exception as e:
                    return None, f"Failed to read CSV: {e}"
            # Assume DataFrame-like
            return obj.copy() if hasattr(obj, "copy") else obj, None

        # Reset base-related state
        self.has_tracking = False
        self.has_comparison = False
        self.base = None
        self.sec = None
        self.id_col_base = None
        self.id_col_sec = None
        self.FI_col = None
        self.Comp_files = []
        self.Comp = None
        self.all_ids = []
        self.current_id = None
        if not hasattr(self, "base_channel_idx"):
            self.base_channel_idx = 0

        comp_df, err = _to_df(comparison)
        if comp_df is None:
            # Fall back to detection click mode
            self.click_mode = True
            return False, err or "Invalid comparison input"

        # Determine available bases from ID columns
        try:
            base_cols = [c for c in comp_df.columns if c.startswith("ID")]
            available_bases = []
            for c in base_cols:
                if c == "ID":
                    available_bases.append("")
                elif c.startswith("ID (") and c.endswith(")"):
                    available_bases.append(c[4:-1])
        except Exception:
            available_bases = []

        if len(available_bases) == 0:
            # No tracking columns found -> click mode
            self.click_mode = True
            return False, "No ID columns found in comparison data"

        # Tracking available
        self.has_tracking = True
        self.has_comparison = len(available_bases) >= 2
        self.base = available_bases[0]
        self.id_col_base = "ID" if self.base == "" else f"ID ({self.base})"
        self.FI_col = "FI" if self.base == "" else f"FI ({self.base})"
        if self.has_comparison:
            self.sec = available_bases[1] if len(available_bases) > 1 else None
            self.id_col_sec = f"ID ({self.sec})" if self.sec else None
        else:
            self.sec = None
            self.id_col_sec = None

        # Duplicate comp per channel and select active
        n_files = max(1, getattr(self, "n_channels", 1))
        for _ in range(n_files):
            try:
                self.Comp_files.append(comp_df.copy())
            except Exception:
                self.Comp_files.append(comp_df)
        self.base_channel_idx = max(0, min(self.base_channel_idx, len(self.Comp_files) - 1))
        self.Comp = self.Comp_files[self.base_channel_idx]

        # Build ID list
        try:
            import numpy as np
            if self.id_col_base in getattr(self, "Comp", {}).columns:
                ids = np.unique(self.Comp[self.id_col_base].dropna())
                self.all_ids = ids
                self.current_id = int(ids[0]) if len(ids) else None
            else:
                self.all_ids = []
                self.current_id = None
        except Exception:
            self.all_ids = []
            self.current_id = None

        # Load coverage if provided
        cov_df, _ = _to_df(coverage) if coverage is not None else (None, None)
        self.Cov = cov_df if cov_df is not None else None

        # Switch UI mode and clear caches so new data will display
        self.click_mode = False
        try:
            if "load_maxproj_cached" in globals():
            # Invalidate max-projection cache when base/secondary or channel toggles affect image sources.
                load_maxproj_cached.cache_clear()
            if "load_slice_cached" in globals():
            # Invalidate slice cache when z or input folders change to avoid stale tiles.
                load_slice_cached.cache_clear()
        except Exception:
            pass

        return True, None

# Instantiate the state
state = State()

# === Optimized Image Loading with LRU Cache ===

# Cached max-projection per (frame, channel_idx); evicts least-recently-used beyond 20 entries.
@lru_cache(maxsize=20)
def load_maxproj_cached(frame, channel_idx=0):
    """Load max projection (cached) for requested frame & channel.

    Returns a 2D numpy array (single channel) or None if unavailable."""
    t = int(frame)
    channel_idx = int(channel_idx)
    if channel_idx >= len(state.channel_sources):
        return None
    channel = state.channel_sources[channel_idx]
    idx = t - state.first_frame
    if idx < 0 or idx >= len(channel["folders"]):
        return None
    folder = channel["folders"][idx]
    folder_path = os.path.join(channel["path"], folder)
    # Try maxproj file first
    maxproj_file = find_maxproj_file(folder_path)
    if maxproj_file:
        maxproj_path = os.path.join(folder_path, maxproj_file)
        try:
            img = tifffile.imread(maxproj_path)
            # Extract specific channel if multi-channel
            if img.ndim == 3 and channel["channel_idx"] < img.shape[2]:
                return img[..., channel["channel_idx"]]
            else:
                return img
        except Exception as e:
            print(f"Error loading maxproj file {maxproj_file}: {e}")
    # Fallback: create from volume
    tif_file = find_tif_file(folder_path)
    if tif_file:
        path = os.path.join(folder_path, tif_file)
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff"):
                with tifffile.TiffFile(path) as tif:
                    if len(tif.pages) == 1:
                        img = tif.pages[0].asarray()
                    else:
                        volume = np.array([page.asarray() for page in tif.pages])
                        img = np.max(volume, axis=0)
            else:
                img = plt.imread(path)
            # Extract specific channel if multi-channel
            if img.ndim == 3 and channel["channel_idx"] < img.shape[2]:
                return img[..., channel["channel_idx"]]
            else:
                return img
        except Exception as e:
            print(f"Error creating maxproj for frame {t}: {e}")
    return None

# Cached slice per (frame, z, channel_idx); evicts least-recently-used beyond 30 entries.
@lru_cache(maxsize=30)
def load_slice_cached(frame, z, channel_idx=0):
    """Load a single z slice (cached) for a frame & channel.

    Falls back safely if slice index exceeds stack depth."""
    t = int(frame)
    z = int(z)
    channel_idx = int(channel_idx)
    if channel_idx >= len(state.channel_sources):
        return None
    channel = state.channel_sources[channel_idx]
    idx = t - state.first_frame
    if idx < 0 or idx >= len(channel["folders"]):
        return None
    folder = channel["folders"][idx]
    folder_path = os.path.join(channel["path"], folder)
    tif_file = find_tif_file(folder_path)
    if tif_file:
        path = os.path.join(folder_path, tif_file)
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff"):
                with tifffile.TiffFile(path) as tif:
                    # Get the slice
                    if len(tif.pages) == 1:
                        img = tif.pages[0].asarray()
                    elif z < len(tif.pages):
                        img = tif.pages[z].asarray()
                    else:
                        img = tif.pages[-1].asarray()
            else:
                # 2D non-tiff image: treat as single slice
                img = plt.imread(path)
            # Extract specific channel if multi-channel
            if img.ndim == 3 and channel["channel_idx"] < img.shape[2]:
                return img[..., channel["channel_idx"]]
            else:
                return img
        except Exception as e:
            print(f"Error loading slice {z} in frame {t}: {e}")
    return None

# === Utility Functions ===

def extract_centered_patch_float(img, center_x, center_y, size):
    """Extract a square patch centered at (x,y) (float allowed).

    Regions extending beyond image boundaries are zero padded. Returns (patch, left, right, top, bottom)."""
    # Support both grayscale (2D) and color (3D) inputs.
    if img.ndim == 2:
        H, W = img.shape
        C = 1
    else:
        H, W, C = img.shape
    # Compute half window size; allow non-integer centers for smooth panning.
    half = size / 2.0
    left = center_x - half
    right = center_x + half
    top = center_y - half
    bottom = center_y + half
    x1 = int(np.floor(left))
    x2 = int(np.ceil(right))
    y1 = int(np.floor(top))
    y2 = int(np.ceil(bottom))
    if img.ndim == 2:
        # Allocate output patch (with padding) before copying valid pixels.
        patch = np.zeros((y2 - y1, x2 - x1), dtype=img.dtype)
    else:
        patch = np.zeros((y2 - y1, x2 - x1, C), dtype=img.dtype)
    x_start = max(0, -x1)
    x_end = (x2 - x1) - max(0, x2 - W)
    y_start = max(0, -y1)
    y_end = (y2 - y1) - max(0, y2 - H)
    bx1 = max(0, x1)
    bx2 = min(W, x2)
    by1 = max(0, y1)
    by2 = min(H, y2)
    if img.ndim == 2:
        patch[y_start:y_end, x_start:x_end] = img[by1:by2, bx1:bx2]
    else:
        patch[y_start:y_end, x_start:x_end, :] = img[by1:by2, bx1:bx2, :]
    return patch, left, right, top, bottom

def parse_color(val):
    """Parse color literals stored as strings (e.g. "(1,0,0)") into tuples if possible."""
    if isinstance(val, str):
        try:
            c = ast.literal_eval(val)
            if isinstance(c, (list, tuple)) and (3 <= len(c) <= state.n_channels):
                return tuple(c)
            else:
                return val
        except Exception:
            return val
    return val

def process_fluorescence_image(images_with_cmaps, percentile=99, multiplier=2, background_percentile=10):
    """Convert one or multiple raw channel images into a single processed RGB image.

    Pipeline per channel:
      1. Percentile-based upper scaling (percentile * multiplier).
      2. Clip to [0, 0.8] to avoid harsh saturation.
      3. Per-channel background estimate (low percentile) and subtraction.
      4. Renormalize to [0,1].
      5. Map to pure additive RGB color (black background guaranteed).
      6. Place channel with optional x-offset (CHANNEL_OFFSET).
    Post multi-channel merge:
      7. Global background subtraction (same percentile) to suppress residual haze.
      8. Final normalization to [0,1].

    Parameters
    ----------
    images_with_cmaps : list[(idx, image, color)] or tuple
        Each element holds a 2D array and a simple color name.
    percentile : float
        High percentile used for dynamic range normalization (default 99).
    multiplier : float
        Multiplier applied to percentile value to extend headroom.
    background_percentile : float
        Low percentile used for background estimation.
    """
    # Handle single channel input
    if not isinstance(images_with_cmaps, list):
        images_with_cmaps = [images_with_cmaps]
    
    if not images_with_cmaps:
        return None
    
    # Get image shape from first image and calculate total width with offsets
    _, first_img, _ = images_with_cmaps[0]
    base_height, base_width = first_img.shape
    
    # Calculate total width needed for all channels with offsets
    num_channels = len(images_with_cmaps)
    total_width = base_width + (num_channels - 1) * CHANNEL_OFFSET
    
    # Create RGB image with expanded width
    rgb_image = np.zeros((base_height, total_width, 3), dtype=np.float64)
    
    # Process each channel with x-offset
    for ch_order, (ch_idx, img, cmap_name) in enumerate(images_with_cmaps):
        # Calculate x-offset for this channel
        x_offset = ch_order * CHANNEL_OFFSET
        
        # Step 1: Normalize using percentile method (like original)
        img_max = multiplier * np.percentile(img, percentile) if img.size else 1
        img_normalized = np.clip(img / (img_max + 1e-8), 0, 0.8)
        
        # Step 2: Background correction per channel
        background = np.percentile(img_normalized, background_percentile)
        
        # Step 3: Subtract background and clip negative values
        img_corrected = np.maximum(0, img_normalized - background)
        
        # Step 4: Renormalize to 0-1 range
        if img_corrected.max() > 0:
            img_corrected = img_corrected / img_corrected.max()
        
        # Step 5: Convert to RGB based on color name - PURE COLOR MAPPING WITH BLACK BACKGROUND
        # Define all color mappings directly
        color_mapping = {
            "red": [1, 0, 0],        # Pure red
            "green": [0, 1, 0],      # Pure green  
            "blue": [0, 0, 1],       # Pure blue
            "white": [1, 1, 1],      # White (all components)
            "grey": [1, 1, 1],       # Grey mapped as white
            "gray": [1, 1, 1],       # Gray mapped as white
            "magenta": [1, 0, 1],    # Magenta = Red + Blue
            "cyan": [0, 1, 1],       # Cyan = Green + Blue  
            "yellow": [1, 1, 0],     # Yellow = Red + Green
            "orange": [1, 0.5, 0],   # Orange = Red + half Green
            "purple": [0.5, 0, 1],   # Purple = half Red + Blue
            "pink": [1, 0.5, 0.5],   # Pink = Red + half Green + half Blue
            "lime": [0.5, 1, 0],     # Lime = half Red + Green
            "turquoise": [0, 1, 0.5] # Turquoise = Green + half Blue
        }
        
        # Apply color mapping with x-offset
        color_name = cmap_name.lower()
        if color_name in color_mapping:
            rgb_values = color_mapping[color_name]
            # Apply to region with x-offset
            end_x = min(total_width, base_width + x_offset)
            for c in range(3):
                rgb_image[:, x_offset:end_x, c] += img_corrected * rgb_values[c]
        else:
            # Fallback: default to white if color not found
            end_x = min(total_width, base_width + x_offset)
            for c in range(3):
                rgb_image[:, x_offset:end_x, c] += img_corrected
    
    # Step 6: Final background removal on combined image (if multiple channels)
    if len(images_with_cmaps) > 1:
        combined_background = np.percentile(rgb_image, background_percentile)
        rgb_image = np.maximum(0, rgb_image - combined_background)
    
    # Step 7: Final normalization for display
    if rgb_image.max() > 0:
        rgb_image = rgb_image / rgb_image.max()
    
    return rgb_image

# === Fluorescence Intensity Data Functions ===

def highlight_axes(ax, active=True):
    """Visually indicate channel activation state (yellow = active, gray = inactive)."""
    color = "yellow" if active else "gray"
    width = 2 if active else 1
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(width)

def get_first_active_channel_idx():
    """Return index of first active channel or None if none active."""
    for idx, label in enumerate(state.channel_labels):
        if state.active_channels[label]:
            return idx
    return None

def FI(ID, Comparison, Coverage, base_name):
    """Fetch fluorescence intensity vectors for a single ID.

    If coverage segmentation exists, returns segmented FI subsets; otherwise just full trace."""
    if Comparison is None:
        return [], [], [], [], []
    ID_comp = Comparison[Comparison[state.id_col_base] == ID]
    if Coverage is not None and not Coverage.empty:
        try:
            ID_cov = Coverage[Coverage[Coverage.columns[0]] == ID]
            if not ID_cov.empty:
                comp = ID_comp.to_numpy()
                cov = ID_cov.to_numpy()
                data_str = str(cov[0][4])
                segments = [row.split(",") for row in data_str.split(";")] if data_str else []
                y_data = np.array(segments, float) if segments else np.array([])
                if y_data.size > 0:
                    FI_y = np.zeros(len(y_data), dtype=object)
                    t_y = np.zeros(len(y_data), dtype=object)
                    for i in range(len(y_data) - 1):
                        start_idx = int(y_data[i][1] - 1)
                        end_idx = int(y_data[i+1][1] - 1)
                        FI_y[i] = comp[start_idx:end_idx, 12]
                        t_y[i] = comp[start_idx:end_idx, 8]
                    FI_y[-1] = comp[int(y_data[-1][1] - 1):, 12]
                    t_y[-1] = comp[int(y_data[-1][1] - 1):, 8]
                    ID_y = y_data[:, 0]
                    FI_x = comp[:, 11]
                    t_full = comp[:, 8]
                    return t_full, FI_x, t_y, ID_y, FI_y
        except Exception as e:
            print(f"Error processing coverage data for ID {ID}: {e}")
    # Fallback: if coverage not used, return FI and t columns directly if available
    if state.FI_col in ID_comp.columns and "t" in ID_comp.columns:
        t_full = ID_comp["t"].values
        FI_x = ID_comp[state.FI_col].values
        return t_full, FI_x, [], [], []
    return [], [], [], [], []

def FI_plot(ID_array, data, t_current=None, current_color=None, ax=None,
        base_name="", sec_name="", is_single_tracking=False):
    """Render FI curve(s) plus optional secondary segments and a time marker.

    Marker color: current_color if exact timepoint exists, else gray (nearest)."""
    ids = np.atleast_1d(ID_array)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    axes = [ax]

    for plot_idx, ID in enumerate(ids):
        t_full, FI_x, t_y, ID_y, FI_y = data
        axx = axes[plot_idx]

        # SINGLE TRACKING
        if is_single_tracking or len(FI_y) == 0:
            t_full = np.array(t_full, float)
            FI_x = np.array(FI_x, float)
            if not FI_x.size:
                axx.text(0.5, 0.5, f"No data for ID {ID}",
                         ha="center", va="center", transform=axx.transAxes)
                continue
            FI_x_norm = (FI_x - FI_x.min()) / (FI_x.max() - FI_x.min())

            # Plot bas (hantera hopp >1 i tid)
            if t_full.size:
    # Break lines at time gaps > 1 to avoid drawing misleading connections.
                gaps = np.where(np.diff(t_full) > 1)[0] + 1
                segments = np.split(np.arange(t_full.size), gaps) if gaps.size else [np.arange(t_full.size)]
                for si, seg in enumerate(segments):
                    if len(seg) == 1:
                        xx = [t_full[seg[0]] - 0.1, t_full[seg[0]] + 0.1]
                        yy = [FI_x_norm[seg[0]], FI_x_norm[seg[0]]]
                        axx.plot(xx, yy, color="red", linewidth=2,
                                 label=f"{base_name} ID: {int(ID)}" if si == 0 else None)
                    else:
                        axx.plot(t_full[seg], FI_x_norm[seg], color="red", linewidth=2,
                                 label=f"{base_name} ID: {int(ID)}" if si == 0 else None)

            # Scatter (base value)
            if t_current is not None and t_full.size:
                exact = np.where(t_full == t_current)[0]
                if exact.size:
                    idx_marker = exact[0]
                    marker_col = current_color if current_color else "red"
                else:
                    idx_marker = np.argmin(np.abs(t_full - t_current))
                    marker_col = "gray"
                # Draw point markers at exact or nearest time index to emphasize current frame.
                axx.scatter(t_full[idx_marker], FI_x_norm[idx_marker],
                            color=marker_col, s=60, zorder=50)

            axx.set_ylabel("Normalized FI")
            axx.set(xlim=(0, state.max_frame), xlabel="Time")
            axx.legend(labelspacing=0.2, fontsize=8)
            continue

        # COMPARISON MODE
        # Base
        t_full = np.array(t_full, float)
        FI_x = np.array(FI_x, float)
        if not FI_x.size:
            axx.text(0.5, 0.5, f"No data for ID {ID}",
                     ha="center", va="center", transform=axx.transAxes)
            continue
        FI_x_norm = (FI_x - FI_x.min()) / (FI_x.max() - FI_x.min()) if FI_x.max() > FI_x.min() else FI_x

        # Plot base
        if t_full.size:
            gaps = np.where(np.diff(t_full) > 1)[0] + 1
            segments = np.split(np.arange(t_full.size), gaps) if gaps.size else [np.arange(t_full.size)]
            for si, seg in enumerate(segments):
                if len(seg) == 1:
                    xx = [t_full[seg[0]] - 0.1, t_full[seg[0]] + 0.1]
                    yy = [FI_x_norm[seg[0]], FI_x_norm[seg[0]]]
                    axx.plot(xx, yy, color="red",
                             label=f"{base_name} ID: {int(ID)}" if si == 0 else None)
                else:
                    axx.plot(t_full[seg], FI_x_norm[seg], color="red",
                             label=f"{base_name} ID: {int(ID)}" if si == 0 else None)

        # Secondary segment colors
        uniq_ids = np.unique(ID_y) if len(ID_y) else []
        color_map = {}
        if len(uniq_ids):
            for i_u, sid in enumerate(uniq_ids):
                color_map[sid] = plt.cm.gist_rainbow((i_u + 1) / (len(uniq_ids) + 1))

            # Plot secondary segments, normalize FI_y globally over all segments
            # Collect all FI_y values into an array
            all_fi_y = np.concatenate([np.array(fy, float) for fy in FI_y if len(fy) > 0]) if len(FI_y) else np.array([])
            if all_fi_y.size > 0 and all_fi_y.max() > all_fi_y.min():
                global_min = all_fi_y.min()
                global_max = all_fi_y.max()
            else:
                global_min = 0.0
                global_max = 1.0
            plotted_sec = set()
            for seg_index, sec_id in enumerate(ID_y):
                x_vals = np.array(t_y[seg_index], float)
                y_vals_raw = np.array(FI_y[seg_index], float)
                if not x_vals.size or not y_vals_raw.size:
                    continue
                # Normalize with global min/max
                if global_max > global_min:
                    y_norm = (y_vals_raw - global_min) / (global_max - global_min)
                else:
                    y_norm = y_vals_raw
                c_sec = color_map.get(sec_id, (0, 0.6, 1, 1))
                label = f"{sec_name} ID: {int(sec_id)}" if (sec_name and sec_id not in plotted_sec) else None
                gaps2 = np.where(np.diff(x_vals) > 1)[0] + 1
                segs2 = np.split(np.arange(x_vals.size), gaps2) if gaps2.size else [np.arange(x_vals.size)]
                for si, s2 in enumerate(segs2):
                    if len(s2) == 1:
                        xx = [x_vals[s2[0]] - 0.1, x_vals[s2[0]] + 0.1]
                        yy = [y_norm[s2[0]], y_norm[s2[0]]]
                        axx.plot(xx, yy, color=c_sec, label=label if si == 0 else None)
                    else:
                        axx.plot(x_vals[s2], y_norm[s2], color=c_sec, label=label if si == 0 else None)
                plotted_sec.add(sec_id)

        # Scatter: ALWAYS at the base"s normalized value
        if t_current is not None and t_full.size:
            exact = np.where(t_full == t_current)[0]
            if exact.size:
                idx_marker = exact[0]
                marker_color = current_color if current_color else "red"
            else:
                idx_marker = np.argmin(np.abs(t_full - t_current))
                marker_color = "gray"
            axx.scatter(t_full[idx_marker], FI_x_norm[idx_marker],
                        color=marker_color, s=60, zorder=50)

        axx.set_ylabel("FI (norm)")
        axx.set(xlim=(0, state.max_frame), xlabel="Time")
        axx.legend(labelspacing=0.2, fontsize=8)

    for extra_ax in axes[len(ids):]:
        extra_ax.axis("off")
    if ax is None:
        plt.show()

# === Detection Plotting Functions ===

def get_column_name(column_type, data_df=None):
    """Resolve column name with preference for exact match; fallback to partial match.

    Time ("t") requires exact match from a whitelist to avoid ambiguity."""
    df = data_df if data_df is not None else state.Comp
    if df is None:
        return None
    # Special handling for time columns - only exact matches
    if column_type == "t":
        for exact_name in ["t", "t0", "timepoint", "time", "frame"]:
            if exact_name in df.columns:
                return exact_name
        return None
    # For other columns: exact match first, then partial
    if column_type in df.columns:
        return column_type
    # Fall back to partial match (for parentheses format)
    for col in df.columns:
        if column_type in col:
            return col
    return None

def filter_detections(det_data, frame, use_maxproj=True, current_z=None, z_tolerance=None):
    """Return detections for current frame (and z-window if exploring single slices)."""
    if det_data.empty:
        return pd.DataFrame()
    # Find time column using the unified function
    time_col = get_column_name("t", det_data)
    if time_col is None:
        print(f"Warning: No time column found in detection data. Available columns: {list(det_data.columns)}")
        return pd.DataFrame()
    # Filter by time
    det_t = det_data[det_data[time_col] == frame]
    # Apply z-filtering if needed
    if not use_maxproj and z_tolerance is not None and current_z is not None:
        z_col = get_column_name("z", det_data)
        if z_col:
            # Convert detection z (assumed 1-based in CSV) to 0-based for comparison.
            z_vals0 = det_t[z_col].astype(float) - 1.0
            det_t = det_t[(z_vals0 >= current_z - z_tolerance) & (z_vals0 <= current_z + z_tolerance)]
    return det_t

def plot_detections_in_patch(ax, frame, left, right, top, bottom, use_maxproj, z_tolerance=None):
    """Scatter detections restricted to patch extent (with optional z filtering)."""
    frame = int(frame)
    has_labels = False
    for det_name, det_info in state.selected_detections.items():
        det_df = det_info["data"]
        det_color = det_info["color"]
        det_t = filter_detections(det_df, frame, use_maxproj=use_maxproj, current_z=state.current_z_plane, z_tolerance=z_tolerance)
        if det_t.empty:
            continue
        # Find x and y columns using the unified function
        x_col = get_column_name("x", det_t)
        y_col = get_column_name("y", det_t)
        if not x_col or not y_col:
            print(f"Warning: No x/y columns found for {det_name}. Available: {list(det_t.columns)}")
            continue
        # Filter to patch region
        mask = (det_t[x_col] >= left) & (det_t[x_col] < right) & (det_t[y_col] >= top) & (det_t[y_col] < bottom)
        close_det = det_t[mask]
        if not close_det.empty:
            ax.scatter(close_det[x_col], close_det[y_col], color=det_color, s=100, label=f"{det_name} Det", zorder=5)
            has_labels = True
    if has_labels:
        ax.legend(fontsize=8, loc="upper right")

def plot_full(ax, frame, box_size, ID_comp_df, current_id, base_name, sec_name,
              sec_color, z_tol_val, use_maxproj=True, patch_center=None):
    """Construct zoom patch (tracking or click mode) with channel composite + markers."""
    frame = int(frame)

    # Help: which channels are active
    active_idxs = [i for i, lbl in enumerate(state.channel_labels) if state.active_channels[lbl]]
    if not active_idxs:
        ax.clear()
        ax.text(0.5, 0.5, "Activate a channel", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    # Detection / Click mode (no tracking or click mode)
    if not state.has_tracking or state.click_mode:
        if not patch_center or patch_center[0] is None or patch_center[1] is None:
            ax.clear()
            ax.text(0.5, 0.5, "Click main image to center view",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        x_patch, y_patch = patch_center
        # Collect patches for universal processing
        patch_channels_data = []
        left = right = top = bottom = None
        for j, ch_idx in enumerate(active_idxs):
            img = load_maxproj_cached(frame, ch_idx) if use_maxproj else load_slice_cached(frame, state.current_z_plane, ch_idx)
            if img is None:
                continue
            patch_img, l, r, t, b = extract_centered_patch_float(img, x_patch, y_patch, int(box_size))
            if patch_img.size == 0:
                continue
            if j == 0:
                left, right, top, bottom = l, r, t, b
            patch_channels_data.append((ch_idx, patch_img, state.cmaps[ch_idx]))
        
        if not patch_channels_data:
            ax.clear(); ax.axis("off"); return
        
        ax.clear()
        # Use universal fluorescence processing function
        processed_patch = process_fluorescence_image(patch_channels_data)
        if processed_patch is not None:
            ax.imshow(processed_patch, extent=[left, right, bottom, top],
                      origin="upper", interpolation="nearest")
        # Detections
        plot_detections_in_patch(ax, frame, left, right, top, bottom, use_maxproj, z_tol_val)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.01, 0.99, f"Center: ({x_patch:.1f}, {y_patch:.1f})",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round", fc="black", alpha=0.65),
                transform=ax.transAxes, va="top")
        return

    # Tracking mode
    if current_id is None or ID_comp_df.empty:
        ax.clear()
        ax.text(0.5, 0.5, "No ID selected", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    id_col = get_column_name("ID")
    x_col_base = get_column_name("x")
    y_col_base = get_column_name("y")
    z_col_base = get_column_name("z")
    if not all([id_col, x_col_base, y_col_base]):
        ax.clear()
        ax.text(0.5, 0.5, "Missing data columns", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    ID_data = ID_comp_df[ID_comp_df[id_col] == current_id]
    if ID_data.empty:
        ax.clear(); ax.axis("off"); return

    # Base exact row
    row_exact_base = ID_data[ID_data["t"] == frame]
    if not row_exact_base.empty:
        rb = row_exact_base.iloc[0]
        x_base = float(rb[x_col_base]); y_base = float(rb[y_col_base])
        z_base = float(rb[z_col_base]) if (z_col_base and z_col_base in row_exact_base.columns) else state.current_z_plane
        base_exact = True
    else:
        x_base = y_base = None
        z_base = state.current_z_plane
        base_exact = False

    # Secondary point
    sec_point = None
    if state.has_comparison and sec_name and state.id_col_sec and state.id_col_sec in ID_data.columns:
        x_col_sec = f"x ({sec_name})" if f"x ({sec_name})" in ID_data.columns else None
        y_col_sec = f"y ({sec_name})" if f"y ({sec_name})" in ID_data.columns else None
        if x_col_sec and y_col_sec:
            row_exact_sec = ID_data[ID_data["t"] == frame]
            if not row_exact_sec.empty:
                rs = row_exact_sec.iloc[0]
                try:
                    sx = float(rs[x_col_sec]); sy = float(rs[y_col_sec])
                    if not (np.isnan(sx) or np.isnan(sy)):
                        sec_point = (sx, sy)
                except Exception:
                    sec_point = None

    # Center
    if patch_center and patch_center[0] is not None:
        x_patch, y_patch = patch_center
    else:
        if x_base is not None:
            x_patch, y_patch = x_base, y_base
        elif sec_point is not None:
            x_patch, y_patch = sec_point
        else:
            ax.clear()
            ax.text(0.5, 0.5, "No center defined", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return

    # Load patches for all active channels
    patch_channels_data = []
    left = right = top = bottom = None
    # z-plane is based on the first active channel (or the base z)
    z_plane = int(round(z_base))
    for j, ch_idx in enumerate(active_idxs):
        img = load_maxproj_cached(frame, ch_idx) if use_maxproj else load_slice_cached(frame, z_plane, ch_idx)
        if img is None:
            continue
        patch_img, l, r, t, b = extract_centered_patch_float(img, x_patch, y_patch, int(box_size))
        if patch_img.size == 0:
            continue
        if j == 0:
            left, right, top, bottom = l, r, t, b
        patch_channels_data.append((ch_idx, patch_img, state.cmaps[ch_idx]))
    
    if not patch_channels_data:
        ax.clear(); ax.axis("off"); return

    ax.clear()
    # Use universal fluorescence processing function
    processed_patch = process_fluorescence_image(patch_channels_data)
    if processed_patch is not None:
        ax.imshow(processed_patch, extent=[left, right, bottom, top],
                  origin="upper", interpolation="nearest")

    # Detections
    plot_detections_in_patch(ax, frame, left, right, top, bottom, use_maxproj, z_tol_val)


    # Base
    if base_exact and x_base is not None and left <= x_base < right and top <= y_base < bottom:
        ax.scatter(x_base, y_base, color="darkred", s = 110, zorder=10,
                   label=f"{base_name}" if base_name else "Base")

    # Secondary
    if sec_point is not None:
        sx, sy = sec_point
        if left <= sx < right and top <= sy < bottom:
            ax.scatter(sx, sy, color=sec_color, s=110, zorder=12,
                       edgecolors="none",
                       label=f"{sec_name}" if sec_name else "Sec")

    # Legend
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right")

    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.01, 0.99,
            f"z = {int(round(z_base))}\nID: {int(current_id)}",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
            transform=ax.transAxes, va="top")

# === Setup Main Figure and UI Elements ===

fig = plt.figure(figsize=(15, 8))
fig.canvas.manager.set_window_title("Adam's Very Cool Viewer (AVCV)")

# Checkbox and control axes
ax_show_channels = fig.add_axes([0.86, 0.937, 0.1, 0.04])
state.show_channels = False
show_channels_checkbox = CheckButtons(ax_show_channels, ["Show channels"], [state.show_channels])

if state.has_tracking:
    ax_click_mode = fig.add_axes([0.29, 0.89, 0.1, 0.04])
    click_mode_checkbox = CheckButtons(ax_click_mode, ["Click Mode"], [state.click_mode])
    ax_maxproj = fig.add_axes([0.15, 0.89, 0.1, 0.04])
else:
    ax_maxproj = fig.add_axes([0.15, 0.89, 0.1, 0.04])

maxproj_checkbox = CheckButtons(ax_maxproj, ["Max Projection"], [True])
state.show_maxproj = True

ax_show_detections = fig.add_axes([0.01, 0.35, 0.05, len(DETECTION_CONFIG)*0.02])
detection_states = [state.active_detections[label] for label in state.available_detections]
show_detections_checkbox = CheckButtons(ax_show_detections, state.available_detections, detection_states)
ax_show_detections.set_title("Displayed\nDetections", pad=15)

ax_channel_select = fig.add_axes([0.01, 0.5, 0.05, len(CHANNEL_CONFIG)*0.02])
channel_states = [state.active_channels[label] for label in state.channel_labels]
channel_checkbox = CheckButtons(ax_channel_select, state.channel_labels, channel_states)
ax_channel_select.set_title("Displayed\nChannels", pad=15)

main_ax = fig.add_axes([0.07, 0.15, 0.4, 0.75])

# FI plot axes (only visible in tracking mode)
# Dedicated axes for FI time-series; toggled visible only in tracking mode.
FI_ax = fig.add_axes([0.51, 0.18, 0.2, 0.28])
if not state.has_tracking:
    # Hide FI axes when tracking is unavailable.
    FI_ax.set_visible(False)

zoom_ax = fig.add_axes([0.475, 0.50, 0.27, 0.39])
axbox = fig.add_axes([0.22, 0.94, 0.09, 0.035])
ax_boxsize = fig.add_axes([0.5, 0.94, 0.05, 0.035])
ax_ztol = fig.add_axes([0.58, 0.94, 0.05, 0.035])
ax_frame = fig.add_axes([0.16, 0.09, 0.31, 0.04])

state.main_zoom = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
state.main_img_obj = None
state.rect_patch = None

ax_zoom_maxproj = fig.add_axes([0.64, 0.937, 0.1, 0.04])
zoom_maxproj_checkbox = CheckButtons(ax_zoom_maxproj, ["Max Projection"], [True])
state.show_zoom_maxproj = True

ax_center_toggle = fig.add_axes([0.75, 0.937, 0.1, 0.04])
center_checkbox = CheckButtons(ax_center_toggle, ["Center view"], [state.center_view])

ax_base_select = fig.add_axes([0.01, 0.650, 0.05, len(BASE_FILES)*0.03])
# Radio for selecting base file (one button per BASE_FILES entry)
_base_file_names = [bf.get("name", f"Base {i+1}") for i, bf in enumerate(BASE_FILES)]
base_radio = RadioButtons(ax_base_select, _base_file_names, active=0)
ax_base_select.set_title("Base\nFile", pad=15)

ax_z_slider = fig.add_axes([0.07, 0.02, 0.4, 0.02])

def _on_base_file_change(label):
    """Switch base/secondary selection and refresh tracking sources when radio changes."""
    try:
        idx = _base_file_names.index(label)
    except ValueError:
        print(f"Unknown base label: {label}")
        return
    state.base_selection(idx)
base_radio.on_clicked(_on_base_file_change)

channel_patch_axes = []
def create_channel_patch_axes():
    """Create/refresh per-channel patch axes (side panels)."""
    global channel_patch_axes
    # Remove old axes
    for ax in channel_patch_axes:
        try:
            fig.delaxes(ax)
        except Exception:
            pass
    channel_patch_axes = []
    n = len(state.channel_labels)
    n_rows = 3
    n_cols = (n + n_rows - 1) // n_rows
    left0 = 0.73
    width = 0.13
    height = 0.22
    h_gap = 0.01
    v_gap = 0.04
    for i, ch in enumerate(state.channel_labels):
        col = i // n_rows
        row = i % n_rows
        left = left0 + col * (width + h_gap)
        bottom = 0.67 - row * (height + v_gap)
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_title(f"Channel {ch}", fontsize=11)
        channel_patch_axes.append(ax)

def plot_channel_patches(frame, patch_center, box_size_val, z_tol_val, use_maxproj=True):
    """Render individual channel patches (if enabled) using unified processing."""
    if not state.show_channels:
        for ax in channel_patch_axes:
            ax.clear()
            ax.axis("off")
        fig.canvas.draw_idle()
        return
    frame = int(frame)
    if patch_center[0] is None or patch_center[1] is None:
        for ax in channel_patch_axes:
            ax.clear()
            ax.axis("off")
        return
    if len(channel_patch_axes) != len(state.channel_labels):
        create_channel_patch_axes()
    x_patch, y_patch = patch_center
    # Plot each channel separately
    for i, ch_label in enumerate(state.channel_labels):
        if i >= len(channel_patch_axes):
            break
        ax = channel_patch_axes[i]
        ax.clear()
        # Load image for this channel
        slice_img = load_maxproj_cached(frame, i) if use_maxproj else load_slice_cached(frame, state.current_z_plane, i)
        if slice_img is None:
            ax.axis("off")
            ax.set_title(f"{ch_label} - No Data", fontsize=10)
            continue
        patch, left, right, top, bottom = extract_centered_patch_float(slice_img, x_patch, y_patch, int(box_size_val))
        # Use universal fluorescence processing function
        processed_patch = process_fluorescence_image([(i, patch, state.cmaps[i])])
        if processed_patch is not None:
            ax.imshow(processed_patch, extent=[left, right, top, bottom], origin="upper", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ch_label}", fontsize=10)
        # Highlight if this channel is active
        highlight_axes(ax, state.active_channels[ch_label])
    fig.canvas.draw_idle()

# === Update and Plot Functions ===

def update_view_center():
    """Update current view center based on ID trajectory (exact > previous > future > None)."""
    if not state.has_tracking or state.click_mode:
        if state.last_click_position[0] is not None and state.last_click_position[1] is not None:
            state.view_center[0] = state.last_click_position[0]
            state.view_center[1] = state.last_click_position[1]
        return
    if state.current_id is None or state.Comp is None:
        return

    frame = state.current_frame
    id_col = get_column_name("ID")
    x_col = get_column_name("x")
    y_col = get_column_name("y")
    if not all([id_col, x_col, y_col]):
        return

    df = state.Comp
    id_mask = (df[id_col] == state.current_id)

    # 1. Exact row
    exact_mask = id_mask & (df["t"] == frame)
    if exact_mask.any():
        state.view_center[0] = float(df.loc[exact_mask, x_col].values[0])
        state.view_center[1] = float(df.loc[exact_mask, y_col].values[0])
        return

    # 2. Latest previous
    prev_mask = id_mask & (df["t"] < frame)
    if prev_mask.any():
        prev_df = df.loc[prev_mask]
        last_t = prev_df["t"].max()
        row = prev_df[prev_df["t"] == last_t].iloc[0]
        state.view_center[0] = float(row[x_col])
        state.view_center[1] = float(row[y_col])
        return

    # 3. Earliest future
    future_mask = id_mask & (df["t"] > frame)
    if future_mask.any():
        future_df = df.loc[future_mask]
        first_t = future_df["t"].min()
        row = future_df[future_df["t"] == first_t].iloc[0]
        state.view_center[0] = float(row[x_col])
        state.view_center[1] = float(row[y_col])
        return

    # 4. No data
    state.view_center[0] = None
    state.view_center[1] = None

def update_plot(*args):
    """Full UI refresh: main composite, zoom patch, FI curve, detections and channel patches."""
    frame = state.current_frame
    sec_color = "red"
    ID_Comp = pd.DataFrame()
    if state.has_tracking and state.current_id is not None and not state.click_mode:
        FI_ax.set_visible(True); FI_ax.axis("on")
        choice = state.Comp[state.Comp[state.id_col_base] == state.current_id]
        ID_Comp = choice
        if state.has_comparison and state.sec:
            FI_data = FI(state.current_id, state.Comp, state.Cov, state.base)
            t_full, FI_x, t_y, ID_y, FI_y = FI_data
            if len(ID_y):
                uniq_ids = np.unique(ID_y)
                colors_list = [plt.cm.gist_rainbow(i / (len(uniq_ids) + 1))
                               for i in range(len(uniq_ids) + 1)] if len(uniq_ids) else ["blue"]
                # find color for secondary that has data exactly now
                chosen = False
                for seg_index, sec_id in enumerate(ID_y):
                    x_vals = np.array(t_y[seg_index], float)
                    if x_vals.size and frame in x_vals:
                        color_idx = min(np.where(uniq_ids == sec_id)[0][0] + 1, len(colors_list) - 1)
                        sec_color = colors_list[color_idx]
                        chosen = True
                        break
                if not chosen:
                    sec_color = "gray"
            else:
                sec_color = "gray"
        else:
            # single tracking: red if exact row, otherwise gray
            choice_t = choice[choice["t"] == frame]
            sec_color = "red" if not choice_t.empty else "gray"

        FI_ax.clear()
        if not (state.has_comparison and state.sec):
            FI_data = FI(state.current_id, state.Comp, state.Cov, state.base)
        FI_plot(state.current_id, FI_data, t_current=frame, current_color=sec_color,
                ax=FI_ax, base_name=state.base, sec_name=state.sec,
                is_single_tracking=(not state.has_comparison))
        FI_ax.set_xlim(0, state.max_frame)
        FI_ax.set_ylim(0, 1.1)
    else:
        FI_ax.clear(); FI_ax.axis("off"); FI_ax.set_visible(False)
        sec_color = "red"
        ID_Comp = pd.DataFrame()

    # Slider color
    try:
        slider_frame.poly.set_color(sec_color); slider_frame._handle.set_color(sec_color)
    except Exception:
        pass
    try:
        z_slider.poly.set_color(sec_color); z_slider._handle.set_color(sec_color)
    except Exception:
        pass

    # Main image
    main_ax.clear(); main_ax.axis("on")
    
    # Collect all active channels for universal processing
    active_channels_data = []
    for i, ch_label in enumerate(state.channel_labels):
        if not state.active_channels[ch_label]:
            continue
        img = load_maxproj_cached(frame, i) if state.show_maxproj else load_slice_cached(frame, state.current_z_plane, i)
        if img is None:
            continue
        active_channels_data.append((i, img, state.cmaps[i]))
    
    if active_channels_data:
        # Use universal fluorescence processing function
        processed_image = process_fluorescence_image(active_channels_data)
        if processed_image is not None:
            state.main_img_obj = [main_ax.imshow(processed_image, origin="upper",
                                               interpolation="nearest")]
        else:
            state.main_img_obj = None
    else:
        state.main_img_obj = None
    main_ax.set_xlim(state.main_zoom["xmin"], state.main_zoom["xmax"])
    main_ax.set_ylim(state.main_zoom["ymin"], state.main_zoom["ymax"])
    main_ax.set_aspect("equal")

    # Zoom-rektangel
    if state.view_center[0] is not None and state.view_center[1] is not None:
        size = float(state.box_size)
        x_c, y_c = state.view_center
        rect = patches.Rectangle((x_c - size/2, y_c - size/2),
                                 size, size, linewidth=2,
                                 edgecolor=sec_color, facecolor="none")
        state.rect_patch = rect
        main_ax.add_patch(rect)
    main_ax.set_xticks([]); main_ax.set_yticks([])

    # Zoom patch
    zoom_ax.clear()
    if state.has_tracking and not state.click_mode:
        plot_full(zoom_ax, frame, state.box_size, ID_Comp, state.current_id,
                  state.base_name, state.sec_name, sec_color, state.z_tol,
                  use_maxproj=state.show_zoom_maxproj, patch_center=state.view_center)
    else:
        plot_full(zoom_ax, frame, state.box_size, pd.DataFrame(), None,
                  state.base_name, state.sec_name, sec_color, state.z_tol,
                  use_maxproj=state.show_zoom_maxproj, patch_center=state.view_center)
    zoom_ax.set_aspect("equal")

    # Detections
    ch_idx = get_first_active_channel_idx()
    if ch_idx is not None:
        bg = load_maxproj_cached(frame, ch_idx) if state.show_maxproj else load_slice_cached(frame, state.current_z_plane, ch_idx)
        if bg is not None:
            dot_factor = (bg.shape[0] + bg.shape[1])
            xlim = main_ax.get_xlim(); ylim = main_ax.get_ylim()
            cur_size = (xlim[1] - xlim[0] + ylim[0] - ylim[1])
            dot_size = dot_factor / cur_size if cur_size != 0 else dot_factor
            labels = []
            for det_name, det_info in state.selected_detections.items():
                det_df = det_info["data"]; det_color = det_info["color"]
                det_t = filter_detections(det_df, frame, use_maxproj=state.show_maxproj,
                                            current_z=state.current_z_plane, z_tolerance=state.z_tol)
                if det_t.empty:
                    continue
                x_col = get_column_name("x", det_t); y_col = get_column_name("y", det_t)
                if not x_col or not y_col:
                    continue
                main_ax.scatter(det_t[x_col], det_t[y_col], color=det_color, s=dot_size,
                                label=f"{det_name} Detections")
                labels.append(det_name)
            if labels:
                main_ax.legend(fontsize=8, markerscale=2, loc="upper right")

    plot_channel_patches(frame, state.view_center, state.box_size, state.z_tol,
                         use_maxproj=state.show_zoom_maxproj)
    fig.canvas.draw_idle()
#

def update_z_plane():
    """Adjust z plane toward natural tracked z for current ID (when not in max projection)."""
    if not state.has_tracking or state.current_id is None or state.click_mode:
        return
    frame = state.current_frame
    id_val = state.current_id
    # Get correct column names using the unified function
    id_col = get_column_name("ID")
    z_col = get_column_name("z")
    if not id_col:
        return
    idx = (state.Comp["t"] == frame) & (state.Comp[id_col] == id_val)
    if any(idx) and z_col:
        z_val = int(round(state.Comp.loc[idx, z_col].values[0])) - 1  # convert to 0-based
    else:
        subdf = state.Comp[(state.Comp[id_col] == id_val) & (state.Comp["t"] <= frame)]
        if not subdf.empty and z_col:
            last_t = subdf["t"].max()
            z_val = int(round(subdf[subdf["t"] == last_t][z_col].values[0])) - 1
        else:
            z_val = 25
    # Clamp within valid range
    z_val = max(0, min(state.max_z_planes - 1, z_val))
    state.current_z_plane = z_val
    if not state.show_maxproj:
        z_slider.set_val(z_val)

# === Event Handlers for UI Interactions ===

def channel_checkbox_callback(label):
    state.active_channels[label] = not state.active_channels[label]
    create_channel_patch_axes()
    update_plot()

channel_checkbox.on_clicked(channel_checkbox_callback)

def on_show_channels_toggle(label):
    state.show_channels = not state.show_channels
    update_plot()

show_channels_checkbox.on_clicked(on_show_channels_toggle)

if state.has_tracking:
    def on_click_mode_toggle(label):
        state.click_mode = not state.click_mode
        if state.click_mode and state.center_view:
            """UI callback `channel_checkbox_callback`.

            Args:
                label: See usage in code.

            Returns:
                None: Operates via side effects or returns implicit values if noted in code.
            """
            # UI event handler: updates state and refreshes relevant artists.
            state.center_view = True
            center_checkbox.set_active(0)
        update_plot()

    click_mode_checkbox.on_clicked(on_click_mode_toggle)

def on_maxproj_toggle(label):
    state.show_maxproj = not state.show_maxproj
    ax_z_slider.set_visible(not state.show_maxproj)
    if not state.show_maxproj:
        if not state.has_tracking:
            state.current_z_plane = state.max_z_planes // 2
            z_slider.set_val(state.current_z_plane)
        else:
            update_z_plane()
    fig.canvas.draw_idle()
    update_plot()

maxproj_checkbox.on_clicked(on_maxproj_toggle)

def on_show_detections_toggle(label):
    """UI callback `on_show_channels_toggle`.

    Args:
        label: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    state.active_detections[label] = not state.active_detections[label]
    state.selected_detections = {k: v for k, v in state.detection_data.items() if state.active_detections[k]}
    update_plot()

show_detections_checkbox.on_clicked(on_show_detections_toggle)

def on_zoom_maxproj_toggle(label):
    state.show_zoom_maxproj = not state.show_zoom_maxproj
    update_plot()

zoom_maxproj_checkbox.on_clicked(on_zoom_maxproj_toggle)

def on_center_toggle(label):
    """UI callback `on_click_mode_toggle`.

    Args:
        label: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    state.center_view = not state.center_view
    if state.center_view:
        update_view_center()
    update_plot()

center_checkbox.on_clicked(on_center_toggle)

def on_id_submit(text):
    if not state.has_tracking or state.click_mode:
        return
    try:
        val = int(text.strip())
        if val in state.all_ids:
            """UI callback `on_maxproj_toggle`.

            Args:
                label: See usage in code.

            Returns:
                None: Operates via side effects or returns implicit values if noted in code.
            """
            # UI event handler: updates state and refreshes relevant artists.
            state.current_id = val
            t_vals = sorted({t for t in state.Comp[state.Comp[state.id_col_base] == val]["t"].values})
            if not t_vals:
                state.current_frame = 1
            elif state.current_frame not in t_vals:
                state.current_frame = t_vals[0]
            slider_frame.set_val(state.current_frame)
            update_view_center()
            if not state.show_maxproj:
                update_z_plane()
            update_plot()
        else:
            print("ID not tracked")
    except Exception as e:
        print("Input a valid ID", e)


if state.has_tracking:
    text_box = TextBox(axbox, f"{state.base} ID:", initial="")
    text_box.on_submit(on_id_submit)
else:
    text_box = TextBox(axbox, "Detection Mode", initial="Click image")
    text_box.set_active(False)

def on_boxsize_submit(text):
    """UI callback `on_show_detections_toggle`.

    Args:
        label: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    try:
        val = float(text.strip())
        if val >= 1:
            state.box_size = val
            update_plot()
    except Exception:
        pass

boxsize_box = TextBox(ax_boxsize, "Box Size:", initial=int(state.box_size))
boxsize_box.on_submit(on_boxsize_submit)

def on_ztol_submit(text):
    """UI callback `on_zoom_maxproj_toggle`.

    Args:
        label: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    try:
        val = float(text.strip())
        if val >= 0:
            state.z_tol = val
            update_plot()
    except Exception:
        pass

ztol_box = TextBox(ax_ztol, "z tol:", initial=int(state.z_tol))
ztol_box.on_submit(on_ztol_submit)

slider_frame = Slider(ax_frame, "Time", 1, state.max_frame, valinit=state.current_frame, valstep=1)
slider_frame.vline.set_visible(False)
state.is_playing = False
state.fps = 5.0

ax_play = fig.add_axes([0.07, 0.09, 0.055, 0.04])
play_button = Button(ax_play, "Play", color="0.9", hovercolor="0.7")
ax_fps = fig.add_axes([0.07, 0.045, 0.07, 0.03])
fps_box = TextBox(ax_fps, "FPS:", initial=int(state.fps))

def on_center_toggle(label):
    """UI callback `on_center_toggle`.

    Args:
        label: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    state.center_view = not state.center_view
    if state.center_view:
        update_view_center()
    update_plot()

center_checkbox.on_clicked(on_center_toggle)

def play_loop(event=None):
    state.is_playing = True
    play_button.label.set_text("Pause")
    if state.current_frame == state.max_frame:
        state.current_frame = 1
        slider_frame.set_val(1)
    for frame in range(state.current_frame + 1, state.max_frame + 1):
        if not state.is_playing:
            break
        state.current_frame = frame
        slider_frame.set_val(frame)
        plt.pause(1.0 / state.fps)
    play_button.label.set_text("Play")
    state.is_playing = False

def toggle_play(event):
    """UI callback `on_id_submit`.

    Args:
        text: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    if state.is_playing:
        state.is_playing = False
        play_button.label.set_text("Play")
    else:
        play_loop()

play_button.on_clicked(toggle_play)

def on_slider_change(val):
    state.current_frame = int(val)
    if state.center_view:
        if state.has_tracking and not state.click_mode:
            update_view_center()
        elif not state.has_tracking:
            update_view_center()
    if not state.show_maxproj and state.has_tracking and not state.click_mode:
        update_z_plane()
    update_plot()

slider_frame.on_changed(on_slider_change)
fps_box.on_submit(lambda text: setattr(state, "fps", float(text.strip())) if text.strip().isdigit() and float(text) > 0 else None)



def on_z_change(val):
    state.current_z_plane = int(val)
    if not state.has_tracking:
        update_plot()
        return
    frame = state.current_frame
    id_val = state.current_id
    if id_val is not None and f"z ({state.base})" in state.Comp.columns and not state.click_mode:
        """UI callback `on_boxsize_submit`.

        Args:
            text: See usage in code.

        Returns:
            None: Operates via side effects or returns implicit values if noted in code.
        """
        # UI event handler: updates state and refreshes relevant artists.
        idx = (state.Comp["t"] == frame) & (state.Comp[state.id_col_base] == id_val)
        if any(idx):
            natural_z = int(round(state.Comp.loc[idx, f"z ({state.base})"].values[0])) - 1
        else:
            subdf = state.Comp[(state.Comp[state.id_col_base] == id_val) & (state.Comp["t"] <= frame)]
            natural_z = int(round(subdf[subdf["t"] == subdf["t"].max()][f"z ({state.base})"].values[0])) - 1 if not subdf.empty else 0
        natural_z = max(0, min(state.max_z_planes - 1, natural_z))
    update_plot()

# Z slider: fixed range 0..max_z_planes-1 (0-based). Does not change after init.
z_slider = Slider(ax_z_slider, "z:", 0, state.max_z_planes - 1, valinit=state.current_z_plane, valstep=1)
ax_z_slider.set_visible(False)
z_slider.vline.set_visible(False)
z_slider.on_changed(on_z_change)

def on_key(event):
    movement = {
        "w": (0, -WASD_STEP), "s": (0, WASD_STEP),
        "a": (-WASD_STEP, 0), "d": (WASD_STEP, 0)
    }
    if event.key in movement and not state.center_view:
        """UI callback `on_ztol_submit`.

        Args:
            text: See usage in code.

        Returns:
            None: Operates via side effects or returns implicit values if noted in code.
        """
        # UI event handler: updates state and refreshes relevant artists.
        if state.view_center[0] is None or state.view_center[1] is None:
            return
        dx, dy = movement[event.key]
        state.view_center[0] += dx
        state.view_center[1] += dy
        update_plot()
    elif event.key in ["left", "right"]:
        if ax_z_slider.get_visible() and event.inaxes == ax_z_slider:
            new_z = state.current_z_plane + (-1 if event.key == "left" else 1)
            if event.key == "left":
                new_z = max(0, new_z)
            else:
                new_z = min(state.max_z_planes, new_z)
            state.current_z_plane = new_z
            z_slider.set_val(new_z)
        else:
            new_frame = state.current_frame + (-1 if event.key == "left" else 1)
            if event.key == "left":
                new_frame = max(1, new_frame)
            else:
                new_frame = min(state.max_frame, new_frame)
            state.current_frame = new_frame
            slider_frame.set_val(new_frame)
            if state.center_view and (not state.has_tracking or not state.click_mode):
                update_view_center()
            if not state.show_maxproj and state.has_tracking and not state.click_mode:
                update_z_plane()
        update_plot()

fig.canvas.mpl_connect("key_press_event", on_key)

def on_scroll(event):
    """Function `play_loop`.

    Args:
        event: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # See docstring for purpose and flow.
    if event.inaxes != main_ax:
        return
    x_mouse, y_mouse = event.xdata, event.ydata
    if x_mouse is None or y_mouse is None:
        return

    # Current view (left/right, top/bottom; origin="upper")
    xlim = main_ax.get_xlim()
    ylim = main_ax.get_ylim()
    left = min(xlim)
    right = max(xlim)
    width = right - left
    top = min(ylim)
    bottom = max(ylim)
    height = bottom - top
    if width <= 0 or height <= 0:
        return

    # Zoom factor
    scale = (1 / ZOOM_FACTOR if event.button == "up" else (ZOOM_FACTOR if event.button == "down" else 1.0))

    # Image size
    if isinstance(state.main_img_obj, list) and state.main_img_obj:
        """UI callback `toggle_play`.

        Args:
            event: See usage in code.

        Returns:
            None: Operates via side effects or returns implicit values if noted in code.
        """
        # UI event handler: updates state and refreshes relevant artists.
        img_shape = state.main_img_obj[0].get_array().shape
    else:
        img_shape = state.main_img_obj.get_array().shape if state.main_img_obj is not None else (0, 0)
    W_img, H_img = img_shape[1], img_shape[0]
    if W_img == 0 or H_img == 0:
        return

    # New window sizes within the image
    new_width = min(max(1.0, width * scale), W_img)
    new_height = min(max(1.0, height * scale), H_img)

    # Relative mouse position
    relx = (x_mouse - left) / width
    rely = (y_mouse - top) / height  # origin="upper": measure from top

    # New top-left
    new_left = x_mouse - relx * new_width
    new_top = y_mouse - rely * new_height

    # Clip within image
    new_left = np.clip(new_left, 0, max(0, W_img - new_width))
    new_top = np.clip(new_top, 0, max(0, H_img - new_height))
    new_right = new_left + new_width
    new_bottom = new_top + new_height

    # Fully zoomed out -> entire image
    if abs(new_width - W_img) < 1e-6 or abs(new_height - H_img) < 1e-6:
        """UI callback `on_slider_change`.

        Args:
            val: See usage in code.

        Returns:
            None: Operates via side effects or returns implicit values if noted in code.
        """
        # UI event handler: updates state and refreshes relevant artists.
        new_left, new_right = 0, W_img
        new_top, new_bottom = 0, H_img

    # Set limits (origin="upper": ylim = bottom, top)
    main_ax.set_xlim(new_left, new_right)
    main_ax.set_ylim(new_bottom, new_top)

    # Spara konsekvent (ymin = bottom, ymax = top) som i panning-koden
    state.main_zoom["xmin"], state.main_zoom["xmax"] = new_left, new_right
    state.main_zoom["ymin"], state.main_zoom["ymax"] = new_bottom, new_top

    main_ax.figure.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)

state.is_panning = False
state.pan_start = {"x": 0, "y": 0, "xlim": (0, 1), "ylim": (0, 1)}
"""UI callback `on_z_change`.

Args:
    val: See usage in code.

Returns:
    None: Operates via side effects or returns implicit values if noted in code.
"""
# UI event handler: updates state and refreshes relevant artists.
state.mouse_press_pos = None
drag_threshold = 2

def on_press(event):
    if event.inaxes == main_ax:
        if event.button != 1:
            return
        xlim = main_ax.get_xlim()
        ylim = main_ax.get_ylim()
        if isinstance(state.main_img_obj, list) and state.main_img_obj:
            img_shape = state.main_img_obj[0].get_array().shape
        else:
            img_shape = state.main_img_obj.get_array().shape if state.main_img_obj is not None else (0, 0)
        W_img, H_img = img_shape[1], img_shape[0]
        # Enable panning if image is zoomed in (not fully visible)
        if not (abs(xlim[1] - xlim[0] - W_img) < 1 and abs(ylim[1] - ylim[0] - H_img) < 1):
            state.is_panning = True
            state.pan_start["x"] = event.xdata
            state.pan_start["y"] = event.ydata
            state.pan_start["xlim"] = xlim
            state.pan_start["ylim"] = ylim
        state.mouse_press_pos = (event.x, event.y)
    elif event.inaxes == zoom_ax:
        select_id_at(event)

fig.canvas.mpl_connect("button_press_event", on_press)

def on_motion(event):
    """UI callback `on_key`.

    Args:
        event: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    if not state.is_panning or event.inaxes != main_ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    # Drag data coordinates
    dx = (event.xdata - state.pan_start["x"]) * DRAG_SENSITIVITY
    dy = (event.ydata - state.pan_start["y"]) * DRAG_SENSITIVITY

    xlim0 = state.pan_start["xlim"]
    ylim0 = state.pan_start["ylim"]

    # Current image size
    if isinstance(state.main_img_obj, list) and state.main_img_obj:
        img_shape = state.main_img_obj[0].get_array().shape
    else:
        img_shape = state.main_img_obj.get_array().shape if state.main_img_obj is not None else (0, 0)
    W_img, H_img = img_shape[1], img_shape[0]

    # View as left/right and top/bottom (top < bottom)
    left0 = min(xlim0)
    width0 = abs(xlim0[1] - xlim0[0])

    top0 = min(ylim0)  # with origin="upper", top is the smaller value
    bottom0 = max(ylim0)
    height0 = bottom0 - top0

    # Update position (clip within image)
    new_left = min(max(0, left0 - dx), max(0, W_img - width0))
    new_top = min(max(0, top0 - dy), max(0, H_img - height0))
    new_right = new_left + width0
    new_bottom = new_top + height0

    # Set limits (origin="upper")
    main_ax.set_xlim(new_left, new_right)
    main_ax.set_ylim(new_bottom, new_top)

    state.main_zoom["xmin"], state.main_zoom["xmax"] = new_left, new_right
    state.main_zoom["ymin"], state.main_zoom["ymax"] = new_bottom, new_top
    main_ax.figure.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_motion)

def on_release(event):
    """UI callback `on_scroll`.

    Args:
        event: See usage in code.

    Returns:
        None: Operates via side effects or returns implicit values if noted in code.
    """
    # UI event handler: updates state and refreshes relevant artists.
    if event.inaxes != main_ax:
        return
    if state.mouse_press_pos is not None:
        x0, y0 = state.mouse_press_pos
        x1, y1 = event.x, event.y
        if np.hypot(x1 - x0, y1 - y0) < drag_threshold:
            select_id_at(event)
    state.is_panning = False
    state.mouse_press_pos = None

fig.canvas.mpl_connect("button_release_event", on_release)

def select_id_at(event):
    if event.inaxes not in [main_ax, zoom_ax]:
        return
    if not event.inaxes.get_images():
        return
    x_click, y_click = event.xdata, event.ydata
    if not state.has_tracking:
        # Detection mode: center view on clicked point
        state.view_center[0] = x_click
        state.view_center[1] = y_click
        state.last_click_position[0] = x_click
        state.last_click_position[1] = y_click
        update_plot()
    elif state.click_mode:
        # Tracking mode click mode: center view on clicked point
        state.view_center[0] = x_click
        state.view_center[1] = y_click
        state.last_click_position[0] = x_click
        state.last_click_position[1] = y_click
        update_plot()
    else:
        # Tracking mode: select nearest ID to clicked point
        frame = state.current_frame
        ids_this_time = state.Comp[state.Comp["t"] == frame]
        if ids_this_time.empty:
            return
        # Use the unified function to get column names
        id_col = get_column_name("ID")
        x_col = get_column_name("x")
        y_col = get_column_name("y")
        z_col = get_column_name("z")
        if not all([id_col, x_col, y_col]):
            return
        xs = ids_this_time[x_col].values.astype(float)
        ys = ids_this_time[y_col].values.astype(float)
        ids = ids_this_time[id_col].values.astype(float)
        if not state.show_maxproj and z_col and z_col in ids_this_time.columns:
            zs = ids_this_time[z_col].values.astype(float)
            current_z = state.current_z_plane
            dists = np.sqrt((xs - x_click)**2 + (ys - y_click)**2 + ((zs - current_z))**2)
        else:
            dists = np.sqrt((xs - x_click)**2 + (ys - y_click)**2)
        nearest_idx = np.argmin(dists)
        nearest_id = ids[nearest_idx]
        state.current_id = int(nearest_id)
        update_view_center()
        text_box.set_val(str(state.current_id))
        update_plot()

# === Final Initialization and Event Loop ===

# Final initialization steps
"""create_channel_patch_axes()
if state.has_tracking and state.current_id is not None:
    update_view_center()

update_plot()
plt.show()"""