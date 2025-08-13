# Interactive Imaging Viewer — User Guide

## What this tool does
An interactive viewer for multi‑channel time series microscopy data (with optional z‑stacks). It provides:
- **Synchronized channels** with unified processing (percentile normalization, background subtraction, pure additive RGB mapping, optional per‑channel x‑offset).
- **Tracking mode** to browse track IDs and plot normalized fluorescence intensity (FI) traces, with optional **comparison** and **coverage** overlays.
- **Detection mode** to visualize detection points from CSVs; **click to center** the view.
- **Fast inspection** with per‑channel patch views and dynamic channel toggles.
- **Keyboard + mouse navigation** for zoom, pan, frame and z navigation.

---

## 1) Requirements

- **Python** 3.9+  
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tifffile`
  - `imagecodecs`

Install (recommended in a virtual environment):

`pip install numpy pandas matplotlib tifffile imagecodecs`

---

## 2) Prepare your working folder

Create a “work” folder (name is up to you; below we call it `work/`) with **one subfolder per imaging channel**. Example:

```
work/
├─ Channel_1/
│  ├─ 001/
│  │  └─ img_001.tif         # single 2D image or a multi-page TIFF (z-stack/time slice)
│  ├─ 002/
│  │  └─ img_002.tif
│  └─ ...                     # 003, 004, ...
├─ Channel_2/
│  ├─ 001/
│  │  └─ img_001.tif
│  └─ 002/
│     └─ img_002.tif
├─ CME_tracks.csv             # optional: tracking table
├─ Dino_tracks.csv            # optional: another tracking table
├─ detections_CME.csv         # optional: detection points for “CME”
└─ detections_Dino.csv        # optional: detection points for “Dino”
```

**Notes**
- If a channel folder (e.g., `Channel_1/`) contains **flat `.tif` files directly** (no `001/`, `002/`… folders), the viewer will **auto‑create numbered subfolders** (`001`, `002`, …), move images, and, if applicable, create a **`_maxproj.tif`** per timepoint (max‑projection for multi‑page TIFFs).
- Each numbered folder typically contains **one** `.tif` (either a 2D image or a stack for that timepoint).
- The viewer supports **2D** and **multi‑page TIFFs** (z‑stacks). For non‑TIFF formats (png/jpg), it will convert to TIFF for consistency.

---

## 3) Quick edits in `AVCV.py` (channel setup)

Open `AVCV.py` and set your **channel folders and colors** in `CHANNEL_CONFIG`. Example:

```python
CHANNEL_CONFIG = [
    {"name": "Ch1", "path": r"work/Channel_1", "color": "red"},
    {"name": "Ch2", "path": r"work/Channel_2", "color": "green"},
]
# Optional x-offset (pixels) to align channels horizontally
CHANNEL_OFFSET = 0
```

- Base (tracking) files shown in Tracking mode:
  ```python
  BASE_FILES = [
      {"file": "CME_tracks.csv", "name": "CME"},
      {"file": "Dino_tracks.csv", "name": "Dino"},
  ]
  ```
- Secondary CSVs used for comparison:
  ```python
  SEC_FILES = [
      {"file": "Dino_tracks.csv", "name": "Dino"},
      {"file": "CME_tracks.csv", "name": "CME"},
  ]
  ```
- Detection overlays:
  ```python
  DETECTION_CONFIG = [
      {"name": "CME",  "file": "detections_CME.csv",  "color": "blue"},
      {"name": "Dino", "file": "detections_Dino.csv", "color": "red"},
  ]
  ```

---

## 4) CSV formats (for Tracking & Detection)

### Tracking CSVs (used by `Comparison.py` and Tracking mode)
Assumed **column order** (headers required):
1. `ID`
2. `t0` (track starting time point)
3. `t` (frame/time index)
4. `x`
5. `y`
6. `z`
7. `FI` (Fluorescence Intensity)
8. `TL` (Track Length

You can compare a **base** table (e.g., `CME_tracks.csv`) to a **secondary** table (e.g., `Dino.csv`). Matching is done by **same `t`** and **nearest 3D neighbor** (x,y,z(.

### Detection CSVs
A simple list of detection points per frame. Typical columns:
- `ID, t, x, y, z, FI` (minimum needed; extra columns are ignored, tracking CSVs can be used as detection CSVs)
You’ll select which detection source(s) to show in the UI.

---

## 5) Run the viewer

From the folder containing `AVCV.py`:

```bash
python AVCV.py
```

A Matplotlib window opens with:
- The main fused view (RGB additive mapping from your channels).
- UI controls (sliders, checkboxes, radio buttons, buttons) for:
  - Channel on/off toggles
  - Frame (and z) navigation
  - Tracking/Detection mode selection
  - Base/Secondary dataset selection
  - Threshold settings for comparison/coverage
  - Text boxes for ID, frame, zoom, etc.

> If your channel folders don’t exist or are empty, you’ll see warnings in the console. Fix the paths in `CHANNEL_CONFIG` or populate the folders and run again.

---

## 6) Basic controls

**Mouse**
- **Scroll**: zoom in/out
- **Drag** (left mouse): pan
- **Click** on a detection point: center the view

**Keyboard**
- **W/A/S/D**: pan up/left/down/right
- **Frame navigation**: use the provided UI controls (and, if exposed, arrow‑like keys for frame/z)
- Other actions (save, reset view, etc.) are exposed via on‑screen buttons and standard Matplotlib keymaps (e.g., `Ctrl+S` to save a figure)

*(Exact key bindings may appear in tooltips or in the console as you interact.)*

---

## 7) Tracking mode workflow

1. Select **Tracking mode** in the UI.
2. Pick a **Base** table (e.g., `CME_tracks.csv`) and, optionally, a **Secondary** table (e.g., `Dino.csv`) for comparison.
3. Enter an **ID** (or use next/previous controls) to jump between tracked objects.
4. The viewer plots the **normalized FI trace** for the selected ID and overlays:
   - **Comparison** (nearest neighbor at same `t` from the secondary table within a distance threshold)
   - **Coverage** (timepoint‑level coverage derived from comparison)
5. If a comparison/coverage CSV doesn’t exist yet, the viewer will generate it lazily (using `Comparison.py`) and store it in your `work/` folder:
   - `Comparison_{Base}_vs_{Secondary}.csv`
   - `ID_Coverage_{Base}_vs_{Secondary}.csv`

---

## 8) Detection mode workflow

1. Select **Detection mode**.
2. Toggle one or more **detection sources** (e.g., “CME”, “Dino”) to overlay their points.
3. **Click** a point to center it; use frame/z controls to step through time or planes.

---

## 9) Tips & troubleshooting

- **Channel paths not found**  
  Check `CHANNEL_CONFIG` paths. Use absolute paths or make them relative to where you run the script.

- **No numbered subfolders**  
  The viewer will **auto‑create** `001/`, `002/`, … and move flat `.tif` files under the channel folder. It also creates missing `_maxproj.tif` files as needed.

- **Comparison seems empty**  
  Ensure both **Base** and **Secondary** tables cover the **same time range** and that the **threshold** is reasonable (in pixels; increase slightly if too strict).

- **Wrong colors**  
  Colors accept standard Matplotlib names (`"red"`, `"green"`, `"blue"`, `"magenta"`, `"cyan"`, `"yellow"`, `"orange"`, `"white"`).

- **Performance**  
  Large stacks may be heavy. Start with fewer channels, or pre‑generate max projections (the viewer will do this per timepoint automatically if missing).

---

## 10) Minimal checklist

- [ ] Python & packages installed (`numpy`, `pandas`, `matplotlib`, `tifffile`, `imagecodecs`)
- [ ] `work/` with per‑channel folders and `001/`, `002/`, … subfolders (or flat TIFFs the app can reorganize)
- [ ] `CHANNEL_CONFIG` updated in `AVCV.py` (paths + colors)
- [ ] Optional CSVs placed in `work/`:
  - Base: `*_tracks.csv`
  - Secondary: `*.csv`
  - Detections: `detections_*.csv`
- [ ] Run `python AVCV.py`
