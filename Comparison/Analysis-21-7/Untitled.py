import os
import tifffile
import stackview
import numpy as np
import napari
import matplotlib.pyplot as plt
import pandas as pd
from skimage import exposure
import math
from matplotlib.widgets import Slider
from ipywidgets import interact, IntSlider
from IPython.display import display, clear_output
import cv2
from PIL import Image
from ipycanvas import MultiCanvas

base = "CME"

if base == "CME":
    sec = "Dino"
if base == "Dino":
    sec = "CME"

def plot(ax, time, boxsize, Comparison, CME_detections, Dino_detections):
    ID1 = Comparison[Comparison["t"] == time].sort_values([f"ID ({base})", "t"], ascending = [True, True])

    Dino_time = Dino_detections[Dino_detections["timepoint"]==time]
    
    CME_time = CME_detections[CME_detections[0]==time]

    selected_folder = folders[time-1]
    
    volume_path = os.path.join(root_dir, selected_folder, "volume.tif")
    volume = tifffile.imread(volume_path)
    
    sec_ID = ID1[f"ID ({sec})"].values[0]
    color = ID1["Color"].values[0]

    selected_folder = folders[time - 1]
    volume_path = os.path.join(root_dir, selected_folder, "volume.tif")
    volume = tifffile.imread(volume_path)

    z_CME = ID1[f"z (CME)"].values[0]
    x_CME = ID1[f"x (CME)"].values[0]
    y_CME = ID1[f"y (CME)"].values[0]
    z_Dino = ID1[f"z (Dino)"].values[0]
    x_Dino = ID1[f"x (Dino)"].values[0]
    y_Dino = ID1[f"y (Dino)"].values[0]
    FI_base = round(ID1[f"FI ({base})"].values[0], 2)
    Distance = round(ID1["Distance"].values[0], 2)

    if base == "CME":
        x_base, y_base, z_base = x_CME, y_CME, z_CME
        x_sec, y_sec, z_sec = x_Dino, y_Dino, z_Dino
        base_color = "darkred"
    if base == "Dino":
        x_base, y_base, z_base = x_Dino, y_Dino, z_Dino
        x_sec, y_sec, z_sec = x_CME, y_CME, z_CME
        base_color = "darkblue"
        
    H, W = volume.shape[1:3]
    y1 = max(0, int(round(y_base - boxsize / 2)))
    y2 = min(H, y1 + boxsize)
    x1 = max(0, int(round(x_base - boxsize / 2)))
    x2 = min(W, x1 + boxsize)

    frame = volume[int(round(z_base)), y1:y2, x1:x2]
    patch_ymin = y1
    patch_xmin = x1
    patch_ymax = patch_ymin + frame.shape[0]
    patch_xmax = patch_xmin + frame.shape[1]

    z_tol=2

    mask_z = (CME_time[4] >= z_CME - z_tol) & (CME_time[4] <= z_CME + z_tol)
    close_points = CME_time[mask_z]
    patch_mask = (close_points[2] >= patch_xmin) & (close_points[2] < patch_xmax) & \
                 (close_points[3] >= patch_ymin) & (close_points[3] < patch_ymax)
    close_points = close_points[patch_mask]
    label_added = False
    for idx, row in close_points.iterrows():
        if row[2] != x_CME:
            label = f"{base} Det" if not label_added else ""
            ax.scatter(row[2], row[3], color="blue", label=label,s=100)
            label_added = True
    
    mask_z = (Dino_time["z"] >= z_CME - z_tol) & (Dino_time["z"] <= z_CME + z_tol)
    close_points = Dino_time[mask_z]
    patch_mask = (close_points["x"] >= patch_xmin) & (close_points["x"] < patch_xmax) & \
                 (close_points["y"] >= patch_ymin) & (close_points["y"] < patch_ymax)
    close_points = close_points[patch_mask]
    label_added = False
    for idx, row in close_points.iterrows():
        if row["x"] != x_Dino:
            ax.scatter(row["x"], row["y"], color="red", label=f"{sec} Det" if not label_added else "",s=100)
            label_added = True

    ax.imshow(
        frame, cmap="gray", vmin=0, vmax=0.3,
        extent=[
            patch_xmin, patch_xmax,
            patch_ymax, patch_ymin
        ]
    )
    
    ax.scatter(x_base, y_base, color=base_color, label=f"{base} Base",s=100)
    if (patch_xmin <= x_sec < patch_xmax) and (patch_ymin <= y_sec < patch_ymax):
        ax.scatter(x_sec, y_sec, color=color, label=f"{sec} Close",s=100)
            
    ax.legend(fontsize=15)
    ax.text(
        patch_xmin+1, patch_ymin+5.1,
        f"t = {time}\n{sec} ID {int(sec_ID)}\nDistance = {Distance}",
        fontsize=15,
        color="black",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(10)
    ax.set_xticks([])
    ax.set_yticks([])

Comp = pd.read_csv(f"{base} Base Output/Comparison.csv")
Dino_det = pd.read_csv("Input Data/detections_Dino.csv")
Dino_det["timepoint"] = Dino_det["timepoint"] + 1
CME_det = pd.read_csv("Input Data/detections_CME.csv", header = None)
Cov = pd.read_csv(f"{base} Base Output/ID_Coverage.csv")

root_dir = "/nfs/scratch2/shared_image_recog_ml/ap2_unnorm"
folders = [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
        and not f.startswith(".")
    ]
folders.sort()


IDs = [10790]

boxsize = 40

for ID in IDs:
    choice = Comp[Comp[f"ID ({base})"] == ID]
    time = choice["t"].values

    uniq = np.unique(choice[f"ID ({sec})"])
    colors = [plt.cm.gist_rainbow(i/len(uniq)) for i in range(len(uniq)+1)][1:]
    uniq_colors = pd.DataFrame({f"ID ({sec})": uniq, "Color": colors})
    ID_Comp = pd.merge(choice, uniq_colors, on = f"ID ({sec})")

    frames = []
    for i, t_now in enumerate(time):
        if i%10 == 0:
            print(f"Generated {round((i+1)/len(time),1)*100} %")
        fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
        plot(ax_temp, int(t_now), boxsize, ID_Comp, CME_det, Dino_det)
        fig_temp.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_temp.canvas.draw()
        img = np.asarray(fig_temp.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)
        plt.close(fig_temp)
    frames = np.array(frames)


backgrounds = []

for t in time:
    selected_folder = folders[int(t)-1]
    volume_path = os.path.join(root_dir, selected_folder, "volume.tif")
    volume = tifffile.imread(volume_path)
    max_projection = np.max(volume, axis=0)
    
    Dino_time = Dino_det[Dino_det.iloc[:, 0]==t]
    CME_time = CME_det[CME_det.iloc[:, 0]==t]

    Comp_non = Comp[Comp["Distance"]<3.5]
    Comp_non_time = Comp_non[Comp_non.iloc[:, 8]==t]

    Dino_ID_non = Comp_non_time["ID (Dino)"].unique()
    CME_ID_non = Comp_non_time["ID (CME)"].unique()

    Dino_clean = Dino_time[~Dino_time["label_id"].isin(Dino_ID_non)].reset_index(drop=True)
    CME_clean = CME_time[~CME_time[1].isin(CME_ID_non)].reset_index(drop=True)
    
    height, width = max_projection.shape
    
    fig, ax = plt.subplots(figsize=(15,15), dpi=100)
    ax.imshow(max_projection, cmap="gray", vmin=0, vmax=0.5,aspect = "auto")
    
    ax.scatter(CME_time.iloc[:,2], CME_time.iloc[:,3], s=10, color="blue", label="CME")
    ax.scatter(Dino_time["x"], Dino_time["y"], s=10, color="red", label="Dino")
    ax.scatter(Comp_non_time["x (CME)"], Comp_non_time["y (CME)"], s=10, color="green", label="Match")
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    backgrounds.append(img)



Image.fromarray(backgrounds[0]).save("debug.png")
def show_frame(i):
    # Hämta bakgrunden för detta frame-index
    bakgrund_kopia = backgrounds[i].copy()  # Kopiera om du vill kunna rita ovanpå
    
    liten_bild = frames[i]   # Detta antar jag fortfarande är "overlay" bilden du vill lägga ovanpå?
    h, w = liten_bild.shape[:2]
    y0 = bakgrund_kopia.shape[0] - h
    x0 = bakgrund_kopia.shape[1] - w
    bakgrund_kopia[y0:y0+h, x0:x0+w] = liten_bild

    plt.figure(figsize=(15,15))
    plt.imshow(bakgrund_kopia)
    plt.axis('off')
    plt.show()

frame_slider = IntSlider(min=0, max=frames.shape[0]-1, step=1, value=0, description='Frame')
interact(show_frame, i=frame_slider)

bg = backgrounds[0].copy()
overlay = frames[0].copy()

# Säkerställ rätt format:
if bg.dtype != np.uint8:
    bg = (bg * 255).astype(np.uint8)
if overlay.dtype != np.uint8:
    overlay = (overlay * 255).astype(np.uint8)
if overlay.ndim == 2:
    overlay = np.stack([overlay]*3, axis=-1)

H, W = bg.shape[:2]
h, w = overlay.shape[:2]

# Startposition overlay, i mitten
x, y = (W - w) // 2, (H - h) // 2
drag = False
ox, oy = 0, 0

def show():
    img = bg.copy()
    # Kontroll så att overlay alltid ligger inom bakgrunden!
    x_clip = max(0, min(W - w, x))
    y_clip = max(0, min(H - h, y))
    # Klistra in overlayn
    img[y_clip:y_clip+h, x_clip:x_clip+w] = overlay
    cv2.imshow('Drag overlay', img)

def mouse(event, mx, my, flags, param):
    global x, y, drag, ox, oy
    if event == cv2.EVENT_LBUTTONDOWN:
        # Kolla om klick i overlay
        if x <= mx <= x + w and y <= my <= y + h:
            drag = True
            ox, oy = mx - x, my - y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag:
            x = mx - ox
            y = my - oy
            show()
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False

cv2.namedWindow('Drag overlay')
cv2.setMouseCallback('Drag overlay', mouse)
show()

print("Dra overlayn med musen. Tryck ESC för att avsluta.")
while True:
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()