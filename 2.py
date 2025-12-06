import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
from sklearn.cluster import KMeans
from sklearn import metrics

df_elev = pd.read_csv('data/track_with_elevation.csv')

# Generating segments

print("Generating segments... ")
segments = []

for i in range(len(df_elev) - 1):
    lat1, lon1, ele1 = df_elev.loc[i, ["lat", "lon", "elevation"]]
    lat2, lon2, ele2 = df_elev.loc[i+1, ["lat", "lon", "elevation"]]

    # -----------------------------
    # a) segment length (meters)
    # -----------------------------
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    dist = 2 * R * math.atan2(np.sqrt(a), np.sqrt(1 - a))

    # -----------------------------
    # b) slope (elevation gain / distance)
    # -----------------------------
    if dist == 0:
        slope = 0
    else:
        slope = (ele2 - ele1) / dist  # meters per meter → unitless

    # -----------------------------
    # c) curvature (perubahan heading)
    # -----------------------------
    # heading i → i+1
    heading1 = math.atan2(
        math.radians(lon2 - lon1),
        math.radians(lat2 - lat1)
    )

    # heading i+1 → i+2 (kalau ada)
    if i < len(df_elev) - 2:
        lat3, lon3 = df_elev.loc[i+2, ["lat", "lon"]]
        heading2 = math.atan2(
            math.radians(lon3 - lon2),
            math.radians(lat3 - lat2)
        )
        curvature = abs(heading2 - heading1)
    else:
        curvature = 0  # segment terakhir

    # simpan
    segments.append({
        "segment_id": i,
        "lat": lat1,
        "lon": lon1,
        "length_m": dist,
        "slope": slope,
        "curvature": curvature
    })

df_seg = pd.DataFrame(segments)

df_track = pd.read_csv('data/device_track_1.csv')
df_emergency = pd.read_csv("data/emergency_events.csv")

def closest_segment(lat, lon):
    d = (df_seg["lat"] - lat)**2 + (df_seg["lon"] - lon)**2
    return d.idxmin()

# Apply closest segments to df_track and df_emergency

print("\nApplying closest segments to device track data and emegency events data...")

df_track["segment_id"] = df_track.apply(
    lambda row: closest_segment(row["latitude"], row["longitude"]), axis=1
)

df_emergency["segment_id"] = df_emergency.apply(
    lambda row: closest_segment(row["latitude"], row["longitude"]), axis=1
)

# hitung fitur tambahan
print("\nGenerating density and offtrack rate...")
seg_stats = df_track.groupby("segment_id").agg({
    "device_id": "count",
    "off_track": "mean"
}).rename(columns={
    "device_id": "density",
    "off_track": "offtrack_rate"
}).reset_index()

df_seg = df_seg.merge(seg_stats, on="segment_id", how="left")

df_seg.fillna({
    "density": 0,
    "offtrack_rate": 0
}, inplace=True)

###
stuck_emg = df_emergency[df_emergency['emergency_type'] == 'Stuck']
sos_emg = df_emergency[df_emergency["emergency_type"] == 'SOS']

def rate(count, density):
    if (count == 0):
        return 0
    else:
        return count/density
    
## Stuck
print("\nGenerating stuck rate...")
stuck_stats = stuck_emg.groupby("segment_id").agg({
    "emergency_id":"count"
}).rename(columns={"emergency_id":"stuck_count"}).reset_index()

temp_stuck = df_seg.merge(stuck_stats, on="segment_id", how="left")

temp_stuck.fillna({
    "stuck_count": 0
}, inplace=True)

temp_stuck['stuck_rate'] = temp_stuck.apply(
    lambda row: rate(row["stuck_count"], row["density"]), axis=1
)

df_seg['stuck_rate'] = temp_stuck['stuck_rate']

### SOS
print("\nGenerating sos rate...")
sos_stats = sos_emg.groupby("segment_id").agg({
    "emergency_id":"count"
}).rename(columns={"emergency_id":"sos_count"}).reset_index()

temp_sos = df_seg.merge(sos_stats, on="segment_id", how="left")

temp_sos.fillna({
    "sos_count": 0
}, inplace=True)

temp_sos['sos_rate'] = temp_sos.apply(
    lambda row: rate(row["sos_count"], row["density"]), axis=1
)

df_seg['sos_rate'] = temp_sos['sos_rate'] 

df_seg.to_csv("output/df_seg.csv", index=False)

print("\nFeature engineering complete.")
print("\nTraining model...")

### Training
features = df_seg[['slope', 'curvature', 'offtrack_rate', 'density', 'stuck_rate', 'sos_rate']]

# Standarisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
kmeans.fit(X)

df_seg["cluster"] = kmeans.labels_

labels = kmeans.labels_
sc = metrics.silhouette_score(X, labels)
db = metrics.davies_bouldin_score(X, labels)
ch = metrics.calinski_harabasz_score(X, kmeans.labels_)

print(f'Calinski-Harabasz Index: {ch}')
print(f'Davies-Bouldin Index: {db}')
print("Silhouette Coefficient:%0.2f" % sc)