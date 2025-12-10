import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import Pipeline
import joblib
import os

elev_file = 'data/track_with_elevation.csv'
device_track = 'data/device_track_1.csv'
emergency_file = 'data/emergency_events.csv'

df_elev = pd.read_csv(elev_file)
df_track = pd.read_csv(device_track)
df_emergency = pd.read_csv(emergency_file)

def generate_segments(df_elev = df_elev):
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

        a = (math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
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
    return df_seg

def apply_segment(df_seg, df):
    def closest_segment(lat, lon):
        d = (df_seg["lat"] - lat)**2 + (df_seg["lon"] - lon)**2
        return d.idxmin()

    df["segment_id"] = df.apply(
        lambda row: closest_segment(row["latitude"], row["longitude"]), axis=1
    )

def generate_feature(df_seg, df_track = df_track, df_emergency = df_emergency):
    # density and offtrack rate
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

    # fungsi untuk menghitung stuck rate dan sos rate
    def rate(count, density):
        if (count == 0):
            return 0
        else:
            return count/density

    # stuck rate
    stuck_emg = df_emergency[df_emergency['emergency_type'] == 'Stuck']
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

    # sos rate
    sos_emg = df_emergency[df_emergency["emergency_type"] == 'SOS']
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

    return df_seg

def kmeans_clustering(segments):
    features = segments[['slope', 'curvature', 'density', 'offtrack_rate', 'stuck_rate', 'sos_rate']]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=3, max_iter=100, random_state=42))
    ])
    
    pipeline.fit(features)

    y_kmeans = pipeline.predict(features)

    segments["cluster"] = y_kmeans

    labels = pipeline.named_steps['kmeans'].labels_
    
    sc = metrics.silhouette_score(features, labels)
    db = metrics.davies_bouldin_score(features, labels)
    ch = metrics.calinski_harabasz_score(features, labels)

    ev = {"silhouette_score" : sc,
          "davies_bouldin_index": db,
          "calinski_harabasz_score": ch}
    
    joblib.dump(pipeline, "output/kmeans.pkl")

    return segments, ev

def main():
    print("Generating segments...")
    df_seg = generate_segments(df_elev)
    apply_segment(df_seg, df_track)
    apply_segment(df_seg, df_emergency)

    print("\nGenerating features...")
    df_seg = generate_feature(df_seg, df_track, df_emergency)
    df_seg.to_csv("output/df_seg.csv", index=False)

    print("\nTraining...")
    df_seg, evaluation_metrics = kmeans_clustering(df_seg)
    print("\nTraining complete")
    print(f"\nResults: \nSilhouette score : {evaluation_metrics['silhouette_score']}",
          f"\nDavies-Bouldin Index : {evaluation_metrics['davies_bouldin_index']}",
          f"\nCalinski-Harabasz Score : {evaluation_metrics['calinski_harabasz_score']}"
    )

if __name__ == "__main__":
    main()
    

    



