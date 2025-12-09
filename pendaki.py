"""
sar_resource_optimization.py

SAR Resource Optimization Model pipeline:
1) Segment GPX trails into fixed-length segments
2) Compute segment-level features from trails + device tracks + events
3) Train a segment-level incident risk model (RandomForest)
4) Forecast incidents per segment per day (simple model)
5) Optimize SAR post locations (ILP maximizing covered risk)
6) Produce risk-based patrol frequencies (ILP / greedy)

Inputs:
 - trails.csv (columns: trail_id,name,geom) where geom is JSON list [{"lat":..,"lon":..},...]
 - device_track.csv (columns include device_id,timestamp,longitude,latitude,emergency_status,condition,off_track)
 - emergency_events.csv (optional) (columns: emergency_id,device_id,timestamp,longitude,latitude,emergency_type)

Outputs:
 - segment_features.csv
 - segment_risk_predictions.csv
 - sar_post_plan.json
 - patrol_plan.csv
"""

import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
import geopandas as gpd
from sklearn.neighbors import BallTree

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ILP solver
import pulp

# plotting (optional)
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------
# Utility geometry functions
# --------------------------
EARTH_R = 6371000.0

def haversine_m(a, b):
    lat1, lon1 = a; lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a_ = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*EARTH_R*math.atan2(math.sqrt(a_), math.sqrt(1-a_))

def point_along_segment(a, b, frac):
    lat = a[0] + (b[0] - a[0]) * frac
    lon = a[1] + (b[1] - a[1]) * frac
    return (lat, lon)

# --------------------------
# 1) Load trails and segment them
# --------------------------
def load_trails(trails_csv_path: str):
    df = pd.read_csv(trails_csv_path)
    # assume geom is JSON list of {"lat":..,"lon":..}
    trails = {}
    for _, row in df.iterrows():
        geom = json.loads(row['geom'])
        pts = [(p['lat'], p['lon']) for p in geom]
        trails[row['name']] = pts
    return trails

def segment_trail(pts, seg_length_m=50):
    """Return list of segments [(start_lat, start_lon),(end_lat,end_lon)] covering trail from start to end."""
    segments = []
    if len(pts) < 2:
        return segments
    # walk along polyline
    i = 0
    cur_pt = pts[0]
    accum = 0.0
    along_idx = 0
    # produce points every seg_length_m along track
    distances = []
    for j in range(len(pts)-1):
        d = haversine_m(pts[j], pts[j+1])
        distances.append(d)
    # create sample points along track at step seg_length_m
    total_len = sum(distances)
    if total_len == 0:
        return segments
    num = max(1, int(total_len // seg_length_m))
    # compute cumulative along track positions
    sample_positions = [i*seg_length_m for i in range(num+1)]
    # produce sample coordinates
    coords = []
    cur_cum = 0.0
    seg_idx = 0
    rem_in_seg = distances[0] if distances else 0
    a = pts[0]; b = pts[1]
    for pos in sample_positions:
        # move along to pos
        while pos > cur_cum + rem_in_seg and seg_idx < len(distances)-1:
            cur_cum += rem_in_seg
            seg_idx += 1
            a = pts[seg_idx]; b = pts[seg_idx+1]
            rem_in_seg = distances[seg_idx]
        frac = (pos - cur_cum) / (rem_in_seg if rem_in_seg>0 else 1)
        frac = max(0.0, min(1.0, frac))
        coords.append(point_along_segment(a,b,frac))
    # make consecutive segments
    for i in range(len(coords)-1):
        segments.append((coords[i], coords[i+1]))
    return segments

# --------------------------
# 2) Feature calculation
# --------------------------
def calc_segment_features(segments, device_track_df, event_df=None):
    """
    Optimized version using Sklearn BallTree for fast spatial queries.
    """
    # 1. Prepare Device Data
    device_points = device_track_df.copy()
    
    # Drop rows with NaN lat/lon
    device_points = device_points.dropna(subset=['latitude', 'longitude'])
    
    # Convert lat/lon to Radians for BallTree (required for 'haversine' metric)
    d_lat_rad = np.deg2rad(device_points['latitude'].values)
    d_lon_rad = np.deg2rad(device_points['longitude'].values)
    device_coords_rad = np.column_stack([d_lat_rad, d_lon_rad])
    
    # Pre-convert status columns to numpy for fast indexing
    # Handle boolean conversions robustly
    d_off_track = device_points['off_track'].astype(str).str.lower().isin(['true', '1', 't']).values.astype(int)
    d_emergency = device_points['emergency_status'].astype(str).str.lower().isin(['true', '1', 't']).values.astype(int)

    # Build Tree for Devices
    print("  -> Building spatial index for device tracks...")
    # metric='haversine' expects inputs in radians
    tree_device = BallTree(device_coords_rad, metric='haversine')
    
    # 2. Prepare Event Data (if exists)
    tree_event = None
    if event_df is not None and not event_df.empty:
        ev_lat = np.deg2rad(event_df['latitude'].values)
        ev_lon = np.deg2rad(event_df['longitude'].values)
        ev_coords = np.column_stack([ev_lat, ev_lon])
        tree_event = BallTree(ev_coords, metric='haversine')

    seg_records = []
    seg_id = 0
    EARTH_RADIUS_METERS = 6371000.0

    print(f"  -> Processing segments for {len(segments)} trails...")
    
    for trail_name, segs in tqdm(segments.items()): # Added tqdm for progress bar
        for s_idx, ((lat1, lon1), (lat2, lon2)) in enumerate(segs):
            seg_id += 1
            
            # Basic geometry
            seg_len = haversine_m((lat1, lon1), (lat2, lon2))
            mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
            
            # Convert query point to radians
            q_lat_rad, q_lon_rad = np.deg2rad(mid_lat), np.deg2rad(mid_lon)
            
            # Buffer radius in radians: distance / earth_radius
            buffer_m = max(30, seg_len / 2 + 20)
            radius_rad = buffer_m / EARTH_RADIUS_METERS
            
            # --- FAST QUERY: Devices ---
            # query_radius returns an array of indices found within radius
            ind_devices = tree_device.query_radius([[q_lat_rad, q_lon_rad]], r=radius_rad)[0]
            
            density = len(ind_devices)
            off_rate = 0.0
            stuck_rate = 0.0
            
            if density > 0:
                # Use numpy indexing which is super fast
                count_off = np.sum(d_off_track[ind_devices])
                count_stuck = np.sum(d_emergency[ind_devices]) # Using emergency as stuck proxy as per original code
                
                off_rate = count_off / density
                stuck_rate = count_stuck / density

            # --- FAST QUERY: Events ---
            close_events = 0
            if tree_event is not None:
                ind_events = tree_event.query_radius([[q_lat_rad, q_lon_rad]], r=radius_rad)[0]
                close_events = len(ind_events)

            # Curvature placeholder (unchanged)
            curvature = 0.0 

            seg_records.append({
                'seg_id': seg_id,
                'trail': trail_name,
                'seg_index': s_idx,
                'length_m': seg_len,
                'mid_lat': mid_lat,
                'mid_lon': mid_lon,
                'density_pts': density,
                'off_rate': off_rate,
                'stuck_rate': stuck_rate,
                'historical_events': close_events,
                'curvature': curvature
            })
            
    seg_df = pd.DataFrame(seg_records)
    return seg_df

# --------------------------
# 3) Train risk model (classification for incident presence)
# --------------------------
def train_risk_model(seg_features_df, label_col='historical_events', threshold=1):
    """
    label_col: historical_events (counts). We'll set binary label: 1 if >= threshold events in history
    Returns model and scaler
    """
    df = seg_features_df.copy()
    df['label'] = (df[label_col] >= threshold).astype(int)
    features = ['length_m','density_pts','off_rate','stuck_rate','historical_events','curvature']
    X = df[features].fillna(0.0).values
    y = df['label'].values
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y if len(np.unique(y))>1 else None)
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train); Xte = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_train)
    # eval
    if len(np.unique(y_test))>1:
        probs = clf.predict_proba(Xte)[:,1]
        auc = roc_auc_score(y_test, probs)
    else:
        auc = None
    print("Risk model trained. AUC (if available):", auc)
    # attach probabilities to original segments
    seg_features_df['risk_prob'] = clf.predict_proba(scaler.transform(df[features].fillna(0.0).values))[:,1]
    return clf, scaler, seg_features_df

# --------------------------
# 4) Forecast incidents & staffing need
# --------------------------
def forecast_incidents(seg_df, days_ahead=7):
    """
    Simple forecasting per segment using historic event counts.
    We'll fit a small regressor (RandomForestRegressor) on features -> historical_events and predict expected incidents.
    """
    df = seg_df.copy()
    features = ['length_m','density_pts','off_rate','stuck_rate','curvature']
    X = df[features].fillna(0.0).values
    y = df['historical_events'].values
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)
    expected = model.predict(X)
    df['expected_incidents_per_day'] = expected  # approximate
    return model, df

# Convert expected incidents -> required SAR staff count rough heuristic
def staff_requirements(expected_incidents, base_response_capacity=1.0):
    """
    Map expected incidents -> needed officers.
    base_response_capacity: incidents per officer per day (tunable)
    """
    req = np.ceil(expected_incidents / base_response_capacity).astype(int)
    return req

# --------------------------
# 5) Optimization: place k posts to maximize covered risk
# --------------------------
def optimize_posts(seg_df, k_posts=5, coverage_radius_m=1000):
    """
    ILP: variables x_j = 1 if place post at segment j midpoint
    maximize sum_j risk_j * covered_j
    covered_j = min(1, sum_{p placed} I(dist(p, j) <= coverage))
    We'll linearize: for each segment j, create cover variables y_j <= sum_{p} a_{p,j} x_p, y_j <=1, maximize sum risk_j * y_j
    """
    segs = seg_df.reset_index(drop=True)
    n = len(segs)
    coords = list(zip(segs['mid_lat'], segs['mid_lon']))
    risk = segs['risk_prob'].values
    # compute coverage matrix a[p,j]
    a = np.zeros((n,n), dtype=int)
    for p in range(n):
        for j in range(n):
            if haversine_m(coords[p], coords[j]) <= coverage_radius_m:
                a[p,j] = 1
    # ILP
    prob = pulp.LpProblem("SAR_Post_Placement", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{p}", cat="Binary") for p in range(n)]
    y = [pulp.LpVariable(f"y_{j}", lowBound=0, upBound=1, cat="Continuous") for j in range(n)]
    # objective
    prob += pulp.lpSum([risk[j] * y[j] for j in range(n)])
    # constraints
    prob += pulp.lpSum(x) <= k_posts
    for j in range(n):
        prob += y[j] <= pulp.lpSum([a[p,j] * x[p] for p in range(n)])
        prob += y[j] <= 1
    # solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = [i for i in range(n) if pulp.value(x[i]) >= 0.5]
    plan = segs.loc[chosen, ['seg_id','trail','seg_index','mid_lat','mid_lon','risk_prob']]
    return plan, pulp.value(prob.objective)

# --------------------------
# 6) Risk-based patrol scheduling
# --------------------------
def patrol_allocation(seg_df, total_officers=10, patrol_budget_hours_per_day=8, travel_speed_m_per_s=1.4):
    """
    Simple allocation:
    - each segment j gets frequency f_j (visits per day) proportional to risk * expected_incidents
    - convert frequencies to officer-hours estimating travel & on-site time
    We'll produce a greedy allocation of officer time to segments
    """
    df = seg_df.copy()
    # score = risk_prob * expected_incidents_per_day
    if 'expected_incidents_per_day' not in df.columns:
        df['expected_incidents_per_day'] = 0.1 + df['historical_events']*0.1
    df['score'] = df['risk_prob'] * (1 + df['expected_incidents_per_day'])
    # normalize into total patrol hours
    total_hours_available = total_officers * patrol_budget_hours_per_day
    # assume baseline visit time per segment = 0.25 hr + travel estimate proportional to local density (small heuristic)
    df['base_visit_hours'] = 0.25 + (df['density_pts'] / (df['density_pts'].max() + 1)) * 0.5
    # greedy: allocate hours to segments with highest score/base_visit_hours until hours exhausted
    df['priority'] = df['score'] / df['base_visit_hours']
    df = df.sort_values('priority', ascending=False)
    allocation = []
    hours_left = total_hours_available
    for _, row in df.iterrows():
        if hours_left <= 0:
            break
        # decide number of visits (integer)
        max_visits_possible = int(hours_left // row['base_visit_hours'])
        if max_visits_possible <= 0:
            continue
        # allocate min(2, max_visits_possible) visits (tunable)
        visits = min(2, max_visits_possible)
        allocation.append({
            'seg_id': row['seg_id'],
            'trail': row['trail'],
            'seg_index': row['seg_index'],
            'visits_per_day': visits,
            'hours_allocated': visits * row['base_visit_hours']
        })
        hours_left -= visits * row['base_visit_hours']
    patrol_df = pd.DataFrame(allocation)
    return patrol_df

# --------------------------
# CLI-like wrapper
# --------------------------
def run_pipeline(trails_csv="trails.csv", device_track_csv="device_track.csv", emergency_csv=None,
                 seg_length_m=50, k_posts=7, coverage_radius_m=1500):
    print("Loading trails...")
    trails = load_trails(trails_csv)  # expects name->pts
    print("Segmenting trails...")
    segments = {}
    for name, pts in trails.items():
        segments[name] = segment_trail(pts, seg_length_m=seg_length_m)
    print("Loading device tracks...")
    dt = pd.read_csv(device_track_csv)
    # ensure lat/long cols named consistently
    if 'latitude' not in dt.columns and 'lat' in dt.columns:
        dt = dt.rename(columns={'lat':'latitude','lon':'longitude'})
    # load emergency events if available
    ev = None
    if emergency_csv:
        ev = pd.read_csv(emergency_csv)
    print("Calculating segment-level features...")
    seg_df = calc_segment_features(segments, dt, event_df=ev)
    seg_df.to_csv("output\segment_features.csv", index=False)
    print("Training risk model...")
    clf, scaler, seg_pred_df = train_risk_model(seg_df, label_col='historical_events', threshold=1)
    seg_pred_df.to_csv("output\segment_risk_predictions.csv", index=False)
    print("Forecasting expected incidents...")
    fr_model, seg_forecast_df = forecast_incidents(seg_pred_df)
    seg_forecast_df.to_csv("output\segment_forecast.csv", index=False)
    print("Computing staff requirements (heuristic)...")
    seg_forecast_df['required_staff'] = staff_requirements(seg_forecast_df['expected_incidents_per_day'].values, base_response_capacity=0.5)
    print("Optimizing SAR post locations...")
    plan, obj = optimize_posts(seg_forecast_df, k_posts=k_posts, coverage_radius_m=coverage_radius_m)
    plan.to_csv("output\sar_post_plan.csv", index=False)
    print("Optimizing patrol allocation...")
    patrol_df = patrol_allocation(seg_forecast_df, total_officers=10, patrol_budget_hours_per_day=8)
    patrol_df.to_csv("output\patrol_plan.csv", index=False)
    print("All outputs saved: output\segment_features.csv, output\segment_risk_predictions.csv, output\segment_forecast.csv, output\sar_post_plan.csv, output\patrol_plan.csv")
    return {
        'segments': seg_df,
        'risk_predictions': seg_pred_df,
        'forecast': seg_forecast_df,
        'post_plan': plan,
        'patrol': patrol_df
    }

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # change file names as needed
    out = run_pipeline(trails_csv="trails.csv", device_track_csv="device_track_full.csv", emergency_csv="emergency_events.csv",
                       seg_length_m=50, k_posts=7, coverage_radius_m=1200)
    # quick plot top risk segments (optional)
    try:
        top = out['risk_predictions'].sort_values('risk_prob', ascending=False).head(20)
        plt.figure(figsize=(6,6))
        plt.scatter(out['risk_predictions']['mid_lon'], out['risk_predictions']['mid_lat'], c=out['risk_predictions']['risk_prob'], cmap='Reds', s=10)
        plt.colorbar(label='risk_prob')
        plt.scatter(top['mid_lon'], top['mid_lat'], c='blue', s=30, marker='x')
        plt.title("Segment risk map (approx.)")
        plt.savefig("segment_risk_map.png", dpi=150)
        print("Saved segment_risk_map.png")
    except Exception as e:
        print("Plot failed:", e)
