import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from itertools import product
import joblib
from pathlib import Path

class TrackAnalyzer:
    """Main class for track segmentation and clustering analysis"""
    
    def __init__(self, output_dir="output"):
        self.df_seg = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_cols = ['slope', 'curvature', 'offtrack_rate', 'density', 'stuck_rate', 'sos_rate']
        self.best_params = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True) 

    def load_data(self, elevation_path, track_path, emergency_path):
        """Load all required datasets"""
        print("Loading data...")
        self.df_elev = pd.read_csv(elevation_path)
        self.df_track = track_path
        self.df_emergency = emergency_path
        print("Data loaded successfully.")
        
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = (math.sin(dphi/2)**2 + 
             math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
        distance = 2 * R * math.atan2(np.sqrt(a), np.sqrt(1 - a))
        
        return distance
    
    def calculate_slope(self, ele1, ele2, distance):
        """Calculate slope between two elevation points"""
        if distance == 0:
            return 0
        return (ele2 - ele1) / distance
    
    def calculate_curvature(self, i, df):
        """
        Calculate curvature (change in heading) at a point
        
        Args:
            i: Current index
            df: DataFrame with lat/lon columns
            
        Returns:
            Curvature value
        """
        if i >= len(df) - 2:
            return 0
            
        lat1, lon1 = df.loc[i, ["lat", "lon"]]
        lat2, lon2 = df.loc[i+1, ["lat", "lon"]]
        lat3, lon3 = df.loc[i+2, ["lat", "lon"]]
        
        # Calculate headings
        heading1 = math.atan2(
            math.radians(lon2 - lon1),
            math.radians(lat2 - lat1)
        )
        heading2 = math.atan2(
            math.radians(lon3 - lon2),
            math.radians(lat3 - lat2)
        )
        
        return abs(heading2 - heading1)
    
    def generate_segments(self):
        """Generate track segments with geometric features"""
        print("\nGenerating segments...")
        segments = []
        
        for i in range(len(self.df_elev) - 1):
            lat1, lon1, ele1 = self.df_elev.loc[i, ["lat", "lon", "elevation"]]
            lat2, lon2, ele2 = self.df_elev.loc[i+1, ["lat", "lon", "elevation"]]
            
            # Calculate segment features
            distance = self.calculate_haversine_distance(lat1, lon1, lat2, lon2)
            slope = self.calculate_slope(ele1, ele2, distance)
            curvature = self.calculate_curvature(i, self.df_elev)
            
            segments.append({
                "segment_id": i,
                "lat": lat1,
                "lon": lon1,
                "length_m": distance,
                "slope": slope,
                "curvature": curvature
            })
        
        self.df_seg = pd.DataFrame(segments)
        print(f"Generated {len(self.df_seg)} segments.")
    
    def find_closest_segment(self, lat, lon):
        """Find the closest segment to given coordinates"""
        distances = (self.df_seg["lat"] - lat)**2 + (self.df_seg["lon"] - lon)**2
        return distances.idxmin()
    
    def map_data_to_segments(self):
        """Map track and emergency data to nearest segments"""
        print("\nMapping data to segments...")
        
        self.df_track["segment_id"] = self.df_track.apply(
            lambda row: self.find_closest_segment(row["latitude"], row["longitude"]), 
            axis=1
        )
        
        self.df_emergency["segment_id"] = self.df_emergency.apply(
            lambda row: self.find_closest_segment(row["latitude"], row["longitude"]), 
            axis=1
        )
        
        print("Mapping complete.")
    
    def calculate_density_and_offtrack(self):
        """Calculate device density and off-track rate per segment"""
        print("\nCalculating density and off-track rate...")
        
        seg_stats = self.df_track.groupby("segment_id").agg({
            "device_id": "count",
            "off_track": "mean"
        }).rename(columns={
            "device_id": "density",
            "off_track": "offtrack_rate"
        }).reset_index()
        
        self.df_seg = self.df_seg.merge(seg_stats, on="segment_id", how="left")
        self.df_seg.fillna({"density": 0, "offtrack_rate": 0}, inplace=True)
    
    def calculate_emergency_rate(self, emergency_type, rate_column):
        """
        Calculate emergency rate per segment
        
        Args:
            emergency_type: Type of emergency ('Stuck' or 'SOS')
            rate_column: Name for the rate column
        """
        print(f"\nCalculating {emergency_type.lower()} rate...")
        
        # Filter emergency events
        emergency_filtered = self.df_emergency[
            self.df_emergency['emergency_type'] == emergency_type
        ]
        
        # Count emergencies per segment
        count_col = f"{emergency_type.lower()}_count"
        emergency_stats = emergency_filtered.groupby("segment_id").agg({
            "emergency_id": "count"
        }).rename(columns={"emergency_id": count_col}).reset_index()
        
        # Merge with segments
        temp_df = self.df_seg.merge(emergency_stats, on="segment_id", how="left")
        temp_df.fillna({count_col: 0}, inplace=True)
        
        # Calculate rate (emergencies per device)
        temp_df[rate_column] = temp_df.apply(
            lambda row: row[count_col] / row["density"] if row["density"] > 0 else 0,
            axis=1
        )
        
        self.df_seg[rate_column] = temp_df[rate_column]
    
    def engineer_features(self):
        """Generate all behavioral features for segments"""
        print("\n=== Feature Engineering ===")
        
        self.calculate_density_and_offtrack()
        self.calculate_emergency_rate('Stuck', 'stuck_rate')
        self.calculate_emergency_rate('SOS', 'sos_rate')
        
        print("\nFeature engineering complete.")
    
    def evaluate_clustering(self):
        """Evaluate clustering quality using multiple metrics"""
        print("\n=== Clustering Evaluation ===")
        
        features = self.df_seg[self.feature_cols]
        X = self.scaler.transform(features)
        labels = self.kmeans.labels_
        
        # Calculate metrics
        ch_score = metrics.calinski_harabasz_score(X, labels)
        db_score = metrics.davies_bouldin_score(X, labels)
        silhouette = metrics.silhouette_score(X, labels)
        
        print(f"Calinski-Harabasz Index: {ch_score:.2f}")
        print(f"Davies-Bouldin Index: {db_score:.2f}")
        print(f"Silhouette Coefficient: {silhouette:.2f}")
        
        return {
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score,
            'silhouette': silhouette
        }
    
    def tune_hyperparameters(self, param_grid=None, feature_sets=None):
        """
        Tune hyperparameters for K-Means clustering
        
        Args:
            param_grid: Dictionary of parameters to tune
            feature_sets: List of feature combinations to try
            
        Returns:
            DataFrame with results and best parameters
        """
        print("\n=== Hyperparameter Tuning ===")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_clusters': [3, 4, 5, 6],
                'max_iter': [100, 300, 500],
                'n_init': [10, 20, 30],
                'algorithm': ['lloyd', 'elkan'],
                'scaler': ['standard', 'minmax', 'robust']
            }
        
        # Default feature sets
        if feature_sets is None:
            all_features = ['slope', 'curvature', 'offtrack_rate', 
                           'density', 'stuck_rate', 'sos_rate']
            feature_sets = [
                all_features,  # All features
                ['slope', 'curvature', 'stuck_rate', 'sos_rate'],  # Geometric + Safety
                ['offtrack_rate', 'density', 'stuck_rate', 'sos_rate'],  # Behavioral only
                ['slope', 'curvature', 'density'],  # Geometric + Traffic
            ]
        
        results = []
        best_score = -np.inf
        best_params = None
        
        # Generate all combinations
        param_combinations = list(product(
            param_grid['n_clusters'],
            param_grid['max_iter'],
            param_grid['n_init'],
            param_grid['algorithm'],
            param_grid['scaler']
        ))
        
        total_combinations = len(param_combinations) * len(feature_sets)
        print(f"Testing {total_combinations} combinations...")
        
        for idx, (n_clust, max_it, n_in, algo, scaler_type) in enumerate(param_combinations):
            for feat_idx, features in enumerate(feature_sets):
                try:
                    # Select scaler
                    if scaler_type == 'standard':
                        scaler = StandardScaler()
                    elif scaler_type == 'minmax':
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    # Prepare features
                    X = scaler.fit_transform(self.df_seg[features])
                    
                    # Train model
                    kmeans = KMeans(
                        n_clusters=n_clust,
                        max_iter=max_it,
                        n_init=n_in,
                        algorithm=algo,
                        random_state=42
                    )
                    labels = kmeans.fit_predict(X)
                    
                    # Evaluate
                    ch_score = metrics.calinski_harabasz_score(X, labels)
                    db_score = metrics.davies_bouldin_score(X, labels)
                    sil_score = metrics.silhouette_score(X, labels)
                    inertia = kmeans.inertia_
                    
                    # Store results
                    result = {
                        'n_clusters': n_clust,
                        'max_iter': max_it,
                        'n_init': n_in,
                        'algorithm': algo,
                        'scaler': scaler_type,
                        'features': ', '.join(features),
                        'n_features': len(features),
                        'calinski_harabasz': ch_score,
                        'davies_bouldin': db_score,
                        'silhouette': sil_score,
                        'inertia': inertia
                    }
                    results.append(result)
                    
                    # Track best model (using Silhouette score)
                    if sil_score > best_score:
                        best_score = sil_score
                        best_params = {
                            'n_clusters': n_clust,
                            'max_iter': max_it,
                            'n_init': n_in,
                            'algorithm': algo,
                            'scaler': scaler_type,
                            'features': features
                        }
                    
                except Exception as e:
                    print(f"Error with params {n_clust}, {max_it}, {n_in}, {algo}, {scaler_type}: {e}")
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Sort by silhouette score (higher is better)
        df_results = df_results.sort_values('silhouette', ascending=False)
        
        print(f"\nTuning complete! Tested {len(results)} configurations.")
        print("\n=== Top 5 Configurations ===")
        print(df_results.head(5).to_string(index=False))
        
        print("\n=== Best Parameters ===")
        for key, value in best_params.items():
            if key == 'features':
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        print(f"Best Silhouette Score: {best_score:.4f}")
        
        return df_results, best_params
    
    def apply_best_params(self, best_params):
        """
        Apply the best parameters found during tuning
        
        Args:
            best_params: Dictionary of best parameters
        """
        print("\n=== Applying Best Parameters ===")

        self.best_params = best_params
        self.feature_cols = best_params['features']

        print(f"Using features: {', '.join(self.feature_cols)}")
        
        # Select scaler
        if best_params['scaler'] == 'standard':
            self.scaler = StandardScaler()
        elif best_params['scaler'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        # Prepare features
        features = self.df_seg[best_params['features']]
        X = self.scaler.fit_transform(features)
        
        # Train final model
        self.kmeans = KMeans(
            n_clusters=best_params['n_clusters'],
            max_iter=best_params['max_iter'],
            n_init=best_params['n_init'],
            algorithm=best_params['algorithm'],
            random_state=42
        )
        self.kmeans.fit(X)
        
        # Assign cluster labels
        self.df_seg["cluster"] = self.kmeans.labels_
        
        print("Best model applied successfully!")
    
    def find_optimal_k_elbow(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using Elbow method
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Dictionary with inertia values for each k
        """
        print("\n=== Elbow Method Analysis ===")
        
        feature_cols = ['slope', 'curvature', 'offtrack_rate', 
                       'density', 'stuck_rate', 'sos_rate']
        X = self.scaler.fit_transform(self.df_seg[feature_cols])
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                sil_score = metrics.silhouette_score(X, kmeans.labels_)
                silhouettes.append(sil_score)
            else:
                silhouettes.append(0)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")
        
        return {
            'k_values': list(k_range),
            'inertia': inertias,
            'silhouette': silhouettes
        }
    
    def save_results(self):
        """Save segmented data with cluster labels"""
        self.df_seg.to_csv(self.output_dir / "segmented.csv", index=False)
        print(f"\nResults saved to {self.output_dir}")

    def save_model(self):
        """Save the trained model and scaler to the output directory"""
        print(f"\nSaving models to {self.output_dir}...")
        
        # 1. Tentukan Nama File
        model_path = self.output_dir / "kmeans_model.pkl"
        scaler_path = self.output_dir / "scaler_model.pkl"
        
        # 2. Simpan (Dump) objek model ke file tersebut
        # Kita pakai joblib karena lebih efisien untuk Scikit-Learn daripada pickle biasa
        joblib.dump(self.kmeans, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")