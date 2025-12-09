"""
Track Segmentation and Clustering Analysis
Analyzes track segments based on geographic and behavioral features
"""

import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


class TrackAnalyzer:
    """Main class for track segmentation and clustering analysis"""
    
    def __init__(self):
        self.df_seg = None
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def load_data(self, elevation_path, track_path, emergency_path):
        """Load all required datasets"""
        print("Loading data...")
        self.df_elev = pd.read_csv(elevation_path)
        self.df_track = pd.read_csv(track_path)
        self.df_emergency = pd.read_csv(emergency_path)
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
    
    def train_clustering_model(self, n_clusters=3, max_iter=100, random_state=42):
        """
        Train K-Means clustering model on segment features
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            random_state: Random seed
        """
        print("\n=== Training Clustering Model ===")
        
        # Select features
        feature_cols = ['slope', 'curvature', 'offtrack_rate', 
                       'density', 'stuck_rate', 'sos_rate']
        features = self.df_seg[feature_cols]
        
        # Standardize features
        X = self.scaler.fit_transform(features)
        
        # Train K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, 
                            random_state=random_state)
        self.kmeans.fit(X)
        
        # Assign cluster labels
        self.df_seg["cluster"] = self.kmeans.labels_
        
        print(f"Model trained with {n_clusters} clusters.")
    
    def evaluate_clustering(self):
        """Evaluate clustering quality using multiple metrics"""
        print("\n=== Clustering Evaluation ===")
        
        feature_cols = ['slope', 'curvature', 'offtrack_rate', 
                       'density', 'stuck_rate', 'sos_rate']
        X = self.scaler.transform(self.df_seg[feature_cols])
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
    
    def save_results(self, output_path="output/df_seg.csv"):
        """Save segmented data with cluster labels"""
        self.df_seg.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    def run_pipeline(self):
        """Execute the complete analysis pipeline"""
        print("="*50)
        print("TRACK SEGMENTATION AND CLUSTERING ANALYSIS")
        print("="*50)
        
        self.load_data(
            'data/track_with_elevation.csv',
            'data/device_track_1.csv',
            'data/emergency_events.csv'
        )
        
        self.generate_segments()
        self.map_data_to_segments()
        self.engineer_features()
        self.train_clustering_model()
        self.evaluate_clustering()
        self.save_results()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)


def main():
    """Main execution function"""
    analyzer = TrackAnalyzer()
    analyzer.run_pipeline()

def main_with_tuning():
    """Main execution with hyperparameter tuning"""
    analyzer = TrackAnalyzer()
    
    print("="*50)
    print("TRACK SEGMENTATION AND CLUSTERING WITH TUNING")
    print("="*50)
    
    # Load and prepare data
    analyzer.load_data(
        'data/track_with_elevation.csv',
        'data/device_track_1.csv',
        'data/emergency_events.csv'
    )
    analyzer.generate_segments()
    analyzer.map_data_to_segments()
    analyzer.engineer_features()
    
    # Option 1: Quick elbow method analysis
    print("\n" + "="*50)
    print("FINDING OPTIMAL K")
    print("="*50)
    elbow_results = analyzer.find_optimal_k_elbow(k_range=range(2, 8))
    
    # Option 2: Full hyperparameter tuning (may take time)
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Define custom parameter grid (smaller for faster execution)
    param_grid = {
        'n_clusters': [3, 4, 5],
        'max_iter': [100, 300],
        'n_init': [10, 20],
        'algorithm': ['lloyd'],
        'scaler': ['standard', 'robust']
    }
    
    df_results, best_params = analyzer.tune_hyperparameters(param_grid=param_grid)
    
    # Save tuning results
    df_results.to_csv("output/tuning_results.csv", index=False)
    print("\nTuning results saved to output/tuning_results.csv")
    
    # Apply best parameters
    analyzer.apply_best_params(best_params)
    analyzer.evaluate_clustering()
    analyzer.save_results()
    
    print("\n" + "="*50)
    print("TUNING AND ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()