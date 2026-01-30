import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask as riomask
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.colors as mcolors
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
FILES = {
    "2015": "Tirupati_Period1_2015_LANDSAT8.tif",
    "2024": "Tirupati_Period2_2024_LANDSAT8.tif"
}

# Define LULC Classes
CLASSES = {
    1: "Forest",
    2: "Water Bodies",
    3: "Agriculture",
    4: "Barren Land",
    5: "Built-up"
}

# Colors for visualization (Premium Palette)
CLASS_COLORS = {
    "Forest": "#228B22",       # Forest Green
    "Water Bodies": "#1E90FF", # Dodger Blue
    "Agriculture": "#DAA520",  # Goldenrod
    "Barren Land": "#DEB887",  # BurlyWood
    "Built-up": "#FF4500"      # Orange Red
}

def load_and_inspect(filepath):
    """Loads a GeoTIFF and prints metadata."""
    print(f"\n--- Inspecting: {filepath} ---")
    with rasterio.open(filepath) as src:
        meta = src.meta
        print(f"Bands: {src.count}")
        print(f"Shape: {src.shape}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        
        # Read all bands
        data = src.read()
        descriptions = src.descriptions
        return data, meta, descriptions

# ==========================================
# CORE PROCESSING PIPELINE
# ==========================================
def auto_label_clusters(kmeans_labels, features, n_clusters=5):
    """
    Maps KMeans clusters to LULC classes based on mean spectral index values.
    Indices: NDVI (idx 7), NDBI (idx 8), NDWI (idx 9), BSI (idx 10)
    """
    cluster_means = []
    for i in range(n_clusters):
        cluster_data = features[kmeans_labels == i]
        cluster_means.append(np.mean(cluster_data, axis=0))
    
    cluster_means = np.array(cluster_means)
    
    # Logic for mapping:
    # 1. Water: Highest NDWI (idx 9)
    water_cluster = np.argmax(cluster_means[:, 9])
    
    # 2. Forest: Highest NDVI (idx 7) among remaining
    remaining = [i for i in range(n_clusters) if i != water_cluster]
    forest_cluster = remaining[np.argmax(cluster_means[remaining, 7])]
    
    # 3. Built-up: Highest NDBI (idx 8) among remaining
    remaining = [i for i in remaining if i != forest_cluster]
    builtup_cluster = remaining[np.argmax(cluster_means[remaining, 8])]
    
    # 4. Barren: Highest BSI (idx 10) among remaining
    remaining = [i for i in remaining if i != builtup_cluster]
    barren_cluster = remaining[np.argmax(cluster_means[remaining, 10])]
    
    # 5. Agriculture: Last remaining
    agri_cluster = [i for i in remaining if i != barren_cluster][0]
    
    mapping = {
        water_cluster: 2,
        forest_cluster: 1,
        builtup_cluster: 5,
        barren_cluster: 4,
        agri_cluster: 3
    }
    
    labels = np.zeros_like(kmeans_labels)
    for k, v in mapping.items():
        labels[kmeans_labels == k] = v
        
    return labels

def train_refine_classifier(features, labels):
    """Refines classification using Random Forest."""
    print("Training Random Forest for refinement...")
    # Sampling for faster training if dataset is huge, but here we use all labels
    # Splitting for a quick validation
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Validation Accuracy:", rf.score(X_test, y_test))
    return rf

import gc

def predict_in_chunks(model, features, chunk_size=1000000):
    """Predicts in chunks to save memory."""
    predictions = []
    for i in range(0, len(features), chunk_size):
        predictions.append(model.predict(features[i:i+chunk_size]))
    return np.concatenate(predictions)

def calculate_confidence_and_transitions(rf_2015, rf_2024, feat_2015, feat_2024, mask, rows, cols, chunk_size=500000):
    """
    Generates confidence maps and probabilistic transition likelihoods.
    Transition Likelihood = P(Class_2015) * P(Class_2024)
    """
    print("Generating AI Confidence & Transition Likelihood Maps...")
    
    conf_2015 = np.zeros(len(feat_2015), dtype=np.float32)
    conf_2024 = np.zeros(len(feat_2024), dtype=np.float32)
    trans_likelihood = np.zeros(len(feat_2024), dtype=np.float32)
    
    for i in range(0, len(feat_2015), chunk_size):
        c15 = feat_2015[i : i + chunk_size]
        c24 = feat_2024[i : i + chunk_size]
        
        p15 = rf_2015.predict_proba(c15)
        p24 = rf_2024.predict_proba(c24)
        
        conf_2015[i : i + chunk_size] = np.max(p15, axis=1)
        conf_2024[i : i + chunk_size] = np.max(p24, axis=1)
        
        # Likelihood of the detected transition
        # We take the product of the probabilities of the predicted classes
        pred15 = np.argmax(p15, axis=1)
        pred24 = np.argmax(p24, axis=1)
        
        # Optimized way to get P(pred) for each pixel
        idx = np.arange(len(pred15))
        trans_likelihood[i : i + chunk_size] = p15[idx, pred15] * p24[idx, pred24]
        
        del p15, p24
        gc.collect()
    
    # Reshape to maps
    # Use the specific mask for the pixels they were extracted from
    m_conf15 = np.zeros(rows * cols, dtype=np.float32); m_conf15[mask] = conf_2015
    m_conf24 = np.zeros(rows * cols, dtype=np.float32); m_conf24[mask] = conf_2024
    m_trans = np.zeros(rows * cols, dtype=np.float32); m_trans[mask] = trans_likelihood
    
    return m_conf15.reshape(rows, cols), m_conf24.reshape(rows, cols), m_trans.reshape(rows, cols)

def perform_classification_pipeline(year, filepath, shp_path=None):
    print(f"\n>>> Processing {year}...")
    with rasterio.open(filepath) as src:
        if shp_path:
            aoi = gpd.read_file(shp_path).to_crs(src.crs)
            out_image, out_transform = riomask(src, aoi.geometry, crop=True)
            meta = src.meta.copy()
            meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
            data = out_image
        else:
            meta = src.meta
            data = src.read()
            
        rows, cols = data.shape[1], data.shape[2]
        select_bands = [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 23]
        selected_data = data[select_bands].astype('float32')
        
    features = selected_data.reshape(len(select_bands), -1).T
    mask = np.all(np.isfinite(features), axis=1) & (np.any(features != 0, axis=1))
    valid_features = features[mask]
    del data, selected_data, features
    gc.collect()

    print(f"Step 1: Clustering for {year}...")
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=4096, n_init=3)
    clusters = kmeans.fit_predict(valid_features)
    
    print(f"Step 2: Auto-labeling for {year}...")
    initial_labels = auto_label_clusters(clusters, valid_features)
    
    print(f"Step 3: Training RF Model for {year}...")
    idx = np.random.choice(len(valid_features), 150000, replace=False)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(valid_features[idx], initial_labels[idx])
    
    print(f"Step 4: Predicting LULC for {year}...")
    preds = predict_in_chunks(rf, valid_features)
    full_map = np.zeros(rows * cols, dtype=np.uint8)
    full_map[mask] = preds
    
    return {
        "map": full_map.reshape(rows, cols),
        "rf": rf,
        "valid_features": valid_features,
        "meta": meta,
        "mask": mask
    }

import json

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    start_time = time.time()
    
    # 1. First Load & Get Master Mask
    # We load both rasters to find pixels that are valid in BOTH years
    def get_features_and_mask(filepath):
        with rasterio.open(filepath) as src:
            select_bands = [1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 24]
            data = src.read(select_bands).astype('float32')
            rows, cols = data.shape[1], data.shape[2]
            features = data.reshape(len(select_bands), -1).T
            mask = np.all(np.isfinite(features), axis=1) & (np.any(features != 0, axis=1))
            return features, mask, rows, cols, src.meta, src.bounds

    f15, m15, rows, cols, meta15, bounds15 = get_features_and_mask(FILES["2015"])
    f24, m24, _, _, meta24, bounds24 = get_features_and_mask(FILES["2024"])
    
    master_mask = m15 & m24  # Shared valid study area
    valid_f15 = f15[master_mask]
    valid_f24 = f24[master_mask]
    del f15, f24, m15, m24
    gc.collect()

    # 2. Train Models & Predict
    def train_and_map(year, features, mask, rows, cols):
        print(f"Processing {year}...")
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=4096, n_init=3)
        clusters = kmeans.fit_predict(features)
        labels = auto_label_clusters(clusters, features)
        
        idx = np.random.choice(len(features), 150000, replace=False)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(features[idx], labels[idx])
        
        preds = rf.predict(features)
        full_map = np.zeros(rows * cols, dtype=np.uint8)
        full_map[mask] = preds
        return full_map.reshape(rows, cols), rf

    map_15, rf15 = train_and_map("2015", valid_f15, master_mask, rows, cols)
    map_24, rf24 = train_and_map("2024", valid_f24, master_mask, rows, cols)

    # 3. AI Confidence & Transition Likelihood
    m_conf15, m_conf24, m_trans = calculate_confidence_and_transitions(
        rf15, rf24, valid_f15, valid_f24, master_mask, rows, cols
    )
    
    # 4. Export Dash Data
    pixel_area_km2 = (30 * 30) / 1e6
    dashboard_stats = {"years": {"2015": {}, "2024": {}}, "class_names": []}
    
    for year, m in [("2015", map_15), ("2024", map_24)]:
        unique, counts = np.unique(m[m != 0], return_counts=True)
        total_area = counts.sum() * pixel_area_km2
        for c_id, count in zip(unique, counts):
            name = CLASSES[c_id]
            area = count * pixel_area_km2
            dashboard_stats["years"][year][name] = {
                "area_km2": round(float(area), 2),
                "percentage": round(float((area / total_area) * 100), 2)
            }

    trans_cm = confusion_matrix(map_15[master_mask.reshape(rows,cols)], map_24[master_mask.reshape(rows,cols)])
    
    # Governance Insights
    forest_to_built = np.sum((map_15 == 1) & (map_24 == 5)) * pixel_area_km2
    dashboard_stats["governance"] = {
        "urban_sprawl_km2": round(float(forest_to_built), 2),
        "avg_confidence": round(float(np.mean(m_conf24[master_mask.reshape(rows,cols)])), 3),
        "high_likelihood_changes": int(np.sum(m_trans > 0.8))
    }

    class_names = [CLASSES[i] for i in sorted(CLASSES.keys())]
    dashboard_stats["class_names"] = class_names
    dashboard_stats["matrix"] = trans_cm.tolist()

    # Geo Bounds
    from rasterio.warp import transform_bounds
    b = transform_bounds(meta15['crs'], 'EPSG:4326', *bounds15)
    dashboard_stats["bounds"] = [[b[1], b[0]], [b[3], b[2]]]
    
    with open("lulc_analytics.json", "w") as f:
        json.dump(dashboard_stats, f, indent=4)
        
    # Save Maps
    cmap = mcolors.ListedColormap([CLASS_COLORS[CLASSES[i]] for i in sorted(CLASSES.keys())])
    plt.imsave("map_2015.png", map_15, cmap=cmap)
    plt.imsave("map_2024.png", map_24, cmap=cmap)
    plt.imsave("confidence_2024.png", m_conf24, cmap='viridis')
    plt.imsave("transition_likelihood.png", m_trans, cmap='magma')
    
    change_map = np.zeros_like(map_15, dtype=np.uint8)
    change_map[master_mask.reshape(rows,cols) & (map_15 != map_24)] = 1
    plt.imsave("change_map.png", change_map, cmap='Reds')

    print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
