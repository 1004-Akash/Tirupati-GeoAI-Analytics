# 🎯 Tirupati GeoAI Analytics: Pixel-Level LULC Intelligence

[![AI-Powered](https://img.shields.io/badge/AI-Random%20Forest-green.svg)](#)
[![Geospatial](https://img.shields.io/badge/GIS-Rasterio%20%2B%20Leaflet-blue.svg)](#)
[![SmartCity](https://img.shields.io/badge/Governance-Decision Support-orange.svg)](#)

An end-to-end, automated remote sensing and AI-based framework for pixel-level Land Use/Land Cover (LULC) change detection and transition analytics for Tirupati District.

## 🚀 Overview
Tirupati has undergone rapid urban transformation. This project provides a **GeoAI Pipeline** and an **Interactive Governance Dashboard** to monitor these changes with pixel-precise accuracy (30m resolution), enabling sustainable urban planning and evidence-based policymaking.

---

## 🛠️ Technical Solution & Features

### 1. AI-Powered Analytics Pipeline (`lulc_change_detection.py`)
- **Multi-Temporal Alignment**: Automated alignment and normalization of Landsat 8 datasets for 2015 and 2024.
- **Hybrid Classification**: Combines Unsupervised `MiniBatchKMeans` for initial cluster discovery with Supervised `RandomForestClassifier` for spatial refinement.
- **Probabilistic Transition Modeling**: Calculates the likelihood of class-to-class conversions using joint probability maps.
- **Memory Efficient**: Processes 17M+ pixels in chunks to support execution in resource-constrained environments like Google Colab.

### 2. Change Intelligence & Stats
- **Transition Matrix**: A full 5x5 matrix capturing pixel-level conversions between:
  - Forest, Water Bodies, Agriculture, Barren Land, Built-up.
- **AI Confidence Mapping**: Generates pixel-level certainty scores to distinguish between reliable change and spectral noise.
- **Urban Sprawl Quantification**: Specifically tracks vegetation-to-urban transitions in square kilometers.

### 3. Interactive Smart City Dashboard (React)
- **Spatial Explorer**: Multi-layer Leaflet integration with:
  - LULC Map Toggles (2015 vs 2024).
  - **Transition Likelihood Heatmap**: Visualizes where land cover is most likely changing.
  - **Hotspot Overlays**: Highlights critical encroachment areas.
- **Analytics & Export**: 
  - Dynamic charts (Area distribution & Growth trends).
  - Transition magnitude tables.
  - Policy recommendations driven by AI insights.

---

## 📂 Project Structure
```text
├── lulc_change_detection.py    # Core GeoAI Pipeline (Python)
├── lulc_analytics.json          # Exported transition statistics
├── dashboard/                   # React + Vite + Tailwind CSS v4 Frontend
│   ├── src/App.jsx              # Main Dashboard Logic
│   └── public/                  # Generated AI maps and overlays
└── requirements.txt             # Dependency list
```

## ⚙️ Installation & Usage

### Backend (Python)
```bash
pip install rasterio numpy pandas scikit-learn matplotlib geopandas
python lulc_change_detection.py
```

### Frontend (Dashboard)
```bash
cd dashboard
npm install
npm run dev
```

---

## 📊 Technical Evaluation Alignment
1. **Accuracy**: Uses RF probability estimates for robust classification.
2. **Completeness**: Full transition matrix with quantitative stats (km² and percentages).
3. **Interpretability**: Confidence maps help planners verify AI outputs.
4. **Scalability**: Chunked processing architecture.
5. **Innovation**: Integrated "Governance Action" labeling for detected transitions.

## 🌍 Impact
This framework contributes to **Sustainable Urban Governance** by providing a replicable workflow deployable across Indian districts, aligning with National Geospatial initiatives.

---
**Developed for the Tirupati GeoAI Challenge.**
**Author:** AI-Developer with SmartCity GeoAI Framework.
