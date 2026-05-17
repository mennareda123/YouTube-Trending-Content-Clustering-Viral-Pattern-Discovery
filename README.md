# YouTube Trending Content Clustering & Viral Pattern Discovery

## Project Overview

Unsupervised deep learning system that clusters YouTube trending videos across US, GB, and India using KMeans, GMM, DBSCAN, SOM, and DEC with PyTorch autoencoders, then predicts viral potential via KNN and RBF classifiers.

## Team Members

| Person | Responsibility |
|--------|----------------|
| Person 1 | Data Engineering, EDA, Feature Engineering, Scaling, PCA/t-SNE |
| Person 2 | Classical Clustering (KMeans, GMM, DBSCAN, Anomaly Detection) |
| Person 3 | Deep Learning (Autoencoder, DEC with PyTorch) |
| Person 4 | Visualization (SOM, PCA/t-SNE, Country Comparison, Anomalies Plotting) |
| Person 5 | Viral Prediction (KNN, RBF), Dashboard, Integration |

## Dataset

| Attribute | Description |
|-----------|-------------|
| Source | YouTube Trending Videos Dataset |
| Countries | US, GB, IN |
| Raw Size | ~120,000 videos |
| After Cleaning | ~100,000 videos |
| Features | views, likes, dislikes, comment_count, tags, category_id, publish_time |

## Features Engineered

| Feature | Formula |
|---------|---------|
| Engagement Ratio | (likes + comments) / views |
| Like Ratio | likes / views |
| Comment Ratio | comments / views |
| Viral Score | (views*0.4 + likes*0.3 + comments*0.3) / max |
| Tags Count | Number of tags in video |
| Title Length | Length of video title |
| Publish Hour | Hour of day (0-23) |
| Publish Day | Day of week (0-6) |

## Models Used

| Model | Type | Purpose |
|-------|------|---------|
| KMeans | Classical | Baseline clustering |
| GMM | Classical | Probabilistic clustering |
| DBSCAN | Classical | Anomaly detection |
| SOM | Classical | 2D visualization map |
| Autoencoder | Deep Learning | Dimensionality reduction to latent space |
| DEC | Deep Learning | Clustering in latent space |
| KNN | Prediction | Viral video classification |
| RBF | Prediction | Viral video classification |

## Results

| Model | Silhouette Score |
|-------|------------------|
| KMeans (Original Space) | 0.32 |
| KMeans (Latent Space) | 0.84 |
| DEC | 0.84 |

## Repository Structure

```
YouTube-Trending-Clustering/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Original CSV and JSON files
│   └── processed/           # Cleaned data, X_scaled.npy, latent_features.npy
│
├── notebooks/
│   ├── ML_PROJECT
│
├── models/
│   ├── autoencoder_model.pth
│   ├── scaler.pkl
│   ├── pca_2d.pkl
│   └── kmeans_model.pkl
│
├── visualizations/
│   ├── som_umatrix.png
│   ├── som_grid.png
│   ├── pca_2d.png
│   ├── tsne.png
│   ├── country_clusters.png
│   ├── anomalies.png
│   ├── autoencoder_loss.png
│   ├── latent_space.png
│   └── models_comparison.png
│
├── results/
│   ├── dec_vs_kmeans_comparison.csv
│   ├── anomalies_analysis.csv
│   └── model_metrics.csv
│
├── streamlit/
│   └── app1.py
│
└── reports/
    └── project_report.pdf
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/mennareda123/YouTube-Trending-Clustering.git
cd YouTube-Trending-Clustering
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Place the CSV and JSON files in `data/raw/` folder.

### 4. Run preprocessing

Open and run `notebooks/01_Data_Preprocessing.ipynb`

### 5. Run clustering models

Run notebooks in order:
- `02_Classical_Clustering.ipynb`
- `03_Deep_Learning_Autoencoder_DEC.ipynb`
- `04_Visualization_SOM.ipynb`

### 6. Run dashboard

```bash
streamlit run app.py
```

## Key Findings

1. DEC achieved Silhouette Score of 0.84, significantly outperforming classical KMeans (0.32)
2. DBSCAN detected approximately 2-3% of videos as anomalies
3. Country comparison showed US prefers entertainment, India prefers music, GB prefers comedy
4. Viral prediction accuracy reached 76% using RBF classifier

## Dashboard Features

- Select between KMeans, GMM, DBSCAN, DEC results
- View cluster distributions by country
- Interactive anomaly detection viewer
- Viral prediction tool for new videos

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Main programming language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Classical ML, clustering, metrics |
| PyTorch | Autoencoder, DEC deep learning |
| Matplotlib, Seaborn | Static visualizations |
| MiniSOM | Self Organizing Map |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard web application |

## Conclusion

The DEC model successfully identified meaningful content clusters in YouTube trending videos with high silhouette score (0.84), proving that deep embedded clustering is superior to classical methods for this type of social media data. The system can effectively detect anomalies, compare trends across countries, and predict viral potential of new videos.

## Future Work

- Implement LSTM-Autoencoder for time series patterns
- Add more countries (Canada, Australia, Germany)
- Integrate BERT for title and tag embeddings
- Deploy dashboard as web application

## License

MIT

## Acknowledgments

- YouTube Trending Dataset from Kaggle
- DEC paper: "Unsupervised Deep Embedding for Clustering Analysis"
- PyTorch and Scikit-learn documentation
```
