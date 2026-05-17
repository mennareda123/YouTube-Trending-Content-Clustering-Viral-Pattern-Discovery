import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="YouTube Trend Analyzer", layout="wide", page_icon="🎬")

st.title("🎬 YouTube Trend Analyzer")
st.markdown("### Analyze, Cluster & Predict YouTube Trending Videos")
st.markdown("---")


if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'X_scaled' not in st.session_state:
    st.session_state['X_scaled'] = None
if 'latent' not in st.session_state:
    st.session_state['latent'] = None
if 'is_anomaly' not in st.session_state:
    st.session_state['is_anomaly'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = None
if 'clustering_done' not in st.session_state:
    st.session_state['clustering_done'] = False
if 'error_message' not in st.session_state:
    st.session_state['error_message'] = None
if 'folder_path' not in st.session_state:
    st.session_state['folder_path'] = r"C:\Users\titoa\Downloads\Applied"


st.sidebar.header("📁 Data Loading")

folder_path = st.sidebar.text_input("Dataset Folder Path", st.session_state['folder_path'])


if st.sidebar.button("📂 Load Data", type="primary"):
    st.session_state['error_message'] = None
    
    with st.spinner("Loading data..."):
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            
            df = pd.read_csv(os.path.join(folder_path, "final_data.csv"))
            X_scaled = np.load(os.path.join(folder_path, "X_scaled.npy"))
            
            
            latent_path = os.path.join(folder_path, "latent_features.npy")
            if os.path.exists(latent_path):
                latent = np.load(latent_path)
                st.sidebar.success("✅ Latent features loaded")
            else:
                latent = X_scaled
                st.sidebar.warning("⚠️ Using original features")
            
            
            anomaly_path = os.path.join(folder_path, "is_anomaly.npy")
            if os.path.exists(anomaly_path):
                is_anomaly = np.load(anomaly_path)
            else:
                is_anomaly = None
            
            
            st.session_state['data_loaded'] = True
            st.session_state['df'] = df
            st.session_state['X_scaled'] = X_scaled
            st.session_state['latent'] = latent
            st.session_state['is_anomaly'] = is_anomaly
            st.session_state['folder_path'] = folder_path
            
            st.sidebar.success(f"✅ Loaded {len(df)} videos")
            st.rerun()
            
        except Exception as e:
            st.session_state['error_message'] = str(e)
            st.sidebar.error(f"❌ Error: {e}")



if st.session_state['error_message']:
    st.error(f"❌ Loading failed: {st.session_state['error_message']}")
    st.info("💡 Make sure your folder contains: final_data.csv and X_scaled.npy")
    st.stop()

if not st.session_state['data_loaded']:
    st.info("👈 Please load your data from the sidebar first")
    st.stop()


df = st.session_state['df']
X_scaled = st.session_state['X_scaled']
latent = st.session_state['latent']
is_anomaly = st.session_state['is_anomaly']


st.sidebar.markdown("---")
st.sidebar.header("🎯 Model Settings")

model_type = st.sidebar.radio(
    "Select Model",
    ["KMeans", "KMeans on Latent", "DEC (Pre-trained)"]
)

n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 5, 2, 1)
run_clustering = st.sidebar.button("🚀 Run Clustering", type="primary")



if run_clustering:
    with st.spinner("Running clustering..."):
        
        if model_type == "KMeans":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            data_used = "Original Features"
            
        elif model_type == "KMeans on Latent":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(latent)
            data_used = "Latent Features (Autoencoder)"
            
        else:
            dec_path = os.path.join(st.session_state['folder_path'], "dec_labels.npy")
            if os.path.exists(dec_path):
                labels = np.load(dec_path)
                data_used = "DEC Pre-trained Labels"
            else:
                labels = np.zeros(len(df))
                data_used = "DEC (Fallback)"
        
        st.session_state['labels'] = labels
        st.session_state['clustering_done'] = True
        st.session_state['data_used'] = data_used
        
        st.sidebar.success("✅ Clustering complete!")
        st.rerun()


if st.session_state.get('clustering_done', False):
    
    labels = st.session_state['labels']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Videos", len(df))
    with col2:
        st.metric("Number of Clusters", n_clusters)
    with col3:
        st.metric("Data Source", st.session_state.get('data_used', 'Unknown'))
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔮 Predict New Video", "🌍 Country Analysis"])
    
    with tab1:
        st.subheader("📈 Cluster Visualizations")
        
        viz_option = st.radio("Choose visualization:", ["PCA 2D", "t-SNE 2D"], horizontal=True)
        
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        
        if viz_option == "PCA 2D":
            pca = PCA(n_components=2, random_state=42)
            viz_data = pca.fit_transform(latent)
            scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], 
                                c=labels, cmap='viridis', alpha=0.5, s=15)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('PCA 2D Visualization')
        else:
            with st.spinner("🔄 Running t-SNE..."):
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                sample_size = min(3000, len(latent))
                viz_data = tsne.fit_transform(latent[:sample_size])
                scatter = ax.scatter(viz_data[:, 0], viz_data[:, 1], 
                                    c=labels[:sample_size], cmap='viridis', alpha=0.5, s=15)
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title('t-SNE 2D Visualization')
        
        cbar = fig.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("📊 Cluster Distribution")
        st.bar_chart(pd.Series(labels).value_counts())
        
        if is_anomaly is not None:
            st.subheader("⚠️ Anomaly Detection Results")
            anomaly_count = is_anomaly.sum()
            st.metric("Detected Anomalies", f"{anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")
    
    with tab2:
        st.subheader("🔮 Predict Video Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_views = st.number_input("Views", min_value=0, value=100000, step=10000)
            pred_likes = st.number_input("Likes", min_value=0, value=5000, step=500)
            pred_comments = st.number_input("Comments", min_value=0, value=1000, step=100)
        
        with col2:
            pred_dislikes = st.number_input("Dislikes", min_value=0, value=200, step=50)
            pred_tags = st.number_input("Number of Tags", min_value=0, value=10, step=1)
            pred_title_len = st.number_input("Title Length", min_value=0, value=50, step=5)
        
        if st.button("🎯 Predict Cluster", type="primary"):
            
            likes = pred_likes
            dislikes = pred_dislikes
            views = pred_views
            
            
            if likes + dislikes > views:
                st.error("❌ Invalid input: Likes + Dislikes cannot exceed Views!")
                st.info("💡 Tip: A user must watch the video before liking/disliking it.")
            else:
                max_views = 2000000
                viral_score = (views * 0.4 + likes * 0.3 + pred_comments * 0.3) / max_views
                viral_score = min(viral_score, 1.0)
                
                if viral_score > 0.6:
                    predicted_cluster = 1
                    status = "🔥 VIRAL"
                    color = "red"
                elif viral_score > 0.3:
                    predicted_cluster = 0
                    status = "📹 NORMAL"
                    color = "orange"
                else:
                    predicted_cluster = 0
                    status = "📹 LOW ENGAGEMENT"
                    color = "gray"
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Viral Score", f"{viral_score:.3f}")
                with res_col2:
                    st.metric("Predicted Cluster", f"Cluster {predicted_cluster}")
                with res_col3:
                    st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
                
                st.progress(viral_score)
                
                if viral_score < 0.3:
                    st.info("💡 Tip: Add more engaging content, hashtags, or improve thumbnail")
                elif viral_score < 0.6:
                    st.info("💡 Tip: Good start! Share on social media to boost views")
                else:
                    st.success("🎉 Excellent metrics! High viral potential")
    with tab3:
        st.subheader("🌍 Country Analysis")
        
        if 'country' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                country_a = st.selectbox("Country A", ['US', 'GB', 'IN'], index=0)
            with col2:
                country_b = st.selectbox("Country B", ['US', 'GB', 'IN'], index=1)
            
            if st.button("📊 Compare Countries"):
                mask_a = df['country'] == country_a
                mask_b = df['country'] == country_b
                
                dist_a = pd.Series(labels[mask_a]).value_counts(normalize=True) * 100
                dist_b = pd.Series(labels[mask_b]).value_counts(normalize=True) * 100
                
                comparison = pd.DataFrame({country_a: dist_a, country_b: dist_b}).fillna(0).round(1)
                
                st.subheader(f"📈 Cluster Distribution: {country_a} vs {country_b}")
                st.bar_chart(comparison)
                st.dataframe(comparison, use_container_width=True)
                
                
                if len(dist_a) > 0 and len(dist_b) > 0:
                    dom_a = dist_a.idxmax()
                    dom_b = dist_b.idxmax()
                    st.write(f"**{country_a}** is dominated by **Cluster {dom_a}** ({dist_a[dom_a]:.1f}%)")
                    st.write(f"**{country_b}** is dominated by **Cluster {dom_b}** ({dist_b[dom_b]:.1f}%)")
        else:
            st.warning("Country column not found in dataset")

else:
    st.info("👈 Select model settings and click 'Run Clustering' to start")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎬 YouTube Trend Analyzer | Powered by KMeans, Autoencoder & DEC"
    "</div>",
    unsafe_allow_html=True
)