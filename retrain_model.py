import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load & preprocess
df = pd.read_csv('customer_segmentation_data.csv')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
current_date = pd.Timestamp.today()
df['Tenure'] = (current_date - df['Dt_Customer']).dt.days
df['Age'] = 2026 - df['Year_Birth']
mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
df['Total_Spend'] = df[mnt_cols].sum(axis=1)
df['Family_Size'] = df['Kidhome'] + df['Teenhome'] + 1

relevant_features = [
    'Income', 'Age', 'Total_Spend', 'Recency',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'NumDealsPurchases', 'Kidhome', 'Teenhome'
]

df = df.dropna(subset=relevant_features)
df_selected = df[relevant_features].copy()

# Scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected)

# PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled_data)

# Train KMeans on PCA data
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(df_pca)
df['Cluster'] = labels

# Re-map cluster labels so they are always consistent:
# 0 = Cautious Spenders (lowest spend)
# 1 = High-Value Spenders (highest spend)
# 2 = Steady Consumers (mid spend)
cluster_avg_spend = df.groupby('Cluster')['Total_Spend'].mean().sort_values()
sorted_clusters = cluster_avg_spend.index.tolist()
# sorted_clusters[0]=lowest, [1]=mid, [2]=highest
remap = {
    sorted_clusters[0]: 0,  # Cautious Spenders
    sorted_clusters[1]: 2,  # Steady Consumers
    sorted_clusters[2]: 1,  # High-Value Spenders
}
df['Cluster'] = df['Cluster'].map(remap)

# Apply same remap to kmeans labels so the saved model is consistent
import numpy as np
new_labels = np.array([remap[l] for l in labels])
kmeans.labels_ = new_labels

print("--- Cluster Averages (after remap) ---")
print(df.groupby('Cluster')[['Income', 'Total_Spend', 'Age', 'Kidhome', 'Teenhome']].mean().round(1))
print("\n--- Number of customers ---")
print(df['Cluster'].value_counts().sort_index())

joblib.dump(kmeans, 'Kmeans_Model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nSaved!")
