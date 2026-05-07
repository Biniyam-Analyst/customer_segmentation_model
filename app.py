import streamlit  as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.sidebar.markdown("**Running file: app.py**")

st.title("📊 Customer Segmentation Dashboard")
st.write("Uses customer information (Income, Total Spend, Age) to identify which customer segment they belong to.")

# 1. Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        # Load the trained model and scaler
        Kmeans = joblib.load('Kmeans_Model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Load the dataset
        df = pd.read_csv('customer_segmentation_data.csv')

        # Recreate the features used in training
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
        current_date = pd.Timestamp.today()
        df['Tenure'] = (current_date - df['Dt_Customer']).dt.days
        df['Age'] = 2026 - df['Year_Birth']
        mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
        df['Total_Spend'] = df[mnt_cols].sum(axis=1)
        df['Family_Size'] = df['Kidhome'] + df['Teenhome'] + 1
        # Use the exact feature names the scaler was trained on
        relevant_features = [
            'Income', 'Age', 'Total_Spend', 'Recency',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'NumDealsPurchases', 'Kidhome', 'Teenhome'
        ]

        df = df.dropna(subset=relevant_features)
        df_selected = df[relevant_features].copy()
        scaled = scaler.transform(df_selected.fillna(0))

        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(scaled)

        # Pre-predict clusters for all customers
        df['Cluster'] = Kmeans.predict(df_pca)
        df['PC1'] = df_pca[:, 0]
        df['PC2'] = df_pca[:, 1]
        mean_values = df_selected.mean()
        return Kmeans, scaler, pca, df, relevant_features, mean_values
    except FileNotFoundError:
        return None, None, None, None, None, None

Kmeans, scaler, pca, df, relevant_features, mean_values = load_model_and_data()

if Kmeans is None:
    st.error("Could not load the model or data! Please make sure the files exist.")
    st.stop()

# Cluster details
cluster_info = {
    0: {
        'name': 'Cautious Spenders', # low income/spending
        'color': '#3498db',
        'strategy': 'Offer value-based incentives to encourage more spending.'
    },
    1: {
        'name': 'High-Value Spenders', #high income/spending
        'color': '#27ae60',
        'strategy': 'Offer premium services and exclusive VIP perks.'
    },
    2: {
        'name': 'Steady Consumers', # medium income/spending
        'color': '#e67e22',
        'strategy': 'Target with loyalty programs and consistent engagement.'
    }
}

# Cluster visualization and data exploration
st.markdown("### Cluster Visualization")
if df is not None:
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        income_cap = df['Income'].quantile(0.99)
        df_plot = df[df['Income'] <= income_cap]
        fig = px.scatter(
            df_plot, x='Income', y='Total_Spend',
            color=df_plot['Cluster'].map(lambda c: cluster_info[c]['name']),
            color_discrete_map={
                cluster_info[0]['name']: '#3498db',
                cluster_info[1]['name']: '#27ae60',
                cluster_info[2]['name']: '#e67e22',
            },
            title='Customer Clusters: Income vs Total Spend',
            labels={'Income': 'Annual Income (k$)', 'Total_Spend': 'Total Spend', 'color': 'Cluster'},
            height=400
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

    with col_plot2:
        color_map = {0: '#3498db', 1: '#27ae60', 2: '#e67e22'}
        fig2 = px.scatter(
            df, x='PC1', y='PC2',
            color=df['Cluster'].map(lambda c: cluster_info[c]['name']),
            color_discrete_map={
                cluster_info[0]['name']: '#3498db',
                cluster_info[1]['name']: '#27ae60',
                cluster_info[2]['name']: '#e67e22',
            },
            title='Visualization of Clusters (PCA)',
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'color': 'Cluster'},
            height=400
        )
        fig2.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig2, use_container_width=True)

# 4. Sidebar Inputs for User
st.sidebar.title("Customer Profile")
st.sidebar.subheader("Enter Customer Details")

income = st.sidebar.number_input('Income', min_value=10000, max_value=2000000, value=52000, step=100)
Age = st.sidebar.slider('Age', min_value=18, max_value=100, value=56, step=1)
Total_Spend = st.sidebar.number_input('Total Spend', min_value=10, max_value=50000, value=600, step=10)

recency = st.sidebar.number_input('Recency', min_value=0, max_value=100, value=10, step=1)
num_web_purchases = st.sidebar.number_input('Num Web Purchases', min_value=0, max_value=50, value=8, step=1)
num_catalog_purchases = st.sidebar.number_input('Num Catalog Purchases', min_value=0, max_value=50, value=10, step=1)
num_store_purchases = st.sidebar.number_input('Num Store Purchases', min_value=0, max_value=50, value=4, step=1)
num_web_visits_month = st.sidebar.number_input('Num Web Visits Month', min_value=0, max_value=30, value=7, step=1)
num_deals_purchases = st.sidebar.number_input('Num Deals Purchases', min_value=0, max_value=30, value=3, step=1)
kidhome = st.sidebar.number_input('Kid Home', min_value=0, max_value=5, value=0, step=1)
teenhome = st.sidebar.number_input('Teen Home', min_value=0, max_value=5, value=0, step=1)

st.sidebar.markdown("---")
segment_btn = st.sidebar.button("Predict Segment", type="primary", use_container_width=True)

# 3. Main page

if segment_btn:
    # Build a full input row using the user values and training averages
    input_row = mean_values.to_dict()
    input_row.update({
        'Income': income,
        'Age': Age,
        'Total_Spend': Total_Spend,
        'Recency': recency,
        'NumWebPurchases': num_web_purchases,
        'NumCatalogPurchases': num_catalog_purchases,
        'NumStorePurchases': num_store_purchases,
        'NumWebVisitsMonth': num_web_visits_month,
        'NumDealsPurchases': num_deals_purchases,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
    }) 

    input_df = pd.DataFrame([input_row], columns=relevant_features)
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)

    predicted_cluster = int(Kmeans.predict(input_pca)[0])
    cluster_details = cluster_info.get(predicted_cluster, {'name': 'Unknown', 'color': '#bdc3c7', 'strategy': 'N/A'})

    # Isolate customers of the predicted cluster
    cluster_customers = df[df['Cluster'] == predicted_cluster]

    # Output section
    st.markdown("---")
    st.header("Segmentation Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Cluster result display
        st.markdown(f"""
        <div style='background-color: {cluster_details['color']}; padding: 2rem; border-radius: 1rem; color: white;'>
            <h2 style='color: white; margin: 0;'>{cluster_details['name']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(" ")
        # Marketing strategy
        st.markdown("### Marketing Strategy")
        st.success(f"**Strategy:** {cluster_details['strategy']}")
        
        # Cluster size
        cluster_size = len(cluster_customers)
        total_customers = len(df)
        st.info(f"**Segment Size:** {cluster_size} customers ({cluster_size/total_customers*100:.1f}% of total)")

    with col2:
        # Position chart
        st.markdown("### Your Position")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Income'], # Annual Income
            y=df['Total_Spend'], # Total Spend
            mode='markers',
            marker=dict(
                color=df['Cluster'], 
                colorscale=['#3498db', '#e67e22', '#27ae60'], 
                size=8
            ),
            name='All Customers'
        ))
        
        # User's position
        fig.add_trace(go.Scatter(
            x=[income],
            y=[Total_Spend],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Your Position'
        ))
        
        fig.update_layout(
            title='Income vs Total Spend',
            xaxis_title='Annual Income (k$)',
            yaxis_title='Total Spend (1-100)',
            height=350,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


    # Three customer segments summary
    st.markdown("### Three Customer Segments")
    cols = st.columns(3)
    
    for cluster_id, info in cluster_info.items():
        with cols[cluster_id]:
            cluster_customers = df[df['Cluster'] == cluster_id]
            cluster_count = len(cluster_customers)
            
            st.markdown(f"""
            <div style='background-color: {info['color']}; padding: 1rem; border-radius: 0.5rem; color: white; text-align: center; height: 130px;'>
                <h4 style='color: white; margin: 0;'>{info['name']}</h4>
                <p style='color: white; font-size: 0.9rem; margin-top: 10px;'>{cluster_count} Customers</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed segment comparison
    st.markdown("---")
    st.subheader("Segment Comparison")
    
    comparison_data = []
    
    # Loop over the 3 clusters
    for cluster_id in range(3):
        cluster_customers = df[df['Cluster'] == cluster_id]
        
        avg_income = cluster_customers['Income'].mean()
        avg_spending = cluster_customers['Total_Spend'].mean()
        avg_age = cluster_customers['Age'].mean()
            
        comparison_data.append({
            'Segment': cluster_info[cluster_id]['name'],
            'Count': len(cluster_customers),
            'Avg Income': f"${avg_income:.0f}k",
            'Avg Spending': f"{avg_spending:.0f}/100",
            'Avg Age': f"{avg_age:.0f}",
            'Strategy': cluster_info[cluster_id]['strategy']
            })
            
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
else:
    # Initial page when the button is not clicked
    st.markdown("---")
    st.info("Enter customer information in the sidebar to find their segment.")
       

# Cluster samples
st.markdown("---")
st.subheader("Cluster Samples")

samples = df.sample(n=min(5, len(df)), random_state=42)[
    ['Cluster', 'Income', 'Age', 'Total_Spend', 'Recency',
     'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
     'NumWebVisitsMonth', 'NumDealsPurchases', 'Kidhome', 'Teenhome']
].copy()

samples['Cluster'] = samples['Cluster'].map(lambda c: cluster_info[c]['name'])
samples.columns = ['Segment', 'Income', 'Age', 'Total Spend', 'Recency',
                   'Web Purchases', 'Catalog Purchases', 'Store Purchases',
                   'Web Visits/Month', 'Deals Purchases', 'Kid Home', 'Teen Home']

st.dataframe(samples.reset_index(drop=True), use_container_width=True, hide_index=True)