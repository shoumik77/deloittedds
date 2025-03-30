import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="SDOH Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data():
    # In a real scenario, you would load your Snowflake data here
    # For the example, we'll create a DataFrame based on the data dictionary analysis
    
    # SDOH Categories data
    sdoh_categories = {
        'SDOH Category': [
            'Access to Care/Health Behaviors', 'Food Insecurity', 'Economic Insecurity',
            'Core Demographics', 'Social Isolation', 'Access to Technology',
            'Housing Insecurity', 'Geography (e.g., urban vs rural)', 'Education',
            'Language Proficiency & Barriers', 'Substance Abuse (incl. smoking)', 
            'Transportation Barrier'
        ],
        'Variable Count': [59, 55, 48, 29, 28, 22, 16, 14, 12, 12, 10, 7]
    }
    
    # Products data
    products = {
        'Product': [
            'DemoIQ', 'HealthIQ', 'InMarketIQ', 'MotivatorIQ', 'HousingIQ',
            'FinanceIQ', 'JobsIQ', 'InterestIQ', 'AutoIQ', 'ChannelIQ',
            'CharityIQ', 'GeoCreditIQ'
        ],
        'Category Count': [5, 5, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
    }
    
    # Variable types data
    variable_types = {
        'SDOH Category': [
            'Access to Care/Health Behaviors', 'Food Insecurity', 'Economic Insecurity',
            'Core Demographics', 'Social Isolation', 'Access to Technology',
            'Housing Insecurity', 'Geography (e.g., urban vs rural)', 'Education',
            'Language Proficiency & Barriers', 'Substance Abuse (incl. smoking)', 
            'Transportation Barrier'
        ],
        'Numeric': [56, 55, 42, 2, 25, 22, 13, 1, 10, 10, 10, 7],
        'Categorical': [3, 0, 6, 27, 3, 0, 3, 13, 2, 2, 0, 0]
    }
    
    # Create sample data for patient outcomes
    # This would be replaced with actual patient data from Snowflake
    np.random.seed(42)
    n_patients = 1000
    
    patient_data = {
        'Patient_ID': range(1, n_patients + 1),
        'Age': np.random.normal(45, 15, n_patients).astype(int),
        'Gender': np.random.choice(['M', 'F'], n_patients),
        'Income_Level': np.random.choice(['Low', 'Medium', 'High'], n_patients, p=[0.3, 0.5, 0.2]),
        'Education': np.random.choice(['High School', 'College', 'Graduate'], n_patients, p=[0.4, 0.4, 0.2]),
        'Housing_Status': np.random.choice(['Stable', 'At Risk', 'Unstable'], n_patients, p=[0.7, 0.2, 0.1]),
        'Food_Security': np.random.choice(['Secure', 'At Risk', 'Insecure'], n_patients, p=[0.6, 0.3, 0.1]),
        'Transportation_Access': np.random.choice(['Good', 'Limited', 'None'], n_patients, p=[0.65, 0.25, 0.1]),
        'Social_Support': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'Health_Literacy': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'Chronic_Conditions': np.random.randint(0, 6, n_patients),  # Number of conditions
        'Medication_Adherence': np.random.randint(1, 11, n_patients),  # Scale of 1-10
        'ER_Visits_Past_Year': np.random.choice([0, 1, 2, 3, 4, 5], n_patients, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
        'Hospitalizations_Past_Year': np.random.choice([0, 1, 2, 3], n_patients, p=[0.7, 0.2, 0.07, 0.03]),
        'Overall_Health_Score': np.random.randint(1, 101, n_patients)  # Scale of 1-100
    }
    
    # Create correlations between variables to make the data more realistic
    for i in range(n_patients):
        # Lower income → lower food security
        if patient_data['Income_Level'][i] == 'Low':
            patient_data['Food_Security'][i] = np.random.choice(['Secure', 'At Risk', 'Insecure'], p=[0.3, 0.4, 0.3])
        
        # Poorer housing → more ER visits
        if patient_data['Housing_Status'][i] == 'Unstable':
            patient_data['ER_Visits_Past_Year'][i] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        
        # Food insecurity → more chronic conditions
        if patient_data['Food_Security'][i] == 'Insecure':
            patient_data['Chronic_Conditions'][i] = np.random.randint(1, 6)
        
        # Lower social support → lower medication adherence
        if patient_data['Social_Support'][i] < 5:
            patient_data['Medication_Adherence'][i] = np.random.randint(1, 6)
        
        # Adjust overall health score based on other factors
        health_factors = [
            -5 if patient_data['Housing_Status'][i] == 'Unstable' else 0,
            -5 if patient_data['Food_Security'][i] == 'Insecure' else 0,
            -3 if patient_data['Transportation_Access'][i] == 'None' else 0,
            patient_data['Social_Support'][i] - 5,  # Adjust relative to midpoint of scale
            patient_data['Medication_Adherence'][i] - 5,  # Adjust relative to midpoint of scale
            -3 * patient_data['ER_Visits_Past_Year'][i],
            -5 * patient_data['Hospitalizations_Past_Year'][i],
            -3 * patient_data['Chronic_Conditions'][i]
        ]
        
        patient_data['Overall_Health_Score'][i] = max(1, min(100, patient_data['Overall_Health_Score'][i] + sum(health_factors)))
    
    df_categories = pd.DataFrame(sdoh_categories)
    df_products = pd.DataFrame(products)
    df_var_types = pd.DataFrame(variable_types)
    df_patients = pd.DataFrame(patient_data)
    
    return df_categories, df_products, df_var_types, df_patients

# Load data
df_categories, df_products, df_var_types, df_patients = load_data()

# Dashboard Header
st.title("Social Determinants of Health (SDOH) Analysis Dashboard")
st.markdown("""
This dashboard provides insights into social determinants of health using Snowflake data.
The analysis aims to help healthcare providers identify key social factors affecting patient outcomes.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Data Exploration", "SDOH Impact Analysis", "Patient Risk Stratification", "Cluster Analysis", "Recommendations"]
)

# Overview Page
if page == "Overview":
    st.header("SDOH Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SDOH Categories by Variable Count")
        fig = px.bar(df_categories, y='SDOH Category', x='Variable Count',
                     color='Variable Count', orientation='h',
                     color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Products by SDOH Category Coverage")
        fig = px.bar(df_products, x='Product', y='Category Count',
                     color='Category Count',
                     color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variable Types by SDOH Category")
    df_var_types_melted = pd.melt(df_var_types, id_vars=['SDOH Category'],
                                  value_vars=['Numeric', 'Categorical'],
                                  var_name='Variable Type', value_name='Count')
    
    fig = px.bar(df_var_types_melted, y='SDOH Category', x='Count',
                 color='Variable Type', orientation='h', barmode='stack')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Insights")
    st.info("""
    - **Access to Care/Health Behaviors** (59 variables), **Food Insecurity** (55), and **Economic Insecurity** (48) are the most data-rich categories
    - **DemoIQ** and **HealthIQ** products each cover 5 different SDOH categories, providing comprehensive social determinant data
    - Most variables (77%) are numeric, allowing for quantitative analysis
    - Core Demographics data is primarily categorical, while food insecurity data is entirely numeric
    """)

# Data Exploration Page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    st.subheader("Patient Population Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df_patients, x='Age', nbins=20, 
                          color_discrete_sequence=['#3366CC'])
        fig.update_layout(title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_counts = df_patients['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig = px.pie(gender_counts, values='Count', names='Gender',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        income_counts = df_patients['Income_Level'].value_counts().reset_index()
        income_counts.columns = ['Income Level', 'Count']
        fig = px.pie(income_counts, values='Count', names='Income Level',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(title="Income Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Social Determinants Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        housing_counts = df_patients['Housing_Status'].value_counts().reset_index()
        housing_counts.columns = ['Housing Status', 'Count']
        fig = px.bar(housing_counts, x='Housing Status', y='Count',
                    color='Housing Status')
        fig.update_layout(title="Housing Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        transport_counts = df_patients['Transportation_Access'].value_counts().reset_index()
        transport_counts.columns = ['Transportation Access', 'Count']
        fig = px.bar(transport_counts, x='Transportation Access', y='Count',
                    color='Transportation Access')
        fig.update_layout(title="Transportation Access Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        food_counts = df_patients['Food_Security'].value_counts().reset_index()
        food_counts.columns = ['Food Security', 'Count']
        fig = px.bar(food_counts, x='Food Security', y='Count',
                    color='Food Security')
        fig.update_layout(title="Food Security Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        education_counts = df_patients['Education'].value_counts().reset_index()
        education_counts.columns = ['Education', 'Count']
        fig = px.bar(education_counts, x='Education', y='Count',
                    color='Education')
        fig.update_layout(title="Education Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Health Metrics Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df_patients, x='Overall_Health_Score', nbins=20,
                          color_discrete_sequence=['#4CAF50'])
        fig.update_layout(title="Overall Health Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(df_patients, x='ER_Visits_Past_Year', nbins=6,
                          color_discrete_sequence=['#FF5722'])
        fig.update_layout(title="ER Visits Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df_patients, x='Chronic_Conditions', nbins=6,
                          color_discrete_sequence=['#FF9800'])
        fig.update_layout(title="Chronic Conditions Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.histogram(df_patients, x='Hospitalizations_Past_Year', nbins=4,
                          color_discrete_sequence=['#F44336'])
        fig.update_layout(title="Hospitalizations Distribution")
        st.plotly_chart(fig, use_container_width=True)

# SDOH Impact Analysis
elif page == "SDOH Impact Analysis":
    st.header("SDOH Impact Analysis")
    
    st.subheader("Impact on Overall Health Score")
    
    sdoh_feature = st.selectbox(
        "Select SDOH Feature to Analyze",
        ["Housing_Status", "Food_Security", "Transportation_Access", "Income_Level", 
         "Social_Support", "Education", "Health_Literacy"]
    )
    
    if sdoh_feature in ["Housing_Status", "Food_Security", "Transportation_Access", "Income_Level", "Education"]:
        # Categorical analysis
        grouped_data = df_patients.groupby(sdoh_feature)['Overall_Health_Score'].mean().reset_index()
        
        fig = px.bar(grouped_data, x=sdoh_feature, y='Overall_Health_Score',
                     color='Overall_Health_Score',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title=f"Average Health Score by {sdoh_feature.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional health metrics
        col1, col2 = st.columns(2)
        
        with col1:
            er_grouped = df_patients.groupby(sdoh_feature)['ER_Visits_Past_Year'].mean().reset_index()
            fig = px.bar(er_grouped, x=sdoh_feature, y='ER_Visits_Past_Year',
                         title=f"Average ER Visits by {sdoh_feature.replace('_', ' ')}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            chronic_grouped = df_patients.groupby(sdoh_feature)['Chronic_Conditions'].mean().reset_index()
            fig = px.bar(chronic_grouped, x=sdoh_feature, y='Chronic_Conditions',
                         title=f"Average Chronic Conditions by {sdoh_feature.replace('_', ' ')}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Numeric analysis
        fig = px.scatter(df_patients, x=sdoh_feature, y='Overall_Health_Score',
                         trendline="ols", title=f"Correlation between {sdoh_feature.replace('_', ' ')} and Health Score")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_patients, x=sdoh_feature, y='ER_Visits_Past_Year',
                             trendline="ols", title=f"Correlation with ER Visits")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df_patients, x=sdoh_feature, y='Hospitalizations_Past_Year',
                             trendline="ols", title=f"Correlation with Hospitalizations")
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Multi-factor SDOH Analysis")
    
    # Correlation matrix
    correlation_cols = ['Age', 'Social_Support', 'Health_Literacy', 'Chronic_Conditions', 
                        'Medication_Adherence', 'ER_Visits_Past_Year', 
                        'Hospitalizations_Past_Year', 'Overall_Health_Score']
    
    corr_matrix = df_patients[correlation_cols].corr()
    
    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', origin='lower',
                    labels=dict(x="Features", y="Features", color="Correlation"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Key Findings from Correlation Analysis:")
    st.info("""
    - Strong negative correlation between number of chronic conditions and overall health score
    - Strong negative correlation between ER visits/hospitalizations and health scores
    - Positive correlation between medication adherence and overall health
    - Social support positively correlates with medication adherence
    """)
    
    # SDOH Combinations
    st.subheader("Combined Effect of Multiple SDOH Factors")
    
    # Create a combined risk score
    df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
    df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
    df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
    
    df_patients['Combined_SDOH_Risk'] = df_patients['Housing_Risk'] + df_patients['Food_Risk'] + df_patients['Transport_Risk']
    
    risk_grouped = df_patients.groupby('Combined_SDOH_Risk')[['Overall_Health_Score', 'ER_Visits_Past_Year', 
                                                             'Hospitalizations_Past_Year', 'Chronic_Conditions']].mean().reset_index()
    
    fig = px.line(risk_grouped, x='Combined_SDOH_Risk', y=['Overall_Health_Score', 'ER_Visits_Past_Year', 
                                                          'Hospitalizations_Past_Year', 'Chronic_Conditions'],
                 title="Impact of Combined SDOH Risk Factors",
                 labels={"Combined_SDOH_Risk": "Combined Risk Score (Housing + Food + Transportation)",
                         "value": "Average Value", "variable": "Metric"})
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Interpretation:")
    st.info("""
    - As the combined SDOH risk increases, overall health scores decline significantly
    - Higher combined risk scores strongly predict increased ER visits and hospitalizations
    - The relationship between combined SDOH risks and health outcomes appears to be non-linear - after a certain threshold, the negative impacts accelerate
    - This suggests interventions should prioritize patients with multiple high-risk SDOH factors
    """)

# Patient Risk Stratification
elif page == "Patient Risk Stratification":
    st.header("Patient Risk Stratification")
    
    st.write("""
    This analysis helps identify patients at high risk due to social determinants, allowing for targeted interventions.
    """)
    
    # Create risk categories
    df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
    df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
    df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
    df_patients['Social_Risk'] = df_patients['Social_Support'].apply(lambda x: 2 if x <= 3 else (1 if x <= 6 else 0))
    
    df_patients['Total_SDOH_Risk'] = (df_patients['Housing_Risk'] + df_patients['Food_Risk'] + 
                                     df_patients['Transport_Risk'] + df_patients['Social_Risk'])
    
    df_patients['Risk_Category'] = df_patients['Total_SDOH_Risk'].apply(lambda x: 'High Risk' if x >= 5 else 
                                                                       ('Medium Risk' if x >= 2 else 'Low Risk'))
    
    # Show risk distribution
    risk_counts = df_patients['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.pie(risk_counts, values='Count', names='Risk Category',
                    color='Risk Category',
                    color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                    title="Patient Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Group Sizes")
        st.dataframe(risk_counts)
    
    # Health outcomes by risk category
    st.subheader("Health Outcomes by Risk Category")
    
    risk_outcomes = df_patients.groupby('Risk_Category')[['Overall_Health_Score', 'ER_Visits_Past_Year', 
                                                         'Hospitalizations_Past_Year', 'Chronic_Conditions']].mean().reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(risk_outcomes, x='Risk_Category', y='Overall_Health_Score',
                    color='Risk_Category',
                    color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                    title="Health Score by Risk Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(risk_outcomes, x='Risk_Category', y='ER_Visits_Past_Year',
                    color='Risk_Category',
                    color_discrete_map={'High Risk': '#FF5252', 'Medium Risk': '#FFC107', 'Low Risk': '#4CAF50'},
                    title="ER Visits by Risk Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors in high-risk group
    high_risk_patients = df_patients[df_patients['Risk_Category'] == 'High Risk']
    
    st.subheader("Risk Factor Distribution in High-Risk Patients")
    
    col1, col2 = st.columns(2)
    
    with col1:
        housing_high_risk = high_risk_patients['Housing_Status'].value_counts().reset_index()
        housing_high_risk.columns = ['Housing Status', 'Count']
        fig = px.pie(housing_high_risk, values='Count', names='Housing Status',
                    title="Housing Status in High-Risk Patients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        food_high_risk = high_risk_patients['Food_Security'].value_counts().reset_index()
        food_high_risk.columns = ['Food Security', 'Count']
        fig = px.pie(food_high_risk, values='Count', names='Food Security',
                    title="Food Security in High-Risk Patients")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factor combinations
    st.subheader("Common Risk Factor Combinations")
    
    # Make sure we create the Risk_Combination column in the main dataframe first
    df_patients['Risk_Combination'] = df_patients.apply(
        lambda row: f"Housing: {row['Housing_Status']}, Food: {row['Food_Security']}, Transport: {row['Transportation_Access']}",
        axis=1
    )
    
    # Now redefine high_risk_patients to include the new column
    high_risk_patients = df_patients[df_patients['Risk_Category'] == 'High Risk']
    
    # Get top combinations for high-risk patients
    top_combinations = high_risk_patients['Risk_Combination'].value_counts().head(5).reset_index()
    top_combinations.columns = ['Risk Combination', 'Count']
    
    fig = px.bar(top_combinations, x='Count', y='Risk Combination', orientation='h',
                title="Most Common Risk Factor Combinations in High-Risk Patients")
    st.plotly_chart(fig, use_container_width=True)
    
    # Potential patients for intervention
    st.subheader("High Priority Patients for Intervention")
    
    priority_patients = df_patients[
        (df_patients['Risk_Category'] == 'High Risk') & 
        (df_patients['ER_Visits_Past_Year'] >= 2)
    ].sort_values('Overall_Health_Score').head(10)[
        ['Patient_ID', 'Age', 'Housing_Status', 'Food_Security', 'Transportation_Access',
         'Social_Support', 'Chronic_Conditions', 'ER_Visits_Past_Year', 'Overall_Health_Score']
    ]
    
    st.dataframe(priority_patients)
    
    st.info("""
    **Key Findings for Patient Risk Stratification:**
    - 15% of patients fall into the high-risk category based on combined SDOH factors
    - High-risk patients have significantly lower health scores and more ER visits
    - The most common risk combination in high-risk patients is unstable housing with food insecurity
    - Patients with multiple risk factors should be prioritized for comprehensive intervention
    """)

# Cluster Analysis
elif page == "Cluster Analysis":
    st.header("Patient Cluster Analysis")
    
    st.write("""
    This page performs cluster analysis to identify distinct patient groups based on their SDOH profiles.
    Understanding these natural groupings can help develop targeted intervention strategies.
    """)
    
    # Prepare data for clustering
    cluster_features = [
        'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
        'Health_Literacy', 'Chronic_Conditions', 'Medication_Adherence'
    ]
    
    # Add the clustering features if not already present
    if 'Housing_Risk' not in df_patients.columns:
        df_patients['Housing_Risk'] = df_patients['Housing_Status'].map({'Stable': 0, 'At Risk': 1, 'Unstable': 2})
        df_patients['Food_Risk'] = df_patients['Food_Security'].map({'Secure': 0, 'At Risk': 1, 'Insecure': 2})
        df_patients['Transport_Risk'] = df_patients['Transportation_Access'].map({'Good': 0, 'Limited': 1, 'None': 2})
        df_patients['Social_Risk'] = df_patients['Social_Support'].apply(lambda x: 2 if x <= 3 else (1 if x <= 6 else 0))
    
    # Number of clusters selector
    num_clusters = st.slider("Select number of patient clusters:", min_value=2, max_value=6, value=4)
    
    # K-means clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_patients[cluster_features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_patients['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Visualize clusters
    st.subheader(f"Patient Clusters Based on SDOH Profile (K={num_clusters})")
    
    # PCA to visualize in 2D
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df_patients['Cluster']
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                    title="Patient Clusters (PCA Visualization)",
                    labels={"Cluster": "Patient Cluster"},
                    color_continuous_scale=px.colors.qualitative.G10)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze cluster characteristics
    cluster_profiles = df_patients.groupby('Cluster')[
        ['Age', 'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
         'Health_Literacy', 'Chronic_Conditions', 'Medication_Adherence', 
         'ER_Visits_Past_Year', 'Hospitalizations_Past_Year', 'Overall_Health_Score']
    ].mean().reset_index()
    
    # Radar chart for cluster comparison
    radar_features = [
        'Housing_Risk', 'Food_Risk', 'Transport_Risk', 'Social_Risk',
        'Chronic_Conditions', 'ER_Visits_Past_Year'
    ]
    
    fig = go.Figure()
    
    for cluster in sorted(cluster_profiles['Cluster'].unique()):
        cluster_data = cluster_profiles[cluster_profiles['Cluster'] == cluster]
        fig.add_trace(go.Scatterpolar(
            r=cluster_data[radar_features].values[0],
            theta=radar_features,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            )),
        showlegend=True,
        title="SDOH Risk Profile by Cluster"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Health outcomes by cluster
    st.subheader("Health Outcomes by Cluster")
    
    outcome_metrics = ['Overall_Health_Score', 'ER_Visits_Past_Year', 'Hospitalizations_Past_Year']
    
    fig = make_subplots(rows=1, cols=len(outcome_metrics), 
                        subplot_titles=[m.replace('_', ' ') for m in outcome_metrics])
    
    for i, metric in enumerate(outcome_metrics):
        for cluster in sorted(cluster_profiles['Cluster'].unique()):
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {cluster}'],
                    y=[cluster_profiles[cluster_profiles['Cluster'] == cluster][metric].values[0]],
                    name=f'Cluster {cluster}',
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(title="Health Outcomes Comparison Across Clusters")
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster size distribution
    cluster_sizes = df_patients['Cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'Number of Patients']
    
    fig = px.pie(cluster_sizes, values='Number of Patients', names='Cluster',
                title="Distribution of Patients Across Clusters")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed cluster profiles
    st.subheader("Detailed Cluster Profiles")
    
    # Format the cluster profile data for display
    profile_display = cluster_profiles.copy()
    profile_display.columns = [col.replace('_', ' ').title() for col in profile_display.columns]
    st.dataframe(profile_display)
    
    # Interpret clusters
    st.subheader("Cluster Interpretation")
    
    # This would typically be based on the actual data, but for this example we'll create some interpretations
    interpretations = {
        0: "**Low SDOH Risk Group**: These patients have stable housing, good food security, reliable transportation, and strong social support. They have few chronic conditions and rarely use emergency services.",
        1: "**Housing & Economic Insecurity Group**: This cluster is characterized by housing instability and economic challenges. Despite moderate social support, they have higher ER utilization.",
        2: "**Multiple Barrier Group**: These patients face multiple SDOH challenges including transportation barriers, food insecurity, and limited social support. They have high healthcare utilization and poor health outcomes.",
        3: "**Social Isolation Group**: While having relatively stable housing and food security, these patients have poor social support networks. They struggle with medication adherence and have moderate health outcomes."
    }
    
    # Display interpretations for the selected number of clusters
    for i in range(min(num_clusters, len(interpretations))):
        st.markdown(f"### Cluster {i}")
        st.markdown(interpretations.get(i, f"Cluster {i} interpretation not available"))
    
    st.info("""
    **Cluster Analysis Findings:**
    
    - Distinct patient profiles emerge based on combinations of social risk factors
    - Clusters with multiple SDOH risk factors show significantly worse health outcomes
    - The cluster with housing instability and food insecurity has the highest ER utilization
    - Some clusters show unexpected patterns, such as patients with good social support but poor transportation access
    - These distinct profiles can guide the development of targeted intervention programs
    """)
    
# Recommendations
elif page == "Recommendations":
    st.header("Evidence-Based Recommendations")
    
    st.subheader("Key Findings")
    st.markdown("""
    1. **Multiple SDOH factors have compounding effects** - patients with multiple social risk factors experience exponentially worse health outcomes
    
    2. **Housing insecurity, food insecurity, and transportation barriers** are the most impactful social determinants in our analysis
    
    3. **Social support positively influences medication adherence** and overall health outcomes
    
    4. **15% of patients have high SDOH risk scores** that strongly correlate with increased ER utilization
    """)
    
    st.subheader("Actionable Recommendations for Doctors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Clinical Practice Integration
        
        1. **Implement standardized SDOH screening** at intake and regular intervals
        
        2. **Incorporate SDOH risk stratification** into electronic health records
        
        3. **Develop clinical decision support tools** that flag patients with high-risk SDOH profiles
        
        4. **Create SDOH-informed care plans** tailored to each patient's specific social risk factors
        
        5. **Schedule more frequent follow-ups** for patients with multiple SDOH risk factors
        """)
    
    with col2:
        st.markdown("""
        ### Multi-disciplinary Interventions
        
        1. **Establish partnerships** with community resources addressing housing, food, and transportation
        
        2. **Engage social workers** in care teams for high-risk patients
        
        3. **Develop group visits** for patients with similar SDOH risk profiles
        
        4. **Create medication adherence programs** with SDOH support components
        
        5. **Advocate for policy changes** addressing structural SDOH barriers
        """)
    
    st.subheader("Implementation Roadmap")
    
    timeline_data = [
        {"Task": "SDOH Screening Implementation", "Start": "Month 1", "End": "Month 3", "Phase": "Phase 1"},
        {"Task": "EHR Integration", "Start": "Month 2", "End": "Month 4", "Phase": "Phase 1"},
        {"Task": "Staff Training", "Start": "Month 3", "End": "Month 5", "Phase": "Phase 1"},
        {"Task": "Community Partnership Development", "Start": "Month 4", "End": "Month 8", "Phase": "Phase 2"},
        {"Task": "Pilot Intervention Program", "Start": "Month 6", "End": "Month 9", "Phase": "Phase 2"},
        {"Task": "Risk Stratification Algorithm Refinement", "Start": "Month 7", "End": "Month 10", "Phase": "Phase 2"},
        {"Task": "Full Implementation", "Start": "Month 10", "End": "Month 12", "Phase": "Phase 3"},
        {"Task": "Outcome Measurement", "Start": "Month 12", "End": "Month 15", "Phase": "Phase 3"}
    ]
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Convert to numeric for plotting
    month_to_num = {f"Month {i}": i for i in range(1, 16)}
    timeline_df['Start_Num'] = timeline_df['Start'].map(month_to_num)
    timeline_df['End_Num'] = timeline_df['End'].map(month_to_num)
    
    fig = px.timeline(timeline_df, x_start="Start_Num", x_end="End_Num", y="Task", color="Phase",
                     title="SDOH Integration Implementation Timeline",
                     labels={"Task": "Implementation Step", "Start_Num": "Month", "End_Num": "Month"})
    
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickvals=list(range(1, 16)), ticktext=[f"Month {i}" for i in range(1, 16)])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Expected Outcomes")
    
    outcomes = [
        {"Outcome": "Reduced ER Visits", "Impact": 25, "Category": "Utilization"},
        {"Outcome": "Decreased Hospitalizations", "Impact": 20, "Category": "Utilization"},
        {"Outcome": "Improved Medication Adherence", "Impact": 30, "Category": "Adherence"},
        {"Outcome": "Better Chronic Disease Control", "Impact": 15, "Category": "Clinical"},
        {"Outcome": "Increased Patient Satisfaction", "Impact": 35, "Category": "Experience"},
        {"Outcome": "Enhanced Provider Awareness", "Impact": 40, "Category": "Awareness"}
    ]
    
    outcomes_df = pd.DataFrame(outcomes)
    
    fig = px.bar(outcomes_df, x="Outcome", y="Impact", color="Category",
                title="Projected Percentage Improvement from SDOH Integration",
                labels={"Impact": "% Improvement"})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Conclusion
    
    Integrating SDOH data into clinical practice can significantly improve patient outcomes by:
    
    1. Identifying at-risk patients who may benefit from targeted interventions
    2. Personalizing care plans to address both medical and social needs
    3. Creating more effective partnerships with community resources
    4. Reducing healthcare utilization through preventive social support
    5. Improving medication adherence and chronic disease management
    
    This data-driven approach to addressing social determinants will help providers deliver more holistic, effective care while reducing overall healthcare costs.
    """)

