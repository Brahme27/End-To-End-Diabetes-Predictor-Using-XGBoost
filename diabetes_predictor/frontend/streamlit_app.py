"""
Diabetes Risk Prediction - Streamlit Frontend
Based on Pima Indians Diabetes Dataset
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def plot_shap_waterfall(explanation):
    """Create a clean horizontal bar chart showing feature impacts"""
    feature_names = explanation['feature_names']
    shap_values = explanation['shap_values']
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': shap_values
    })
    
    # Sort by absolute impact
    df['Abs_Impact'] = df['Impact'].abs()
    df = df.sort_values('Abs_Impact', ascending=False).head(10)  # Top 10 features
    df = df.sort_values('Impact', ascending=True)  # Sort for better visualization
    
    # Create color based on impact direction
    colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in df['Impact']]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=df['Impact'],
        y=df['Feature'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:+.3f}" for val in df['Impact']],
        textposition='outside',
    ))
    
    fig.update_layout(
        title={
            'text': "Top Factors Affecting Your Risk",
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        xaxis_title="Impact on Risk Score",
        yaxis_title="",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='#E8E8E8',
            zerolinecolor='#333',
            zerolinewidth=2
        ),
        margin=dict(l=150, r=50, t=80, b=50)
    )
    
    return fig

def show_impact_summary(explanation):
    """Show top 3 impacting features in a clean summary"""
    feature_names = explanation['feature_names']
    shap_values = explanation['shap_values']
    
    # Get top 3 by absolute impact
    impacts = list(zip(feature_names, shap_values))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    st.markdown("### 🎯 Key Factors")
    
    for i, (feature, impact) in enumerate(impacts[:3], 1):
        impact_pct = impact * 100
        if impact > 0:
            st.error(f"**{i}. {feature}**: Increases risk by **{abs(impact_pct):.1f}%** ⬆️")
        else:
            st.success(f"**{i}. {feature}**: Decreases risk by **{abs(impact_pct):.1f}%** ⬇️")

# Header
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Based on Pima Indians Diabetes Dataset")
st.markdown("---")

# Sidebar for input
st.sidebar.header("📋 Patient Information")
st.sidebar.markdown("Enter patient details below:")

# Input fields with proper organization
with st.sidebar:
    st.subheader("Demographics")
    age = st.slider("Age (years)", 21, 100, 33, help="Minimum age is 21 in this dataset")
    pregnancies = st.slider("Number of Pregnancies", 0, 17, 3, help="Number of times pregnant")
    
    st.subheader("Physical Measurements")
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 32.0, 0.1, help="Body Mass Index")
    blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", 0, 140, 72, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20, help="Triceps skin fold thickness")
    
    st.subheader("Lab Values")
    glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120, help="Plasma glucose concentration after 2-hour oral glucose tolerance test")
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80, help="2-Hour serum insulin (0 if not measured)")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.001, help="Genetic influence score")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Patient Summary")
    
    # Display patient info in a clean format
    summary_data = {
        "Feature": ["Age", "Pregnancies", "BMI", "Blood Pressure", "Glucose", "Insulin", "Skin Thickness", "Pedigree Function"],
        "Value": [
            f"{age} years",
            f"{pregnancies}",
            f"{bmi:.1f} kg/m²",
            f"{blood_pressure} mmHg",
            f"{glucose} mg/dL",
            f"{insulin} mu U/ml" if insulin > 0 else "Not measured",
            f"{skin_thickness} mm" if skin_thickness > 0 else "Not measured",
            f"{diabetes_pedigree_function:.3f}"
        ]
    }
    st.table(pd.DataFrame(summary_data))

with col2:
    st.subheader("🎯 Risk Factors")
    risk_factors = []
    if bmi >= 30:
        risk_factors.append("⚠️ High BMI (Obese)")
    if blood_pressure >= 90:
        risk_factors.append("⚠️ High Blood Pressure")
    if glucose >= 140:
        risk_factors.append("⚠️ High Glucose (Diabetic range)")
    elif glucose >= 126:
        risk_factors.append("⚠️ Elevated Glucose")
    if pregnancies >= 6:
        risk_factors.append("⚠️ High Pregnancies")
    if diabetes_pedigree_function > 0.8:
        risk_factors.append("⚠️ Strong Family History")
    if age >= 45:
        risk_factors.append("⚠️ Age Factor")
    
    if risk_factors:
        for rf in risk_factors:
            st.warning(rf)
    else:
        st.success("✅ No major risk factors")

st.markdown("---")

# Prediction button
if st.button("🔍 Predict Diabetes Risk", type="primary", width='stretch'):
    # Prepare data for API
    input_data = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree_function": diabetes_pedigree_function,
        "age": age
    }
    
    with st.spinner("🔄 Analyzing patient data..."):
        try:
            # Call backend API
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=input_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Prediction Complete!")
                st.markdown("---")
                
                # Display results in columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(
                        label="Diabetes Risk",
                        value=result["prediction"],
                        delta="High Risk" if result["prediction"] == "Positive" else "Low Risk"
                    )
                
                with res_col2:
                    confidence = result["probability"] * 100
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%"
                    )
                
                with res_col3:
                    risk_level = "High" if result["probability"] > 0.7 else "Moderate" if result["probability"] > 0.4 else "Low"
                    st.metric(
                        label="Risk Level",
                        value=risk_level
                    )
                
                # Visualization
                st.subheader("📈 Risk Assessment Visualization")
                
                # Gauge chart for risk probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result["probability"] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': 'lightgreen'},
                            {'range': [40, 70], 'color': 'yellow'},
                            {'range': [70, 100], 'color': 'lightcoral'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')
                
                # SHAP Explanation
                if result.get("explanation"):
                    st.markdown("---")
                    
                    # Show key impacting factors
                    show_impact_summary(result["explanation"])
                    
                    st.markdown("")  # Spacing
                    
                    # Show detailed chart
                    with st.expander("📊 View Detailed Impact Analysis", expanded=True):
                        st.info("🔴 Red bars increase risk  |  🟢 Teal bars decrease risk")
                        shap_fig = plot_shap_waterfall(result["explanation"])
                        st.plotly_chart(shap_fig, width='stretch')
                
                st.markdown("---")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if result["prediction"] == "Positive":
                    st.error("⚠️ **High Risk Detected** - Immediate medical consultation recommended")
                    st.markdown("""
                    **Recommended Actions:**
                    - 🏥 Schedule an appointment with your healthcare provider
                    - 📊 Get a comprehensive diabetes screening test
                    - 🥗 Consider dietary modifications
                    - 🏃 Increase physical activity
                    - 💊 Discuss preventive medications with your doctor
                    """)
                else:
                    st.success("✅ **Low Risk** - Maintain healthy lifestyle")
                    st.markdown("""
                    **Preventive Measures:**
                    - 🥗 Maintain a balanced diet
                    - 🏃 Regular exercise (150 min/week)
                    - ⚖️ Monitor your weight
                    - 📅 Annual health checkups
                    - 🚭 Avoid smoking and excessive alcohol
                    """)
                
                # Show model details
                with st.expander("🔬 View Model Details"):
                    st.json({
                        "Model Version": result.get("model_version", "v1.0"),
                        "Prediction": result["prediction"],
                        "Probability": f"{result['probability']:.4f}",
                        "Timestamp": result.get("timestamp", "N/A")
                    })
                    
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Please ensure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>🏥 <b>Diabetes Risk Prediction System</b> | Built with Streamlit & FastAPI</p>
        <p><i>This tool is for educational purposes. Always consult healthcare professionals for medical advice.</i></p>
    </div>
""", unsafe_allow_html=True)

# Batch prediction feature in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    if st.sidebar.button("Process Batch"):
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} records")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head())
            
            st.info("🔄 Batch prediction feature coming soon!")
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
