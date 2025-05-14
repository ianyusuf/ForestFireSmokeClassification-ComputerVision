import streamlit as st
import eda
import prediction

sc = st.sidebar.image('fire-res.png')

# Sidebar Navigation
st.sidebar.markdown("### **Fire Res National Forestry Protector**")
st.sidebar.markdown("Protect the forests, Protect the planet")

# Radio button
nav = st.sidebar.radio("Navigation", options=[
    "📊 Exploratory Data Analysis",
    "🔍 Model Prediction"
])

# Navigation function
if nav == "📊 Exploratory Data Analysis":
    eda.run()
elif nav == "🔍 Model Prediction":
    prediction.run()

# Sidebar Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 **About the Model**")

# Accuracy
st.sidebar.markdown("🎯 **Model Accuracy:**")
model_accuracy = 0.8
st.sidebar.progress(model_accuracy)
st.sidebar.write(f"{model_accuracy * 100:.2f}%")

# Accuracy Information
st.sidebar.markdown(
    "ℹ️ This model successfully classifies 80% of the images correctly, showing quite good performance in distinguishing between fire, non-fire, and smoke classes."
)

    