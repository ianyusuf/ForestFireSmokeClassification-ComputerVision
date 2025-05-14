import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def run():

    #Title
    st.title('🌳 Forest Fire & Smoke Detection')

    st.write('Forest fires are one of the environmental disasters that have a major impact on life. This disaster can cause ecosystem damage, air pollution, and threaten human safety. One solution that can be used for early mitigation is to utilize Computer Vision technology to automatically detect the presence of fire and smoke through images.')

    #Image
    image = Image.open('forest.png')
    st.image(image)
    
    # Dataset Distribution
    st.write('## 📊 Exploratory Data Analysis')
    st.markdown('---')

    data = {
        "Class": [
            "Fire", "Non-Fire", "Smoke"
        ],
        "Number of Images": [1000, 1000, 1000]
    }

    df = pd.DataFrame(data)

    # Dataset from Google Drive
    images_by_class = {
        "fire": [
            "fire/fire1.jpg",
            "fire/fire2.png",
            "fire/fire3.jpeg",
            "fire/fire4.png",
            "fire/fire5.png"
        ],
        "non-fire": [
            "non-fire/non-fire1.png",
            "non-fire/non-fire2.png",
            "non-fire/non-fire3.png",
            "non-fire/non-fire4.jpg",
            "non-fire/non-fire5.png"
        ],
        "smoke": [
            "smoke/smoke1.png",
            "smoke/smoke2.jpg",
            "smoke/smoke3.jpg",
            "smoke/smoke4.png",
            "smoke/smoke5.jpg"
        ]
    }

    st.write("## Forest Fire Image Samples")

    # Display in grid
    for cls, image_paths in images_by_class.items():
        st.subheader(cls.capitalize())
        cols = st.columns(len(image_paths))
        for i, path in enumerate(image_paths):
            img = Image.open(path)
            with cols[i]:
                st.image(img, use_container_width=True)

    st.write(" ")

    # Insight text
    st.markdown("""
    **Fire Image**: Defined by the dominant colors of bright red and orange, often accompanied by forms of burning flames or spreading flames. The characteristic of this class is the high intensity of light and a clearly visible burning pattern defining the presence of an active fire.

    **Smoke Image**: Defined by a whitish gray color that covers the image area. Its shape obscures the objects behind it. The uniqueness of this class lies in its opaque texture and transparent gradation, which indicates the presence of fire smoke.

    **Non-fire Image**: Defined as an image without any indication of fire or smoke, usually showing a calm forest scene with a blue sky. There are no flaming shapes or smoke haze. Its visual calmness clearly distinguishes it from the other two classes.
    """)

    st.markdown('---')

    # Pie Chart
    st.subheader("Class Distribution in Training and Test Set")

    fig, ax = plt.subplots()
    ax.pie(df["Number of Images"],
        labels=df["Class"],
        autopct='%1.1f%%',
        startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    st.pyplot(fig)

    # Insight text
    st.markdown("""
    The figure above shows the data distribution in the training set for three classes: fire, smoke, and non-fire.

    Each class has almost the same proportion, which is around 33.3%.

    This shows that the dataset is balanced, so the model does not need to be given special treatment such as oversampling or undersampling to overcome class imbalance (imbalanced data).
    """)

if __name__ == '__main__':
    run()