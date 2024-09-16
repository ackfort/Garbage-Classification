import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import io

# โหลดโมเดลที่ฝึกมาแล้ว
model = load_model('Image_classify.keras')

# หมวดหมู่ที่ใช้ในการจำแนก
data_cat = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# กำหนดขนาดภาพ
img_height = 300
img_width = 300

# สร้าง Navbar ด้วย sidebar
st.sidebar.title('Navigation')
nav = st.sidebar.radio("Go to", ["Home", "About Model", "History"])

# เก็บประวัติการ classify
if 'history' not in st.session_state:
    st.session_state.history = []

# กำหนดเนื้อหาตาม Navbar
if nav == "Home":
    st.title('Image Classification Model')
    st.write("Upload an image to classify it into different categories.")

    # รับภาพจากผู้ใช้
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # แสดงภาพที่ผู้ใช้อัพโหลด
        image_load = Image.open(uploaded_file)
        
        # ตรวจสอบว่าภาพเป็น RGBA หรือไม่ ถ้าใช่แปลงเป็น RGB
        if image_load.mode != "RGB":
            image_load = image_load.convert("RGB")
        
        st.image(image_load, caption='Uploaded Image', use_column_width=True)
        
        # ปรับขนาดและเตรียมภาพสำหรับการทำนาย
        image_load = image_load.resize((img_height, img_width))
        img_arr = tf.keras.preprocessing.image.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)  # เพิ่มมิติสำหรับ batch

        # เพิ่มปุ่มสำหรับทำนาย
        if st.button('Classify'):
            # ทำนายผล
            predict = model.predict(img_bat)

            # คำนวณค่าความน่าจะเป็น
            score = tf.nn.softmax(predict[0])

            # แสดงผลการจำแนกภาพและค่าความแม่นยำ
            st.subheader('Prediction Result:')
            result = data_cat[np.argmax(score)]
            accuracy = np.max(score) * 100
            st.write(f'**Veg/Fruit in image is:** {result}')
            st.write(f'**With accuracy of:** {accuracy:.2f}%')

            # เก็บรูปและประวัติการทำนาย
            buffer = io.BytesIO()
            image_load.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            st.session_state.history.append({
                "image": image_data,
                "image_name": uploaded_file.name,
                "result": result,
                "accuracy": accuracy
            })

            # แสดงค่าความน่าจะเป็นของทุกหมวดหมู่
            st.subheader("Probability for each category:")
            for i, cat in enumerate(data_cat):
                # ใช้ progress bar เพื่อแสดงค่าความน่าจะเป็น
                st.write(f"{cat}: {score[i] * 100:.2f}%")
                st.progress(float(score[i]))

elif nav == "About Model":
    st.title("About the Model")
    st.write("""
        This image classification model is trained to classify various types of trash. 
        It uses a deep learning approach and TensorFlow for the prediction.
        The dataset includes categories like cardboard, glass, metal, paper, plastic, and trash.
    """)

elif nav == "History":
    st.title("Classification History")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history):
            st.write(f"{idx + 1}. **Image**: {entry['image_name']}, **Prediction**: {entry['result']}, **Accuracy**: {entry['accuracy']:.2f}%")
            st.image(entry['image'], caption=f"Classified as {entry['result']} with {entry['accuracy']:.2f}% accuracy", use_column_width=True)
    else:
        st.write("No history available.")
