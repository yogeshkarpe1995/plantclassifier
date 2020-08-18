from fastai.vision import *
import os
import time
import streamlit as strm
import cv2
import imutils

# App Styles
css_html = """
    <style>

        .main{
            background-color: #000000cf;
            color : aliceblue;
        }
        .btn{
            background-color: aliceblue;
        }

    </style>
"""


def file_selector(folder_path='yolo', file_type=(), sbar=True, key=""):
    filenames = os.listdir(folder_path)
    filenames = [fname for fname in filenames if fname.endswith(file_type)]

    if not sbar:
        selected_filename = strm.selectbox('', filenames)
    else:
        selected_filename = strm.sidebar.selectbox('', filenames, key=key)

    return os.path.join(folder_path, selected_filename)


def show_loading(st):
    import base64
    file_ = open("loading.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img style="margin-left:250px; margin-top:150px" src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )


def write_res(res_img, class_label):
    # res_img = cv2.imread(img_path)
    x = 18 * len(class_label)
    cv2.rectangle(res_img, (0, 0), (x, 40), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(res_img, (0, 0), (x, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(res_img, class_label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)
    return res_img


predictor = load_learner('fastai_model', 'farmizo_plant_classifier.pkl')
strm.markdown(css_html, unsafe_allow_html=True)
strm.markdown("<h1 style='text-align:center;'>Farmizo Plant Classifier App</h1>", unsafe_allow_html=True)
strm.sidebar.markdown('## **Select input image**')
img_path = file_selector('test_data', file_type=('.jpg', '.jpeg', '.png'), sbar=True)
# strm.markdown("## **Result**")
strm.markdown('<hr>', unsafe_allow_html=True)

img_place = strm.empty()
if strm.sidebar.button('Predict'):
    show_loading(img_place)
    start_time = time.time()
    pred_class, pred_idx, outputs = predictor.predict(open_image(img_path))
    prediction_class = predictor.data.classes[try_int(pred_class)]
    end_time = time.time()
    img_place.markdown(
        '#### *' + f'Predicted Class: {prediction_class} .......Elapsed time: {end_time - start_time:.3f}' + '*')

    strm.markdown('<br>', unsafe_allow_html=True)
    res_img = imutils.resize(cv2.imread(img_path), width=640, height=480)
    strm.image(write_res(res_img, prediction_class), channels='BGR')


if strm.sidebar.button("Data Info"):
    strm.markdown("<h1 style='text-align:center;'>Categories</h1>", unsafe_allow_html=True)
    image_paths = os.listdir("test_data")
    strm.write(predictor.data.classes)
    strm.markdown("<hr>", unsafe_allow_html=True)
    strm.markdown("<h1 style='text-align:center;'>Test Images</h1>", unsafe_allow_html=True)
    strm.markdown("<br>", unsafe_allow_html=True)

    image_lst = []
    for pths in image_paths:
        image_lst.append(os.path.join("test_data", pths))
    strm.image(image_lst, width=100)
