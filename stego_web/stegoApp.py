import streamlit as st
from stegano import lsb
from PIL import Image
import io
import argparse
import CNet
from glob import glob
import base64
import shutil
import os

def get_image_down_link(img):
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}">Download stego</a>'
    return href

def myParseArgs(debug_bool):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-TEST_DIR',
        '--TEST_DIR',
        help='The path to load test_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )
    if debug_bool:
        args = parser
    else:
        args = parser.parse_args()
    return args

st.set_page_config("基于LSB的最低有效位隐写", layout='wide')
st.title("The Webapp of Steganography based on LSB and Steganalysis")
st.sidebar.title("Select the Mode of This Webapp")
option = st.sidebar.radio(
    'Embedding or Extract or Steganalysis?',
     ['Embedding','Extract','Steganalysis-jpg','Steganalysis-png'])

if option == "Embedding":
    col1, col2 = st.beta_columns(2)
    st.subheader("Embedding what you want~")
    raw_image = col1.file_uploader("upload your raw image：",['png','jpg','jpeg'])
    message = col2.text_input("Message","")
    if raw_image and message:
        origin_image = Image.open(raw_image)
        origin_image = origin_image.convert("RGB")
        if st.button("开始嵌入"):
            stego = lsb.hide(origin_image,message)
            st.write(f"大小：{stego.size}")
            st.image(stego,caption="Stego")
            st.markdown(get_image_down_link(stego), unsafe_allow_html=True)

elif option == "Extract":
    st.subheader("Extract what you have embeded!")
    stego_input = st.file_uploader("upload your stego：",['png','jpg','jpeg'])
    if st.button("解密！"):
        remessage = lsb.reveal(stego_input)
        st.info(remessage)
elif option == "Steganalysis-jpg":
    st.subheader("Steganalysis the Image That You Input!")
    input_filepath = st.file_uploader("upload your images：",['jpg'], accept_multiple_files=True)
    if input_filepath:
        for i in range(len(input_filepath)):
            # print(input_filepath[i].name)
            input_images = Image.open(input_filepath[i])
            input_images.save('./data/'+str(input_filepath[i].name))
    if st.button("Steganalysis！！！"):
        debug_bool = True
        TEST_DIR = './data'
        image_info = [x.split('\\')[-1].split(".jpg")[0] for x in glob('./data/*.jpg')]
        args = myParseArgs(debug_bool=debug_bool)
        if debug_bool:
            args.statePath = './model_params_jpg.pt'
            args.TEST_DIR = TEST_DIR
        message_ste = CNet.main(args)
        st.info(image_info)
        st.info(message_ste)
    if st.button("Clear Cache!"):
        shutil.rmtree("./data")
        os.mkdir("./data")
elif option == "Steganalysis-png":
    st.subheader("Steganalysis the Image That You Input!")
    input_filepath = st.file_uploader("upload your images：",['png'], accept_multiple_files=True)
    if input_filepath:
        for i in range(len(input_filepath)):
            # print(input_filepath[i].name)
            input_images = Image.open(input_filepath[i])
            input_images.save('./data/'+str(input_filepath[i].name))
    if st.button("Steganalysis！！！"):
        debug_bool = True
        TEST_DIR = './data'
        image_info = [x.split('\\')[-1].split(".png")[0] for x in glob('./data/*.png')]
        args = myParseArgs(debug_bool=debug_bool)
        if debug_bool:
            args.statePath = './model_params_png.pt'
            args.TEST_DIR = TEST_DIR
        message_ste = CNet.main(args)
        st.info(image_info)
        st.info(message_ste)
    if st.button("Clear Cache!"):
        shutil.rmtree("./data")
        os.mkdir("./data")
