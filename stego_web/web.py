import cv2
import imutils
import numpy as np
import streamlit as st
from PIL import Image

from bwm import watermark

st.set_page_config("基于LSB的最低有效位隐写", layout='wide')
st.title("基于LSB的最低有效位隐写")
type_ = st.sidebar.radio("工具列表：", ["嵌入秘密信息", "提取秘密信息"])

if type_ == '嵌入秘密信息':
    st.subheader("嵌入秘密信息")
    col1, col2 = st.beta_columns(2)
    out1 = col1.file_uploader("上传原始图片:", ['png', 'jpg', 'jpeg'])
    if out1:
        origin_img = Image.open(out1)
        origin_img.save("data/origin.png")
        col1.write(f"大小: {origin_img.size}")
        col1.image(origin_img, caption='原图')

    # out2 = col2.file_uploader("上传水印图片:", ['png', 'jpg', 'jpeg'])
    out2 = col2.text_input('秘密信息', '')
    # title = st.text_input('秘密信息', '')
    if out2:
        watermark_img = Image.open(out2)
        col2.write(f"大小: {watermark_img.size}")
        watermark_img.save("data/watermark.png")
        col2.image(watermark_img, caption='水印')
    
    # TODO: just list here, have not implemented yet
    expander = st.beta_expander("高级选项")
    bs=expander.number_input("分块大小:", min_value=1, step=1, help="设定分块大小,因为限定长宽相同,所以只需要传一个整数就行了,对于大图可以使用更大的数,如8,更大的形状使得对原图影响更小,而且运算时间减少,但对鲁棒性没有提高,注意太大会使得水印信息超过图片的承载能力")
    dwt_deep = expander.number_input("小波变换次数:", min_value=1, step=1, help="设定小波变换的次数,次数增加会提高鲁棒性,但会减少图片承载水印的能力,通常取1,2,3")

    if out1 and out2:
        if st.button("开始嵌入"):
            bwm1 = watermark(4399,2333,36,20)
            bwm1.read_ori_img("data/origin.png")
            bwm1.read_wm("data/watermark.png")
            watermarked_img = bwm1.embed('data/watermarked.png')
            watermarked_img = imutils.opencv2matplotlib(watermarked_img.astype(np.uint8))

            st.image(watermarked_img, caption='嵌水印后图片', clamp=True)
            if st.button('Save'):
                st.stop()
else:
    st.subheader("提取秘密信息")


    out = st.file_uploader("上传带水印图片:", ['png', 'jpg', 'jpeg'])
    if out:
        embedding_img = Image.open(out)
        embedding_img.save("data/watermarked.png")
        st.image(embedding_img, caption='水印')

    st.write("输入水印大小:")
    col3, col4 = st.beta_columns(2)
    width = col3.number_input("宽:", min_value=1, step=1)
    height = col4.number_input("高:", min_value=1, step=1)
    if out:
        if st.button("提取水印"):
            embedding_img = np.asarray(embedding_img)[:, :, :3]
            bwm1 = watermark(4399,2333,36,20, wm_shape=(width, height))
            watermark_img = bwm1.extract('data/watermarked.png', "data/extract_wm.png")
            st.write(watermark_img.shape)
            st.image(watermark_img, caption='提取的水印', clamp=True)
