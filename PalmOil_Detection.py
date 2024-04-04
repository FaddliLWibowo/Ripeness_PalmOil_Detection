import streamlit as st
import cv2
import numpy as np
import av
import torch
import tempfile
from PIL import Image

@st.cache_resource
def load_model():
    # model = torch.hub.load('ultralytics/yolov5','custom',path="weights/best.pt")
    model = torch.hub.load('yolov5', 'custom', path="weights/YOLOv5-CBAM-TR-BiFPN.pt", source='local') # local repo
    return model

demo_img = "palm-oil.png"
demo_video = "PalmOil_Video.mp4"

st.title('Ripeness Palm Oil Detection')
st.sidebar.title('App Mode')


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App','Run on Image','Run on Video','Run on WebCam'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Ripeness Palm Oil Detection App created with custom trained models using Improved YoloV5</h5>",unsafe_allow_html=True)
    
    st.markdown("- <h5>Select the App Mode in the SideBar</h5>",unsafe_allow_html=True)

    st.markdown("- <h5>Upload the Image and Detect the Palm Oil in Images</h5>",unsafe_allow_html=True)

    st.markdown("- <h5>Upload the Video and Detect the Palm Oil in Videos</h5>",unsafe_allow_html=True)

    st.markdown("- <h5>Live Detection</h5>",unsafe_allow_html=True)

    st.markdown("- <h5>Click Start to start the camera</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>",unsafe_allow_html=True)
    
    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5
## 🔗 Links
[![twitter](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FaddliLWibowo)
""")
    

if app_mode == 'Run on Image':
    st.subheader("Detected Palm Oil:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)
    
    # predict the image
    model = load_model()
    results = model(image)
    length = len(results.xyxy[0])
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
    st.subheader("Output Image")
    st.image(output,use_column_width=True)
    
if app_mode == 'Run on Video':
    st.subheader("Detected Palm Oil:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    #Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)
    
    # predict the video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model(frame)
        length = len(results.xyxy[0])
        output = np.squeeze(results.render())
        text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
        stframe.image(output)
        
if app_mode == 'Run on WebCam':
    st.subheader("Detected Palm Oil:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    
    st.subheader("Output")
    stframe = st.empty()
    
    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")
    
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture(-1)
    if(run):
        while(True):
            if(stop):
                break
            ret,frame = cam.read()
            if not ret:
                st.write("Error: Failed to read frame from webcam.")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            output = np.squeeze(results.render())
            text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
            stframe.image(output)
