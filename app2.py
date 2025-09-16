# import cv2
# import numpy as np
# import Sports2D
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# # # Initialize pose estimator
# # pose_estimator = Sports2D.Sports2D()

# # class PoseTransformer(VideoTransformerBase):
# #     def transform(self, frame):
# #         # Convert to numpy array
# #         img = frame.to_ndarray(format="bgr24")
        
# #         # Convert to RGB
# #         rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
# #         # Pose estimation
# #         pose_data = Sports2D.process(rgb_frame)
# #         keypoints = pose_data.get('keypoints', [])

# #         # Draw keypoints (optional: wrist, elbow, shoulder only)
# #         if keypoints:
# #             # Example indices: 5-left shoulder, 6-right shoulder, 7-left elbow, 8-right elbow, 9-left wrist, 10-right wrist
# #             joint_indices = [5, 6, 7, 8, 9, 10]
# #             for i in joint_indices:
# #                 if i < len(keypoints):
# #                     x, y = keypoints[i]
# #                     cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

# #         return img

# st.title("Live Pose Estimation with Sports2D + Streamlit WebRTC")

# webrtc_streamer(key="pose-estimation")

# # webrtc_streamer(
# #     key="pose-estimation",
# #     video_transformer_factory=PoseTransformer,
# #     media_stream_constraints={"video": True, "audio": False},
# # )

"""Video transforms with OpenCV"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

_type = st.radio("Select transform type", ("noop", "cartoon", "edges", "rotate"))


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    if _type == "noop":
        pass
    elif _type == "cartoon":
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(img))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)
    elif _type == "edges":
        # perform edge detection
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    elif _type == "rotate":
        # rotate image
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    "This demo is based on "
    "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
    "Many thanks to the project."
)