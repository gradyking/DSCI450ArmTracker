# import streamlit as st
# import cv2
# import numpy as np
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
# import mediapipe as mp
# from scipy import stats
# import matplotlib.pyplot as plt

# st.title("Arm Height Tracker with Distribution Demo")

# # -----------------------
# # Mediapipe setup
# # -----------------------
# mp_pose = mp.solutions.pose
# POSE = mp_pose.Pose(static_image_mode=False,
#                     min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5)

# # -----------------------
# # Video Transformer
# # -----------------------
# class ArmHeightTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.height_data = []

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         h, w, _ = img.shape
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = POSE.process(img_rgb)

#         heights = {}
#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             try:
#                 # pixel coordinates
#                 def y_px(lm_obj): return int(lm_obj.y * h)
#                 left_wrist = y_px(lm[mp_pose.PoseLandmark.LEFT_WRIST])
#                 right_wrist = y_px(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
#                 left_elbow = y_px(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
#                 right_elbow = y_px(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
#                 left_shoulder = y_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
#                 right_shoulder = y_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
#                 avg_shoulder = (left_shoulder + right_shoulder) // 2

#                 heights = {
#                     "left_wrist": left_wrist,
#                     "right_wrist": right_wrist,
#                     "left_elbow": left_elbow,
#                     "right_elbow": right_elbow,
#                     "avg_shoulder": avg_shoulder
#                 }

#                 # Draw landmarks for preview
#                 for coord in [left_wrist, right_wrist, left_elbow, right_elbow, avg_shoulder]:
#                     cv2.circle(img, (w//2, coord), 5, (0, 255, 0), -1)

#                 # record mean wrist/elbow height relative to shoulders
#                 mean_arm_height = np.mean([
#                     left_wrist - avg_shoulder,
#                     right_wrist - avg_shoulder,
#                     left_elbow - avg_shoulder,
#                     right_elbow - avg_shoulder
#                 ])
#                 self.height_data.append(mean_arm_height)
#                 if len(self.height_data) > 200:
#                     self.height_data.pop(0)

#             except Exception as e:
#                 print("Landmark error:", e)

#         # return frame for preview
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # -----------------------
# # Streamlit WebRTC component
# # -----------------------
# ctx = webrtc_streamer(
#     key="arm-heights",
#     mode=WebRtcMode.SENDRECV,
#     video_transformer_factory=ArmHeightTransformer,
#     media_stream_constraints={"video": True, "audio": False},
#     async_transform=True,
# )

# # -----------------------
# # Display distribution plot
# # -----------------------
# if ctx.video_transformer:
#     transformer = ctx.video_transformer
#     if len(transformer.height_data) > 10:
#         data = np.array(transformer.height_data)
#         fig, ax = plt.subplots(figsize=(6,3))
#         ax.hist(data, bins=15, density=True, alpha=0.7, color='skyblue')

#         # overlay normal fit
#         mu, sigma = data.mean(), data.std()
#         x = np.linspace(data.min(), data.max(), 100)
#         ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label="Normal fit")
#         ax.set_title("Distribution of arm heights (wrist/elbow relative to shoulder)")
#         ax.set_xlabel("Relative height (pixels)")
#         ax.set_ylabel("Probability density")
#         ax.legend()
#         st.pyplot(fig)

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import mediapipe as mp
from scipy import stats
import matplotlib.pyplot as plt

st.title("Arm Tracker with XY Coordinates Demo")

# -----------------------
# Mediapipe setup
# -----------------------
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# -----------------------
# Video Transformer
# -----------------------
class ArmXYTransformer(VideoTransformerBase):
    def __init__(self):
        self.xy_data = []  # store (x, y) relative to avg shoulder

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = POSE.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                # convert normalized coordinates to pixels
                def px_coords(lm_obj):
                    return int(lm_obj.x * w), int(lm_obj.y * h)

                left_wrist = px_coords(lm[mp_pose.PoseLandmark.LEFT_WRIST])
                right_wrist = px_coords(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
                left_elbow = px_coords(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
                right_elbow = px_coords(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
                left_shoulder = px_coords(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
                right_shoulder = px_coords(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])

                avg_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                                (left_shoulder[1] + right_shoulder[1]) // 2)

                # Draw landmarks
                for coord in [left_wrist, right_wrist, left_elbow, right_elbow, avg_shoulder]:
                    cv2.circle(img, coord, 5, (0, 255, 0), -1)

                # Store XY relative to avg shoulder
                for coord in [left_wrist, right_wrist, left_elbow, right_elbow]:
                    rel_x = coord[0] - avg_shoulder[0]
                    rel_y = coord[1] - avg_shoulder[1]
                    self.xy_data.append((rel_x, rel_y))
                    if len(self.xy_data) > 200:
                        self.xy_data.pop(0)

            except Exception as e:
                print("Landmark error:", e)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------
# Streamlit WebRTC component
# -----------------------
ctx = webrtc_streamer(
    key="arm-xy",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=ArmXYTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# -----------------------
# Display XY scatter plot
# -----------------------
if ctx.video_transformer:
    transformer = ctx.video_transformer
    if len(transformer.xy_data) > 10:
        data = np.array(transformer.xy_data)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data[:,0], -data[:,1], alpha=0.6, color='skyblue')  # invert Y for visualization
        ax.set_title("XY coordinates of wrist/elbow relative to shoulders")
        ax.set_xlabel("X relative to shoulder (pixels)")
        ax.set_ylabel("Y relative to shoulder (pixels)")
        ax.grid(True)
        st.pyplot(fig)
