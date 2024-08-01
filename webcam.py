# import cv2
# import torch
# import os

# # Set model path to point to the correct path for best.pt
# model_path = os.path.join('exp7', 'weights', 'best.pt')

# # Load the pre-trained YOLOv5 model with the fixed checkpoint
# model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# # Set the confidence threshold
# model.conf = 0.35

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform inference
#     results = model(frame)

#     # Parse results
#     for pred in results.pred[0]:
#         xmin, ymin, xmax, ymax, conf, cls = pred
#         if conf >= 0.35:  # Apply confidence threshold
#             label = results.names[int(cls)]
#             cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
#             cv2.putText(frame, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#     # Display the frame
#     cv2.imshow('Webcam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # import cv2
# # import torch
# # import os
# # import numpy as np

# # # Set model path to point to the correct path for best.pt
# # model_path = os.path.join('exp7', 'weights', 'best.pt')

# # # Load the pre-trained YOLOv5 model with the fixed checkpoint
# # model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# # # Set the confidence threshold
# # model.conf = 0.35

# # # Variables to track gun detection state and timestamps
# # gun_detected = False
# # gun_detected_start_time = 0
# # last_notification_time = 0

# # # Time thresholds (in seconds)
# # detection_threshold = 3
# # notification_cooldown = 30

# # def detect_objects(frame):
# #     """
# #     Perform object detection on the provided frame using YOLOv5.
    
# #     Args:
# #         frame (numpy.ndarray): The frame to process.

# #     Returns:
# #         list: List of detection results. Each result is a tuple (xmin, ymin, xmax, ymax, confidence, label).
# #     """
# #     # Perform inference
# #     results = model(frame)

# #     detections = []
# #     # Parse results
# #     for pred in results.pred[0]:
# #         xmin, ymin, xmax, ymax, conf, cls = pred
# #         if conf >= 0.35:  # Apply confidence threshold
# #             label = results.names[int(cls)]
# #             detections.append((int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin), f'{label} {conf:.2f}'))
    
# #     return detections
