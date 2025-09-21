"""
YOLO TFLite Real-time Object Detection with Spatial Relationships

HOW TO RUN:
1. Make sure you have the required packages:
   pip install opencv-python tensorflow numpy

2. Place your TFLite model file as 'best_float32.tflite' in the same folder

3. Run the script:
   python run_pytorch_inference.py

4. The script will:
   - Open your webcam
   - Show live video feed
   - Process every 30 frames (~1 second) for object detection
   - Display bounding boxes, object names, and spatial relationships
   - Show "PROCESSING..." when analyzing a frame
   - Display results on the last processed frame

5. Press 'q' to quit

FEATURES:
- Detects objects with confidence scores
- Shows spatial relationships (e.g., "glass on table")
- Displays object names and bounding boxes
- Processes frames periodically for better accuracy
- Console output shows detailed detection info

REQUIREMENTS:
- Webcam (camera index 0)
- TFLite model file: best_float32.tflite
- Python packages: opencv-python, tensorflow, numpy
"""

import cv2
import numpy as np
import tensorflow as tf
import itertools
import time

def _letterbox_resize(image: np.ndarray, target_size: tuple) -> tuple:
    """Resize image with unchanged aspect ratio using padding (letterbox) to target_size (w, h)."""
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, (pad_w, pad_h)

def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert xywh format to xyxy format."""
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return xyxy

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Simple NMS returning kept indices."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return kept

# ==========================
# 1️⃣ Load TFLite Model
# ==========================
MODEL_PATH = "C:/Users/ASUS/OneDrive/Desktop/od/best_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

print(f"Model input shape: {input_details[0]['shape']}")
print(f"Model output shape: {output_details[0]['shape']}")

# ==========================
# 2️⃣ Open Webcam
# ==========================
cap = cv2.VideoCapture(0)  # 0 = default camera
if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam")

# Class names (12 classes based on your model output)
CLASS_NAMES = [
    "bed", "table", "pillow", "glass", "chair", "sofa",
    "book", "lamp", "tv", "cup", "bottle", "unknown"
]

print("✅ Webcam started. Press 'q' to quit.")

# ==========================
# 3️⃣ Frame Capture and Processing
# ==========================
frame_count = 0
process_every_n = 30  # Process 1 in every 30 frames (every ~1 second at 30fps)
last_processed_frame = None
processing = False

# FPS calculation
fps_counter = 0
fps_start_time = time.time()
fps = 0

# ==========================
# 4️⃣ Live Detection Loop
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Show live feed
    display_frame = frame.copy()
    
    # Add status text
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if processing:
        cv2.putText(display_frame, "PROCESSING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif last_processed_frame is not None:
        # Show last processed results
        display_frame = last_processed_frame.copy()
        cv2.putText(display_frame, f"Last Processed: Frame {frame_count - process_every_n}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Live Object Detection", display_frame)
    
    # Process frame every N frames
    if frame_count % process_every_n == 0 and not processing:
        processing = True
        print(f"\n🔍 Processing frame {frame_count}...")
        
        # Process the frame in a separate thread or just process it
        try:
            # Preprocess frame using letterbox resize (crucial for YOLO)
            orig_h, orig_w = frame.shape[:2]
            resized, scale, (pad_w, pad_h) = _letterbox_resize(frame, (input_width, input_height))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            input_data = np.expand_dims(img_rgb, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = output_data[0]  # Remove batch dimension
            
            # Parse YOLO output format
            num_classes = pred.shape[0] - 4  # 16 - 4 = 12 classes
            boxes_xywh = pred[0:4, :].T  # (8400, 4) in xywh format
            class_scores = pred[4:, :].T  # (8400, 12) class probabilities
            
            class_ids = class_scores.argmax(axis=1)
            class_conf = class_scores.max(axis=1)
            
            print(f"Debug - Max confidence: {class_conf.max():.3f}, Detections above 0.25: {(class_conf >= 0.25).sum()}")
            
            # Filter by confidence threshold
            conf_threshold = 0.1
            mask = class_conf >= conf_threshold
            boxes_xywh = boxes_xywh[mask]
            class_ids = class_ids[mask]
            class_conf = class_conf[mask]
            
            print(f"Debug - After filtering: {len(boxes_xywh)} detections")
            
            detected_objects = []
            if len(boxes_xywh) > 0:
                # Convert to xyxy format
                boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
                
                # Debug: Print raw box coordinates
                print(f"Debug - Raw boxes (first 3): {boxes_xywh[:3]}")
                print(f"Debug - Converted to xyxy (first 3): {boxes_xyxy[:3]}")
                
                # Apply NMS
                keep_indices = _nms(boxes_xyxy, class_conf, iou_threshold=0.45)
                boxes_xyxy = boxes_xyxy[keep_indices]
                class_ids = class_ids[keep_indices]
                class_conf = class_conf[keep_indices]
                
                print(f"Debug - After NMS: {len(boxes_xyxy)} detections")
                
                # Map boxes back to original image coordinates
                # Check if boxes are in normalized coordinates (0-1) or pixel coordinates
                if boxes_xyxy.max() <= 1.0:
                    print("Debug - Boxes appear to be normalized (0-1), converting to pixels")
                    # Convert from normalized coordinates to pixel coordinates
                    boxes_xyxy[:, [0, 2]] *= input_width   # x coordinates
                    boxes_xyxy[:, [1, 3]] *= input_height  # y coordinates
                
                # Now map from input image space to original image space
                # First, undo the letterbox padding
                boxes_xyxy[:, [0, 2]] -= pad_w
                boxes_xyxy[:, [1, 3]] -= pad_h
                
                # Then scale back to original image size
                boxes_xyxy = boxes_xyxy / max(1e-6, scale)
                
                # Clip to image boundaries
                boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, orig_w - 1)
                boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, orig_h - 1)
                
                # Debug: Print coordinates before and after transformation
                print(f"Debug - Boxes after transformation: {boxes_xyxy}")
                print(f"Debug - Scale: {scale}, Pad: ({pad_w}, {pad_h}), Orig size: {orig_w}x{orig_h}")
                
                # Convert to integers for drawing
                boxes_xyxy = boxes_xyxy.astype(int)
                
                for i, (box, cls_id, score) in enumerate(zip(boxes_xyxy, class_ids, class_conf)):
                    x1, y1, x2, y2 = box
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                    
                    # Debug: Print box coordinates
                    print(f"Debug - Box {i}: {class_name} {score:.3f} at ({x1},{y1},{x2},{y2})")
                    print(f"Debug - Frame size: {frame.shape[1]}x{frame.shape[0]}")
                    
                    detected_objects.append({
                        "class": class_name,
                        "score": score,
                        "bbox": (x1, y1, x2, y2)
                    })
                    
                    # Draw bounding box with thicker lines and different colors
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Thicker green box
                    
                    # Draw label background for better visibility
                    label = f"{class_name} {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Also draw a red circle at the center for visibility
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)

            # Find Relations
            relations = []
            if len(detected_objects) >= 2:
                for obj1, obj2 in itertools.permutations(detected_objects, 2):
                    x1_min, y1_min, x1_max, y1_max = obj1["bbox"]
                    x2_min, y2_min, x2_max, y2_max = obj2["bbox"]

                    # Check if obj1 is above obj2 and horizontally overlapping
                    horizontal_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                    vertical_distance = y2_min - y1_max
                    
                    # More lenient conditions for relationship detection
                    if (horizontal_overlap > 20 and  # At least 20px horizontal overlap
                        vertical_distance < 100 and  # Within 100px vertically
                        y1_max < y2_max):  # obj1 is above obj2
                        relations.append(f"{obj1['class']} on {obj2['class']}")

            # Show detected objects summary at top
            if detected_objects:
                object_names = [obj["class"] for obj in detected_objects]
                summary_text = f"Detected: {', '.join(object_names)}"
                text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (5, 5), (5 + text_size[0] + 10, 30), (0, 0, 0), -1)
                cv2.putText(frame, summary_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show relations with better visibility
            y_offset = 40
            for i, rel in enumerate(relations[:5]):  # Show max 5 relations
                # Draw background for text
                text_size = cv2.getTextSize(rel, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (5, y_offset - 20), (5 + text_size[0] + 10, y_offset + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, rel, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)  # Yellow text on black background
                y_offset += 30

            # Store the processed frame
            last_processed_frame = frame.copy()
            print(f"✅ Processed frame {frame_count}: Found {len(detected_objects)} objects")
            if relations:
                print(f"   Relations: {', '.join(relations)}")
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
        finally:
            processing = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
