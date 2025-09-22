# ==========================
# 3️⃣ Frame Capture and Processing
# ==========================
frame_count = 0
process_every_n = 1  # Process every frame for visible boxes on the stream
last_processed_frame = None
processing = False

# FPS calculation
fps_counter = 0
fps_start_time = time.time()
fps = 0

# Initialize video writer to save annotated output
output_video_path = "C:/Users/ASUS/OneDrive/Desktop/od/output_detections.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
writer = cv2.VideoWriter(output_video_path, fourcc, vid_fps, (frame_w, frame_h))
print(f"💾 Saving annotated video to: {output_video_path} @ {vid_fps:.1f} FPS, {frame_w}x{frame_h}")

# ==========================
# 4️⃣ Improved Detection Loop with Slower Display
# ==========================
process_interval_sec = 2  # process every 2 seconds
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames_to_skip = int(fps_video * process_interval_sec)

frame_count = 0
last_processed_frame = None

while True:
    # Skip frames until next processing interval
    for _ in range(frames_to_skip - 1):  # -1 because we will read one below
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    # Read frame to process
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    print(f"\n🔍 Processing frame {frame_count}...")
    try:
        # Preprocess frame
        orig_h, orig_w = frame.shape[:2]
        resized, scale, (pad_w, pad_h) = _letterbox_resize(frame, (input_width, input_height))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_data = np.expand_dims(img_rgb, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Parse output
        num_classes = output_data.shape[0] - 4
        boxes_xywh = output_data[0:4, :].T
        class_scores = output_data[4:, :].T

        class_ids = class_scores.argmax(axis=1)
        class_conf = class_scores.max(axis=1)

        # Filter by confidence
        conf_threshold = 0.1
        mask = class_conf >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        class_conf = class_conf[mask]

        detected_objects = []

        if len(boxes_xywh) > 0:
            boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
            keep_indices = _nms(boxes_xyxy, class_conf, iou_threshold=0.45)
            boxes_xyxy = boxes_xyxy[keep_indices]
            class_ids = class_ids[keep_indices]
            class_conf = class_conf[keep_indices]

            # Map boxes back to original frame
            if boxes_xyxy.max() <= 1.0:
                boxes_xyxy[:, [0, 2]] *= input_width
                boxes_xyxy[:, [1, 3]] *= input_height

            boxes_xyxy[:, [0, 2]] -= pad_w
            boxes_xyxy[:, [1, 3]] -= pad_h
            boxes_xyxy = boxes_xyxy / max(1e-6, scale)
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, orig_w - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, orig_h - 1)
            boxes_xyxy = boxes_xyxy.astype(int)

            for box, cls_id, score in zip(boxes_xyxy, class_ids, class_conf):
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                if class_name.lower() == "unknown":
                    continue  # skip unknowns

                x1, y1, x2, y2 = box
                detected_objects.append({
                    "class": class_name,
                    "score": score,
                    "bbox": (x1, y1, x2, y2)
                })

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                label = f"{class_name} {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)

        # Detect relationships if >=2 objects
        relations = []
        if len(detected_objects) >= 2:
            detections_for_relation = [{
                'label': obj['class'],
                'x_min': obj['bbox'][0],
                'y_min': obj['bbox'][1],
                'x_max': obj['bbox'][2],
                'y_max': obj['bbox'][3]
            } for obj in detected_objects]
            relations = detect_relationships(detections_for_relation, vertical_threshold=0.2, horizontal_threshold=0.2)
            relations = list(dict.fromkeys(relations))

        last_processed_frame = frame.copy()
        print(f"✅ Processed frame {frame_count}: Found {len(detected_objects)} objects")
        if relations:
            print(f"   Relations: {', '.join(relations)}")

    except Exception as e:
        print(f"❌ Error processing frame: {e}")

    # Display last processed frame and hold it for the interval
    if last_processed_frame is not None:
        cv2.imshow("Live Object Detection", last_processed_frame)
        if 'writer' in locals() and writer is not None:
            try:
                writer.write(last_processed_frame)
            except Exception:
                pass

        # Hold frame for interval duration
        if cv2.waitKey(int(process_interval_sec * 1000)) & 0xFF == ord('q'):
            break

cap.release()
if 'writer' in locals() and writer is not None:
    writer.release()
cv2.destroyAllWindows()


"""

import cv2
import numpy as np
import tensorflow as tf
from indoor import detect_relationships
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
# 2️⃣ Open Video File (for testing)
# ==========================
VIDEO_PATH = r"C:\Users\ASUS\OneDrive\Desktop\od\test\WhatsApp Video 2025-09-22 at 23.19.56_0f51541b.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open video file: {VIDEO_PATH}")

# NOTE: Live stream code retained below for future use
# cap = cv2.VideoCapture(0)  # 0 = default camera
# if not cap.isOpened():
#     raise RuntimeError("❌ Could not access webcam")

# Class names (12 classes based on your model output)
CLASS_NAMES = [
    "bed", "table", "pillow", "glass", "chair", "sofa",
    "book", "lamp", "tv", "cup", "bottle", "unknown"
]

print("✅ Video opened. Press 'q' to quit.")

# ==========================
# 4️⃣ Tiled Detection Loop
# ==========================
process_interval_sec = 2  # process every 2 seconds
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames_to_skip = int(fps_video * process_interval_sec)

frame_count = 0
last_processed_frame = None

tile_rows, tile_cols = 2, 2  # 2x2 grid
overlap = 0.1  # 10% overlap to catch boundary objects

while True:
    # Skip frames until next processing interval
    for _ in range(frames_to_skip - 1):
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    # Read frame to process
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    print(f"\n🔍 Processing frame {frame_count}...")

    try:
        orig_h, orig_w = frame.shape[:2]
        detected_objects = []

        # Determine tile sizes
        tile_h = int(orig_h / tile_rows)
        tile_w = int(orig_w / tile_cols)

        for r in range(tile_rows):
            for c in range(tile_cols):
                # Compute tile coordinates with overlap
                y1 = max(0, r * tile_h - int(tile_h * overlap))
                y2 = min(orig_h, (r + 1) * tile_h + int(tile_h * overlap))
                x1 = max(0, c * tile_w - int(tile_w * overlap))
                x2 = min(orig_w, (c + 1) * tile_w + int(tile_w * overlap))

                tile = frame[y1:y2, x1:x2]

                # Preprocess tile
                resized, scale, (pad_w, pad_h) = _letterbox_resize(tile, (input_width, input_height))
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_data = np.expand_dims(img_rgb, axis=0)

                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # Parse output
                boxes_xywh = output_data[0:4, :].T
                class_scores = output_data[4:, :].T
                class_ids = class_scores.argmax(axis=1)
                class_conf = class_scores.max(axis=1)

                # Filter by confidence
                mask = class_conf >= 0.1
                boxes_xywh = boxes_xywh[mask]
                class_ids = class_ids[mask]
                class_conf = class_conf[mask]

                if len(boxes_xywh) == 0:
                    continue

                # Convert to xyxy
                boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
                keep_indices = _nms(boxes_xyxy, class_conf, iou_threshold=0.45)
                boxes_xyxy = boxes_xyxy[keep_indices]
                class_ids = class_ids[keep_indices]
                class_conf = class_conf[keep_indices]

                # Map back to original frame
                if boxes_xyxy.max() <= 1.0:
                    boxes_xyxy[:, [0, 2]] *= input_width
                    boxes_xyxy[:, [1, 3]] *= input_height

                boxes_xyxy[:, [0, 2]] -= pad_w
                boxes_xyxy[:, [1, 3]] -= pad_h
                boxes_xyxy = boxes_xyxy / max(1e-6, scale)
                boxes_xyxy[:, 0::2] += x1  # offset by tile x
                boxes_xyxy[:, 1::2] += y1  # offset by tile y
                boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, orig_w - 1)
                boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, orig_h - 1)
                boxes_xyxy = boxes_xyxy.astype(int)

                for box, cls_id, score in zip(boxes_xyxy, class_ids, class_conf):
                    class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
                    if class_name.lower() == "unknown":
                        continue

                    x1b, y1b, x2b, y2b = box
                    detected_objects.append({
                        "class": class_name,
                        "score": score,
                        "bbox": (x1b, y1b, x2b, y2b)
                    })

        # Apply NMS again across all tiles
        if detected_objects:
            all_boxes = np.array([obj['bbox'] for obj in detected_objects])
            all_scores = np.array([obj['score'] for obj in detected_objects])
            keep = _nms(all_boxes, all_scores, iou_threshold=0.45)
            detected_objects = [detected_objects[i] for i in keep]

        # Draw boxes and labels
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name, score = obj['class'], obj['score']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            label = f"{class_name} {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)

        # Detect relationships if >=2 objects
        relations = []
        if len(detected_objects) >= 2:
            detections_for_relation = [{
                'label': obj['class'],
                'x_min': obj['bbox'][0],
                'y_min': obj['bbox'][1],
                'x_max': obj['bbox'][2],
                'y_max': obj['bbox'][3]
            } for obj in detected_objects]
            relations = detect_relationships(detections_for_relation, vertical_threshold=0.2,
                                             horizontal_threshold=0.2)
            relations = list(dict.fromkeys(relations))

        last_processed_frame = frame.copy()
        print(f"✅ Processed frame {frame_count}: Found {len(detected_objects)} objects")
        if relations:
            print(f"   Relations: {', '.join(relations)}")

    except Exception as e:
        print(f"❌ Error processing frame: {e}")

    # Display and save
    if last_processed_frame is not None:
        cv2.imshow("Live Object Detection", last_processed_frame)
        if 'writer' in locals() and writer is not None:
            try:
                writer.write(last_processed_frame)
            except Exception:
                pass

        # Hold frame for interval duration
        if cv2.waitKey(int(process_interval_sec * 1000)) & 0xFF == ord('q'):
            break

cap.release()
if 'writer' in locals() and writer is not None:
    writer.release()
cv2.destroyAllWindows()

"""