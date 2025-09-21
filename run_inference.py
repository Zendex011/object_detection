import argparse
import os
import time
from typing import List, Tuple, Optional

import numpy as np


def _lazy_import_interpreter():
	"""Return a TFLite Interpreter from tflite_runtime if available, else TensorFlow."""
	try:
		from tflite_runtime.interpreter import Interpreter  # type: ignore
	except Exception:
		from tensorflow.lite import Interpreter  # type: ignore
	return Interpreter


def _lazy_import_ultralytics():
	"""Return YOLO from ultralytics."""
	try:
		from ultralytics import YOLO
		return YOLO
	except ImportError:
		raise ImportError("ultralytics not installed. Install with: pip install ultralytics")


def _cv2():
	import cv2  # lazy import to speed module import
	return cv2


def _letterbox_resize(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
	"""
	Resize image with unchanged aspect ratio using padding (letterbox) to target_size (w, h).
	Returns: resized_image, scale, (pad_w, pad_h)
	"""
	h, w = image.shape[:2]
	tw, th = target_size
	scale = min(tw / w, th / h)

	new_w, new_h = int(round(w * scale)), int(round(h * scale))
	resized = _cv2().resize(image, (new_w, new_h), interpolation=_cv2().INTER_LINEAR)
	canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
	pad_w = (tw - new_w) // 2
	pad_h = (th - new_h) // 2
	canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
	return canvas, scale, (pad_w, pad_h)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
	xyxy = np.empty_like(xywh)
	xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
	xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
	xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
	xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
	return xyxy


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
	"""Simple NMS returning kept indices."""
	if boxes.size == 0:
		return []
	# boxes: (N, 4) in xyxy
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


def run_yolo_tflite(
	image: np.ndarray,
	interpreter,
	input_details,
	output_details,
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
	"""
	Run YOLOv8 TFLite model on an image array.

	Args:
		image: Input image as numpy array (BGR format).
		interpreter: Loaded TFLite interpreter.
		input_details: Input tensor details.
		output_details: Output tensor details.
		conf_threshold: Confidence threshold.
		iou_threshold: IoU threshold for NMS.

	Returns:
		List of detections as tuples: (class_id, score, (x1, y1, x2, y2)) in original image coordinates.
	"""
	orig_h, orig_w = image.shape[:2]

	# Assume a single input: (1, H, W, 3)
	ih, iw = input_details[0]["shape"][1], input_details[0]["shape"][2]
	resized, scale, (pad_w, pad_h) = _letterbox_resize(image, (iw, ih))
	img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	img_input = np.expand_dims(img_rgb, axis=0)

	# Set input
	interpreter.set_tensor(input_details[0]["index"], img_input)
	interpreter.invoke()

	# Read outputs (support common Ultralytics formats)
	outputs = [interpreter.get_tensor(o["index"]) for o in output_details]

	# Normalize output to a unified format: boxes_xyxy, scores, classes
	boxes_xyxy = []
	scores = []
	classes = []

	if len(outputs) == 1:
		pred = outputs[0]
		# Common: (1, 84, 8400) -> 4 box + nc class logits
		if pred.ndim == 3 and pred.shape[0] == 1 and pred.shape[1] >= 5:
			pred = pred[0]
			num_outputs = pred.shape[1]
			if num_outputs >= 6:
				num_classes = pred.shape[0] - 4
				boxes_xywh = pred[0:4, :].T  # (N, 4)
				class_scores = pred[4:, :].T  # (N, C)
				class_ids = class_scores.argmax(axis=1)
				class_conf = class_scores.max(axis=1)
				object_conf = class_conf  # Some exports separate objectness; if not, use class max
				score = object_conf
				mask = score >= conf_threshold
				boxes_xywh = boxes_xywh[mask]
				class_ids = class_ids[mask]
				score = score[mask]
				boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
			else:
				raise RuntimeError("Unexpected TFLite output shape for single-tensor prediction.")
		else:
			raise RuntimeError("Unsupported prediction tensor shape.")
	elif len(outputs) >= 3:
		# Some TFLite models output boxes, classes, scores, count
		# Try to detect by shapes
		# boxes: (1, N, 4), classes: (1, N), scores: (1, N)
		b = None
		c = None
		s = None
		for out in outputs:
			if out.ndim == 3 and out.shape[-1] == 4:
				b = out[0]
			elif out.ndim == 2:
				if s is None:
					s = out[0]
				else:
					c = out[0]
		if b is None or c is None or s is None:
			raise RuntimeError("Could not parse multi-tensor TFLite outputs.")
		mask = s >= conf_threshold
		boxes_xyxy = b[mask]
		scores = s[mask]
		classes = c[mask].astype(int)
	else:
		raise RuntimeError("Unknown TFLite output format.")

	boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)
	if not isinstance(scores, np.ndarray) or scores == []:
		# Build scores/classes if single-tensor path not set them yet
		scores = score  # type: ignore[name-defined]
		classes = class_ids  # type: ignore[name-defined]
	else:
		scores = np.array(scores, dtype=np.float32)
		classes = np.array(classes, dtype=np.int32)

	# NMS per class
	final_indices = []
	for cls_id in np.unique(classes):
		cls_mask = classes == cls_id
		keep = _nms(boxes_xyxy[cls_mask], scores[cls_mask], iou_threshold)
		mapped = np.where(cls_mask)[0][keep]
		final_indices.extend(mapped.tolist())

	final_indices = np.array(final_indices, dtype=np.int32)
	boxes_xyxy = boxes_xyxy[final_indices]
	scores = scores[final_indices]
	classes = classes[final_indices]

	# Map boxes back to original image coords (undo letterbox)
	# Current boxes are in resized image space (iw x ih)
	boxes_xyxy[:, [0, 2]] -= pad_w
	boxes_xyxy[:, [1, 3]] -= pad_h
	boxes_xyxy = boxes_xyxy / max(1e-6, scale)
	boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clip(0, orig_w - 1)
	boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clip(0, orig_h - 1)

	# Return detections
	results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
	for cls_id, sc, box in zip(classes.tolist(), scores.tolist(), boxes_xyxy.astype(int).tolist()):
		results.append((int(cls_id), float(sc), (int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
	return results


def run_yolo_pytorch(
	image: np.ndarray,
	model,
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
	"""
	Run YOLOv8 PyTorch model on an image array.

	Args:
		image: Input image as numpy array (BGR format).
		model: Loaded YOLO model.
		conf_threshold: Confidence threshold.
		iou_threshold: IoU threshold for NMS.

	Returns:
		List of detections as tuples: (class_id, score, (x1, y1, x2, y2)) in original image coordinates.
	"""
	results = model(image, conf=conf_threshold, iou=iou_threshold)
	result = results[0]
	
	detections = []
	if result.boxes is not None:
		boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
		confidences = result.boxes.conf.cpu().numpy()
		class_ids = result.boxes.cls.cpu().numpy().astype(int)
		
		for box, conf, cls_id in zip(boxes, confidences, class_ids):
			x1, y1, x2, y2 = box.astype(int)
			detections.append((int(cls_id), float(conf), (int(x1), int(y1), int(x2), int(y2))))
	
	return detections


def draw_detections(image: np.ndarray, detections: List[Tuple[int, float, Tuple[int, int, int, int]]], 
                   class_names: Optional[List[str]] = None) -> np.ndarray:
	"""Draw bounding boxes and labels on the image."""
	cv2 = _cv2()
	vis = image.copy()
	
	# Default class names if not provided
	if class_names is None:
		class_names = [
			'bed', 'sofa', 'chair', 'table', 'lamp', 'tv', 'laptop', 'wardrobe',
			'window', 'door', 'potted plant', 'photo frame'
		]
	
	for cls_id, score, (x1, y1, x2, y2) in detections:
		# Draw bounding box
		cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
		
		# Prepare label
		class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
		label = f"{class_name}: {score:.2f}"
		
		# Draw label background
		label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
		cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
		
		# Draw label text
		cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	
	return vis


def run_camera_inference(
	model_path: str = "best.pt",
	model_type: str = "pytorch",  # "pytorch" or "tflite"
	camera_id: int = 0,
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	display_fps: bool = True,
	class_names: Optional[List[str]] = None,
):
	"""
	Run real-time object detection on camera feed.

	Args:
		model_path: Path to model file (.pt for PyTorch, .tflite for TFLite).
		model_type: Type of model ("pytorch" or "tflite").
		camera_id: Camera device ID (usually 0 for default camera).
		conf_threshold: Confidence threshold for detections.
		iou_threshold: IoU threshold for NMS.
		display_fps: Whether to display FPS counter.
		class_names: List of class names for display.
	"""
	cv2 = _cv2()
	
	# Load model
	print(f"Loading {model_type} model from: {model_path}")
	
	if model_type.lower() == "pytorch":
		YOLO = _lazy_import_ultralytics()
		model = YOLO(model_path)
		interpreter = None
		input_details = None
		output_details = None
	elif model_type.lower() == "tflite":
		Interpreter = _lazy_import_interpreter()
		interpreter = Interpreter(model_path=model_path)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		model = None
	else:
		raise ValueError("model_type must be 'pytorch' or 'tflite'")
	
	print("✅ Model loaded successfully!")
	
	# Initialize camera
	print(f"Initializing camera {camera_id}...")
	cap = cv2.VideoCapture(camera_id)
	
	if not cap.isOpened():
		raise RuntimeError(f"Could not open camera {camera_id}")
	
	# Set camera properties for better performance
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 30)
	
	print("✅ Camera initialized successfully!")
	print("\nPress 'q' to quit, 's' to save current frame")
	
	# FPS calculation
	fps_counter = 0
	fps_start_time = time.time()
	fps = 0
	
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				print("❌ Failed to read from camera")
				break
			
			# Run inference
			if model_type.lower() == "pytorch":
				detections = run_yolo_pytorch(frame, model, conf_threshold, iou_threshold)
			else:  # tflite
				detections = run_yolo_tflite(frame, interpreter, input_details, output_details, 
				                           conf_threshold, iou_threshold)
			
			# Draw detections
			annotated_frame = draw_detections(frame, detections, class_names)
			
			# Calculate and display FPS
			if display_fps:
				fps_counter += 1
				if fps_counter % 30 == 0:  # Update FPS every 30 frames
					fps_end_time = time.time()
					fps = 30 / (fps_end_time - fps_start_time)
					fps_start_time = fps_end_time
				
				cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
				           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			
			# Display detection count
			cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 70), 
			           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			
			# Show frame
			cv2.imshow("YOLOv8 Real-time Object Detection", annotated_frame)
			
			# Handle key presses
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('s'):
				# Save current frame
				timestamp = int(time.time())
				filename = f"camera_detection_{timestamp}.jpg"
				cv2.imwrite(filename, annotated_frame)
				print(f"💾 Saved frame as: {filename}")
	
	except KeyboardInterrupt:
		print("\n⏹️  Stopped by user")
	
	finally:
		# Cleanup
		cap.release()
		cv2.destroyAllWindows()
		print("✅ Camera released and windows closed")


def main():
	parser = argparse.ArgumentParser(description="Run YOLOv8 real-time object detection on camera feed.")
	parser.add_argument("--model", default="best.pt", help="Path to model file (.pt or .tflite)")
	parser.add_argument("--type", choices=["pytorch", "tflite"], default="pytorch", 
	                   help="Model type: pytorch or tflite")
	parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
	parser.add_argument("--no-fps", action="store_true", help="Disable FPS display")
	args = parser.parse_args()
	
	# Check if model file exists
	if not os.path.isfile(args.model):
		print(f"❌ Model file not found: {args.model}")
		return
	
	# Run camera inference
	run_camera_inference(
		model_path=args.model,
		model_type=args.type,
		camera_id=args.camera,
		conf_threshold=args.conf,
		iou_threshold=args.iou,
		display_fps=not args.no_fps,
	)


if __name__ == "__main__":
	main()