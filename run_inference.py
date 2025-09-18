import argparse
import os
from typing import List, Tuple

import numpy as np


def _lazy_import_interpreter():
	"""Return a TFLite Interpreter from tflite_runtime if available, else TensorFlow."""
	try:
		from tflite_runtime.interpreter import Interpreter  # type: ignore
	except Exception:
		from tensorflow.lite import Interpreter  # type: ignore
	return Interpreter


def _letterbox_resize(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
	"""
	Resize image with unchanged aspect ratio using padding (letterbox) to target_size (w, h).
	Returns: resized_image, scale, (pad_w, pad_h)
	"""
	h, w = image.shape[:2]
	tw, th = target_size
	scale = min(tw / w, th / h)
\n+	new_w, new_h = int(round(w * scale)), int(round(h * scale))
	resized = _cv2().resize(image, (new_w, new_h), interpolation=_cv2().INTER_LINEAR)
	canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
	pad_w = (tw - new_w) // 2
	pad_h = (th - new_h) // 2
	canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
	return canvas, scale, (pad_w, pad_h)


def _cv2():
	import cv2  # lazy import to speed module import
	return cv2


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
	image_path: str,
	model_path: str = "best_float32.tflite",
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	output_path: str = "output_detections.jpg",
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
	"""
	Run YOLOv8 TFLite model on an image.

	Args:
		image_path: Path to input image.
		model_path: Path to TFLite model (default: best_float32.tflite in CWD).
		conf_threshold: Confidence threshold.
		iou_threshold: IoU threshold for NMS.
		output_path: Where to save visualization image.

	Returns:
		List of detections as tuples: (class_id, score, (x1, y1, x2, y2)) in original image coordinates.
	"""
	if not os.path.isfile(image_path):
		raise FileNotFoundError(f"Image not found: {image_path}")
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"Model not found: {model_path}")

	cv2 = _cv2()
	image_bgr = cv2.imread(image_path)
	if image_bgr is None:
		raise ValueError(f"Failed to read image: {image_path}")
	orig_h, orig_w = image_bgr.shape[:2]

	Interpreter = _lazy_import_interpreter()
	interpreter = Interpreter(model_path=model_path)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Assume a single input: (1, H, W, 3)
	ih, iw = input_details[0]["shape"][1], input_details[0]["shape"][2]
	resized, scale, (pad_w, pad_h) = _letterbox_resize(image_bgr, (iw, ih))
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

	# Draw and save
	vis = image_bgr.copy()
	for cls_id, sc, box in zip(classes.tolist(), scores.tolist(), boxes_xyxy.astype(int).tolist()):
		x1, y1, x2, y2 = box
		cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
		label = f"{cls_id}:{sc:.2f}"
		cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	cv2.imwrite(output_path, vis)

	# Return detections
	results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
	for cls_id, sc, box in zip(classes.tolist(), scores.tolist(), boxes_xyxy.astype(int).tolist()):
		results.append((int(cls_id), float(sc), (int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
	return results


def main():
	parser = argparse.ArgumentParser(description="Run YOLOv8 TFLite inference on an image.")
	parser.add_argument("image", help="Path to input image")
	parser.add_argument("--model", default="best_float32.tflite", help="Path to TFLite model")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
	parser.add_argument("--out", default="output_detections.jpg", help="Output image path")
	args = parser.parse_args()

	results = run_yolo_tflite(
		image_path=args.image,
		model_path=args.model,
		conf_threshold=args.conf,
		iou_threshold=args.iou,
		output_path=args.out,
	)
	print(f"Detections ({len(results)}):")
	for cls_id, score, (x1, y1, x2, y2) in results:
		print(f"class={cls_id}, score={score:.3f}, box=({x1},{y1},{x2},{y2})")


if __name__ == "__main__":
	main()


