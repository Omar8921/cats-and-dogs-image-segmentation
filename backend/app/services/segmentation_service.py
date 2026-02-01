from deep_learning.inference import run_image_segmentation
import cv2, base64

def segment_image(img_bytes: bytes):
    mask, overlay = run_image_segmentation(img_bytes)

    _, mask_buffer = cv2.imencode(".png", mask)
    _, overlay_buffer = cv2.imencode(".png", overlay)

    mask_b64 = base64.b64encode(mask_buffer).decode("utf-8")
    overlay_b64 = base64.b64encode(overlay_buffer).decode("utf-8")

    return {
        "mask": mask_b64,
        "overlay": overlay_b64
    }