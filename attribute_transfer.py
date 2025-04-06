# attribute_transfer.py

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def get_mouth_box_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    mx = x + int(w * 0.2)
    mw = int(w * 0.6)
    my = y + int(h * 2 / 3)
    mh = int(h / 3)
    return (mx, my, mw, mh)

def gen_gaussian_pyramid(img, levels):
    gp = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        gp.append(img)
    return gp

def gen_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lap = gp[i] - up
        lp.append(lap)
    lp.append(gp[-1])
    return lp

def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(lp[i].shape[1], lp[i].shape[0]))
        img = img + lp[i]
    return img

def pyramid_blend(img1, img2, mask, levels=6):
    gp1 = gen_gaussian_pyramid(img1, levels)
    gp2 = gen_gaussian_pyramid(img2, levels)
    gp_mask = gen_gaussian_pyramid(mask, levels)
    lp1 = gen_laplacian_pyramid(gp1)
    lp2 = gen_laplacian_pyramid(gp2)
    blended = [l1 * m + l2 * (1 - m) for l1, l2, m in zip(lp1, lp2, gp_mask)]
    return reconstruct_from_laplacian(blended)

def match_color(source, target):
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    for i in range(3):
        s_mean, s_std = source[..., i].mean(), source[..., i].std()
        t_mean, t_std = target[..., i].mean(), target[..., i].std()
        if s_std > 0:
            source[..., i] = ((source[..., i] - s_mean) / s_std) * t_std + t_mean
    return np.clip(source, 0, 255).astype(np.uint8)

def run_emotion_transfer(source_pil, target_pil):
    # Convert to OpenCV
    source_img = pil_to_cv2(source_pil)
    target_img = pil_to_cv2(target_pil)

    # Resize for consistency
    source_img = cv2.resize(source_img, (512, 512))
    target_img = cv2.resize(target_img, (512, 512))

    source_box = get_mouth_box_opencv(source_img)
    target_box = get_mouth_box_opencv(target_img)
    if source_box is None or target_box is None:
        raise Exception("Could not detect mouth region.")

    sx, sy, sw, sh = source_box
    tx, ty, tw, th = target_box
    ty += 10  # small vertical alignment

    # Prepare mouth patch
    source_mouth = source_img[sy:sy + sh, sx:sx + sw]
    source_mouth_resized = cv2.resize(source_mouth, (tw, th))
    target_mouth = target_img[ty:ty + th, tx:tx + tw]
    source_mouth_matched = match_color(source_mouth_resized, target_mouth)

    # Copy patch onto target
    warped = target_img.copy()
    warped[ty:ty + th, tx:tx + tw] = source_mouth_matched

    # Create elliptical mask
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    mask_shift_y = 5
    scale_factor = 0.85
    scaled_tw = int(tw * scale_factor)
    scaled_th = int(th * scale_factor)
    center = (tx + tw // 2, ty + th // 2 + mask_shift_y)
    axes = (scaled_tw // 2, scaled_th // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.

    # Blend images
    blended = pyramid_blend(warped.astype(np.float32), target_img.astype(np.float32), mask)
    final_result = np.clip(blended, 0, 255).astype(np.uint8)

    return cv2_to_pil(final_result)
