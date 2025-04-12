import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def normalize_keypoints(keypoints):
    """
    Normalize keypoints to [0,1] range.
    keypoints: np.ndarray of shape (T, 299) or (299,)
    """
    try:
        keypoints = np.array(keypoints)
        # Avoid division by zero
        max_val = np.max(np.abs(keypoints))
        if max_val > 0:
            keypoints = keypoints / max_val
        # Clip to [0,1]
        keypoints = np.clip(keypoints, 0, 1)
        logging.info(f"Normalized keypoints: shape={keypoints.shape}, range=[{keypoints.min()}, {keypoints.max()}]")
        return keypoints
    except Exception as e:
        logging.error(f"Normalization failed: {e}")
        return keypoints

def augment_keypoints(keypoints, scale_range=(0.9, 1.1), rotation_angle=10):
    """
    Apply data augmentation: random scaling and rotation.
    keypoints: np.ndarray of shape (T, 299)
    """
    try:
        keypoints = np.array(keypoints)
        T, D = keypoints.shape
        
        # Separate modalities
        hand_kp = keypoints[:, :84]    # 42 left + 42 right
        face_kp = keypoints[:, 84:224] # 140
        body_kp = keypoints[:, 224:299] # 75
        
        # Random scaling
        scale = np.random.uniform(scale_range[0], scale_range[1])
        hand_kp *= scale
        face_kp *= scale
        body_kp *= scale
        
        # Random rotation for x, y coordinates
        angle = np.radians(np.random.uniform(-rotation_angle, rotation_angle))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        def rotate_kp(kp, num_points, has_confidence=False):
            """Rotate x, y coordinates for num_points keypoints."""
            rotated = kp.copy()
            for i in range(num_points):
                x_idx = i * (3 if has_confidence else 2)
                y_idx = x_idx + 1
                x, y = kp[:, x_idx], kp[:, y_idx]
                rotated[:, x_idx] = x * cos_a - y * sin_a
                rotated[:, y_idx] = x * sin_a + y * cos_a
            return rotated
        
        # Rotate hands (21 keypoints each, no confidence)
        hand_kp[:, :42] = rotate_kp(hand_kp[:, :42], 21)
        hand_kp[:, 42:84] = rotate_kp(hand_kp[:, 42:84], 21)
        # Rotate face (70 keypoints, no confidence)
        face_kp = rotate_kp(face_kp, 70)
        # Rotate body (25 keypoints, with confidence)
        body_kp = rotate_kp(body_kp, 25, has_confidence=True)
        
        # Recombine
        augmented = np.concatenate([hand_kp, face_kp, body_kp], axis=1)
        logging.info(f"Augmented keypoints: shape={augmented.shape}, scale={scale:.2f}, angle={np.degrees(angle):.2f}°")
        return augmented
    except Exception as e:
        logging.error(f"Augmentation failed: {e}")
        return keypoints

def preprocess_keypoints(keypoints, normalize=True, augment=False):
    """
    Preprocess keypoints with optional normalization and augmentation.
    keypoints: np.ndarray of shape (T, 299) or (299,)
    """
    keypoints = np.array(keypoints)
    
    if normalize:
        keypoints = normalize_keypoints(keypoints)
    
    if augment:
        keypoints = augment_keypoints(keypoints)
    
    return keypoints