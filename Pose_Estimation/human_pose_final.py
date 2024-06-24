import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Mask R-CNN configuration for inference

class InferenceConfig(Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)


model_dir = './'  
mask_rcnn_model = modellib.MaskRCNN(
    mode="inference", config=InferenceConfig(), model_dir=model_dir)
mask_rcnn_model.load_weights('mask_rcnn_coco.h5', by_name=True)

# pose estimation model
def create_pose_estimation_model(num_keypoints, input_tensor=None):
    base_model = ResNet50(
        include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    final_output = Dense(
        num_keypoints * 2, activation='linear')(x)  # 2 for x and y
    model = Model(inputs=base_model.input, outputs=final_output)
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    return model


def load_dataset(annotations_path, images_dir):
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)
    images = []
    keypoints = []
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        image_file = f"{image_id}.jpg"
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip images that can't be loaded
        kp = ann['keypoints'] if 'keypoints' in ann else []
        images.append(image)
        keypoints.append(kp)
    return images, keypoints


def preprocess_image_mask_rcnn(image, target_size=(1024, 1024)):
    h, w = image.shape[:2]
    scale = min(target_size[1] / h, target_size[0] / w)

    new_h = int(h * scale)
    new_w = int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    top_pad = (target_size[1] - new_h) // 2
    bottom_pad = target_size[1] - new_h - top_pad
    left_pad = (target_size[0] - new_w) // 2
    right_pad = target_size[0] - new_w - left_pad
    pad_color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(
        resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=pad_color)

    return padded_image, new_w, new_h, left_pad, top_pad


def preprocess_image_pose_estimation(image, image_size=(224, 224)):
    image = cv2.resize(image, image_size)
    image = image.astype('float32') / 255.0
    return image



skeleton = [
    (1, 2),  # head to neck
    (2, 3),  # neck to right shoulder
    (2, 4),  # neck to left shoulder
    (3, 5),  # right shoulder to right elbow
    (4, 6),  # left shoulder to left elbow
    (5, 7),  # right elbow to right wrist
    (6, 8),  # left elbow to left wrist
    (9, 10),  # right hip to left hip
    (9, 11),  # right hip to right knee
    (10, 12),  # left hip to left knee
    (11, 13),  # right knee to right ankle
    (12, 14)  # left knee to left ankle
]

# Function to preprocess the data
def preprocess_data(images, keypoints, num_keypoints, image_size=(224, 224)):
    preprocessed_images = []
    preprocessed_keypoints = []
    for img, kps in zip(images, keypoints):
        img = cv2.resize(img, image_size)
        img = img.astype('float32') / 255.0

        kp_array = np.array(kps).reshape(-1, 3)
        kp_coords = kp_array[:, :2].astype(np.float32)
        kp_visible = kp_array[:, 2] > 0

        original_size = img.shape[:2]
        x_scale = image_size[0] / original_size[1]
        y_scale = image_size[1] / original_size[0]
        kp_coords[:, 0] *= x_scale
        kp_coords[:, 1] *= y_scale

        kps_processed = kp_coords[kp_visible].flatten()
        pad_size = (num_keypoints * 2) - len(kps_processed)
        kps_processed = np.pad(kps_processed, (0, pad_size), mode='constant')

        preprocessed_images.append(img)
        preprocessed_keypoints.append(kps_processed)

    return np.array(preprocessed_images), np.array(preprocessed_keypoints)


def train_model(model, images, keypoints, epochs, batch_size=32):
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x=images, y=keypoints, batch_size=batch_size,
                        epochs=epochs, validation_split=0.1)
    return history

def evaluate_model(model, images, keypoints):
    predicted_keypoints = model.predict(images)
    true_keypoints_flat = []
    predicted_keypoints_flat = []
    for true_kp, pred_kp in zip(keypoints, predicted_keypoints):
        true_keypoints_flat.append(true_kp.flatten())
        predicted_keypoints_flat.append(pred_kp.flatten())
    mse = mean_squared_error(true_keypoints_flat, predicted_keypoints_flat)
    print(f"Mean Squared Error on the test set: {mse:.4f}")
    return mse

def compare_angles(expert_keypoints, newbie_keypoints):
    """
    Compare angles between expert and newbie using their keypoints.
    """
    angle_differences = {}

    # Define the skeleton connections
    skeleton = {
        'right_elbow': (4, 6, 8),
        'left_elbow': (5, 7, 9),
        'right_knee': (10, 12, 14),
        'left_knee': (11, 13, 15)
    }

    for joint, (pt1, pt2, pt3) in skeleton.items():
        expert_angle = calculate_angle(expert_keypoints[pt1-1][:2], expert_keypoints[pt2-1][:2], expert_keypoints[pt3-1][:2])
        newbie_angle = calculate_angle(newbie_keypoints[pt1-1][:2], newbie_keypoints[pt2-1][:2], newbie_keypoints[pt3-1][:2])

        angle_differences[joint] = abs(expert_angle - newbie_angle)

    return angle_differences

num_keypoints = 15 
pose_estimation_model = create_pose_estimation_model(num_keypoints)


train_annotations_path = 'DL\\Project\\yoga_pose_annotations_coco.json'
train_images_dir = 'DL\\Project\\yoga_dataset_train'
train_images, train_keypoints = load_dataset(
    train_annotations_path, train_images_dir)
preprocessed_train_images, preprocessed_train_keypoints = preprocess_data(
    train_images, train_keypoints, num_keypoints)


input_tensor = Input(shape=(224, 224, 3))
pose_estimation_model = create_pose_estimation_model(
    num_keypoints=15, input_tensor=input_tensor)


history = train_model(pose_estimation_model, preprocessed_train_images,
                      preprocessed_train_keypoints, epochs=1000, batch_size=32)



# Performance Plots
plt.style.use('ggplot')
plt.figure(figsize=(18, 6))


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='royalblue', marker='o', linewidth=2)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout(pad=2)


plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='darkorange', marker='s', linewidth=2)
plt.title('Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout(pad=2)

plt.show()


plt.style.use('ggplot')
plt.figure(figsize=(18, 6))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='royalblue', marker='o', linewidth=2)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout(pad=2)

plt.subplot(1, 2, 2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange', marker='s', linewidth=2)
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout(pad=2)

plt.show()


test_image_path = 'DL\Project\yoga_dataset_test\7.jpg'
test_image = cv2.imread(test_image_path)
test_image_preprocessed, new_w, new_h, left_pad, top_pad = preprocess_image_mask_rcnn(
    test_image)


mask_color = (0, 0, 255)
keypoint_color = (0, 255, 0)
skeleton_color = (255, 255, 255)


image = cv2.imread(test_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detection_results = mask_rcnn_model.detect([image], verbose=0)[0]

output_image = image.copy()
r = detection_results

# custom color (red) for the masks
RED_COLOR = np.array([1.0, 0.0, 0.0])

colors = [RED_COLOR] * len(r['class_ids']) 
visualize.display_instances(image=image,
                            boxes=r['rois'],
                            masks=r['masks'],
                            class_ids=r['class_ids'],
                            class_names=CLASS_NAMES,
                            scores=r['scores'],
                            colors=colors) 

person_indices = np.where(r['class_ids'] == 1)[0] 
person_boxes = r['rois'][person_indices]

fig, ax = plt.subplots(1, figsize=(16, 16))
ax.imshow(image)

# Iterate over detected objects
for i in range(detection_results['rois'].shape[0]):
    if detection_results['class_ids'][i] == 1: 
        y1, x1, y2, x2 = detection_results['rois'][i]


        person_image_preprocessed = preprocess_image_pose_estimation(
            output_image[y1:y2, x1:x2])
        keypoints_pred = pose_estimation_model.predict(
            np.expand_dims(person_image_preprocessed, axis=0))[0]
        keypoints_with_visibility = np.array(keypoints_pred).reshape(-1, 3)

        scale_x, scale_y = (x2 - x1) / 224.0, (y2 - y1) / 224.0
        keypoints_with_visibility[:, 0] = keypoints_with_visibility[:, 0] * scale_x + x1
        keypoints_with_visibility[:, 1] = keypoints_with_visibility[:, 1] * scale_y + y1

        for x, y, v in keypoints_with_visibility:
            if v > 0:
                cv2.circle(output_image, (int(x), int(y)), 5, keypoint_color, -1)
        
        for start, end in skeleton:
            start_idx, end_idx = start - 1, end - 1
            if start_idx < len(keypoints_with_visibility) and end_idx < len(keypoints_with_visibility):
                start_point = keypoints_with_visibility[start_idx]
                end_point = keypoints_with_visibility[end_idx]
                if start_point[2] > 0 and end_point[2] > 0:
                    cv2.line(output_image, (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])), skeleton_color, 2)

# Load an expert image and preprocess it for Mask R-CNN
expert_image_path = 'DL\\Project\\yoga_dataset_test\\expert.jpg'  # Change this to your expert image path
expert_image = cv2.imread(expert_image_path)
expert_image_preprocessed, new_w_expert, new_h_expert, left_pad_expert, top_pad_expert = preprocess_image_mask_rcnn(expert_image)

# Detect objects in the expert image (assuming only people are of interest)
expert_results = mask_rcnn_model.detect([expert_image_preprocessed], verbose=1)
expert_r = expert_results[0]

# Set the colors for the mask, keypoints, and skeleton
mask_color = (0, 0, 255)  # Red color in BGR
keypoint_color = (0, 255, 0)  # Green color
skeleton_color = (255, 255, 255)  # White color

# Create a copy of the expert image to draw on
output_expert_image = expert_image.copy()

# Process each detected person in the expert image
for i in range(expert_r['rois'].shape[0]):
    if expert_r['class_ids'][i] == 1:  # Check for the person class
        y1, x1, y2, x2 = expert_r['rois'][i]
        cv2.rectangle(output_expert_image, (x1, y1), (x2, y2),
                      (255, 0, 0), 2)  # Draw bounding box in blue

        # Extract the mask for the detected person, resize it to the bounding box, then create a full-size mask
        person_mask = expert_r['masks'][:, :, i].astype(np.uint8)
        resized_person_mask = cv2.resize(person_mask, (x2 - x1, y2 - y1))
        full_image_mask = np.zeros(output_expert_image.shape[:2], dtype=bool)
        full_image_mask[y1:y2, x1:x2] = resized_person_mask.astype(bool)

        # Create a colored mask with the same dimensions as the expert image
        colored_mask = np.zeros_like(output_expert_image, dtype=np.uint8)
        colored_mask[full_image_mask] = mask_color

        # Blend the colored mask with the expert image
        output_expert_image = cv2.addWeighted(output_expert_image, 1, colored_mask, 0.5, 0)

        # Predict keypoints for the person in the expert image
        person_image_preprocessed = preprocess_image_pose_estimation(
            expert_image[y1:y2, x1:x2])
        keypoints_pred_expert = pose_estimation_model.predict(
            np.expand_dims(person_image_preprocessed, axis=0))[0]
        keypoints_with_visibility_expert = np.array(keypoints_pred_expert).reshape(-1, 3)

        # Scale and translate keypoints back to the original expert image size
        scale_x_expert, scale_y_expert = (new_w_expert / 224.0, new_h_expert / 224.0)
        pad_x_expert, pad_y_expert = (left_pad_expert, top_pad_expert)

        keypoints_with_visibility_expert[:, 0] = (
            keypoints_with_visibility_expert[:, 0] * scale_x_expert) + x1 + pad_x_expert
        keypoints_with_visibility_expert[:, 1] = (
            keypoints_with_visibility_expert[:, 1] * scale_y_expert) + y1 + pad_y_expert

        # Draw keypoints and skeleton on the expert image
        for x, y, v in keypoints_with_visibility_expert:
            if v > 0:  # Visible keypoints
                cv2.circle(output_expert_image, (int(x), int(y)),
                           5, keypoint_color, -1)

        # Draw skeleton connections
        for start, end in skeleton:
            start_idx, end_idx = start - 1, end - 1
            if start_idx < len(keypoints_with_visibility_expert) and end_idx < len(keypoints_with_visibility_expert):
                start_point = keypoints_with_visibility_expert[start_idx]
                end_point = keypoints_with_visibility_expert[end_idx]
                if start_point[2] > 0 and end_point[2] > 0:
                    cv2.line(output_expert_image, (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])), skeleton_color, 2)
                             

angle_errors = compare_angles(keypoints_with_visibility_expert, keypoints_with_visibility)
for joint, error in angle_errors.items():
    print(f"Angle difference at {joint}: {error:.2f} degrees")

# Display the final image
cv2_imshow(output_image)
cv2_imshow(output_expert_image)
