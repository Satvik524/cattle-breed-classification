import numpy as np
from PIL import Image
import tensorflow as tf


CLASS_NAMES = [
    "Amritmahal",
    "Ayrshire",
    "Bachaur",
    "Badri",
    "Bargur",
    "Bhelai",
    "Dagri",
    "Dangi",
    "Deoni",
    "Gangatari",
    "Gaolao",
    "Ghumsari",
    "Gir",
    "Hallikar",
    "Hariana",
    "Himachali_pahari",
    "Kangayam",
    "Kankrej",
    "Kenkatha",
    "Khariar",
    "Kherigarh",
    "Khillari",
    "Konkan_kapila",
    "Kosali",
    "Krishna_valley",
    "Ladakhi",
    "Lakhimi",
    "Malnad_gidda",
    "Malvi",
    "Mewati",
    "Motu",
    "Nagori",
    "Nari",
    "Nimari",
    "Ongole",
    "Poda_thirupu",
    "Ponwar",
    "Pulikulam",
    "Punganur",
    "Purnea",
    "Rathi",
    "Red_kandhari",
    "Red_sindhi",
    "Sahiwal",
    "Shweta_Kapila",
    "Siri",
    "Tharparkar",
    "Thutho",
    "Umblachery",
    "Vechur",
]


def crop_cow(image, yolo_model):
    results = yolo_model.predict(
        source=np.array(image),
        verbose=False,
        classes=[19],
        conf=0.30
    )

    res = results[0]

    print("Number of detections:", len(res.boxes))

    for i, box in enumerate(res.boxes):
        print(
            i,
            "class:", int(box.cls.item()),
            "confidence:", box.conf.item(),
            "box:", box.xyxy[0].cpu().numpy()
        )

    if len(res.boxes) == 0:
        return None

    best_idx = res.boxes.conf.argmax().item()
    best_cow_box = res.boxes.xyxy[best_idx].cpu().numpy()

    img = res.orig_img

    x1, y1, x2, y2 = map(int, best_cow_box)

    h, w = img.shape[:2]

    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped_img = img[y1:y2, x1:x2]

    return Image.fromarray(cropped_img)


def preprocessing_image(image):
    target_size = (224, 224)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)

    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    image_arr = np.expand_dims(image_arr, axis=0)

    return image_arr


def predict_breed(image, yolo_model, cattle_model):
    cropped_image = crop_cow(image, yolo_model)

    if cropped_image is None:
        return None

    processed_image = preprocessing_image(cropped_image)

    prediction = cattle_model.predict(processed_image, verbose=0)

    probabilities = prediction[0]

    best_index = np.argmax(probabilities)

    predicted_breed = CLASS_NAMES[best_index]

    confidence = float(probabilities[best_index])

    top_k_indices = np.argsort(probabilities)[-3:][::-1]

    top_predictions = []

    for idx in top_k_indices:
        top_predictions.append(
            {
                "breed": CLASS_NAMES[idx],
                "confidence": float(probabilities[idx])
            }
        )

    return {
        "predicted_breed": predicted_breed,
        "confidence": confidence,
        "top_predictions": top_predictions
    }

