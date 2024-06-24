from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('yolov8m')

# Faire des prédictions sur la vidéo
result = model.predict('input_videos/input_video.mp4', save=True)

# Initialiser une liste pour stocker les scores de confiance
confidence_scores = []

# Parcourir toutes les détections dans le premier résultat
for box in result[0].boxes:
    # Ajouter le score de confiance à la liste
    confidence_scores.append(box.confidence)

# Calculer la moyenne des scores de confiance
average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

# Afficher le résultat
print(f"Average prediction confidence: {average_confidence * 100:.2f}%")

# Afficher les détails des boîtes
print("Boxes:")
for box in result[0].boxes:
    print(box)
    print (box)
