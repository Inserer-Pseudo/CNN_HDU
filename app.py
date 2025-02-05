import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Répertoire pour stocker temporairement les images uploadées
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle CNN entraîné pour reconnaître les gestes
model = load_model("rock_paper_scissors_cnn.h5")
classes = ['Pierre', 'Feuille', 'Ciseau']

# Essai d'ouvrir la caméra
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("[WARN] La caméra n'est pas accessible.")

def process_image(img, filename):
    """prépare l'image pour le modèle."""
    #resized = cv2.resize(img, (150, 150))
    
    # Sauvegarde de l'image redimensionnée
    #cv2.imwrite(filename, resized)

    # Make image color values to be float.
    resized = tf.cast(img, tf.float32)
    # Make image color values to be in [0..1] range.
    resized = resized / 255.
    # Make sure that image has a right size
    resized = tf.image.resize(resized, [150, 150])
    
    #return resized.astype(np.float32) / 255.0
    return resized

def predict_gesture(img):
    """Prédit le geste basé sur l'image traitée."""
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

def determine_winner(gesture1, gesture2):
    """Détermine le gagnant."""
    if gesture1 == gesture2:
        return "Égalité"
    elif (gesture1 == "Pierre" and gesture2 == "Ciseau") or \
         (gesture1 == "Feuille" and gesture2 == "Pierre") or \
         (gesture1 == "Ciseau" and gesture2 == "Feuille"):
        return "Joueur Gauche gagne"
    else:
        return "Joueur Droit gagne"

@app.route('/')
def index():
    """Affiche la page principale."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Renvoie le flux vidéo."""
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                height, width, _ = frame.shape
                cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'La capture a échoué.'}), 500

    height, width, _ = frame.shape
    mid = width // 2
    img_left, img_right = frame[:, :mid], frame[:, mid:]
    
    left_filename = 'savedImage_left.jpg'
    right_filename = 'savedImage_right.jpg'

    left_gesture = predict_gesture(process_image(img_left, left_filename))
    right_gesture = predict_gesture(process_image(img_right, right_filename))
    winner = determine_winner(left_gesture, right_gesture)

    return jsonify({'left_gesture': left_gesture, 'right_gesture': right_gesture, 'winner': winner})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == "":
        return redirect(url_for('index'))
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': "L'image uploadée est invalide."}), 400

        height, width, _ = img.shape
        mid = width // 2
        img_left, img_right = img[:, :mid], img[:, mid:]
        
        left_filename = 'savedImage_left.jpg'
        right_filename = 'savedImage_right.jpg'

        left_gesture = predict_gesture(process_image(img_left, left_filename))
        right_gesture = predict_gesture(process_image(img_right, right_filename))
        winner = determine_winner(left_gesture, right_gesture)

        os.remove(filepath)

        return jsonify({'left_gesture': left_gesture, 'right_gesture': right_gesture, 'winner': winner})

if __name__ == '__main__':
    app.run(debug=True)
