
import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def calculate_score(distance, ring_width=10):
    for score in range(10, 0, -1):
        if distance <= score * ring_width:
            return score
    return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_blur = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=20)

    total_score = 0
    shots = 0
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        h, w = img.shape
        center_x, center_y = w // 2, h // 2
        for (x, y, r) in circles:
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            score = calculate_score(distance)
            total_score += score
            shots += 1
            cv2.circle(img_color, (x, y), r, (0, 0, 255), 2)
            cv2.putText(img_color, str(score), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img_color)

    return jsonify({
        'total_score': total_score,
        'shots': shots,
        'image_url': f'/results/{result_filename}'
    })

@app.route('/results/<filename>')
def result_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
