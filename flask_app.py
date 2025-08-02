from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from utils import preprocess_image, predict_mask

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/seifeldenelmizayen/Desktop/Cellula_Computer Vision/Fifth week/uploads'
OUTPUT_FOLDER = '/Users/seifeldenelmizayen/Desktop/Cellula_Computer Vision/Fifth week/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(filepath)

            # Predict mask
            mask = predict_mask(filepath)

            # Save the mask to outputs/
            output_path = os.path.join(OUTPUT_FOLDER, f"mask_{filename}.png")
            plt.imsave(output_path, mask, cmap='gray')

            return render_template('home.html', output_path=output_path)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
