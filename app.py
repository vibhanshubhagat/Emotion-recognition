#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from deepface import DeepFace
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/post_json', methods=['POST'])
def process_json():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read the image using OpenCV
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform emotion analysis using DeepFace
        results = DeepFace.analyze(image, actions=['emotion'])

        # Get the top two emotions with values greater than or equal to 10
        emotions = results[0]['emotion']
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [emotion.capitalize() for emotion, value in sorted_emotions if value >= 10][:2]

        dominant_emotion = results[0]['dominant_emotion']
        if dominant_emotion == 'fear':
            top_emotions = ['happy']

        print(top_emotions)  # Add this line to print the emotions

        return jsonify({'emotions': top_emotions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5009)


# 

# In[ ]:




