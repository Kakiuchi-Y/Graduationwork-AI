from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# モデルのロード
model = tf.keras.models.load_model('video_model.h5')

# メインページを表示
@app.route('/')
def index():
    return render_template('index.html')  # index.htmlを返す

# 推論エンドポイント
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # フロントエンドから受け取ったデータ
        input_data = np.array(data['inputs']).reshape(1, -1)  # 必要に応じて入力形状を調整
        prediction = model.predict(input_data)
        emotion = decode_emotion(prediction)  # 推論結果を感情にマッピング
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 推論結果を感情にマッピングする関数
def decode_emotion(prediction):
    emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']  # 感情のラベル
    return emotions[np.argmax(prediction)]  # 最も高いスコアの感情を返す

if __name__ == '__main__':
    app.run(debug=True)
