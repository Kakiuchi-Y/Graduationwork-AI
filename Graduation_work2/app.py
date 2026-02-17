import os
from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import tempfile

app = Flask(__name__)

# モデルのロード
model = tf.keras.models.load_model('video_model.h5')

# メインページを表示
@app.route('/')
def index():
    return render_template('index.html')  # index.html を返す

# 動画からフレームを抽出する関数
def process_video(file):
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        file.save(temp_file.name)  # ファイルを一時的に保存
        temp_file_path = temp_file.name
    
    # VideoCaptureにファイルパスを渡す
    cap = cv2.VideoCapture(temp_file_path)

    if not cap.isOpened():
        raise ValueError("動画を開けませんでした。")

    frames = []

    # 動画の最初の100フレームを取得
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break

        # フレームを64x64にリサイズ
        frame = cv2.resize(frame, (64, 64))  # モデルに合わせたサイズに変更
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケールに変換
        frame = frame / 255.0  # 正規化
        frames.append(frame)

    cap.release()

    # 一時ファイルを削除
    os.remove(temp_file_path)

    # フレームを(100, 64, 64, 1)の形状に変換
    frames = np.array(frames)  # (100, 64, 64)の形状
    frames = np.expand_dims(frames, axis=-1)  # チャンネル次元を追加して(100, 64, 64, 1)に

    # (1, 100, 64, 64, 1) の形にするためにバッチ次元を追加
    frames = np.expand_dims(frames, axis=0)

    return frames

# 推論エンドポイント
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            raise ValueError("File not found in the request")
        
        file = request.files['file']
        
        # ファイルが画像または動画であることを確認
        if not file.content_type.startswith(('image', 'video')):
            return jsonify({'error': 'Uploaded file is not an image or video'}), 400
        
        # 動画の場合、フレームを処理
        if file.content_type.startswith('video'):
            frames = process_video(file)
            img = frames  # フレームを3D入力として処理
        
        # 画像の場合、画像を処理
        elif file.content_type.startswith('image'):
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((64, 64))  # 64x64にリサイズ
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=-1)  # グレースケール
            img = np.expand_dims(img, axis=0)  # バッチサイズの次元を追加
        
        # モデルによる予測
        prediction = model.predict(img)
        emotion = decode_emotion(prediction)
        
        return jsonify({'emotion': emotion})
    
    except Exception as e:
        print(f"Error: {str(e)}")  # エラーの詳細をコンソールに表示
        return jsonify({'error': str(e)}), 400

# 推論結果を感情にマッピングする関数
def decode_emotion(prediction):
    emotions = ['Happy', 'Sad', 'hate', 'Surprised', 'tired']  # 感情のラベル
    return emotions[np.argmax(prediction)]  # 最も高いスコアの感情を返す

if __name__ == '__main__':
    app.run(debug=True)
