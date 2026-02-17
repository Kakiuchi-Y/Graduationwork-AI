document.addEventListener('DOMContentLoaded', () => {
    startCamera();
});

// カメラを起動し、録画を開始する関数
async function startCamera() {
    const video = document.getElementById('video');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // 5秒ごとに録画して送信
        startRecording(stream);
    } catch (error) {
        console.error('カメラを起動できません:', error);
    }
}

// MediaRecorderを使って動画を録画し、5秒ごとにサーバーに送信
function startRecording(stream) {
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    let chunks = [];

    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };

    recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        chunks = []; // 次回の録画のためにリセット

        sendVideo(blob);
        recorder.start();
        
        // 5秒ごとに録画停止→送信
        setTimeout(() => recorder.stop(), 5000);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 5000); // 最初の録画開始
}

// 録画した動画をサーバーに送信する関数
async function sendVideo(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'video.webm');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to fetch');
        }

        const data = await response.json();
        document.getElementById('emotion').textContent = `予測された感情: ${data.emotion}`;
    } catch (error) {
        console.error('予測エラー:', error);
    }
}
