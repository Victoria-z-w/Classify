from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载模型
model = torch.jit.load('ClassifyModel.pt')
model.eval()
if torch.cuda.is_available():
    model.cuda()

# 预处理流水线
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': '请选择有效的图片文件'}), 400

        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': '无法读取图片'}), 400

        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.width == 0 or img.height == 0:
                return jsonify({'error': '图片尺寸无效'}), 400
            image = img.convert('RGB')

        tensor = preprocess(image).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, 1)

        # 提前保存标量结果
        class_idx_value = class_idx.item()
        confidence_value = confidence.item()

        class_names = [
            "complex", "frog_eye_leaf_spot", "frog_eye_leaf_spot-complex", "healthy",
            "powdery_mildew", "powdery_mildew-complex", "rust", "rust-complex",
            "rust-frog_eye_leaf_spot", "scab", "scab-frog_eye_leaf_spot",
            "scab-frog_eye_leaf_spot-complex"
        ]

        return jsonify({
            'class': class_names[class_idx_value],
            'confidence': round(float(confidence_value), 4)
        })

    except Exception as e:
        error_msg = f"预测失败: {str(e)}"
        print(f"错误详细信息: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

    finally:
        # 释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 删除非必要变量
        try:
            del tensor, outputs, probabilities
        except NameError:
            pass

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)