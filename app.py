import os
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XRayClassifier")



app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MODEL_PATH': 'best_model.h5',
    'MAX_CONTENT_LENGTH': 5 * 1024 * 1024
})

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class ModelLoader:
    """模型加载器（兼容性优化版）"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.custom_objects = {
            'Adam': tf.keras.optimizers.legacy.Adam,
            'Optimizer': tf.keras.optimizers.legacy.Optimizer
        }

    def load(self):
        try:
            model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=self.custom_objects,
                compile=False
            )
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise


def predict_image(img_path):
    try:
        # 加载模型
        model = ModelLoader(app.config['MODEL_PATH']).load()

        # 预处理图像
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 预测
        prediction = model.predict(img_array, verbose=0)[0][0]
        class_idx = 1 if prediction > 0.5 else 0

        return {
            'status': 'success',
            'class': 'PNEUMONIA' if class_idx == 1 else 'NORMAL',
            'confidence': round(float(prediction if class_idx else 1 - prediction), 4),
            'prediction_score': round(float(prediction), 4),  # 统一键名
            'device': 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
        }
    except Exception as e:
        logger.exception("预测异常")
        return {
            'status': 'error',
            'message': f'系统错误: {str(e)}'
        }


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', message='未选择文件')

        file = request.files['file']
        if not file or file.filename == '':
            return render_template('error.html', message='无效文件')



        try:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            result = predict_image(save_path)

            if result['status'] == 'error':
                return render_template('error.html', message=result['message'])

            return render_template('result.html',
                                   image_path=f"uploads/{filename}",
                                   result=result)
        except Exception as e:
            return render_template('error.html', message=f'文件处理错误: {str(e)}')

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)