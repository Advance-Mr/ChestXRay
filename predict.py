# predict.py（增强交互版）
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def predict_image(model_path, img_path, img_size=(224, 224)):
    # 加载模型
    model = load_model(model_path)

    # 加载并预处理图像
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 预测
    prediction = model.predict(img_array)[0][0]
    class_idx = 1 if prediction > 0.5 else 0
    confidence = prediction if class_idx == 1 else 1 - prediction

    return {
        'class': 'PNEUMONIA' if class_idx == 1 else 'NORMAL',
        'confidence': float(confidence),
        'prediction_score': float(prediction)
    }


def select_files():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 选择模型文件
    model_path = filedialog.askopenfilename(
        title="选择模型文件",
        filetypes=[("Keras Model", "*.h5 *.keras"), ("All Files", "*.*")]
    )

    if not model_path:
        print("未选择模型文件")
        return

    # 选择图片文件
    img_path = filedialog.askopenfilename(
        title="选择X光片",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
    )

    if not img_path:
        print("未选择图片文件")
        return

    return model_path, img_path


if __name__ == "__main__":
    import sys

    # 命令行模式
    if len(sys.argv) == 3:
        result = predict_image(sys.argv[1], sys.argv[2])

    # 交互模式
    else:
        print("\n请通过文件选择对话框进行操作：")
        files = select_files()
        if not files:
            sys.exit()

        model_path, img_path = files
        result = predict_image(model_path, img_path)

    # 输出结果
    print("\n预测结果:")
    print(f"图片路径: {img_path}")
    print(f"诊断类别: {result['class']}")
    print(f"置信度: {result['confidence']:.2%}")
    print(f"原始评分: {result['prediction_score']:.4f}")