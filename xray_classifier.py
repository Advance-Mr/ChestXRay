# xray_classifier.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# 配置参数
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def prepare_data(dataset_dir='chest_xray'):
    # 计算类别权重
    pneumonia_count = 1528  # 根据你的数据集实际数量
    normal_count = 1349  # 根据你的数据集实际数量
    total = pneumonia_count + normal_count
    class_weights = {
        0: total / (2 * normal_count),  # NORMAL权重
        1: total / (2 * pneumonia_count)  # PNEUMONIA权重
    }

    # 数据增强配置
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 数据流
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, test_generator, class_weights


def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history):
    # 创建可视化画布
    plt.figure(figsize=(18, 6))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves', fontsize=14)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves', fontsize=14)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)

    # 保存和显示
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()


def evaluate_model(model, test_generator):
    # 模型评估
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()


def main():
    # 数据准备
    train_gen, test_gen, class_weights = prepare_data()

    # 模型构建
    model = build_model()
    model.summary()

    # 回调设置
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # 训练模型
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_gen,
        validation_steps=test_gen.samples // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[checkpoint]
    )

    # 可视化训练过程
    plot_history(history)

    # 加载最佳模型并评估
    model.load_weights('best_model.h5')
    evaluate_model(model, test_gen)


if __name__ == "__main__":
    main()