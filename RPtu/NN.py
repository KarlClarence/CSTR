import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_and_preprocess_data(filepath):
    """加载并预处理数据"""
    df = pd.read_csv(filepath)

    # 输入特征：初始状态+操作参数
    X = df[['k0', 'delta_h', 'Q', 'cA0_input', 'cA_init', 'T_init']].values

    # 输出目标：最终状态
    y = df[['cA_final', 'T_final']].values

    # 数据标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    return X_scaled, y_scaled, x_scaler, y_scaler


def build_model(input_dim, output_dim):
    """构建神经网络模型"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_dim)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    return model


def plot_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def evaluate_model(model, X_test, y_test, y_scaler):
    """评估模型性能"""
    # 预测并反标准化
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # 计算误差
    cA_error = np.abs(y_true[:, 0] - y_pred[:, 0])
    T_error = np.abs(y_true[:, 1] - y_pred[:, 1])

    print("\n模型评估结果:")
    print(f"浓度预测平均绝对误差: {np.mean(cA_error):.4f} kmol/m³")
    print(f"温度预测平均绝对误差: {np.mean(T_error):.2f} K")

    # 绘制预测 vs 真实值
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3)
    plt.plot([min(y_true[:, 0]), max(y_true[:, 0])],
             [min(y_true[:, 0]), max(y_true[:, 0])], 'r--')
    plt.xlabel('真实浓度')
    plt.ylabel('预测浓度')
    plt.title('浓度预测')

    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3)
    plt.plot([min(y_true[:, 1]), max(y_true[:, 1])],
             [min(y_true[:, 1]), max(y_true[:, 1])], 'r--')
    plt.xlabel('真实温度 (K)')
    plt.ylabel('预测温度 (K)')
    plt.title('温度预测')

    plt.tight_layout()
    plt.savefig('predictions_vs_actuals.png')
    plt.show()


if __name__ == "__main__":
    # 加载数据
    X, y, x_scaler, y_scaler = load_and_preprocess_data('data/cstr_dataset.csv')

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建模型
    model = build_model(X.shape[1], y.shape[1])

    # 训练模型
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=100,
                        batch_size=64,
                        verbose=1)

    # 保存模型
    model.save('cstr_nn_model.h5')
    print("模型已保存为 cstr_nn_model.h5")

    # 可视化训练过程
    plot_history(history)

    # 评估模型
    evaluate_model(model, X_test, y_test, y_scaler)