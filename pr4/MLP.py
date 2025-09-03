# Enhanced Multi-Layer Perceptron (MLP) in TensorFlow
# Dataset: MNIST Handwritten Digits
# Enhanced with advanced features: regularization, callbacks, multiple optimizers

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load and preprocess dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data (0-255 → 0-1) with improved preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add noise reduction (optional)
# x_train = tf.image.resize(x_train[..., tf.newaxis], [28, 28]).numpy().squeeze()
# x_test = tf.image.resize(x_test[..., tf.newaxis], [28, 28]).numpy().squeeze()

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# 2. Define multiple MLP architectures
def create_basic_mlp():
    """Basic MLP architecture"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def create_enhanced_mlp():
    """Enhanced MLP with regularization and batch normalization"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        
        # First hidden layer with regularization
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second hidden layer
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Third hidden layer
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer
        Dense(10, activation='softmax')
    ])
    return model

def create_deep_mlp():
    """Deep MLP with more layers"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        
        # Layer 1
        Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Layer 2
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Layer 4
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer
        Dense(10, activation='softmax')
    ])
    return model

# 3. Choose model architecture
print("\nAvailable MLP architectures:")
print("1. Basic MLP (2 hidden layers)")
print("2. Enhanced MLP (3 hidden layers with regularization)")
print("3. Deep MLP (4 hidden layers with advanced regularization)")

# For demonstration, we'll use the enhanced MLP
model = create_enhanced_mlp()

# 4. Define multiple optimizers
optimizers = {
    'adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    'sgd': SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    'rmsprop': RMSprop(learning_rate=0.001, rho=0.9)
}

# Choose optimizer (Adam is generally best for MLPs)
chosen_optimizer = 'adam'
print(f"\nUsing optimizer: {chosen_optimizer}")

# 5. Compile model with advanced metrics
model.compile(
    optimizer=optimizers[chosen_optimizer],
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# 6. Define advanced callbacks
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_mlp_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 7. Advanced training with validation split
print("\nStarting training with enhanced features...")
print("Note: Training with reduced epochs for demonstration purposes")
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,  # Reduced epochs for faster demonstration
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 8. Comprehensive model evaluation
print("\n" + "="*50)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*50)

# Basic evaluation
test_loss, test_acc, test_top_k_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Top-5 Accuracy: {test_top_k_acc*100:.2f}%")

# Detailed predictions for classification report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# 9. Advanced visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training history - Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training history - Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

# Sample predictions
sample_indices = np.random.choice(len(x_test), 9, replace=False)
for i, idx in enumerate(sample_indices[:9]):
    if i < 9:  # Only show first 9
        row = i // 3
        col = i % 3
        if i == 0:  # Use the remaining subplot for sample predictions
            axes[1, 1].remove()
            # Create a new subplot grid for sample predictions
            gs = fig.add_gridspec(3, 3, left=0.55, right=0.95, top=0.45, bottom=0.05)
            for j, sample_idx in enumerate(sample_indices):
                ax = fig.add_subplot(gs[j//3, j%3])
                ax.imshow(x_test[sample_idx], cmap='gray')
                pred_label = y_pred_classes[sample_idx]
                true_label = y_true_classes[sample_idx]
                confidence = np.max(y_pred[sample_idx]) * 100
                color = 'green' if pred_label == true_label else 'red'
                ax.set_title(f'P:{pred_label} T:{true_label}\n{confidence:.1f}%', 
                           color=color, fontsize=8)
                ax.axis('off')
            break

plt.tight_layout()
plt.show()

# 10. Advanced prediction analysis
print("\n" + "="*50)
print("PREDICTION ANALYSIS")
print("="*50)

# Analyze prediction confidence
confidences = np.max(y_pred, axis=1)
print(f"\nPrediction Confidence Statistics:")
print(f"Mean confidence: {np.mean(confidences)*100:.2f}%")
print(f"Median confidence: {np.median(confidences)*100:.2f}%")
print(f"Min confidence: {np.min(confidences)*100:.2f}%")
print(f"Max confidence: {np.max(confidences)*100:.2f}%")

# Find most and least confident predictions
most_confident_idx = np.argmax(confidences)
least_confident_idx = np.argmin(confidences)

print(f"\nMost confident prediction:")
print(f"Index: {most_confident_idx}, Predicted: {y_pred_classes[most_confident_idx]}, "
      f"Actual: {y_true_classes[most_confident_idx]}, Confidence: {confidences[most_confident_idx]*100:.2f}%")

print(f"\nLeast confident prediction:")
print(f"Index: {least_confident_idx}, Predicted: {y_pred_classes[least_confident_idx]}, "
      f"Actual: {y_true_classes[least_confident_idx]}, Confidence: {confidences[least_confident_idx]*100:.2f}%")

# 11. Model comparison function
def compare_models():
    """Compare different MLP architectures"""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*50)
    
    models = {
        'Basic MLP': create_basic_mlp(),
        'Enhanced MLP': create_enhanced_mlp(),
        'Deep MLP': create_deep_mlp()
    }
    
    results = {}
    
    for name, model_arch in models.items():
        print(f"\nTraining {name}...")
        model_arch.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Quick training for comparison
        history = model_arch.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=10,
            batch_size=128,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model_arch.evaluate(x_test, y_test, verbose=0)
        results[name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'parameters': model_arch.count_params()
        }
        
        print(f"{name} - Accuracy: {test_acc*100:.2f}%, Parameters: {model_arch.count_params():,}")
    
    return results

# Uncomment to run model comparison
# comparison_results = compare_models()

print("\n" + "="*50)
print("ENHANCED MLP TRAINING COMPLETED!")
print("="*50)
print(f"Final Model Accuracy: {test_acc*100:.2f}%")
print(f"Model saved as: best_mlp_model.h5")
print("\nEnhancements implemented:")
print("✓ Multiple architecture options")
print("✓ Advanced regularization (L1, L2, Dropout)")
print("✓ Batch normalization")
print("✓ Multiple optimizer options")
print("✓ Advanced callbacks (Early stopping, LR reduction)")
print("✓ Comprehensive evaluation metrics")
print("✓ Advanced visualization")
print("✓ Prediction confidence analysis")
print("✓ Model comparison framework")
