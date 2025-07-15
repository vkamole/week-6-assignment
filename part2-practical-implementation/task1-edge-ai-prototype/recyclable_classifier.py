import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class RecyclableClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize the recyclable item classifier
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of recyclable categories (plastic, paper, glass, metal)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_model(self):
        """
        Create a lightweight CNN model optimized for edge deployment
        Uses MobileNetV2 as base with custom classification head
        """
        # Use MobileNetV2 as backbone (pre-trained on ImageNet)
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers for transfer learning
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, train_dir, val_dir, batch_size=32):
        """
        Prepare training and validation data with augmentation
        
        Args:
            train_dir: Directory containing training images
            val_dir: Directory containing validation images
            batch_size: Batch size for training
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=10):
        """
        Train the model with early stopping and model checkpointing
        """
        if self.model is None:
            raise ValueError("Model not created and compiled.")
        
        # Callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_recyclable_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def predict_image(self, image_path):
        """
        Predict recyclable type for a single image
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape[:2]
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # Class labels
        class_labels = ['Glass', 'Metal', 'Paper', 'Plastic']
        
        return {
            'predicted_class': class_labels[predicted_class],
            'confidence': confidence,
            'all_probabilities': dict(zip(class_labels, predictions[0]))
        }
    
    def plot_training_history(self):
        """
        Plot training history (accuracy and loss)
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = RecyclableClassifier()
    
    # Create and compile model
    model = classifier.create_model()
    classifier.compile_model()
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Note: In a real implementation, you would:
    # 1. Prepare your dataset with proper directory structure
    # 2. Train the model
    # 3. Convert to TensorFlow Lite
    # 4. Deploy on edge device
    
    print("\nModel created successfully!")
    print("Ready for training with recyclable item dataset.")
