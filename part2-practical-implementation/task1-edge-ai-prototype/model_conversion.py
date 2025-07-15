import tensorflow as tf
import numpy as np
import os
from pathlib import Path

class ModelConverter:
    """
    Convert TensorFlow models to TensorFlow Lite for edge deployment
    """
    
    def __init__(self, model_path):
        """
        Initialize converter with trained model
        
        Args:
            model_path: Path to saved TensorFlow model
        """
        self.model_path = model_path
        self.model = None
        self.tflite_model = None
        
    def load_model(self):
        """
        Load the trained TensorFlow model
        """
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def convert_to_tflite(self, optimization_level='default', quantize=True):
        """
        Convert TensorFlow model to TensorFlow Lite
        
        Args:
            optimization_level: 'default', 'optimize_for_size', or 'optimize_for_latency'
            quantize: Whether to apply quantization for further size reduction
        """
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        # Create TensorFlow Lite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        if optimization_level == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif optimization_level == 'optimize_for_size':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        elif optimization_level == 'optimize_for_latency':
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        
        # Apply quantization if requested
        if quantize:
            converter.target_spec.supported_types = [tf.float16]
            # For int8 quantization, you would need representative dataset
            # converter.representative_dataset = representative_data_gen
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        try:
            # Convert the model
            self.tflite_model = converter.convert()
            print("Model converted to TensorFlow Lite successfully!")
            return self.tflite_model
        except Exception as e:
            print(f"Error converting model: {e}")
            return None
    
    def save_tflite_model(self, output_path):
        """
        Save TensorFlow Lite model to file
        
        Args:
            output_path: Path to save the .tflite model
        """
        if self.tflite_model is None:
            print("No TensorFlow Lite model to save. Convert model first.")
            return False
        
        try:
            with open(output_path, 'wb') as f:
                f.write(self.tflite_model)
            print(f"TensorFlow Lite model saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def get_model_info(self, tflite_model_path):
        """
        Get information about the TensorFlow Lite model
        
        Args:
            tflite_model_path: Path to the .tflite model file
        """
        try:
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Get model size
            model_size = os.path.getsize(tflite_model_path)
            
            print("=== TensorFlow Lite Model Information ===")
            print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
            print(f"Input shape: {input_details[0]['shape']}")
            print(f"Input type: {input_details[0]['dtype']}")
            print(f"Output shape: {output_details[0]['shape']}")
            print(f"Output type: {output_details[0]['dtype']}")
            
            return {
                'model_size_mb': model_size / 1024 / 1024,
                'input_shape': input_details[0]['shape'],
                'input_type': input_details[0]['dtype'],
                'output_shape': output_details[0]['shape'],
                'output_type': output_details[0]['dtype']
            }
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def compare_models(self, original_model_path, tflite_model_path):
        """
        Compare original and TensorFlow Lite model sizes
        
        Args:
            original_model_path: Path to original TensorFlow model
            tflite_model_path: Path to TensorFlow Lite model
        """
        try:
            # Get file sizes
            original_size = os.path.getsize(original_model_path)
            tflite_size = os.path.getsize(tflite_model_path)
            
            # Calculate compression ratio
            compression_ratio = original_size / tflite_size
            size_reduction = (1 - tflite_size / original_size) * 100
            
            print("=== Model Comparison ===")
            print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
            print(f"TensorFlow Lite model size: {tflite_size / 1024 / 1024:.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            print(f"Size reduction: {size_reduction:.1f}%")
            
            return {
                'original_size_mb': original_size / 1024 / 1024,
                'tflite_size_mb': tflite_size / 1024 / 1024,
                'compression_ratio': compression_ratio,
                'size_reduction_percent': size_reduction
            }
        except Exception as e:
            print(f"Error comparing models: {e}")
            return None

class TFLiteInference:
    """
    Run inference with TensorFlow Lite model
    """
    
    def __init__(self, model_path):
        """
        Initialize TFLite inference
        
        Args:
            model_path: Path to .tflite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """
        Load TensorFlow Lite model
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("TensorFlow Lite model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            return False
    
    def predict(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: Preprocessed input data (numpy array)
        """
        if self.interpreter is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return output_data
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def predict_recyclable(self, image_path):
        """
        Predict recyclable type for an image
        
        Args:
            image_path: Path to image file
        """
        if self.interpreter is None:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0
            
            # Convert to float32 (TFLite requirement)
            img_array = img_array.astype(np.float32)
            
            # Run inference
            output = self.predict(img_array)
            
            if output is not None:
                # Get prediction results
                predicted_class = np.argmax(output, axis=1)[0]
                confidence = np.max(output)
                
                # Class labels
                class_labels = ['Glass', 'Metal', 'Paper', 'Plastic']
                
                return {
                    'predicted_class': class_labels[predicted_class],
                    'confidence': float(confidence),
                    'all_probabilities': dict(zip(class_labels, output[0]))
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Example conversion workflow
    print("=== TensorFlow Lite Model Conversion Example ===")
    
    # Note: Replace with actual model path
    model_path = "best_recyclable_model.h5"
    
    # Initialize converter
    converter = ModelConverter(model_path)
    
    # Load model (this would work with a real trained model)
    # converter.load_model()
    
    # Convert to TensorFlow Lite
    # tflite_model = converter.convert_to_tflite(quantize=True)
    
    # Save TensorFlow Lite model
    # converter.save_tflite_model("recyclable_model.tflite")
    
    # Get model information
    # converter.get_model_info("recyclable_model.tflite")
    
    # Compare models
    # converter.compare_models(model_path, "recyclable_model.tflite")
    
    print("\nConversion script ready!")
    print("Replace model_path with your actual trained model to run conversion.")
