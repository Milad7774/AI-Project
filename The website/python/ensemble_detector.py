# ======================================================================
# FAKE IMAGE DETECTION ENSEMBLE SYSTEM
# ======================================================================

# IMPORTANT: These imports MUST be at the VERY TOP to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Now continue with other imports
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os
import datetime
import sys

print("=" * 80)
print("🛡️  ADVANCED FAKE IMAGE DETECTION ENSEMBLE SYSTEM")
print("=" * 80)

# Create main results directory
RESULTS_DIR = "ensemble_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"📁 Results will be saved to: {RESULTS_DIR}")

# =============================
#        GPU CONFIGURATION
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Using device: {device}")

# =============================
#        MODEL 1: JPEG DCT MODEL
# =============================
class JPEGDetector:
    """JPEG DCT-based detector"""
    def __init__(self, model_path="model_complete_gpu.pth", scaler_path="scaler_gpu.pkl"):
        self.device = device
        self.name = "JPEG-DCT"
        
        # Load scaler
        print(f"📦 Loading {self.name} model...")
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"   ✅ Scaler loaded from {scaler_path}")
        except:
            print(f"   ⚠️ Scaler not found, using default scaling")
            self.scaler = None
        
        # Load model architecture
        class ImageDetectorNN(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return torch.sigmoid(self.network(x))
        
        # Load model checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = ImageDetectorNN(checkpoint['input_size']).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"   ✅ Model loaded from {model_path}")
            print(f"   📊 Original accuracy: {checkpoint.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error loading JPEG model: {e}")
            self.model = None
    
    def extract_features(self, image):
        """Extract DCT features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize and apply DCT
        img_resized = cv2.resize(gray, (128, 128))
        img_float = np.float32(img_resized) / 255.0
        dct = cv2.dct(img_float)
        
        # Extract first 16x16 coefficients
        features = dct[:16, :16].flatten()  # 256 features
        return features
    
    def predict(self, image_path):
        """Predict single image"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Load and process image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": "Cannot read image"}
            
            # Extract features
            features = self.extract_features(img)
            
            # Scale features if scaler available
            if self.scaler:
                features = self.scaler.transform([features])
            else:
                features = np.array([features])
            
            # Predict
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                probability = self.model(features_tensor).item()
            
            # Convert to prediction
            is_fake = probability > 0.5
            confidence = probability if is_fake else 1 - probability
            
            return {
                "model": self.name,
                "is_fake": bool(is_fake),
                "fake_probability": float(probability),
                "real_probability": float(1 - probability),
                "confidence": float(confidence),
                "decision": "FAKE" if is_fake else "REAL"
            }
        
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# =============================
#        MODEL 2: EfficientNet B0
# =============================
class EfficientNetDetector:
    """EfficientNet B0 detector"""
    def __init__(self, model_path="no_fft_model.pth"):
        self.device = device
        self.name = "EfficientNet-B0"
        self.is_loaded = False
        
        print(f"📦 Loading {self.name} model...")
        
        try:
            import timm
            
            # Create architecture
            class EfficientNetClassifier(nn.Module):
                def __init__(self, num_classes=2):
                    super().__init__()
                    # Backbone (EfficientNet B0 from timm)
                    self.backbone = timm.create_model("efficientnet_b0", 
                                                     pretrained=False,
                                                     num_classes=0)
                    
                    # Classifier
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.4),
                        nn.Linear(1280, 512),
                        nn.ReLU(),
                        nn.BatchNorm1d(512),
                        nn.Dropout(0.3),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, x):
                    features = self.backbone(x)
                    return self.classifier(features)
            
            # Create model
            self.model = EfficientNetClassifier(num_classes=2).to(self.device)
            
            # Load weights
            print(f"   📂 Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)
            self.model.eval()
            self.is_loaded = True
            
            # Transformations
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            print(f"   ✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"   ❌ Error loading EfficientNet: {e}")
            print(f"   🔍 Check if 'timm' is installed: pip install timm")
            self.model = None
    
    def predict(self, image_path):
        """Predict single image"""
        if not self.is_loaded or self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                fake_prob = probabilities[0, 0].item()
                real_prob = probabilities[0, 1].item()
            
            # Decision
            is_fake = fake_prob > 0.5
            decision = "FAKE" if is_fake else "REAL"
            confidence = fake_prob if is_fake else real_prob
            
            return {
                "model": self.name,
                "is_fake": bool(is_fake),
                "fake_probability": float(fake_prob),
                "real_probability": float(real_prob),
                "confidence": float(confidence),
                "decision": decision
            }
        
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# =============================
#        MODEL 3: XGBoost Noise Detector
# =============================
class XGBoostNoiseDetector:
    """XGBoost-based noise pattern detector"""
    def __init__(self, model_path="xgb_noise_model.pkl", scaler_path="xgb_scaler.pkl", csv_path=None):
        self.device = device
        self.name = "XGBoost-Noise"
        
        print(f"📦 Loading {self.name} model...")
        
        try:
            import xgboost as xgb
            
            # Load XGBoost model
            self.model = joblib.load(model_path)
            print(f"   ✅ XGBoost model loaded from {model_path}")
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            print(f"   ✅ Scaler loaded from {scaler_path}")
            
            # Load reference data from CSV if provided
            self.reference_data = {}
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    self.reference_data[row['image']] = {
                        'fake_prob': row['xgb_fake_probability'],
                        'true_label': row['true_label']
                    }
                print(f"   ✅ Loaded {len(self.reference_data)} reference samples from CSV")
            
            print(f"   ✅ XGBoost noise detector initialized")
            self.is_loaded = True
            
        except Exception as e:
            print(f"   ❌ Error loading XGBoost model: {e}")
            print(f"   🔍 Install xgboost: pip install xgboost")
            self.model = None
            self.scaler = None
            self.is_loaded = False
    
    def extract_noise_features(self, image):
        """Extract noise-based features from image"""
        features = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Basic statistics
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.var(gray))
        features.append(np.median(gray))
        
        # 2. Noise analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        
        features.append(np.mean(noise))
        features.append(np.std(noise))
        features.append(np.var(noise))
        features.append(np.max(noise))
        
        # 3. Frequency domain analysis
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))
        features.append(np.max(magnitude))
        
        # 4. Edge features
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.sum(edges > 0) / edges.size)
        
        # 5. Local binary patterns
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 6. Color features if color image
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            for channel in range(3):
                features.append(np.mean(hsv[:, :, channel]))
                features.append(np.std(hsv[:, :, channel]))
                features.append(np.mean(lab[:, :, channel]))
                features.append(np.std(lab[:, :, channel]))
        
        # Ensure we have enough features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def _extract_lbp_features(self, gray, radius=1, n_points=8):
        """Extract Local Binary Pattern features"""
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            n_bins = 10
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            return hist.tolist()
        except:
            return [0.0] * 10
    
    def predict(self, image_path):
        """Predict single image using XGBoost model"""
        if not self.is_loaded or self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Check if image is in reference data
            img_name = os.path.basename(image_path)
            if img_name in self.reference_data:
                fake_probability = self.reference_data[img_name]['fake_prob']
                print(f"   📊 Using reference data for {img_name}")
            else:
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    return {"error": "Cannot read image"}
                
                # Extract features
                features = self.extract_noise_features(img)
                
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Predict with XGBoost
                fake_probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Convert to prediction
            is_fake = fake_probability > 0.5
            confidence = fake_probability if is_fake else 1 - fake_probability
            
            return {
                "model": self.name,
                "is_fake": bool(is_fake),
                "fake_probability": float(fake_probability),
                "real_probability": float(1 - fake_probability),
                "confidence": float(confidence),
                "decision": "FAKE" if is_fake else "REAL"
            }
        
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            # Fallback to simple noise analysis
            try:
                img = cv2.imread(image_path)
                if img is None:
                    return {"error": f"Prediction error: {str(e)}"}
                
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = cv2.absdiff(gray, blurred)
                noise_variance = np.var(noise)
                
                fake_probability = min(1.0, noise_variance / 5000)
                
                is_fake = fake_probability > 0.6
                confidence = fake_probability if is_fake else 1 - fake_probability
                
                return {
                    "model": self.name + " (fallback)",
                    "is_fake": bool(is_fake),
                    "fake_probability": float(fake_probability),
                    "real_probability": float(1 - fake_probability),
                    "confidence": float(confidence),
                    "decision": "FAKE" if is_fake else "REAL"
                }
            except:
                return {"error": f"Prediction error: {str(e)}"}

# =============================
#        ENSEMBLE MANAGER
# =============================
class EnsembleDetector:
    """Main ensemble that combines all models"""
    def __init__(self, jpeg_model_path="model_complete_gpu.pth",
                 jpeg_scaler_path="scaler_gpu.pkl",
                 efficientnet_path="no_fft_model.pth",
                 xgb_model_path="xgb_noise_model.pkl",
                 xgb_scaler_path="xgb_scaler.pkl",
                 noise_csv_path=None,
                 results_dir=RESULTS_DIR):
        
        print("\n" + "="*80)
        print("🤝 INITIALIZING ENSEMBLE DETECTOR")
        print("="*80)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(results_dir, f"run_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Initialize all detectors
        self.detectors = {
            "JPEG": JPEGDetector(jpeg_model_path, jpeg_scaler_path),
            "EfficientNet": EfficientNetDetector(efficientnet_path),
            "XGBoost-Noise": XGBoostNoiseDetector(xgb_model_path, xgb_scaler_path, noise_csv_path)
        }
        
        # Model weights
        self.weights = {
            "JPEG": 0.35,
            "EfficientNet": 0.45,
            "XGBoost-Noise": 0.20
        }
        
        # Track active models
        self.active_models = []
        for name, detector in self.detectors.items():
            if name == "JPEG" and detector.model is not None:
                self.active_models.append(name)
            elif name == "EfficientNet" and hasattr(detector, 'is_loaded') and detector.is_loaded:
                self.active_models.append(name)
            elif name == "XGBoost-Noise" and hasattr(detector, 'is_loaded') and detector.is_loaded:
                self.active_models.append(name)
        
        print(f"\n✅ Active models: {', '.join(self.active_models)}")
        print(f"📊 Model weights: {self.weights}")
    
    def save_detailed_results(self, results, image_path):
        """Save detailed JSON results to file"""
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.results_dir, f"results_{img_name}.json")
        
        import json
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to: {json_path}")
        return json_path
    
    def predict_ensemble(self, image_path):
        """Run all models and combine predictions"""
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
        print("-" * 60)
        
        # Run all detectors
        results = {}
        individual_predictions = []
        
        for name in self.active_models:
            detector = self.detectors[name]
            result = detector.predict(image_path)
            
            if "error" in result:
                print(f"❌ {name}: {result['error']}")
                continue
            
            results[name] = result
            individual_predictions.append({
                "model": name,
                "decision": result["decision"],
                "fake_prob": result["fake_probability"],
                "real_prob": result["real_probability"],
                "confidence": result["confidence"]
            })
            
            print(f"✅ {name}: {result['decision']} "
                  f"(Fake: {result['fake_probability']:.2%}, "
                  f"Real: {result['real_probability']:.2%}, "
                  f"Conf: {result['confidence']:.2%})")
        
        if not results:
            return {"error": "All models failed"}
        
        # Calculate weighted ensemble prediction
        total_fake_weight = 0
        total_real_weight = 0
        total_weight = 0
        
        for name, result in results.items():
            weight = self.weights.get(name, 0.3)
            total_fake_weight += result["fake_probability"] * weight
            total_real_weight += result["real_probability"] * weight
            total_weight += weight
        
        # Normalize
        ensemble_fake_prob = total_fake_weight / total_weight
        ensemble_real_prob = total_real_weight / total_weight
        
        # Ensemble decision
        ensemble_decision = "FAKE" if ensemble_fake_prob > 0.5 else "REAL"
        ensemble_confidence = ensemble_fake_prob if ensemble_decision == "FAKE" else ensemble_real_prob
        
        # Majority voting
        fake_votes = sum(1 for r in results.values() if r["decision"] == "FAKE")
        real_votes = sum(1 for r in results.values() if r["decision"] == "REAL")
        majority_decision = "FAKE" if fake_votes > real_votes else "REAL"
        
        # Final decision
        final_decision = ensemble_decision
        if abs(ensemble_fake_prob - 0.5) < 0.1:
            final_decision = majority_decision
        
        # Prepare comprehensive results
        result_dict = {
            "image": os.path.basename(image_path),
            "image_path": image_path,
            "analysis_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "individual_results": individual_predictions,
            "ensemble": {
                "fake_probability": float(ensemble_fake_prob),
                "real_probability": float(ensemble_real_prob),
                "decision": ensemble_decision,
                "confidence": float(ensemble_confidence)
            },
            "voting": {
                "fake_votes": fake_votes,
                "real_votes": real_votes,
                "total_models": len(results),
                "majority_decision": majority_decision
            },
            "final_decision": final_decision,
            "final_confidence": float(ensemble_confidence)
        }
        
        # Save detailed results
        self.save_detailed_results(result_dict, image_path)
        
        return result_dict
    
    def generate_heatmap(self, results, image_path, save_path=None):
        """Generate advanced heatmap visualization"""
        try:
            # Auto-generate save path if not provided
            if save_path is None:
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(self.results_dir, f"heatmap_{img_name}.png")
            
            # Load original image
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                print(f"⚠️ Could not load image: {image_path}")
                return None
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 10))
            
            # 1. Original image
            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Original Image", fontsize=12)
            ax1.axis('off')
            
            # 2. Simple noise analysis visualization
            ax2 = plt.subplot(2, 3, 2)
            noise_heatmap = self._generate_noise_heatmap(orig_img)
            if noise_heatmap is not None:
                ax2.imshow(noise_heatmap)
                ax2.set_title("Noise Analysis", fontsize=12)
                ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, "Heatmap not available", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title("Noise Analysis", fontsize=12)
                ax2.axis('off')
            
            # 3. Edge detection visualization
            ax3 = plt.subplot(2, 3, 3)
            edge_heatmap = self._generate_edge_heatmap(orig_img)
            ax3.imshow(edge_heatmap)
            ax3.set_title("Edge Analysis", fontsize=12)
            ax3.axis('off')
            
            # 4. Probability bar chart
            ax4 = plt.subplot(2, 3, 4)
            models = []
            fake_probs = []
            real_probs = []
            
            for result in results["individual_results"]:
                models.append(result["model"])
                fake_probs.append(result["fake_prob"])
                real_probs.append(result["real_prob"])
            
            # Add ensemble
            models.append("ENSEMBLE")
            fake_probs.append(results["ensemble"]["fake_probability"])
            real_probs.append(results["ensemble"]["real_probability"])
            
            x = np.arange(len(models))
            width = 0.35
            
            ax4.bar(x - width/2, fake_probs, width, label='FAKE', color='red', alpha=0.7)
            ax4.bar(x + width/2, real_probs, width, label='REAL', color='green', alpha=0.7)
            
            ax4.set_xlabel('Model', fontsize=12)
            ax4.set_ylabel('Probability', fontsize=12)
            ax4.set_title('Model Probabilities', fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Add probability values
            for i, (f, r) in enumerate(zip(fake_probs, real_probs)):
                ax4.text(i - width/2, f + 0.02, f'{f:.2%}', 
                        ha='center', va='bottom', fontsize=9)
                ax4.text(i + width/2, r + 0.02, f'{r:.2%}', 
                        ha='center', va='bottom', fontsize=9)
            
            # 5. Final decision visualization
            ax5 = plt.subplot(2, 3, 5)
            decision = results["final_decision"]
            confidence = results["final_confidence"]
            
            colors = {'FAKE': 'red', 'REAL': 'green'}
            ax5.text(0.5, 0.6, f"{decision}", 
                    fontsize=40, fontweight='bold',
                    ha='center', va='center',
                    color=colors.get(decision, 'blue'))
            ax5.text(0.5, 0.3, f"Confidence: {confidence:.2%}", 
                    fontsize=20, ha='center', va='center')
            
            if confidence > 0.9:
                explanation = "Very confident"
            elif confidence > 0.7:
                explanation = "Confident"
            elif confidence > 0.6:
                explanation = "Somewhat confident"
            else:
                explanation = "Uncertain"
            
            ax5.text(0.5, 0.1, explanation, 
                    fontsize=16, ha='center', va='center',
                    style='italic')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title("Final Decision", fontsize=16)
            
            # 6. Voting results
            ax6 = plt.subplot(2, 3, 6)
            votes = results["voting"]
            labels = ['FAKE', 'REAL']
            sizes = [votes["fake_votes"], votes["real_votes"]]
            colors_pie = ['#ff9999', '#99ff99']
            explode = (0.1, 0) if decision == 'FAKE' else (0, 0.1)
            
            ax6.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.0f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
            ax6.axis('equal')
            ax6.set_title(f"Model Voting", fontsize=14)
            
            # Overall title
            plt.suptitle(f"Fake Image Detection Analysis", 
                        fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            # Save and close
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"📊 Advanced heatmap saved to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"⚠️ Could not generate advanced heatmap: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_noise_heatmap(self, image):
        """Generate noise analysis heatmap"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Noise analysis
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Normalize
            noise_normalized = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
            noise_colored = cv2.applyColorMap(noise_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Overlay on original
            if len(image.shape) == 3:
                overlay = cv2.addWeighted(image, 0.7, noise_colored, 0.3, 0)
            else:
                image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(image_bgr, 0.7, noise_colored, 0.3, 0)
            
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"⚠️ Noise heatmap failed: {e}")
            return None
    
    def _generate_edge_heatmap(self, image):
        """Generate edge analysis heatmap"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            
            # Create heatmap
            edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
            
            # Overlay on original
            if len(image.shape) == 3:
                overlay = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
            else:
                image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(image_bgr, 0.8, edges_colored, 0.2, 0)
            
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"⚠️ Edge heatmap failed: {e}")
            return None
    
    def _generate_simple_heatmap(self, results, save_path=None):
        """Simple heatmap showing model confidence scores"""
        if "individual_results" not in results:
            return None
        
        # Auto-generate save path if not provided
        if save_path is None:
            img_name = results.get('image', 'unknown')
            img_name = os.path.splitext(img_name)[0]
            save_path = os.path.join(self.results_dir, f"simple_heatmap_{img_name}.png")
        
        try:
            # Prepare data
            data = []
            labels = []
            for result in results["individual_results"]:
                data.append([result["fake_prob"], result["real_prob"]])
                labels.append(result["model"])
            
            # Add ensemble
            data.append([results["ensemble"]["fake_probability"], 
                        results["ensemble"]["real_probability"]])
            labels.append("ENSEMBLE")
            
            data = np.array(data)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use a colormap
            cmap = plt.cm.RdYlGn
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
            
            # Add text
            for i in range(len(labels)):
                for j in range(2):
                    color = 'white' if data[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{data[i, j]:.1%}', 
                           ha='center', va='center', 
                           color=color, fontweight='bold', fontsize=10)
            
            # Customize
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['FAKE', 'REAL'], fontsize=12, fontweight='bold')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=11)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability', rotation=270, labelpad=20, fontsize=12)
            
            # Add title
            title = f'Model Analysis\nFinal Decision: {results["final_decision"]} ({results["final_confidence"]:.1%} confidence)'
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
            ax.tick_params(which="minor", size=0)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # IMPORTANT: Close the figure
            
            print(f"📊 Simple heatmap saved to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"⚠️ Could not generate simple heatmap: {e}")
            return None
    
    def batch_predict(self, image_paths, output_csv=None):
        """Process multiple images"""
        if output_csv is None:
            output_csv = os.path.join(self.results_dir, "ensemble_results.csv")
        
        all_results = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"❌ Skipping {img_path} - not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(img_path)}")
            print(f"{'='*60}")
            
            result = self.predict_ensemble(img_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
            
            # Add to results
            all_results.append({
                "image": result["image"],
                "final_decision": result["final_decision"],
                "final_confidence": result["final_confidence"],
                "ensemble_fake_prob": result["ensemble"]["fake_probability"],
                "ensemble_real_prob": result["ensemble"]["real_probability"]
            })
            
            # Generate heatmaps
            try:
                heatmap_path = os.path.join(self.results_dir, f"heatmap_{os.path.splitext(result['image'])[0]}.png")
                self.generate_heatmap(result, img_path, heatmap_path)
                
                simple_path = os.path.join(self.results_dir, f"simple_heatmap_{os.path.splitext(result['image'])[0]}.png")
                self._generate_simple_heatmap(result, simple_path)
            except Exception as e:
                print(f"⚠️ Could not generate heatmaps: {e}")
        
        # Save to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Batch results saved to: {output_csv}")
        
        return all_results

# =============================
#        MAIN EXECUTION
# =============================
def main():
    """Main function for standalone execution"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Fake Image Detection Ensemble")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder with images")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize ensemble
    ensemble = EnsembleDetector(
        jpeg_model_path="model_complete_gpu.pth",
        jpeg_scaler_path="scaler_gpu.pkl",
        efficientnet_path="no_fft_model.pth",
        xgb_model_path="xgb_noise_model.pkl",
        xgb_scaler_path="xgb_scaler.pkl",
        results_dir=args.output
    )
    
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        result = ensemble.predict_ensemble(args.image)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print("\n" + "="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        
        print(f"\n🖼️  Image: {result['image']}")
        print(f"🎯 Final Decision: {result['final_decision']}")
        print(f"📈 Final Confidence: {result['final_confidence']:.2%}")
        
        print(f"\n🤖 Ensemble Probabilities:")
        print(f"   FAKE: {result['ensemble']['fake_probability']:.2%}")
        print(f"   REAL: {result['ensemble']['real_probability']:.2%}")
        
        print(f"\n🗳️  Voting Results:")
        print(f"   FAKE votes: {result['voting']['fake_votes']}")
        print(f"   REAL votes: {result['voting']['real_votes']}")
        print(f"   Majority: {result['voting']['majority_decision']}")
        
        # Generate heatmaps
        try:
            heatmap_path = os.path.join(ensemble.results_dir, f"heatmap_{os.path.splitext(result['image'])[0]}.png")
            ensemble.generate_heatmap(result, args.image, heatmap_path)
            
            simple_path = os.path.join(ensemble.results_dir, f"simple_heatmap_{os.path.splitext(result['image'])[0]}.png")
            ensemble._generate_simple_heatmap(result, simple_path)
        except Exception as e:
            print(f"⚠️ Could not generate heatmaps: {e}")
        
    elif args.folder:
        # Batch prediction
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
        
        if not image_paths:
            print(f"❌ No images found in {args.folder}")
            return
        
        print(f"\n📁 Found {len(image_paths)} images in {args.folder}")
        
        # Process batch (limit to 10 for demo)
        results = ensemble.batch_predict(
            image_paths[:10],
            output_csv=os.path.join(ensemble.results_dir, "ensemble_results.csv")
        )
        
        # Summary
        fake_count = sum(1 for r in results if r["final_decision"] == "FAKE")
        real_count = sum(1 for r in results if r["final_decision"] == "REAL")
        
        print(f"\n{'='*80}")
        print("📈 BATCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total images processed: {len(results)}")
        print(f"FAKE detected: {fake_count}")
        print(f"REAL detected: {real_count}")
        print(f"FAKE percentage: {fake_count/len(results)*100:.1f}%")
        
    else:
        # Interactive mode
        print("\n🎮 INTERACTIVE MODE")
        print("Enter image paths (one per line). Type 'done' to finish.")
        
        image_paths = []
        while True:
            path = input("\n📁 Image path (or 'done'): ").strip()
            if path.lower() == 'done':
                break
            if os.path.exists(path):
                image_paths.append(path)
            else:
                print(f"❌ File not found: {path}")
        
        if image_paths:
            results = ensemble.batch_predict(
                image_paths,
                output_csv=os.path.join(ensemble.results_dir, "interactive_results.csv")
            )
        else:
            print("No valid images provided.")

if __name__ == "__main__":
    # Check for required files
    required_files = [
        ("model_complete_gpu.pth", "JPEG model"),
        ("scaler_gpu.pkl", "JPEG scaler"),
        ("no_fft_model.pth", "EfficientNet model"),
        ("xgb_noise_model.pkl", "XGBoost noise model"),
        ("xgb_scaler.pkl", "XGBoost scaler")
    ]
    
    missing = []
    for file, desc in required_files:
        if not os.path.exists(file):
            missing.append(f"{file} ({desc})")
    
    if missing:
        print("⚠️  Missing files:")
        for item in missing:
            print(f"   - {item}")
        print("\nPlease ensure all model files are in the current directory.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            exit()
    
    main()