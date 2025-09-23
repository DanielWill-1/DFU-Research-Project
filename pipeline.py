# === pipeline.py (Improved) ===
"""
pipeline.py — inference pipeline adapted from your notebook cells.
Key functions:
 - get_analyzer() -> returns initialized DFUAnalyzer singleton
 - run_dfu_analysis_pipeline(image_path, ...) -> runs full pipeline and returns structured dict

Place model files alongside or update the CLASSIFIER_MODEL_PATH and SEGMENTATION_MODEL_PATH.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from lime import lime_image
from skimage.segmentation import mark_boundaries
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

# --- Configuration ---
# For best results, use absolute paths or ensure your environment is set correctly.
CLASSIFIER_MODEL_PATH = os.environ.get('CLASSIFIER_MODEL_PATH', 'dfu_classifier.pth')
SEGMENTATION_MODEL_PATH = os.environ.get('SEGMENTATION_MODEL_PATH', 'unet_segmentation_model.pth')
PIXELS_PER_MM = None
DEPTH_UNIT_TO_MM = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_analyzer_instance = None


# --- Utility: Fix state_dict prefix ---
def fix_state_dict_prefix(state_dict: Dict[str, Any], prefix: str = 'model.') -> OrderedDict:
    """Ensures all keys in a state_dict start with a given prefix."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            new_sd[prefix + k] = v
        else:
            new_sd[k] = v
    return new_sd


# --- GradCAM class (Improved with context manager) ---
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.handles = []

    def _save_grad(self, name: str):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def _save_act(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handles.append(module.register_forward_hook(self._save_act(name)))
                self.handles.append(module.register_full_backward_hook(self._save_grad(name)))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = [] # Avoid trying to remove twice

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(DEVICE)
        output = self.model(x)
        self.model.zero_grad()
        
        # Target the predicted class for backpropagation
        pred_class_score = output.max()
        pred_class_score.backward()

        cams = []
        for name in self.target_layers:
            grads = self.gradients.get(name)
            acts = self.activations.get(name)
            if grads is None or acts is None:
                print(f"Warning: Gradients or activations not found for layer: {name}")
                continue
            
            pooled_grad = torch.mean(grads, dim=[0, 2, 3])
            cam = (acts[0] * pooled_grad[:, None, None]).sum(dim=0).cpu().numpy()
            cam = np.maximum(cam, 0)
            
            if cam.size > 0:
                cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
                if cam.max() > 0:
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                cams.append(cam)
        
        if not cams:
            return np.zeros((x.shape[2], x.shape[3]))
        
        return np.mean(cams, axis=0)

    @contextmanager
    def apply(self):
        """Context manager to ensure hooks are always removed."""
        self._register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()


# --- DFUAnalyzer class (Core logic) ---
class DFUAnalyzer:
    def __init__(self, classifier_path: str, segmenter_path: str, img_size: int = 224, use_midas: bool = True):
        self.img_size = img_size
        self.device = DEVICE
        self._load_models(classifier_path, segmenter_path)
        self._define_transforms()
        self.use_midas = use_midas
        if self.use_midas:
            self._init_midas()

    def _load_models(self, classifier_path: str, segmenter_path: str):
        # **FIXED**: Robust classifier loading
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found at: {classifier_path}")
        
        self.classifier = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1)
        
        # Keep track of initial weights of the final layer for a sanity check
        initial_head_weights = self.classifier.get_classifier().weight.clone().to(self.device)
        
        # **FIX 1**: Added weights_only=True to address the FutureWarning
        sd = torch.load(classifier_path, map_location=self.device, weights_only=True)
        
        if 'model' in sd: # Handle checkpoints that save the whole model object
             sd = sd['model']

        if not list(sd.keys())[0].startswith('model.'):
            # **FIX 2**: Changed 'add_prefix' to the correct argument 'prefix'
            sd = fix_state_dict_prefix(sd, prefix='model.')
        
        # Load with strict=False to be flexible, but we will check it after
        self.classifier.load_state_dict(sd, strict=False)
        self.classifier.to(self.device).eval()

        # **CRUCIAL CHECK**: Verify if the final layer weights were actually loaded
        final_head_weights = self.classifier.get_classifier().weight.to(self.device)
        if torch.equal(initial_head_weights, final_head_weights):
            print("---")
            print("⚠️ WARNING: Classifier head weights did not change after loading the model.")
            print("This is the likely cause of 'confidence stuck at 0.5'.")
            print("Potential Mismatches:")
            print("  1. The final layer in your saved model has a different name (e.g., 'fc' vs 'classifier').")
            print("  2. The number of classes in the saved model is different.")
            print("---")

        # Segmenter
        if not os.path.exists(segmenter_path):
            raise FileNotFoundError(f"Segmentation model not found at: {segmenter_path}")
        self.segmenter = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, in_channels=3, classes=1)
        
        # **FIX 1**: Added weights_only=True here as well
        sd2 = torch.load(segmenter_path, map_location=self.device, weights_only=True)
        
        if 'model' in sd2: # Also handle checkpoints for the segmenter
             sd2 = sd2['model']

        try:
            self.segmenter.load_state_dict(sd2)
        except RuntimeError:
            # **FIX 2**: Changed 'add_prefix' to the correct argument 'prefix'
            sd2_fixed = fix_state_dict_prefix(sd2, prefix='model.')
            self.segmenter.load_state_dict(sd2_fixed, strict=False)
        self.segmenter.to(self.device).eval()

        # Segmenter
        if not os.path.exists(segmenter_path):
            raise FileNotFoundError(f"Segmentation model not found at: {segmenter_path}")
        self.segmenter = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, in_channels=3, classes=1)
        sd2 = torch.load(segmenter_path, map_location=self.device)
        try:
            self.segmenter.load_state_dict(sd2)
        except RuntimeError:
            sd2_fixed = fix_state_dict_prefix(sd2, add_prefix='model.')
            self.segmenter.load_state_dict(sd2_fixed, strict=False)
        self.segmenter.to(self.device).eval()

    def _define_transforms(self):
        self.cls_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.seg_transform = Compose([
            Resize(self.img_size, self.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _init_midas(self):
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', verbose=False).to(self.device).eval()
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', verbose=False)
            self.midas_transform = midas_transforms.dpt_transform
        except Exception as e:
            print(f'MiDaS load failed: {e}. Depth analysis will be disabled.')
            self.midas = None
            self.midas_transform = None
            self.use_midas = False
    
    # (Other private methods like _infer_depth_midas, _lime_predict_wrapper, etc. remain the same)
    # ... [No changes needed for the other helper methods, so they are omitted for brevity] ...
    def _infer_depth_midas(self, pil_image):
        if self.midas is None or self.midas_transform is None: return None
        img_np = np.array(pil_image)
        if img_np.ndim == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        input_tensor = self.midas_transform(img_np).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=pil_image.size[::-1],
                mode='bicubic', align_corners=False).squeeze()
        return prediction.cpu().numpy()

    def _lime_predict_wrapper(self, images):
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        tensors = torch.stack([self.cls_transform(p) for p in pil_images]).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.classifier(tensors)).cpu().numpy()
        return np.hstack([1 - probs, probs]) # LIME expects (N, num_classes)

    @staticmethod
    def _compute_relative_depth_for_mask(depth_map, mask):
        h, w = depth_map.shape
        if mask.shape != (h, w): mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masked_depth = depth_map.copy(); masked_depth[mask == 0] = np.nan
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        border_mask = (dilated > 0) & (mask == 0)
        skin_depth_values = depth_map[border_mask]
        skin_depth = np.nanmedian(skin_depth_values) if skin_depth_values.size > 0 else np.nanmedian(depth_map)
        relative_depth = -(depth_map - skin_depth)
        relative_depth[mask == 0] = np.nan
        return relative_depth

    @staticmethod
    def _get_segmented_region_depth(mask, depth_map):
        if mask.shape != depth_map.shape: mask = cv2.resize(mask, depth_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
        vals = depth_map[mask > 0]; vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if vals.size > 0 else None

    @staticmethod
    def _get_middle_region_depth(mask, depth_map, erosion_kernel=(15, 15)):
        if mask.shape != depth_map.shape: mask = cv2.resize(mask, depth_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
        kernel = np.ones(erosion_kernel, np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        middle_depth_values = depth_map[eroded > 0]; middle_depth_values = middle_depth_values[~np.isnan(middle_depth_values)]
        return float(np.nanmedian(middle_depth_values)) if middle_depth_values.size > 0 else None


    def analyze(self, img_path: str, pixels_per_mm: Optional[float] = None, depth_unit_to_mm: Optional[float] = None, cam_layers: List[str] = ['model.conv_head']) -> Dict[str, Any]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Input image not found at: {img_path}")
        
        results = {}
        original_pil = Image.open(img_path).convert('RGB')
        
        # **IMPROVEMENT**: Let transforms handle resizing to avoid redundant operations
        # The resized version will be created as needed from the full-resolution image
        
        # Classification
        input_tensor_cls = self.cls_transform(original_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.classifier(input_tensor_cls)
            prob = torch.sigmoid(logit).item()
        
        label = 'Abnormal' if prob > 0.5 else 'Normal'
        conf = prob if label == 'Abnormal' else 1 - prob
        results['Classification'] = f"{label} (Confidence: {conf:.2f})"
        results['raw_confidence'] = prob

        # Store a resized version of the original image for plotting
        resized_original_np = np.array(original_pil.resize((self.img_size, self.img_size)))
        results['original_image_resized'] = resized_original_np

        if label == 'Normal':
            return results

        # Segmentation
        # Use a numpy array of the original image for albumentations
        img_np_full = np.array(original_pil)
        seg_in = self.seg_transform(image=img_np_full)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask_prob = torch.sigmoid(self.segmenter(seg_in)).cpu().squeeze().numpy()
        
        seg_mask_resized = (mask_prob > 0.5).astype(np.uint8)
        results['segmentation_mask_resized'] = seg_mask_resized
        
        # Upscale mask to full size for measurements
        seg_mask_full_size = cv2.resize(seg_mask_resized, original_pil.size, interpolation=cv2.INTER_NEAREST)

        # GradCAM
        cam_extractor = GradCAM(self.classifier, cam_layers)
        with cam_extractor.apply(): # Use context manager
            results['grad_cam_map_resized'] = cam_extractor(input_tensor_cls)
        
        # LIME
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image=resized_original_np,
            classifier_fn=self._lime_predict_wrapper,
            top_labels=1,
            hide_color=0,
            num_samples=300, # **IMPROVEMENT**: Reduced for faster inference
            segmentation_fn=lime_image.SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        )
        _, lime_mask_resized = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=30, hide_rest=True
        )
        lime_overlay_resized = mark_boundaries(resized_original_np / 255.0, lime_mask_resized)
        
        # Mask LIME explanation to only show on the segmented ulcer region for clarity
        seg_mask_3ch = np.stack([seg_mask_resized] * 3, axis=-1)
        results['lime_overlay_resized'] = np.where(seg_mask_3ch, lime_overlay_resized, resized_original_np / 255.0).astype(np.float32)

        # Measurements
        contours, _ = cv2.findContours(seg_mask_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measurements = {"Area (pixels)": 0, "Max Width (pixels)": 0}
        if contours:
            c = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(c)
            _, _, w, _ = cv2.boundingRect(c)
            measurements["Area (pixels)"] = area_px
            measurements["Max Width (pixels)"] = float(w)
            if pixels_per_mm is not None and pixels_per_mm > 0:
                measurements["Estimated Area (mm²)"] = area_px / (pixels_per_mm ** 2)
                measurements["Estimated Max Width (mm)"] = w / pixels_per_mm
        results['measurements'] = measurements

        # MiDaS Depth Analysis
        depth_results = {"note": "MiDaS disabled or failed to initialize."}
        if self.use_midas and self.midas is not None:
            try:
                depth_map = self._infer_depth_midas(original_pil)
                if depth_map is not None:
                    relative_depth = self._compute_relative_depth_for_mask(depth_map, seg_mask_full_size)
                    seg_depth = self._get_segmented_region_depth(seg_mask_full_size, relative_depth)
                    center_depth = self._get_middle_region_depth(seg_mask_full_size, relative_depth)
                    depth_results = {"Segment Depth (units)": seg_depth, "Segment Center Depth (units)": center_depth}
                    if depth_unit_to_mm is not None and seg_depth is not None:
                        depth_results["Segment Depth (mm)"] = seg_depth * float(depth_unit_to_mm)
                        if center_depth is not None:
                           depth_results["Segment Center Depth (mm)"] = center_depth * float(depth_unit_to_mm)
                else:
                    depth_results = {"error": "MiDaS inference returned None."}
            except Exception as e:
                depth_results = {"error": str(e)}
        results['depth_analysis'] = depth_results
        return results

    def plot_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """Generates and displays/saves a plot of the analysis results."""
        if 'original_image_resized' not in results:
            print("No data to plot.")
            if 'Classification' in results:
                 print(f"Result: {results['Classification']}")
            return
        
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        fig.suptitle(f"DFU Analysis Result: {results.get('Classification', 'N/A')}", fontsize=16)

        # 1. Original Image
        axes[0].imshow(results['original_image_resized'])
        axes[0].set_title("1. Original Image")
        axes[0].axis('off')
        
        # 2. Segmentation Mask
        axes[1].imshow(results['segmentation_mask_resized'], cmap='gray')
        axes[1].set_title("2. Segmentation Mask")
        axes[1].axis('off')
        
        # 3. Grad-CAM Heatmap
        grad_cam_map = results.get('grad_cam_map_resized', np.zeros_like(results['original_image_resized'][:,:,0], dtype=float))
        axes[2].imshow(grad_cam_map, cmap='jet')
        axes[2].set_title("3. Grad-CAM Heatmap")
        axes[2].axis('off')
        
        # 4. Grad-CAM Overlay on Ulcer
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * grad_cam_map), cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(results['original_image_resized'], 0.6, heatmap_colored_rgb, 0.4, 0)
        seg_mask_3ch = np.stack([results['segmentation_mask_resized']] * 3, axis=-1)
        masked_overlay = np.where(seg_mask_3ch, overlay, results['original_image_resized'])
        axes[3].imshow(masked_overlay)
        axes[3].set_title("4. Grad-CAM on Ulcer")
        axes[3].axis('off')

        # 5. LIME Explanation
        axes[4].imshow(results['lime_overlay_resized'])
        axes[4].set_title("5. LIME Explanation")
        axes[4].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        # **FIXED**: Added plt.show() to display the plot
        plt.show()

# --- Singleton accessor ---
def get_analyzer() -> DFUAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        try:
            print("Initializing DFUAnalyzer...")
            _analyzer_instance = DFUAnalyzer(CLASSIFIER_MODEL_PATH, SEGMENTATION_MODEL_PATH, use_midas=True)
            print("DFUAnalyzer initialized successfully.")
        except Exception as e:
            _analyzer_instance = 'ERROR'
            print(f"FATAL: DFUAnalyzer failed to initialize: {e}")
            raise
    if _analyzer_instance == 'ERROR':
        raise RuntimeError('Analyzer is in a failed state. Please check model paths and dependencies.')
    return _analyzer_instance

# --- Public pipeline function ---
def run_dfu_analysis_pipeline(image_path: str, pixels_per_mm: float = PIXELS_PER_MM, depth_unit_to_mm: float = DEPTH_UNIT_TO_MM) -> Optional[Dict[str, Any]]:
    try:
        analyzer = get_analyzer()
        results = analyzer.analyze(image_path, pixels_per_mm=pixels_per_mm, depth_unit_to_mm=depth_unit_to_mm)
        
        # Create a copy of results for returning, without large arrays
        results_for_return = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
        results_for_return['info'] = "Image arrays removed for smaller return size."
        
        return results_for_return
    except Exception as e:
        print(f"An error occurred during the analysis pipeline: {e}")
        return None

# --- Example Usage (Testable Script) ---
if __name__ == '__main__':
    print(f"Running pipeline in test mode on device: {DEVICE}")

    # Create a dummy image for testing if no image is available
    TEST_IMAGE_PATH = 'test_image.png'
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Creating a dummy image at: {TEST_IMAGE_PATH}")
        dummy_img_data = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)
        Image.fromarray(dummy_img_data).save(TEST_IMAGE_PATH)

    try:
        # 1. Get the analyzer instance (it will initialize on first call)
        analyzer = get_analyzer()

        # 2. Run the full analysis
        # Pass the full results dictionary, including arrays, to the plot function
        full_results = analyzer.analyze(TEST_IMAGE_PATH)

        if full_results:
            # 3. Print the results (without the large image arrays)
            results_to_print = {k: v for k, v in full_results.items() if not isinstance(v, np.ndarray)}
            import json
            print("\n--- Analysis Results ---")
            print(json.dumps(results_to_print, indent=2))
            print("------------------------\n")

            # 4. Plot the results and save to a file
            print("Generating plot...")
            analyzer.plot_results(full_results, output_path='analysis_summary.png')
        else:
            print("Analysis failed to produce results.")

    except FileNotFoundError as fnf_error:
        print(f"\nERROR: Could not find a model file. Please ensure '{CLASSIFIER_MODEL_PATH}' and '{SEGMENTATION_MODEL_PATH}' are in the correct location.")
        print(f"Details: {fnf_error}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test run: {e}")