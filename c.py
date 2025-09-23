import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from timm import create_model
import segmentation_models_pytorch as smp
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os
import tempfile
import json
from groq import Groq # <--- NEW: Import Groq

# ===================================================================
# Configuration
# ===================================================================
CLASSIFIER_MODEL_PATH = "dfu_classifier.pth"
SEGMENTATION_MODEL_PATH = "unet_segmentation_model.pth"
DEFAULT_PIXELS_PER_MM = 11.81

# ===================================================================
# NEW: Groq LLM API Caller Function
# ===================================================================
def call_groq_llm(prompt: str) -> str:
    """Sends a prompt to the Groq API and returns the model's response."""
    try:
        # Use st.secrets for Streamlit deployment
        groq_api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=groq_api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful medical analysis assistant specializing in DFU."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the LLM: {e}")
        return None

# ===================================================================
# Core Classes (GradCAM and DFUAnalyzer)
# NOTE: These classes remain unchanged from your previous version.
# ===================================================================
class GradCAM:
    # ... (Your GradCAM class code here, no changes needed)
    """ Helper class for extracting Grad-CAM heatmaps from a model. """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.handles = []
        self._register_hooks()

    def _save_grad(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def _save_act(self, name):
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

    def __call__(self, x, device):
        x = x.to(device)
        output = self.model(x)
        self.model.zero_grad()
        pred_class_score = output.sum()
        pred_class_score.backward()
        cams = []
        for name in self.target_layers:
            grads = self.gradients.get(name)
            acts = self.activations.get(name)
            if grads is None or acts is None: continue
            pooled_grad = torch.mean(grads, dim=[0, 2, 3])
            cam = (acts[0] * pooled_grad[:, None, None]).sum(dim=0).cpu().numpy()
            cam = np.maximum(cam, 0)
            if cam.size > 0:
                cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
                cam -= cam.min()
                if cam.max() > 0: cam /= cam.max()
                cams.append(cam)
        if not cams: return np.zeros((x.shape[2], x.shape[3]))
        return np.mean(cams, axis=0)

class DFUAnalyzer:
    # ... (Your DFUAnalyzer class code here, no changes needed)
    """ An analyzer to perform classification, segmentation, and XAI on DFU images. """
    def __init__(self, classifier_path, segmenter_path, img_size=224, use_midas=True):
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models(classifier_path, segmenter_path)
        self._define_transforms()
        self.use_midas = use_midas
        if self.use_midas:
            self._init_midas()

    def _load_models(self, classifier_path, segmenter_path):
        with st.spinner(f"Loading models to {self.device}..."):
            try:
                self.classifier = create_model("efficientnet_b0", pretrained=False, num_classes=1)
                self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                self.classifier.to(self.device).eval()
                self.segmenter = smp.Unet(
                    encoder_name="efficientnet-b0", encoder_weights="imagenet",
                    in_channels=3, classes=1, activation=None
                ).to(self.device)
                self.segmenter.load_state_dict(torch.load(segmenter_path, map_location=self.device))
                self.segmenter.eval()
            except FileNotFoundError as e:
                st.error(f"ERROR: Model file not found at '{e.filename}'.")
                st.stop()

    def _define_transforms(self):
        self.cls_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.seg_transform = Compose([
            Resize(self.img_size, self.img_size),
            Normalize(),
            ToTensorV2()
        ])

    def _lime_predict_wrapper(self, images):
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        tensors = torch.stack([self.cls_transform(p) for p in pil_images]).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.classifier(tensors)).cpu().numpy()
        return np.concatenate([1 - probs, probs], axis=1)

    def _init_midas(self):
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = midas_transforms.dpt_transform
        except Exception as e:
            st.warning(f"Could not load MiDaS model: {e}. Depth analysis will be disabled.")
            self.midas, self.midas_transform, self.use_midas = None, None, False

    def _infer_depth_midas(self, pil_image):
        if not self.use_midas: return None
        img_np = np.array(pil_image)
        if img_np.ndim == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        input_tensor = self.midas_transform(img_np).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=pil_image.size[::-1], mode="bicubic", align_corners=False
            ).squeeze()
        return prediction.cpu().numpy()

    @staticmethod
    def _compute_relative_depth_for_mask(depth_map, mask):
        if mask.shape != depth_map.shape:
            mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_depth = depth_map.copy()
        masked_depth[mask == 0] = np.nan
        kernel = np.ones((5, 5), np.uint8)
        border_mask = (cv2.dilate(mask, kernel, iterations=2) > 0) & (mask == 0)
        skin_depth_values = depth_map[border_mask]
        skin_depth = np.nanmedian(skin_depth_values) if skin_depth_values.size > 0 else np.nanmedian(depth_map)
        relative_depth = -(depth_map - skin_depth)
        relative_depth[mask == 0] = np.nan
        return relative_depth

    @staticmethod
    def _get_segmented_region_depth(mask, depth_map):
        vals = depth_map[mask > 0]
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if vals.size > 0 else None

    @staticmethod
    def _get_middle_region_depth(mask, depth_map, erosion_kernel=(15, 15)):
        kernel = np.ones(erosion_kernel, np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        vals = depth_map[eroded > 0]
        vals = vals[~np.isnan(vals)]
        return float(np.nanmedian(vals)) if vals.size > 0 else None

    def analyze(self, img_path, pixels_per_mm=None, depth_unit_to_mm=None, lime_num_samples=150, cam_layers=["conv_head"]):
        results = {}
        original_pil = Image.open(img_path).convert("RGB")
        original_resized_for_models = np.array(original_pil.resize((self.img_size, self.img_size)))
        results['original_image_resized'] = original_resized_for_models
        # 1. Classification
        input_tensor = self.cls_transform(original_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = torch.sigmoid(self.classifier(input_tensor)).item()
        label = "Abnormal" if pred > 0.5 else "Normal"
        conf = pred if label == "Abnormal" else 1 - pred
        results['Classification'] = f"{label} (Confidence: {conf:.2f})"
        results['Classification_Label'] = label
        if label == "Normal":
            return {'Classification': results['Classification'], 'Classification_Label': label}
        # 2. Segmentation
        img_cv_full = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        seg_in = self.seg_transform(image=img_cv_full)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask_prob = torch.sigmoid(self.segmenter(seg_in)).cpu().squeeze().numpy()
        seg_mask_full_size = cv2.resize((mask_prob > 0.5).astype(np.uint8), original_pil.size, interpolation=cv2.INTER_NEAREST)
        results['segmentation_mask_resized'] = (mask_prob > 0.5).astype(np.uint8)
        # 3. Grad-CAM
        cam_extractor = GradCAM(self.classifier, cam_layers)
        results['grad_cam_map_resized'] = cam_extractor(input_tensor.clone(), self.device)
        cam_extractor.remove_hooks()
        # 4. LIME
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            results['original_image_resized'], self._lime_predict_wrapper, top_labels=1, hide_color=0, num_samples=lime_num_samples
        )
        _, lime_mask_resized = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True, num_features=30, hide_rest=True
        )
        lime_overlay_full = mark_boundaries(results['original_image_resized'] / 255.0, lime_mask_resized)
        results['lime_overlay_resized'] = lime_overlay_full
        # 5. Measurements
        contours, _ = cv2.findContours(seg_mask_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measurements = {"Area (pixels)": 0, "Max Width (pixels)": 0}
        if contours:
            c = max(contours, key=cv2.contourArea)
            _, _, w, _ = cv2.boundingRect(c)
            area_px, width_px = float(cv2.contourArea(c)), float(w)
            measurements["Area (pixels)"] = area_px
            measurements["Max Width (pixels)"] = width_px
            if pixels_per_mm is not None and pixels_per_mm > 0:
                measurements["Estimated Area (mm²)"] = area_px / (pixels_per_mm ** 2)
                measurements["Estimated Max Width (mm)"] = width_px / pixels_per_mm
        results['measurements'] = measurements
        # 6. MiDaS Depth Analysis
        depth_results = {"note": "MiDaS disabled or failed."}
        if self.use_midas:
            depth_map_full = self._infer_depth_midas(original_pil)
            if depth_map_full is not None:
                relative_depth_full = self._compute_relative_depth_for_mask(depth_map_full, seg_mask_full_size)
                seg_depth_unit = self._get_segmented_region_depth(seg_mask_full_size, relative_depth_full)
                center_depth_unit = self._get_middle_region_depth(seg_mask_full_size, relative_depth_full)
                depth_results = {
                    "Segment Depth (units)": seg_depth_unit, "Segment Center Depth (units)": center_depth_unit
                }
                if depth_unit_to_mm and seg_depth_unit:
                    depth_results["Segment Depth (mm)"] = seg_depth_unit * float(depth_unit_to_mm)
                    if center_depth_unit:
                        depth_results["Segment Center Depth (mm)"] = center_depth_unit * float(depth_unit_to_mm)
        results['depth_analysis'] = depth_results
        return results

    def plot_results(self, results):
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        fig.suptitle(f"Analysis for: {results.get('image_name', 'Unknown Image')}", fontsize=16)
        axes[0].imshow(results['original_image_resized']); axes[0].set_title(f"1. Original\n{results['Classification']}"); axes[0].axis('off')
        axes[1].imshow(results['segmentation_mask_resized'], cmap='gray'); axes[1].set_title("2. Segmentation Mask"); axes[1].axis('off')
        axes[2].imshow(results['grad_cam_map_resized'], cmap='jet'); axes[2].set_title("3. Grad-CAM Heatmap"); axes[2].axis('off')
        heatmap_colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * results['grad_cam_map_resized']), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(results['original_image_resized'], 0.6, heatmap_colored, 0.4, 0)
        masked_overlay = np.where(np.stack([results['segmentation_mask_resized']] * 3, axis=-1), overlay, results['original_image_resized'])
        axes[3].imshow(masked_overlay); axes[3].set_title("4. Grad-CAM on Ulcer"); axes[3].axis('off')
        axes[4].imshow(results['lime_overlay_resized']); axes[4].set_title("5. LIME Explanation"); axes[4].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

# ===================================================================
# Streamlit App Logic
# ===================================================================

@st.cache_resource
def get_analyzer():
    analyzer = DFUAnalyzer(CLASSIFIER_MODEL_PATH, SEGMENTATION_MODEL_PATH, use_midas=True)
    return analyzer

st.set_page_config(layout="wide", page_title="Diabetic Foot Ulcer Analysis")

# --- NEW: Initialize session state ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'llm_report' not in st.session_state:
    st.session_state.llm_report = None

st.title("🩺 Diabetic Foot Ulcer (DFU) Analysis Pipeline")
st.markdown("This tool uses a multi-stage deep learning pipeline to analyze DFU images, followed by an interactive questionnaire to generate a comprehensive summary.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    pixels_per_mm = st.number_input("Pixels per Millimeter (for calibration)", min_value=1.0, value=DEFAULT_PIXELS_PER_MM, step=0.1)
    enable_depth_mm = st.checkbox("Enable Depth in mm Conversion")
    depth_unit_to_mm = None
    if enable_depth_mm:
        depth_unit_to_mm = st.number_input("Depth Model Units to mm", min_value=0.001, value=1.0, step=0.1)
    lime_samples = st.slider("LIME Explanation Quality", 100, 1000, 200, 50, help="Higher values are more accurate but slower.")

# --- Main App Body ---
if uploaded_file is None:
    st.info("👈 Please upload an image using the sidebar to begin analysis.")
else:
    analyzer = get_analyzer()
    image_name = uploaded_file.name
    st.image(uploaded_file, caption=f"Uploaded: {image_name}", width=250)

    if st.button("🚀 Analyze Image", use_container_width=True):
        st.session_state.analysis_results = None # Reset previous results
        st.session_state.llm_report = None      # Reset previous report
        with st.spinner("Analyzing image... Please wait."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_image_path = tmp_file.name
            
            analysis_results = analyzer.analyze(
                temp_image_path,
                pixels_per_mm=pixels_per_mm,
                depth_unit_to_mm=depth_unit_to_mm,
                lime_num_samples=lime_samples
            )
            os.remove(temp_image_path)
            st.session_state.analysis_results = analysis_results

# --- Display analysis results if they exist in session state ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.header("📊 Analysis Results")
    
    class_label = results.get('Classification_Label', 'Unknown')
    if class_label == "Abnormal":
        st.error(f"**Classification:** {results['Classification']}")
    else:
        st.success(f"**Classification:** {results['Classification']}")
    
    if class_label == "Normal":
        st.info("Image classified as Normal. No further analysis is required.")
    else:
        # Display plots and metrics for Abnormal cases
        results['image_name'] = image_name
        fig = analyzer.plot_results(results)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📏 Measurements")
            if 'measurements' in results:
                for key, value in results['measurements'].items():
                    st.metric(label=key, value=f"{value:.2f}")
        with col2:
            st.subheader("🔬 Depth Analysis (MiDaS)")
            if 'depth_analysis' in results:
                for key, value in results['depth_analysis'].items():
                # Check if the value is a number (float) before formatting
                    if isinstance(value, float):
                        st.metric(label=key, value=f"{value:.4f}")
                    # Otherwise, just display it as a string (for notes or errors)
                    else:
                        st.metric(label=key, value=str(value) if value is not None else "N/A")

        st.markdown("---")
        # --- NEW: Symptom Questionnaire and LLM Report Section ---
        st.header("📝 Symptom Questionnaire for Final Report")
        st.info("Please answer the following questions based on the patient's symptoms.")

        symptom_questions = {
            "redness": "Is there any unusual redness around the wound?",
            "swelling": "Do you notice any swelling in the foot or ankle?",
            "discharge": "Is there any fluid or pus draining from the ulcer?",
            "odor": "Is there a foul odor coming from the wound?",
            "pain": "Is the patient experiencing increased pain at the ulcer site?",
            "warmth": "Does the area around the wound feel warm to the touch?",
        }

        user_symptoms = {}
        # Use a form to group the radio buttons
        with st.form("symptom_form"):
            for key, question in symptom_questions.items():
                user_symptoms[key] = st.radio(question, ('No', 'Yes'), key=key, horizontal=True)
            
            submitted = st.form_submit_button("Generate Final Report", use_container_width=True)

        if submitted:
            with st.spinner("Contacting LLM to generate the report..."):
                # Prepare prompt for the LLM
                symptom_summary = "\n".join([f"- {key.capitalize()}: {val}" for key, val in user_symptoms.items()])
                
                # Clean up image results for the prompt
                detection_data = results.get('measurements', {})
                detection_data.update(results.get('depth_analysis', {}))
                detection_data['Classification'] = results.get('Classification')
                detection_summary = "\n".join([f"- {k.replace('_',' ').capitalize()}: {v}" for k, v in detection_data.items() if v is not None])

                final_prompt = f"""
                You are a medical analysis assistant specializing in Diabetic Foot Ulcers (DFU).
                Your task is to provide a clear, easy-to-understand summary based on patient-reported symptoms and a computer vision analysis of a wound image.
                **DO NOT PROVIDE A DIAGNOSIS.** Instead, explain what the findings might indicate and strongly recommend consulting a healthcare professional.

                **Patient-Reported Symptoms:**
                {symptom_summary}

                **Computer Vision Image Analysis Results:**
                {detection_summary}

                **Your Task:**
                1. Explain what the combined symptoms and image analysis results might mean in simple terms.
                2. Interpret the potential severity based on the findings (e.g., mention if signs of infection like warmth, discharge, and odor are present).
                3. Clearly and firmly recommend the patient see a doctor or wound care specialist immediately.
                """
                
                llm_report = call_groq_llm(final_prompt)
                st.session_state.llm_report = llm_report

# Display the final LLM report if it exists
if st.session_state.llm_report:
    st.markdown("---")
    st.header("📋 Final Analysis & Recommendations")
    st.markdown(st.session_state.llm_report)
    st.warning("**Disclaimer:** This is not a medical diagnosis. Please consult a qualified healthcare provider for any medical concerns.")