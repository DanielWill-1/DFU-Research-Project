# === app.py ===
"""
Streamlit app for DFU research demo.
Place this file as `app.py` and the pipeline below as `pipeline.py` in the same folder.
Run: streamlit run app.py
Requirements (example):
streamlit
torch
torchvision
timm
segmentation-models-pytorch
albumentations
albumentations-pytorch
pillow
opencv-python
matplotlib
lime
scikit-image
requests
"""

import streamlit as st
from PIL import Image
import numpy as np
import io

# Import pipeline functions
from pipeline import get_analyzer, run_dfu_analysis_pipeline

st.set_page_config(page_title="DFU Demo", layout="wide")
st.title("Diabetic Foot Ulcer — DFU Analysis Demo")

st.markdown("Upload an image (jpg/png). The app will run classification → segmentation → Grad-CAM & LIME → MiDaS depth (if available) and show results.")

uploaded = st.file_uploader("Upload foot image", type=["jpg", "jpeg", "png"]) 

if uploaded is not None:
    try:
        # Read image bytes and save to a temporary in-memory file path for pipeline
        image = Image.open(uploaded).convert("RGB")
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Save to a temp file path because pipeline expects a path
        tmp_bytes = io.BytesIO()
        image.save(tmp_bytes, format='PNG')
        tmp_bytes.seek(0)

        # Write to a temp file (Streamlit ephemeral file system)
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(tmp_bytes.read())
            tmp_path = tmp.name

        st.info("Running analysis — this may take a little while on first run (models will load).")

        # Run pipeline (this returns a dictionary of results)
        results = run_dfu_analysis_pipeline(tmp_path)

        if results is None:
            st.warning("Analysis did not return results. Check server logs for errors.")
        else:
            st.success("Analysis complete")
            # Show classification
            st.subheader("Classification")
            st.write(results.get('Classification', 'No classification result'))

            # Show measurements if present
            if 'measurements' in results:
                st.subheader("Measurements")
                st.json(results['measurements'])

            # Show depth analysis summary
            if 'depth_analysis' in results:
                st.subheader("Depth Analysis (MiDaS)")
                st.json(results['depth_analysis'])

            # If the analyzer produced plot outputs (it may have displayed them), we'll ask analyzer to plot to a matplotlib figure and show it
            analyzer = get_analyzer()
            try:
                # The analyzer has a plot_results method that uses matplotlib; re-run a plot in-memory
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(14, 5))
                analyzer.plot_results(results)
                # plot_results opens its own figure, so we capture and show last figure
                st.pyplot(plt.gcf())
            except Exception as e:
                st.write("Could not render detailed plot in the web UI:", e)

            # Provide JSON download
            import json
            st.download_button("Download analysis JSON", data=json.dumps(results, indent=2), file_name="dfu_analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload an image to start analysis.")

