from typing import final
import streamlit as st
import cv2
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def main():

    st.header("Sneha ")
    bg_image = st.sidebar.file_uploader("Background image:", type=["jpeg","png", "jpg"])

    stroke_width = 3
    stroke_color = (0,255,0)
    bg_color = "#eee"
    realtime_update = True
    drawing_mode = 'point'
    point_display_radius = 3
    if bg_image is not None:
        original = Image.open(bg_image)
        img = np.array(original)
        st.sidebar.image(img, use_column_width=True)
        w,h = original.size


        # Create a canvas component
        canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=h,
        width=w,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
        )

        # Do something interesting with the image data and paths
        # if canvas_result.image_data is not None:
        #     st.image(canvas_result.image_data)
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

if __name__ == "__main__":
    main()