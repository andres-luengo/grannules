import streamlit as st
import pandas as pd
from streamlit_bokeh import streamlit_bokeh
from bokeh.plotting import figure

from grannules.utils.psd import nu_max
from grannules.utils.scalingrelations import compare_psd

# Streamlit app title
st.title("Red Giant Power Spectrum Viewer")

# Sidebar sliders for user input
st.sidebar.header("Star Parameters")
mass = st.sidebar.slider("Mass (M)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
radius = st.sidebar.slider("Radius (R☉)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
temperature = st.sidebar.slider("Temperature (K)", min_value=3000, max_value=8000, value=5000, step=100)
metallicity = st.sidebar.slider("Metallicity [Fe/H]", min_value=-2.5, max_value=0.5, value=0.0, step=0.1)

# Calculate nu_max
nu_max_val = nu_max(mass, radius, temperature)

# Generate and display the power spectrum using compare_psd
st.header("Power Spectrum")
psd_plot = compare_psd(
    M=mass, R=radius, Teff=temperature, FeH=metallicity, KepMag=12, phase=0
)

# Convert Holoviews Overlay to Bokeh figure
bokeh_fig = figure(
    title="Power Spectrum",
    x_axis_label="Frequency (μHz)",
    y_axis_label="Power (ppm²/μHz)",
    x_axis_type="log",
    y_axis_type="log",
    width=800,
    height=600,
)
for element in psd_plot:
    if hasattr(element, "dframe"):  # Use dframe() to extract data
        data = element.dframe()
        bokeh_fig.line(
            data.iloc[:, 0],  # x values
            data.iloc[:, 1],  # y values
            legend_label=element.label,
            line_width=2,
            color=element.opts.get("color", "black")  # Default to black if no color
        )

# Embed the Bokeh figure in Streamlit using streamlit_bokeh
streamlit_bokeh(bokeh_fig)

# Display the selected parameters
st.sidebar.subheader("Selected Parameters")
st.sidebar.write(f"Mass: $${mass} \\mathrm{{M}}_\\odot$$")
st.sidebar.write(f"Radius: $${radius} \\mathrm{{R}}_\\odot$$")
st.sidebar.write(f"Temperature: $${temperature} \\mathrm{{K}}$$")
st.sidebar.write(f"Metallicity: $${metallicity} \\left[ \\mathrm{{Fe}}/ \\mathrm{{H}} \\right]$$")