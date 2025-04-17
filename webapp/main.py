import streamlit as st
from streamlit_bokeh import streamlit_bokeh
from grannules.utils.scalingrelations import compare_psd_bokeh

# Streamlit app title
st.set_page_config(page_title = "RG PSD Viewer", page_icon="./favicon.png", layout = "wide")
st.title("Red Giant Power Spectrum Viewer")

# Sidebar sliders for user input
st.sidebar.header("Star Parameters")
mass = st.sidebar.slider("Mass ($$\\mathrm{M}_\\odot$$)", min_value=0.5, max_value=5.0, value=1.55, step=0.01)
radius = st.sidebar.slider("Radius ($$\\mathrm{R}_\\odot$$)", min_value=5.0, max_value=20.0, value=13.26, step=0.01)
temperature = st.sidebar.slider("Temperature ($$\\mathrm{K}$$)", min_value=4200, max_value=5400, value=4751, step=100)
metallicity = st.sidebar.slider("Metallicity ($$\\left[ \\mathrm{Fe}/\\mathrm{H} \\right]$$)", min_value=-2.0, max_value=1.0, value=-0.08, step=0.01)
magnitude = st.sidebar.slider("Magnitude (Kepler Band)", min_value=7.0, max_value=15.0, value=9.196, step=0.001)
phase = st.sidebar.slider("Phase", min_value=0, max_value=2, value=2, step=1)
st.sidebar.text("(0 = Any | 1 = Red Giant Branch | 2 = Helium Burning)")

# User input for KIC number
kic_number_input = st.sidebar.text_input("Enter KIC Number (optional)", value="757137")
try:
    kic_number = int(kic_number_input)
except ValueError:
    kic_number = None

# Generate and display the power spectrum using compare_psd_bokeh
st.header("Power Spectrum")
bokeh_fig = compare_psd_bokeh(
    M=mass, R=radius, Teff=temperature, FeH=metallicity, KepMag=magnitude, phase=phase, KIC=kic_number
)
streamlit_bokeh(bokeh_fig)