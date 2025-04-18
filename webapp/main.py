import streamlit as st
from streamlit_bokeh import streamlit_bokeh
import sys
sys.path.append("../grannules")
from grannules.utils.scalingrelations import compare_psd_bokeh, _pd_cache
from lightkurve import LightkurveError
import shutil

# Streamlit app title
st.set_page_config(page_title = "RG PSD Viewer", page_icon="webapp/favicon.png", layout = "wide")
st.title("Red Giant Power Spectrum Viewer")


# Sidebar sliders for user input
st.sidebar.header("Star Parameters")
mass = st.sidebar.slider("Mass ($$\\mathrm{M}_\\odot$$)", min_value=0.5, max_value=5.0, value=1.55, step=0.01)
radius = st.sidebar.slider("Radius ($$\\mathrm{R}_\\odot$$)", min_value=5.0, max_value=20.0, value=13.26, step=0.01)
temperature = st.sidebar.slider("Temperature ($$\\mathrm{K}$$)", min_value=4200, max_value=5400, value=4751, step=100)
metallicity = st.sidebar.slider("Metallicity ($$\\left[ \\mathrm{Fe}/\\mathrm{H} \\right]$$)", min_value=-2.0, max_value=1.0, value=-0.08, step=0.01)
magnitude = st.sidebar.slider("Apparent Magnitude (Kepler Band)", min_value=7.0, max_value=15.0, value=9.196, step=0.001, help = "I should probably turn this into distance instead...")
phase = st.sidebar.selectbox(
    "Phase", 
    ["Unclassified", "Red Giant Branch", "Helium Burning"],
    index = 2
)

phase_num = {
    "Unclassified": 0, 
    "Red Giant Branch": 1, 
    "Helium Burning": 2
}[phase]

st.sidebar.divider()

# User input for KIC number
kic_number_input = st.sidebar.text_input("Enter KIC Number (optional)", value="757137")
try:
    kic_number = int(kic_number_input)
except ValueError:
    kic_number = None

# Generate and display the power spectrum using compare_psd_bokeh
st.header("Power Spectrum")
try:
    bokeh_fig = compare_psd_bokeh(
        M=mass, R=radius, Teff=temperature, FeH=metallicity, KepMag=magnitude, phase=phase_num, KIC=kic_number, cache = st.session_state
    )
    streamlit_bokeh(bokeh_fig)
except LightkurveError as e:
    # scary
    shutil.rmtree("~/.lightkurve")
    st.button("Rerun")
    st.write(":red[Something with the lightkurve cache went wrong. Please press the rerun button.]")

info_column, banner_column = st.sidebar.columns([1, 2], vertical_alignment = "center")

info_column.write("[What is this?](https://grannules.readthedocs.io/en/latest/#id4)")
info_column.write("[Github Repo](https://grannules.readthedocs.io/en/latest/#id4)")

banner_column.image("webapp/grannules banner resized.png")