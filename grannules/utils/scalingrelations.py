r"""
Provides the :class:`SRPredictor` class, which allows the user to predict
:math:`H,\,P,\,` and :math:`\tau` using the :math:`\nu_\mathrm{max}`
scaling relations in `de Assis Peralta et al. 2018`_, complemented with my own
fit for :math:`\alpha` using the data from the same study.

.. _de Assis Peralta et al. 2018: https://doi.org/10.48550/arXiv.1805.04296
"""

import numpy as np
import pandas as pd

from ..neural_net import NNPredictor
from .psd import PSD, nu_max
from bokeh.plotting import figure

# i might have made this a sibling of NNPredictor, but i don't think it's 
# important enough to add that much complexity
class SRPredictor():
    r"""
    Uses :math:`\nu_\mathrm{max}` scaling relations to predict red giant 
    granulation parameters. 
    
    Defaults to using the relations for :math:`H,\, P,\,`
    and :math:`\tau` found in `de Assis Peralta et al. 2018`_ Table B5, and
    a provided fit for :math:`\alpha`.

    :param relations: If `use_phase` is False, overrides the "all" fits in the
        default relations with entries in this dict. Use keys "H", "P", "tau"
        and "alpha" to replace the respective scaling relations. The values
        should be tuples of float of length 2, where the first item is the
        coefficient of :math:`\nu_\mathrm{max}`, and the second is the exponent.
        Defaults to an empty dictionary (no overrides).
    :type relations:  dict[str, tuple[float, float]]
    :param relations_with_phase: This parameter is a tuple of length 3, where
        each element is a dict just like `relations`. Each dict corresponds to
        the scaling relations for the following stellar evolution phases:

        * 0: Unclassified/All
        * 1: Red Giant Branch
        * 2: Red Clump/Helium Burning

        If `use_phase` is True, each dict will override the default scaling
        relation of its respective phase, as with the `relations` parameter.
    :type relations_with_phase: tuple[dict[str, tuple[float, float]]]
    :param use_phase: Whether to use different scaling relations based on the 
        phase of the input star.
    :type use_phase: bool
    """
    def __init__(
            self, 
            relations : dict[str, tuple[float, float]] = {},
            relations_with_phase : tuple[dict[str, tuple[float, float]]] = ({}, {}, {}),
            use_phase = True
        ):
        self.use_phase = use_phase
        if use_phase:
            self._coefficients = []
            self._powers = []
            relations_with_phase_dict = [
                # 0: Unclassified
                {
                    "H" : (2.021e7, -1.9029),
                    "P" : (1.306e7, -1.9180),
                    "tau" : (5.72e4, -0.5332),
                    "alpha" : (3.61265536, -0.00768064) # my fit :)
                },
                # 1 : RGB
                {
                    "H" : (2.19e7, -1.863),
                    "P" : (4.9e6, -1.635),
                    "tau" : (3.62e4, -0.379),
                    "alpha" : (6.59235356, -0.23847137) # my fit :)
                },
                # 2 : RC/HeB
                {
                    "H" : (8.0e8, -2.921),
                    "P" : (9.10e7, -2.496),
                    "tau" : (2.20e4, -0.306),
                    "alpha" : (14.60572466, -0.42630839) # my fit :)
                }
            ]
            for phase in range(3):
                relations_with_phase_dict[phase].update(relations_with_phase[phase])
                self._coefficients.append([duplet[0] for duplet in relations_with_phase_dict[phase].values()])
                self._powers.append([duplet[1] for duplet in relations_with_phase_dict[phase].values()])
            self._params = relations_with_phase_dict[0].keys()
        else:
            relations_dict = {
                "H" : (2.021e7, -1.9029),
                "P" : (1.306e7, -1.9180),
                "tau" : (5.72e4, -0.5332),
                "alpha" : (3.031339584469758, 0.0) # N/A using the mean of alpha
            }
            relations_dict.update(relations)
            self._coefficients = [duplet[0] for duplet in relations_dict.values()]
            self._powers = [duplet[1] for duplet in relations_dict.values()]
            self._params = relations_dict.keys()
    

    def predict(self, nu_max, phases = 0) -> pd.DataFrame:
        index = None

        if type(nu_max) is pd.Series:
            index = nu_max.index
            nu_max = nu_max.values
        
        if isinstance(phases, int):
            phases = np.ones_like(nu_max) * phases
        
        if not isinstance(nu_max, np.ndarray):
            nu_max = np.array(nu_max).reshape((-1))
        if (not isinstance(phases, np.ndarray)) and (not phases is None):
            phases = np.array(phases).reshape((-1))
        nu_max_rows = nu_max.reshape((-1, 1))
        if index is None:
            index = np.arange(len(nu_max_rows))
        pred = pd.DataFrame(index=index, columns=self._params, dtype=float)
        if self.use_phase:
            for phase in range(3):
                pred.loc[phases == phase, :] = \
                    self._coefficients[phase] * nu_max_rows[phases == phase] ** self._powers[phase]
        else:
            pred.loc[:, :] = self._coefficients * nu_max_rows ** self._powers
        return pred

_pd_cache = {}
def compare_psd(
                M: float | None = None,
                R: float | None = None,
                Teff: float | None = None,
                FeH: float | None = None,
                phase: float | None = None,
                KepMag: float | None = None,
                KIC = None,
                *,
                x: pd.Series | None = None
):

    """
    Compare the power spectral density (PSD) of a red giant derived from the
    default NNPredictor model and the scaling relations. Has functionality to
    optionally fetch observed PSD using lightkurve.

    .. important::
        This method requires :mod:`holoviews` to be installed. 
        
        To fetch Kepler lightcurves (i.e. running with :code:`KIC is not None`),
        :mod:`lightkurve` is also required.

    :param M: Stellar mass in solar units. Default is None.
    :type M: float, optional
    :param R: Stellar radius in solar units. Default is None.
    :type R: float, optional
    :param Teff: Effective temperature of the star in Kelvin. Default is None.
    :type Teff: float, optional
    :param FeH: Metallicity of the star ([Fe/H]). Default is None.
    :type FeH: float, optional
    :param phase: Evolutionary phase of the star. Default is None.
    :type phase: float, optional
    :param KepMag: Kepler magnitude of the star. Default is None.
    :type KepMag: float, optional
    :param KIC: Kepler Input Catalog (KIC) identifier for the star. Default is
        None.
    :type KIC: int, optional
    :param x: A pandas Series containing the stellar parameters (M, R, Teff,
        FeH, phase, KepMag). If provided, individual parameters should not be
        passed. Default is None.
    :type x: pd.Series, optional

    :return: A Holoviews Overlay object containing the PSD plots for the star, 
        including:

        * The true PSD (if KIC is provided).
        * The binned PSD (if KIC is provided).
        * PSD predicted using scaling relations.
        * PSD predicted using a neural network.
        * A vertical line indicating the predicted nu_max value.

    :rtype: :type:`holoviews.Overlay`

    :raises ValueError: If neither a pandas Series (`x`) nor individual stellar 
        parameters are provided.

    .. note::
        * The function uses Lightkurve to search, download, and process the 
          light curve data for the given KIC.
        * The PSD is computed and compared against predictions from scaling 
          relations and a neural network.
        * The function caches the PSD for a given KIC to avoid redundant 
          computations.
    """
    import holoviews as hv
    hv.extension("bokeh")

    series_given = x is not None
    values_given = None not in {M, R, Teff, FeH, phase, KepMag}
    if not series_given:
        x = pd.Series(
            data = [M, R, Teff, FeH, phase, KepMag],
            index = ["M", "R", "Teff", "FeH", "phase", "KepMag"]
        )
    elif not values_given:
        raise ValueError(
            "Must provide stellar parameters with either series (x = ) or "
            "individual parameters."
        )

    if KIC in _pd_cache:
        print("Reading from cache...")
        psd = _pd_cache[KIC]
    elif KIC is not None:
        import lightkurve as lk
        print("Searching...")
        search_result = lk.search_lightcurve(f"KIC {KIC}")
        print("Downloading...")
        lc = search_result.download_all()
        print("Processing lightcurve...")
        lc = lc.stitch(lambda x : x.normalize('ppm'))
        psd = lc.to_periodogram(normalization='psd')
        _pd_cache[KIC] = psd
    
    # pd.power *= 2
    # dpd["power"] /= 2
    plots = []

    if KIC is not None:
        dpd = psd.to_table().to_pandas() # lol
        plots.append(
            hv.Curve(
                (dpd["frequency"], dpd["power"]),
                label='True'
            ).opts(
                line_width=1.5,
                color='gray',
            )
        )
        elements_per_bin = 50
        spd = psd.bin(elements_per_bin).to_table().to_pandas()
        # spd["power"] /= 2
        plots.append(
            hv.Curve(
                (spd["frequency"], spd["power"]),
                label='True, Binned'
            ).opts(
                line_width=1.5,
                color='black',
            )
        )

    # # deassis-ish fit
    # da_params = df.loc[KIC, ["H", "P", "tau", "alpha"]]
    # nu_max = df.loc[KIC, "nu_max"]
    # da_psd = PSD(dpd["frequency"], nu_max, *da_params)
    # plots.append(
    #     hv.Curve(
    #         (dpd["frequency"], da_psd),
    #         label="DeAssis Fit"
    #     ).opts(
    #         line_width=1.5,
    #         color='red'
    #     )
    # )

    # scaling relations
    frequency = np.logspace(np.log10(5), np.log10(300))
    nu_max_val = nu_max(M, R, Teff)
    sr_predictor = SRPredictor()
    sr_y_pred = sr_predictor.predict(nu_max_val, phase)
    sr_y_pred.values
    sr_psd = PSD(frequency, nu_max_val, *(sr_y_pred.values[0]))[0]
    plots.append(
        hv.Curve(
            (frequency, sr_psd),
            label="Scaling Relations"
        ).opts(
            line_width=1.5,
            color='blue'
        )
    )

    # neural net
    X = pd.DataFrame([x])
    # X_ = X_transform.transform(X)
    # nn_y_pred = model.apply(best_params, X_, training=False)
    # nn_y_pred = y_transform.inverse_transform(nn_y_pred)
    nn_predictor = NNPredictor.get_default_predictor()
    nn_y_pred = nn_predictor.predict(X)
    nn_psd = PSD(frequency, nu_max_val, *nn_y_pred[0])[0]
    plots.append(
        hv.Curve(
            (frequency, nn_psd),
            label="Neural Net"
        ).opts(
            line_width=1.5,
            color='green'
        )
    )

    plots.append(hv.VLine(nu_max_val).opts(color='black', line_dash='dashed', line_width=1.5))
    # print(sr_y_pred["P"])

    p = hv.Overlay(plots).opts(
        logx=True, logy=True,
        width=600, height=600,
        xlabel="Frequency (uHz)",
        ylabel="Power (ppm^2/uHz)",
        title=f"KIC {KIC} Power Spectrum",
        legend_position="bottom_left"
    )
    return p

def compare_psd_bokeh(
                M: float | None = None,
                R: float | None = None,
                Teff: float | None = None,
                FeH: float | None = None,
                phase: float | None = None,
                KepMag: float | None = None,
                KIC = None,
                *,
                x: pd.Series | None = None,
                cache = _pd_cache
):
    """
    Compare the power spectral density (PSD) of a red giant derived from the
    default NNPredictor model and the scaling relations, returning a Bokeh figure.

    :param M: Stellar mass in solar units. Default is None.
    :type M: float, optional
    :param R: Stellar radius in solar units. Default is None.
    :type R: float, optional
    :param Teff: Effective temperature of the star in Kelvin. Default is None.
    :type Teff: float, optional
    :param FeH: Metallicity of the star ([Fe/H]). Default is None.
    :type FeH: float, optional
    :param phase: Evolutionary phase of the star. Default is None.
    :type phase: float, optional
    :param KepMag: Kepler magnitude of the star. Default is None.
    :type KepMag: float, optional
    :param KIC: Kepler Input Catalog (KIC) identifier for the star. Default is
        None.
    :type KIC: int, optional
    :param x: A pandas Series containing the stellar parameters (M, R, Teff,
        FeH, phase, KepMag). If provided, individual parameters should not be
        passed. Default is None.
    :type x: pd.Series, optional

    :return: A Bokeh figure containing the PSD plots for the star, including:
        * PSD predicted using scaling relations.
        * PSD predicted using a neural network.
        * The true PSD (if KIC is provided).
        * A vertical line indicating the predicted nu_max value.
    :rtype: bokeh.plotting.figure
    """
    series_given = x is not None
    values_given = None not in {M, R, Teff, FeH, phase, KepMag}
    if not series_given:
        x = pd.Series(
            data = [M, R, Teff, FeH, phase, KepMag],
            index = ["M", "R", "Teff", "FeH", "phase", "KepMag"]
        )
    elif not values_given:
        raise ValueError(
            "Must provide stellar parameters with either series (x = ) or "
            "individual parameters."
        )

    # Frequency range
    frequency = np.logspace(np.log10(5), np.log10(300))

    # Scaling relations prediction
    nu_max_val = nu_max(M, R, Teff)
    sr_predictor = SRPredictor()
    sr_y_pred = sr_predictor.predict(nu_max_val, phase)
    sr_psd = PSD(frequency, nu_max_val, *(sr_y_pred.values[0]))[0]

    # Neural network prediction
    X = pd.DataFrame([x])
    nn_predictor = NNPredictor.get_default_predictor()
    nn_y_pred = nn_predictor.predict(X)
    nn_psd = PSD(frequency, nu_max_val, *nn_y_pred[0])[0]

    # Create Bokeh figure
    bokeh_fig = figure(
        title="Power Spectrum",
        x_axis_label="Frequency (μHz)",
        y_axis_label="Power (ppm²/μHz)",
        x_axis_type="log",
        y_axis_type="log",
        x_range = (1, 300),
        width=800,
        height=800,
    )

    # KIC lookup functionality
    if KIC is not None:
        import lightkurve as lk
        if KIC in cache:
            psd = cache[KIC]
        else:
            search_result = lk.search_lightcurve(f"KIC {KIC}")
            download = search_result.download_all()
            if download is not None:
                lc = download.stitch(lambda x: x.normalize('ppm'))
                psd = lc.to_periodogram(normalization='psd')
                cache[KIC] = psd
            else:
                psd = None

    if psd is not None:
        dpd = psd.to_table().to_pandas()
        bokeh_fig.line(
            dpd["frequency"], dpd["power"], legend_label="True PSD",
            line_width=2, color="gray"
        )

        # Add binned PSD
        elements_per_bin = 50
        spd = psd.bin(elements_per_bin).to_table().to_pandas()
        bokeh_fig.line(
            spd["frequency"], spd["power"], legend_label="True PSD (Binned)",
            line_width=2, color="black"
        )

    # Add scaling relations plot
    bokeh_fig.line(
        frequency, sr_psd, legend_label="Scaling Relations",
        line_width=2, color="blue"
    )

    # Add neural network plot
    bokeh_fig.line(
        frequency, nn_psd, legend_label="Neural Net",
        line_width=2, color="green"
    )

    # Add vertical line for nu_max
    bokeh_fig.line(
        [nu_max_val, nu_max_val], [min(sr_psd.min(), nn_psd.min()), max(sr_psd.max(), nn_psd.max())],
        legend_label="nu_max", line_width=2, color="red", line_dash="dashed"
    )

    return bokeh_fig