r"""
Scaling Relations Predictor
***************************


"""

import numpy as np
import pandas as pd

class SRPredictor():
    
    def __init__(
            self, 
            relations : dict[str, tuple[float, float]] = {},
            relations_with_phase : list[dict[str, tuple[float, float]]] = [{}, {}, {}],
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
                    "alpha" : (14.60572466, -0.42630839) # N/A using the mean of alpha
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
    

    def predict(self, nu_max, phases = None) -> pd.DataFrame:
        index = None

        if type(nu_max) is pd.Series:
            index = nu_max.index
            nu_max = nu_max.values
        
        if not nu_max is np.ndarray:
            nu_max = np.array(nu_max).reshape((-1))
        if (not phases is np.ndarray) and (not phases is None):
            phases = np.array(phases).reshape((-1))
        nu_max_rows = nu_max.reshape((-1, 1))
        if index is None:
            index = np.arange(len(nu_max_rows))
        pred = pd.DataFrame(index=index, columns=self._params)
        if self.use_phase:
            for phase in range(3):
                pred.loc[phases == phase, :] = \
                    self._coefficients[phase] * nu_max_rows[phases == phase] ** self._powers[phase]
        else:
            pred.loc[:, :] = self._coefficients * nu_max_rows ** self._powers
        return pred

pd_cache = {}
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
    import lightkurve as lk
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

    if KIC in pd_cache:
        print("Reading from cache...")
        pd = pd_cache[KIC]
    else:
        print("Searching...")
        search_result = lk.search_lightcurve(f"KIC {KIC}")
        print("Downloading...")
        lc = search_result.download_all()
        print("Processing lightcurve...")
        lc = lc.stitch(lambda x : x.normalize('ppm')) # WHY
        pd = lc.to_periodogram(normalization='psd')
        pd_cache[KIC] = pd
    # pd.power *= 2
    dpd = pd.to_table().to_pandas() # lol
    # dpd["power"] /= 2
    plots = []
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
    spd = pd.bin(elements_per_bin).to_table().to_pandas()
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
    nu_max_val = nu_max(M, R, Teff)
    sr_predictor = SRPredictor()
    sr_y_pred = sr_predictor.predict(nu_max_val, phase)
    sr_y_pred.values
    sr_psd = PSD(dpd["frequency"], nu_max_val, *(sr_y_pred.values[0]))
    plots.append(
        hv.Curve(
            (dpd["frequency"], sr_psd),
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
    nn_psd = PSD(dpd["frequency"], nu_max_val, *nn_y_pred[0])
    plots.append(
        hv.Curve(
            (dpd["frequency"], nn_psd),
            label="Neural Net"
        ).opts(
            line_width=1.5,
            color='green'
        )
    )
    
    plots.append(hv.VLine(nu_max_val).opts(color='black', line_dash='dashed', line_width=1.5))
    print(sr_y_pred["P"])

    p = hv.Overlay(plots).opts(
        logx=True, logy=True,
        width=600, height=600,
        xlabel="Frequency (uHz)",
        ylabel="Power (ppm^2/uHz)",
        title=f"KIC {KIC} Power Spectrum",
        legend_position="bottom_left"
    )
    return p