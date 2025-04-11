import numpy as np
import pandas as pd

from neural_net import NNPredictor
from forward_model_lib import SRPredictor


NYQUIST = 283.2114

# PSD model per deassis
def damping(nu):
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta

def PSD(nu, nu_max, H, P, tau, alpha, reshape = True):
    if reshape:
        nu = np.reshape(nu, (1, -1))
        
        nu_max = np.reshape(nu_max, (-1, 1))
        H = np.reshape(H, (-1, 1))
        P = np.reshape(P,(-1, 1))
        tau = np.reshape(tau, (-1, 1))
        alpha = np.reshape(alpha, (-1, 1))
    
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha)
    g = excess(nu, nu_max, H)
    # g = 0
    return eta * (b + g)

def excess(nu, nu_max, H):
    FWHM = 0.66 * nu_max ** 0.88
    G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
    return G

def granulation(nu, P, tau, alpha):
    granulation = (P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha))
    if not np.isfinite(granulation).all():
        return nu * np.inf
    return granulation

def nu_max(M, R, Teff, nu_max_solar = 3090, Teff_solar = 5777):
    return nu_max_solar * M * (R **-2) * ((Teff / Teff_solar)**-0.5)

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