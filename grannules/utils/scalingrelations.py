import numpy as np
import pandas as pd

from typing import Iterable

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


def main(): pass

if __name__ == '__main__': main()