from grannules import NNPredictor

import pandas as pd

print("Serializing...", end = " ")
n = NNPredictor.from_pickle("nn.pkl")
n.serialize()
print("Done.")

print("Deserializing...", end = " ")
n2 = NNPredictor._default_from_serialize()
print("Done.")

print("Running predict on deserialized...", end = " ")
test_data = pd.DataFrame({
    "M" : [1.2],
    "R" : [2],
    "Teff" : [4500.0],
    "FeH" : [-1.0],
    "KepMag" : [13.0],
    "phase" : [0]
})
print(n2.predict(test_data))
print("Done.")