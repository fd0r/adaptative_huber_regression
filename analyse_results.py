import json
import numpy as np

# Data generated from benchmark_synthetic_data.json

with open("results.json", "r") as file:
    data = json.loads(file.read())

print("""\\begin{center}
\\begin{tabular}{ || c | c | c | c | c || }
\\hline
Noise distribution & dimension & Regressor & l2 error mean & l2 error std \\\\ [0.5ex]
\\hline\\hline""")
for noise, item in data.items():
    for dimension, sub_item in item.items():
        for reg_name, sub_data in sub_item.items():
            print(
                noise,
                dimension,
                reg_name,
                np.round(np.mean(sub_data),3),
                np.round(np.std(sub_data),3),
                sep=" \t& ",
                end="\\\\\n"
            )
            print("\\hline")
    print("\\hline")
    
print("""\\end{tabular}
\\end{center}""")