import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
import paxplot
import tikzplotlib
from matplotlib import pyplot as plt
import pandas.plotting

optuna_db = "optuna.db"
name_trial_id = "trial_id"
name_objective = "objective"
name_param_name = "param_name"
name_param_value = "param_value"
params_to_include = [name_objective, "half_period", "num_harmonics"]
df_columns_mapping = {"objective": "Objective", "half_period": "Half-Period", "num_harmonics": "Number of Harmonics"}


def main():
    con = sqlite3.connect(optuna_db)
    con.row_factory = sqlite3.Row
    cur = con.execute(
        f"""
        SELECT
            t.trial_id AS {name_trial_id},
            v.value AS {name_objective},
            p.param_name AS {name_param_name},
            p.param_value AS {name_param_value}
        FROM trials AS t
        INNER JOIN trial_values AS v ON t.trial_id = v.trial_id
        INNER JOIN trial_params AS p ON t.trial_id = p.trial_id
        WHERE t.study_id = ?
        """,
        [50]
    )

    results = defaultdict(lambda: {})
    for row in cur:
        trial_id = row[name_trial_id]
        param_name = row[name_param_name]
        trial_results = results[trial_id]
        trial_results[name_trial_id] = trial_id
        trial_results[name_objective] = row[name_objective]
        trial_results[param_name] = row[name_param_value]

    df = pd.DataFrame(results.values())[params_to_include].rename(columns=df_columns_mapping)

    cols = df.columns
    color_col = 0
    paxfig = paxplot.pax_parallel(n_axes=len(df.columns))
    paxfig.plot(df.to_numpy())
    paxfig.set_labels(df.columns)
    paxfig.set_ticks(ax_idx=0, ticks=list(np.linspace(-8, -2.8, 5)))
    paxfig.set_ticks(ax_idx=1, ticks=[0.0, 2.5, 5.0, 7.5, 10.0])
    paxfig.set_ticks(ax_idx=2, ticks=[1, 8, 16, 24, 32])
    paxfig.add_colorbar(ax_idx=color_col, cmap="coolwarm", colorbar_kwargs={"label": df.columns[color_col]})
    paxfig.savefig("report/graphics/optuna-study-parallel-coordinates.pgf")
    plt.show()


if __name__ == '__main__':
    main()
