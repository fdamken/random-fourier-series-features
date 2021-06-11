from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from ingredients import dataset
from rfsf.util.tensor_util import unpickle_str
from scripts.plotting.common import plot_process
from scripts.plotting.util import savefig
from scripts.util.sacred_util import load_experiment


ex, load_config, load_metrics, load_run, load_model = load_experiment()


@ex.main
def main(__figures_dir: str):
    data = dataset.load_data()
    model_states = load_metrics()["model_state"]

    model = load_model()
    steps = model_states["steps"] + ["Final"]
    state_dicts = model_states["values"] + [load_run()["result"]["model_state"]]

    print(f"Plotting GP for {len(steps)} steps.")
    filenames = []
    for step, state_dict in tqdm(zip(steps, state_dicts)):
        model.load_state_dict(unpickle_str(state_dict))
        model.eval()

        if type(step) == int:
            filename = f"{step:010d}"
        elif type(step) == str:
            filename = f"{step.lower()}"
        else:
            assert False, f"unexpected step type {type(step)}"
        plt.close(savefig(plot_process(model, data, 0, f"{dataset.get_title()}, Iter. {step}", y_lim=(-2, 2)), __figures_dir, filename, formats=["png"]))
        filenames.append(filename)

    print("Plotting finished. Generating GIF.")
    frames = [Image.open(f"{__figures_dir}/{filename}.png") for filename in tqdm(filenames)]
    frames[0].save(f"{__figures_dir}/gp.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)


if __name__ == "__main__":
    ex.run()
