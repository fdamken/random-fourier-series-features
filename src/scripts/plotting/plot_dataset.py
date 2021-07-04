from matplotlib import pyplot as plt
from sacred import Experiment

from ingredients import dataset
from ingredients.dataset import dataset_ingredient
from rfsf.util.tensor_util import to_numpy


ex = Experiment(ingredients=[dataset_ingredient])


@ex.automain
def main():
    (train_x, train_y), (test_x, test_y) = dataset.load_data()

    fig, ax = plt.subplots()
    ax.scatter(to_numpy(train_x), to_numpy(train_y), color="black", marker="*", s=100, label="Observed Data", zorder=3)
    ax.plot(to_numpy(test_x), to_numpy(test_y), color="black", label="True Func.", zorder=0)
    ax.set_title(dataset.get_title())
    ax.legend()
    fig.show()
