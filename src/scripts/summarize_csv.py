from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    file = args.file

    df = pd.read_csv(file)
    print(df.describe().round(4))


if __name__ == "__main__":
    main()
