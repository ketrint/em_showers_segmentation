import click
import numpy as np
import torch
from preprocessing import preprocess_dataset
import sys
import os


@click.command()
@click.option('--input_datafile', type=str, default='./data_new/data/200_test.pt')
@click.option('--output_datafile', type=str, default='./data_new/data/200test_preprocessed.pt')
def main(
        input_datafile='./data_new/data/200_test.pt',
        output_datafile='./data_new//data/200test_preprocessed.pt'
):
    print("Lets start")
    showers = preprocess_dataset(input_datafile)
    torch.save(showers, output_datafile)


if __name__ == "__main__":
    main()
