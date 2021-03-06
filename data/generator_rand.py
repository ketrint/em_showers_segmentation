import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import deque
from opera_distance_metric import generate_k_nearest_graph, opera_distance_metric_py, generate_radius_graph
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
import click

# !cd tools/ && python setup_opera_distance_metric.py build_ext --inplace


def pmc_to_ship_format(showers_df, num_showers_in_brick):
    gb = showers_df.groupby('signal')
    showers = [gb.get_group(x) for x in gb.groups]
    showers_with_brick_id = []
    to_split = [
        238, 109, 132, 196, 162, 279, 138, 153, 226, 230, 189, 223, 192, 223, 265, 74,
    234, 155, 189, 156, 275, 187, 175, 224, 66, 158, 122, 114, 88, 87, 272, 243, 293,
    236, 246, 209, 61, 96, 248, 240, 191, 156, 146, 56, 79, 90, 260, 255, 144, 161, 75, 50, 289, 299, 130, 
    56,78,94,69,98,108,124,111,105, 130,145,167,158,172,183,192,201,210,210,230,222, 245,267,287,279,291,
    300, 51, 56, 67,89,200,143,165, 234, 289, 57,87,62,65,98, 100,200,287,76, 81, 72,93,102,106,298,121
    ]
    brick_list = []
    for i in range(len(to_split)):
        brick_list.extend(to_split[i]*[i])
    for brick, shower in zip(brick_list, showers):
        shower = shower.copy(deep=True)
        n = len(shower)
        shower['brick_id'] = brick
        showers_with_brick_id.append(shower)
    return showers_with_brick_id


def gen_one_shower(df_brick, knn=False, r=250, k=5, symmetric=False, directed=False, e=0.00005, scale=1e4):
    print('Start!')
    from opera_distance_metric import generate_k_nearest_graph, opera_distance_metric_py, generate_radius_graph
    if knn:
        edges_from, edge_to, dist = generate_k_nearest_graph(
            df_brick[["brick_id", "SX", "SY", "SZ", "TX", "TY"]].values,
            k,
            e=e,
            symmetric=symmetric, directed=directed)
        edges = np.vstack([edges_from, edge_to])
        dist = np.array(dist)
        edge_index = torch.LongTensor(edges)
    else:
        edges_from, edge_to, dist = generate_radius_graph(
            df_brick[["brick_id", "SX", "SY", "SZ", "TX", "TY"]].values,
            r,
            e=e,
            symmetric=symmetric, directed=directed)
        edges = np.vstack([edges_from, edge_to])
        dist = np.array(dist)
        edge_index = torch.LongTensor(edges)

    x = torch.FloatTensor(df_brick[["SX", "SY", "SZ", "TX", "TY"]].values / np.array([scale, scale, scale, 1., 1.]))
    shower_data = torch.FloatTensor(
        df_brick[["ele_P", "ele_SX", "ele_SY", "ele_SZ", "ele_TX", "ele_TY", "numtracks", "signal"]].values / np.array(
            [1., scale, scale, scale, 1., 1., 1., 1.]))
    edge_attr = torch.log(torch.FloatTensor(dist).view(-1, 1))
    y = torch.LongTensor(df_brick.signal.values)
    shower = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        shower_data=shower_data,
        pos=x,
        edge_attr=edge_attr,
        y=y
    )
    return shower


def gen_torch_showers(df, knn=False, r=250, k=5, symmetric=False, directed=False, e=0.00005, scale=1e4):
    df_bricks = [df[df.brick_id == brick_id] for brick_id in list(df.brick_id.unique())]
    showers = Parallel(n_jobs=10)(
        delayed(gen_one_shower)(df_brick, knn=knn, r=r, k=k, symmetric=symmetric, directed=directed, e=e, scale=scale) for df_brick in
        df_bricks)
    return showers


@click.command()
@click.option('--df_file', type=str, default='./data/showers.df')
@click.option('--output_file', type=str, default='./data/train.pt')
@click.option('--knn', type=bool, default=True)
@click.option('--k', type=int, default=10)
@click.option('--r', type=int, default=400)
@click.option('--directed', type=bool, default=False)
@click.option('--symmetric', type=bool, default=False)
@click.option('--e', type=float, default=10)
@click.option('--num_showers_in_brick', type=int, default=200)
def main(
        df_file='./data/showers.df',
        output_file='./data/train.pt',
        knn=True,
        k=10,
        r=400,
        directed=False,
        symmetric=False,
        e=10,
        num_showers_in_brick=200
):
    showers = pd.read_csv(df_file)
    showers = pmc_to_ship_format(showers, num_showers_in_brick)
    df = pd.concat(showers)
    showers = gen_torch_showers(df=df, knn=knn, k=k, r=r, symmetric=symmetric, directed=directed, e=e)
    torch.save(showers, output_file)


if __name__ == "__main__":
    main()
