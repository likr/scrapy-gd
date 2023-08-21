import networkx as nx
import numpy as np
import requests
import egraph as eg
from scipy.spatial.distance import pdist, squareform


def layout(graph, distance, iterations, eps, seed):
    drawing = eg.Drawing.initial_placement(graph)
    rng = eg.Rng.seed_from(seed)
    sgd = eg.FullSgd.new_with_distance_matrix(distance)
    scheduler = sgd.scheduler(iterations, eps)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)

    return drawing


def layout_da(graph, distance, iterations, eps, l, seed):
    drawing = eg.Drawing.initial_placement(graph)
    rng = eg.Rng.seed_from(seed)
    sgd = eg.FullSgd.new_with_distance_matrix(distance)
    scheduler = sgd.scheduler(iterations, eps)

    def new_distance(i, j, d, w):
        d1 = math.hypot(drawing.x(i) - drawing.x(j), drawing.y(i) - drawing.y(j))
        d2 = distance.get(i, j)
        return l * d1 + (1 - l) * d2

    def new_weight(i, j, d, w):
        return d ** -2

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
        sgd.update_distance(new_distance)
        sgd.update_weight(new_weight)
    scheduler.run(step)

    return drawing

import json
import math
import egraph as eg


def layout_da_sparse(graph, distance, unit_edge_length, iterations, eps, l, pivot, seed):
    drawing = eg.Drawing.initial_placement(graph)
    rng = eg.Rng.seed_from(seed)
    sgd = eg.SparseSgd(graph, lambda _: unit_edge_length, pivot, rng)
    scheduler = sgd.scheduler(iterations, eps)

    def new_distance(i, j, d, w):
        d1 = math.hypot(drawing.x(i) - drawing.x(j), drawing.y(i) - drawing.y(j))
        d2 = distance.get(i, j)
        return l * d1 + (1 - l) * d2

    def new_weight(i, j, d, w):
        return d ** -2

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
        sgd.update_distance(new_distance)
        sgd.update_weight(new_weight)
    scheduler.run(step)

    return drawing


def convert_graph(nx_graph, unit_edge_length):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node({})
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], {})
    return graph, eg.all_sources_dijkstra(graph, lambda _: unit_edge_length)


def eliminate_negative_eigen_values(d, n):
    D = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            D[i, j] = d.get(i, j)
    H = np.identity(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    K = -0.5 * H @ (D ** 2) @ H
    e, v = np.linalg.eigh(K)
    pos_e = np.array([x if x >= 0 else 0 for x in e])
    X = np.sqrt(pos_e) * v
    D_a = squareform(pdist(X))
    distance_a = eg.DistanceMatrix(n)
    for i in range(n):
        for j in range(n):
            distance_a.set(i, j, D_a[i, j])
    return distance_a


def eliminate_small_eigen_values(d, n, p, min_d):
    D = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            D[i, j] = d.get(i, j)
    H = np.identity(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    K = -0.5 * H @ (D ** 2) @ H
    e, v = np.linalg.eigh(K)
    q = np.percentile(np.abs(e), p)
    e_a = np.array([ei if abs(ei) >= q else 0 for ei in e])
    X2 = np.ones((n, n), dtype=np.float32) * [sum(e_a[k] * v[i, k] * v[i, k] for k in range(n)) for i in range(n)]
    D_a = np.maximum(X2.T - 2 * v @ np.diag(e_a) @ v.T + X2, min_d * np.ones((n, n))) ** 0.5
    distance_a = eg.DistanceMatrix(n)
    for i in range(n):
        for j in range(n):
            distance_a.set(i, j, D_a[i, j] if i != j else 0)
    return distance_a


quality_metrics_keys = [
    'angular_resolution',
    'aspect_ratio',
    'crossing_angle',
    'crossing_number',
    'gabriel_graph_property',
    'ideal_edge_lengths',
    'neighborhood_preservation',
    'node_resolution',
    'stress',
]


def quality_metrics(graph, drawing, distance):
    crossing_edges = eg.crossing_edges(graph, drawing)
    return {
        'angular_resolution': eg.angular_resolution(graph, drawing),
        'aspect_ratio': eg.aspect_ratio(drawing),
        'crossing_angle': eg.crossing_angle(graph, drawing, crossing_edges),
        'crossing_number': eg.crossing_number(graph, drawing, crossing_edges),
        'gabriel_graph_property': eg.gabriel_graph_property(graph, drawing),
        'ideal_edge_lengths': eg.ideal_edge_lengths(graph, drawing, distance),
        'neighborhood_preservation': eg.neighborhood_preservation(graph, drawing),
        'node_resolution': eg.node_resolution(drawing),
        'stress': eg.stress(drawing, distance),
    }
