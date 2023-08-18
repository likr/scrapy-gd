import datetime
import json
import os
import scrapy
import networkx as nx
import egraph as eg
import requests


WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')


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


def convert_graph(nx_graph, unit_edge_length):
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node({})
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], {})
    return graph, eg.all_sources_dijkstra(graph, lambda _: unit_edge_length)


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


class FullSgdSpider(scrapy.Spider):
    name = "full_sgd"

    def __init__(self, graph=None, unit_edge_length=30, iterations=100, eps=0.1, seed_start=0, seed_stop=100, *args, **kwargs):
        super(FullSgdSpider, self).__init__(*args, **kwargs)
        self.graphs = []
        if graph:
            self.graphs.append(graph)
        self.unit_edge_length = int(unit_edge_length)
        self.iterations = int(iterations)
        self.eps = float(eps)
        self.seed_start = int(seed_start)
        self.seed_stop = int(seed_stop)
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(FullSgdSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.engine_stopped,
                                signal=scrapy.signals.engine_stopped)
        return spider

    def start_requests(self):
        for graph in self.graphs:
            url = f'https://storage.googleapis.com/vdslab-share/graph-benchmark/json/{graph}.json'
            yield scrapy.Request(url, self.parse, meta={'graph': graph})

    def parse(self, response):
        params = {
            'iterations': self.iterations,
            'eps': self.eps,
        }
        data = json.loads(response.body.decode())
        nx_graph = nx.node_link_graph(data)
        graph, distance = convert_graph(nx_graph, self.unit_edge_length)
        for s in range(self.seed_start, self.seed_stop):
            drawing = layout(graph, distance, seed=s, **params)
            q = quality_metrics(graph, drawing, distance)
            q['unit_edge_length'] = f'uniform {self.unit_edge_length}'
            q['seed'] = s
            q['algorithm'] = 'FullSgd'
            q['graph'] = response.meta['graph']
            q['params'] = json.dumps(params)
            yield q


    def engine_stopped(self):
        requests.post(WEBHOOK_URL, json={
            'content': f'''scraped
Graph: {self.graphs[0]} (uniform edge {self.unit_edge_length})
Algorithm: FullSgd
Params: {{"iterations": {self.iterations}, "eps": {self.eps}}}
Seed: {self.seed_start}-{self.seed_stop}
https://storage.cloud.google.com/vdslab/gdresult/{self.timestamp}.jsonl
```shellsession
gsutil cp gs://vdslab/gdresult/{self.timestamp}.jsonl .
```
'''
        })
