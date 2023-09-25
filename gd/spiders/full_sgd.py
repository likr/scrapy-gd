import datetime
import json
import os
import scrapy
import networkx as nx
import egraph as eg
import requests
from .algorithm import *


WEBHOOK_URL = os.environ.get('WEBHOOK_URL', '')


class FullSgdSpider(scrapy.Spider):
    name = "full_sgd"

    def __init__(self, graph=None, unit_edge_length=30, iterations=100, eps=0.1, seed_start=0, seed_stop=100, *args, **kwargs):
        super(FullSgdSpider, self).__init__(*args, **kwargs)
        self.graph = graph
        self.algorithm = self.name
        self.unit_edge_length = int(unit_edge_length)
        self.params = {
            'iterations': int(iterations),
            'eps': float(eps),
        }
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
        if self.graph:
            url = f'https://storage.googleapis.com/vdslab-share/graph-benchmark/json/{self.graph}.json'
            yield scrapy.Request(url, self.parse)

    def parse(self, response):
        data = json.loads(response.body.decode())
        nx_graph = nx.node_link_graph(data)
        graph, distance = convert_graph(nx_graph, self.unit_edge_length)
        for s in range(self.seed_start, self.seed_stop):
            drawing = layout(graph, distance, iterations=self.params['iterations'], eps=self.params['eps'], seed=s)
            q = quality_metrics(graph, drawing, distance)
            q['edge_length'] = f'uniform {self.unit_edge_length}'
            q['seed'] = s
            q['algorithm'] = 'FullSgd'
            q['graph'] = self.graph
            q['params'] = self.params
            yield q


    def engine_stopped(self):
        if self.graph:
            filename = f'{self.algorithm}-{self.graph}-{self.timestamp}'
            requests.post(WEBHOOK_URL, json={
                'content': f'''scraped
Graph: {self.graph} (uniform edge {self.unit_edge_length})
Algorithm: {self.algorithm}
Params: {json.dumps(self.params)}
Seed: {self.seed_start}-{self.seed_stop}
https://storage.cloud.google.com/vdslab/gdresult/{filename}.jsonl
```shellsession
gsutil cp gs://vdslab/gdresult/{filename}.jsonl .
```
'''
            })
