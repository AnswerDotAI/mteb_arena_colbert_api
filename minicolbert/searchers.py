import srsly
from colbert.infra import ColBERTConfig
from colbert import Searcher


datasets = {
    "wikipedia": "./data/wikipedia_int2doc.json",
    "arxiv": "./data/arxiv_int2doc.json",
    "stackexchange": "./data/stackexchange_int2doc.json",
}

model = "answerdotai/AnswerAI-ColBERTv2.5-small"
config = ColBERTConfig(
    nbits=2,
    nranks=1,
    root="experiments/",
    ncells=8,
    ndocs=8192,
    bsize=8,
    query_maxlen=32,
    centroid_score_threshold=0.35,
)

SEARCHERS = {}

COLLECTIONS = {
    dataset_name: srsly.read_json(dataset_path)
    for dataset_name, dataset_path in datasets.items()
}

for dataset_name in datasets.keys():
    SEARCHERS[dataset_name] = Searcher(
        index=f"{dataset_name}_V1_{model}", config=config
    )
