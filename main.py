import os
import fastapi
import transformers
import math
import time
from colbert import Searcher
from minicolbert.searchers import SEARCHERS, COLLECTIONS

SUPPORTED_DATASETS = ["wikipedia", "arxiv", "stackexchange"]
TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    "answerdotai/AnswerAI-ColBERTv2.5-small"
)
VALID_TOKEN = os.environ.get("AUTH_TOKEN")
app = fastapi.FastAPI()


def _update_searcher_with_querylen(query: str, searcher: Searcher) -> Searcher:
    query_len = len(TOKENIZER.encode(query))
    max_query_len = min(query_len + 8, int(math.ceil(query_len / 16) * 16))
    max_query_len = min(max_query_len, 510)
    searcher.config.query_max_len = max_query_len
    searcher.checkpoint.query_tokenizer.query_maxlen = max_query_len

    return searcher


@app.get("/query")
def query(
    query: str,
    dataset: str,
    token: str = fastapi.Depends(fastapi.security.HTTPBearer()),
) -> fastapi.responses.JSONResponse:
    if token.credentials != VALID_TOKEN:
        return fastapi.responses.JSONResponse(
            status_code=401, content={"detail": "Invalid token"}
        )

    if dataset not in SUPPORTED_DATASETS:
        return fastapi.responses.JSONResponse(
            status_code=400, content={"detail": "Unsupported dataset"}
        )

    tick = time.time()
    searcher = SEARCHERS[dataset]
    searcher = _update_searcher_with_querylen(query, searcher)
    raw_results = searcher.search(query)
    top_document_id = raw_results[0][0]
    top_document = COLLECTIONS[dataset][str(top_document_id)]
    return {
        "document": top_document,
        "document_id": top_document_id,
        "elapsed": f"{(time.time() - tick) * 1000:.2f}ms",
    }
