from opensearchpy import OpenSearch, RequestsHttpConnection
import os

CLIENT = None

def get_opensearch_client() -> OpenSearch:
    global CLIENT
    if CLIENT is None:
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", 9200))
        CLIENT = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(os.getenv("OPENSEARCH_USER", "admin"), os.getenv("OPENSEARCH_PASS", "admin")),
            use_ssl=True, verify_certs=False,
            connection_class=RequestsHttpConnection,
            max_retries=3, retry_on_timeout=True, timeout=30
        )
    return CLIENT
