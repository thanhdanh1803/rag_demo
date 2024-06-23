import boto3
from langchain_community.document_loaders import TextLoader, PyPDFLoader, \
    Docx2txtLoader
from langchain_community.vectorstores import OpenSearchVectorSearch, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from opensearchpy import RequestsHttpConnection, OpenSearch
from requests_aws4auth import AWS4Auth

from src.configs.constants import (AWS_REGION,
                                   AWS_OPENSEARCH_URL)

service = "es"
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(refreshable_credentials=credentials, region=AWS_REGION,
                   service=service)

opensearch_client = OpenSearch(
    hosts=[{'host': AWS_OPENSEARCH_URL, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)


def list_docs() -> list[str]:
    indexes_response = opensearch_client.indices.get_alias()
    return [index for index in indexes_response.keys() if
            not index.startswith(".")]


def document_vectorize(document_path) -> VectorStore:
    normalize_path = document_path.replace(" ", "_").lower()
    file_name = (normalize_path.split("/")[-1]).split(".")[0]
    if file_name in list_docs():
        return get_document(file_name)

    file_type = normalize_path.split(".")[-1]
    if file_type == "txt":
        text_loader = TextLoader(document_path)
    elif file_type == "pdf":
        text_loader = PyPDFLoader(document_path)
    elif file_type == "docx":
        text_loader = Docx2txtLoader(document_path)
    else:
        raise ValueError(f"Unsupported file type {file_type}")

    document = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=100)
    split_docs = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    return OpenSearchVectorSearch.from_documents(
        split_docs, embeddings,
        opensearch_url=f"https://{AWS_OPENSEARCH_URL}",
        index_name=file_name,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def get_document(doc_name) -> VectorStore:
    embeddings = OpenAIEmbeddings()
    return OpenSearchVectorSearch(
        opensearch_url=f"https://{AWS_OPENSEARCH_URL}",
        embedding_function=embeddings,
        index_name=doc_name,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
