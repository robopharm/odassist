import os
import time
import boto3
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch,RequestsHttpConnection,helpers
from requests_aws4auth import AWS4Auth
from pprint import pprint

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)


embed_model = "amazon.titan-embed-text-v2:0"
region = "us-west-2"
index_name = "load_pdf"
embedding = BedrockEmbeddings(
    region_name = region,
    model_id = embed_model
)

service = 'es'
index_name = 'sample_index'
pdf_dir_name = "/robopharm/odassist/dbr-hackathon"
host_url='https://'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, my_region, service, session_token=credentials.token)


client = OpenSearch(
    hosts=[host_url],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    connection_class=RequestsHttpConnection,
    timeout=60,
)


def process_bulk_data(bulk_data, batch_size=300):
    """Process bulk data in smaller batches"""
    for i in range(0, len(bulk_data), batch_size):
        yield bulk_data[i:i + batch_size]

files = []
for file in os.listdir(pdf_dir_name):
    if file.endswith(".pdf"):
        files.append(os.path.join(pdf_dir_name, file))

for file in files:
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()

    bulk_data = []
    for chunk in pages:
        bulk_data.append({
            "_op_type": "index",
            "_index": index_name,
            "_source":
                {
                    "text": chunk.page_content,
                    "metadata":{
                        "source": chunk.metadata['source'],
                        "api_name": "bcancer"
                        },
                    "vector_field": embedding.embed_query(chunk.page_content)
                }
            }
        )


    for data_chunk in process_bulk_data(bulk_data):
        success, failed = helpers.bulk(client, data_chunk)
