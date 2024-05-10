import pathway as pw

# This creates a connector that tracks files in a given directory.
data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./files/chp14.pdf",
        format="plaintext_by_file",
        mode="streaming",
        with_metadata=True,
    )
)

print(*data_sources)


# We now build the VectorStore pipeline

from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from pathway.xpacks.llm import parsers
import time

PATHWAY_PORT = 8765

# Choose document transformers
text_splitter = TokenCountSplitter()
#embedder = OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"])
embedder = LiteLLMEmbedder(model='ollama/all-minilm')

# The `PathwayVectorServer` is a wrapper over `pathway.xpacks.llm.vector_store` to accept LangChain transformers.
# Fell free to fork it to develop bespoke document processing pipelines.
vector_server = VectorStoreServer(
    *data_sources,
    parser=parsers.ParseUnstructured(),
    embedder=embedder,
    splitter=text_splitter,
)
vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=True, with_cache=False)
time.sleep(30)  # Workaround for Colab - messages from threads are not visible unless a cell is running

# You can connect to the Pathway+LlamaIndex server using any client - Pathway's, Langchain's or LlamaIndex's!
client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT+1,
)

client.query("pathway")

