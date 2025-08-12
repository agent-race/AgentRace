import agentscope
from agentscope.service import ServiceToolkit
from agentscope.service import get_help
from agentscope.service import openai_image_to_text,openai_audio_to_text,google_search,execute_python_code
from agentscope.service import ServiceResponse, ServiceExecStatus
from PyPDF2 import PdfReader
from agentscope.agents import ReActAgent,LlamaIndexAgent,DialogAgent
from agentscope.message import Msg
from agentscope.msghub import msghub
from together import Together
import json
from pydantic import BaseModel, Field
from agentscope.message.block import (
    TextBlock,
    ImageBlock,
    AudioBlock,
)
import os
import pandas as pd
from docx import Document
import time
import requests
import io

import whisper
from typing import cast
from pydub import AudioSegment
from pathlib import Path
from collections.abc import Sequence
from urllib.parse import urlparse
import logging
import dotenv
dotenv.load_dotenv()
together_api_key=os.getenv("TOGETHER_API_KEY")
openai_api_key=os.getenv("OPENAI_API_KEY")
google_api_key=os.getenv("GOOGLE_API_KEY")
google_cse_id=os.getenv("GOOGLE_CSE_ID")
together_base_url=os.getenv("TOGETHER_BASE_URL")
if together_base_url is None or together_api_key is None or openai_api_key is None or google_api_key is None or google_cse_id is None:
    raise ValueError("Please set TOGETHER_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, TOGETHER_BASE_URL and HF_TOKEN in your environment variables, you can refer to the .evn.example file for more details.")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
from functools import wraps
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))

def traced_tool(fn, tool_name=None):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from time import perf_counter
        start_time = perf_counter()

        with tracer.start_as_current_span(fn.__name__) as span:
            span.set_attribute("input", str(args))
            try:
                result = fn(*args, **kwargs)
                span.set_attribute("output", str(result)[:100])
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                duration = perf_counter() - start_time
                name = tool_name or fn.__name__
                logging.info(f"tool_name: {name}, tool_time: {duration:.10f}")
                print(f"{name} finished, time: {duration:.6f} seconds")
    return wrapper

def ReAct(name:str):
    @traced_tool
    def vedio_tool(file_path:str)-> ServiceResponse:
        """
        Convert an vedio file to text 

        Args:
            file_path (`str`):
                The file path to the vedio file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            video = AudioSegment.from_file(Path(file_path), format=file_path[-3:])
            audio = video.split_to_mono()[0]
            file_str = str(file_path)[:-4] + ".mp3"
            audio.export(file_str, format="mp3")
            model = whisper.load_model(name="base")
            model = cast(whisper.Whisper, model)
            result = model.transcribe(str(file_path))
            return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=result["text"])
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def pdf_tool(file_path: str) -> ServiceResponse:
        """
        Convert an pdf file to text 

        Args:
            file_path (`str`):
                The file path to the pdf file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            text = ""
            if urlparse(file_path).scheme in ('http', 'https'):
                response = requests.get(file_path)
                pdf_file = io.BytesIO(response.content)
                reader = PdfReader(pdf_file)
            else:
                reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
            return ServiceResponse(status=ServiceExecStatus.SUCCESS, content=text.strip())
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def csv_tool(file_path:str)->ServiceResponse:
        """
        Convert an csv file to text 

        Args:
            file_path (`str`):
                The file path to the csv file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            df = pd.read_csv(file_path)
            # 转为字符串（带表头，不带行号）
            csv_str = df.to_string(index=False)
            return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=csv_str)
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def xlsx_tool(file_path:str)->ServiceResponse:
        """
        Convert an xlsx file to text 

        Args:
            file_path (`str`):
                The file path to the xlsx file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            excel_file = pd.read_excel(file_path, sheet_name=None)
            result = ""
            for sheet_name, df in excel_file.items():
                result += f"Sheet: {sheet_name}\n"
                result += df.to_string(index=False) + "\n\n"
            return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=result.strip())
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def txt_tool(file_path:str)->ServiceResponse:
        """
        Convert an txt file to text 

        Args:
            file_path (`str`):
                The file path to the txt file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                txt_str = f.read()
            return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=txt_str)
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def docx_tool(file_path:str)->ServiceResponse:
        """
        Convert an docx file to text 

        Args:
            file_path (`str`):
                The file path to the docx file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` is the transcribed text.
        """
        try:
            doc = Document(file_path)
            docx_str = "\n".join([para.text for para in doc.paragraphs])
            return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=docx_str)
        except Exception as e:
            return ServiceResponse(ServiceExecStatus.ERROR, str(e))
        
    @traced_tool
    def audio_tool(file_path:str)->ServiceResponse:
        """
        Convert an audio file to text using OpenAI's transcription service.

        Args:
            file_path (`str`):
                The file path or URL to the audio file that needs to be
                transcribed.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` contains a dictionary with key 'transcription' and
                value as the transcribed text.
        """
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=openai_audio_to_text(file_path,api_key=openai_api_key).content)
    
    @traced_tool
    def vision_tool(file_path:str)->ServiceResponse:
        """
        Generate descriptive text for given image(s) using a specified model, and
        return the generated text.

        Args:
            file_path (`str`):
                The URL or list of URLs pointing to the images that need to be
                described.

        Returns:
            `ServiceResponse`:
                A dictionary with two variables: `status` and `content`.
                If `status` is `ServiceExecStatus.SUCCESS`,
                the `content` contains the generated text description(s).
        """
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=openai_image_to_text(file_path,api_key=openai_api_key,model="gpt-4o-2024-08-06").content)
    
    
    @traced_tool
    def web_browser_tool(query:str)->ServiceResponse:
        """
        Search question in Google Search API and return the searching results

        Args:
            query (`str`):
                The search query string.

        Returns:
            `ServiceResponse`: A dictionary with two variables: `status` and
            `content`. The `status` variable is from the ServiceExecStatus enum,
            and `content` is a list of search results or error information,
            which depends on the `status` variable.
            For each searching result, it is a dictionary with keys 'title',
            'link', and 'snippet'.
        """
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=google_search(query,api_key=google_api_key,cse_id=google_cse_id).content)
    
    
    @traced_tool
    def python_tool(code:str)->ServiceResponse:
        """
        Execute a piece of python code.

        This function can run Python code provided in string format. It has the
        option to execute the code within a Docker container to provide an
        additional layer of security, especially important when running
        untrusted code.


        Args:
            code (`str`, optional):
                The Python code to be executed.


        Returns:
            `ServiceResponse`: A ServiceResponse containing two elements:
            `output` and `error`. Both `output` and `error` are strings that
            capture the standard output and standard error of the code
            execution, respectively.

        Note:
            IPython-specific operations such as `plt.show()` for displaying
            matplotlib plots are currently not supported. This limitation stems
            from the non-interactive nature of the execution environment.

        """
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=execute_python_code(code,timeout=300,use_docker=False,maximum_memory_bytes=None).content)
    
    toolkit = ServiceToolkit()
    toolkit.add(python_tool)
    toolkit.add(csv_tool)
    toolkit.add(xlsx_tool)
    toolkit.add(docx_tool)
    toolkit.add(vedio_tool)
    toolkit.add(txt_tool)
    toolkit.add(pdf_tool)
    toolkit.add(audio_tool)
    toolkit.add(vision_tool)
    toolkit.add(web_browser_tool)
    
    # irrelevant tools
    from agents.irrelevant_tools.irrelevant_tools_agentscope import twoSum
    from agents.irrelevant_tools.irrelevant_tools_agentscope import lengthOfLongestSubstring
    from agents.irrelevant_tools.irrelevant_tools_agentscope import findMedianSortedArrays
    from agents.irrelevant_tools.irrelevant_tools_agentscope import longestPalindrome
    from agents.irrelevant_tools.irrelevant_tools_agentscope import convertZ
    from agents.irrelevant_tools.irrelevant_tools_agentscope import reverseX
    from agents.irrelevant_tools.irrelevant_tools_agentscope import myAtoi
    from agents.irrelevant_tools.irrelevant_tools_agentscope import isPalindrome
    from agents.irrelevant_tools.irrelevant_tools_agentscope import isMatch
    from agents.irrelevant_tools.irrelevant_tools_agentscope import maxArea

    from agents.irrelevant_tools.irrelevant_tools_agentscope import longestCommonPrefix
    from agents.irrelevant_tools.irrelevant_tools_agentscope import threeSum
    from agents.irrelevant_tools.irrelevant_tools_agentscope import isValidBrackets
    from agents.irrelevant_tools.irrelevant_tools_agentscope import generateParenthesis
    from agents.irrelevant_tools.irrelevant_tools_agentscope import groupAnagrams
    from agents.irrelevant_tools.irrelevant_tools_agentscope import lengthOfLastWord
    from agents.irrelevant_tools.irrelevant_tools_agentscope import addBinary
    from agents.irrelevant_tools.irrelevant_tools_agentscope import minDistance
    from agents.irrelevant_tools.irrelevant_tools_agentscope import largestNumber
    from agents.irrelevant_tools.irrelevant_tools_agentscope import reverseString

    twoSum = traced_tool(twoSum)
    lengthOfLongestSubstring = traced_tool(lengthOfLongestSubstring)
    findMedianSortedArrays = traced_tool(findMedianSortedArrays)
    longestPalindrome = traced_tool(longestPalindrome)
    convertZ = traced_tool(convertZ)
    reverseX = traced_tool(reverseX)
    myAtoi = traced_tool(myAtoi)
    isPalindrome = traced_tool(isPalindrome)
    isMatch = traced_tool(isMatch)
    maxArea = traced_tool(maxArea)
    longestCommonPrefix = traced_tool(longestCommonPrefix)
    threeSum = traced_tool(threeSum)
    isValidBrackets = traced_tool(isValidBrackets)
    generateParenthesis = traced_tool(generateParenthesis)
    groupAnagrams = traced_tool(groupAnagrams)
    lengthOfLastWord = traced_tool(lengthOfLastWord)
    addBinary = traced_tool(addBinary)
    minDistance = traced_tool(minDistance)
    largestNumber = traced_tool(largestNumber)
    reverseString = traced_tool(reverseString)


    # toolkit.add(twoSum)
    # toolkit.add(lengthOfLongestSubstring)
    # toolkit.add(findMedianSortedArrays)
    # toolkit.add(longestPalindrome)
    # toolkit.add(convertZ)
    # toolkit.add(reverseX)
    # toolkit.add(myAtoi)
    # toolkit.add(isPalindrome)
    # toolkit.add(isMatch)
    # toolkit.add(maxArea)

    # toolkit.add(longestCommonPrefix)
    # toolkit.add(threeSum)
    # toolkit.add(isValidBrackets)
    # toolkit.add(generateParenthesis)
    # toolkit.add(groupAnagrams)
    # toolkit.add(lengthOfLastWord)
    # toolkit.add(addBinary)
    # toolkit.add(minDistance)
    # toolkit.add(largestNumber)
    # toolkit.add(reverseString)


    ReAct_Agent = ReActAgent(
        name="Friday",
        model_config_name=name,
        service_toolkit=toolkit,
        sys_prompt="你是一个名为 Friday 的助手",
        max_iters=10,
        verbose=True,
    )
    return ReAct_Agent


# -*- coding: utf-8 -*-
"""
This module is an integration of the Llama index RAG
into AgentScope package
"""
retrieved_time=[]

import copy
import os.path
from typing import Any, Optional, List, Union
from loguru import logger

try:
    import llama_index
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding,
        Embedding,
    )
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.legacy.retrievers.bm25_retriever import BM25Retriever

    from llama_index.core.vector_stores.types import VectorStore
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core.schema import (
        Document,
        TransformComponent,
        BaseNode,
    )
except ImportError:
    llama_index = None
    BaseRetriever = None
    BaseEmbedding = None
    Embedding = None
    IngestionPipeline = None
    SimpleDocumentStore = None
    BM25Retriever = None
    VectorStore = None
    SentenceSplitter = None
    VectorStoreIndex = None
    StorageContext = None
    load_index_from_storage = None
    PrivateAttr = None
    Document = None
    TransformComponent = None
    BaseNode = None

from agentscope.manager import FileManager, ModelManager
from agentscope.models import ModelWrapperBase
from agentscope.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from agentscope.rag import Knowledge, RetrievedChunk


try:

    class _EmbeddingModel(BaseEmbedding):
        """
        Wrapper for ModelWrapperBase to an embedding model can be used
        in Llama Index pipeline.
        """

        _emb_model_wrapper: ModelWrapperBase = PrivateAttr()

        def __init__(
            self,
            emb_model: ModelWrapperBase,
            embed_batch_size: int = 1,
        ) -> None:
            """
            Dummy wrapper to convert a ModelWrapperBase to llama Index
            embedding model

            Args:
                emb_model (`ModelWrapperBase`):
                    Embedding model in ModelWrapperBase
                embed_batch_size (`int`):
                    Batch size, defaults to 1
            """
            super().__init__(
                model_name="Temporary_embedding_wrapper",
                embed_batch_size=embed_batch_size,
            )
            self._emb_model_wrapper = emb_model

        def _get_query_embedding(self, query: str) -> List[float]:
            """
            Get embedding for query

            Args:
                query (`str`): Query to be embedded

            Returns:
                `List[float]`: Embedding
            """
            # Note: AgentScope embedding model wrapper returns list
            # of embedding
            return list(self._emb_model_wrapper(query).embedding[0])

        def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
            """
            Get embedding for list of strings

            Args:
                 texts (` List[str]`): Texts to be embedded

            Returns:
                `List[float]`: List of embeddings
            """
            results = [
                list(self._emb_model_wrapper(t).embedding[0]) for t in texts
            ]
            return results

        def _get_text_embedding(self, text: str) -> Embedding:
            """
            Get embedding for a single string
            Args:
                 text (`str`): Texts to be embedded

            Returns:
                `List[float]`: Embedding
            """
            return list(self._emb_model_wrapper(text).embedding[0])

        # TODO: use proper async methods, but depends on model wrapper
        async def _aget_query_embedding(self, query: str) -> List[float]:
            """The asynchronous version of _get_query_embedding."""
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            """Asynchronously get text embedding."""
            return self._get_text_embedding(text)

        async def _aget_text_embeddings(
            self,
            texts: List[str],
        ) -> List[List[float]]:
            """Asynchronously get text embeddings."""
            return self._get_text_embeddings(texts)

except Exception:

    class _EmbeddingModel:  # type: ignore[no-redef]
        """
        A dummy embedding model for passing tests when
        llama-index is not install
        """

        def __init__(self, emb_model: ModelWrapperBase):
            self._emb_model_wrapper = emb_model


class LlamaIndexKnowledge(Knowledge):
    """
    This class is a wrapper with the llama index RAG.
    """

    knowledge_type: str = "llamaindex_knowledge"

    def __init__(
        self,
        knowledge_id: str,
        emb_model: Union[ModelWrapperBase, BaseEmbedding, None] = None,
        knowledge_config: Optional[dict] = None,
        model: Optional[ModelWrapperBase] = None,
        persist_root: Optional[str] = None,
        additional_sparse_retrieval: Optional[bool] = False,
        overwrite_index: Optional[bool] = False,
        showprogress: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the knowledge component based on the
        llama-index framework: https://github.com/run-llama/llama_index

        Notes:
            In LlamaIndex, one of the most important concepts is index,
            which is a data structure composed of Document objects, designed to
            enable querying by an LLM. The core workflow of initializing RAG is
            to convert data to index, and retrieve information from index.
            For example:
            1) preprocessing documents with data loaders
            2) generate embedding by configuring pipeline with embedding models
            3) store the embedding-content to vector database
                the default dir is " ~/.cache/agentscope/{knowledge_id}"

        Args:
            knowledge_id (`str`):
                The id of the RAG knowledge unit.
            emb_model (`ModelWrapperBase`):
                The embedding model used for generate embeddings
            knowledge_config (`dict`):
                The configuration for llama-index to
                generate or load the index.
            model (`ModelWrapperBase`):
                The language model used for final synthesis
            persist_root (`str`):
                The root directory for index persisting
            overwrite_index (`Optional[bool]`):
                Whether to overwrite the index while refreshing
            showprogress (`Optional[bool]`):
                Whether to show the indexing progress
        """
        super().__init__(
            knowledge_id=knowledge_id,
            emb_model=emb_model,
            knowledge_config=knowledge_config,
            model=model,
            **kwargs,
        )
        if llama_index is None:
            raise ImportError(
                "LlamaIndexKnowledge require llama-index installed. "
                "Try a stable llama-index version, such as "
                "`pip install llama-index==0.10.30`",
            )

        if persist_root is None:
            persist_root = FileManager.get_instance().cache_dir or "./"
        self.persist_dir = os.path.join(persist_root, knowledge_id)
        logger.info(f"** persist_dir: {self.persist_dir}")
        self.emb_model = emb_model
        self.overwrite_index = overwrite_index
        self.showprogress = showprogress
        self.index = None

        # if use mix retrieval with bm25 in addition to dense retrieval
        self.additional_sparse_retrieval = additional_sparse_retrieval
        self.bm25_retriever = None

        # ensure the emb_model is compatible with LlamaIndex
        if isinstance(emb_model, ModelWrapperBase):
            self.emb_model = _EmbeddingModel(emb_model)
        elif isinstance(self.emb_model, BaseEmbedding):
            pass
        else:
            raise TypeError(
                f"Embedding model does not support {type(self.emb_model)}.",
            )
        # then we can initialize the RAG
        self._init_rag()

    def _init_rag(self, **kwargs: Any) -> None:
        """
        Initialize the RAG. This includes:
            * if the persist_dir exists, load the persisted index
            * if not, convert the data to index
            * if needed, update the index
            * set the retriever to retrieve information from index

        Notes:
            * the index is persisted in the self.persist_dir
            * the refresh_index method is placed here for testing, it can be
                called externally. For example, updated the index periodically
                by calling rag.refresh_index() during the execution of the
                agent.
        """
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)
        try:
            self._load_index()
        except Exception as e:
            logger.warning(
                f"index loading error: {str(e)}, recomputing index...",
            )
            self._data_to_index()
        self._get_retriever()
        logger.info(
            f"RAG with knowledge ids: {self.knowledge_id} "
            f"initialization completed!\n",
        )

    def _load_index(self) -> None:
        """
        Load the persisted index from persist_dir.
        """
        # load the storage_context
        storage_context = StorageContext.from_defaults(
            persist_dir=self.persist_dir,
        )
        # construct index from
        self.index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.emb_model,
        )
        logger.info(f"index loaded from {self.persist_dir}")

    def _data_to_index(
        self,
        vector_store: Optional[VectorStore] = None,
    ) -> List[BaseNode]:
        """
        Convert the data to index by configs. This includes:
            * load the d_data_to_ata to documents by using information
              from configs
            * set the transformations associated with documents
            * convert the documents to nodes
            * convert the nodes to index

        Notes:
            As each selected file type may need to use a different loader
            and transformations, knowledge_config is a list of configs.

        Args:
            vector_store (`Optional[VectorStore]`):
                Vector store in LlamaIndex

        Returns:
           ` List[BaseNode]`: list of processed nodes
        """
        nodes = []
        # load data to documents and set transformations
        # using information in knowledge_config
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            nodes_docs = self._docs_to_nodes(
                documents=documents,
                transformations=transformations,
            )
            nodes = nodes + nodes_docs
        # convert nodes to index
        if vector_store is None:
            self.index = VectorStoreIndex(
                nodes=nodes,
                embed_model=self.emb_model,
            )
            logger.info("index calculation completed.")
            # persist the calculated index
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("index persisted.")
        else:
            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            storage_context = StorageContext.from_defaults(
                docstore=docstore,
                vector_store=vector_store,
            )
            self.index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.emb_model,
            )
            logger.info("[Update Mode] Added documents to VDB")
            storage_context.docstore.persist(
                os.path.join(self.persist_dir, "docstore.json"),
            )

        return nodes

    def _data_to_docs(
        self,
        query: Optional[str] = None,
        config: dict = None,
    ) -> Any:
        """
        This method set the loader as needed, or just use the
        default setting. Then use the loader to load data from
        dir to documents.

        Notes:
            We can use simple directory loader (SimpleDirectoryReader)
            to load general documents, including Markdown, PDFs,
            Word documents, PowerPoint decks, images, audio and video.
            Or use SQL loader (DatabaseReader) to load database.

        Args:
            query (`Optional[str]`):
                Optional, used when the data is in a database.
            config (`dict`):
                Optional, used when the loader config is in a config file.
        Returns:
            `Any`: loaded documents
        """
        loader = self._set_loader(config=config).get("loader")
        # let the doc_id be the filename for each document
        loader.filename_as_id = True
        if query is None:
            documents = loader.load_data()
        else:
            # this is for querying a database,
            # does not work for loading a document directory
            documents = loader.load_data(query)
        logger.info(f"loaded {len(documents)} documents")
        return documents

    def _docs_to_nodes(
        self,
        documents: List[Document],
        transformations: Optional[list[Optional[TransformComponent]]] = None,
    ) -> Any:
        """
        Convert the loaded documents to nodes using transformations.

        Args:
            documents (`List[Document]`):
                Documents to be processed, usually expected to be in
                 llama index Documents.
            transformations (`Optional[list[TransformComponent]]`):
                Optional, specifies the transformations (operators) to
                process documents (e.g., split the documents into smaller
                chunks)
        Return:
            `Any`: Return the index of the processed document
        """
        # nodes, or called chunks, is a presentation of the documents
        # we build nodes by using the IngestionPipeline
        # for each document with corresponding transformations
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # stack up the nodes from the pipeline
        nodes = pipeline.run(
            documents=documents,
            show_progress=self.showprogress,
        )
        logger.info("nodes generated.")
        return nodes

    def _set_loader(self, config: dict) -> Any:
        """
        Set the loader as needed, or just use the default setting.

        Args:
            config (`dict`): A dictionary containing configurations
        """
        if "load_data" in config:
            # we prepare the loader from the configs
            loader = self._prepare_args_from_config(
                config=config.get("load_data", {}),
            )
        else:
            # we prepare the loader by default
            try:
                from llama_index.core import SimpleDirectoryReader
            except ImportError as exc_inner:
                raise ImportError(
                    " LlamaIndexAgent requires llama-index to be install."
                    "Please run `pip install llama-index`",
                ) from exc_inner
            loader = {
                "loader": SimpleDirectoryReader(
                    input_dir="set_default_data_path",
                ),
            }
        logger.info("loaders are ready.")
        return loader

    def _set_transformations(self, config: dict) -> Any:
        """
        Set the transformations as needed, or just use the default setting.

        Args:
            config (`dict`): A dictionary containing configurations.
        """
        if "store_and_index" in config:
            temp = self._prepare_args_from_config(
                config=config.get("store_and_index", {}),
            )
            transformations = temp.get("transformations")
        else:
            transformations = [
                SentenceSplitter(
                    chunk_size=self.knowledge_config.get(
                        "chunk_size",
                        DEFAULT_CHUNK_SIZE,
                    ),
                    chunk_overlap=self.knowledge_config.get(
                        "chunk_overlap",
                        DEFAULT_CHUNK_OVERLAP,
                    ),
                ),
            ]
        # adding embedding model as the last step of transformation
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html
        transformations.append(self.emb_model)
        logger.info("transformations are ready.")
        # as the last step, we need to repackage the transformations in dict
        transformations = {"transformations": transformations}
        return transformations

    def _get_retriever(
        self,
        similarity_top_k: int = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Set the retriever as needed, or just use the default setting.

        Args:
            retriever (`Optional[BaseRetriever]`):
                Passing a retriever in LlamaIndexKnowledge
            rag_config (`dict`):
                RAG configuration, including similarity top k index.
        """
        # set the retriever
        logger.info(
            f"similarity_top_k" f"={similarity_top_k or DEFAULT_TOP_K}",
        )
        retriever = self.index.as_retriever(
            embed_model=self.emb_model,
            similarity_top_k=similarity_top_k or DEFAULT_TOP_K,
            **kwargs,
        )

        if not self.bm25_retriever:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.index.docstore.docs.values(),
                similarity_top_k=similarity_top_k,
            )
        else:
            self.bm25_retriever.similarity_top_k = similarity_top_k

        logger.info("retriever is ready.")

        return retriever

    def retrieve(
        self,
        query: str,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> list[Union[RetrievedChunk, str]]:
        """
        This is a basic retrieve function for knowledge.
        It will build a retriever on the fly and return the
        result of the query.
        Args:
            query (`str`):
                Query is expected to be a question in string
            similarity_top_k (`int`):
                The number of most similar data returned by the
                retriever.
            to_list_strs (`bool`):
                Whether returns the list of strings;
                if False, return list of RetrievedChunk
            retriever (`BaseRetriever`):
                For advanced usage, user can pass their own retriever.
        Return:
            `list[Union[RetrievedChunk, str]]`: List of retrieved content

        More advanced query processing can refer to
        https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook.html
        """
        start=time.perf_counter()
        if retriever is None:
            retriever = self._get_retriever(similarity_top_k)
        dense_retrieved = retriever.retrieve(str(query))
        retrieved_res = []

        for node in dense_retrieved:
            retrieved_res.append(
                RetrievedChunk(
                    score=node.score,
                    content=node.get_content(),
                    metadata=node.metadata,
                    embedding=node.embedding,
                    hash=node.node.hash,
                ),
            )

        if self.additional_sparse_retrieval and self.bm25_retriever:
            bm25_retrieved = self.bm25_retriever.retrieve(str(query))
            sparse_retrieved = [x for x in bm25_retrieved if x.score > 0]
            bm25_scores = [x.score for x in bm25_retrieved]
            logger.info(f"bm25 scores {bm25_scores}")
            for node in sparse_retrieved:
                retrieved_res.append(
                    RetrievedChunk(
                        score=node.score,
                        content=node.get_content(),
                        metadata=node.metadata,
                        embedding=node.embedding,
                        hash=node.node.hash,
                    ),
                )

        if to_list_strs:
            results = []
            for chunk in retrieved_res:
                results.append(str(chunk.content))
            return results
        end=time.perf_counter()
        retrieved_time.append(end-start)
        return retrieved_res

    def refresh_index(self) -> None:
        """
        Refresh the index when needed.
        """
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            # store and indexing for each file type
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            self._insert_docs_to_index(
                documents=documents,
                transformations=transformations,
            )

    def _insert_docs_to_index(
        self,
        documents: List[Document],
        transformations: TransformComponent,
    ) -> None:
        """
        Add documents to the index. Given a list of documents, we first test if
        the doc_id is already in the index. If not, we add the doc to the
        list. If yes, and the over-write flag is enabled,
        we delete the old doc and add the new doc to the list.
        Lastly, we generate nodes for all documents on the list, and insert
        the nodes to the index.

        Args:
            documents (`List[Document]`):
                List of documents to be added.
            transformations (`TransformComponent`):
                Transformations that onvert the documents into nodes.
        """
        # this is the pipeline that generate the nodes
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # we need to generate nodes from this list of documents
        insert_docs_list = []
        for doc in documents:
            if doc.doc_id not in self.index.ref_doc_info.keys():
                # if the doc_id is not in the index, we add it to the list
                insert_docs_list.append(doc)
                logger.info(
                    f"add new documents to index, " f"doc_id={doc.doc_id}",
                )
            else:
                if self.overwrite_index:
                    # if we enable overwrite index, we delete the old doc
                    self.index.delete_ref_doc(
                        ref_doc_id=doc.doc_id,
                        delete_from_docstore=True,
                    )
                    # then add the same doc to the list
                    insert_docs_list.append(doc)
                    logger.info(
                        f"replace document in index, " f"doc_id={doc.doc_id}",
                    )
        logger.info("documents scan completed.")
        # we generate nodes for documents on the list
        nodes = pipeline.run(
            documents=insert_docs_list,
            show_progress=True,
        )
        logger.info("nodes generated.")
        # insert the new nodes to index
        self.index.insert_nodes(nodes=nodes)
        logger.info("nodes inserted to index.")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def _delete_docs_from_index(
        self,
        documents: List[Document],
    ) -> None:
        """
        Delete the nodes that are associated with a list of documents.

        Args:
            documents (`List[Document]`): List of documents to be deleted.
        """
        doc_id_list = [doc.doc_id for doc in documents]
        for key in self.index.ref_doc_info.keys():
            if key in doc_id_list:
                self.index.delete_ref_doc(
                    ref_doc_id=key,
                    delete_from_docstore=True,
                )
                logger.info(f"docs deleted from index, doc_id={key}")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("nodes delete completed.")

    @classmethod
    def default_config(
        cls,
        knowledge_id: str,
        data_dirs_and_types: dict[str, list[str]] = None,
        knowledge_config: Optional[dict] = None,
    ) -> dict:
        """
        Generate default config for loading data from directories and using the
        default operations to preprocess the data for RAG usage.
        Args:
            knowledge_id (`str`):
                User-defined unique id for the knowledge
            data_dirs_and_types (`dict[str, list[str]]`):
                Dictionary of data paths (keys) to the data types
                (file extensions) for knowledgebase
                (e.g., [".md", ".py", ".html"])
            knowledge_config (`optional[dict]`):
                Complete indexing configuration, used for more advanced
                applications. Users can customize
                - loader,
                - transformations,
                - ...
                Examples can refer to../examples/conversation_with_RAG_agents/

        Returns:
            `dict`: A default config of LlamaIndexKnowledge
        """
        data_dirs_and_types = (
            data_dirs_and_types if data_dirs_and_types else {}
        )

        default_knowledge_config = {
            "knowledge_id": "",
            "data_processing": [],
        }
        default_loader_config = {
            "load_data": {
                "loader": {
                    "create_object": True,
                    "module": "llama_index.core",
                    "class": "SimpleDirectoryReader",
                    "init_args": {},
                },
            },
        }
        default_init_config = {
            "input_dir": "",
            "recursive": True,
            "required_exts": [],
        }
        # generate default knowledge config
        default_knowledge_config["knowledge_id"] = knowledge_id
        for data_dir, types in data_dirs_and_types.items():
            loader_config = copy.deepcopy(default_loader_config)
            loader_init = copy.deepcopy(default_init_config)
            loader_init["input_dir"] = data_dir
            loader_init["required_exts"] = types
            loader_config["load_data"]["loader"]["init_args"] = loader_init
            default_knowledge_config["data_processing"].append(loader_config)

        if knowledge_config is None:
            return default_knowledge_config
        else:
            default_knowledge_config.update(knowledge_config)
            return default_knowledge_config

    @classmethod
    def build_knowledge_instance(
        cls,
        knowledge_id: str,
        knowledge_config: Optional[dict] = None,
        data_dirs_and_types: dict[str, list[str]] = None,
        emb_model_config_name: Optional[str] = None,
        model_config_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Knowledge:
        """
        Building an instance of the LlamaIndex knowledge

        Args:
            knowledge_id (`str`):
                User-defined unique id for the knowledge
            knowledge_config (`optional[dict]`):
                Complete indexing configuration, used for more advanced
                applications. Users can customize
                - loader,
                - transformations,
                - ...
                Examples can refer to../examples/conversation_with_RAG_agents/
            data_dirs_and_types (`dict[str, list[str]]`):
                Dictionary of data paths (keys) to the data types
                (file extensions) for knowledge
                (e.g., [".md", ".py", ".html"])
            emb_model_config_name (`Optional[str]`):
                Name of the embedding model.
                This should be specified here or in the knowledge_config dict.
                If specified both here and in the knowledge_config,
                the input parameter takes a higher priority than the
                one knowledge_config.
            model_config_name (`Optional[str]`):
                Name of the language model.
                Optional, can be None and not specified in knowledge_config.
                If specified both here and in the knowledge_config,
                the input parameter takes a higher priority than the
                one knowledge_config.

        Returns:
            `Knowledge`: A knowledge instance

            A simple example of importing data to Knowledge object:

            .. code-block:: python

                knowledge_bank.add_data_as_knowledge(
                    knowledge_id="agentscope_tutorial_rag",
                    emb_model_config_name="qwen_emb_config",
                    data_dirs_and_types={
                        "../../docs/sphinx_doc/en/source/tutorial": [".md"],
                    },
                    persist_dir="./rag_storage/tutorial_assist",
                )

        """
        model_manager = ModelManager.get_instance()
        if emb_model_config_name is None and (
            knowledge_config is None
            or "emb_model_config_name" not in knowledge_config
        ):
            raise ValueError(
                "Must specify embedding model by providing value to"
                "'emb_model_config_name' key in  in knowledge config"
                "of LlamaIndexKnowledge. For example"
                """
                {
                    "knowledge_id": "xxx_rag",
                    "knowledge_type": "llamaindex_knowledge",
                    "emb_model_config_name": "qwen_emb_config",
                    ....
                }
                """,
            )
        if emb_model_config_name is None:
            emb_model_config_name = knowledge_config.get(
                "emb_model_config_name",
            )
        # model_name is optional
        if knowledge_config is not None and model_config_name is None:
            model_config_name = knowledge_config.get("model_config_name")
        knowledge_config = cls.default_config(
            knowledge_id=knowledge_id,
            data_dirs_and_types=data_dirs_and_types,
            knowledge_config=knowledge_config,
        )
        return cls(
            knowledge_id=knowledge_id,
            emb_model=model_manager.get_model_by_config_name(
                emb_model_config_name,
            ),
            knowledge_config=knowledge_config,
            model=model_manager.get_model_by_config_name(model_config_name)
            if model_config_name
            else None,
            **kwargs,
        )
RAG_LLM_TIME=[]
class TimedModelWrapper:
    def __init__(self, original_model):
        self.model = original_model

    def __call__(self, prompt):
        start_time = time.perf_counter()
        result = self.model(prompt)
        end_time = time.perf_counter()
        RAG_LLM_TIME.append(end_time-start_time)
        return result  # 保持返回结果不变
    def format(self,*args: Union[Msg, list[Msg], None],multi_agent_mode: bool = True,):
        return self.model.format(*args)

def RAG(name:str):
    agentscope.init(
        model_configs=[
        {
            "model_type": "openai_embedding",
            "config_name": "embeddings",
            "model_name": "text-embedding-3-large",#需要能返回usage模块
            "api_key":  openai_api_key,
            #"client_args": {"base_url":"" , },
        },],
        use_monitor=True,  # 是否监控 token 使用情况
        save_code=False,  # 是否保存此次运行的代码
        save_log=False,  # 是否保存日志
        save_dir="./results",  # 保存目录
    )
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    start=time.perf_counter()
    local_knowledge = LlamaIndexKnowledge.build_knowledge_instance(
        knowledge_id="agentscope_mmlu_db",
        data_dirs_and_types={os.path.join(parent_dir,"data","MMLU","dev"): [".csv"]},
        emb_model_config_name="embeddings",
    )
    end=time.perf_counter()
    RAG_Agent =LlamaIndexAgent(
        name="Friday",
        sys_prompt="你是一个名为 Friday 的助手。",
        model_config_name=name,
        knowledge_list=[local_knowledge],
        knowledge_id_list=["agentscope_mmlu_db"],
        similarity_top_k=5,
    )
    RAG_Agent.model=TimedModelWrapper(RAG_Agent.model)
    return [RAG_Agent,end-start]
    
def MoA(name,moa_scale):
    agentscope.init(
        model_configs=[
        {
            "model_type": "openai_chat",
            "config_name": "workAI1",
            "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "api_key": together_api_key  ,# API 密钥
            "client_args": {"base_url": together_base_url,  },
            "generate_args": {"temperature": 0,},
        },
        {
            "model_type": "openai_chat",
            "config_name": "workAI2",
            "model_name": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "api_key": together_api_key  ,# API 密钥
            "client_args": {"base_url": together_base_url,  },
            "generate_args": {"temperature": 0,  },
        },
        {
            "model_type": "openai_chat",
            "config_name": "workAI3",
            "model_name": "deepseek-ai/DeepSeek-V3",
            "api_key": together_api_key  ,# API 密钥
            "client_args": {"base_url": together_base_url,  },
            "generate_args": {"temperature": 0, },
        },
        ],
        use_monitor=False,  # 是否监控 token 使用情况
        save_code=False,  # 是否保存此次运行的代码
        save_log=False,  # 是否保存日志
        save_dir="./results",  # 保存目录
    )
    if moa_scale==3:
        alice = DialogAgent(
            name="Alice",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice 的助手。",
        )
        bob = DialogAgent(
            name="Bob",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob 的助手。",
        )
        charles = DialogAgent(
            name="Charles",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles 的助手。",
        )
        dave = DialogAgent(
            name="Dave",
            model_config_name=name,
            sys_prompt="You are an assistant called Dave,you should synthesize the answers from Alice, Bob, and Charles to arrive at the final response.",
        )
        
        return [alice,bob,charles,dave]
    elif moa_scale==6:
        alice1 = DialogAgent(
            name="Alice1",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice1 的助手。",
        )
        bob1 = DialogAgent(
            name="Bob1",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob1 的助手。",
        )
        charles1 = DialogAgent(
            name="Charles1",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles1 的助手。",
        )
        alice2 = DialogAgent(
            name="Alice2",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice2 的助手。",
        )
        bob2 = DialogAgent(
            name="Bob2",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob2 的助手。",
        )
        charles2 = DialogAgent(
            name="Charles2",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles2 的助手。",
        )

        dave = DialogAgent(
            name="Dave",
            model_config_name=name,
            sys_prompt="You are an assistant called Dave,you should synthesize the answers from Alice1, Bob1, Charles1, Alice2, Bob2 and Charles2 to arrive at the final response.",
        )
        return [alice1,bob1,charles1,alice2,bob2,charles2,dave]
    elif moa_scale==9:
        alice1 = DialogAgent(
            name="Alice1",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice1 的助手。",
        )
        bob1 = DialogAgent(
            name="Bob1",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob1 的助手。",
        )
        charles1 = DialogAgent(
            name="Charles1",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles1 的助手。",
        )
        alice2 = DialogAgent(
            name="Alice2",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice2 的助手。",
        )
        bob2 = DialogAgent(
            name="Bob2",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob2 的助手。",
        )
        charles2 = DialogAgent(
            name="Charles2",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles2 的助手。",
        )
        alice3 = DialogAgent(
            name="Alice3",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice3 的助手。",
        )
        bob3 = DialogAgent(
            name="Bob3",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob3 的助手。",
        )
        charles3 = DialogAgent(
            name="Charles3",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles3 的助手。",
        )

        dave = DialogAgent(
            name="Dave",
            model_config_name=name,
            sys_prompt="You are an assistant called Dave,you should synthesize the answers from Alice1, Bob1, Charles1, Alice2, Bob2, Charles2, Alice3, Bob3 and Charles3 to arrive at the final response.",
        )
        return [alice1,bob1,charles1,alice2,bob2,charles2,alice3,bob3,charles3,dave]
    elif moa_scale==12:
        alice1 = DialogAgent(
            name="Alice1",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice1 的助手。",
        )
        bob1 = DialogAgent(
            name="Bob1",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob1 的助手。",
        )
        charles1 = DialogAgent(
            name="Charles1",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles1 的助手。",
        )
        alice2 = DialogAgent(
            name="Alice2",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice2 的助手。",
        )
        bob2 = DialogAgent(
            name="Bob2",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob2 的助手。",
        )
        charles2 = DialogAgent(
            name="Charles2",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles2 的助手。",
        )
        alice3 = DialogAgent(
            name="Alice3",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice3 的助手。",
        )
        bob3 = DialogAgent(
            name="Bob3",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob3 的助手。",
        )
        charles3 = DialogAgent(
            name="Charles3",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles3 的助手。",
        )
        alice4 = DialogAgent(
            name="Alice4",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice4 的助手。",
        )
        bob4 = DialogAgent(
            name="Bob4",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob4 的助手。",
        )
        charles4 = DialogAgent(
            name="Charles4",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles4 的助手。",
        )

        dave = DialogAgent(
            name="Dave",
            model_config_name=name,
            sys_prompt="You are an assistant called Dave,you should synthesize the answers from Alice1, Bob1, Charles1, Alice2, Bob2, Charles2, Alice3, Bob3, Charles3, Alice4, Bob4 and Charles4 to arrive at the final response.",
        )
        return [alice1,bob1,charles1,alice2,bob2,charles2,alice3,bob3,charles3,alice4,bob4,charles4,dave]
    elif moa_scale==15:
        alice1 = DialogAgent(
            name="Alice1",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice1 的助手。",
        )
        bob1 = DialogAgent(
            name="Bob1",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob1 的助手。",
        )
        charles1 = DialogAgent(
            name="Charles1",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles1 的助手。",
        )
        alice2 = DialogAgent(
            name="Alice2",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice2 的助手。",
        )
        bob2 = DialogAgent(
            name="Bob2",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob2 的助手。",
        )
        charles2 = DialogAgent(
            name="Charles2",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles2 的助手。",
        )
        alice3 = DialogAgent(
            name="Alice3",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice3 的助手。",
        )
        bob3 = DialogAgent(
            name="Bob3",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob3 的助手。",
        )
        charles3 = DialogAgent(
            name="Charles3",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles3 的助手。",
        )
        alice4 = DialogAgent(
            name="Alice4",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice4 的助手。",
        )
        bob4 = DialogAgent(
            name="Bob4",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob4 的助手。",
        )
        charles4 = DialogAgent(
            name="Charles4",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles4 的助手。",
        )
        alice5 = DialogAgent(
            name="Alice5",
            model_config_name="workAI1",
            sys_prompt="你是一个名叫 Alice5 的助手。",
        )
        bob5 = DialogAgent(
            name="Bob5",
            model_config_name="workAI2",
            sys_prompt="你是一个名叫 Bob5 的助手。",
        )
        charles5 = DialogAgent(
            name="Charles5",
            model_config_name="workAI3",
            sys_prompt="你是一个名叫 Charles5 的助手。",
        )

        dave = DialogAgent(
            name="Dave",
            model_config_name=name,
            sys_prompt="You are an assistant called Dave,you should synthesize the answers from Alice1, Bob1, Charles1, Alice2, Bob2, Charles2, Alice3, Bob3, Charles3, Alice4, Bob4, Charles4, Alice5, Bob5 and Charles5 to arrive at the final response.",
        )
        return [alice1,bob1,charles1,alice2,bob2,charles2,alice3,bob3,charles3,alice4,bob4,charles4,alice5,bob5,charles5,dave]



def AgentScopeAgent(agent_type,api_key=None,moa_scale=3):
    agentscope.init(
        model_configs=[
        {
            "model_type": "openai_chat",
            "config_name": "openAI",
            "model_name": "gpt-4o-2024-08-06",
            "api_key": api_key  ,# API 密钥
            "client_args": {"base_url":os.environ['OPENAI_BASE_URL'] , },
            "generate_args": {"temperature": 0,},
        },],
        use_monitor=True,  # 是否监控 token 使用情况
        save_code=False,  # 是否保存此次运行的代码
        save_log=False,  # 是否保存日志
        save_dir="./results",  # 保存目录
    )
    name="openAI"
    if agent_type == "ReAct":
        return ReAct(name)
    elif agent_type == "RAG":
        return RAG(name)
    elif agent_type == "MoA":
        return MoA(name,moa_scale)


def AgentScopeAgentRun(agent_type,query,agent,moa_scale=3):
    if agent_type == "ReAct":
        task=Msg(
            role="user",
            content=query,
            name="user",
        )
        response=agent(task)
        return response.content
    elif agent_type =="RAG":
        retrieved_time.clear()
        RAG_LLM_TIME.clear()
        task=Msg(
            role="user",
            content=query,
            name="user",
        )
        agent.memory.clear()
        start=time.perf_counter()
        response=agent(task)
        end=time.perf_counter()
        total_time=end-start
        usage=agentscope.print_llm_usage()
        return [response.content,total_time,retrieved_time,RAG_LLM_TIME,usage]
    elif agent_type =="MoA" and moa_scale==3:
        greeting=Msg(
            role="user",
            content=query,
            name="user",
        )
        agent[0].memory.clear()
        agent[1].memory.clear()
        agent[2].memory.clear()
        agent[3].memory.clear()
        name0="root"
        name1="agent1"
        name2="agent2"
        name3="agent3"
        name4="aggregation"

        
        with msghub(
            participants=[agent[3]],
        ) as hub:
            # 分别工作
            hub.add(agent[0])
            ans0=agent[0](greeting)
            hub.delete(agent[0])

            hub.add(agent[1])
            ans1=agent[1](greeting)
            hub.delete(agent[1])

            hub.add(agent[2])
            ans2=agent[2](greeting)
            hub.delete(agent[2])
            # 总结
            agent[3]()

            
        communication_sizes=[len(str(greeting).encode("utf-8")),len(str(greeting).encode("utf-8")),len(str(greeting).encode("utf-8")),len(str(ans0).encode("utf-8")),len(str(ans1).encode("utf-8")),len(str(ans2).encode("utf-8"))]
        package_size=[118,118,118,124,122,126]


        logging.info(f"source_agent_name: {name0}, target_agent_name: {name1}, message_size: {communication_sizes[0]}, packaging_size: {package_size[0]}")
        logging.info(f"source_agent_name: {name0}, target_agent_name: {name2}, message_size: {communication_sizes[1]}, packaging_size: {package_size[1]}")
        logging.info(f"source_agent_name: {name0}, target_agent_name: {name3}, message_size: {communication_sizes[2]}, packaging_size: {package_size[2]}")
        logging.info(f"source_agent_name: {name1}, target_agent_name: {name4}, message_size: {communication_sizes[3]}, packaging_size: {package_size[3]}")
        logging.info(f"source_agent_name: {name2}, target_agent_name: {name4}, message_size: {communication_sizes[4]}, packaging_size: {package_size[4]}")
        logging.info(f"source_agent_name: {name3}, target_agent_name: {name4}, message_size: {communication_sizes[5]}, packaging_size: {package_size[5]}")

        return agent[3].memory.get_memory()[-1].content
    elif agent_type =="MoA" and moa_scale!=3:
        greeting=Msg(
            role="user",
            content=query,
            name="user",
        )
        for i in range(moa_scale+1):
            agent[i].memory.clear()
        name_head="root"
        name_tail="aggregation"
        name=[]
        for i in range(1,moa_scale+1):
            name.append("agent"+str(i))


        ans=[]
        with msghub(
            participants=[agent[moa_scale]],
        ) as hub:
            # 分别工作
            for i in range(moa_scale):
                hub.add(agent[i])
                ans.append(agent[i](greeting))
                hub.delete(agent[i])
            # 总结
            agent[moa_scale]()

        communication_sizes=[]
        package_size=[]
        temp=[125,123,127]
        for i in range(moa_scale):
            communication_sizes.append(len(str(greeting).encode("utf-8")))
            package_size.append(118)
        for i in range(moa_scale):
            communication_sizes.append(len(str(ans[i]).encode("utf-8")))
            package_size.append(temp[i%3])
        
        for i in range(moa_scale):
            logging.info(f"source_agent_name: {name_head}, target_agent_name: {name[i]}, message_size: {communication_sizes[i]}, packaging_size: {package_size[i]}")
        for i in range(moa_scale):
            logging.info(f"source_agent_name: {name[i]}, target_agent_name: {name_tail}, message_size: {communication_sizes[i+moa_scale]}, packaging_size: {package_size[i+moa_scale]}")

        return agent[moa_scale].memory.get_memory()[-1].content

