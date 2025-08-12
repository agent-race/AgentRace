import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.agent.workflow.multi_agent_workflow import AgentWorkflow
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent import AgentRunner
from llama_index.core.workflow.errors import *
# from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader, ImageReader, VideoAudioReader, PandasExcelReader, PandasCSVReader
from pathlib import Path
import sys
import os
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))
import weave
sys.path.append(os.getcwd())

from llama_index.core.instrumentation.events.span import SpanDropEvent
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.packs.mixture_of_agents import MixtureOfAgentsPack
from llama_index.packs.mixture_of_agents.base import MixtureOfAgentWorkflow

from llama_index.llms.openai import OpenAI
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher(__name__)


from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMChatEndEvent,
    LLMChatStartEvent
)
from llama_index.core.instrumentation.events.agent import AgentToolCallEvent
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent
from typing import Any, Dict, Optional


from llama_index.core.callbacks.schema import CBEventType, EventPayload


import logging

global agent_index
agent_index = {}

class ModelEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ModelEventHandler"

    def handle(self, event) -> None: # https://docs.llamaindex.ai/en/stable/examples/instrumentation/instrumentation_observability_rundown/
        """Logic for handling event."""
        if isinstance(event, LLMCompletionEndEvent):
            print('LLMCompletionEndEvent time:', event.timestamp)
            print(f"LLM Prompt length: {len(event.prompt)}")
            print(f"LLM Completion: {str(event.response.text)}")
        elif isinstance(event, LLMCompletionStartEvent):
            print('LLMCompletionStartEvent time:', event.timestamp)
        elif isinstance(event, LLMChatStartEvent):
            print('LLMChatStartEvent time:', event.timestamp)
            messages_str = "\n".join([str(x) for x in event.messages])
        elif isinstance(event, LLMChatEndEvent):
            print('LLMChatEndEvent time:', event.timestamp)
            messages_str = "\n".join([str(x) for x in event.messages])
            print(f"LLM Input Messages length: {len(messages_str)}")
            print(f"LLM Response: {str(event.response.message)}")
        elif isinstance(event, EmbeddingEndEvent):
            print(f"Embedding {len(event.chunks)} text chunks")
        elif isinstance(event, AgentToolCallEvent):
            print('ToolCallResult time:', event.tool_name)
        elif isinstance(event, SpanDropEvent):
            print('ToolCallEvent time:', event.err_str)

from llama_index.core.instrumentation import get_dispatcher
# root dispatcher
root_dispatcher = get_dispatcher()
# register event handler
root_dispatcher.add_event_handler(ModelEventHandler())


from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
# processor = SimpleSpanProcessor(ConsoleSpanExporter())
# provider.add_span_processor(processor)
from functools import wraps
def traced_tool(tool_name=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs): 
            span = tracer.start_span(fn.__name__)
            span.set_attribute("input", str(args))
            try:
                result = fn(*args, **kwargs)
                span.set_attribute("output", str(result)[:100])
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.end()
                name = tool_name if tool_name else fn.__name__
                # print(f"{name} finished, time: {(span.end_time - span.start_time)/1e9}")
                logging.info(f"tool_name: {tool_name}, tool_time: {(span.end_time - span.start_time)/1e9}") 
        return wrapper
    return decorator

from pympler import asizeof
def communication_size(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # if kwargs['ev'].llm.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo":
        #     source_agent_name = 'Agent 1'
        # elif kwargs['ev'].llm.model == "Qwen/Qwen2.5-7B-Instruct-Turbo":
        #     source_agent_name = 'Agent 2'
        # elif kwargs['ev'].llm.model == "deepseek-ai/DeepSeek-V3":
        #     source_agent_name = 'Agent 3'
        source_agent_name = agent_index[id(kwargs['ev'].llm)]
        target_agent_name = 'Root Agent'
        # message_size = asizeof.asizeof(kwargs['ev'])
        message_size = len(str(kwargs['ev']).encode("utf-8"))
        # packaging_size = message_size - asizeof.asizeof(kwargs['ev'].messages)
        packaging_size = message_size - len(str(kwargs['ev'].messages).encode('utf-8'))
        logging.info(f"source_agent_name: {target_agent_name}, target_agent_name: {source_agent_name}, message_size: {message_size}, packaging_size: {packaging_size}")
        
        result = await func(*args, **kwargs) 

        message_size = len(str(result).encode('utf-8'))
        packaging_size = message_size - len(str(result.result).encode('utf-8'))
        logging.info(f"source_agent_name: {source_agent_name}, target_agent_name: {target_agent_name}, message_size: {message_size}, packaging_size: {packaging_size}")
        return result
    return wrapper


MixtureOfAgentWorkflow.agenerate_with_references = communication_size(MixtureOfAgentWorkflow.agenerate_with_references)

from llama_index.core.callbacks.token_counting import get_llm_token_counts, TokenCountingEvent
def patched_on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if (
            event_type == CBEventType.LLM
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            self.llm_token_counts.append(
                get_llm_token_counts(
                    token_counter=self._token_counter,
                    payload=payload,
                    event_id=event_id,
                )
            )

            if self._verbose:
                # self._print(
                #     "LLM Prompt Token Usage: "
                #     f"{self.llm_token_counts[-1].prompt_token_count}\n"
                #     "LLM Completion Token Usage: "
                #     f"{self.llm_token_counts[-1].completion_token_count}",
                # )
                prompt_token = self.llm_token_counts[-1].prompt_token_count
                completion_token = self.llm_token_counts[-1].completion_token_count
                logging.info(f'LLM Prompt Token Usage: {prompt_token}')
                logging.info(f'LLM Completion Token Usage: {completion_token}')
        elif (
            event_type == CBEventType.EMBEDDING
            and event_type not in self.event_ends_to_ignore
            and payload is not None
        ):
            total_chunk_tokens = 0
            for chunk in payload.get(EventPayload.CHUNKS, []):
                self.embedding_token_counts.append(
                    TokenCountingEvent(
                        event_id=event_id,
                        prompt=chunk,
                        prompt_token_count=self._token_counter.get_string_tokens(chunk),
                        completion="",
                        completion_token_count=0,
                    )
                )
                total_chunk_tokens += self.embedding_token_counts[-1].total_token_count

            if self._verbose:
                self._print(f"Embedding Token Usage: {total_chunk_tokens}")

TokenCountingHandler.on_event_end = patched_on_event_end

class EvalGoogleSearchToolSpec(GoogleSearchToolSpec):
    @traced_tool(tool_name='web_browser_tool')
    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        return super().google_search(query)
    
class EvalCodeInterpreterToolSpec(CodeInterpreterToolSpec):
    @traced_tool(tool_name='python_tool')
    def code_interpreter(self, code: str):
        """
        A function to execute python code, and return the stdout and stderr.

        You should import any libraries that you wish to use. You have access to any libraries the user has installed.

        The code passed to this functuon is executed in isolation. It should be complete at the time it is passed to this function.

        You should interpret the output and errors returned from this function, and attempt to fix any problems.
        If you cannot fix the error, show the code to the user and ask for help

        It is not possible to return graphics or other complicated data from this function. If the user cannot see the output, save it to a file and tell the user.
        """
        return super().code_interpreter(code=code)
    
@traced_tool(tool_name='pdf_tool')
def PDFLoader(path: str) -> str:
    """
    Load a PDF document from a path.

    Args:
        path (str): The path to the PDF document.

    Returns:
        str: The loaded PDF document data.
    """
    reader = SimpleDirectoryReader(input_files=[path])
    data = reader.load_data()
    return data


@traced_tool(tool_name='docx_tool')
def DocLoader(path: str) -> str:
    """
    Load a Doc or Docx document from a path.

    Args:
        path (str): The path to the Doc(x) document.

    Returns:
        str: The loaded Doc(x) document data.
    """
    reader = SimpleDirectoryReader(input_files=[path])
    data = reader.load_data()
    return data.text


@traced_tool(tool_name='txt_tool')
def TxtLoader(path: str) -> str:
    """
    Load a txt document from a path.

    Args:
        path (str): The path to the txt document.

    Returns:
        str: The loaded txt document data.
    """
    reader = SimpleDirectoryReader(input_files=[path])
    data = reader.load_data()
    return data[0].text


@traced_tool(tool_name='csv_tool')
def CsvLoader(path: str) -> str:
    """
    Loads a CSV file and returns the document in this CSV file.
    Args:
        path (str): The path to the CSV file.
    """
    try:
        reader = PandasCSVReader()
        documents = reader.load_data(file=Path(path))
        return str(documents)
    except Exception as e:
        return f"Error: {e}"


@traced_tool(tool_name='xlsx_tool')
def XlsxLoader(path: str) -> str:
    """
    Loads an Excel file and returns the document in this Excel file.
    Args:
        path (str): The path to the Excel file.
    """
    try:
        reader = PandasExcelReader(pandas_config = {"engine": 'openpyxl'})
        documents = reader.load_data(file=Path(path))
        return str(documents)
    except Exception as e:
        return f"Error: {e}"


@traced_tool(tool_name='python_tool')
def PythonREPLTool(code: str) -> str:
    """
    Executes Python code and returns the output.
    Args:
        code (str): The Python code to execute.
    """
    try:
        exec(code)
        return "Code executed."
    except Exception as e:
        return f"Error: {e}"


@traced_tool(tool_name='vision_tool')
def ImageLoader(path: str) -> str:
    """
    Loads an image from a path and returns the document in this image.
    Args:
        path (str): The path to the image file.
    """
    try:
        reader = ImageReader(parse_text=True)
        documents = reader.load_data(file=path)
        return documents[0].text
    except Exception as e:
        return f"Error: {e}"

@traced_tool(tool_name='video_tool')
def Mp4Loader(path: str) -> str:
    """
    Loads a .mp4 file from a path and returns the document in this video.
    Args:
        path (str): The path to the .mp4 file.
    """
    try:
        reader = VideoAudioReader()
        documents = reader.load_data(file=Path(path))
        return documents[0].text
    except Exception as e:
        return f"Error: {e}"

@traced_tool(tool_name='video_tool')
def MovLoader(path: str) -> str:
    """
    Loads a .mov file from a path and returns the document in this video.
    Args:
        path (str): The path to the .mov file.
    """
    try:
        reader = VideoAudioReader()
        documents = reader.load_data(file=Path(path))
        return documents[0].text
    except Exception as e:
        return f"Error: {e}"


@traced_tool(tool_name='audio_tool')
def AudioLoader(path: str) -> str:
    """
    Loads an audio from a path and returns the document in this audio.
    Args:
        path (str): The path to the audio file.
    """
    try:
        reader = VideoAudioReader()
        documents = reader.load_data(file=Path(path))
        return documents[0].text
    except Exception as e:
        return f"Error: {e}"

# irrelevant tools
from agents.irrelevant_tools.irrelevant_tools import twoSum
from agents.irrelevant_tools.irrelevant_tools import lengthOfLongestSubstring
from agents.irrelevant_tools.irrelevant_tools import findMedianSortedArrays
from agents.irrelevant_tools.irrelevant_tools import longestPalindrome
from agents.irrelevant_tools.irrelevant_tools import convertZ
from agents.irrelevant_tools.irrelevant_tools import reverseX
from agents.irrelevant_tools.irrelevant_tools import myAtoi
from agents.irrelevant_tools.irrelevant_tools import isPalindrome
from agents.irrelevant_tools.irrelevant_tools import isMatch
from agents.irrelevant_tools.irrelevant_tools import maxArea

from agents.irrelevant_tools.irrelevant_tools import longestCommonPrefix
from agents.irrelevant_tools.irrelevant_tools import threeSum
from agents.irrelevant_tools.irrelevant_tools import isValidBrackets
from agents.irrelevant_tools.irrelevant_tools import generateParenthesis
from agents.irrelevant_tools.irrelevant_tools import groupAnagrams
from agents.irrelevant_tools.irrelevant_tools import lengthOfLastWord
from agents.irrelevant_tools.irrelevant_tools import addBinary
from agents.irrelevant_tools.irrelevant_tools import minDistance
from agents.irrelevant_tools.irrelevant_tools import largestNumber
from agents.irrelevant_tools.irrelevant_tools import reverseString

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


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]

class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]

# 每次运行都初始化db
class RAGWorkflow(Workflow):

    def __init__(self,
        timeout = 10.0,
        disable_validation = False,
        verbose: bool = False,
        service_manager = None,
        num_concurrent_runs = None,
        mmlu_path=None): # if mmul_path is None it will build db everytime
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            service_manager=service_manager,
            num_concurrent_runs=num_concurrent_runs,
        )
        self.llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHER_API_KEY'])
        self.embed_model = TogetherEmbedding(model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=os.environ['TOGETHER_API_KEY'])
        token_count = TokenCountingHandler(verbose=True)  # verbose=True 会打印 token 使用情况
        self.callback = CallbackManager([token_count])
        if mmlu_path is not None:
            documents = SimpleDirectoryReader(os.path.join(mmlu_path, 'dev')).load_data()
            # documents += SimpleDirectoryReader(os.path.join(mmlu_path, 'auxiliary_train')).load_data()
            self.vsindex = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model = self.embed_model
            )
        else:
            self.vsindex = None
    

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if self.vsindex is not None:
            return StopEvent(result=self.vsindex)
        if not dirname:
            return None
        # documents = EvalGoogleSearchToolSpec(key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_ENGINE']).google_search(query=query)
        # documents = SimpleDirectoryReader(os.path.join(dirname, 'dev')).load_data()
        documents = SimpleDirectoryReader(os.path.join(dirname, 'dev')).load_data()
        
        # image_reader = ImageReader(parse_text=True)
        # documents += image_reader.load_data(dirname)

        # for filename in os.listdir(dirname):
        #     if Path(os.path.join(dirname ,filename)).name.endswith(".mp3") or Path(os.path.join(dirname ,filename)).name.endswith(".mp4"):
        #         audio_video_reader = VideoAudioReader()
        #         documents += audio_video_reader.load_data(file=Path(os.path.join(dirname ,filename)))
        #     if Path(os.path.join(dirname ,filename)).name.endswith(".png") or Path(os.path.join(dirname ,filename)).name.endswith(".jpg") or Path(os.path.join(dirname ,filename)).name.endswith(".jpeg"):
        #         documents += image_reader.load_data(os.path.join(dirname ,filename))

        
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model = self.embed_model
        )
        return StopEvent(result=index)

    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=5)
        nodes = await retriever.aretrieve(query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm = self.llm
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = self.llm
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

# 仅一次初始化db
class RAGWorkflow2(Workflow):

    def __init__(self,
        timeout = 30.0,
        disable_validation = False,
        verbose: bool = False,
        service_manager = None,
        num_concurrent_runs = None,
        mmlu_path=None): # if mmul_path is None it will build db everytime
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            service_manager=service_manager,
            num_concurrent_runs=num_concurrent_runs,
        )
        # self.llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHER_API_KEY'])
        self.llm = OpenAI(model='gpt-4o', temperature=0, api_base=os.environ['OPENAI_BASE_URL'],)
        # self.embed_model = TogetherEmbedding(model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=os.environ['TOGETHER_API_KEY'])
        # self.embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')
        self.embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2', device='cpu')
        token_count = TokenCountingHandler(verbose=True)  # verbose=True 会打印 token 使用情况
        self.callback = CallbackManager([token_count])
        if mmlu_path is not None:
            embedding_start_time = time.time()
            documents = SimpleDirectoryReader(os.path.join(mmlu_path, 'dev')).load_data()
            # documents += SimpleDirectoryReader(os.path.join(mmlu_path, 'auxiliary_train')).load_data()
            self.vsindex = VectorStoreIndex.from_documents(
                documents=documents,
                embed_model = self.embed_model
            )
            embedding_end_time = time.time()
            logging.info(f"database_vectorization time: {embedding_end_time-embedding_start_time}")
        else:
            self.vsindex = None

    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        retrieve_start_time = time.time()
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=5)
        # if self.vsindex is None:
        #     nodes = await retriever.aretrieve(query)
        # else:
        #     nodes = retriever.retrieve(query)
        nodes = await retriever.aretrieve(query)
        retrieve_end_time = time.time()
        logging.info(f"retrieve time: {retrieve_end_time-retrieve_start_time}")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm = self.llm
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        rerank_start_time = time.time()
        llm = self.llm
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        rerank_end_time = time.time()
        logging.info(f"rerank time: {rerank_end_time-rerank_start_time}")
        return StopEvent(result=response)


def load_llamaIndexagent(model_name, workflow, api_key=None):
    
    if workflow == 'ReAct':
        # 创建 TokenCountingHandler 实例
        token_count = TokenCountingHandler(verbose=True)  # verbose=True 会打印 token 使用情况
        # 创建 CallbackManager 并添加 token_count
        callback = CallbackManager([token_count])

    if model_name == 'deepseek-chat' or model_name == 'deepseek-reasoner':
        # llm = DeepSeek(model="deepseek-chat", api_key=os.environ['DS_API_KEY'], callback_manager=callback)
        llm = DeepSeek(model="deepseek-chat", api_key=os.environ['DS_API_KEY'], temperature=0)
        # llm = TogetherLLM(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", api_key=os.environ['TOGETHER_API_KEY'])
    else:
        llm = OpenAI(model="gpt-4o", api_key=api_key, temperature=0, api_base=os.environ['OPENAI_BASE_URL'])
    
    if workflow == "ReAct":
        @weave.op()
        def omni_run(self, task: str):
            runner = self.run
            async def run_react(runner, task):
                response = await runner(user_msg=task)
                return response
            import asyncio
            return asyncio.run(run_react(runner, task))
        ReActAgent.omni_run = omni_run
        wf = ReActAgent(
            llm=llm, 
            tools=[EvalGoogleSearchToolSpec(key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_ENGINE']).google_search, 
                   TxtLoader,
                   PDFLoader,
                   DocLoader,
                   CsvLoader,
                   XlsxLoader,
                   ImageLoader,
                   Mp4Loader,
                   MovLoader,
                   AudioLoader,
                   EvalCodeInterpreterToolSpec().code_interpreter,

                    # irrelevant tools
                    # twoSum,
                    # lengthOfLongestSubstring,
                    # findMedianSortedArrays,
                    # longestPalindrome,
                    # convertZ,
                    # reverseX,
                    # myAtoi,
                    # isPalindrome,
                    # isMatch,
                    # maxArea,

                    # longestCommonPrefix,
                    # threeSum,
                    # isValidBrackets,
                    # generateParenthesis,
                    # groupAnagrams,
                    # lengthOfLastWord,
                    # addBinary,
                    # minDistance,
                    # largestNumber,
                    # reverseString,
                ],
        )
        # wf = AgentWorkflow.from_tools_or_functions(
        #     [EvalGoogleSearchToolSpec(key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_ENGINE']).google_search, TxtLoader],
        #     llm=llm,
        #     system_prompt="You are an agent that can perform basic operations using tools."
        # )
        return wf
    
    elif workflow == "RAG":
        @weave.op()
        def omni_run(self, task: str):
            runner = self.run
            async def run_react(runner, task):
                # index = await runner(dirname='./data/MMLU')
                # result = await runner(query=task, index=index)
                result = await runner(query=task, index=self.vsindex)
                return result
            import asyncio
            return asyncio.run(run_react(runner, task))
        RAGWorkflow2.omni_run = omni_run
        # wf = RAGWorkflow(mmlu_path=None)
        wf = RAGWorkflow2(mmlu_path='./data/MMLU')
        return wf
    
    elif workflow == "MoA":
        reference_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3"
        ]
        reference_llms = []
        for multiple in range(1):
            reference_llms = reference_llms + [TogetherLLM(model=model_name, api_key=os.environ['TOGETHER_API_KEY'], temperature=0) for model_name in reference_models]
        # aggregation_llm = TogetherLLM(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key=os.environ['TOGETHER_API_KEY'], temperature=0)
        aggregation_llm = llm
        
        for llm_index, each_llm in enumerate(reference_llms):
            agent_index[id(each_llm)] = 'Agent ' + str(llm_index + 1)

        @weave.op()
        def omni_run(self, task: str):
            runner = self._wf.run
            async def run_moa(runner, task):
                result = await runner(query_str=task)
                return result
            import asyncio
            response = asyncio.run(run_moa(runner=runner, task=task))
            return response
        MixtureOfAgentsPack.omni_run = omni_run

        wf = MixtureOfAgentsPack(llm=aggregation_llm, reference_llms=reference_llms, num_layers=1)
        return wf

if __name__ == "__main__":
    import weave
    import sys
    sys.path.append(os.getcwd())
    from eval.token_count import init_token_count
    # init_token_count('./results/llamaindex-test')
    weave.init('llamaindex-test')
    # import agentops
    # agentops.init(os.environ['AGENTOPS_API_KEY'])
    # agent = load_llamaIndexagent('deepseek-chat', 'ReAct')
    # result = agent.omni_run(task="请你介绍一下./data_test/test.txt 的内容")
    # print(result.response.blocks[0].text)
    
    
    agent = load_llamaIndexagent('gpt-4o', 'ReAct')
    result = agent.omni_run(task="search information about UCAS using google")
    print(result)

    # agent = load_llamaIndexagent('deepseek-chat', 'MoA')
    # result = agent.omni_run(task="What is the capital of France?")
    # print(result)


    # # 创建 TokenCountingHandler 实例
    # token_counter = TokenCountingHandler(verbose=True)  # verbose=True 会打印 token 使用情况
    # # 创建 CallbackManager 并添加 token_counter
    # callback_manager = CallbackManager([token_counter])

    # llm = DeepSeek(model="deepseek-chat", api_key=os.environ['DS_API_KEY'],callback_manager=callback_manager) # https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/

    # """
    # ---- ReAct ----
    # """
    # workflow = AgentWorkflow.from_tools_or_functions( # ReActAgent
    #     # [multiply, add, save_file],
    #     # [EvalGoogleSearchToolSpec(key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_KEY']).google_search],
    #     [PDFLoader, ImageLoader, AudioLoader, Mp4Loader, MovLoader, EvalGoogleSearchToolSpec(key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_ENGINE']).google_search],
    #     llm=llm,
    #     # system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    #     system_prompt="You are an agent that can perform translation using tools.",
    # )

    # """
    # ---- ReAct ----
    # """
    # # workflow = RAGWorkflow()

    # from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
    # wflow_ckptr = WorkflowCheckpointer(workflow=workflow)
    
    # async def main():
    #     """
    #     ---- ReAct & MoA ----
    #     """
    #     response = await wflow_ckptr.run(user_msg="Translate the file './rag_data_test/test.mp3' into Chinese")
    #     # response = await wflow_ckptr.run(user_msg="Google 'UCAS' and tell me the result")
    #     print(response)

    #     """
    #     ---- RAG ----
    #     """
    #     # index = await workflow.run(dirname="rag_data_test")
    #     # result = await workflow.run(query="张三哪年出生的?", index=index)
    #     # async for chunk in result.async_response_gen():
    #     #     print(chunk, end="", flush=True)        
        

    # import asyncio
    # asyncio.run(main())

    # # 获取 token 统计信息
    # print(f"Total LLM Token Count: {token_counter.total_llm_token_count}")
    # print(f"Total Embedding Token Count: {token_counter.total_embedding_token_count}")
    # token_counter.reset_counts()

    # message_size = 0
    # for run_id, ckpts in wflow_ckptr.checkpoints.items():
    #     print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")
    #     for ckpt in ckpts:
    #         message_size += len(str(ckpt.input_event)) + len(str(ckpt.ctx_state))

    # print(f'message_size: ',message_size)
    

