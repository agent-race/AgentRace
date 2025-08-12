import random
import requests
import urllib
import asyncio

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
from PyPDF2 import PdfReader
# from llama_index.llms.together import TogetherLLM
# from llama_index.embeddings.together import TogetherEmbedding

import whisper
from typing import cast
from pydub import AudioSegment
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from PIL import Image
import pandas as pd
from docx import Document

from pathlib import Path
from httpx import AsyncClient
custom_http_client = AsyncClient(timeout=300)
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))
import logging
logging.getLogger().setLevel(logging.INFO)
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
# processor = SimpleSpanProcessor(ConsoleSpanExporter())
# provider.add_span_processor(processor)
from functools import wraps
from pympler import asizeof
import weave

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
                logging.info(f"tool_name: {name}, tool_time: {(span.end_time - span.start_time)/1e9}") 
        return wrapper
    return decorator


def traced_tool_fn(fn, tool_name=None):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        span = tracer.start_span(fn.__name__)
        span.set_attribute("input", str(args))
        try:
            result = fn(*args, **kwargs)
            span.set_attribute("output", str(result)[:100])  # 记录截断的输出
            return result
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            if tool_name:
                name = tool_name
                logging.info(f"tool_name: {name}, tool_time: {(span.end_time - span.start_time)/1e9}") 
            else:
                print(f"{fn.__name__} finished, time: {(span.end_time - span.start_time)/1e9}")
    return wrapper

def communication_size_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if func.__name__ == 'chat_with_agent_1':
            agent_name = 'Agent 1'
        elif func.__name__ == 'chat_with_agent_2':
            agent_name = 'Agent 2'
        elif func.__name__ == 'chat_with_agent_3':
            agent_name = 'Agent 3'
        elif func.__name__ == 'chat_with_agent_4':
            agent_name = 'Agent 4'
        elif func.__name__ == 'chat_with_agent_5':
            agent_name = 'Agent 5'
        elif func.__name__ == 'chat_with_agent_6':
            agent_name = 'Agent 6'
        elif func.__name__ == 'chat_with_agent_7':
            agent_name = 'Agent 7'
        elif func.__name__ == 'chat_with_agent_8':
            agent_name = 'Agent 8'
        elif func.__name__ == 'chat_with_agent_9':
            agent_name = 'Agent 9'
        elif func.__name__ == 'chat_with_agent_10':
            agent_name = 'Agent 10'
        elif func.__name__ == 'chat_with_agent_11':
            agent_name = 'Agent 11'
        elif func.__name__ == 'chat_with_agent_12':
            agent_name = 'Agent 12'
        elif func.__name__ == 'chat_with_agent_13':
            agent_name = 'Agent 13'
        elif func.__name__ == 'chat_with_agent_14':
            agent_name = 'Agent 14'
        elif func.__name__ == 'chat_with_agent_15':
            agent_name = 'Agent 15'
        logging.info(f"source_agent_name: Root Agent, target_agent_name: {agent_name}, message_size: {len(str(kwargs['task']).encode('utf-8'))}, packaging_size: 0")
        result = await func(*args, **kwargs) 
        logging.info(f"source_agent_name: {agent_name}, target_agent_name: Root Agent, message_size: {len(str(result).encode('utf-8'))}, packaging_size: 0")
        return result
    return wrapper

@traced_tool(tool_name='web_browser_tool')
def google_search(query, num=None):
    """
    Make a query to the Google search engine to receive a list of results.

    Args:
        query (str): The query to be passed to Google search.
        num (int, optional): The number of search results to return. Defaults to None.

    Returns:
        str: The JSON response from the Google search API.

    Raises:
        ValueError: If the 'num' is not an integer between 1 and 10.
    """
    try:
        QUERY_URL_TMPL = ("https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}")
        url = QUERY_URL_TMPL.format(
            key=os.environ['GOOGLE_KEY'], engine=os.environ['GOOGLE_ENGINE'], query=urllib.parse.quote_plus(str(query))
        )

        if num is not None:
            if not 1 <= num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={num}"

        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Error: {e}"

from langchain_experimental.utilities.python import PythonREPL
from pydantic import Field
def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)

@traced_tool(tool_name='python_tool')
def PythonTool(query:str) -> str:
    """
    Tool for running python code in a REPL.
    A Python shell. Use this to execute python commands. 
    Input should be a valid python command. 
    If you want to see the output of a value, you should print it out 
    with `print(...)`.

    Args:
        query (str): The Python code to execute.
    Returns: 
        result (str): The stdout result of the executed code as a string.
    """
    try:
        python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return python_repl.run(query)
    except Exception as e:
        return f"Error: {e}"


@traced_tool(tool_name='pdf_tool')
def PDFLoader(path: str) -> str:
    """
    Load a PDF document from a path.

    Args:
        path (str): The path to the PDF document.

    Returns:
        str: The loaded PDF document data.
    """
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
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
        image = Image.open(path)
        processor = DonutProcessor.from_pretrained(
                            "naver-clova-ix/donut-base-finetuned-cord-v2"
                        )
        model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-cord-v2"
        )

        device = 'cpu'
        model.to(device)

        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=3,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        # remove first task start token
        text_str = re.sub(r"<.*?>", "", sequence, count=1).strip()

        return text_str
    except Exception as e:
        return f"Error: {e}"

@traced_tool(tool_name='video_tool')
def VideoLoader(path: str) -> str:
    """
    Loads a video from a path and returns the document in this video.
    Args:
        path (str): The path to the video file.
    """
    try:
        video = AudioSegment.from_file(Path(path), format=path[-3:])
        audio = video.split_to_mono()[0]
        file_str = path[:-4] + ".mp3"
        audio.export(file_str, format="mp3")
        model = whisper.load_model(name="base")
        model = cast(whisper.Whisper, model)
        result = model.transcribe(path)
        return result["text"]
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
        model = whisper.load_model(name="base")
        model = cast(whisper.Whisper, model)
        result = model.transcribe(path)
        return result["text"]
    except Exception as e:
        return f"Error: {e}"

@traced_tool(tool_name='csv_tool')
def CsvLoad(path:str):
    try:
        df = pd.read_csv(path)
        csv_str = df.to_string(index=False)
        return csv_str
    except Exception as e:
        return f"error: {str(e)}"

@traced_tool(tool_name='xlsx_tool')
def XlsxLoader(path: str, sheet_name: str = None):
    """
    load .xlsx file
    Args:
        path (str): The path to the .xlsx file.
        sheet_name (str, optional): The name of the sheet to read. Defaults to None.
    Returns:
        str: The contents of the .xlsx file.
    """
    try:
        excel_file = pd.read_excel(path, sheet_name=None)
        result = ""
        for sheet_name, df in excel_file.items():
            result += f"Sheet: {sheet_name}\n"
            result += df.to_string(index=False) + "\n\n"
        return result.strip()
    except Exception as e:
        return f"error: {str(e)}"

@traced_tool(tool_name='docx_tool')
def DocLoader(path: str):
    """
    load .doc and .docx file
    Args:
        path (str): The path to the .doc file.
    Returns:
        str: The contents of the .doc file.
    """
    try:
        doc = Document(path)
        docx_str = "\n".join([para.text for para in doc.paragraphs])
        return docx_str
    except Exception as e:
        return f"error: {str(e)}"

@traced_tool(tool_name='txt_tool')
def TxtLoader(path:str):
    """
    load .txt file
    Args:
        path (str): The path to the .txt file.
    Returns:
        str: The contents of the .txt file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt_str = f.read()
        return txt_str
    except Exception as e:
        return f"error: {str(e)}"
    
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

twoSumTool = traced_tool_fn(twoSum, tool_name='twoSum')
lengthOfLongestSubstringTool = traced_tool_fn(lengthOfLongestSubstring, tool_name='lengthOfLongestSubstring')
findMedianSortedArraysTool = traced_tool_fn(findMedianSortedArrays, tool_name='findMedianSortedArrays')
longestPalindromeTool = traced_tool_fn(longestPalindrome, tool_name='longestPalindrome')
convertZTool = traced_tool_fn(convertZ, tool_name='convertZ')
reverseXTool = traced_tool_fn(reverseX, tool_name='reverseX')
myAtoiTool = traced_tool_fn(myAtoi, tool_name='myAtoi')
isPalindromeTool = traced_tool_fn(isPalindrome, tool_name='isPalindrome')
isMatchTool = traced_tool_fn(isMatch, tool_name='isMatch')
maxAreaTool = traced_tool_fn(maxArea, tool_name='maxArea')

longestCommonPrefixTool = traced_tool_fn(longestCommonPrefix, tool_name='longestCommonPrefix')
threeSumTool = traced_tool_fn(threeSum, tool_name='threeSum')
isValidBracketsTool = traced_tool_fn(isValidBrackets, tool_name='isValidBrackets')
generateParenthesisTool = traced_tool_fn(generateParenthesis, tool_name='generateParenthesis')
groupAnagramsTool = traced_tool_fn(groupAnagrams, tool_name='groupAnagrams')
lengthOfLastWordTool = traced_tool_fn(lengthOfLastWord, tool_name='lengthOfLastWord')
addBinaryTool = traced_tool_fn(addBinary, tool_name='addBinary')
minDistanceTool = traced_tool_fn(minDistance, tool_name='minDistance')
largestNumberTool = traced_tool_fn(largestNumber, tool_name='largestNumber')
reverseStringTool = traced_tool_fn(reverseString, tool_name='reverseString')

class MoAWorkflow():

    def __init__(self):
        reference_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3"
            # "microsoft/WizardLM-2-8x22B"
        ]
        aggregation_model = "gpt-4o"

        reference_llms = [OpenAIModel(
                model_name,  # model library available at https://www.together.ai/models
                provider=OpenAIProvider(
                    base_url=os.environ['TOGETHER_BASE_URL'],
                    api_key=os.environ['TOGETHER_API_KEY'],
                ),
            ) for model_name in reference_models]
        aggregation_llm = OpenAIModel(
            aggregation_model,
            provider=OpenAIProvider(
                base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'],
                http_client=custom_http_client
            ),
        )
        
        self.agents = []
        for idx in range(15):
            chat_agent = Agent(
                    model=reference_llms[idx%3],
                    deps_type=str,
                    system_prompt='An AI agent is to perform the following task:',
                    tools=[]
                )
            self.agents.append(chat_agent)

        @communication_size_async
        async def chat_with_agent_1(task: str) -> str:
            r = await self.agents[0].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_2(task: str) -> str:
            r = await self.agents[1].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_3(task: str) -> str:
            r = await self.agents[2].run(task, model_settings={'temperature': 0.0})
            return r.output

        @communication_size_async
        async def chat_with_agent_4(task: str) -> str:
            r = await self.agents[3].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_5(task: str) -> str:
            r = await self.agents[4].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_6(task: str) -> str:
            r = await self.agents[5].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_7(task: str) -> str:
            r = await self.agents[6].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_8(task: str) -> str:
            r = await self.agents[7].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_9(task: str) -> str:
            r = await self.agents[8].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_10(task: str) -> str:
            r = await self.agents[9].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_11(task: str) -> str:
            r = await self.agents[10].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_12(task: str) -> str:
            r = await self.agents[11].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_13(task: str) -> str:
            r = await self.agents[12].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_14(task: str) -> str:
            r = await self.agents[13].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        @communication_size_async
        async def chat_with_agent_15(task: str) -> str:
            r = await self.agents[14].run(task, model_settings={'temperature': 0.0})
            return r.output
        
        self.root_agent = Agent(
                model=aggregation_llm,
                deps_type=str,
                system_prompt='Your task is to aggregate all agents results to solve complex tasks.\nYou analyze the input, input the task to all tools that can run a single agent, and synthesize the results from all agents into a final response.',
                tools=[Tool(chat_with_agent_1), 
                       Tool(chat_with_agent_2), 
                       Tool(chat_with_agent_3),
                    #    Tool(chat_with_agent_4), 
                    #    Tool(chat_with_agent_5), 
                    #    Tool(chat_with_agent_6),
                    #    Tool(chat_with_agent_7), 
                    #    Tool(chat_with_agent_8), 
                    #    Tool(chat_with_agent_9),
                    #    Tool(chat_with_agent_10), 
                    #    Tool(chat_with_agent_11), 
                    #    Tool(chat_with_agent_12),
                    #    Tool(chat_with_agent_13), 
                    #    Tool(chat_with_agent_14), 
                    #    Tool(chat_with_agent_15),
                       ]
            )
        
    @weave.op()
    def omni_run(self, task: str) -> str:
        try:
            result = self.root_agent.run_sync(task, model_settings={'temperature': 0.0})
        except UnexpectedModelBehavior as e:
            return 'Pydantic_ai.exceptions.UnexpectedModelBehavior: Received empty model response.'
        return result.output
        
        
        # agents = self.agents
        # root_agent = self.root_agent
        # async def run_sync(task: str):
        #         results = await asyncio.gather(*[agent.run(task) for agent in agents])
        #         final_output = await root_agent.run(f"please aggregate the following results: {[result for result in results]}, based on the task: {task}")
        #         all_messages = [result.all_messages() for result in results]
        #         all_messages.append(final_output.all_messages())
        #         return final_output, all_messages
        # import asyncio
        # return asyncio.run(run_sync(task))
    

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from faiss.swigfaiss import IndexFlatL2
import faiss
import pickle


@dataclass
class Deps:
    embed_model: SentenceTransformer
    index: IndexFlatL2

async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    retrieve_start_time = time.time()
    query_embedding = context.deps.embed_model.encode([search_query])
    index = context.deps.index
    with open("./db/index.pkl", "rb") as f:
        docs = pickle.load(f)
    D, I = index.search(query_embedding, 5)
    retrieve_end_time = time.time()
    logging.info(f"retrieve time: {retrieve_end_time - retrieve_start_time}")
    return '\n\n'.join(
        f'# {docs[row]}\n'
        for row in I[0]
    )


class RAGWorkflow():
    def __init__(self):
        self.llm = OpenAIModel(
            "gpt-4o",
            provider=OpenAIProvider(
                base_url=os.environ['OPENAI_BASE_URL'],
                api_key=os.environ['OPENAI_API_KEY'],
                http_client=custom_http_client
            ),
        )
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag_agent = Agent(
                model=self.llm,
                # deps_type=VectorStoreIndex,
                tools=[Tool(retrieve, takes_ctx=True)],  
                system_prompt=(
                    "You're a RAG agent. please search information from the given task to build a knowledge base and then retrieve relevant information from the knowledge base."
                ),
            )
        if not os.path.exists("./agents/db/index.faiss"):
            embedding_start_time = time.time()
            from agents.support import create_vector_db
            create_vector_db()
            embedding_end_time = time.time()
            logging.info(f"database_vectorization time: {embedding_end_time - embedding_start_time}")
        self.index = faiss.read_index("./db/index.faiss")
        self.deps = Deps(embed_model=SentenceTransformer('all-MiniLM-L6-v2'), index=self.index)

    @weave.op()
    def omni_run(self, task: str):
        result =  self.rag_agent.run_sync(task, stream=False, deps=self.deps, model_settings={'temperature': 0.0})
        return result.output


def PydanticAgent(model_name,agent_type, api_key=None,):
    if model_name == 'deepseek-chat' or model_name == 'deepseek-reasoner':
        model = OpenAIModel(
            model_name="deepseek-chat",
            provider=DeepSeekProvider(api_key=os.getenv('DS_API_KEY')),
        )
    else:
        model = OpenAIModel(
            model_name='gpt-4o',
            provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.environ['OPENAI_BASE_URL'], http_client=custom_http_client),
        )

    if agent_type == "ReAct":
        @weave.op()
        def omni_run(self, task: str):
            result = self.run_sync(task, model_settings={'temperature': 0.0})
            return result.output
        Agent.omni_run = omni_run
        workflow = Agent(
            model=model,
            deps_type=str,  
            system_prompt=(
                "You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.\nUse the following format:Question: the input question or request\nThought: you should always think about what to do\nAction: the action to take (if any)\nAction Input: the input to the action (e.g., search query)\nObservation: the result of the action\n... (this process can repeat multiple times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question or request\nBegin!\nQuestion: {input}\n"
            ),
            tools = [Tool(VideoLoader, takes_ctx=False), 
                     Tool(AudioLoader, takes_ctx=False), 
                     Tool(ImageLoader, takes_ctx=False), 
                     Tool(PDFLoader, takes_ctx=False), 
                     Tool(google_search, takes_ctx=False), 
                     Tool(CsvLoad, takes_ctx=False), 
                     Tool(XlsxLoader, takes_ctx=False),
                     Tool(DocLoader, takes_ctx=False),
                     Tool(TxtLoader, takes_ctx=False),
                     Tool(PythonTool, takes_ctx=False),

                    # irrelevant tools
                    # Tool(twoSumTool, takes_ctx=False),
                    # Tool(lengthOfLongestSubstringTool, takes_ctx=False),
                    # Tool(findMedianSortedArraysTool, takes_ctx=False),
                    # Tool(longestPalindromeTool, takes_ctx=False),
                    # Tool(convertZTool, takes_ctx=False),
                    # Tool(reverseXTool, takes_ctx=False),
                    # Tool(myAtoiTool, takes_ctx=False),
                    # Tool(isPalindromeTool, takes_ctx=False),
                    # Tool(isMatchTool, takes_ctx=False),
                    # Tool(maxAreaTool, takes_ctx=False),

                    # Tool(longestCommonPrefixTool, takes_ctx=False),
                    # Tool(threeSumTool, takes_ctx=False),
                    # Tool(isValidBracketsTool, takes_ctx=False),
                    # Tool(generateParenthesisTool, takes_ctx=False),
                    # Tool(groupAnagramsTool, takes_ctx=False),
                    # Tool(lengthOfLastWordTool, takes_ctx=False),
                    # Tool(addBinaryTool, takes_ctx=False),
                    # Tool(minDistanceTool, takes_ctx=False),
                    # Tool(largestNumberTool, takes_ctx=False),
                    # Tool(reverseStringTool, takes_ctx=False),

                    ]
        )

    elif agent_type == 'RAG':
        workflow = RAGWorkflow()
    elif agent_type == 'MoA':

        workflow = MoAWorkflow()

    return workflow






if __name__ == '__main__':
    model = OpenAIModel(
        'deepseek-chat',
        provider=DeepSeekProvider(api_key=os.getenv('DS_API_KEY')),
    )
    import weave
    import sys
    sys.path.append(os.getcwd())
    import eval.token_count
    weave.init('pydantic-test')
    # import agentops
    # agentops.init(os.environ['AGENTOPS_API_KEY'])

    agent = PydanticAgent('gpt-4o', 'ReAct')
    result = agent.omni_run("question: What's the last line of the rhyme under the flavor name on the headstone visible in the background of the photo of the oldest flavor's headstone in the Ben & Jerry's online flavor graveyard as of the end of 2022?")
    print(result)


    """
    ----- ReAct -----
    """
    # agent = Agent(
    #     model=model,
    #     deps_type=str,
    #     tools=[Tool(VideoLoader, takes_ctx=False)],  
    #     system_prompt=(
    #         "You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.\nUse the following format:Question: the input question or request\nThought: you should always think about what to do\nAction: the action to take (if any)\nAction Input: the input to the action (e.g., search query)\nObservation: the result of the action\n... (this process can repeat multiple times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question or request\nBegin!\nQuestion: {input}\n"
    #     ),
    # )

    # dice_result = agent.run_sync("Translate the mp3 file './rag_data_test/test.mp3' into Chinese", deps='Anne')  

    # print('------------------------------------------------------------------------------------------')
    # print(f"Total massage size: {len(str(dice_result.all_messages()))}")

    # from pydantic_ai.messages import ModelRequest, ModelResponse
    # for am in dice_result.all_messages():
        
    #     if isinstance(am, ModelRequest):
    #         # print(f'message: {am.timestamp}')
    #         for mr in am.parts:
    #             try:
    #                 print(f'request: {mr.part_kind}, timestamp: {mr.timestamp}')
    #             except:
    #                 print(f'request: {mr.part_kind}')
    #     elif isinstance(am, ModelResponse):
    #         for mr in am.parts:
    #             print(f'response: {mr.part_kind}, timestamp: {am.timestamp}')
    #     print('***************************************************************************************')
    # print(f"Total token usage: {dice_result.usage().total_tokens}")


    """
    ----- RAG -----
    """

    # workflow = RAGWorkflow()
    # result = workflow.run("What is the capital of France?")
    # print(result)

    """
    ----- MoA -----
    """

    # workflow = MoAWorkflow(
    #     llm=model,
    #     moa_args=[
    #         {
    #             'description': 'ReAct',
    #             'system_prompt': (
    #                 "You're a react agent, you should give some useful information."
    #             ),
    #             'tools': [],
    #         },
    #         {
    #             'description': 'ReAct',
    #             'system_prompt': (
    #                 "You're a react agent, you should give some useful information."
    #             ),
    #             'tools': [Tool(AudioLoader, takes_ctx=False)],  
    #         },
    #         {
    #             'description': 'ReAct',
    #             'system_prompt': (
    #                 "You're a react agent, you should give some useful information."
    #             ),
    #             'tools': [Tool(google_search, takes_ctx=False)],
    #         }
    #     ],
    #     root_agent_idx = 0
    # )

    # async def main():
    #     result = await workflow.run_sync("What is the capital of France?")
    #     print(result)
    # asyncio.run(main())


    