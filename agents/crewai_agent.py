from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
from functools import wraps
import os
import logging
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(
    filename=os.path.join(parent_dir,"results","crewai","log","react_gaia-4o-t20.log"),  
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
    )

def traced_tool(fn, tool_name=None):
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
            if tool_name:
                name = tool_name
                time_span = (span.end_time - span.start_time)/1e9
                print(f"{name} finished, time: {time_span:.2f} seconds")
                logging.info(f"tool_name: {tool_name}, tool_time: {time_span}") 
            else:
                time_span = (span.end_time - span.start_time)/1e9
                print(f"{fn.__name__} finished, time: {time_span:.2f} seconds")
    return wrapper

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool,PDFSearchTool
from crewai.tools import BaseTool
from crewai import LLM
import base64
from typing import Optional,Type 
from pathlib import Path
from openai import OpenAI
import os
from pydantic import BaseModel, field_validator
from crewai_tools import LlamaIndexTool
from llama_index.readers.file import VideoAudioReader,ImageReader
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
# from langchain_openai import ChatOpenAI
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from dotenv import load_dotenv
load_dotenv()
import time 
import weave
# import psutil

# CPU 8-17
# psutil.Process(os.getpid()).cpu_affinity(list(range(8, 18)))


class ImagePromptSchema(BaseModel):
    """Input for Vision Tool."""

    image_path_url: str = "The image path or URL."

    @field_validator("image_path_url")
    def validate_image_path_url(cls, v: str) -> str:
        if v.startswith("http"):
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file does not exist: {v}")

        # Validate supported formats
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp",'.bmp','.tiff'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported image format. Supported formats: {valid_extensions}"
            )

        return v

class VisionTool(BaseTool):
    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an image.The image_path is called image_path_url."
    )
    _client: Optional[OpenAI] = None
    api_key: str = None
    base_url: str = None
    model: str = None 
    image_path_url: Optional[str] = None  
    if image_path_url is None:
        args_schema: Type[BaseModel] = ImagePromptSchema


    def client(self) -> OpenAI:
        """Cached OpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _run(self,**kwargs) -> str:
        try:
            image_path_url = self.image_path_url 
            if image_path_url is None:
                image_path_url = kwargs.get("image_path_url")
                ImagePromptSchema(image_path_url=image_path_url)
            if not image_path_url:
                return "Image Path or URL is required."


            if image_path_url.startswith("http"):
                image_data = image_path_url
            else:
                try:
                    base64_image = self._encode_image(image_path_url)
                    image_data = f"data:image/jpeg;base64,{base64_image}"
                except Exception as e:
                    return f"Error processing image: {str(e)}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        

import pandas as pd
from docx import Document
class ServiceExecStatus:
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"

class ServiceResponse:
    def __init__(self, status, content):
        self.status = status
        self.content = content
def xlsx_load(path:str)->ServiceResponse:
    try:
        df = pd.read_excel(path)
        xlsx_str = df.to_csv(index=False)
        return xlsx_str
    except Exception as e:
        return f"Error: {e}"

def csv_load(path:str)->ServiceResponse:
    try:
        df = pd.read_csv(path)
        csv_str = df.to_csv(index=False)
        return csv_str
    except Exception as e:
        return f"Error: {e}"

def txt_load(path:str)->ServiceResponse:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt_str = f.read()
        return txt_str
    except Exception as e:
        return f"Error: {e}"

def docs_load(path:str)->ServiceResponse:
    try:
        doc = Document(path)
        docx_str = "\n".join([para.text for para in doc.paragraphs])
        return docx_str
    except Exception as e:
        return f"Error: {e}"


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

def VideoLoader(path: str) -> str:
    """
    Loads an audio from a path and returns the document in this video.
    Args:
        path (str): The path to the audio file.
    """
    try:
        reader = VideoAudioReader()
        documents = reader.load_data(file=Path(path))
        return documents[0].text
    except Exception as e:
        return f"Error: {e}"    
     

from PyPDF2 import PdfReader     
def PDFLoader(path: str) -> str:
    """
    Load a PDF document from a path.

    Args:
        path (str): The path to the PDF document.

    Returns:
        str: The loaded PDF document data.
    """
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text



        
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

from llama_index.core.tools import FunctionTool
Audio_tool=FunctionTool.from_defaults(AudioLoader, name="AudioLoader", description="useful for when you need to load a audio document from a path")
Video_tool=FunctionTool.from_defaults(VideoLoader,name="VideoLoader",description="useful for when you need to load a video document from a path")
Xlsx_tool=FunctionTool.from_defaults(xlsx_load,name="XlsxLoader",description="useful for when you need to load a xlsx document from a path")
Pdf_tool=FunctionTool.from_defaults(PDFLoader,name="PDFLoader",description="useful for when you need to load a pdf document from a path")
Csv_tool=FunctionTool.from_defaults(csv_load,name="CSVLoader",description="useful for when you need to load a csv document from a path")
Docx_tool=FunctionTool.from_defaults(docs_load,name="DocxLoader",description="useful for when you need to load a docx document from a path")
Txt_tool=FunctionTool.from_defaults(txt_load,name="TxtLoader",description="useful for when you need to load a txt document from a path")
Code_execute=FunctionTool.from_defaults(PythonREPLTool,name="Code_excute",description="useful for when you need to execute a python code")
Image_tool=FunctionTool.from_defaults(ImageLoader,name="vision_tool",description="useful for when you need to load a image document from a path")

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


twoSum = FunctionTool.from_defaults(twoSum, name="twoSum", description=twoSum.__doc__)
lengthOfLongestSubstring = FunctionTool.from_defaults(lengthOfLongestSubstring, name="lengthOfLongestSubstring", description=lengthOfLongestSubstring.__doc__)
findMedianSortedArrays = FunctionTool.from_defaults(findMedianSortedArrays, name="findMedianSortedArrays", description=findMedianSortedArrays.__doc__)
longestPalindrome = FunctionTool.from_defaults(longestPalindrome, name="longestPalindrome", description=longestPalindrome.__doc__)
convertZ = FunctionTool.from_defaults(convertZ, name="convertZ", description=convertZ.__doc__)
reverseX = FunctionTool.from_defaults(reverseX, name="reverseX", description=reverseX.__doc__)
myAtoi = FunctionTool.from_defaults(myAtoi, name="myAtoi", description=myAtoi.__doc__)
isPalindrome = FunctionTool.from_defaults(isPalindrome, name="isPalindrome", description=isPalindrome.__doc__)
isMatch = FunctionTool.from_defaults(isMatch, name="isMatch", description=isMatch.__doc__)
maxArea = FunctionTool.from_defaults(maxArea, name="maxArea", description=maxArea.__doc__)

longestCommonPrefix = FunctionTool.from_defaults(longestCommonPrefix, name="longestCommonPrefix", description=longestCommonPrefix.__doc__)
threeSum = FunctionTool.from_defaults(threeSum, name="threeSum", description=threeSum.__doc__)
isValidBrackets = FunctionTool.from_defaults(isValidBrackets, name="isValidBrackets", description=isValidBrackets.__doc__)
generateParenthesis = FunctionTool.from_defaults(generateParenthesis, name="generateParenthesis", description=generateParenthesis.__doc__)
groupAnagrams = FunctionTool.from_defaults(groupAnagrams, name="groupAnagrams", description=groupAnagrams.__doc__)
lengthOfLastWord = FunctionTool.from_defaults(lengthOfLastWord, name="lengthOfLastWord", description=lengthOfLastWord.__doc__)
addBinary = FunctionTool.from_defaults(addBinary, name="addBinary", description=addBinary.__doc__)
minDistance = FunctionTool.from_defaults(minDistance, name="minDistance", description=minDistance.__doc__)
largestNumber = FunctionTool.from_defaults(largestNumber, name="largestNumber", description=largestNumber.__doc__)
reverseString = FunctionTool.from_defaults(reverseString, name="reverseString", description=reverseString.__doc__)
 
class Workflow(Crew):
        def __init__(self,
        agents = [],
        tasks = [],
        verbose=True,
        process = Process.sequential,
        manager_agent=None,
        ):
            super().__init__(
                agents=agents,
                tasks=tasks,
                verbose=verbose, 
                process = process,
                manager_agent=manager_agent,
                ) 
        # @weave.op()
        def omni_run(self, task,excpected_output='an answer'):
            task1 = Task(
                description=task,
                agent=self.agents[0],
                expected_output=excpected_output,
            )
            self.tasks.append(task1)
            result = self.kickoff()
            self.tasks.clear()  
            return result
        # @weave.op()
        def omni_run_RAG(self, task,excpected_output='an answer',vector_search_result=None):
            task1 = Task(
                description="THE quesition is:\n"+task+"\nsome similar questions are:\n"+vector_search_result[0]+"\n"+vector_search_result[1]+"\n"+vector_search_result[2]+"\n"+vector_search_result[3]+"\n"+vector_search_result[4]+"\ngive a final answer.just need a answer,don't need to explain.DONT need to add any other words.",
                agent=self.agents[0],
                expected_output=excpected_output,
            )
            self.tasks.append(task1)
            result = self.kickoff()
            self.tasks.clear()  
            return result
        # @weave.op()
        def omni_run_MOA(self, task,excpected_output='an answer'):
            task1 = Task(
                description=task,
                expected_output=excpected_output,
                agent=self.agents[-1],
            )
            self.tasks.append(task1)
            result = self.kickoff()
            self.tasks.clear()  
            return result


from crewai.utilities.events.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
def calculate_byte_size(string: str, encoding: str = "utf-8") -> int:
    return len(string.encode(encoding))

@crewai_event_bus.on(AgentExecutionStartedEvent)
def handle_execution_started(source,event):
    agents=[]
    for i in range (1,16):
        agents.append(f"agent{i}")
    if event.agent.role in agents:
        message_size = calculate_byte_size(event.task_prompt)
        logging.info(f"source_agent_name: root, target_agent_name: {event.agent.role}, message_size: {message_size}, packaging_size: 0 ")

@crewai_event_bus.on(AgentExecutionCompletedEvent)
def handle_execution_completed(source,event):
    agents=[]
    for i in range (1,16):
        agents.append(f"agent{i}")
    if event.agent.role in agents:
        message_size = calculate_byte_size(event.output)
        logging.info(f"source_agent_name: {event.agent.role}, target_agent_name: aggregation, message_size: {message_size}, packaging_size: 0")


def CrewAIAgent(agent_type,moa_num=3):
    # Tools
    llm0 = LLM(
    model="openai/gpt-4o",  
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.environ['OPENAI_BASE_URL'],
    temperature=0,
    top_k=1
    )
    vision_tool = VisionTool( 
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",  # 确认模型名称正确
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    
    # the tools that the agent have
    web_browser_tool = SerperDevTool()# search tool
    audio_tool = LlamaIndexTool.from_tool(Audio_tool)#audio tool using Whisper to transcribe audio to text
    video_tool = LlamaIndexTool.from_tool(Video_tool)#video tool using Whisper to transcribe video to text
    xlsx_tool = LlamaIndexTool.from_tool(Xlsx_tool)#xlsx tool using pandas to read xlsx file
    pdf_tool = LlamaIndexTool.from_tool(Pdf_tool)#pdf tool using pdfminer to read pdf file
    csv_tool= LlamaIndexTool.from_tool(Csv_tool)#csv tool using pandas to read csv file
    docx_tool = LlamaIndexTool.from_tool(Docx_tool)#docx tool using python-docx to read docx file
    txt_tool = LlamaIndexTool.from_tool(Txt_tool)#txt tool using python-docx to read docx file
    code_execute_tool = LlamaIndexTool.from_tool(Code_execute)#python code execute tool using python exec
    image_tool=LlamaIndexTool.from_tool(Image_tool)

    # irrelevant tools
    twoSumTool = LlamaIndexTool.from_tool(twoSum)
    lengthOfLongestSubstringTool = LlamaIndexTool.from_tool(lengthOfLongestSubstring)
    findMedianSortedArraysTool = LlamaIndexTool.from_tool(findMedianSortedArrays)
    longestPalindromeTool = LlamaIndexTool.from_tool(longestPalindrome)
    convertZTool = LlamaIndexTool.from_tool(convertZ)
    reverseXTool = LlamaIndexTool.from_tool(reverseX)
    myAtoiTool = LlamaIndexTool.from_tool(myAtoi)
    isPalindromeTool = LlamaIndexTool.from_tool(isPalindrome)
    isMatchTool = LlamaIndexTool.from_tool(isMatch)
    maxAreaTool = LlamaIndexTool.from_tool(maxArea)
    longestCommonPrefixTool = LlamaIndexTool.from_tool(longestCommonPrefix)
    threeSumTool = LlamaIndexTool.from_tool(threeSum)
    isValidBracketsTool = LlamaIndexTool.from_tool(isValidBrackets)
    generateParenthesisTool = LlamaIndexTool.from_tool(generateParenthesis)
    groupAnagramsTool = LlamaIndexTool.from_tool(groupAnagrams)
    lengthOfLastWordTool = LlamaIndexTool.from_tool(lengthOfLastWord)
    addBinaryTool = LlamaIndexTool.from_tool(addBinary)
    minDistanceTool = LlamaIndexTool.from_tool(minDistance)
    largestNumberTool = LlamaIndexTool.from_tool(largestNumber)
    reverseStringTool = LlamaIndexTool.from_tool(reverseString)


    
    # @traced_tool
    # def audio_run(*args, **kwargs):
    #     return audio_tool._run(*args, **kwargs)
    # audio_tool._run = audio_run


    # add time seeing
    pdf_tool._run = traced_tool(pdf_tool._run,tool_name="pdf_tool")
    web_browser_tool._run = traced_tool(web_browser_tool._run,tool_name="web_browser_tool")
    txt_tool._run = traced_tool(txt_tool._run,tool_name="txt_tool")
    docx_tool._run = traced_tool(docx_tool._run,tool_name="docx_tool")
    csv_tool._run = traced_tool(csv_tool._run,tool_name="csv_tool")
    xlsx_tool._run = traced_tool(xlsx_tool._run,tool_name="xlsx_tool")
    audio_tool._run = traced_tool(audio_tool._run,tool_name="audio_tool")
    video_tool._run = traced_tool(video_tool._run,tool_name="video_tool")
    vision_tool._run = traced_tool(vision_tool._run,tool_name="vision_tool")
    image_tool._run = traced_tool(image_tool._run,tool_name="vision_tool")
    code_execute_tool._run = traced_tool(code_execute_tool._run,tool_name="python_tool")

    # irrelevant tools
    twoSumTool._run = traced_tool(twoSumTool._run, tool_name="twoSum")
    lengthOfLongestSubstringTool._run = traced_tool(lengthOfLongestSubstringTool._run, tool_name="lengthOfLongestSubstring")
    findMedianSortedArraysTool._run = traced_tool(findMedianSortedArraysTool._run, tool_name="findMedianSortedArrays")
    longestPalindromeTool._run = traced_tool(longestPalindromeTool._run, tool_name="longestPalindrome")
    convertZTool._run = traced_tool(convertZTool._run, tool_name="convertZ")
    reverseXTool._run = traced_tool(reverseXTool._run, tool_name="reverseX")
    myAtoiTool._run = traced_tool(myAtoiTool._run, tool_name="myAtoi")
    isPalindromeTool._run = traced_tool(isPalindromeTool._run, tool_name="isPalindrome")
    isMatchTool._run = traced_tool(isMatchTool._run, tool_name="isMatch")
    maxAreaTool._run = traced_tool(maxAreaTool._run, tool_name="maxArea")
    longestCommonPrefixTool._run = traced_tool(longestCommonPrefixTool._run, tool_name="longestCommonPrefix")
    threeSumTool._run = traced_tool(threeSumTool._run, tool_name="threeSum")
    isValidBracketsTool._run = traced_tool(isValidBracketsTool._run, tool_name="isValidBrackets")
    generateParenthesisTool._run = traced_tool(generateParenthesisTool._run, tool_name="generateParenthesis")
    groupAnagramsTool._run = traced_tool(groupAnagramsTool._run, tool_name="groupAnagrams")
    lengthOfLastWordTool._run = traced_tool(lengthOfLastWordTool._run, tool_name="lengthOfLastWord")
    addBinaryTool._run = traced_tool(addBinaryTool._run, tool_name="addBinary")
    minDistanceTool._run = traced_tool(minDistanceTool._run, tool_name="minDistance")
    largestNumberTool._run = traced_tool(largestNumberTool._run, tool_name="largestNumber")
    reverseStringTool._run = traced_tool(reverseStringTool._run, tool_name="reverseString")


    tools = [
        web_browser_tool,
        audio_tool,
        video_tool,
        vision_tool,
        pdf_tool,
        csv_tool,
        txt_tool,
        docx_tool,
        xlsx_tool,
        code_execute_tool,

        twoSumTool,
        lengthOfLongestSubstringTool,
        findMedianSortedArraysTool,
        longestPalindromeTool,
        convertZTool,
        reverseXTool,
        myAtoiTool,
        isPalindromeTool,
        isMatchTool,
        maxAreaTool,

        longestCommonPrefixTool,
        threeSumTool,
        isValidBracketsTool,
        generateParenthesisTool,
        groupAnagramsTool,
        lengthOfLastWordTool,
        addBinaryTool,
        minDistanceTool,
        largestNumberTool,
        reverseStringTool,

        # image_tool
    ]

    # agent
    if agent_type == 'ReAct':
        # https://docs.crewai.com/concepts/tools
        agent = Agent(
            role="ReAct",
            goal="You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.\nUse the following format:Question: the input question or request\nThought: you should always think about what to do\nAction: the action to take (if any)\nAction Input: the input to the action (e.g., search query)\nObservation: the result of the action\n... (this process can repeat multiple times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question or request\nBegin!\nQuestion: {input}\n Document parameters: pdf(pdf_tool), image_path_urle(vision_tool), path(audio/video_tool)\n。",
            backstory="You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.",
            tools=tools,
            llm=llm0,
            verbose=False,
            allow_delegation=False,   
        )
        workflow=Workflow(
        agents=[agent],
        tasks=[],
        verbose=True, 
        process = Process.sequential
    )
    elif agent_type == 'RAG':
        # Create an agent with the knowledge store
        agent = Agent(
            role="RAG",
            goal="You are a specialized agent for RAG tasks.You just need to give the answer of the question. Don't need any othter word.Such as the answer is a number 5 ,you need output '5'.Or the answer is A,you need to output 'A'.",
            backstory="""You are a RAG agent.""",
            tools=[],
            llm=llm0,
            verbose=True,
            allow_delegation=False,          
        )
        workflow=Workflow(
        agents=[agent],
        tasks=[],
        verbose=True, 
        process = Process.sequential
        )



    elif agent_type == 'MoA':
        llm1=LLM(
            model="openai/THUDM/GLM-Z1-Rumination-32B-0414",  
            api_key=os.getenv("SILICON_FLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
        )
        llm2=LLM(
            model="openai/Pro/Qwen/Qwen2.5-7B-Instruct",  
            api_key=os.getenv("SILICON_FLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
        )
        llm3=LLM(
            model="openai/deepseek-ai/DeepSeek-V3",  
            api_key=os.getenv("SILICON_FLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0,
        )
        agent1=Agent(
            role="agent1",
            goal="You are one of the  agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            verbose=True,
            allow_delegation=False,
            llm=llm2
        )
        agent2=Agent(
            role="agent2",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent3=Agent(
            role="agent3",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm1,
            allow_delegation=False,
            verbose=True
        )
        agent4=Agent(
            role="agent4",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm1,
            allow_delegation=False,
            verbose=True
        )
        agent5=Agent(
            role="agent5",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm2,
            allow_delegation=False,
            verbose=True
        )
        agent6=Agent(
            role="agent6",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent7=Agent(
            role="agent7",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm1,
            allow_delegation=False,
            verbose=True
        )
        agent8=Agent(
            role="agent8",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm2,
            allow_delegation=False,
            verbose=True
        )
        agent9=Agent(
            role="agent9",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent10=Agent(
            role="agent10",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm1,
            allow_delegation=False,
            verbose=True
        )
        agent11=Agent(
            role="agent11",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm2,
            allow_delegation=False,
            verbose=True
        )
        agent12=Agent(
            role="agent12",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent13=Agent(
            role="agent13",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent14=Agent(
            role="agent14",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        agent15=Agent(
            role="agent15",
            goal="You are one of the agents, you have to make your answers as perfect as possible, there will be a management agent to choose the most perfect answer among the three agents as output, you have to do your best to be selected",
            backstory="You need to be the best",
            tools=[],
            llm=llm3,
            allow_delegation=False,
            verbose=True
        )
        if moa_num==3:
            agent = Agent(
                role="MoA",
                goal="You are an agent manager, and You need to assign the questions you receive to each of your all 3 agents, and summarize their answers to get a more complete answer\nYou must give question to all the all agents, and you must summarize their answers to get a more complete answer.\nYou need to be the best",
                backstory="Get the greatest answer by using your all 3 agents",
                tools=[],
                llm=llm0,
                allow_delegation=True,
                verbose=True
            )
            workflow=Workflow(
            agents=[agent1, agent2, agent3,agent],
            # manager_agent=agent,
            tasks=[],
            verbose=True, 
            process = Process.sequential
            )
        elif moa_num==6:
            agent = Agent(
                role="MoA",
                goal="You are an agent manager, and You need to assign the questions you receive to each of your all 6 agents, and summarize their answers to get a more complete answer\nYou must give question to all the all agents, and you must summarize their answers to get a more complete answer.\nYou need to be the best",
                backstory="Get the greatest answer by using your all 6 agents",
                tools=[],
                llm=llm0,
                allow_delegation=True,
                verbose=True
            )
            workflow=Workflow(
            agents=[agent1, agent2, agent3,agent4,agent5,agent6,agent],
            # manager_agent=agent,
            tasks=[],
            verbose=True, 
            process = Process.sequential
            )
        elif moa_num==9:
            agent = Agent(
                role="MoA",
                goal="You are an agent manager, and You need to assign the questions you receive to each of your all 9 agents, and summarize their answers to get a more complete answer\nYou must give question to all the all agents, and you must summarize their answers to get a more complete answer.\nYou need to be the best",
                backstory="Get the greatest answer by using your all 9 agents",
                tools=[],
                llm=llm0,
                allow_delegation=True,
                verbose=True
            )
            workflow=Workflow(
            agents=[agent1, agent2, agent3,agent4,agent5,agent6,agent7,agent8,agent9,agent],
            # manager_agent=agent,
            tasks=[],
            verbose=True, 
            process = Process.sequential
            )
        elif moa_num==12:
            agent = Agent(
                role="MoA",
                goal="You are an agent manager, and You need to assign the questions you receive to each of your all 12 agents, and summarize their answers to get a more complete answer\nYou must give question to all the all agents, and you must summarize their answers to get a more complete answer.\nYou need to be the best",
                backstory="Get the greatest answer by using your all 12 agents",
                tools=[],
                llm=llm0,
                allow_delegation=True,
                verbose=True
            )
            workflow=Workflow(
            agents=[agent1, agent2, agent3,agent4,agent5,agent6,agent7,agent8,agent9,agent10,agent11,agent12,agent],
            # manager_agent=agent,
            tasks=[],
            verbose=True, 
            process = Process.sequential
            )
        elif moa_num==15:
            agent = Agent(
                role="MoA",
                goal="You are an agent manager, and You need to assign the questions you receive to each of your all 9 agents, and summarize their answers to get a more complete answer\nYou must give question to all the all agents, and you must summarize their answers to get a more complete answer.\nYou need to be the best",
                backstory="Get the greatest answer by using your all 9 agents",
                tools=[],
                llm=llm0,
                allow_delegation=True,
                verbose=True
            )
            workflow=Workflow(
            agents=[agent1, agent2, agent3,agent4,agent5,agent6,agent7,agent8,agent9,agent10,agent11,agent12,agent13,agent14,agent15,agent],
            # manager_agent=agent,
            tasks=[],
            verbose=True, 
            process = Process.sequential
            )
    else :
        print("Invalid agent type. Please choose from 'ReAct', 'RAG', or 'MoA'.")    
    return workflow

if __name__ == "__main__":
    workflow = CrewAIAgent('MoA', moa_num=3)
    workflow.omni_run_MOA(task='instruction: How did US states get their names?')