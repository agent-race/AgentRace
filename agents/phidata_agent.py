from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.deepseek import DeepSeekChat
from phi.model.together import Together

from phi.tools.python import PythonTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.website import WebsiteTools
from phi.tools.csv_tools import CsvTools
from phi.tools.file import FileTools
from phi.tools import Toolkit, Function
import whisper
from typing import cast
from PyPDF2 import PdfReader
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from PIL import Image
import whisper
from typing import cast
from pydub import AudioSegment
from pathlib import Path
import pandas as pd
from docx import Document
from textwrap import dedent

from pathlib import Path
from pydantic import ValidationError
from phi.run.response import RunResponse
from uuid import uuid4
from collections import deque
from typing import cast
from phi.run.response import RunEvent, RunResponse
from phi.model.base import Model
from phi.model.message import Message
from phi.model.response import ModelResponse, ModelResponseEvent
from phi.memory.agent import AgentRun

from pydantic import BaseModel
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
# from agno.vectordb.qdrant import Qdrant
# from phi.embedder.openai import OpenAIEmbedder

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import dotenv
dotenv.load_dotenv(os.path.join(parent_dir,".env"))
from typing import List, Iterator
import logging
logging.getLogger().setLevel(logging.INFO)
import time
import json
from pympler import asizeof
import weave
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
# processor = SimpleSpanProcessor(ConsoleSpanExporter())
# provider.add_span_processor(processor)
from functools import wraps
def traced_tool_dec(tool_name=None):
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
                logging.info(f"tool_name: {name}, tool_time: {(span.end_time - span.start_time)/1e9}") 
        return wrapper
    return decorator

def traced_tool(fn, tool_name=None):
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

def trace_retrieve(fn):
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
            logging.info(f"retrieve time: {(span.end_time - span.start_time)/1e9}")
    return wrapper
def communication_size_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs) 
        print(f"Communication Size: {sys.getsizeof(result)}") 
        return result
    return wrapper

class LogCapture:
    def __init__(self):
        self.logs: List[str] = []
    
    def capture(self, logger_name: str = "phi"):
        """开始捕获指定日志记录器的输出"""
        self.logger = logging.getLogger(logger_name)
        self.original_handlers = self.logger.handlers.copy()
        
        # 添加我们的捕获处理器
        self.capture_handler = logging.Handler()
        self.capture_handler.emit = self._handle_log
        self.logger.addHandler(self.capture_handler)
    
    def _handle_log(self, record):
        """处理日志记录"""
        self.logs.append(record.getMessage())
    
    def stop(self):
        """停止捕获并恢复原始处理器"""
        if hasattr(self, 'logger') and hasattr(self, 'original_handlers'):
            self.logger.handlers = self.original_handlers
    
    def get_logs(self) -> List[str]:
        """获取捕获的日志"""
        return self.logs.copy()

def patched_get_transfer_function(self, member_agent: "Agent", index: int) -> Function:
    def _transfer_task_to_agent(
        task_description: str, expected_output: str, additional_information: str
    ) -> Iterator[str]:
        # Update the member agent session_data to include leader_session_id, leader_agent_id and leader_run_id
        if member_agent.session_data is None:
            member_agent.session_data = {}
        member_agent.session_data["leader_session_id"] = self.session_id
        member_agent.session_data["leader_agent_id"] = self.agent_id
        member_agent.session_data["leader_run_id"] = self.run_id

        # -*- Run the agent
        member_agent_messages = f"{task_description}\n\nThe expected output is: {expected_output}"
        try:
            if additional_information is not None and additional_information.strip() != "":
                member_agent_messages += f"\n\nAdditional information: {additional_information}"
        except Exception as e:
            logging.warning(f"Failed to add additional information to the member agent: {e}")

        member_agent_session_id = member_agent.session_id
        member_agent_agent_id = member_agent.agent_id

        # Create a dictionary with member_session_id and member_agent_id
        member_agent_info = {
            "session_id": member_agent_session_id,
            "agent_id": member_agent_agent_id,
        }
        # Update the leader agent session_data to include member_agent_info
        if self.session_data is None:
            self.session_data = {"members": [member_agent_info]}
        else:
            if "members" not in self.session_data:
                self.session_data["members"] = []
            # Check if member_agent_info is already in the list
            if member_agent_info not in self.session_data["members"]:
                self.session_data["members"].append(member_agent_info)

        if self.stream and member_agent.is_streamable:
            member_agent_run_response_stream = member_agent.run(member_agent_messages, stream=True)
            for member_agent_run_response_chunk in member_agent_run_response_stream:
                yield member_agent_run_response_chunk.content  # type: ignore
        else:
            if member_agent.name == 'Chat Agent 1':
                target_agent_name = 'Agent 1'
            elif member_agent.name == 'Chat Agent 2':
                target_agent_name = 'Agent 2'
            elif member_agent.name == 'Chat Agent 3':
                target_agent_name = 'Agent 3'
            target_agent_name = member_agent.name[5:]
            message_size = len((str(member_agent_messages)).encode("utf-8"))
            logging.info(f"source_agent_name: Root Agent, target_agent_name: {target_agent_name}, message_size: {message_size}, packaging_size: 0")
            member_agent_run_response: RunResponse = member_agent.run(member_agent_messages, stream=False)
            message_size = len(str(member_agent_run_response).encode('utf-8'))
            packaging_size = message_size - len(str(member_agent_run_response.messages).encode('utf-8'))
            logging.info(f"source_agent_name: {target_agent_name}, target_agent_name: Root Agent, message_size: {message_size}, packaging_size: {packaging_size}")
            if member_agent_run_response.content is None:
                yield "No response from the member agent."
            elif isinstance(member_agent_run_response.content, str):
                yield member_agent_run_response.content
            elif issubclass(type(member_agent_run_response.content), BaseModel):
                try:
                    yield member_agent_run_response.content.model_dump_json(indent=2)
                except Exception as e:
                    yield str(e)
            else:
                try:
                    yield json.dumps(member_agent_run_response.content, indent=2)
                except Exception as e:
                    yield str(e)
        yield self.team_response_separator

    # Give a name to the member agent
    agent_name = member_agent.name.replace(" ", "_").lower() if member_agent.name else f"agent_{index}"
    if member_agent.name is None:
        member_agent.name = agent_name

    transfer_function = Function.from_callable(_transfer_task_to_agent)
    transfer_function.name = f"transfer_task_to_{agent_name}"
    transfer_function.description = dedent(f"""\
    Use this function to transfer a task to {agent_name}
    You must provide a clear and concise description of the task the agent should achieve AND the expected output.
    Args:
        task_description (str): A clear and concise description of the task the agent should achieve.
        expected_output (str): The expected output from the agent.
        additional_information (Optional[str]): Additional information that will help the agent complete the task.
    Returns:
        str: The result of the delegated task.
    """)

    # If the member agent is set to respond directly, show the result of the function call and stop the model execution
    if member_agent.respond_directly:
        transfer_function.show_result = True
        transfer_function.stop_after_tool_call = True

    return transfer_function

Agent.get_transfer_function = patched_get_transfer_function

class FileLoadTools(Toolkit):
    """
    FileLoadTools is a toolkit that provides tools for loading files from a local directory.
    """
    def __init__(self,):
        super().__init__(name='file_load_tools')
        self.register(self.load_txt_file)
        self.register(self.load_csv_file)
        self.register(self.load_xlsx_file)
        self.register(self.load_doc_file)
        self.register(self.load_audio_file)
        self.register(self.load_video_file)
        self.register(self.load_image_file)
        self.register(self.load_pdf_file)
    
    @traced_tool_dec(tool_name='txt_tool')
    def load_txt_file(self, path: str):
        """
        load .txt file
        Args:
            path (str): The path to the .txt file.
        Returns:
            str: The contents of the .txt file.
        """
        return FileTools().read_file(path)
    
    @traced_tool_dec(tool_name='csv_tool')
    def load_csv_file(self, path: str, row_limit: int = None):
        """
        load .csv file
        Args:
            path (str): The path to the .csv file.
            row_limit (int, optional): The maximum number of rows to read. Defaults to None.
        Returns:
            str: The contents of the .csv file.
        """
        return CsvTools().read_csv_file(path, row_limit)
    
    @traced_tool_dec(tool_name='xlsx_tool')
    def load_xlsx_file(self, path: str, sheet_name: str = None):
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

    @traced_tool_dec(tool_name='docx_tool')
    def load_doc_file(self, path: str):
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
    
    @traced_tool_dec(tool_name='audio_tool')
    def load_audio_file(self, path: str):
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
    
    @traced_tool_dec(tool_name='video_tool')
    def load_video_file(self, path: str):
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
    
    @traced_tool_dec(tool_name='vision_tool')
    def load_image_file(self, path: str):
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
    
    @traced_tool_dec(tool_name='pdf_tool')
    def load_pdf_file(self, path: str):
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

GoogleSearch.google_search = traced_tool(GoogleSearch.google_search, tool_name="web_browser_tool")
PythonTools.run_python_code = traced_tool(PythonTools.run_python_code, tool_name="python_tool")


Agent.search_knowledge_base = trace_retrieve(Agent.search_knowledge_base)

def PhidataAgent(model_name, agent_type, api_key=None):
    if model_name == 'deepseek/deepseek-chat' or model_name == 'deepseek/deepseek-reasoner':
        model = DeepSeekChat(id="deepseek-chat", api_key=os.environ['DS_API_KEY'] if api_key is None else api_key)
    # elif model_name == 'anthropic/claude-3-7-sonnet-20240620':  # https://docs.phidata.com/models/anthropic
    #     model=Claude(id="claude-3-5-sonnet-20240620")
    else:
        model=OpenAIChat(id="gpt-4o", api_key=os.environ['OPENAI_API_KEY'] if api_key is None else api_key, base_url=os.environ['OPENAI_BASE_URL'], temperature=0.0)

    if agent_type == "ReAct":
        @weave.op()
        def omni_run(self, task: str):
            result = self.run(task, stream=False)
            return result.content
        Agent.omni_run = omni_run
        workflow = Agent(
            name="ReAct Agent",
            model=model,
            tools=[GoogleSearch(), 
                   FileLoadTools(),
                   PythonTools(run_code=True)],
            instructions=["You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.\nUse the following format:Question: the input question or request\nThought: you should always think about what to do\nAction: the action to take (if any)\nAction Input: the input to the action (e.g., search query)\nObservation: the result of the action\n... (this process can repeat multiple times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question or request\nBegin!\nQuestion: {input}\n"],
            show_tool_calls=True,
            markdown=True,
        )
        return workflow
    elif agent_type == "RAG":
        @weave.op()
        def omni_run(self, task: str):
            result = self.run(task, stream=False)
            return result.content
        Agent.omni_run = omni_run
        knowledge_base = CSVKnowledgeBase(
            path="./data/MMLU/dev",
            vector_db=ChromaDb(
                collection="csv_data",
                embedder=SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
            ),
            
        )
        workflow = Agent(
            name="RAG Agent",
            model=model,
            knowledge=knowledge_base,
            search_knowledge=True,
            instructions=["You are a RAG-based assistant. You analyze the question, and call the search_knowledge_base tool to retrieve relevant documents from the knowledge base, and then respond accordingly."]
        )

        embedding_start_time = time.time()
        workflow.knowledge.load(recreate=True, upsert=True)
        embedding_end_time = time.time()
        logging.info(f"database_vectorization time: {embedding_end_time - embedding_start_time}")

    elif agent_type == "MoA":
        reference_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3"
        ]
        @weave.op()
        def omni_run(self, task: str):
            result = self.run(task, stream=False)
            return result.content
        Agent.omni_run = omni_run
        agents = []
        for idx in range(3):
            agents.append(Agent(
                    name="Chat Agent " + str(idx + 1),
                    role="Answer the question",
                    model=Together(id=reference_models[idx%3], api_key=os.environ['TOGETHER_API_KEY'], temperature=0),
                    tools=[],
                    show_tool_calls=True,
                    markdown=True,
                )
            )
        workflow = Agent(
            team=agents,
            instructions=["Transfer task to all chat agents (There are 3 agents in your team)", "Aggreagate responses from all chat agents"],
            show_tool_calls=True,
            markdown=True,
            model=model
            # model = OpenAIChat(id="gpt-4o", api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'], temperature=0)
        )
    return workflow


if __name__ == '__main__':
    import weave
    import sys
    sys.path.append(os.getcwd())
    import eval.token_count
    weave.init('phidata-test')
    # import agentops
    # agentops.init(os.environ['AGENTOPS_API_KEY'])
    web_agent = Agent(
        name="Web Agent",
        model=DeepSeekChat(id="deepseek-chat", api_key=os.environ['DS_API_KEY']),
        tools=[GoogleSearch(), PythonTools(), WebsiteTools(), CsvTools()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True,
        debug_mode=False,
        telemetry=True
    )
    # agent = PhidataAgent('deepseek/deepseek-chat', 'ReAct')
    # result = agent.omni_run('Tell me the content of the file at path: requirements.txt')
    # print(result)

    # agent = PhidataAgent('deepseek/deepseek-chat', 'RAG')
    # result = agent.omni_run('Tell me the captial city of the USA.')
    # print(result.content)

    agent = PhidataAgent('gpt-4o', 'MoA')
    result = agent.omni_run('instruction: I want to get better at networking at work')
    print(result)

    # log_capture = LogCapture()
    # log_capture.capture()  # 开始捕获

    # web_agent.print_response("Tell me about the recent works about agent frameworks?", stream=True, )

    # # 获取捕获的日志
    # captured_logs = log_capture.get_logs()
    # for log in captured_logs:
    #     print(log)

    # log_capture.stop() 

    """
    ---- RAG ----
    """
    # knowledge_base = PDFUrlKnowledgeBase(
    #         urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    #         # Use LanceDB as the vector database
    #         vector_db=LanceDb(
    #             table_name="recipes",
    #             uri="tmp/lancedb",
    #             search_type=SearchType.vector,
    #             embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    #         ),
    #     )
        
    #     agent = EvalAgent(
    #         name="ReAct Agent",
    #         model=model,
    #         tools=[GoogleSearch(),WebsiteTools(), PythonTools()],
    #         instructions=["Always include sources"],
    #         show_tool_calls=True,
    #         markdown=True,
    #         knowledge_base=knowledge_base,
    #     )
