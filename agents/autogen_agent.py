from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, ModelFamily
from autogen_core.tools import FunctionTool
from autogen.coding import LocalCommandLineCodeExecutor
import tempfile
temp_dir = tempfile.TemporaryDirectory()
from autogen import UserProxyAgent,ConversableAgent
import asyncio
import nest_asyncio
from datasets import load_dataset
from PyPDF2 import PdfReader
from dataclasses import dataclass
from typing import List
from together import Together
import pandas as pd
from docx import Document
import whisper
from typing import cast
from pydub import AudioSegment
from pathlib import Path
from typing import cast
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from PIL import Image
from pympler import asizeof
from pathlib import Path
import os
import logging
import time
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

'''
weave.init('autogen')


provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
'''
tracer = trace.get_tracer(__name__)
def traced_tool(fn=None, *, tool_name=None):
    if fn is None:
        return lambda f: traced_tool(f, tool_name=tool_name)
    @wraps(fn)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        with tracer.start_as_current_span(tool_name or fn.__name__) as span:
            span.set_attribute("tool.input", str(args))
            try:
                result = await fn(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time
                logging.info(f"tool_name: {tool_name or fn.__name__}, tool_time: {duration:.4f}")
                span.set_attribute("tool.output", str(result)[:300])
                return result
            except Exception as e:
                span.record_exception(e)
                raise
    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        with tracer.start_as_current_span(tool_name or fn.__name__) as span:
            span.set_attribute("tool.input", str(args))
            try:
                result = fn(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time
                logging.info(f"tool_name: {tool_name or fn.__name__}, tool_time: {duration:.4f}")
                span.set_attribute("tool.output", str(result)[:300])
                return result
            except Exception as e:
                span.record_exception(e)
                raise
    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

def ReAct(model_name,api_key):
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            },
        base_url="https://api3.apifans.com/v1",
        api_key=api_key,
        temperature=0
        )
    executor = LocalCommandLineCodeExecutor(
        timeout=10,
        work_dir=temp_dir.name
    )
    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,
        code_execution_config={"executor": executor},
        human_input_mode="NEVER",
    )
    @traced_tool(tool_name="python_tool")
    def python_executor(code_block:str)->str:
        return code_executor_agent.generate_reply(messages=[{"role": "user", "content": code_block}])
    python_tool = FunctionTool(
        python_executor, description="useful when you need to execute python code"
    )
    @traced_tool(tool_name="web_browser_tool")
    def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:  # type: ignore[type-arg]
        import os
        import time
        import requests
        from bs4 import BeautifulSoup
        from dotenv import load_dotenv
        load_dotenv()
        google_api_key = os.environ["GOOGLE_API_KEY"]
        search_engine_id = os.environ["GOOGLE_CSE_ID"]
        if not search_engine_id or not search_engine_id:
            raise ValueError("API key or Search Engine ID not found in environment variables")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": google_api_key, "cx": search_engine_id, "q": query, "num": num_results}
        response = requests.get(url, params=params)  # type: ignore[arg-type]
        if response.status_code != 200:
            print(response.json())
            raise Exception(f"Error in API request: {response.status_code}")
    
        results = response.json().get("items", [])
    
        def get_page_content(url: str) -> str:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                words = text.split()
                content = ""
                for word in words:
                    if len(content) + len(word) + 1 > max_chars:
                        break
                    content += " " + word
                return content.strip()
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                return ""
    
        enriched_results = []
        for item in results:
            body = get_page_content(item["link"])
            enriched_results.append(
                {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
            )
            time.sleep(1)  # Be respectful to the servers
    
        return enriched_results
    google_search_tool = FunctionTool(
        google_search, description="Search Google for information, returns results with a snippet and body content"
    )

    @traced_tool(tool_name="pdf_tool")
    def pdf_load(file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    load_pdf = FunctionTool(
        pdf_load, description="useful when you need to read pdf documents"
    )
    
    @traced_tool(tool_name="csv_tool")
    def csv_load(path:str)->str:
        try:
            df = pd.read_csv(path)
            csv_str = df.to_string(index=False)
            return csv_str
        except Exception as e:
            return "csv load error"
    load_csv = FunctionTool(
        csv_load, description="useful when you need to read csv documents"
    )
    
    @traced_tool(tool_name="xlsx_tool")
    def xlsx_load(path:str)->str:
        try:
            excel_file = pd.read_excel(path, sheet_name=None)
            result = ""
            for sheet_name, df in excel_file.items():
                result += f"Sheet: {sheet_name}\n"
                result += df.to_string(index=False) + "\n\n"
            return result.strip()
        except Exception as e:
            return "xlsx load error"
    load_xlsx = FunctionTool(
        xlsx_load, description="useful when you need to read xlsx documents"
    )
    
    @traced_tool(tool_name="txt_tool")
    def txt_load(path:str)->str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                txt_str = f.read()
            return txt_str
        except Exception as e:
            return "txt load error"
    load_txt = FunctionTool(
        txt_load, description="useful when you need to read txt documents"
    )
    
    @traced_tool(tool_name="docx_tool")
    def docs_load(path:str)->str:
        try:
            doc = Document(path)
            docx_str = "\n".join([para.text for para in doc.paragraphs])
            return docx_str
        except Exception as e:
            return "docs load error"
    load_docs = FunctionTool(
        docs_load, description="useful when you need to read docs documents"
    )
    
    @traced_tool(tool_name="vision_tool")
    def image_load(path:str):
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
    load_image = FunctionTool(
        image_load, description="useful when you need to read jpg or png documents"
    )
    
    @traced_tool(tool_name="video_tool")
    def video_load(path:str):
        video = AudioSegment.from_file(Path(path), format=file[-3:])
        audio = video.split_to_mono()[0]
        file_str = path[:-4] + ".mp3"
        audio.export(file_str, format="mp3")
        model = whisper.load_model(name="base")
        model = cast(whisper.Whisper, model)
        result = model.transcribe(path)
        return result["text"]
    load_video = FunctionTool(
        video_load, description="useful when you need to read mp4 or mov documents"
    )
    
    @traced_tool(tool_name="audio_tool")
    def audio_load(path:str):
        model = whisper.load_model(name="base")
        model = cast(whisper.Whisper, model)
        result = model.transcribe(path)
        return result["text"]
    load_audio = FunctionTool(
        audio_load, description="useful when you need to read mp3 documents"
    )
    ReAct_prompt = "You are a ReAct-based assistant.\nYou analyze the question, decide whether to call a tool or directly answer, and then respond accordingly.\nUse the following format:Question: the input question or request\nThought: you should always think about what to do\nAction: the action to take (if any)\nAction Input: the input to the action (e.g., search query)\nObservation: the result of the action\n... (this process can repeat multiple times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question or request\nBegin!\nQuestion: {input}\n"
    
    agent = AssistantAgent(
        name="react_agent",
        model_client=model_client,
        tools=[python_tool,google_search_tool,pdf_load,csv_load,xlsx_load,txt_load,docs_load,load_image,load_video,load_audio],
        system_message=ReAct_prompt,
        reflect_on_tool_use=True,
        model_client_stream=False,
        memory=None
    )
    return agent

@dataclass
class WorkerTask:
    task: str
    previous_results: List[str]

@dataclass
class WorkerTaskResult:
    result: str

@dataclass
class UserTask:
    task: str

@dataclass
class FinalResult:
    result: str
class WorkerAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
    ) -> None:
        super().__init__(description="Worker Agent")
        self._model_client = model_client
    @message_handler
    async def handle_task(self, message: WorkerTask, ctx: MessageContext) -> WorkerTaskResult:
        if message.previous_results:
            # If previous results are provided, we need to synthesize them to create a single prompt.
            system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
            system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(message.previous_results)])
            model_result = await self._model_client.create(
                [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
            )
        else:
            # If no previous results are provided, we can simply pass the user query to the model.
            message_size = len(str(UserMessage(content=message.task, source="user")).encode("utf-8"))
            packaging_size = message_size- len(str(message.task).encode("utf-8"))
            logging.info(f"source_agent_name: Root Agent, target_agent_name: Agent {self.id.key}, message_size: {message_size}, packaging_size: {packaging_size}")
            model_result = await self._model_client.create([UserMessage(content=message.task, source="user")])
        assert isinstance(model_result.content, str)
        message_size = len(str(WorkerTaskResult(result=model_result.content)).encode("utf-8"))
        packaging_size = message_size- len(str(model_result.content).encode("utf-8"))
        logging.info(f"source_agent_name: Agent {self.id.key}, target_agent_name: aggregation_agent, message_size: {message_size}, packaging_size: {packaging_size}")
        return WorkerTaskResult(result=model_result.content)
class OrchestratorAgent(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_agent_types: List[str],
        num_layers: int,
    ) -> None:
        super().__init__(description="Aggregator Agent")
        self._model_client = model_client
        self._worker_agent_types = worker_agent_types
        self._num_layers = num_layers
    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> FinalResult:
        # Create task for the first layer.
        worker_task = WorkerTask(task=message.task, previous_results=[])
        # Iterate over layers.
        for i in range(self._num_layers - 1):
            # Assign workers for this layer.
            worker_ids = [
                AgentId(worker_type,f"{j+1}")
                for j, worker_type in enumerate(self._worker_agent_types)
            ]
            # Dispatch tasks to workers.
            results = await asyncio.gather(*[self.send_message(worker_task, worker_id) for worker_id in worker_ids])
            # Prepare task for the next layer.
            worker_task = WorkerTask(task=message.task, previous_results=[r.result for r in results])
        # Perform final aggregation.
        system_prompt = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"
        system_prompt += "\n" + "\n\n".join([f"{i+1}. {r}" for i, r in enumerate(worker_task.previous_results)])
        model_result = await self._model_client.create(
            [SystemMessage(content=system_prompt), UserMessage(content=message.task, source="user")]
        )
        assert isinstance(model_result.content, str)
        return FinalResult(result=model_result.content)
        
async def MoA(input_text):
    task = (input_text)
    runtime = SingleThreadedAgentRuntime()
    worker_models = [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-V3"
    ]
    orchestrator_model = "gpt-4o"
    for i, model_name in enumerate(worker_models):
        await WorkerAgent.register(
            runtime,
            f"worker_{i}",
            factory=(lambda model=model_name: WorkerAgent(
                model_client = OpenAIChatCompletionClient(
                    model=model,
                    model_info={
					"vision": False,
					"function_calling": False,
					"json_output": False,
					"family": ModelFamily.UNKNOWN,
					},
                    base_url=os.environ["TOGETHER_URL"],
                    api_key=os.environ["TOGETHER_API_KEY"],
                    temperature=0
                    )
            ))
        )
    await OrchestratorAgent.register(
        runtime,
        "orchestrator",
        lambda: OrchestratorAgent(
            model_client = OpenAIChatCompletionClient(
                model=orchestrator_model,
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_URL"],
                temperature=0
            ),
            worker_agent_types=[f"worker_{i}" for i in range(len(worker_models))],
            num_layers=2
        )
    )
    runtime.start()
    result = await runtime.send_message(UserTask(task=task), AgentId("orchestrator", "default"))
    await runtime.stop_when_idle()
    return result.result

def RAG(model_name):
    from autogen import UserProxyAgent, AssistantAgent
    if model_name == "Together":
        config_list = [
        {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "api_key": os.environ["TOGETHER_API_KEY"],
            "base_url": os.environ["TOGETHER_URL"],
        }
        ]
    elif model_name == "OpenAI":
        config_list = [
        {
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"],
        }
        ]
    import time
    from agents.support import create_vector_db,load_vector_db
    start_time=time.time()
    create_vector_db()
    end_time=time.time()
    global embbeding_time
    embedding_time=end_time-start_time
    logging.info(f"database_vectorization time: {embedding_time}")
    db=load_vector_db()

    def vector_search(query: str, top_k: int = 5) -> str:
        results = db.search(query, k=top_k)
        return "\n".join(i for i in results)

    user_proxy = UserProxyAgent(
        name="user_proxy", 
        human_input_mode="NEVER",
        is_termination_msg=lambda x: True,
        code_execution_config={"use_docker": False}
    )
    
    assistant = AssistantAgent(
        name="assistant", 
        llm_config={"config_list": config_list},
        system_message="You are a helpful assistant. You can answer questions and provide information based on the context provided.",
    )

    import weave
    class rag_agent:
        def __init__(self, user_proxy, assistant):
            self.user_proxy = user_proxy
            self.assistant = assistant

        @weave.op()
        def omni_run(self, input_text:str):
            start_time = time.time()
            search_text=vector_search(input_text)
            end_time = time.time()
            retrieve_time = end_time - start_time
            logging.info(f"retrieve time: {retrieve_time}")

            result=self.user_proxy.initiate_chat(
                self.assistant,
                message=input_text+"some similar questions are:\n"+search_text+"\ngive a final answer.",
            )
            final_message = result.chat_history[-1]["content"]
            
            return final_message,
    return rag_agent(user_proxy, assistant)

class autogen_agent:
    def __init__(self, model_name,agent_type):
        self.agent_type = agent_type
        if self.agent_type == "RAG":
            self.agent=RAG(model_name)
        elif self.agent_type == "MoA":
            self.agent=None
        elif self.agent_type == "ReAct":
            self.agent=ReAct(model_name)
        else:
            print("agent type not supported")
            self.agent=None
    async def omni_run(self, input_text:str):
        if self.agent_type == "RAG":
            return self.agent.omni_run(input_text)
        elif self.agent_type == "MoA":
            return await MoA(input_text)
        elif self.agent_type == "ReAct":
            return await Console(self.agent.run_stream(task=input_text))
        else:
            print("agent type not supported")
            return None
