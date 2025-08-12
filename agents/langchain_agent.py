from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from langchain.agents import Tool,AgentExecutor,create_react_agent
from langchain_core.tools import tool
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import PyPDFLoader,UnstructuredImageLoader,UnstructuredExcelLoader,Docx2txtLoader, UnstructuredFileLoader
from langchain_community.document_loaders.assemblyai import TranscriptFormat,AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel, Field
import asyncio
# import nest_asyncio
from langchain_core.messages import HumanMessage, SystemMessage
# from pympler import asizeof
import weave
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import os
from dotenv import load_dotenv
load_dotenv()

#from langchain.callbacks import OpenAICallbackHandler
#handler = OpenAICallbackHandler()

import logging
import time
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

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

class LangchainAgent:
    def __init__(self,model_name,agent_type):
        if model_name == "OpenAI":
            llm = ChatOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_URL"],
                disable_streaming=True,
                model="gpt-4o",
                temperature=0
                )
        elif model_name == "Together":
            llm = ChatOpenAI(
                api_key=os.getenv("TOGETHER_API_KEY"),
                model="meta-llama/Llama-3-70b-chat-hf", 
                base_url=os.environ["TOGETHER_URL"],
                temperature=0
            )
        else:
            print("Model not supported")
        
        if agent_type == "ReAct":
            self.agent = ReACt(llm)
        elif agent_type == "RAG":
            self.agent = RAG(llm)
        elif agent_type == "MoA":
            self.agent = None
    
    @weave.op()
    async def omni_run(self, question):
        if self.agent:
            return self.agent.invoke({"input": question})["output"]
        else:
            return await MoA(question)

async def MoA(question):
    message_size=len(question.encode('utf-8'))
    for i in range(len(reference_models)):
        logging.info(f"source_agent_name: Root Agent, target_agent_name: Agent {i+1}, message_size: {message_size}, packaging_size: 0")
    
    aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    
    Responses from models:"""
    reference_models = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3"
    ]
    async def run_model(model_id):
        model = ChatTogether(
            model=model_id,
            together_api_key=os.environ["TOGETHER_API_KEY"],
            temperature=0
        )
        try:
            response = await model.ainvoke([HumanMessage(content=question)])
            return response.content
        except Exception as e:
            print(f"[{model_id}] Error: {e}")
            return f"Error from model {model_id}: {e}"
    # reference models reply
    results = await asyncio.gather(*[run_model(m) for m in reference_models])
    formatted_list = [f"{i+1}. {res}" for i, res in enumerate(results)]
    for i in range(len(reference_models)):
        message_size=len(formatted_list[i].encode('utf-8'))
        packaging_size = message_size- len(results[i].encode('utf-8'))
        logging.info(f"source_agent_name: Agent {i+1}, target_agent_name: aggregation_agent, message_size: {message_size}, packaging_size: {packaging_size}")
    formatted_outputs = "\n".join(formatted_list)
    
    aggregation_prompt = [
        SystemMessage(content=aggregator_system_prompt + "\n" + formatted_outputs),
        HumanMessage(content=question)
    ]
    aggregator = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_URL"],
        stream_usage = True,
        model="gpt-4o",
        temperature=0
    )
    final_response = await aggregator.ainvoke(aggregation_prompt)
    return final_response.content


def ReACt(llm):
    from langchain_google_community import GoogleSearchAPIWrapper
    import os
#    os.environ["GOOGLE_CSE_ID"]
#    os.environ["GOOGLE_API_KEY"]
    search = GoogleSearchAPIWrapper()
    google_search = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=traced_tool(search.run, tool_name="web_browser_tool")
    )
    
    Python_excutor=Tool (
        name="Python_executor",
        description = "useful for when you need to execute python code",
        func=traced_tool(PythonREPLTool().run, tool_name="python_tool")
    )

    class Path(BaseModel):
        path: str = Field(description="path to file")

    @tool(name_or_callable = "pdf_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a PDF document from a path"
    )
    @traced_tool(tool_name="pdf_tool")
    def pdf_load(path: str) -> str:
        loader = PyPDFLoader(
            path,
            mode ="single",
        )
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @tool(name_or_callable = "docx_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a docx document from a path"
    )
    @traced_tool(tool_name="docx_tool")
    def docx_load(path: str) -> str:
        loader = Docx2txtLoader(path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @tool(name_or_callable = "txt_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a txt document from a path"
    )
    @traced_tool(tool_name="txt_tool")
    def txt_load(path: str) -> list:
        loader = UnstructuredFileLoader(path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @tool(name_or_callable = "mp3_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a mp3 document from a path"
    )
    @traced_tool(tool_name="audio_tool")
    def mp3_load(path: str) -> str:
        loader = AssemblyAIAudioTranscriptLoader(
            file_path = path,
            transcript_format = TranscriptFormat.TEXT,
        )
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @tool(name_or_callable = "image_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a jpg or png document from a path"
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
        
    @tool(name_or_callable = "csv_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a csv document from a path"
    )
    @traced_tool(tool_name="csv_tool")
    def csv_load(path:str) -> str:
        loader = CSVLoader(file_path=path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
    
    @tool(name_or_callable = "xlsx_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a xlsx document from a path"
    )
    @traced_tool(tool_name="xlsx_tool")
    def xlsx_load(path:str) -> str:
        loader = UnstructuredExcelLoader(file_path=path, mode="single")
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)

    @tool(name_or_callable = "video_loader", 
        args_schema = Path, 
        return_direct = True,
        description = "useful for when you need to load a video document from a path"
    )
    @traced_tool(tool_name="video_tool")
    def video_load(path: str) -> str:
        import whisper
        from pydub import AudioSegment
        from typing import cast
        video = AudioSegment.from_file(Path(path), format=path[-3:])
        audio = video.split_to_mono()[0]
        file_str = path[:-4] + ".mp3"
        audio.export(file_str, format="mp3")
        model = whisper.load_model(name="base")
        model = cast(whisper.Whisper, model)
        result = model.transcribe(path)
        return result["text"]
    
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

    StructuredTool.from_function
    twoSumTool = StructuredTool.from_function(twoSum)
    lengthOfLongestSubstringTool = StructuredTool.from_function(lengthOfLongestSubstring)
    findMedianSortedArraysTool = StructuredTool.from_function(findMedianSortedArrays)
    longestPalindromeTool = StructuredTool.from_function(longestPalindrome)
    convertZTool = StructuredTool.from_function(convertZ)
    reverseXTool = StructuredTool.from_function(reverseX)
    myAtoiTool = StructuredTool.from_function(myAtoi)
    isPalindromeTool = StructuredTool.from_function(isPalindrome)
    isMatchTool = StructuredTool.from_function(isMatch)
    maxAreaTool = StructuredTool.from_function(maxArea)

    longestCommonPrefixTool = StructuredTool.from_function(longestCommonPrefix)
    threeSumTool = StructuredTool.from_function(threeSum)
    isValidBracketsTool = StructuredTool.from_function(isValidBrackets)
    generateParenthesisTool = StructuredTool.from_function(generateParenthesis)
    groupAnagramsTool = StructuredTool.from_function(groupAnagrams)
    lengthOfLastWordTool = StructuredTool.from_function(lengthOfLastWord)
    addBinaryTool = StructuredTool.from_function(addBinary)
    minDistanceTool = StructuredTool.from_function(minDistance)
    largestNumberTool = StructuredTool.from_function(largestNumber)
    reverseStringTool = StructuredTool.from_function(reverseString)


    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    For example, you can use Python_excutor to execute python code to make sure its correctness.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''
    prompt = PromptTemplate.from_template(template)
    tools =[google_search,Python_excutor,pdf_load,docx_load,txt_load,mp3_load,image_load,csv_load,xlsx_load,video_load,
            # irrelevant tools
            # twoSumTool,
            # lengthOfLongestSubstringTool,
            # findMedianSortedArraysTool,
            # longestPalindromeTool,
            # convertZTool,
            # reverseXTool,
            # myAtoiTool,
            # isPalindromeTool,
            # isMatchTool,
            # maxAreaTool,

            # longestCommonPrefixTool,
            # threeSumTool,
            # isValidBracketsTool,
            # generateParenthesisTool,
            # groupAnagramsTool,
            # lengthOfLastWordTool,
            # addBinaryTool,
            # minDistanceTool,
            # largestNumberTool,
            # reverseStringTool,
            ]
    agent=create_react_agent(llm, tools,prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    return agent_executor
    
def RAG(llm):
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    from data.mmlu import merge_csv_files_in_folder
    import time

    start_time=time.time()
    dataset=merge_csv_files_in_folder("data/MMLU/dev")
    
    raw_docs = []
    for item in dataset:
        text = item[0].replace(",please answer A,B,C,or D.",",")+f"answer is {item[1]}."
        raw_docs.append(Document(page_content=text))
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(raw_docs, embedding_model)
    db.save_local("db")
    end_time=time.time()
    global embedding_time
    embedding_time = end_time - start_time
    logging.info(f"database_vectorization time: {embedding_time}")

    from langchain.chains import RetrievalQA
    from langchain.vectorstores import FAISS
    
    class MyRetrievalQA(RetrievalQA):
        def invoke(self,q):
            query = q["input"]
            start_time = time.time()
            docs=self.retriever.get_relevant_documents(query)
            end_time = time.time()
            retrieve_time = end_time - start_time
            logging.info(f"retrieve time: {retrieve_time}")

            result = self.combine_documents_chain.run({"input_documents": docs,"question": query})
            end_time = time.time()
            return {'output':result}

    
    agent_executor = MyRetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        )
    return agent_executor
