import yaml
import os
import time
from eval.summary import summary_log
parent_dir = os.path.dirname(os.path.abspath(__file__))

class benchmark:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def run(self):
        frameworks = self.config['frameworks']
        datasets = self.config['datasets']

        for framework in frameworks:
            for dataset in datasets:
                self._run_single_test(framework, dataset)

    def _run_single_test(self, framework, dataset):
        if framework=="langchain":
            if dataset=="mmlu":
                from agentrun.langchain_run import mmlu_test
                mmlu_test()
                time.sleep(2)
                summary_log(file_path=os.path.join(parent_dir,"results","langchain","log","langchain_rag.log"),save_path=os.path.join(parent_dir,"results","langchain","mmlu"),reverse=False,weave_op=True,millisecond=True,include_op_name=True,time_lens=23)
            elif dataset=="alpaca_eval":
                from agentrun.langchain_run import moa_test
                moa_test()
            elif dataset=="gaia":
                from agentrun.langchain_run import react_test
                react_test()
        if framework=="autogen":
            if dataset=="mmlu":
                from agentrun.autogen_run import mmlu_RAG
                mmlu_RAG()
                time.sleep(2)
                summary_log(file_path=os.path.join(parent_dir,"results","autogen","log","autogen_rag.log"),save_path=os.path.join(parent_dir,"results","autogen","mmlu"),reverse=False,weave_op=True,millisecond=True,include_op_name=True,time_lens=23)
        if framework=="agentscope":
            if dataset=="gaia":
                from agentrun.agentscoperun import react_GAIA_test
                react_GAIA_test()
                time.sleep(2)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","react-gaia.log"),save_path=os.path.join(parent_dir,"results","agentscope","react-gaia"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
            if dataset=="mmlu":
                from agentrun.agentscoperun import rag_test
                rag_test()
            if dataset=="humaneval":
                from agentrun.agentscoperun import react_humaneval_test
                react_humaneval_test(1)
                react_humaneval_test(2)
                react_humaneval_test(3)
                time.sleep(2)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","react-humaneval1.log"),save_path=os.path.join(parent_dir,"results","agentscope","react-humaneval1"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","react-humaneval2.log"),save_path=os.path.join(parent_dir,"results","agentscope","react-humaneval1"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","react-humaneval3.log"),save_path=os.path.join(parent_dir,"results","agentscope","react-humaneval1"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
            if dataset=="alpaca_eval":
                from agentrun.agentscoperun import moa_test
                moa_test(3)
                moa_test(6)
                moa_test(9)
                moa_test(12)
                moa_test(15)
                time.sleep(2)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","moa3.log"),save_path=os.path.join(parent_dir,"results","agentscope","moa3"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","moa6.log"),save_path=os.path.join(parent_dir,"results","agentscope","moa6"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","moa9.log"),save_path=os.path.join(parent_dir,"results","agentscope","moa9"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","moa12.log"),save_path=os.path.join(parent_dir,"results","agentscope","moa12"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
                summary_log(file_path=os.path.join(parent_dir,"results","agentscope","log","moa15.log"),save_path=os.path.join(parent_dir,"results","agentscope","moa15"),reverse=False,weave_op=False,millisecond=True,include_op_name=True,time_lens=23)
        if framework == 'llamaindex':
            if dataset=="gaia":
                from agentrun.llamaindex_run import react_gaia
                react_gaia()
            if dataset=="mmlu":
                from agentrun.llamaindex_run import rag_mmlu
                rag_mmlu()
            if dataset=="humaneval":
                from agentrun.llamaindex_run import react_human_eval
                react_human_eval()
            if dataset=="alpaca_eval":
                from agentrun.llamaindex_run import moa_alpacaeval
                moa_alpacaeval()
        if framework == 'phidata':
            if dataset=="gaia":
                from agentrun.phidata_run import react_gaia
                react_gaia()
            if dataset=="mmlu":
                from agentrun.phidata_run import rag_mmlu
                rag_mmlu()
            if dataset=="humaneval":
                from agentrun.phidata_run import react_human_eval
                react_human_eval()
            if dataset=="alpaca_eval":
                from agentrun.phidata_run import moa_alpacaeval
                moa_alpacaeval()
        if framework == 'pydantic':
            if dataset=="gaia":
                from agentrun.pydantic_run import react_gaia
                react_gaia()
            if dataset=="mmlu":
                from agentrun.pydantic_run import rag_mmlu
                rag_mmlu()
            if dataset=="humaneval":
                from agentrun.pydantic_run import react_human_eval
                react_human_eval()
            if dataset=="alpaca_eval":
                from agentrun.pydantic_run import moa_alpacaeval
                moa_alpacaeval()
        if framework == 'crewai':
            if dataset=="gaia":
                from agentrun.crewai_run import react_gaia_eval
                react_gaia_eval()
            if dataset=="mmlu":
                from agentrun.crewai_run import rag_eval
                rag_eval()
            if dataset=="humaneval":
                from agentrun.crewai_run import react_human_eval
                react_human_eval()
            if dataset=="alpaca_eval":
                from agentrun.crewai_run import moa_eval
                moa_eval(moa_num=3)

benchmark("configs/config.yaml").run()