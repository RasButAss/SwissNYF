import torch
import pandas as pd
from tqdm import tqdm
from llama_index.llms import AzureOpenAI
import json

class BaseRetriever:
    def __init__(self, top_k:int = 5,device='cuda', verbose=True, model_name = 'model_name') -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.verbose = verbose
        self.top_k = top_k

    def set_tool_def(self, tool_def: list) -> None:
        """create a list of tools from json definition"""
        all_tools = []

        # print("setting tools for retriever via base")
        for obj in tool_def:
            if type(obj) is not dict:
                obj = json.loads(obj)
            api_name = obj['tool']
            api_desc = obj['description']
            if 'arguments' in obj:
                for args in obj['arguments']:
                    arg_name = args['name']
                    arg_desc = args['description']
                    arg_type = args['type']
            all_tools.append((api_name, api_desc))

        # print("All tools after being set in retriever")
        self.all_tools = all_tools
        

    def add_tool_def(self, new_tool_def: list) -> None:
        """add new tools, accepts a list of tools in json format """

        new_tools = []
        
        for obj in new_tool_def:
            if type(obj) is not dict:
                obj = json.loads(obj)
            api_name = obj['tool']
            api_desc = obj['description']
            for args in obj['arguments']:
                arg_name = args['name']
                arg_desc = args['description']
                arg_type = args['type']
            new_tools.append((api_name, api_desc))
        self.all_tools.extend(new_tools)


    def filter(self, query, top_k=5) -> list:
        """returns the top_k tools for a given query"""
        pass 


