"""
@Author: Rossi
Created At: 2021-02-21
"""

import json
import time

from mako.template import Template

from Broca.faq_engine.index import ESIndex, VectorIndex
from Broca.message import BotMessage


class FAQAgent:
    def __init__(self, agent_name, es_index, vector_index, threshold, topk, prompt_threshold, 
            template, prompt_template):
        self.agent_name = agent_name
        self.es_index = es_index
        self.vector_index = vector_index
        self.threshold = threshold
        self.topk = topk
        self.prompt_threshold = prompt_threshold
        self.template = template
        self.prompt_template = prompt_template

    @classmethod
    def from_config(cls, config):
        agent_name = config["agent_name"]
        es_config = config["es_index"]
        es_index = ESIndex.from_config(es_config)
        vector_index_config = config["vector_index"]
        vector_index = VectorIndex.from_config(vector_index_config)
        if config["build_index_at_start"]:
            es_index.build_index_from_file(config["document_file"])
            time.sleep(5)  # wait until the es index gets ready
            vector_index.build_index(es_index)
        vector_index.load_index()
        threshold = config["threshold"]
        topk = config["topk"]
        prompt_threshold = config["prompt_threshold"]
        template = Template(filename=config["template"])
        prompt_template = Template(filename=config["prompt_template"])
        return cls(agent_name, es_index, vector_index, threshold, topk, prompt_threshold, template, prompt_template)

    @classmethod
    def from_config_file(cls, config_file):
        with open(config_file, encoding="utf-8") as fi:
            config = json.load(fi)
            return cls.from_config(config)

    def handle_message(self, message):
        """Respond to the user message by retriving documents from the knowledge base. 
    
        Args:
            message ([type]): [description]
        """
        query = message.text
        candidates, similarities = self.vector_index.retrieve(query, self.topk)
        selected = [candidate for candidate, similarity in zip(candidates, similarities) if similarity >= self.threshold]
        result = {}
        if selected:
            documents = self.es_index.get_answer_by_question_ids(selected)
            response = self.template.render(documents=documents)
            result["response"] = BotMessage(message.sender_id, response.strip())
        else:
            selected = [candidate for candidate, similarity in zip(candidates, similarities) if similarity >= self.prompt_threshold]
            if selected:
                documents = self.es_index.get_documents_by_ids(selected)
                prompt = self.prompt_template.render(documents=documents)
                result["prompt"] = BotMessage(message.sender_id, prompt.strip())
        return result
