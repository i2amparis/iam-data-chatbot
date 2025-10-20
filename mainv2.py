import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from data_utils import data_query
from utils.yaml_loader import load_all_yaml_files, yaml_to_documents
from manager import MultiAgentManager
from utils_query import (
    get_available_models,
    get_available_scenarios,
    get_available_variables_from_yaml)

from pathlib import Path

#Variable Definitions for Energy Systems
variable_dict = load_all_yaml_files('definitions/variable')

def setup_logging(debug: bool = False):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler('chatbot.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console_handler)

def load_definitions():
    region_path = Path('definitions/region').resolve()
    variable_path = Path('definitions/variable').resolve()
    region_yaml = load_all_yaml_files(str(region_path))
    variable_yaml = load_all_yaml_files(str(variable_path))
    return yaml_to_documents(region_yaml), yaml_to_documents(variable_yaml)

region_docs , variable_docs = load_definitions()

def docs_from_records(records: list) -> List[Document]:
    docs = []
    for rec in records:
        if rec is None:
            continue
        desc = (rec.get("description") or rec.get("modelName") or "").strip()
        asum = (rec.get("assumptions") or "").strip()
        if not desc and not asum:
            continue
        content = desc + (f"\n\nAssumptions: {asum}" if asum else "")
        doc = Document(
            page_content=content,
            metadata={
                "modelName": rec.get("modelName", ""),
                "variable": rec.get("variable", ""),
                "unit": rec.get("unit", ""),
                "study": rec.get("study", ""),
                "scenario": rec.get("scenario", ""),
                "type": "model" if "modelName" in rec else "timeseries"
            }
        )
        docs.append(doc)
    return docs

def build_faiss_index(docs: list, embeddings) -> FAISS:
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(index_dir)
    return store

def display_plot_from_base64(base64_string: str):
    try:
        if "data:image/png;base64," in base64_string:
            base64_data = base64_string.split("data:image/png;base64,")[1]
        else:
            base64_data = base64_string
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")

class IAMParisBot:
    def __init__(self, streaming: bool = True):
        self.streaming = streaming
        self.logger = logging.getLogger(__name__)
        self.history: List[Tuple[str, str]] = []
        self.load_env()

    def load_env(self):
        load_dotenv(override=True)
        required = ["OPENAI_API_KEY", "REST_MODELS_URL", "REST_API_FULL"]
        self.env = {k: os.getenv(k) for k in required}
        if missing := [k for k, v in self.env.items() if not v]:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    def fetch_json(self, url: str, params=None, payload=None, cache=True) -> list:
        os.makedirs("cache", exist_ok=True)
        cache_file = f"cache/{url.split('/')[-1]}_{hash(str(params) + str(payload))}.json"
        if cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return pd.read_json(f).to_dict('records')
        if payload:
            resp = requests.post(url, json=payload)
        else:
            resp = requests.get(url, params=params)
        print(f"API call to {url}: status {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data") if isinstance(data, dict) else data
        print(f"Records fetched: {len(records)}")
        with open(cache_file, 'w') as f:
            pd.DataFrame(records).to_json(f)
        return records

    def create_qa_chain(self, vs: FAISS) -> ConversationalRetrievalChain:
        memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
        system_tpl = """You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Always:
- Provide direct answers without restating the question
- Use Markdown formatting with headers and lists
- Reference IAM PARIS data when available
- Include IAM PARIS links
- Format numbers with units

Context: ```{context}```"""
        user_tpl = "Question: ```{question}```"
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_tpl),
            HumanMessagePromptTemplate.from_template(user_tpl)
        ])
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=self.streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if self.streaming else None,
            api_key=self.env["OPENAI_API_KEY"]
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--query", type=str, help="Single query to process and exit")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    bot = IAMParisBot(streaming=not args.no_stream)
    models = bot.fetch_json(bot.env["REST_MODELS_URL"], params={"limit": -1}, cache=False)
    # Remove limit to get all available data for energy-systems workspace
    ts_payload = {"workspace_code": ["energy-systems"]}
    logger.info(f"Fetching timeseries with payload: {ts_payload}")
    ts = bot.fetch_json(bot.env["REST_API_FULL"], payload=ts_payload, cache=False)
    logger.info(f"Timeseries records fetched: {len(ts)}")
    print(f"ts fetch: {len(ts)} records")

    region_docs, variable_docs = load_definitions()
    all_docs = docs_from_records(models + ts) + region_docs + variable_docs
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=bot.env["OPENAI_API_KEY"])
    faiss_index = build_faiss_index(chunks, embeddings)

    shared_resources = {
        "models": models,
        "ts": ts,
        "vector_store": faiss_index,
        "env": bot.env,
        "bot": bot
    }

    manager = MultiAgentManager(shared_resources, streaming=not args.no_stream)

    if args.query:
        # Process single query and exit
        history = []
        response = manager.route_query(args.query, history)
        if response.startswith("![Plot]"):
            display_plot_from_base64(response)
            print("Response: [Plot Image]")
        else:
            print("Response:", response)
        return

    print("\nWelcome to the IAM PARIS Climate Policy Assistant! Type 'exit' to quit.\n")

    print("Examples (filtered for energy-systems workspace):")
    # Filter models to those available in the loaded ts data
    available_models = set(r.get('modelName', '') for r in ts if r and r.get('modelName'))
    model_names = [name for name in available_models if name][:5]
    print("Models:", ", ".join(sorted(model_names)))
    print("Scenarios:", ", ".join(get_available_scenarios(ts)[:5]))

    # Filter variables to those available in the loaded ts data
    available_variables = set(r.get('variable', '') for r in ts if r and r.get('variable'))
    variable_names = [name for name in available_variables if name][:10]
    print("Variables:", ", ".join(sorted(variable_names)))

    history = []
    while True:
        try:
            query = input("YOU: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            response = manager.route_query(query, history)
            if response.startswith("![Plot]"):
                display_plot_from_base64(response)
                history.append((query, "[Plot Image]"))
            else:
                print("\nBOT:", response, "\n")
                history.append((query, response))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("\nBOT: An error occurred. Please try again.\n")

if __name__ == "__main__":
    main()
