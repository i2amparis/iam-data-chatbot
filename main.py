import os
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
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import messages_from_dict, messages_to_dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # File handler for all logs
        logging.FileHandler('chatbot.log'),
        # Custom stream handler for errors only
        logging.StreamHandler(stream=open(os.devnull, 'w'))  # Suppress console output
    ]
)

def setup_logging(debug: bool = False):
    """Configure logging with optional debug mode"""
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler - always log everything to file
    file_handler = logging.FileHandler('chatbot.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    if debug:
        # Console handler - only used in debug mode
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console_handler)

def docs_from_records(records: list) -> List[Document]:
    """Convert API records to Document objects for vector store"""
    docs = []
    for rec in records:
        # Extract description and assumptions
        desc = (rec.get("description") or rec.get("modelName") or "").strip()
        asum = (rec.get("assumptions") or "").strip()
        if not desc and not asum:
            continue
            
        # Combine text with assumptions if present
        content = desc + (f"\n\nAssumptions: {asum}" if asum else "")
        
        # Create document with metadata
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
    """Build or load FAISS vector store"""
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        return FAISS.load_local(
            index_dir, 
            embeddings,
            allow_dangerous_deserialization=True
        )
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(index_dir)
    return store

from data_utils import data_query

def display_plot_from_base64(base64_string: str):
    """Display plot image from base64 string"""
    try:
        # Extract base64 data from markdown string
        if "data:image/png;base64," in base64_string:
            base64_data = base64_string.split("data:image/png;base64,")[1]
        else:
            base64_data = base64_string

        # Validate base64 data
        if not base64_data or not base64_data.strip():
            raise ValueError("Empty base64 data")

        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception as decode_error:
            raise ValueError(f"Invalid base64 data: {decode_error}")

        # Create PIL Image from bytes
        try:
            image = Image.open(BytesIO(image_bytes))
        except Exception as image_error:
            raise ValueError(f"Invalid image data: {image_error}")

        # Display using matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(f"Error displaying plot: {e}")
    except Exception as e:
        print(f"Unexpected error displaying plot: {e}")

class IAMParisBot:
    def __init__(self, streaming: bool = True):
        self.streaming = streaming
        self.logger = logging.getLogger(__name__)
        self.load_env()
        self.history: List[Tuple[str, str]] = []
        
    def load_env(self) -> None:
        """Load and validate environment variables"""
        load_dotenv(override=True)
        required = ["OPENAI_API_KEY", "REST_MODELS_URL", "REST_API_FULL"] 
        env = {var: os.getenv(var) for var in required}
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
        missing = [k for k, v in env.items() if not v]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
        self.env = env

    def fetch_json(self, url: str, params: Optional[Dict] = None,
                  payload: Optional[Dict] = None, cache: bool = True) -> list:
        """Fetch JSON with caching - handles both API formats"""
        cache_file = f"cache/{url.split('/')[-1]}_{hash(str(params) + str(payload))}.json"
        os.makedirs("cache", exist_ok=True)

        if cache and os.path.exists(cache_file):
            self.logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'r') as f:
                return pd.read_json(f).to_dict('records')

        self.logger.info(f"Fetching data from {url}")
        resp = requests.post(url, json=payload) if payload else requests.get(url, params=params)
        resp.raise_for_status()
        json_data = resp.json()

        # Handle different API response formats
        if "data" in json_data:
            # IAM PARIS results API format
            data = json_data["data"]
        elif isinstance(json_data, list):
            # Direct list response (CMS API)
            data = json_data
        else:
            # Single object response - wrap in list
            data = [json_data]

        self.logger.info(f"Retrieved {len(data)} records from API")

        if cache:
            with open(cache_file, 'w') as f:
                pd.DataFrame(data).to_json(f)

        return data

    def create_qa_chain(self, vs: FAISS) -> ConversationalRetrievalChain:
        """Create optimized QA chain with IAM PARIS specific prompting"""
        # New memory configuration using ChatMessageHistory
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
        
        system_tpl = """You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

    Always:
    - Provide direct answers without restating the question
    - Use Markdown formatting for responses with proper headers (##) and lists (-)
    - Reference specific IAM PARIS data points when available 
    - Clearly indicate when information comes from external sources
    - Include relevant IAM PARIS links when referencing specific studies
    - Format numerical values with proper units
    - Keep answers focused and data-driven

    Available IAM PARIS resources:
    - Model documentation: https://iamparis.eu/models
    - Results database: https://iamparis.eu/results
    - Study descriptions: https://iamparis.eu/studies
    
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
        
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )

    def plot_time_series(self, records: list, year: Optional[int] = None,
                        model: Optional[str] = None, variable: Optional[str] = None,
                        save: bool = False) -> None:
        """Enhanced plotting with better formatting and save option"""
        df = pd.DataFrame(records)
        if df.empty:
            self.logger.warning("No data available to plot")
            return
            
        years = sorted([c for c in df.columns if c.isdigit()], key=int)
        if not years:
            self.logger.warning("No year data available")
            return
            
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn')
        
        dfc = df.copy()
        if model:
            dfc = dfc[dfc['modelName'].str.lower() == model.lower()]
        if variable:
            dfc = dfc[dfc['variable'].str.contains(variable, case=False, na=False)]
            
        unit = dfc['unit'].iloc[0] if 'unit' in dfc.columns and not dfc.empty else ''
        
        if year and str(year) in years:
            vals = dfc.groupby('modelName')[str(year)].mean()
            ax = vals.plot(kind='bar', color='skyblue', alpha=0.7)
            plt.title(f"Values in {year}\n{'Model: ' + model if model else 'All Models'}", 
                     pad=20, fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(vals):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
                
        else:
            for name, grp in dfc.groupby('modelName'):
                pts = [grp[str(y)].mean() for y in years]
                plt.plot(years, pts, marker='o', label=name, linewidth=2, markersize=6)
                
            plt.title(f"Time Series\n{'Model: ' + model if model else 'All Models'}", 
                     pad=20, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.xlabel("Year", fontsize=10, labelpad=10)
        plt.ylabel(f"Value ({unit})" if unit else "Value", fontsize=10, labelpad=10)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plots/plot_{timestamp}.png"
            os.makedirs("plots", exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved as {filename}")
            
        plt.show()

    def run(self):
        """Main interaction loop"""
        self.logger.info("Loading data...")
        
        # Load model and time series data
        models = self.fetch_json(self.env["REST_MODELS_URL"], params={"limit": -1})

        # Try to get data with time series (year columns)
        ts = self.fetch_json(
            self.env["REST_API_FULL"],
            payload={
                "study": ["paris-reinforce"],
                "workspace_code": ["eu-headed"],
                "include_years": True  # Request time series data
            }
        )

        # If no time series data, try alternative data sources
        if not ts or not any(str(col).isdigit() for record in ts for col in record.keys()):
            self.logger.info("No time series data found, trying alternative data source...")
            ts = self.fetch_json(
                self.env["REST_API_FULL"],
                payload={
                    "study": ["all"],  # Try getting all studies
                    "limit": 1000  # Increase limit to get more data
                }
            )
        
        # Create vector store and QA chain
        docs = docs_from_records(models + ts)
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80
        ).split_documents(docs)
        
        emb = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.env["OPENAI_API_KEY"]
        )
        vs = build_faiss_index(chunks, emb)
        chain = self.create_qa_chain(vs)
        
        self.logger.info("Bot ready for interaction")
        print("\nWelcome to the IAM PARIS Climate Policy Assistant!")
        print("Ask me about climate models, scenarios, policies and data.")
        print("Type 'help' for available commands or 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input without keeping it in the response
                query = input("YOU: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if not query:
                    continue
                
                # Handle data query
                ans = data_query(query, models, ts)
                if ans:
                    print("\nBOT:", ans)
                    print()
                    self.history.append((query, ans))
                    continue

                # Handle QA chain response with streaming output
                if self.streaming:
                    print("\nBOT: ", end="", flush=True)
                    # Use callback handler to stream output
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                    print()
                    ans = resp.get("answer", "").strip()
                else:
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                    ans = resp.get("answer", "").strip()

                # Store in history without the "You:" part
                if ans:
                    # Check if the answer is a plot markdown string
                    if ans.startswith("![Plot]"):
                        # Display the plot image visually
                        display_plot_from_base64(ans)
                        # Append to history but do not print the base64 markdown text
                        self.history.append((query, "[Plot Image]"))
                    else:
                        if not self.streaming:
                            print("\nBOT:", ans)
                        print()
                        self.history.append((query, ans))
                else:
                    if self.streaming:
                        print("I cannot answer that question.")
                    else:
                        print("\nBOT: I cannot answer that question.")
                    print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                self.logger.error(f"Error processing query: {str(e)}")
                print("\nBOT: Sorry, I encountered an error. Please try again.")

import logging
from manager import MultiAgentManager

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IAM PARIS Climate Policy Assistant with Multi-Agent Support")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (shows all logs)")
    args = parser.parse_args()

    # Set up logging based on debug flag
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    # Initialize the original bot to access shared resources and plotting
    bot = IAMParisBot(streaming=not args.no_stream)
    logger.info("Loading data and resources for multi-agent system...")

    # Load data and resources as in IAMParisBot.run but without starting loop
    logger.info("Fetching model data...")
    models = bot.fetch_json(bot.env["REST_MODELS_URL"], params={"limit": -1})
    logger.info(f"Fetched {len(models)} model records")

    logger.info("Fetching time series data...")
    # Try to get data with time series (year columns)
    ts = bot.fetch_json(
        bot.env["REST_API_FULL"],
        payload={
            "study": ["paris-reinforce"],
            "workspace_code": ["eu-headed"],
            "include_years": True  # Request time series data
        }
    )

    # If no time series data, try alternative data sources with year columns
    if not ts or not any(str(col).isdigit() for record in ts for col in record.keys()):
        logger.info("No time series data found, trying alternative data source...")
        ts = bot.fetch_json(
            bot.env["REST_API_FULL"],
            payload={
                "study": ["all"],  # Try getting all studies
                "limit": 1000,  # Increase limit to get more data
                "include_years": True  # Ensure we get year columns
            }
        )

        # If still no year columns, try one more time with different parameters
        if not ts or not any(str(col).isdigit() for record in ts for col in record.keys()):
            logger.info("Still no year columns, trying with specific studies...")
            ts = bot.fetch_json(
                bot.env["REST_API_FULL"],
                payload={
                    "study": ["paris-reinforce", "emissions-gap", "1.5c-scenarios"],
                    "workspace_code": ["eu-headed", "global"],
                    "include_years": True,
                    "limit": 2000
                }
            )
            
            # Debug: Log sample of time series data to understand structure
            if ts:
                logger.info(f"Sample time series record: {ts[0] if ts else 'No data'}")
                sample_keys = list(ts[0].keys())[:10]  # First 10 keys
                logger.info(f"Time series data keys: {sample_keys}")
                # Check for any numeric columns that might be years
                numeric_cols = [k for k in ts[0].keys() if str(k).isdigit()]
                logger.info(f"Numeric columns (potential years): {numeric_cols}")
    logger.info(f"Fetched {len(ts)} time series records")

    # Debug: Check if models have modelName field
    if models:
        sample_model = models[0] if models else {}
        logger.info(f"Sample model data: {sample_model}")
        model_names = [r.get('modelName', '') for r in models if r.get('modelName')]
        logger.info(f"Found {len(model_names)} models with modelName: {model_names[:5]}...")

    # Debug: Check time series data
    if ts:
        sample_ts = ts[0] if ts else {}
        logger.info(f"Sample time series data: {sample_ts}")
        scenarios = list(set(r.get('scenario', '') for r in ts if r.get('scenario')))
        logger.info(f"Found scenarios: {scenarios[:5]}...")

    # Prepare documents and vector store
    docs = docs_from_records(models + ts)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    ).split_documents(docs)

    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=bot.env["OPENAI_API_KEY"]
    )
    vs = build_faiss_index(chunks, emb)

    # Shared resources dictionary
    shared_resources = {
        "models": models,
        "ts": ts,
        "vector_store": vs,
        "env": bot.env,
        "bot": bot  # For plotting agent to call plot_time_series
    }

    # Initialize multi-agent manager
    manager = MultiAgentManager(shared_resources, streaming=not args.no_stream)

    # Pre-compute common plots for faster access
    from simple_plotter import simple_plotter
    # Removed precompute_common_plots call as caching is disabled and method removed
    # simple_plotter.precompute_common_plots(ts, models)

    logger.info("Multi-agent manager initialized. Ready for interaction.")

    print("\nWelcome to the IAM PARIS Climate Policy Assistant with Multi-Agent Support!")
    print("Ask me about climate models, scenarios, policies, data, or request plots.")
    print("Type 'help' for available commands or 'exit' to quit.\n")

    history = []

    while True:
        try:
            query = input("YOU: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            # Route query to appropriate agent
            response = manager.route_query(query, history)

            # Check if the response is a plot markdown string
            if response.startswith("![Plot]"):
                # Display the plot image visually
                display_plot_from_base64(response)
                # Append to history but do not print the base64 markdown text
                history.append((query, "[Plot Image]"))
            else:
                print("\nBOT:", response, "\n")
                # Append to history
                history.append((query, response))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("\nBOT: Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    main()
