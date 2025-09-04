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

def data_query(question: str, model_data: list, ts_data: list) -> str:
    """Direct data lookup without using the LLM"""
    q = question.lower()
    
    # Handle plot requests
    if any(word in q for word in ['plot', 'show', 'graph', 'visualize']):
        try:
            # Debug information
            logging.info(f"Total records in ts_data: {len(ts_data)}")
            
            # Create DataFrame
            df = pd.DataFrame(ts_data)
            logging.info(f"DataFrame columns: {df.columns}")
            
            # Filter for emissions variables
            emissions_vars = []
            if any(word in q for word in ['emission', 'co2', 'ghg']):
                emissions_vars = [v for v in df['variable'].unique() 
                                if 'emission' in str(v).lower() or 'co2' in str(v).lower()]
                logging.info(f"Found emission variables: {emissions_vars}")
                
                if emissions_vars:
                    df = df[df['variable'].isin(emissions_vars)]
                else:
                    return """## No Emission Data Found

I couldn't find any emission-related variables in the current dataset. Available variables are:
""" + "\n".join([f"- {v}" for v in sorted(df['variable'].unique())]) + """

Try:
- Using `list variables` to see all available variables
- Plotting a different variable
- Checking the data source at [IAM PARIS Results](https://iamparis.eu/results)
"""
            
            # Get year columns
            year_cols = [col for col in df.columns if str(col).isdigit()]
            logging.info(f"Year columns found: {year_cols}")
            
            if not year_cols:
                return """## No Time Series Data Available

The current dataset doesn't contain any year-based data columns. This might be because:
- The data hasn't been loaded correctly
- The selected variables don't have time series data
- The data format is different than expected

Try:
1. Refreshing the data connection
2. Using `list variables` to see available data
3. Checking the [IAM PARIS Results](https://iamparis.eu/results) for available time series"""
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot data for each model/variable combination
            for (model_name, var), group in df.groupby(['modelName', 'variable']):
                years = sorted(year_cols)
                values = [float(group[year].mean()) for year in years if not pd.isna(group[year].mean())]
                if values:  # Only plot if we have values
                    plt.plot(years[:len(values)], values, marker='o', 
                            label=f"{model_name}: {var}", linewidth=2)
            
            if plt.gca().get_lines():  # Check if any lines were plotted
                plt.title("Emissions Time Series")
                plt.xlabel("Year")
                plt.ylabel(f"Value ({df['unit'].iloc[0]})" if 'unit' in df else "Value")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # Save plot
                os.makedirs("plots", exist_ok=True)
                filename = f"plots/timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                return f"""## Time Series Plot Generated

I've created a plot showing emissions over time across different models. 
You can find it saved as: `{filename}`

The plot shows:
- Data from {min(year_cols)} to {max(year_cols)}
- {len(df['modelName'].unique())} different models
- {len(emissions_vars)} emission-related variables

Want to:
- Focus on a specific model? Try `plot emissions for [model name]`
- Look at different variables? Try `list variables`
- Compare specific years? Include a year in your query

_Data source: [IAM PARIS Results](https://iamparis.eu/results)_"""
            else:
                return """## No Plottable Data Found

While I found some data, I couldn't create a meaningful plot. This might be because:
- The values are missing or invalid
- The time series is incomplete
- The data format isn't suitable for plotting

Try:
1. Using a different variable
2. Specifying a particular model
3. Checking the raw data with `list variables`"""
                
        except Exception as e:
            logging.error(f"Error creating plot: {str(e)}")
            return f"""## Error Creating Plot

I encountered an error: `{str(e)}`

Troubleshooting steps:
1. Check if the variable exists using `list variables`
2. Try specifying a model with `plot [model_name] emissions`
3. Make sure the data contains time series information

_If the problem persists, there might be an issue with the data source._"""
    
    # List models (conversational)
    if re.search(r"\bmodels?\b.*\b(available|list)\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r.get('modelName')})
        if not models:
            return "I couldn't find any models in the data right now. Try `help` or refresh the data."
        # Build a natural sentence rather than a dry bullet list
        if len(models) <= 6:
            model_str = ", ".join(models[:-1]) + (" and " + models[-1] if len(models) > 1 else models[0])
            return f"I found these models in the IAM PARIS dataset: {model_str}. Which one would you like to know more about?"
        # For many models, give a short hint and invite follow-up
        return (f"There are {len(models)} models available. "
                "You can ask for details about a specific model using `info [model name]`, "
                "or say `list variables` to see the kinds of outputs available.")
    
    # List variables (conversational)
    if 'list variables' in q:
        vars = sorted({r.get('variable', '') for r in ts_data if r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."
        # Present a friendly sample and hint for full list
        sample = vars[:8]
        more = "" if len(vars) <= 8 else f" and {len(vars)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return (f"I can work with variables like {sample_str}{more}. "
                "If you want the complete list, say `list variables` again or specify a variable to plot (e.g. `plot CO2 emissions`).")
    
    # List scenarios (conversational)
    if re.search(r"\bscenarios?\b.*\b(available|included|list)\b", q):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r.get('scenario')})
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."
        return ("I see several scenarios in the data. If you tell me which one interests you I can compare results or plot variables "
                "for that scenario. Try `list scenarios` to get the exact names.")
    
    # Model info or general models query — now conversational and data-driven
    if any(w in q for w in ('info', 'details', 'describe', 'about', 'tell me about')):
        return conversational_models_overview(model_data, ts_data, q)
    
    # Help command
    if 'help' in q:
        return ("Tell me what you want to do and I'll help. Examples:\n"
                "- Ask about models: `list models` or `info GCAM`\n"
                "- Explore variables: `list variables` or `plot CO2 emissions`\n"
                "- Visualize results: `plot emissions for GCAM`\n"
                "If you want more conversational guidance, just say 'suggest' or ask a question in plain language.")
    
    return ""

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
        """Fetch JSON with caching"""
        cache_file = f"cache/{url.split('/')[-1]}_{hash(str(params) + str(payload))}.json"
        os.makedirs("cache", exist_ok=True)
        
        if cache and os.path.exists(cache_file):
            self.logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'r') as f:
                return pd.read_json(f).to_dict('records')
                
        self.logger.info(f"Fetching data from {url}")
        resp = requests.post(url, json=payload) if payload else requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        
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
        ts = self.fetch_json(
            self.env["REST_API_FULL"],
            payload={
                "study": ["paris-reinforce"],
                "workspace_code": ["eu-headed"]
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

                # Handle QA chain response
                if self.streaming:
                    print("\nBOT: ", end="", flush=True)
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                    print()
                else:
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                ans = resp.get("answer", "").strip()

                # Store in history without the "You:" part
                if ans:
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

def main():
    parser = argparse.ArgumentParser(description="IAM PARIS Climate Policy Assistant")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (shows all logs)")
    args = parser.parse_args()
    
    # Set up logging based on debug flag
    setup_logging(args.debug)
    
    bot = IAMParisBot(streaming=not args.no_stream)
    bot.run()

if __name__ == "__main__":
    main()