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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
    
    # List models
    if re.search(r"\bmodels?\b.*\b(available|list)\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r.get('modelName')})
        response = """## Available Climate Models

I found the following models in the IAM PARIS database:

"""
        for model in models:
            response += f"- {model}\n"
        
        response += "\nWould you like to:\n"
        response += "- Get detailed information about a specific model? (use `info [model name]`)\n"
        response += "- See what variables these models can analyze? (use `list variables`)\n"
        response += "- Learn about specific scenarios? (use `list scenarios`)\n\n"
        response += "_Data source: [IAM PARIS Models](https://iamparis.eu/models)_"
        return response
    
    # List variables
    if 'list variables' in q:
        vars = sorted({r.get('variable', '') for r in ts_data if r.get('variable')})
        response = """## Available Variables

Here are all the variables I can find in the current dataset:

"""
        for var in vars:
            response += f"- {var}\n"
        
        response += "\nYou can:\n"
        response += "- Plot any variable over time using `plot [variable] [year]`\n"
        response += "- Compare variables across different models\n"
        response += "- Ask questions about specific trends\n\n"
        response += "_Data source: [IAM PARIS Results](https://iamparis.eu/results)_"
        return response
    
    # List scenarios
    if re.search(r"\bscenarios?\b.*\b(available|included|list)\b", q):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r.get('scenario')})
        response = """## Available Scenarios

Let me show you the scenarios included in our database:

"""
        for scenario in scenarios:
            scenario_name = scenario.replace("_", " ").title()
            response += f"- {scenario_name}\n"
        
        response += "\nEach scenario represents different policy and technology pathways. "
        response += "Would you like to:\n"
        response += "- Learn more about a specific scenario?\n"
        response += "- Compare results across scenarios?\n"
        response += "- See how variables change in different scenarios?\n\n"
        response += "_Data source: [IAM PARIS Scenarios](https://iamparis.eu/results)_"
        return response
    
    # Model info or general models query
    if any(w in q for w in ('info', 'details', 'describe', 'about', 'tell me about')):
        # If asking about specific model
        for model in model_data:
            name = model.get('modelName', '').lower()
            if name in q:
                desc = model.get('description', 'No description available.')
                response = f"""## {model['modelName']}

{desc}

[View detailed documentation](https://iamparis.eu/detailed_model_doc)

### Available Data
- Variables: {len({r.get('variable') for r in ts_data if r.get('modelName') == model['modelName']})} variables
- Time periods: {len({c for c in ts_data[0].keys() if c.isdigit()})} years

_Data source: [IAM PARIS Models](https://iamparis.eu/models)_"""
                return response

        # If asking about models in general
        response = """## IAM PARIS Models Overview

The platform includes several integrated assessment models that analyze climate policy impacts:

### Key Models

- **GCAM (Global Change Assessment Model)**
  - Global integrated assessment model representing human and Earth system dynamics
  - Explores interactions between energy systems, agriculture, land use, economy and climate
  - Provides comprehensive analysis of climate policy impacts

- **PROMETHEUS**
  - Focuses on energy system dynamics and technological change
  - Features stochastic analysis capabilities
  - Specialized in energy market projections

- **TIMES-GEO**
  - Detailed technology-rich model
  - Analyzes energy system transformations
  - Provides regional and sectoral insights

[View detailed model documentation](https://iamparis.eu/detailed_model_doc)

### Data Availability
- Multiple scenarios and policy pathways
- Time series from present to 2100
- Regional and sectoral breakdowns

_Data source: [IAM PARIS Models](https://iamparis.eu/models)_"""
        return response
    
    # Help command
    if 'help' in q:
        return """## How Can I Help You?

Here's what you can ask me about:

### Explore Data 📊
- `list models` - See all available climate models
- `list variables` - Browse available data variables
- `list scenarios` - Explore different policy scenarios
- `list studies` - Check out research studies

### Visualize Results 📈
- `plot [model/variable] [year]` - Create custom visualizations
- `info [model name]` - Learn about specific models

### General Questions 💡
You can also ask me about:
- Climate policies and their impacts
- Model comparisons and assumptions
- Specific variables and their trends
- Technical details about the models

_Just type your question or command, and I'll help you explore the IAM PARIS data!_"""
    
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
        # Updated memory initialization
        memory = ConversationBufferMemory(
            return_messages=True,
            input_key="question",
            output_key="answer",
            memory_key="chat_history"  # Add this line
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
                query = input("You: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if not query:
                    continue
                    
                # Try data query first
                ans = data_query(query, models, ts)
                if ans:
                    print(f"\nBot: {ans}")
                    self.history.append((query, ans))
                    continue
                
                # Fall back to QA chain with streaming
                if self.streaming:
                    print("\nBot: ", end="", flush=True)
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                    ans = resp.get("answer", "").strip()
                    if not ans:
                        print("I cannot answer that question.")
                    # Answer is already streamed via callback
                else:
                    # Non-streaming mode - only print the answer
                    resp = chain.invoke({
                        "question": query,
                        "chat_history": self.history
                    })
                    ans = resp.get("answer", "").strip()
                    if not ans:
                        ans = "I cannot answer that question."
                    print(f"\nBot: {ans}")
            
                self.history.append((query, ans))
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                self.logger.error(f"Error processing query: {str(e)}")
                print("\nSorry, I encountered an error. Please try again.")

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