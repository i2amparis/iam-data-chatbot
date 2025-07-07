import os
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ─── 1) Load & validate environment ────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
REST_MODELS_URL = os.getenv("REST_MODELS_URL")
REST_API_URL    = os.getenv("REST_API_FULL")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not REST_MODELS_URL or not REST_API_URL:
    raise RuntimeError("Missing REST_MODELS_URL or REST_API_FULL in environment")

# ─── 2) Fetch metadata & time series via REST ──────────────────────────────────
def fetch_model_metadata():
    resp = requests.get(REST_MODELS_URL, params={"limit": -1})
    resp.raise_for_status()
    return resp.json().get("data", [])

def fetch_time_series(filters):
    resp = requests.post(REST_API_URL, json=filters)
    resp.raise_for_status()
    return resp.json().get("data", [])

# ─── 3) Prepare Documents & FAISS ───────────────────────────────────────────────
def records_to_documents(records):
    docs = []
    for rec in records:
        desc = (rec.get("description") or rec.get("modelName") or "").strip()
        asum = (rec.get("assumptions") or "").strip()
        if not desc and not asum:
            continue
        text = desc + (f"\n\nAssumptions: {asum}" if asum else "")
        meta = {k: v for k, v in rec.items() if k not in ("description", "assumptions")}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(index_dir)
    return vs

# ─── 4) Set up LLM Retrieval Chain (with optional streaming) ────────────────────
def create_chat_chain(vs, streaming: bool = False):
    # 1) Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 2) Prompt templates
    system_template = r"""
You are an expert climate policy assistant.
- Always mention uncertainties if asked for model forecasts.
- Prioritize peer-reviewed data when available.
- If a user's question is ambiguous, ask for clarification.
Use the following pieces of context to answer the user's question.
If you don’t see something in the context, feel free to use your own knowledge to fill in gaps.
---------------
Context: ```{context}```
"""
    user_template = "Question: ```{question}```"

    # Build the combined ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ])

    # 3) Streaming callback (new API)
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else None

    # 4) Instantiate the LLM with streaming & callbacks
    llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        temperature=0,
        streaming=streaming,
        callbacks=callbacks,           # ← use `callbacks=…`
        api_key=OPENAI_API_KEY,
    )

    # 5) Retriever
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 6) Build the chain, passing in our `prompt`
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
    )

# ─── 5) Plotting helper ─────────────────────────────────────────────────────────
def plot_time_series(records, year=None):
    df = pd.DataFrame(records)
    if df.empty:
        print("No data available to plot.")
        return

    if "years" in df.columns:
        yrs = pd.json_normalize(df.pop("years"))
        df = pd.concat([df, yrs], axis=1)

    year_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    if not year_cols:
        print("No year columns found in data.")
        return

    plt.figure()
    if year and str(year) in year_cols:
        pivot = df.groupby("modelName")[str(year)].mean()
        pivot.plot(kind="bar")
        plt.title(f"Values in {year}")
        plt.ylabel(df.get("unit", [""])[0])
    else:
        for model, group in df.groupby("modelName"):
            vals = [group[str(y)].mean() for y in year_cols]
            plt.plot(year_cols, vals, label=model)
        plt.title("Time Series")
        plt.ylabel(df.get("unit", [""])[0])
        plt.legend()
    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()

def main():
    # Fetch & combine records
    model_data = fetch_model_metadata()
    ts_data    = fetch_time_series({
        "study": ["paris-reinforce"],
        "workspace_code": ["eu-headed"],
    })
    combined = model_data + ts_data
    for rec in combined:
        rec.setdefault("description", rec.get("modelName", ""))
        rec.setdefault("assumptions", "")

    # Build documents & FAISS index
    docs     = records_to_documents(combined)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(docs)
    vs       = build_vectorstore(chunks)

    # Create chain with streaming enabled
    chain = create_chat_chain(vs, streaming=True)

    def chat(q: str) -> str:
        resp = chain.invoke({"question": q})
        return resp.get("answer", resp)

    print("Welcome to the Climate and Energy Dataset Explorer powered by Holistic SA! (type 'exit' or 'quit' to close)\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Bot: Goodbye!")
            break

        # Handle plotting requests
        if any(tok in user.lower() for tok in ("plot", "chart", "graph")):
            match = re.search(r"\b(20\d{2})\b", user)
            year = int(match.group(1)) if match else None
            plot_time_series(ts_data, year)
            continue

        # Otherwise stream the answer
        print("Bot:", end=" ", flush=True)
        chat(user)   # tokens will stream via StreamingStdOutCallbackHandler
        print()      # newline after streaming completes

if __name__ == "__main__":
    main()
