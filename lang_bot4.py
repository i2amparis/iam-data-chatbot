import os
import re
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Tuple

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


def load_env() -> None:
    """
    Load environment variables and validate required keys.
    """
    load_dotenv(override=True)
    required = ["OPENAI_API_KEY", "REST_MODELS_URL", "REST_API_FULL"]
    env = {var: os.getenv(var) for var in required}
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    globals().update(env)


def fetch_json(url: str, params=None, payload=None) -> list:
    """
    GET or POST JSON data and return 'data'.
    """
    resp = requests.post(url, json=payload) if payload is not None else requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("data", [])


def flatten_years(ts_data: list) -> list:
    flat = []
    for rec in ts_data:
        d = rec.copy()
        years = d.pop("years", {})
        items = years.items() if isinstance(years, dict) else []
        if not isinstance(years, dict):
            for elt in years:
                if isinstance(elt, dict):
                    items.extend(elt.items())
        for year, val in items:
            if year.isdigit(): d[year] = val
        flat.append(d)
    return flat


def docs_from_records(records: list) -> list:
    docs = []
    for rec in records:
        desc = (rec.get("description") or rec.get("modelName") or "").strip()
        asum = (rec.get("assumptions") or "").strip()
        if not desc and not asum: continue
        content = desc + (f"\n\nAssumptions: {asum}" if asum else "")
        meta = {k: v for k, v in rec.items() if k not in ("description","assumptions")}
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def build_faiss_index(docs: list, embeddings) -> FAISS:
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(index_dir)
    return store


def create_qa_chain(vs: FAISS, streaming: bool) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(return_messages=True, input_key="question", output_key="answer")
    # System prompt revised to incorporate manager guidance. The assistant should base its answers on
    # the IAM PARIS platform whenever possible. If data is missing, it should say so and only then
    # rely on general knowledge, clearly noting that it comes from outside IAM PARIS. It should also
    # ask clarifying questions when the query is ambiguous and suggest follow‑up questions.
    system_tpl = (
        "You are an expert climate‑policy assistant using the IAM PARIS database. "
        "Base your answers on the IAM PARIS platform. "
        "If requested information isn’t available in the platform, clearly state that and then rely on general knowledge, "
        "noting that it comes from outside IAM PARIS. "
        "Ask clarifying questions if the user’s query is ambiguous, such as requesting a specific study, scenario, variable, or region. "
        "Suggest next questions after each answer.\nContext: ```{context}```"
    )
    user_tpl = "Question: ```{question}```"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_tpl),
        HumanMessagePromptTemplate.from_template(user_tpl)
    ])
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, streaming=streaming,
                      callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
                      api_key=os.getenv("OPENAI_API_KEY"))
    retr = vs.as_retriever(search_type="similarity", search_kwargs={"k":5})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retr,
                                                 memory=memory, chain_type="stuff",
                                                 combine_docs_chain_kwargs={"prompt":prompt},
                                                 verbose=False)


def plot_time_series(records: list, year: int=None, model: str=None, variable: str=None) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        print("No data to plot."); return
    years = sorted([c for c in df.columns if c.isdigit()], key=int)
    if not years:
        print("No year data."); return
    dfc = df.copy()
    if model:
        dfc = dfc[dfc['modelName'].str.lower() == model.lower()]
    if variable:
        dfc = dfc[dfc['variable'].str.contains(variable, case=False, na=False)]
    plt.figure()
    unit = dfc['unit'].iloc[0] if 'unit' in dfc.columns and not dfc.empty else ''
    if year and str(year) in years:
        vals = dfc.groupby('modelName')[str(year)].mean()
        vals.plot(kind='bar')
        plt.title(f"Values in {year} for {'model ' + model if model else 'all models'}")
        plt.ylabel(unit)
    else:
        for name, grp in dfc.groupby('modelName'):
            pts = [grp[str(y)].mean() for y in years]
            plt.plot(years, pts, label=name)
        plt.title(f"Time Series for {'model ' + model if model else 'all models'}")
        plt.ylabel(unit)
        plt.legend()
    plt.xlabel("Year")
    plt.tight_layout()
    plt.show()


def data_query(question: str, model_data: list, ts_data: list) -> str:
    q = question.lower()
    if 'help' in q:
        return "Ask models, variables, years or plots: 'list models', 'plot for 2050', 'plot gcam', 'plot all models', etc."

    # list available models
    if re.search(r"\bmodels?\b.*\b(available|list)\b", q):
        models = sorted({r['modelName'] for r in model_data})
        return "Models: " + ", ".join(models)

    # list variables
    if 'list variables' in q:
        return "Variables: " + ", ".join(sorted({r['variable'] for r in ts_data if 'variable' in r}))
    # list years
    if 'list years' in q:
        yrs = sorted({k for r in ts_data for k in r if k.isdigit()}); return "Years: " + ", ".join(yrs)

    # list available scenarios
    # Capture queries asking which scenarios are available or to list scenarios. Pull the field
    # `scenario` or `scenarioName` from the time‑series data. If none exist, direct the user
    # to the IAM PARIS results page to choose a study.
    if re.search(r"\bscenarios?\b.*\b(available|list)\b", q):
        scenarios = sorted({
            (r.get('scenario') or r.get('scenarioName'))
            for r in ts_data
            if r.get('scenario') or r.get('scenarioName')
        })
        # Filter out empty values
        scenarios = [s for s in scenarios if s]
        if scenarios:
            return "Scenarios: " + ", ".join(scenarios)
        else:
            return (
                "Scenario information isn’t provided in the current dataset. "
                "Please choose a modelling study from https://iamparis.eu/results to explore its specific scenarios."
            )
    # model info
    if any(w in q for w in ('info','details','describe')):
        for r in model_data:
            if r.get('modelName','').lower() in q:
                return (f"Model: {r['modelName']}\n"
                        f"Description: {r.get('description','N/A')}\n"
                        f"Institution: {r.get('institute','N/A')}\n"
                        f"Type: {r.get('model_type','N/A')}")

    # extract year if present
    m = re.search(r"(20\d{2})", q)
    yr = int(m.group(1)) if m else None

    # handle plot requests
    if 'plot' in q or 'chart' in q or 'graph' in q:
        # determine selected model filter
        models_available = sorted({r['modelName'] for r in ts_data if 'modelName' in r})
        selected = next((m for m in models_available if m.lower() in q), None)
        if yr and selected:
            plot_time_series(ts_data, year=yr, model=selected)
        elif selected:
            plot_time_series(ts_data, model=selected)
        elif yr:
            plot_time_series(ts_data, year=yr)
        else:
            plot_time_series(ts_data)
        return "Displayed plot."

    # handle year data listing
    if yr:
        df = pd.DataFrame(ts_data)
        if str(yr) not in df.columns:
            return f"No data found for {yr}."
        dfy = df[['modelName', str(yr), 'unit']].dropna(subset=[str(yr)])
        summ = dfy.groupby(['modelName', 'unit'], as_index=False)[str(yr)].mean()
        lines = [f"{row['modelName']}: {row[str(yr)]:.2f} {row['unit']}" for _, row in summ.iterrows()]
        return f"Data for {yr}:\n" + "\n".join(lines)

    # general inquiry about available data
    if re.search(r"\bwhat\b.*\bkind of data\b.*\bavailable\b", q):
        # provide a high level overview of variable categories and suggest studies
        variables = sorted({r.get('variable','') for r in ts_data if r.get('variable')})
        # group some variables into categories for illustration
        examples = []
        # emissions examples
        emissions_vars = [v for v in variables if any(term in v.lower() for term in ['emission','co2','ghg'])]
        if emissions_vars:
            examples.append(f"Emissions data such as {emissions_vars[0]}")
        # energy examples
        energy_vars = [v for v in variables if any(term in v.lower() for term in ['energy','electricity','power'])]
        if energy_vars:
            examples.append(f"Energy data such as {energy_vars[0]}")
        # socioeconomic examples
        socio_vars = [v for v in variables if any(term in v.lower() for term in ['gdp','population','income'])]
        if socio_vars:
            examples.append(f"Socioeconomic data such as {socio_vars[0]}")
        example_txt = "; ".join(examples) if examples else ", ".join(variables[:3])
        return (
            "The IAM PARIS platform contains a variety of data. "
            "For example, it includes emissions variables (e.g. CO₂ emissions), energy variables (e.g. primary and secondary energy generation), "
            "and socioeconomic variables (e.g. GDP). These variables cover multiple models, scenarios and regions. "
            f"Here are a few example variables: {example_txt}. "
            "You can explore detailed modelling studies and results at https://iamparis.eu/results."
        )

    # generic future emissions question – ask for a specific study or scenario from IAM PARIS
    if 'emissions' in q and 'future' in q and 'post-glasgow' not in q:
        return (
            "Emissions trajectories vary by modelling study, model, scenario and region. "
            "Please specify a study or scenario from the IAM PARIS results page (https://iamparis.eu/results) to explore its data."
        )

    # post-glasgow context – prompt for variable and region after study selection
    if 'post-glasgow' in q:
        return (
            "In the context of the post‑Glasgow targets, emissions trajectories vary by study and region. "
            "Please select a specific study and indicate the variable (e.g. CO₂ emissions) and region of interest. "
            "After selecting, I can provide statistics and plot a timeseries for the chosen variable."
        )

    # differences between GCAM and GEMINI‑E3
    if 'difference' in q and ('gcam' in q and 'gemini' in q):
        return (
            "GCAM (Global Change Assessment Model) is a technology‑rich integrated assessment model that represents the interactions of energy, agriculture, land use and climate systems. "
            "It simulates the substitution of low‑carbon for high‑carbon technologies based on costs and policy constraints, providing a long‑term global perspective on mitigation pathways. "
            "GEMINI‑E3, by contrast, is a multi‑country, multi‑sector computable general equilibrium model that represents economic markets under perfect competition. "
            "It focuses on short‑ to medium‑term economic impacts of policies such as carbon taxes and trade measures."
        )

    # request for more information about models
    if re.search(r"\btell me something more about (these )?models\b", q) or 'can you tell me more about' in q:
        return (
            "The IAM PARIS platform provides detailed documentation for each model. "
            "For example, GCAM is a global integrated assessment model combining human and Earth systems, while GEMINI‑E3 is a recursive general equilibrium model. "
            "You can explore overviews, key features, geographic coverage and other details for each model at https://iamparis.eu/detailed_model_doc."
        )

    return ""


def main():
    load_env()
    p = argparse.ArgumentParser("Climate Data Chatbot")
    p.add_argument("--no-stream",action="store_true",help="Disable streaming")
    args = p.parse_args()
    print("Loading data...")
    models = fetch_json(os.getenv("REST_MODELS_URL"),params={"limit":-1})
    ts = flatten_years(fetch_json(os.getenv("REST_API_FULL"),payload={"study":["paris-reinforce"],"workspace_code":["eu-headed"]}))
    docs = docs_from_records(models + ts)
    chunks = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=80).split_documents(docs)
    emb = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.getenv("OPENAI_API_KEY"))
    vs = build_faiss_index(chunks,emb)
    chain = create_qa_chain(vs, streaming=not args.no_stream)
    print("Ready! Type your question or 'exit'.")
    history: List[Tuple[str,str]] = []
    while True:
        u = input("You: ").strip()
        if u.lower() in ('exit','quit'): break
        if not u: continue
        ans = data_query(u, models, ts)
        if ans:
            print("Bot:", ans)
        else:
            print("Bot: ", end="", flush=True)
            r = chain.invoke({"question": u, "chat_history": history})
            out = r.get("answer") or "No answer."
            print(out)
            ans = out
        history.append((u, ans))

if __name__ == '__main__':
    main()
