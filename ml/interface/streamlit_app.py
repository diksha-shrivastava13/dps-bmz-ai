import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse

# Environment Variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
az_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
az_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# models
embed_model = OpenAIEmbedding(embed_batch_size=10)
llm = AzureOpenAI(engine="gpt-4o", model="gpt-4o", temperature=0.0,
                  api_key=az_openai_api_key, azure_endpoint=az_openai_endpoint)

# Configurations
Settings.embed_model = embed_model
Settings.llm = llm


# File Processing Functions
def create_index(file_path):
    loader = SimpleDirectoryReader(input_files=[file_path], required_exts=[".pdf"])
    docs = loader.load_data()
    index = VectorStoreIndex.from_documents(docs)

    return index


def overview_report(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Generating Overview...</p>""", unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    program_report_prompt = (
        "Context Information is below.\n"
        "----------------------------\n"
        "{context_str}\n"
        "----------------------------\n"
        "You're a very helpful data scientist. You create comprehensive and detailed long reports"
        "from large multi-page documents such that there's no need to read the original document and"
        "your generated report covers all the necessary data, quantitative and qualitative to assist"
        "with decision making. You should not miss any important information, you should mention all"
        "the numbers clearly, along with this you should output a very detailed history of the document,"
        "and everything it entails. You should cover all the sections of the documents. Make the report as"
        "large and as detailed as possible. Pay special attention to risks, indicators and on generating a"
        "comprehensive report out of all wirkungsmatrix and appendix tables available in the report."
        "Respond in German. Don't mention the file name (temp_file.pdf) itself in your report."
    )
    qa_prompt_template = PromptTemplate(program_report_prompt)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_template}
    )
    overview = query_engine.query(
        "You're a very helpful data scientist. You create comprehensive and detailed long reports"
        "from large multi-page documents such that there's no need to read the original document and"
        "your generated report covers all the necessary data, quantitative and qualitative to assist"
        "with decision making. You should not miss any important information, you should mention all"
        "the numbers clearly, along with this you should output a very detailed history of the document,"
        "and everything it entails. You should cover all the sections of the documents. Respond in German.")
    with ov_tab:
        st.markdown(overview)
    return overview


def key_value_pairs(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Extracting KURZBESCHREIBUNG...</p>""",
                                  unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    key_value_data = query_engine.query(
        "Extract all the key-value pairs from the first 10 pages of "
        "the document. Make use of the KURZBESCHREIBUNG table data in order to display all the key-value pairs"
        "about the report in the table. This should necessarily include the title of the report, country, "
        "theme, year, duration, budget and all other information present in the KURZBESCHREIBUNG table."
        "Give the answer in the markdown format. Do NOT enclose this data within ``` ``` markers. If there's"
        "any missing field, you are allowed to use the header of the file and its content to fill it."
        "Respond in German.")
    with kv_tab:
        st.markdown(key_value_data)
    return key_value_data


def risk_analysis(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Analysing Risks...</p>""", unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    risk_data = query_engine.query(
        "Determine whether the mission/program/project is at risk or not. Determine the level of risk associated"
        "with it. Give a comprehensive report on risk analysis of the document, necessarily listing all the reasons,"
        "any recommendations that are provided for these risks in the documents and call to actions. Pay particular "
        "attention on the key indicators mentioned in the report and if there's any risk or anomaly there."
        "Respond in German."
    )
    with rg_tab:
        st.markdown(risk_data)
    return risk_data


def wirkungsmatrix(file_path):
    processing_container.markdown("""<p style="color: #3ae2a5;">Extracting Wirkungsmatrix...</p>""",
                                  unsafe_allow_html=True)
    parser = LlamaParse(result_type="markdown", num_workers=8)
    documents = parser.load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    processing_container.markdown("""<p style="color: #3ae2a5;">Gathering Insights from Wirkungsmatrix...</p>""",
                                  unsafe_allow_html=True)

    wirkungsdata1 = query_engine.query(
        "Detail all the data in Wirkungsmatrix des Moduls, including Ziele, Indikatoren, Quellen, Annahmen, "
        "Modulzielindikator, Programmzielindikator, Output, Outputindikator. Respond in German."
    )
    wirkungsdata2 = query_engine.query(
        "Detail all the data in Wirkungsmatrix des"
        "Moduls including Outputs, Wesentliche Aktivitäten zu Outputs, Inputs / Geplante Instrumente, Annahmen."
        "Respond in German."
    )
    wirkungsdata3 = query_engine.query(
        "Detail all the information in Wirkungslogik für ein Modul, including the connections and flows between "
        "Programmziel und Zeithorizont, Programmzielindikator, Modulziel und Zeithorizont, Modulzielindikator,"
        "Output and which of these factors are dependent on or affect each other. Respond in German."
    )
    processing_container.markdown("""<p style="color: #3ae2a5;">Halfway there...</p>""",
                                  unsafe_allow_html=True)
    wirkungsdata4 = query_engine.query(
        "Detail all the information from Berichterstattung über die Kostenentwicklung in EUR, including "
        "Kostenzeile GIZ-Schema, Kostenschätzung laut Angebot (Planwert), Ist-Kosten kumuliert bis Berichtsstichtag, "
        "Bis zum Ende der Laufzeit verblei- bende Mittel, Erläuterung bei vorhersehbaren signifikanten Abweichungen "
        "von der Kostenschätzung*. Respond in German."
    )
    wirkungsdata5 = query_engine.query(
        "Detail all the information from Ist-Kosten und angepasste Prognose pro Output bilat./reg. Vorhaben. in German"
    )
    wirkungsdata6 = query_engine.query(
        "Detail all the information from Karte mit Kennzeichnung der projektregion in english. If there's a map"
        "involved, list all the countries the map is describing. Respond in German."
    )
    wirkungsdata = [wirkungsdata1, wirkungsdata2, wirkungsdata3, wirkungsdata4, wirkungsdata5, wirkungsdata6]
    with w_tab:
        st.markdown(wirkungsdata[0])
        st.markdown(wirkungsdata[1])
        st.markdown(wirkungsdata[2])
        st.markdown(wirkungsdata[3])
        st.markdown(wirkungsdata[4])
        st.markdown(wirkungsdata[5])
    return wirkungsdata


def next_steps(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Recommending Actions...</p>""",
                                  unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    next_steps_data = query_engine.query(
        "List all the mentioned calls to actions and recommended steps mentioned in the document, arrange them"
        "according to highest risk and urgency, top to bottom. Apart from the recommended steps in the documents,"
        "you should also include the assumptions and risks and the actions you think should be taken in order to "
        "make the program a success. Respond in German."
    )
    with ac_tab:
        st.markdown(next_steps_data)
    return next_steps_data


def recommended_fields_generation(index):
    processing_container.markdown("""<p style="color: #3ae2a5;">Almost done...</p>""", unsafe_allow_html=True)
    query_engine = index.as_query_engine()
    recommended_fields_data = query_engine.query(
        "Now that you have a good look at the document, as a data analyst who needs to provide enough insight and data"
        "to your manager so they can make the correct data-driven decisions on all factors, what information and fields"
        " would be the most useful to you? Make a list of all such fields and give your reasons on why it would be"
        "useful in order for you to assist in making data driven decisions and always have insights into how the"
        "program is processing and drive it to success. Respond in German."
    )
    with f_tab:
        with col1:
            st.markdown(recommended_fields_data)
    return recommended_fields_data


def user_query_answer(index, user_query):
    query_engine = index.as_query_engine()
    query_answer = query_engine.query(user_query + " Make the answer as detailed and as comprehensive as required."
                                      + " Make sure to use the documents as context to answer the question."
                                      + " If you cannot find an answer from the documents, tell the user to go through"
                                        "the original document they uploaded. Respond in German.")
    return query_answer


def display_information_once():
    file_path = "temp_file.pdf"
    if uploaded_file is not None:
        # time.sleep(5)
        processing_container.markdown("""<p style="color: #3ae2a5;">Creating Index...</p>""", unsafe_allow_html=True)
        query_index_ = create_index(file_path)

        # data
        overview_report(query_index_)
        key_value_pairs(query_index_)
        risk_analysis(query_index_)
        wirkungsmatrix(file_path)
        next_steps(query_index_)
        recommended_fields_generation(query_index_)

        processing_container.markdown("""<p style="color: #3ae2a5;">Processing complete!</p>""", unsafe_allow_html=True)


st.set_page_config(
    page_title="PortFolio Navigator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.markdown("<h1 style='text-align: center;'>📈 Portfolio Navigator </h1>", unsafe_allow_html=True)
st.markdown(" ")
st.markdown(
    "<p style='text-align: center;'>Upload your Document to Portfolio Navigator and get overview, insights, "
    "history and analysis.", unsafe_allow_html=True)

# Upload functionality
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
processing_container = st.empty()
processing_container.text(" ")

# Tab Overview
ov_tab, kv_tab, rg_tab, w_tab, ac_tab, f_tab = st.tabs(
    ["Overview", "General Information", "Risk Analysis", "Wirkungsmatrix", "Recommended Actions", "Suggestions"]
)

# Search Functionality within Suggestion Tab
with f_tab:
    with st.container():
        col1, col2 = st.columns(2)
        with col2:
            # Capture the input from the text input widget
            input_search_query = st.text_input("Search:", value=st.session_state.get('search_query', ''),
                                               key="search_query")

if uploaded_file is not None:
    temp_filepath = Path("temp_file.pdf")
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.read())
    processing_container.text("File uploaded...")
    time.sleep(2)
    processing_container.text("Processing...")
    display_information_once()
    # uploaded_file = None
else:
    st.markdown("""<p style="color: #3ae2a5;">Please upload a PDF file.</p>""", unsafe_allow_html=True)

# Conditionally update the session state if the input has changed
if 'search_query' not in st.session_state or input_search_query != st.session_state['search_query']:
    uploaded_file = None
    st.session_state['search_query'] = input_search_query

# Perform the search operation using the search term from session state
if st.session_state['search_query']:
    with col2:
        answer_processing_container = st.empty()
        answer_processing_container.markdown("""<p style="color: #3ae2a5;">Processing request...</p>""",
                                             unsafe_allow_html=True)
        filepath = "temp_file.pdf"
        query_index = create_index(filepath)
        answer = user_query_answer(query_index, st.session_state['search_query'])
        if answer:
            answer_processing_container.markdown("""<p style="color: #3ae2a5;">Found results!</p>""",
                                                 unsafe_allow_html=True)
            st.markdown(answer)
        else:
            st.markdown("No answer found for your query.")
