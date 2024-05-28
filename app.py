import streamlit as st
import requests
import vertexai
from vertexai.generative_models import Content, FunctionDeclaration, GenerativeModel, Part, Tool
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import os

# Load environment variables
load_dotenv()

# Load secrets from secrets.toml
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
SCOPUS_API_KEY = st.secrets["SCOPUS_API_KEY"]
google_credentials = st.secrets["google"]["credentials"]

credentials_dict = {
    "type": google_credentials["type"],
    "project_id": google_credentials["project_id"],
    "private_key_id": google_credentials["private_key_id"],
    "private_key": google_credentials["private_key"].replace('\\n', '\n'),
    "client_email": google_credentials["client_email"],
    "client_id": google_credentials["client_id"],
    "auth_uri": google_credentials["auth_uri"],
    "token_uri": google_credentials["token_uri"],
    "auth_provider_x509_cert_url": google_credentials["auth_provider_x509_cert_url"],
    "client_x509_cert_url": google_credentials["client_x509_cert_url"],
    "universe_domain": google_credentials["universe_domain"]
}

# Authenticate and connect to Google Sheets
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict)

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

# Initialize Vertex AI
project_id = google_credentials["project_id"]
vertexai.init(project=project_id, location="us-central1")

# Set up Google Gemini-Pro AI model
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Scopus AI Chatbot",
    page_icon="https://seeklogo.com/images/G/google-ai-logo-996E85F6FD-seeklogo.com.png",
    layout="centered",
)

# Add title
st.title("🤖 Scopus AI Chatbot")

# Language selection
langcols = st.columns([0.2, 0.8])
with langcols[0]:
  lang = st.selectbox('Select your language',
  ('English', 'Español', 'Français', 'Deutsch',
  'Italiano', 'Português', 'Polski', 'Nederlands',
  'Русский', '日本語', '한국어', '中文', 'العربية',
  'हिन्दी', 'Türkçe', 'Tiếng Việt', 'Bahasa Indonesia',
  'ภาษาไทย', 'Română', 'Ελληνικά', 'Magyar', 'Čeština',
  'Svenska', 'Norsk', 'Suomi', 'Dansk', 'हिन्दी', 'हिन्�'), index=0)

if 'lang' not in st.session_state:
    st.session_state.lang = lang

# Define FunctionDeclaration for Scopus search
search_scopus_func = FunctionDeclaration(
    name="search_scopus",
    description="Search for open access articles on Scopus",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for finding articles on Scopus"
            }
        },
        "required": ["query"]
    }
)

# Create a Tool with the FunctionDeclaration
scopus_tool = Tool(
    function_declarations=[search_scopus_func],
)

# Load SentenceTransformer model
@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-mpnet-base-v2')

sentence_model = load_sentence_transformer_model()

# Function to search Scopus for open access articles with metadata including abstract
def search_scopus(query):
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        "X-ELS-APIKey": SCOPUS_API_KEY,
    }
    params = {
        "query": f"{query} AND openaccess(1)",
        "count": 10,  # Retrieve more results for better vector search
        "field": "dc:title,dc:creator,prism:coverDate,prism:doi,dc:description"
    }
    response = requests.get(url, headers=headers, params=params)
    results = response.json()
    return results.get('search-results', {}).get('entry', [])

# Function to append message to chat session
def append_message(role, content):
    st.session_state.chat_session.append({"role": role, "content": content})

# Load text model
@st.cache_resource
def load_model(model_name):
    model = GenerativeModel(model_name)
    return model

model = load_model("gemini-1.5-pro-001")

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = []

# Welcome message
if 'welcome' not in st.session_state or lang != st.session_state.lang:
    st.session_state.lang = lang
    welcome_prompt = f"""
    Give a welcome message to the user and suggest what they can do (e.g., search academic papers, answer questions based on Scopus data, etc.). Generate the answer in {lang}.
    """
    welcome = model.generate_content(Content(role="user", parts=[Part.from_text(welcome_prompt)]))
    st.session_state.welcome = welcome.candidates[0].text
    append_message('ai', st.session_state.welcome)

# Display chat messages
for message in st.session_state.chat_session:
    if message['role'] == 'ai':
        with st.chat_message('ai'):
            st.write(message['content'])
    else:
        with st.chat_message('user'):
            st.write(message['content'])

# User input
prompt = st.chat_input("Escribe tu mensaje" if lang == 'Español' else "Write your message")

if prompt:
    append_message('user', prompt)
    
    with st.spinner('Wait a moment, I am thinking...'):
        # Generate a direct answer using the model
        answer_prompt = f"Provide an explanation or answer based on the following query: {prompt}"
        answer_response = model.generate_content(Content(role="user", parts=[Part.from_text(answer_prompt)]))
        explanation = answer_response.candidates[0].text

        append_message('ai', explanation)

        # Perform a Scopus search for references related to the explanation
        scopus_results = search_scopus(prompt)
        if scopus_results:
            response_text = f"\nFound open access documents. Here are the top results:\n\n"
            
            # Encode query and article abstracts
            query_embedding = sentence_model.encode(prompt, convert_to_tensor=True)
            abstracts = [result.get('dc:description', 'No abstract available') for result in scopus_results]
            title_authors = [(result.get('dc:title', 'No title available'), result.get('dc:creator', 'No creator available')) for result in scopus_results]
            
            article_embeddings = sentence_model.encode(abstracts, convert_to_tensor=True)

            # Compute similarity scores
            query_embedding_np = query_embedding.cpu().numpy()
            article_embeddings_np = article_embeddings.cpu().numpy()
            similarities = np.dot(article_embeddings_np, query_embedding_np) / (np.linalg.norm(article_embeddings_np, axis=1) * np.linalg.norm(query_embedding_np))

            # Rank results by similarity
            ranked_results = sorted(zip(similarities, scopus_results, title_authors), key=lambda x: x[0], reverse=True)
            
            detailed_info = ""
            for score, result, (title, creator) in ranked_results:
                date = result.get('prism:coverDate', 'No date available')
                doi = result.get('prism:doi', 'No DOI available')
                abstract = result.get('dc:description', 'No abstract available')
                
                if doi != 'No DOI available':
                    link = f"https://doi.org/{doi}"
                else:
                    link = "No direct link available"
                
                detailed_info += f"Title: {title}\nAuthor: {creator}\nDate: {date}\nDOI: {doi}\nSimilarity Score: {score:.4f}\nLink: {link}\n\n"

            # Generate detailed response
            detailed_prompt = f"""Based on the search results for the query '{prompt}', generate a detailed response including the titles, authors, publication dates, DOIs. Here are the results:\n\n{detailed_info}"""

            detailed_response = model.generate_content(Content(role="user", parts=[Part.from_text(detailed_prompt)]))

            final_output = detailed_response.candidates[0].text.strip()
            response_text += final_output

            append_message('ai', response_text)
        else:
            no_results = "\nNo open access results found."
            append_message('ai', no_results)

        st.rerun()