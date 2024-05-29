import streamlit as st
import requests
import vertexai
from vertexai.generative_models import Content, FunctionDeclaration, GenerativeModel, Part, Tool
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import numpy as np
import google.auth
import os
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64

def decrypt_json_file(encrypted_file, secret_key):
    key = secret_key.encode('utf-8')
    key = key[:16].ljust(16, b'\0')
    with open(encrypted_file, 'rb') as f:
        encrypted_data = base64.b64decode(f.read())
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
    json_data = decrypted_data.decode('utf-8')
    return json.loads(json_data)

load_dotenv()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
SCOPUS_API_KEY = st.secrets["SCOPUS_API_KEY"]
GOOGLE_CREDENTIALS = st.secrets["GOOGLE_CREDENTIALS"]['gc']

project_id = GOOGLE_CREDENTIALS["project_id"]
vertexai.init(project=project_id, location="us-central1")

secret_key = st.secrets["secret_key"]
decrypted_json = decrypt_json_file('gdc_encrypted.json', secret_key)
temp_key_path = 'temp_decrypted_key.json'
with open(temp_key_path, 'w') as f:
    json.dump(decrypted_json, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
credentials, project_id = google.auth.default()
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="Scopus AI Chatbot",
    page_icon="https://seeklogo.com/images/G/google-ai-logo-996E85F6FD-seeklogo.com.png",
    layout="centered",
)

st.title("🤖 Scopus AI Chatbot")

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

scopus_tool = Tool(
    function_declarations=[search_scopus_func],
)

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer_model()

def search_scopus(query):
    url = "https://api.elsevier.com/content/search/scopus"
    headers = {
        "X-ELS-APIKey": SCOPUS_API_KEY,
    }
    params = {
        "query": f"{query} AND openaccess(1)",
        "count": 10,
        "field": "dc:title,dc:creator,prism:coverDate,prism:doi,dc:description"
    }
    response = requests.get(url, headers=headers, params=params)
    results = response.json()
    return results.get('search-results', {}).get('entry', [])

def append_message(role, content):
    st.session_state.chat_session.append({"role": role, "content": content})

@st.cache_resource
def load_model(model_name):
    model = GenerativeModel(model_name)
    return model

model = load_model("gemini-1.0-pro")

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = []

if 'welcome' not in st.session_state or lang != st.session_state.lang:
    st.session_state.lang = lang
    welcome_prompt = f"""
    Give a welcome message to the user and suggest what they can do (e.g., search academic papers, answer questions based on Scopus data, etc.). Generate the answer in {lang}.
    """
    welcome = model.generate_content(Content(role="user", parts=[Part.from_text(welcome_prompt)]))
    st.session_state.welcome = welcome.candidates[0].text
    append_message('ai', st.session_state.welcome)

for message in st.session_state.chat_session:
    if message['role'] == 'ai':
        with st.chat_message('ai'):
            st.write(message['content'])
    else:
        with st.chat_message('user'):
            st.write(message['content'])

prompt = st.chat_input("Escribe tu mensaje" if lang == 'Español' else "Write your message")

def generate_response_with_retry(prompt, retries=3):
    for _ in range(retries):
        responses = model.generate_content(Content(role="user", parts=[Part.from_text(prompt)]), stream=True)
        explanation = ""
        for response in responses:
            explanation += response.text
        if explanation.strip():
            return explanation
        st.write("Response blocked by safety filters, retrying...")
    return "Sorry, I'm unable to provide a response at this time due to safety filters."

if prompt:
    append_message('user', prompt)
    
    with st.spinner('Wait a moment, I am thinking...'):
        explanation = generate_response_with_retry(prompt)
        append_message('ai', explanation)
        
        scopus_results = search_scopus(prompt)
        if scopus_results:
            response_text = f"\nFound open access documents. Here are the top results:\n\n"
            query_embedding = sentence_model.encode(prompt, convert_to_tensor=True)
            abstracts = [result.get('dc:description', 'No abstract available') for result in scopus_results]
            title_authors = [(result.get('dc:title', 'No title available'), result.get('dc:creator', 'No creator available')) for result in scopus_results]
            
            article_embeddings = sentence_model.encode(abstracts, convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().numpy()
            article_embeddings_np = article_embeddings.cpu().numpy()
            similarities = np.dot(article_embeddings_np, query_embedding_np) / (np.linalg.norm(article_embeddings_np, axis=1) * np.linalg.norm(query_embedding_np))

            ranked_results = sorted(zip(similarities, scopus_results, title_authors), key=lambda x: x[0], reverse=True)
            
            detailed_info = ""
            for score, result, (title, creator) in ranked_results:
                date = result.get('prism:coverDate', 'No date available')
                doi = result.get('prism:doi', 'No DOI available')
                abstract = result.get('dc:description', 'No abstract available')
                link = f"https://doi.org/{doi}" if doi != 'No DOI available' else "No direct link available"
                detailed_info += f"Title: {title}\nAuthor: {creator}\nDate: {date}\nDOI: {doi}\nSimilarity Score: {score:.4f}\nLink: {link}\n\n"

            detailed_prompt = f"""Based on the search results for the query '{prompt}', generate a detailed response including the titles, authors, publication dates, DOIs. Here are the results:\n\n{detailed_info}"""
            detailed_responses = model.generate_content(Content(role="user", parts=[Part.from_text(detailed_prompt)]), stream=True)
            final_output = "".join([response.text for response in detailed_responses])
            response_text += final_output.strip()

            append_message('ai', response_text)
        else:
            append_message('ai', explanation + "\nNo open access results found.")

        st.rerun()
