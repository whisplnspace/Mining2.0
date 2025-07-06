import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate

# Set Streamlit page config
st.set_page_config(page_title="MinerlexAI", page_icon="🤖", layout="wide")
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("🔴 Error: Gemini API key is missing. Set it in a .env file.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

# RAG setup
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = None
rag_model = Ollama(model="deepseek-r1:1.5b")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

@st.cache_data(show_spinner=False)
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name
    docs = PDFPlumberLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True).split_documents(docs)
    return chunks

def index_docs(documents):
    global vector_store
    vector_store = FAISS.from_documents(documents, embeddings)

def retrieve_docs(query):
    if vector_store is None:
        st.warning("📂 Please upload and process a PDF to enable contextual answers.")
        return []
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    return (prompt | rag_model).invoke({"question": question, "context": context})

# Translation & TTS
LANGUAGE_CODES = {"English": "en_XX", "Hindi (हिन्दी)": "hi_IN", "Bengali (বাংলা)": "bn_IN", "Tamil (தமிழ்)": "ta_IN", "Telugu (తెలుగు)": "te_IN", "Marathi (मराठी)": "mr_IN", "Gujarati (ગુજરાતી)": "gu_IN", "Malayalam (മലയാളം)": "ml_IN", "Kannada (ಕನ್ನಡ)": "kn_IN", "Odia (ଓଡ଼ିଆ)": "or_IN", "Urdu (اردو)": "ur_PK", "Assamese (অসমীয়া)": "as_IN"}
GTTS_LANGUAGE_CODES = {"English": "en", "Hindi (हिन्दी)": "hi", "Bengali (বাংলা)": "bn", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te", "Marathi (मराठी)": "mr", "Gujarati (ગુજરાતી)": "gu", "Malayalam (മലയാളം)": "ml", "Kannada (ಕನ್ನಡ)": "kn", "Urdu (اردو)": "ur", "Assamese (অসমীয়া)": "as"}

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
translator_model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate_text(text, target_language):
    if target_language == "English":
        return text
    try:
        model_inputs = tokenizer(text, return_tensors="pt")
        tokens = translator_model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[LANGUAGE_CODES.get(target_language, "en_XX")]
        )
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def generate_audio(text, language):
    try:
        tts = gTTS(text, lang=GTTS_LANGUAGE_CODES.get(language, "en"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None

# --- 🌌 Custom CSS (Fixed language dropdown) ---
custom_css = """
<style>
body {
    background-color: #1e1e2f;
    font-family: 'Poppins', sans-serif;
    color: #ffffff;
}

h1, h3 {
    text-align: center;
    color: #FF6B6B;
}

.stButton > button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border: none;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    width: 100%;
    font-size: 0.9rem;
    transition: 0.3s ease-in-out;
    box-shadow: 0 0 10px rgba(0, 115, 255, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0072ff, #00c6ff);
    transform: scale(1.02);
}

.stTextInput > div > input {
    background-color: #2c2f4a;
    color: #ffffff;
    border-radius: 10px;
    padding: 0.6rem;
    border: 1px solid #444;
    font-size: 0.9rem;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #2c2f4a !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
    font-size: 0.9rem;
    z-index: 9999 !important;
}

.sidebar .stFileUploader, .css-1djdyxw {
    background-color: #2a2c3e !important;
    border-radius: 10px;
}

audio {
    width: 100% !important;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# UI Layout
st.markdown("""
    <div style='text-align:center; padding-top:10px;'>
        <h1>🚀 MinerlexAI</h1>
        <h3>Revolutionizing Mining Law with AI-Powered Precision</h3>
    </div>
""", unsafe_allow_html=True)

# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Your PDF", type="pdf")
if uploaded_file:
    with st.spinner("📖 Processing PDF..."):
        chunks = process_pdf(uploaded_file)
        index_docs(chunks)
        st.sidebar.success("✅ PDF processed and indexed.")
        st.sidebar.write(f"📝 Uploaded File: `{uploaded_file.name}`")
        st.sidebar.write(f"📄 Chunks Indexed: {len(chunks)}")

selected_language = st.selectbox("🌐 Choose Language:", list(LANGUAGE_CODES.keys()))
user_input = st.text_input("💬 Type your message:")

col1, col2 = st.columns([1, 1])
with col1:
    send_clicked = st.button("📩 Send")
with col2:
    speak_clicked = st.button("🎙️ Speak")

if speak_clicked:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Listening... Speak clearly.")
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)
            st.success(f"🗣️ You said: {user_input}")
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("⚠️ API unavailable. Check your internet.")

if user_input and (send_clicked or speak_clicked):
    with st.spinner("🔍 Generating Answer..."):
        docs = retrieve_docs(user_input)
        if docs:
            answer = answer_question(user_input, docs)
        else:
            chat_session = model.start_chat(history=[])
            answer = chat_session.send_message(user_input).text

        translated = translate_text(answer, selected_language)

        st.subheader("💡 AI Response:")
        st.success(answer)

        st.subheader(f"🌐 Translation ({selected_language}):")
        st.write(translated)

        audio_path = generate_audio(translated, selected_language)
        if audio_path:
            st.audio(audio_path, format="audio/mp3")
