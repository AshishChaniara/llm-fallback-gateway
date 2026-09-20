import streamlit as st
import litellm
import os
import time

# Configure LiteLLM
litellm.drop_params = True

st.set_page_config(page_title="Multi-Model Fallback Router", page_icon="🔀", layout="wide")

st.title("🔀 Unified LLM API Gateway")
st.markdown("A production-ready gateway with Automatic Fallback Routing using LiteLLM. Ensures zero downtime when primary models hit rate limits or token expirations.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration & Keys")
    st.markdown("Enter your keys below. Keys are strictly kept in the session and environment variables.")
    
    openai_key = st.text_input("OpenAI API Key (Primary)", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    gemini_key = st.text_input("Google Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    groq_key = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    openrouter_key = st.text_input("OpenRouter API Key", type="password", value=os.environ.get("OPENROUTER_API_KEY", ""))
    
    # Update environment dynamically
    if openai_key: os.environ["OPENAI_API_KEY"] = openai_key
    if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key
    if groq_key: os.environ["GROQ_API_KEY"] = groq_key
    if openrouter_key: os.environ["OPENROUTER_API_KEY"] = openrouter_key
    
    st.header("🔀 Extended Fallback Chain")
    st.markdown("1. 🟢 **OpenAI**: `gpt-4o-mini`")
    st.markdown("2. 🟡 **Google Gemini**: `gemini/gemini-1.5-flash`")
    st.markdown("3. 🟡 **Google Gemini**: `gemini/gemini-1.5-pro`")
    st.markdown("4. 🔴 **Groq**: `groq/llama3-8b-8192`")
    st.markdown("5. 🔴 **Groq**: `groq/llama-3.1-70b-versatile`")
    st.markdown("6. 🔴 **Groq**: `groq/mixtral-8x7b-32768`")
    st.markdown("7. 🔴 **Groq**: `groq/gemma2-9b-it`")
    st.markdown("8. 🟣 **OpenRouter (Free)**: `openrouter/meta-llama/llama-3.1-8b-instruct:free`")
    st.markdown("9. 🟣 **OpenRouter (Free)**: `openrouter/google/gemini-flash-1.5-exp:free`")
    st.markdown("10. 🟣 **OpenRouter (Free)**: `openrouter/microsoft/phi-3-mini-128k-instruct:free`")
    
    st.header("📊 Metrics & Logs")
    metrics_placeholder = st.empty()
    st.subheader("Console Logs")
    logs_placeholder = st.empty()

# --- Initialize State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(msg):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    with logs_placeholder.container():
        for log in st.session_state.logs[-10:]:
            st.text(log)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Inference & Fallback Handling ---
if prompt := st.chat_input("Enter your message here..."):
    if not any([openai_key, gemini_key, groq_key, openrouter_key]):
        st.warning("⚠️ Please provide at least one API key in the sidebar to start.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Processing request...")
        
        # Prepare messages
        litellm_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        start_time = time.time()
        
        # Define the extended fallback matrix
        fallback_models = [
            {"model": "gemini/gemini-1.5-flash"},
            {"model": "gemini/gemini-1.5-pro"},
            {"model": "groq/llama3-8b-8192"},
            {"model": "groq/llama-3.1-70b-versatile"},
            {"model": "groq/mixtral-8x7b-32768"},
            {"model": "groq/gemma2-9b-it"},
            {"model": "openrouter/meta-llama/llama-3.1-8b-instruct:free"},
            {"model": "openrouter/google/gemini-flash-1.5-exp:free"},
            {"model": "openrouter/microsoft/phi-3-mini-128k-instruct:free"}
        ]
        
        full_response = ""
        model_used = "Unknown"
        error_occurred = False
        
        try:
            add_log("Attempting primary model: gpt-4o-mini")
            
            # LiteLLM handles the fallback internally across all models in the list
            response = litellm.completion(
                model="gpt-4o-mini",
                messages=litellm_messages,
                fallbacks=fallback_models,
                timeout=15
            )
            
            full_response = response.choices[0].message.content
            model_used = response.model
            
            if model_used != "gpt-4o-mini":
                add_log(f"⚠️ Primary model failed. Fallback activated: {model_used}")
            else:
                add_log(f"✅ Success with primary model: {model_used}")
                
            message_placeholder.markdown(full_response)
            
        except litellm.AuthenticationError as e:
            error_msg = f"Authentication Error (Expired/Invalid Token): {e}"
            st.error(error_msg)
            add_log(error_msg)
            error_occurred = True
        except litellm.RateLimitError as e:
            error_msg = f"Rate Limit Error (Exhausted Free Tier): {e}"
            st.error(error_msg)
            add_log(error_msg)
            error_occurred = True
        except litellm.Timeout as e:
            error_msg = f"Timeout Error: {e}"
            st.error(error_msg)
            add_log(error_msg)
            error_occurred = True
        except Exception as e:
            error_msg = f"Unexpected Error: {e}"
            st.error(error_msg)
            add_log(error_msg)
            error_occurred = True

        latency = time.time() - start_time
        
        if not error_occurred:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            with metrics_placeholder.container():
                st.metric(label="Active Model", value=model_used)
                st.metric(label="Latency", value=f"{latency:.2f} s")
