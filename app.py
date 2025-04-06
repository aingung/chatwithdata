import streamlit as st
import pandas as pd
import google.generativeai as genai

# Set up the Streamlit app layout 
st.title("My Chatbot and Data Analysis App") 
st.subheader("Conversation and Data Analysis")

# Capture Gemini API Key
gemini_api_key = st.secrets['gemini_api_key']

# Initialize the Gemini Model
model = None
data_context = ""
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite") 
        st.success("Gemini API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")

# Initialize session state for storing chat history and data 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "data_context" not in st.session_state:
    st.session_state.data_context = ""

# Display previous chat history 
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Add a file uploader for CSV data
st.subheader("Upload CSV for Analysis")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        st.session_state.uploaded_data = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded and read.")
        st.write("### Uploaded Data Preview")
        st.dataframe(st.session_state.uploaded_data.head())

        # Prepare data context for model
        description = st.session_state.uploaded_data.describe(include='all').to_string()
        sample_rows = st.session_state.uploaded_data.head(3).to_string(index=False)
        st.session_state.data_context = f"This dataset has the following stats:\n{description}\n\nHere are sample rows:\n{sample_rows}"

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

# Checkbox for indicating data analysis need 
analyze_data_checkbox = st.checkbox("Analyze CSV Data with AI")

# Capture user input and generate bot response
if user_input := st.chat_input("Ask anything about your data or start a chat..."):
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    if model:
        try:
            if st.session_state.uploaded_data is not None:
                prompt = f"You are a data analyst. Here is the dataset context:\n\n{st.session_state.data_context}\n\nNow answer the question: {user_input}"
                response = model.generate_content(prompt)
                bot_response = response.text
                st.session_state.chat_history.append(("assistant", bot_response))
                st.chat_message("assistant").markdown(bot_response)
            else:
                bot_response = "Please upload a CSV file first to analyze."
                st.session_state.chat_history.append(("assistant", bot_response))
                st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    else:
        st.warning("Please configure the Gemini API Key to enable chat responses.")
