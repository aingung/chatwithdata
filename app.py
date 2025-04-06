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
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        st.success("File successfully uploaded and read.")
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        # Prepare detailed context for model
        description = df.describe(include='all').to_string()
        sample_rows = df.head(3).to_string(index=False)
        columns_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])

        st.session_state.data_context = (
            f"You are a helpful data analyst AI. The user uploaded a dataset with the following structure:\n"
            f"Columns and Types:\n{columns_info}\n\n"
            f"Descriptive Statistics:\n{description}\n\n"
            f"Sample Records:\n{sample_rows}"
        )

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
                prompt = (
                    f"{st.session_state.data_context}\n\n"
                    f"Now answer the user's question: {user_input}"
                )
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
