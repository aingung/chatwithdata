import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import traceback

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
    st.session_state.uploaded_data = []
if "data_context" not in st.session_state:
    st.session_state.data_context = ""
if "data_dictionary" not in st.session_state:
    st.session_state.data_dictionary = None

# Display previous chat history 
for role, message in st.session_state.chat_history:
    st.chat_message(role).markdown(message)

# Allow multiple file uploads for CSV
st.subheader("Upload CSV Files for Analysis")
uploaded_files = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True)
if uploaded_files:
    all_contexts = []
    st.session_state.uploaded_data = []  # Reset stored data
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            st.session_state.uploaded_data.append((file.name, df))
            st.success(f"File '{file.name}' uploaded and read.")
            st.write(f"### Preview of {file.name}")
            st.dataframe(df.head())

            # Build context for each file
            description = df.describe(include='all').to_string()
            sample_rows = df.head(3).to_string(index=False)
            columns_info = "\n".join([f"- {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            file_context = (
                f"File: {file.name}\n"
                f"Columns and Types:\n{columns_info}\n\n"
                f"Descriptive Statistics:\n{description}\n\n"
                f"Sample Records:\n{sample_rows}\n"
            )
            all_contexts.append(file_context)

        except Exception as e:
            st.error(f"An error occurred while reading file '{file.name}': {e}")

    # Store combined context
    st.session_state.data_context = (
        "You are a helpful data analyst AI. The user uploaded multiple datasets. Here is the context for each:\n\n"
        + "\n\n".join(all_contexts)
    )

# Upload optional data dictionary
st.subheader("Upload Data Dictionary (Optional)")
dict_file = st.file_uploader("Choose a CSV data dictionary file", type=["csv"], key="dict_file")
if dict_file is not None:
    try:
        data_dict = pd.read_csv(dict_file)
        st.session_state.data_dictionary = data_dict
        st.success("Data dictionary successfully uploaded and read.")
        st.write("### Data Dictionary Preview")
        st.dataframe(data_dict)

        # Append dictionary to data context
        dict_info = data_dict.to_string(index=False)
        st.session_state.data_context += f"\n\nData Dictionary:\n{dict_info}"

    except Exception as e:
        st.error(f"An error occurred while reading the data dictionary file: {e}")

# Checkbox for indicating data analysis need 
analyze_data_checkbox = st.checkbox("Analyze CSV Data with AI")

# Capture user input and generate bot response
if user_input := st.chat_input("Ask anything about your data or start a chat..."):
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    if model:
        if st.session_state.uploaded_data:
            for file_name, df in st.session_state.uploaded_data:
                df_name = "df"
                example_record = df.head(2).to_string(index=False)
                data_dict_text = "\n".join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
                question = user_input

                # 1. Generate code using Gemini
                code_prompt = f"""
You are a helpful Python code generator.
Your goal is to write Python code snippets based on the user's question and the provided DataFrame information.
Here's the context:
**User Question:**
{question}
**DataFrame Name:**
{df_name}
**DataFrame Details:**
{data_dict_text}
**Sample Data (Top 2 Rows):**
{example_record}

**Instructions:**
1. Write Python code that addresses the user's question by querying or manipulating the DataFrame.
2. **Crucially, use the `exec()` function to execute the generated code.**
3. Do not import pandas
4. Change date column type to datetime
5. **Store the result of the executed code in a variable named `ANSWER`.**
6. Assume the DataFrame is already loaded into a pandas DataFrame object named `{df_name}`.
7. Keep the generated code concise and focused on answering the question.
"""

                response = model.generate_content(code_prompt)
                generated_code = response.text

                st.markdown("#### 🧠 Generated Python Code")
                st.code(generated_code, language='python')

                try:
                    local_vars = {df_name: df.copy()}
                    exec(generated_code, {}, local_vars)
                    answer_result = local_vars.get("ANSWER", "No result in variable ANSWER")
                    st.session_state.chat_history.append(("assistant", f"Executed the following code:\n```python\n{generated_code}\n```\n\n**Result:**\n{answer_result}"))
                    st.chat_message("assistant").markdown(f"**Result Preview:**\n{answer_result}")

                    # 2. Generate explanation and summary
                    explain_prompt = f'''
The user asked: "{question}",
Here is the result:\n{str(answer_result)}
Answer the question and summarize the findings,
Include your opinion of the persona of this customer if relevant.
'''
                    explain_response = model.generate_content(explain_prompt)
                    explanation_text = explain_response.text
                    st.session_state.chat_history.append(("assistant", explanation_text))
                    st.chat_message("assistant").markdown(f"**Summary & Interpretation:**\n{explanation_text}")

                except Exception as e:
                    error_msg = f"⚠️ An error occurred during code execution: {e}\n\n{traceback.format_exc()}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))
        else:
            bot_response = "Please upload one or more CSV files first to analyze."
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
    else:
        st.warning("Please configure the Gemini API Key to enable chat responses.")
