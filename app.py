from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import pandas as pd
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# Use the current directory for Hugging Face Spaces
dataset_folder = "./data"  # Assuming files are in a 'data/' folder

# Verify the folder exists
if not os.path.exists(dataset_folder):
    print(f"Warning: Dataset folder '{dataset_folder}' not found. Using current directory instead.")
    dataset_folder = "."  # Fallback: Look in the current directory

# Print available files for debugging
print("Available files:", os.listdir(dataset_folder))

import warnings

# Ignore DtypeWarning
warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)

# Load all CSV files in the dataset folder
dataframes = []
for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):  # Check if the file is a CSV
        try:
            # Read first few rows to identify column names
            sample_df = pd.read_csv(
                os.path.join(dataset_folder, file),
                nrows=5,  # Read only first 5 rows for column type inference
                encoding="utf-8",
                errors="replace"  # Replace encoding errors with a placeholder
            )

            column_types = {col: str for col in sample_df.columns}  # Force all columns to string
            
            # Read the entire file with enforced column types
            df = pd.read_csv(
                os.path.join(dataset_folder, file),
                dtype=column_types,  # Apply enforced string types
                low_memory=False,  # Avoid chunk-based reading issues
                encoding="utf-8",
                errors="replace"
            ).fillna('')  # Fill NaN values with empty strings
            
            dataframes.append(df)  # Append DataFrame to the list
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Merge all CSV files into one DataFrame (only if there are valid files)
if dataframes:
    full_data = pd.concat(dataframes, ignore_index=True)
else:
    print("Warning: No valid CSV files found in the dataset folder.")
    full_data = pd.DataFrame()  # Create an empty DataFrame as a fallback
    

def load_dataset_metadata(dataset_folder):
    """Loads metadata from all CSV files in the dataset folder."""
    dataframes = []
    metadata_list = []
    
    for file in os.listdir(dataset_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dataset_folder, file))
            dataframes.append((file, df))

            # Generate table metadata
            columns = df.columns.tolist()
            table_metadata = f"""
            Table: {file.replace('.csv', '')}
            Columns:
            {', '.join(columns)}
            """
            metadata_list.append(table_metadata)
    
    return dataframes, metadata_list

def create_metadata_embeddings(metadata_list):
    """Creates embeddings for all table metadata."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(metadata_list)
    return embeddings, model

def find_best_fit(embeddings, model, user_query, metadata_list):
    """Finds the best matching table based on user query."""
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, embeddings)
    best_match_index = similarities.argmax()
    return metadata_list[best_match_index]

def create_prompt(user_query, table_metadata):
    """Generates a direct and structured SQL prompt with stricter formatting."""
    system_prompt = f"""
    You are an AI assistant that generates precise SQL queries based on user questions.

    **Table Name & Columns:**
    {table_metadata}

    **User Query:**
    {user_query}

    **Output Format (STRICT):**
    - Provide ONLY the SQL query.
    - Do NOT include explanations, comments, or unnecessary text.
    - Ensure the table and column names match exactly.
    - If the query is impossible, return: "ERROR: Unable to generate query."

    **Example Queries:**
    - User: "Show all startups founded in 2020."
    - AI Response: SELECT * FROM startups WHERE founded_year = 2020;
    
    - User: "List the top 5 startups by total funding."
    - AI Response: SELECT name, total_funding FROM startups ORDER BY total_funding DESC LIMIT 5;
    """
    return system_prompt


def generate_sql_query(system_prompt):
    """Uses Groq API to generate an SQL query with better debugging."""
    try:
        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama3-70b-8192"
        )

        # Debug: Print entire response
        print("🔍 Full API Response:", chat_completion)

        # Extract AI response
        result = chat_completion.choices[0].message.content.strip()
        print(f"✅ AI Response: {result}")  # Debugging

        # Check if the response starts with "SELECT"
        if result.lower().startswith("select"):
            return result
        else:
            print("⚠️ AI did not generate a valid SQL query!")
            return "⚠️ AI response is not a valid SQL query."

    except Exception as e:
        print(f"❌ API Error: {e}")
        return "⚠️ API failed. Check logs."


def response(user_query, dataset_folder):
    """Processes the user query and returns an SQL query."""
    dataframes, metadata_list = load_dataset_metadata(dataset_folder)
    embeddings, model = create_metadata_embeddings(metadata_list)
    table_metadata = find_best_fit(embeddings, model, user_query, metadata_list)
    system_prompt = create_prompt(user_query, table_metadata)
    return generate_sql_query(system_prompt)

dataset_folder = "./data"  # Change this based on where your files are uploaded
user_query = "Show me the top 10 startups with the highest funding."

def sql_query_interface(user_query):
    return response(user_query, dataset_folder)

# Define Gradio UI
iface = gr.Interface(
    fn=sql_query_interface,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Generated SQL Query"),
    title="AI-Powered SQL Query Generator"
)

# Run Gradio app
if __name__ == "__main__":
    iface.launch()