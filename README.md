AI-Powered SQL Query Generator for Startup Insights
This project is an AI-driven SQL Query Generator that allows users to ask startup-related business questions in natural language and receive AI-generated SQL queries based on the Kaggle: Startups & Investments dataset.

🔹 Key Features

✅ Natural Language Processing (NLP): Uses Sentence Transformers (all-MiniLM-L6-v2) to embed and understand metadata.

✅ Similarity Matching: Applies cosine similarity to match user queries to the most relevant table.

✅ LLM Integration: Generates SQL queries using Groq’s API (llama3-70b-8192).

✅ Multi-CSV Support: Dynamically loads and processes multiple dataset files.

✅ Gradio UI: Provides an interactive web-based interface for easy user interaction.


This tool simplifies database querying by converting natural language questions into precise SQL queries, making startup data analysis more accessible. 🚀

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
