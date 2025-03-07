# Mobile_information_chatbot
Setup Instructions
Prerequisites:

Python 3.7 or higher
pip package manager

Installation:

Create and activate a virtual environment (recommended):
bash
Copy python -m venv phone_chatbot_env
source phone_chatbot_env/bin/activate   # On Windows: phone_chatbot_env\Scripts\activate

Install required packages:
bash
Copy pip install requests beautifulsoup4 transformers torch numpy

Save the code:
Save the provided Python code as Chatbot_Mobile_Details.py
Run the chatbot:
bash
Copy python Chatbot_Mobile_Details.py


Usage Examples:
Once running, you can interact with the chatbot using commands like:

"What are the latest phones right now?"
"Tell me about the iPhone 15 Pro"
"Compare Samsung Galaxy S24 and iPhone 15"
"What are the specs of the Google Pixel 8?"

Notes:

The chatbot requires an active internet connection to scrape data from GSMArena
Initial queries might be slower due to model loading and first-time web requests
The system has special handling for iPhone models to better match their naming conventions

This chatbot demonstrates the effective integration of web scraping and NLP technologies to create a practical, user-friendly assistant for mobile phone information retrieval.RetryClaude does not have the ability to run the code it generates yet.Claude can make mistakes. Please double-
