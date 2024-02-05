import openai
from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os 
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
api_key_openai = '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'



embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(model_name=embed_model_id)


pinecone.init(
    api_key="04e55b76-d298-4d5f-85fe-90ae3b0e4ce4",
    environment='gcp-starter'
)



index = pinecone.Index('igcseeco')



text_field = 'text'
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)



text_field = 'text'
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

llm = OpenAI(openai_api_key=api_key_openai,temperature=0, model_name="gpt-3.5-turbo")
pipe = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff",retriever = vectorstore.as_retriever(),
        return_source_documents=True

)



def get_response(message):
    assistant_response = pipe(message)
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(assistant_response['result'])
    sentiment_score = sentiment_scores['compound']
    print(assistant_response, sentiment_score)

    sentiment_score *= 100
    return assistant_response['result'], sentiment_score



app = Flask(__name__)


@app.route('/')
def home(): 
        return render_template('chatbot.html')


@app.route('/ask', methods=['POST'])
def ask():
    message = request.form['message']
    assistant_response, sentiment_score = get_response(message)
    # selected_model = request.form['model'] 
    
    response_data = {
        'response': assistant_response,
        'sentiment_score': sentiment_score,
    }
    
    return jsonify(response_data)


if __name__ == "__main__":
        app.run()