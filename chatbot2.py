import openai
from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

openai.api_key = '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'
# conversation = [{"role": "system", "content": "Act like a girlfriend of someone - shy and saying 'babe'"}]
# conversation = [{"role": "system", "content": "You are a gang member in the UK - a road man in pure slang/ speak like a UK Roadman at all times even when you explain things."}]
conversation = [{"role": "system", "content": "You are a helpful Assistant. Only speak hindi and act as a translater for teaching english to a hindi audience."}]



def get_response(message):
    conversation.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
    assistant_response = response['choices'][0]['message']['content']
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(assistant_response)
    sentiment_score = sentiment_scores['compound']
    
    conversation.append({"role": "assistant", "content": assistant_response})
    print(assistant_response, sentiment_score)
    sentiment_score *= 100
    return assistant_response, sentiment_score



app = Flask(__name__)


@app.route('/')
def home(): 
        return render_template('chatbot.html')


@app.route('/ask', methods=['POST'])
def ask():
    message = request.form['message']
    assistant_response, sentiment_score = get_response(message)
    selected_model = request.form['model'] 
    
    response_data = {
        'response': assistant_response,
        'sentiment_score': sentiment_score,
        'model' : selected_model
    }
    
    return jsonify(response_data)


if __name__ == "__main__":
        app.run()

