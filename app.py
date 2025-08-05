from flask import Flask, render_template, request
import pickle

# Load the saved model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    result = "SPAM" if prediction[0] == 1 else "HAM"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
