# app.py

from flask import Flask, request, render_template
import pickle

# Load the trained model from the pickle file
with open('email_classifier_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Initialize the Flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email_text = request.form['email']
        prediction = clf.predict([email_text])[0]  # Predict if the email is spam or not
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template('index.html', prediction=result, email_text=email_text)  # Pass result back to the same page
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
