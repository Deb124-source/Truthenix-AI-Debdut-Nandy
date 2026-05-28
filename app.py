from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = None
vectorizer = None


def _load_artifacts():
    global model, vectorizer
    if model is not None and vectorizer is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Missing vectorizer artifact: {VECTORIZER_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)


@app.route('/', methods=['GET'])
def home():
    # Ensure template rendering actually returns content.
    return render_template('index.html') + ''



@app.route('/predict', methods=['POST'])
def predict():
    _load_artifacts()

    # Support both JSON and HTML form submissions.
    data = request.get_json(silent=True) or {}
    text = data.get('text')

    if text is None:
        # HTML form submits under the textarea name="news"
        text = request.form.get('news', '')

    if not isinstance(text, str) or not text.strip():
        # Render the page with an error-like prediction.
        return render_template(
            'index.html',
            prediction='Please enter some news text.',
            color='fake',
            user_input=text if isinstance(text, str) else ''
        )


    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # Normalize prediction display + styling.
    pred_label = int(pred) if isinstance(pred, (int, float)) or str(pred).isdigit() else str(pred)

    # Common conventions: 1=fake or 0=fake. If your dataset uses a different mapping,
    # adjust here.
    # We’ll handle both safely.
    if str(pred_label) == '1':
        color = 'fake'
    else:
        color = 'real'

    return render_template('index.html', prediction=pred_label, color=color, user_input=text)




if __name__ == '__main__':
    # Bind to 127.0.0.1 so browser to localhost works.
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='127.0.0.1', port=port, debug=False)

