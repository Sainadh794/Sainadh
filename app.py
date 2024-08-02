from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

app = Flask(__name__)

# Home page with file upload form
@app.route('/')
def home():
    return render_template('index.html')

# Handle file upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        data = pd.read_csv(file)

        # Example preprocessing: standardize features
        features = data.select_dtypes(include=['float64', 'int64']).columns
        if len(features) > 0:
            scaler = StandardScaler()
            data[features] = scaler.fit_transform(data[features])

        # Example analysis: SVM Classifier
        if 'target' in data.columns:
            X = data[features]
            y = data['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train SVM model
            svm = SVC(kernel='linear')
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)

            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_html = report_df.to_html()

        else:
            report_html = "Target variable not found in the dataset."

        # Generate a summary of the dataset
        summary = data.describe(include='all').to_html()
        return f"<h1>Data Summary</h1>{summary}<h1>Classification Report</h1>{report_html}"

if __name__ == '__main__':
    app.run(debug=True)
