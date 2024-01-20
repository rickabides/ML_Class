from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend such as Agg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import base64

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('Salary_dataset.csv')  # Replace with your dataset path
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

def create_plot():
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, model.predict(X_test), color='blue', linewidth=3)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Linear Regression Model')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def home():
    plot_url = create_plot()
    if request.method == 'POST':
        try:
            years_exp = float(request.form['years_exp'])
            return redirect(url_for('predict', years_exp=years_exp))
        except ValueError:
            # Handle invalid input
            return render_template('index.html', plot_url=plot_url, error="Please enter a valid number.")
    
    return render_template('index.html', plot_url=plot_url)

@app.route('/predict')
def predict():
    years_exp = request.args.get('years_exp', type=float)
    prediction = model.predict([[years_exp]])[0]
    return render_template('results.html', years_exp=years_exp, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
