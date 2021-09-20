from flask import Flask, render_template, request
from flask_restful import Api, Resource
from joblib import load

app = Flask(__name__)
api = Api(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model = load('svc_model.joblib.pkl')
        count_vect = load('vec.joblib.pkl')
        message = request.form['message']
        pred_industry = model.predict(count_vect.transform([message]))
    return render_template('result.html', prediction=pred_industry[0])


class Model(Resource):
    def get(self, job_title):
        model = load('svc_model.joblib.pkl')
        count_vect = load('vec.joblib.pkl')
        pred_industry = model.predict(count_vect.transform([job_title]))
        return pred_industry[0], 200


api.add_resource(Model, "/api/<string:job_title>")

if __name__ == '__main__':
    app.run()
