from flask import Flask,render_template,request
from breastcancerdetector.predict import predict_cancer

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():

    classifier = predict_cancer.CancerClassifier()

    if request.method == 'POST':
        try:
            clump_thickness = request.form['Clump_Thickness']
            uniformity_of_cell_zize = request.form['Uniformity_of_Cell_Size']
            uniformity_of_cell_shape = request.form['Uniformity_of_Cell_Shape']
            marginal_adhesion = request.form['Marginal_Adhesion']
            single_epithelial_cell_size = request.form['Single_Epithelial_Cell_Size']
            bare_nuclei = request.form['Bare_Nuclei']
            bland_chromatin = request.form['Bland_Chromatin']
            normal_nucleoli = request.form['Normal_Nucleoli']
            mitoses = request.form['Mitoses']

            data = [clump_thickness, uniformity_of_cell_zize, uniformity_of_cell_shape,
                    marginal_adhesion, single_epithelial_cell_size, bare_nuclei, bland_chromatin,
                    normal_nucleoli, mitoses]

            my_prediction = classifier.predict(data)
        except:
            return render_template("error.html")
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)