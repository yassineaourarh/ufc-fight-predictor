from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle et le scaler
model = load_model('model_ufc.keras')  # Assure-toi que le chemin est correct
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées dans le formulaire
        data = [
            float(request.form['BlueAvgSigStrLanded']),
            float(request.form['RedAvgSigStrLanded']),
            float(request.form['BlueAvgSigStrPct']),
            float(request.form['RedAvgSigStrPct']),
            float(request.form['BlueAvgTDLanded']),
            float(request.form['RedAvgTDLanded']),
            float(request.form['BlueReachCms']),
            float(request.form['RedReachCms']),
            float(request.form['BlueHeightCms']),
            float(request.form['RedHeightCms']),
            float(request.form['BlueAge']),
            float(request.form['RedAge']),
        ]

        # Convertir les données en numpy array pour que le modèle puisse les accepter
        data = np.array(data).reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(data)
        
        # Décider du résultat basé sur la prédiction
        result = "Bleu gagne" if prediction[0] > 0.5 else "Rouge gagne"
        
        # Afficher la prédiction et la probabilité
        probability = float(prediction[0])
        
        # Retourner une réponse JSON avec la prédiction
        return render_template('result.html', prediction=result, probability=probability)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)