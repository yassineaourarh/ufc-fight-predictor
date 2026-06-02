# 🥊 UFC Fight Predictor — du nettoyage de données au déploiement

> Pipeline complet de prédiction de l'issue de combats UFC : nettoyage et
> analyse exploratoire en **R**, comparaison de modèles de Machine Learning
> (ensemble *stacking*), réseau de neurones **Keras**, puis mise en production
> via une application web **Flask**.

![R](https://img.shields.io/badge/R-caret%20%7C%20randomForest-276DC3)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-D00000)
![Flask](https://img.shields.io/badge/Flask-deployed-000000)

<!-- 👉 Ajoute ici une capture de l'app Flask et/ou un graphe d'EDA :
     ![Démo](docs/demo.png) -->


### 🎥 Présentation du projet — Partie 1

[![Regarder la vidéo 1](https://img.youtube.com/vi/_-jlQYU9I4c/maxresdefault.jpg)](https://www.youtube.com/watch?v=_-jlQYU9I4c)

*Cliquez sur l'image pour lancer la première vidéo sur YouTube.*


### 🎥 Présentation du projet — Partie 2

[![Regarder la vidéo 2](https://img.youtube.com/vi/Cs4n6_aqARk/maxresdefault.jpg)](https://www.youtube.com/watch?v=Cs4n6_aqARk)

*Cliquez sur l'image pour lancer la deuxième vidéo sur YouTube.*

---

## 🎯 Objectif

Prédire le vainqueur (coin **Bleu** vs **Rouge**) d'un combat UFC à partir des
statistiques des deux combattants : frappes significatives, takedowns, allonge,
taille, âge. Le projet couvre **toute la chaîne**, de la donnée brute jusqu'à
un modèle servi en ligne.

## 🔬 Démarche

### 1. Données & nettoyage (`analysis/dataprocessing.Rmd`)
- Source : *Ultimate UFC Dataset* (Kaggle) — fichier `ufc-master.csv`.
- Sélection de colonnes, **imputation des valeurs manquantes** par la moyenne.
- Filtrage des combattants sans historique et des lignes trop incomplètes.
- **Analyse exploratoire** (ggplot2) : distributions d'âge, allonge vs victoires,
  cotes stratifiées par vainqueur, densités (KDE).
- **Sélection de variables** par importance Random Forest.

### 2. Modèles de Machine Learning (`analysis/UFCProjetPrevisionFinal.Rmd`)
- Découpage 85 / 15, centrage-réduction, **PCA** (95 % de variance) pour réduire
  la colinéarité.
- Comparaison de **5 approches** :

| Modèle | Type |
|---|---|
| Naïve Bayes | Génératif (sur composantes PCA) |
| KNN | Distance — `k` optimisé sur 1–10 |
| Random Forest | Ensemble bagging |
| AdaBoost | Ensemble boosting |
| **Stacking** | Méta-modèle (régression logistique sur les 4 précédents) |

- Évaluation par **accuracy**, **MAE** et matrices de confusion.
- Le **stacking** donne la meilleure performance d'ensemble.


### 3. Réseau de neurones (`analysis/ReseauNeurones.Rmd`)
- Implémenté en R via **`keras3`** + **`reticulate`** (backend TensorFlow).
- Architecture : `Dense(12, relu) → Dense(4, relu) → Dense(1, sigmoid)`.
- Perte `binary_crossentropy`, optimiseur Adam, 100 époques.
- Modèle exporté en `model_ufc.keras` — c'est lui qui est servi par l'app Flask.

### 4. Mise en production (`app/app.py`)
- API **Flask** chargeant `model_ufc.keras`.
- Formulaire web → 12 statistiques de combat → prédiction du vainqueur et
  probabilité associée, en temps réel.

## 🧰 Stack technique

**R** (caret, randomForest, e1071, class, adabag, dplyr, ggplot2) ·
**Keras / TensorFlow** (via `keras3` + `reticulate`) ·
**Python / Flask** (déploiement) · **Kaggle UFC dataset**

## 📁 Structure du dépôt

```
ufc-fight-predictor/
├── README.md
├── analysis/
│   ├── dataprocessing.Rmd          # nettoyage + EDA + sélection de variables
│   ├── UFCProjetPrevisionFinal.Rmd # ML classique + stacking
│   └── ReseauNeurones.Rmd          # réseau de neurones Keras
├── app/
│   ├── app.py                      # API Flask
│   ├── templates/                  # index.html, result.html
│   └── model_ufc.keras             # modèle entraîné
├── docs/
│   └── presentation.pdf            # diaporama du projet
└── .gitignore
```

## ▶️ Lancer l'application

```bash
cd app
pip install flask tensorflow numpy scikit-learn joblib
python app.py
# → http://localhost:5001
```

## 📊 Données

Le dataset provient de Kaggle (*Ultimate UFC Dataset*). Les fichiers CSV ne sont
**pas** versionnés ici (trop volumineux / licence) : télécharge `ufc-master.csv`
depuis Kaggle et place-le à la racine de `analysis/` avant de ré-exécuter les
notebooks. Les fichiers dérivés (`ufc_data_cleaned_filtered.csv`, etc.) sont
régénérés par `dataprocessing.Rmd`.

## 🗺️ Pistes d'amélioration

- Ajouter **XGBoost / Gradient Boosting** (amorcé, non finalisé).
- Réentraîner le réseau avec features normalisées et comparer.
- Empaqueter l'app Flask dans un **Dockerfile** + tests automatisés.

---

*Projet académique de Data Science — analyse en R, modélisation comparative et
déploiement.*
