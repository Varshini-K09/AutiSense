from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os, json, base64
from io import BytesIO

app = Flask(__name__)

MODEL_PATH = 'model/autism_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def retrain_model():
    df1 = pd.read_csv('data/final_1.csv')
    df2 = pd.read_csv('data/final_2.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Family_ASD'] = le.fit_transform(df['Family_ASD'])
    df['Jaundice'] = le.fit_transform(df['Jaundice'])
    df['Autism'] = le.fit_transform(df['Autism'])
    x = df.drop('Autism', axis=1)
    y = df['Autism']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    importances = model.feature_importances_
    features = list(x.columns)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model, acc, cm, report, importances, features

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, transparent=True)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def make_confusion_matrix_chart(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('#0d1117')
    colors = [['#1a3a4a', '#ff6b35'], ['#2ecc71', '#1a3a4a']]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=colors[i][j], alpha=0.85))
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=22, fontweight='bold', color='white')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Autism', 'Autism'], color='#a0aec0', fontsize=11)
    ax.set_yticklabels(['No Autism', 'Autism'], color='#a0aec0', fontsize=11)
    ax.set_xlabel('Predicted', color='#a0aec0', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual', color='#a0aec0', fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix', color='white', fontsize=14, pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    ax.tick_params(colors='#a0aec0')
    return fig_to_b64(fig)

def make_feature_importance_chart(importances, features):
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('#0d1117')
    colors = ['#00d4ff' if imp > np.median(importances) else '#4a5568' for imp in importances[sorted_idx]]
    bars = ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color=colors, height=0.6, edgecolor='none')
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#a0aec0', fontsize=9)
    ax.set_xlabel('Importance Score', color='#a0aec0', fontsize=11)
    ax.set_title('Feature Importance', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='#a0aec0', labelsize=9)
    ax.set_facecolor('#0d1117')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    fig.tight_layout()
    return fig_to_b64(fig)

def make_prediction_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=False))
    fig.patch.set_alpha(0)
    ax.set_facecolor('#0d1117')
    ax.axis('off')
    theta = np.linspace(0, np.pi, 200)
    r_outer, r_inner = 1.0, 0.6
    # Background arc
    ax.fill_between(np.cos(theta), np.sin(theta)*r_inner, np.sin(theta)*r_outer,
                    color='#1a2744', alpha=0.9)
    # Value arc
    theta_val = np.linspace(0, np.pi * prob, 200)
    if prob < 0.4:
        color = '#2ecc71'
    elif prob < 0.7:
        color = '#f39c12'
    else:
        color = '#e74c3c'
    ax.fill_between(np.cos(theta_val), np.sin(theta_val)*r_inner, np.sin(theta_val)*r_outer,
                    color=color, alpha=0.9)
    ax.text(0, 0.15, f'{prob*100:.1f}%', ha='center', va='center',
            fontsize=26, fontweight='bold', color='white')
    ax.text(0, -0.1, 'Autism Probability', ha='center', va='center',
            fontsize=11, color='#a0aec0')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.1)
    return fig_to_b64(fig)

def make_aq10_chart(scores):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor('#0d1117')
    labels = [f'A{i+1}' for i in range(10)]
    colors = ['#e74c3c' if s == 1 else '#2d3748' for s in scores]
    bars = ax.bar(labels, scores, color=colors, width=0.6, edgecolor='none')
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'], color='#a0aec0')
    ax.set_xlabel('AQ-10 Questions', color='#a0aec0', fontsize=11)
    ax.set_title('AQ-10 Screening Responses', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#a0aec0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    total = sum(scores)
    ax.text(9.5, 1.25, f'Total: {total}/10', ha='right', color='#00d4ff', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig_to_b64(fig)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Map form values
    aq_scores = [int(data.get(f'a{i}', 0)) for i in range(1, 11)]
    age = int(data.get('age', 0))
    gender = 1 if data.get('gender') == 'm' else 0  # LabelEncoder: f=0, m=1
    family_asd = 1 if data.get('family_asd') == 'yes' else 0
    jaundice = 1 if data.get('jaundice') == 'yes' else 0
    eye_contact = int(data.get('eye_contact', 1))
    behavioural = int(data.get('behavioural', 50))

    input_data = pd.DataFrame([{
        'A1_Score': aq_scores[0], 'A2_Score': aq_scores[1], 'A3_Score': aq_scores[2],
        'A4_Score': aq_scores[3], 'A5_Score': aq_scores[4], 'A6_Score': aq_scores[5],
        'A7_Score': aq_scores[6], 'A8_Score': aq_scores[7], 'A9_Score': aq_scores[8],
        'A10_Score': aq_scores[9], 'Age': age, 'Gender': gender,
        'Family_ASD': family_asd, 'Jaundice': jaundice,
        'Eye_Contact_Frequency': eye_contact, 'Behavioural_Score': behavioural
    }])

    # Retrain to get metrics
    model, acc, cm, report, importances, features = retrain_model()

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Charts
    gauge_b64 = make_prediction_gauge(prob)
    cm_b64 = make_confusion_matrix_chart(cm)
    fi_b64 = make_feature_importance_chart(importances, features)
    aq_b64 = make_aq10_chart(aq_scores)

    precision = round(report['1']['precision'] * 100, 1)
    recall = round(report['1']['recall'] * 100, 1)
    f1 = round(report['1']['f1-score'] * 100, 1)

    return render_template('result.html',
        prediction=int(prediction),
        probability=round(prob * 100, 1),
        accuracy=round(acc * 100, 1),
        precision=precision,
        recall=recall,
        f1=f1,
        gauge=gauge_b64,
        cm_chart=cm_b64,
        fi_chart=fi_b64,
        aq_chart=aq_b64,
        age=age,
        aq_total=sum(aq_scores)
    )

if __name__ == '__main__':
    app.run(debug=True)
