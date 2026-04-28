
from flask import Flask, render_template, request, jsonify, send_file, session
import pickle
import numpy as np
import pandas as pd
import json
import os
import io
import datetime
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'sleepguard_secret_key_2025'

gbc_model = pickle.load(open('models/gradient_boosting.pkl', 'rb'))
qda_model = pickle.load(open('models/qda.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))

with open('models/results.json', 'r') as f:
    model_results = json.load(f)

dataset = pd.read_csv('Sleep_Data_Sampled.csv')
DISORDER_CLASSES = list(label_encoders['Sleep Disorder'].classes_)
prediction_history = []

RECOMMENDATIONS = {
    'Healthy': {
        'icon': 'check', 'color': '#10b981',
        'message': 'Your sleep health appears normal.',
        'tips': [
            'Continue maintaining a regular sleep schedule.',
            'Keep up with physical activity and stress management.',
            'Aim for 7-9 hours of quality sleep per night.',
            'Avoid screens 1 hour before bedtime.'
        ]
    },
    'Insomnia': {
        'icon': 'alert', 'color': '#f59e0b',
        'message': 'Indicators suggest possible Insomnia.',
        'tips': [
            'Consult a healthcare professional for proper diagnosis.',
            'Establish a consistent sleep-wake schedule.',
            'Limit caffeine and alcohol intake, especially in evening.',
            'Practice relaxation techniques like meditation.',
            'Create a cool, dark, quiet sleep environment.'
        ]
    },
    'Sleep Apnea': {
        'icon': 'warning', 'color': '#ef4444',
        'message': 'Indicators suggest possible Sleep Apnea.',
        'tips': [
            'Consult a sleep specialist for a proper sleep study.',
            'Maintain a healthy weight through diet and exercise.',
            'Sleep on your side rather than your back.',
            'Avoid alcohol and sedatives before bedtime.',
            'Consider using a humidifier in your bedroom.'
        ]
    }
}

CHATBOT_RESPONSES = {
    'insomnia': "Insomnia is difficulty falling or staying asleep. Tips: Maintain a fixed sleep schedule, avoid caffeine after 2 PM, try relaxation techniques like deep breathing, keep your room dark and cool, and limit screen time before bed.",
    'sleep apnea': "Sleep Apnea causes breathing to repeatedly stop during sleep. Symptoms include loud snoring, gasping during sleep, and daytime fatigue. Treatment options include CPAP machines, lifestyle changes, and in some cases surgery.",
    'how much sleep': "Adults need 7-9 hours of sleep per night. Teenagers need 8-10 hours, and children need even more. Quality matters as much as quantity.",
    'improve sleep': "To improve sleep: 1) Stick to a consistent schedule, 2) Exercise regularly but not before bed, 3) Avoid caffeine and heavy meals at night, 4) Make your bedroom dark, quiet and cool, 5) Try a relaxing bedtime routine.",
    'stress': "Stress is a major sleep disruptor. Try: Practice mindfulness meditation, do progressive muscle relaxation, write a worry journal before bed, exercise during the day, and consider speaking with a therapist.",
    'snoring': "Snoring can indicate Sleep Apnea. Tips: Sleep on your side, elevate your head, maintain healthy weight, avoid alcohol before bed, stay hydrated, and use nasal strips. If loud and frequent, consult a doctor.",
    'melatonin': "Melatonin is a natural sleep hormone. Dim lights 1-2 hours before bed, avoid blue light from screens, eat melatonin-rich foods like cherries and walnuts. Consult doctor before taking supplements.",
    'nap': "Best napping practices: Keep naps under 20-30 minutes, nap before 3 PM, avoid napping if you have insomnia. Power naps boost alertness without affecting nighttime sleep.",
    'exercise': "Exercise improves sleep quality. Exercise at least 30 minutes daily, finish vigorous exercise 3-4 hours before bedtime, yoga and stretching are great evening activities.",
    'diet': "Foods that help sleep: Warm milk, chamomile tea, almonds, kiwi, fatty fish, tart cherries. Avoid before bed: Caffeine, spicy foods, heavy meals, alcohol, sugary snacks.",
    'hello': "Hello! I'm Serenova AI Sleep Assistant. Ask me anything about sleep health!",
    'hi': "Hi there! I'm here to help with sleep-related questions. Try asking about insomnia, sleep tips, or stress management!",
    'help': "I can help with: Sleep disorders, sleep improvement tips, stress management, diet & exercise for sleep, napping advice, and more!",
    'thank': "You're welcome! Take care of your sleep health. Good sleep is the foundation of good health!",
    'dream': "Dreams occur during REM sleep. Maintain good sleep hygiene, avoid heavy meals before bed, and manage stress for better dreams.",
    'caffeine': "Caffeine stays in your body for 6-8 hours. Avoid coffee, tea, and energy drinks after 2 PM. Chocolate also contains caffeine!",
    'default': "I'm not sure about that topic. I can help with: sleep disorders, improving sleep quality, stress management, diet tips, exercise advice, napping, and more!"
}


TREND_FILE = 'sleep_trend_data.json'

def calculate_sleep_score(data):
    score = 0
    breakdown = {}
    sd = float(data.get('sleep_duration', 7))
    ds = 30 if 7<=sd<=9 else 22 if 6<=sd<7 or 9<sd<=10 else 15 if 5<=sd<6 or 10<sd<=11 else 8
    score += ds; breakdown['sleep_duration'] = {'score':ds,'max':30,'value':sd}
    q = int(data.get('quality_of_sleep', 5)); qs = round((q/10)*25)
    score += qs; breakdown['quality_of_sleep'] = {'score':qs,'max':25,'value':q}
    st = int(data.get('stress_level', 5)); ss = round(((10-st)/10)*20)
    score += ss; breakdown['stress_level'] = {'score':ss,'max':20,'value':st}
    a = int(data.get('physical_activity', 30))
    acs = 10 if a>=60 else 8 if a>=45 else 6 if a>=30 else 3
    score += acs; breakdown['physical_activity'] = {'score':acs,'max':10,'value':a}
    hr = int(data.get('heart_rate', 72))
    hs = 10 if 60<=hr<=75 else 7 if 55<=hr<60 or 75<hr<=85 else 4
    score += hs; breakdown['heart_rate'] = {'score':hs,'max':10,'value':hr}
    bmi = data.get('bmi_category', 'Normal')
    bs = 5 if bmi in ['Normal','Normal Weight'] else 3 if bmi=='Overweight' else 1
    score += bs; breakdown['bmi_category'] = {'score':bs,'max':5,'value':bmi}
    if score>=85: rating,color,emoji,tip = 'Excellent','#10B981','🌟','Bahut acchi sleep health! Aise hi maintain karo.'
    elif score>=70: rating,color,emoji,tip = 'Good','#3B82F6','😊','Acchi sleep health. Thoda stress kam karo.'
    elif score>=50: rating,color,emoji,tip = 'Fair','#F59E0B','😐','Average. Sleep quality improve karo.'
    elif score>=30: rating,color,emoji,tip = 'Poor','#F97316','😟','Kharab sleep health. Lifestyle change karo.'
    else: rating,color,emoji,tip = 'Critical','#EF4444','🚨','Bahut kharab. Doctor se milo.'
    return {'score':score,'rating':rating,'color':color,'emoji':emoji,'tip':tip,'breakdown':breakdown}

def load_trends():
    if os.path.exists(TREND_FILE):
        with open(TREND_FILE, 'r') as f: return json.load(f)
    return []

def save_trend(entry):
    trends = load_trends()
    trends.append(entry)
    if len(trends)>30: trends = trends[-30:]
    with open(TREND_FILE, 'w') as f: json.dump(trends, f, indent=2)
    return trends


@app.route('/')
def home():
    return render_template('index.html',
                           occupations=sorted(label_encoders['Occupation'].classes_),
                           bmi_categories=sorted(label_encoders['BMI Category'].classes_))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        gender = label_encoders['Gender'].transform([data['gender']])[0]
        age = int(data['age'])
        occupation = label_encoders['Occupation'].transform([data['occupation']])[0]
        sleep_duration = float(data['sleep_duration'])
        quality_of_sleep = int(data['quality_of_sleep'])
        physical_activity = int(data['physical_activity'])
        stress_level = int(data['stress_level'])
        bmi_category = label_encoders['BMI Category'].transform([data['bmi_category']])[0]
        heart_rate = int(data['heart_rate'])
        daily_steps = int(data['daily_steps'])
        systolic_bp = int(data['systolic_bp'])
        diastolic_bp = int(data['diastolic_bp'])

        features = np.array([[gender, age, occupation, sleep_duration,
                              quality_of_sleep, physical_activity, stress_level,
                              bmi_category, heart_rate, daily_steps,
                              systolic_bp, diastolic_bp]])
        features_scaled = scaler.transform(features)

        model_name = data.get('model', 'gbc')
        if model_name == 'gbc':
            prediction = gbc_model.predict(features_scaled)[0]
            probabilities = gbc_model.predict_proba(features_scaled)[0]
            model_display = 'Gradient Boosting Classifier'
        else:
            prediction = qda_model.predict(features_scaled)[0]
            probabilities = qda_model.predict_proba(features_scaled)[0]
            model_display = 'Quadratic Discriminant Analysis'

        predicted_class = label_encoders['Sleep Disorder'].inverse_transform([prediction])[0]
        rec = RECOMMENDATIONS[predicted_class]

        prob_dict = {}
        for i, cls in enumerate(DISORDER_CLASSES):
            prob_dict[cls] = round(float(probabilities[i]) * 100, 2)

        history_entry = {
            'id': len(prediction_history) + 1,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input': {
                'Gender': data['gender'], 'Age': data['age'],
                'Occupation': data['occupation'],
                'Sleep Duration': data['sleep_duration'],
                'Quality of Sleep': data['quality_of_sleep'],
                'Physical Activity': data['physical_activity'],
                'Stress Level': data['stress_level'],
                'BMI Category': data['bmi_category'],
                'Heart Rate': data['heart_rate'],
                'Daily Steps': data['daily_steps'],
                'Blood Pressure': f"{data['systolic_bp']}/{data['diastolic_bp']}"
            },
            'prediction': predicted_class,
            'model': model_display,
            'probabilities': prob_dict
        }
        prediction_history.append(history_entry)

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'model': model_display,
            'probabilities': prob_dict,
            'recommendation': rec,
            'history_id': history_entry['id']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/results')
def results():
    return render_template('results.html', results=model_results)


@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


@app.route('/api/visualization-data')
def viz_data():
    df = dataset.copy()
    disorder_dist = df['Sleep Disorder'].value_counts().to_dict()
    avg_sleep = df.groupby('Sleep Disorder')['Sleep Duration'].mean().round(2).to_dict()
    avg_stress = df.groupby('Sleep Disorder')['Stress Level'].mean().round(2).to_dict()
    avg_hr = df.groupby('Sleep Disorder')['Heart Rate'].mean().round(1).to_dict()
    avg_activity = df.groupby('Sleep Disorder')['Physical Activity Level'].mean().round(1).to_dict()
    avg_quality = df.groupby('Sleep Disorder')['Quality of Sleep'].mean().round(2).to_dict()
    avg_steps = df.groupby('Sleep Disorder')['Daily Steps'].mean().round(0).to_dict()

    gender_disorder = df.groupby(['Gender', 'Sleep Disorder']).size().unstack(fill_value=0).to_dict()
    bmi_disorder = df.groupby(['BMI Category', 'Sleep Disorder']).size().unstack(fill_value=0).to_dict()

    occ_disorder = df.groupby(['Occupation', 'Sleep Disorder']).size().unstack(fill_value=0)
    occ_data = {}
    for col in occ_disorder.columns:
        occ_data[col] = occ_disorder[col].to_dict()

    return jsonify({
        'disorder_distribution': disorder_dist,
        'avg_sleep_duration': avg_sleep,
        'avg_stress_level': avg_stress,
        'avg_heart_rate': avg_hr,
        'avg_activity': avg_activity,
        'avg_quality': avg_quality,
        'avg_steps': avg_steps,
        'gender_disorder': gender_disorder,
        'bmi_disorder': bmi_disorder,
        'occupation_disorder': occ_data,
        'model_results': model_results
    })


@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/api/history')
def api_history():
    return jsonify(prediction_history[::-1])


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    prediction_history.clear()
    return jsonify({'success': True})


@app.route('/download-report/<int:history_id>')
def download_report(history_id):
    entry = None
    for h in prediction_history:
        if h['id'] == history_id:
            entry = h
            break
    if not entry:
        return "Report not found", 404

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(88, 28, 235)
    pdf.cell(0, 15, 'Sleep Disorder Forecast Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, f'Generated on: {entry["timestamp"]}', ln=True, align='C')
    pdf.cell(0, 6, f'Model: {entry["model"]}', ln=True, align='C')
    pdf.ln(10)

    color_map = {'Healthy': (16, 185, 129), 'Insomnia': (245, 158, 11), 'Sleep Apnea': (239, 68, 68)}
    pred_color = color_map.get(entry['prediction'], (100, 100, 100))

    pdf.set_fill_color(240, 240, 250)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(*pred_color)
    pdf.cell(0, 14, f'Prediction: {entry["prediction"]}', ln=True, align='C', fill=True)
    pdf.ln(5)

    rec = RECOMMENDATIONS[entry['prediction']]
    pdf.set_font('Helvetica', 'I', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, rec['message'], ln=True, align='C')
    pdf.ln(8)

    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, 'Patient Details', ln=True)
    pdf.set_draw_color(88, 28, 235)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(60, 60, 60)
    inp = entry['input']
    details = [
        ('Gender', inp['Gender']), ('Age', inp['Age']),
        ('Occupation', inp['Occupation']),
        ('Sleep Duration', f'{inp["Sleep Duration"]} hours'),
        ('Quality of Sleep', f'{inp["Quality of Sleep"]}/10'),
        ('Physical Activity', f'{inp["Physical Activity"]} min/day'),
        ('Stress Level', f'{inp["Stress Level"]}/10'),
        ('BMI Category', inp['BMI Category']),
        ('Heart Rate', f'{inp["Heart Rate"]} bpm'),
        ('Daily Steps', inp['Daily Steps']),
        ('Blood Pressure', inp['Blood Pressure']),
    ]
    for i, (label, value) in enumerate(details):
        bg = i % 2 == 0
        if bg:
            pdf.set_fill_color(248, 248, 255)
        pdf.cell(80, 8, f'  {label}', fill=bg)
        pdf.cell(100, 8, str(value), fill=bg, ln=True)
    pdf.ln(8)

    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, 'Prediction Probabilities', ln=True)
    pdf.set_draw_color(88, 28, 235)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    for cls, prob in entry['probabilities'].items():
        c = color_map.get(cls, (100, 100, 100))
        pdf.set_text_color(60, 60, 60)
        pdf.cell(60, 8, cls)
        pdf.set_text_color(*c)
        pdf.cell(30, 8, f'{prob}%')
        pdf.set_fill_color(*c)
        pdf.cell(prob * 0.9, 6, '', fill=True)
        pdf.ln(9)
    pdf.ln(8)

    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.set_draw_color(88, 28, 235)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(60, 60, 60)
    for i, tip in enumerate(rec['tips'], 1):
        pdf.cell(0, 7, f'  {i}. {tip}', ln=True)
    pdf.ln(8)

    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, 'Disclaimer: This is a ML prediction, not a medical diagnosis. Consult a healthcare professional.', ln=True, align='C')

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f'Sleep_Report_{entry["id"]}.pdf',
                     mimetype='application/pdf')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower().strip()
    response = CHATBOT_RESPONSES['default']
    for keyword, reply in CHATBOT_RESPONSES.items():
        if keyword in user_msg:
            response = reply
            break
    return jsonify({'response': response})


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/sleep-score')
def sleep_score_page():
    return render_template('sleep_score.html')


@app.route('/calculate-score', methods=['POST'])
def calculate_score():
    data = request.json
    result = calculate_sleep_score(data)
    trend_entry = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'score': result['score'], 'rating': result['rating'],
        'sleep_duration': float(data.get('sleep_duration', 0)),
        'quality': int(data.get('quality_of_sleep', 0)),
        'stress': int(data.get('stress_level', 0))
    }
    save_trend(trend_entry)
    return jsonify(result)


@app.route('/trends')
def trends_page():
    return render_template('trends.html')


@app.route('/get-trends')
def get_trends():
    return jsonify(load_trends())


@app.route('/clear-trends', methods=['POST'])
def clear_trends():
    if os.path.exists(TREND_FILE): os.remove(TREND_FILE)
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
