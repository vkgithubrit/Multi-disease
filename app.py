
# import streamlit as st 
# import pickle 
# import os
# from streamlit_option_menu import option_menu

# st.set_page_config(page_title="Mulitple Disease Prediction",layout="wide", page_icon="👨‍🦰🤶")

# working_dir = os.path.dirname(os.path.abspath(__file__))

# diabetes_model = pickle.load(open(f'{working_dir}/notebook/diabetes_model.pkl','rb'))
# heart_disease_model = pickle.load(open(f'{working_dir}/notebook/heart_model.pkl','rb'))
# parkinsons_disease_model = pickle.load(open(f'{working_dir}/notebook/parkinsons_model.pkl','rb'))

# NewBMI_Overweight=0
# NewBMI_Underweight=0
# NewBMI_Obesity_1=0
# NewBMI_Obesity_2=0 
# NewBMI_Obesity_3=0
# NewInsulinScore_Normal=0 
# NewGlucose_Low=0
# NewGlucose_Normal=0 
# NewGlucose_Overweight=0
# NewGlucose_Secret=0

# with st.sidebar:
#     selected = option_menu("Mulitple Disease Prediction", 
#                 ['Diabetes Prediction',
#                  'Heart Disease Prediction',
#                  'Parkinsons Disease Prediction'],
#                  menu_icon='hospital-fill',
#                  icons=['activity','heart', 'person'],
#                  default_index=0)

# # if selected == 'Diabetes Prediction':
# #     st.title("Diabetes Prediction Using Machine Learning")

# #     col1, col2, col3 = st.columns(3)

# #     with col1:
# #         Pregnancies = st.text_input("Number of Pregnancies")
# #     with col2:
# #         Glucose = st.text_input("Glucose Level")
# #     with col3:
# #         BloodPressure = st.text_input("BloodPressure Value")
# #     with col1:
# #         SkinThickness = st.text_input("SkinThickness Value")
# #     with col2:
# #         Insulin = st.text_input("Insulin Value")
# #     with col3:
# #         BMI = st.text_input("BMI Value")
# #     with col1:
# #         DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
# #     with col2:
# #         Age = st.text_input("Age")
# #     diabetes_result = ""
# #     if st.button("Diabetes Test Result"):
# #         if float(BMI)<=18.5:
# #             NewBMI_Underweight = 1
# #         elif 18.5 < float(BMI) <=24.9:
# #             pass
# #         elif 24.9<float(BMI)<=29.9:
# #             NewBMI_Overweight =1
# #         elif 29.9<float(BMI)<=34.9:
# #             NewBMI_Obesity_1 =1
# #         elif 34.9<float(BMI)<=39.9:
# #             NewBMI_Obesity_2=1
# #         elif float(BMI)>39.9:
# #             NewBMI_Obesity_3 = 1
        
# #         if 16<=float(Insulin)<=166:
# #             NewInsulinScore_Normal = 1

# #         if float(Glucose)<=70:
# #             NewGlucose_Low = 1
# #         elif 70<float(Glucose)<=99:
# #             NewGlucose_Normal = 1
# #         elif 99<float(Glucose)<=126:
# #             NewGlucose_Overweight = 1
# #         elif float(Glucose)>126:
# #             NewGlucose_Secret = 1

# #         user_input=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
# #                     BMI,DiabetesPedigreeFunction,Age, NewBMI_Underweight,
# #                     NewBMI_Overweight,NewBMI_Obesity_1,
# #                     NewBMI_Obesity_2,NewBMI_Obesity_3,NewInsulinScore_Normal, 
# #                     NewGlucose_Low,NewGlucose_Normal, NewGlucose_Overweight,
# #                     NewGlucose_Secret]
        
# #         user_input = [float(x) for x in user_input]
# #         prediction = diabetes_model.predict([user_input])
# #         if prediction[0]==1:
# #             diabetes_result = "The person has diabetic"
# #         else:
# #             diabetes_result = "The person has no diabetic"
# #     st.success(diabetes_result)
# # ---------- Diabetes Prediction block (REPLACE the existing block) ----------
# if selected == 'Diabetes Prediction':
#     st.title("Diabetes Prediction Using Machine Learning")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         Pregnancies = st.text_input("Number of Pregnancies")
#     with col2:
#         Glucose = st.text_input("Glucose Level")
#     with col3:
#         BloodPressure = st.text_input("BloodPressure Value")
#     with col1:
#         SkinThickness = st.text_input("SkinThickness Value")
#     with col2:
#         Insulin = st.text_input("Insulin Value")
#     with col3:
#         BMI = st.text_input("BMI Value")
#     with col1:
#         DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
#     with col2:
#         Age = st.text_input("Age")

#     diabetes_result = ""
#     if st.button("Diabetes Test Result"):
#         # --- Defensive conversion of raw inputs ---
#         try:
#             raw_Pregnancies = float(Pregnancies)
#             raw_Glucose = float(Glucose)
#             raw_BloodPressure = float(BloodPressure)
#             raw_SkinThickness = float(SkinThickness)
#             raw_Insulin = float(Insulin)
#             raw_BMI = float(BMI)
#             raw_DPF = float(DiabetesPedigreeFunction)
#             raw_Age = float(Age)
#         except Exception as e:
#             st.error("Please fill all fields with valid numeric values. Error: " + str(e))
#             st.stop()

#         # --- compute engineered flags (re-initialized locally) ---
#         NewBMI_Overweight = 0
#         NewBMI_Underweight = 0
#         NewBMI_Obesity_1 = 0
#         NewBMI_Obesity_2 = 0 
#         NewBMI_Obesity_3 = 0
#         NewInsulinScore_Normal = 0 
#         NewGlucose_Low = 0
#         NewGlucose_Normal = 0 
#         NewGlucose_Overweight = 0
#         NewGlucose_Secret = 0

#         b = raw_BMI
#         if b <= 18.5:
#             NewBMI_Underweight = 1
#         elif 18.5 < b <= 24.9:
#             pass
#         elif 24.9 < b <= 29.9:
#             NewBMI_Overweight = 1
#         elif 29.9 < b <= 34.9:
#             NewBMI_Obesity_1 = 1
#         elif 34.9 < b <= 39.9:
#             NewBMI_Obesity_2 = 1
#         else:
#             NewBMI_Obesity_3 = 1

#         if 16 <= raw_Insulin <= 166:
#             NewInsulinScore_Normal = 1

#         g = raw_Glucose
#         if g <= 70:
#             NewGlucose_Low = 1
#         elif 70 < g <= 99:
#             NewGlucose_Normal = 1
#         elif 99 < g <= 126:
#             NewGlucose_Overweight = 1
#         else:
#             NewGlucose_Secret = 1

#         # --- Raw 8-feature vector (likely what the trained model expects) ---
#         raw_features = [
#             raw_Pregnancies, raw_Glucose, raw_BloodPressure,
#             raw_SkinThickness, raw_Insulin, raw_BMI,
#             raw_DPF, raw_Age
#         ]

#         # --- Engineered flags (10) appended form a full 18-feature vector ---
#         engineered_flags = [
#             NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
#             NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
#             NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
#             NewGlucose_Secret
#         ]
#         full_input = raw_features + engineered_flags

#         # --- Choose which to pass to the model depending on model expectation ---
#         import numpy as np
#         expected = getattr(diabetes_model, "n_features_in__", None)

#         try:
#             if expected is None:
#                 # Unknown: try raw first (safe), fallback to full
#                 X_try = np.array(raw_features, dtype=float).reshape(1, -1)
#                 try:
#                     pred = diabetes_model.predict(X_try)
#                 except Exception:
#                     X_try2 = np.array(full_input, dtype=float).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try2)
#             else:
#                 if expected == len(raw_features):
#                     X_try = np.array(raw_features, dtype=float).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 elif expected == len(full_input):
#                     X_try = np.array(full_input, dtype=float).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 else:
#                     # Clear actionable error for you to fix (retrain or change inputs)
#                     raise ValueError(
#                         f"Model expects {expected} features. Raw features = {len(raw_features)}, "
#                         f"Full (raw+engineered) = {len(full_input)}. "
#                         "Either retrain the model with engineered features or pass only the raw 8 features."
#                     )
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         # --- Interpret prediction ---
#         if pred[0] == 1:
#             diabetes_result = "The person has diabetic"
#         else:
#             diabetes_result = "The person has no diabetic"

#     st.success(diabetes_result)

# if selected == 'Heart Disease Prediction':
#     st.title("Heart Disease Prediction Using Machine Learning")
#     col1, col2, col3  = st.columns(3)

#     with col1:
#         age = st.text_input("Age")
#     with col2:
#         sex = st.text_input("Sex")
#     with col3:
#         cp = st.text_input("Chest Pain Types")
#     with col1:
#         trtbps = st.text_input("Resting Blood Pressure")
#     with col2:
#         chol = st.text_input("Serum Cholestroal in mg/dl")
#     with col3:
#         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
#     with col1:
#         restecg = st.text_input('Resting Electrocardiographic results')

#     with col2:
#         thalachh = st.text_input('Maximum Heart Rate achieved')

#     with col3:
#         exng = st.text_input('Exercise Induced Angina')

#     with col1:
#         oldpeak = st.text_input('ST depression induced by exercise')

#     with col2:
#         slp = st.text_input('Slope of the peak exercise ST segment')

#     with col3:
#         caa = st.text_input('Major vessels colored by flourosopy')

#     with col1:
#         thall = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
#     heart_disease_result = ""
#     if st.button("Heart Disease Test Result"):
#         user_input = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]
#         user_input = [float(x) for x in user_input]
#         prediction = heart_disease_model.predict([user_input])
#         if prediction[0]==1:
#             heart_disease_result = "This person is having heart disease"
#         else:
#             heart_disease_result = "This person does not have any heart disease"
#     st.success(heart_disease_result)

# if selected == 'Parkinsons Disease Prediction':
    
#     st.title("Parkinsons Disease Prediction using ML")

#     col1, col2, col3, col4= st.columns(4)

#     with col1:
#         MDVPFOHZ = st.text_input('MDVP:Fo(Hz)')

#     with col2:
#         MDVPFHIHZ = st.text_input('MDVP:Fhi(Hz)')

#     with col3:
#         MDVPFLOHZ = st.text_input('MDVP:Flo(Hz)')

#     with col4:
#         MDVPJITTERPERCENTAGE = st.text_input('MDVP:Jitter(%)')

#     with col1:
#         MDVPJITTERPERABS = st.text_input('MDVP:Jitter(Abs)')

#     with col2:
#         MDVPRAP = st.text_input('MDVP:RAP')

#     with col3:
#         MDVPPPQ = st.text_input('MDVP:PPQ')

#     with col4:
#         JITTERDDP = st.text_input('Jitter:DDP')

#     with col1:
#         MDVPSHIMMER = st.text_input('MDVP:Shimmer')

#     with col2:
#         MDVPSHIMMERDB = st.text_input('MDVP:Shimmer(dB)')

#     with col3:
#         SHIMMERAPQ3 = st.text_input('Shimmer:APQ3')

#     with col4:
#         SHIMMERAPQ5 = st.text_input('Shimmer:APQ5')

#     with col1:
#         MDVPAPQ = st.text_input('MDVP:APQ')

#     with col2:
#         SHIMMERDDA = st.text_input('Shimmer:DDA')

#     with col3:
#         NHR = st.text_input('NHR')

#     with col4:
#         HNR = st.text_input('HNR')

#     with col1:
#         RPDE = st.text_input('RPDE')

#     with col2:
#         DFA = st.text_input('DFA')

#     with col3:
#         SPREAD1 = st.text_input('spread1')

#     with col4:
#         SPREAD2 = st.text_input('spread2')

#     with col1:
#         D2 = st.text_input('D2')

#     with col2:
#         PPE = st.text_input('PPE')

#     parkinsons_result = ""

#     if st.button("Parkinsons's Test Result"):
#         user_input = [
#             MDVPFOHZ, MDVPFHIHZ, MDVPFLOHZ, MDVPJITTERPERCENTAGE,
#             MDVPJITTERPERABS, MDVPRAP, MDVPPPQ, JITTERDDP, MDVPSHIMMER,
#             MDVPSHIMMERDB, SHIMMERAPQ3, SHIMMERAPQ5, MDVPAPQ,
#             SHIMMERDDA, NHR, HNR, RPDE, DFA, SPREAD1, SPREAD2, D2, PPE
#         ]

#         user_input = [float(x) for x in user_input]
#         prediction = parkinsons_disease_model.predict([user_input])

#         if prediction[0] == 1:
#             parkinsons_result = "This person has Parkinsons disease"
#         else:
#             parkinsons_result = "This person does not have Parkinsons disease"

#     st.success(parkinsons_result)



# app.py
# import os
# import json
# import pickle
# import numpy as np
# import streamlit as st
# import streamlit.components.v1 as components
# from streamlit_option_menu import option_menu

# # ------------------ Page config ------------------
# st.set_page_config(
#     page_title="Multi-Disease Prediction",
#     page_icon="❤️🧠🩺",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ------------------ Styling (small custom CSS) ------------------
# st.markdown(
#     """
#     <style>
#     /* page background & centering */
#     .stApp { background: linear-gradient(180deg,#f7fbff 0%, #ffffff 60%); }
#     .header {
#         display:flex; align-items:center; gap:12px;
#     }
#     .app-title { font-size:28px; font-weight:700; margin:0; }
#     .app-sub { color:#666; margin:0; font-size:13px; }
#     .card {
#         background: white;
#         border-radius:14px;
#         padding:18px;
#         box-shadow: 0 6px 18px rgba(20,30,60,0.06);
#         margin-bottom: 12px;
#     }
#     .small { font-size:12px; color:#666; }
#     .input-label { font-weight:600; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ------------------ Helper: speak + tone using client JS ------------------
# def speak_with_tone(message: str, tone: str = "neutral"):
#     """
#     Play short tone then use browser TTS to speak `message`.
#     tone: 'alert' (positive/disease), 'ok' (negative/no disease), 'neutral' (no tone)
#     """
#     data = {"message": message, "tone": tone}
#     js = f"""
#     <script>
#     (function() {{
#         try {{
#             const payload = {json.dumps(data)};
#             const msgText = payload.message || "";
#             const tone = payload.tone || "neutral";

#             function playTone(toneName) {{
#                 if (!window.AudioContext && !window.webkitAudioContext) return Promise.resolve();
#                 const AudioCtx = window.AudioContext || window.webkitAudioContext;
#                 const ctx = new AudioCtx();
#                 const o = ctx.createOscillator();
#                 const g = ctx.createGain();
#                 o.connect(g);
#                 g.connect(ctx.destination);

#                 if (toneName === "alert") {{
#                     o.frequency.value = 880;
#                     g.gain.value = 0.02;
#                     o.type = "sawtooth";
#                 }} else if (toneName === "ok") {{
#                     o.frequency.value = 520;
#                     g.gain.value = 0.01;
#                     o.type = "sine";
#                 }} else {{
#                     return Promise.resolve();
#                 }}

#                 return new Promise((resolve) => {{
#                     o.start();
#                     setTimeout(() => {{
#                         o.stop();
#                         ctx.close();
#                         resolve();
#                     }}, 220);
#                 }});
#             }}

#             function speakText(text) {{
#                 if (!("speechSynthesis" in window)) return;
#                 const utterance = new SpeechSynthesisUtterance(text);
#                 utterance.rate = 1;
#                 // utterance.lang = 'en-US';
#                 window.speechSynthesis.cancel();
#                 window.speechSynthesis.speak(utterance);
#             }}

#             playTone(tone).then(() => {{
#                 setTimeout(() => speakText(msgText), 90);
#             }});
#         }} catch(e) {{ console.error(e); }}
#     }})();
#     </script>
#     """
#     # tiny height so it doesn't occupy visible space
#     components.html(js, height=0)


# # ------------------ Sidebar: menu and settings ------------------
# with st.sidebar:
#     selected = option_menu(
#         "Multi-Disease Prediction",
#         ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "About"],
#         icons=["activity", "heart", "person", "info-circle"],
#         menu_icon="hospital-fill",
#         default_index=0,
#     )
#     st.sidebar.markdown("---")
#     enable_sound = st.sidebar.checkbox("Enable sound / voice feedback", value=True)
#     st.sidebar.markdown("### App settings")
#     sample_theme = st.sidebar.radio("Theme", ["Light (default)", "Compact"], index=0)
#     st.sidebar.markdown(
#         "This app uses browser TTS (no server uploads). For best results use Chrome / Edge on desktop."
#     )


# # ------------------ Load models (safe) ------------------
# working_dir = os.path.dirname(os.path.abspath(__file__))

# def safe_load_model(path):
#     try:
#         return pickle.load(open(path, "rb"))
#     except Exception as e:
#         st.error(f"Failed to load model at {path}: {e}")
#         st.stop()

# # paths
# diabetes_model_path = os.path.join(working_dir, "notebook", "diabetes_model.pkl")
# heart_model_path = os.path.join(working_dir, "notebook", "heart_model.pkl")
# parkinsons_model_path = os.path.join(working_dir, "notebook", "parkinsons_model.pkl")

# diabetes_model = safe_load_model(diabetes_model_path)
# heart_disease_model = safe_load_model(heart_model_path)
# parkinsons_disease_model = safe_load_model(parkinsons_model_path)


# # ------------------ Header ------------------
# col_h1, col_h2 = st.columns([6, 1])
# with col_h1:
#     st.markdown(
#         """
#         <div class="header">
#           <div>
#             <h1 class="app-title">Multi-Disease Prediction</h1>
#             <p class="app-sub">Predict Diabetes, Heart Disease, and Parkinson's — with voice feedback & improved UI</p>
#           </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# with col_h2:
#     st.image("https://img.icons8.com/fluency/48/000000/medical-doctor.png", width=48)

# st.markdown("")  # spacing


# # ------------------ Utility: float conversion with error message ------------------
# def to_float_list(values, labels=None):
#     out = []
#     try:
#         for v in values:
#             out.append(float(v))
#     except Exception as e:
#         # helpful message for the user
#         if labels:
#             idx = len(out)
#             raise ValueError(f"Invalid value for '{labels[idx]}': '{values[idx]}' -- please enter a number.")
#         else:
#             raise ValueError(f"Invalid numeric input: {e}")
#     return out


# # ------------------ DIABETES UI & LOGIC ------------------
# if selected == "Diabetes Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Diabetes Prediction using ML")
#     st.markdown("<div class='small'>Enter patient details — inputs are validated before prediction.</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     with st.form(key="diabetes_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 2")
#             SkinThickness = st.text_input("SkinThickness", placeholder="e.g. 20")
#             DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.5")
#         with c2:
#             Glucose = st.text_input("Glucose Level", placeholder="e.g. 120")
#             Insulin = st.text_input("Insulin", placeholder="e.g. 85")
#             Age = st.text_input("Age", placeholder="e.g. 45")
#         with c3:
#             BloodPressure = st.text_input("Blood Pressure", placeholder="e.g. 70")
#             BMI = st.text_input("BMI", placeholder="e.g. 28.1")
#             st.markdown("**Flags**: Underweight / Overweight / Obesity categories are auto-derived.")

#         submit_diabetes = st.form_submit_button("Diabetes Test Result")

#     diabetes_result = ""
#     if submit_diabetes:
#         # validate and convert
#         try:
#             raw_vals = to_float_list(
#                 [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age],
#                 labels=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
#             )
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         # feature engineering flags
#         raw_Pregnancies, raw_Glucose, raw_BloodPressure, raw_SkinThickness, raw_Insulin, raw_BMI, raw_DPF, raw_Age = raw_vals

#         NewBMI_Overweight = 0
#         NewBMI_Underweight = 0
#         NewBMI_Obesity_1 = 0
#         NewBMI_Obesity_2 = 0
#         NewBMI_Obesity_3 = 0
#         NewInsulinScore_Normal = 0
#         NewGlucose_Low = 0
#         NewGlucose_Normal = 0
#         NewGlucose_Overweight = 0
#         NewGlucose_Secret = 0

#         b = raw_BMI
#         if b <= 18.5:
#             NewBMI_Underweight = 1
#         elif 18.5 < b <= 24.9:
#             pass
#         elif 24.9 < b <= 29.9:
#             NewBMI_Overweight = 1
#         elif 29.9 < b <= 34.9:
#             NewBMI_Obesity_1 = 1
#         elif 34.9 < b <= 39.9:
#             NewBMI_Obesity_2 = 1
#         else:
#             NewBMI_Obesity_3 = 1

#         if 16 <= raw_Insulin <= 166:
#             NewInsulinScore_Normal = 1

#         g = raw_Glucose
#         if g <= 70:
#             NewGlucose_Low = 1
#         elif 70 < g <= 99:
#             NewGlucose_Normal = 1
#         elif 99 < g <= 126:
#             NewGlucose_Overweight = 1
#         else:
#             NewGlucose_Secret = 1

#         raw_features = [
#             raw_Pregnancies, raw_Glucose, raw_BloodPressure,
#             raw_SkinThickness, raw_Insulin, raw_BMI,
#             raw_DPF, raw_Age
#         ]

#         engineered_flags = [
#             NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
#             NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
#             NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
#             NewGlucose_Secret
#         ]

#         full_input = raw_features + engineered_flags

#         # choose correct feature length
#         expected = getattr(diabetes_model, "n_features_in__", None)
#         try:
#             if expected is None:
#                 # try raw then full
#                 X_try = np.array(raw_features).reshape(1, -1)
#                 try:
#                     pred = diabetes_model.predict(X_try)
#                 except Exception:
#                     X_try2 = np.array(full_input).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try2)
#             else:
#                 if expected == len(raw_features):
#                     X_try = np.array(raw_features).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 elif expected == len(full_input):
#                     X_try = np.array(full_input).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 else:
#                     raise ValueError(
#                         f"Model expects {expected} features. Raw={len(raw_features)}, Full={len(full_input)}. "
#                         "Please retrain model or align feature inputs."
#                     )
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             diabetes_result = "The person has diabetes"
#             st.success(diabetes_result)
#             if enable_sound:
#                 speak_with_tone(diabetes_result, tone="alert")
#         else:
#             diabetes_result = "The person does NOT have diabetes"
#             st.success(diabetes_result)
#             if enable_sound:
#                 speak_with_tone(diabetes_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)  # close card


# # ------------------ HEART DISEASE UI & LOGIC ------------------
# if selected == "Heart Disease Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Heart Disease Prediction using ML")
#     st.markdown("<div class='small'>Fill in clinical features. Use numeric encoded categories where required (see placeholders).</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     with st.form("heart_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             age = st.text_input("Age", placeholder="e.g. 54")
#             trtbps = st.text_input("Resting Blood Pressure (trtbps)", placeholder="e.g. 130")
#             restecg = st.text_input("Resting ECG (0/1/2)", placeholder="0")
#             oldpeak = st.text_input("ST depression (oldpeak)", placeholder="e.g. 1.2")
#         with c2:
#             sex = st.text_input("Sex (1 = male, 0 = female)", placeholder="1")
#             chol = st.text_input("Cholesterol (mg/dl)", placeholder="e.g. 233")
#             thalachh = st.text_input("Max Heart Rate achieved (thalachh)", placeholder="150")
#             slp = st.text_input("Slope (0/1/2)", placeholder="1")
#         with c3:
#             cp = st.text_input("Chest Pain type (0-3)", placeholder="1")
#             fbs = st.text_input("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", placeholder="0")
#             exng = st.text_input("Exercise induced angina (1/0)", placeholder="0")
#             caa = st.text_input("Major vessels colored (0-3)", placeholder="0")
#             thall = st.text_input("thal (0 normal,1 fixed,2 reversible)", placeholder="1")
#         submit_heart = st.form_submit_button("Heart Disease Test Result")

#     heart_disease_result = ""
#     if submit_heart:
#         try:
#             inputs = to_float_list(
#                 [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall],
#                 labels=["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]
#             )
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         try:
#             pred = heart_disease_model.predict(np.array(inputs).reshape(1, -1))
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             heart_disease_result = "This person is having heart disease"
#             st.error(heart_disease_result)
#             if enable_sound:
#                 speak_with_tone(heart_disease_result, tone="alert")
#         else:
#             heart_disease_result = "This person does NOT have heart disease"
#             st.success(heart_disease_result)
#             if enable_sound:
#                 speak_with_tone(heart_disease_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)


# # ------------------ PARKINSONS UI & LOGIC ------------------
# if selected == "Parkinsons Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Parkinsons Disease Prediction using ML")
#     st.markdown("<div class='small'>Provide the voice / signal features. All numeric.</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     # We'll present inputs in a grid. For long lists, placeholders can guide the user.
#     with st.form("parkinsons_form"):
#         cols = st.columns(4)
#         names = [
#             ("MDVP:Fo(Hz)","MDVPFOHZ"), ("MDVP:Fhi(Hz)","MDVPFHIHZ"),
#             ("MDVP:Flo(Hz)","MDVPFLOHZ"), ("MDVP:Jitter(%)","MDVPJITTERPERCENTAGE"),
#             ("MDVP:Jitter(Abs)","MDVPJITTERPERABS"), ("MDVP:RAP","MDVPRAP"),
#             ("MDVP:PPQ","MDVPPPQ"), ("Jitter:DDP","JITTERDDP"),
#             ("MDVP:Shimmer","MDVPSHIMMER"), ("MDVP:Shimmer(dB)","MDVPSHIMMERDB"),
#             ("Shimmer:APQ3","SHIMMERAPQ3"), ("Shimmer:APQ5","SHIMMERAPQ5"),
#             ("MDVP:APQ","MDVPAPQ"), ("Shimmer:DDA","SHIMMERDDA"),
#             ("NHR","NHR"), ("HNR","HNR"),
#             ("RPDE","RPDE"), ("DFA","DFA"),
#             ("spread1","SPREAD1"), ("spread2","SPREAD2"),
#             ("D2","D2"), ("PPE","PPE")
#         ]
#         values = []
#         for i, (label, key) in enumerate(names):
#             col = cols[i % 4]
#             val = col.text_input(label, placeholder="e.g. 0.0", key=f"par_{i}")
#             values.append(val)

#         submit_parkinsons = st.form_submit_button("Parkinsons Test Result")

#     parkinsons_result = ""
#     if submit_parkinsons:
#         try:
#             vals = to_float_list(values, labels=[n[0] for n in names])
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         try:
#             pred = parkinsons_disease_model.predict(np.array(vals).reshape(1, -1))
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             parkinsons_result = "This person has Parkinsons disease"
#             st.error(parkinsons_result)
#             if enable_sound:
#                 speak_with_tone(parkinsons_result, tone="alert")
#         else:
#             parkinsons_result = "This person does NOT have Parkinsons disease"
#             st.success(parkinsons_result)
#             if enable_sound:
#                 speak_with_tone(parkinsons_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)


# # ------------------ ABOUT ------------------
# if selected == "About":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("About this app")
#     st.write(
#         """
#         - Predictions are produced by pre-trained models loaded from the `notebook/` folder.
#         - Voice and tone playback are performed **in the user's browser** (no audio uploads).
#         - If a prediction errors, check that you supplied the correct number / ordering of features matching your trained models.
#         """
#     )
#     st.markdown("**Developer tips**")
#     st.write(
#         """
#         1. If the diabetes model fails with a feature mismatch, either retrain with only the raw 8 features
#            or with the 18 features (raw + engineered).
#         2. Use the sidebar to disable sounds if you are in a quiet environment.
#         """
#     )
#     st.markdown("</div>", unsafe_allow_html=True)




# # app.py
# import os
# import json
# import pickle
# import numpy as np
# import streamlit as st
# import streamlit.components.v1 as components
# from streamlit_option_menu import option_menu

# # ------------------ Page config ------------------
# st.set_page_config(
#     page_title="Multi-Disease Prediction",
#     page_icon="❤️🧠🩺",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ------------------ Styling (updated CSS for readability) ------------------
# st.markdown(
#     """
#     <style>
#     /* page background & centering */
#     .stApp { background: linear-gradient(180deg,#f7fbff 0%, #ffffff 60%); color: #1f2937 !important; }

#     /* header */
#     .header { display:flex; align-items:center; gap:12px; }
#     .app-title { font-size:28px; font-weight:700; margin:0; color:#0f172a !important; opacity:1 !important; }
#     .app-sub { color:#334155; margin:0; font-size:13px; opacity:1 !important; }

#     /* card */
#     .card { background: white; border-radius:14px; padding:18px; box-shadow: 0 6px 18px rgba(20,30,60,0.06); margin-bottom: 12px; color:#0f172a !important; }
#     .small { font-size:12px; color:#475569 !important; }
#     .input-label { font-weight:600; color:#0f172a !important; }

#     /* input boxes */
#     .stTextInput > div > div > input, .stTextInput > div > div > textarea {
#         background: #f1f5f9;           /* light background for inputs */
#         color: #0f172a !important;       /* visible text */
#         border-radius: 8px;
#         padding: 10px;
#     }
#     /* placeholder color */
#     .stTextInput input::placeholder { color: #64748b !important; opacity: 1 !important; }

#     /* sidebar adjust */
#     .sidebar .stMarkdown p, .sidebar .stText { color: #cbd5e1 !important; }

#     /* make streamlit alerts fully visible even if theme changed */
#     .stAlert, .stAlert > div { color: #0f172a !important; background-color: transparent !important; }

#     /* style our custom result banners (used by show_result helper) */
#     .result-banner { border-radius: 8px; padding: 14px 18px; font-weight: 600; margin: 12px 0; color: #06253a; }
#     .result-ok { background: #d1fae5; border: 1px solid #86efac; }
#     .result-alert { background: #fee2e2; border: 1px solid #fca5a5; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ------------------ Helper: speak + tone using client JS ------------------
# def speak_with_tone(message: str, tone: str = "neutral"):
#     """
#     Play short tone then use browser TTS to speak `message`.
#     tone: 'alert' (positive/disease), 'ok' (negative/no disease), 'neutral' (no tone)
#     """
#     data = {"message": message, "tone": tone}
#     js = f"""
#     <script>
#     (function() {{
#         try {{
#             const payload = {json.dumps(data)};
#             const msgText = payload.message || "";
#             const tone = payload.tone || "neutral";

#             function unlockAudio() {{
#                 // Create a short silent buffer to ensure audio context can play after user gesture
#                 return new Promise((resolve) => {{
#                     try {{
#                         const AudioCtx = window.AudioContext || window.webkitAudioContext;
#                         if (!AudioCtx) return resolve();
#                         const ctx = new AudioCtx();
#                         const o = ctx.createOscillator();
#                         const g = ctx.createGain();
#                         o.connect(g); g.connect(ctx.destination);
#                         g.gain.value = 0; o.start(); setTimeout(() => {{ o.stop(); ctx.close(); resolve(); }}, 50);
#                     }} catch(e) {{ resolve(); }}
#                 }});
#             }}

#             function playTone(toneName) {{
#                 if (!window.AudioContext && !window.webkitAudioContext) return Promise.resolve();
#                 const AudioCtx = window.AudioContext || window.webkitAudioContext;
#                 const ctx = new AudioCtx();
#                 const o = ctx.createOscillator();
#                 const g = ctx.createGain();
#                 o.connect(g);
#                 g.connect(ctx.destination);

#                 if (toneName === "alert") {{
#                     o.frequency.value = 880;
#                     g.gain.value = 0.02;
#                     o.type = "sawtooth";
#                 }} else if (toneName === "ok") {{
#                     o.frequency.value = 520;
#                     g.gain.value = 0.01;
#                     o.type = "sine";
#                 }} else {{
#                     return Promise.resolve();
#                 }}

#                 return new Promise((resolve) => {{
#                     o.start();
#                     setTimeout(() => {{ o.stop(); ctx.close(); resolve(); }}, 220);
#                 }});
#             }}

#             function speakText(text) {{
#                 if (!("speechSynthesis" in window)) return;
#                 const utterance = new SpeechSynthesisUtterance(text);
#                 utterance.rate = 1;
#                 window.speechSynthesis.cancel();
#                 window.speechSynthesis.speak(utterance);
#             }}

#             // Sequence: unlock audio (no-op if already allowed), tone (if any), then TTS
#             unlockAudio().then(() => {{
#                 playTone(tone).then(() => {{ setTimeout(() => speakText(msgText), 90); }});
#             }});

#         }} catch(e) {{ console.error(e); }}
#     }})();
#     </script>
#     """
#     components.html(js, height=0)


# # ------------------ Helper: visible result banner ------------------
# def show_result(message: str, positive: bool):
#     """
#     Render a visible result banner. positive=True -> green banner, else red.
#     """
#     cls = "result-ok" if positive else "result-alert"
#     html = f'<div class="result-banner {cls}">{message}</div>'
#     st.markdown(html, unsafe_allow_html=True)


# # ------------------ Sidebar: menu and settings ------------------
# with st.sidebar:
#     selected = option_menu(
#         "Multi-Disease Prediction",
#         ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "About"],
#         icons=["activity", "heart", "person", "info-circle"],
#         menu_icon="hospital-fill",
#         default_index=0,
#     )
#     st.sidebar.markdown("---")
#     enable_sound = st.sidebar.checkbox("Enable sound / voice feedback", value=True)
#     st.sidebar.markdown("### App settings")
#     sample_theme = st.sidebar.radio("Theme", ["Light (default)", "Compact"], index=0)
#     st.sidebar.markdown(
#         "This app uses browser TTS (no server uploads). For best results use Chrome / Edge on desktop."
#     )


# # ------------------ Load models (safe) ------------------
# working_dir = os.path.dirname(os.path.abspath(__file__))


# def safe_load_model(path):
#     try:
#         return pickle.load(open(path, "rb"))
#     except Exception as e:
#         st.error(f"Failed to load model at {path}: {e}")
#         st.stop()


# # paths
# DIABETES_MODEL_PATH = os.path.join(working_dir, "notebook", "diabetes_model.pkl")
# HEART_MODEL_PATH = os.path.join(working_dir, "notebook", "heart_model.pkl")
# PARKINSONS_MODEL_PATH = os.path.join(working_dir, "notebook", "parkinsons_model.pkl")


# # load models
# if not os.path.exists(DIABETES_MODEL_PATH):
#     st.error(f"Missing file: {DIABETES_MODEL_PATH}")
#     st.stop()
# if not os.path.exists(HEART_MODEL_PATH):
#     st.error(f"Missing file: {HEART_MODEL_PATH}")
#     st.stop()
# if not os.path.exists(PARKINSONS_MODEL_PATH):
#     st.error(f"Missing file: {PARKINSONS_MODEL_PATH}")
#     st.stop()


# diabetes_model = safe_load_model(DIABETES_MODEL_PATH)
# heart_disease_model = safe_load_model(HEART_MODEL_PATH)
# parkinsons_disease_model = safe_load_model(PARKINSONS_MODEL_PATH)


# # ------------------ Header ------------------
# col_h1, col_h2 = st.columns([6, 1])
# with col_h1:
#     st.markdown(
#         """
#         <div class="header">
#           <div>
#             <h1 class="app-title">Multi-Disease Prediction</h1>
#             <p class="app-sub">Predict Diabetes, Heart Disease, and Parkinson's — with voice feedback & improved UI</p>
#           </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
# with col_h2:
#     st.image("https://img.icons8.com/fluency/48/000000/medical-doctor.png", width=48)

# st.markdown("")  # spacing


# # ------------------ Utility: float conversion with error message ------------------
# def to_float_list(values, labels=None):
#     out = []
#     try:
#         for v in values:
#             out.append(float(v))
#     except Exception as e:
#         if labels:
#             idx = len(out)
#             raise ValueError(f"Invalid value for '{labels[idx]}': '{values[idx]}' -- please enter a number.")
#         else:
#             raise ValueError(f"Invalid numeric input: {e}")
#     return out


# # ------------------ DIABETES UI & LOGIC ------------------
# if selected == "Diabetes Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Diabetes Prediction using ML")
#     st.markdown("<div class='small'>Enter patient details — inputs are validated before prediction.</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     with st.form(key="diabetes_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             Pregnancies = st.text_input("Number of Pregnancies", placeholder="e.g. 2")
#             SkinThickness = st.text_input("SkinThickness", placeholder="e.g. 20")
#             DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", placeholder="e.g. 0.5")
#         with c2:
#             Glucose = st.text_input("Glucose Level", placeholder="e.g. 120")
#             Insulin = st.text_input("Insulin", placeholder="e.g. 85")
#             Age = st.text_input("Age", placeholder="e.g. 45")
#         with c3:
#             BloodPressure = st.text_input("Blood Pressure", placeholder="e.g. 70")
#             BMI = st.text_input("BMI", placeholder="e.g. 28.1")
#             st.markdown("**Flags**: Underweight / Overweight / Obesity categories are auto-derived.")

#         submit_diabetes = st.form_submit_button("Diabetes Test Result")

#     diabetes_result = ""
#     if submit_diabetes:
#         try:
#             raw_vals = to_float_list(
#                 [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age],
#                 labels=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
#             )
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         raw_Pregnancies, raw_Glucose, raw_BloodPressure, raw_SkinThickness, raw_Insulin, raw_BMI, raw_DPF, raw_Age = raw_vals

#         NewBMI_Overweight = 0
#         NewBMI_Underweight = 0
#         NewBMI_Obesity_1 = 0
#         NewBMI_Obesity_2 = 0
#         NewBMI_Obesity_3 = 0
#         NewInsulinScore_Normal = 0
#         NewGlucose_Low = 0
#         NewGlucose_Normal = 0
#         NewGlucose_Overweight = 0
#         NewGlucose_Secret = 0

#         b = raw_BMI
#         if b <= 18.5:
#             NewBMI_Underweight = 1
#         elif 18.5 < b <= 24.9:
#             pass
#         elif 24.9 < b <= 29.9:
#             NewBMI_Overweight = 1
#         elif 29.9 < b <= 34.9:
#             NewBMI_Obesity_1 = 1
#         elif 34.9 < b <= 39.9:
#             NewBMI_Obesity_2 = 1
#         else:
#             NewBMI_Obesity_3 = 1

#         if 16 <= raw_Insulin <= 166:
#             NewInsulinScore_Normal = 1

#         g = raw_Glucose
#         if g <= 70:
#             NewGlucose_Low = 1
#         elif 70 < g <= 99:
#             NewGlucose_Normal = 1
#         elif 99 < g <= 126:
#             NewGlucose_Overweight = 1
#         else:
#             NewGlucose_Secret = 1

#         raw_features = [
#             raw_Pregnancies, raw_Glucose, raw_BloodPressure,
#             raw_SkinThickness, raw_Insulin, raw_BMI,
#             raw_DPF, raw_Age
#         ]

#         engineered_flags = [
#             NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
#             NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
#             NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
#             NewGlucose_Secret
#         ]

#         full_input = raw_features + engineered_flags

#         expected = getattr(diabetes_model, "n_features_in__", None)
#         try:
#             if expected is None:
#                 X_try = np.array(raw_features).reshape(1, -1)
#                 try:
#                     pred = diabetes_model.predict(X_try)
#                 except Exception:
#                     X_try2 = np.array(full_input).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try2)
#             else:
#                 if expected == len(raw_features):
#                     X_try = np.array(raw_features).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 elif expected == len(full_input):
#                     X_try = np.array(full_input).reshape(1, -1)
#                     pred = diabetes_model.predict(X_try)
#                 else:
#                     raise ValueError(
#                         f"Model expects {expected} features. Raw={len(raw_features)}, Full={len(full_input)}. "
#                         "Please retrain model or align feature inputs."
#                     )
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             diabetes_result = "The person has diabetes"
#             show_result(diabetes_result, positive=False)
#             if enable_sound:
#                 speak_with_tone(diabetes_result, tone="alert")
#         else:
#             diabetes_result = "The person does NOT have diabetes"
#             show_result(diabetes_result, positive=True)
#             if enable_sound:
#                 speak_with_tone(diabetes_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)


# # ------------------ HEART DISEASE UI & LOGIC ------------------
# if selected == "Heart Disease Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Heart Disease Prediction using ML")
#     st.markdown("<div class='small'>Fill in clinical features. Use numeric encoded categories where required (see placeholders).</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     with st.form("heart_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             age = st.text_input("Age", placeholder="e.g. 54")
#             trtbps = st.text_input("Resting Blood Pressure (trtbps)", placeholder="e.g. 130")
#             restecg = st.text_input("Resting ECG (0/1/2)", placeholder="0")
#             oldpeak = st.text_input("ST depression (oldpeak)", placeholder="e.g. 1.2")
#         with c2:
#             sex = st.text_input("Sex (1 = male, 0 = female)", placeholder="1")
#             chol = st.text_input("Cholesterol (mg/dl)", placeholder="e.g. 233")
#             thalachh = st.text_input("Max Heart Rate achieved (thalachh)", placeholder="150")
#             slp = st.text_input("Slope (0/1/2)", placeholder="1")
#         with c3:
#             cp = st.text_input("Chest Pain type (0-3)", placeholder="1")
#             fbs = st.text_input("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", placeholder="0")
#             exng = st.text_input("Exercise induced angina (1/0)", placeholder="0")
#             caa = st.text_input("Major vessels colored (0-3)", placeholder="0")
#             thall = st.text_input("thal (0 normal,1 fixed,2 reversible)", placeholder="1")
#         submit_heart = st.form_submit_button("Heart Disease Test Result")

#     heart_disease_result = ""
#     if submit_heart:
#         try:
#             inputs = to_float_list(
#                 [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall],
#                 labels=["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]
#             )
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         try:
#             pred = heart_disease_model.predict(np.array(inputs).reshape(1, -1))
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             heart_disease_result = "This person is having heart disease"
#             show_result(heart_disease_result, positive=False)
#             if enable_sound:
#                 speak_with_tone(heart_disease_result, tone="alert")
#         else:
#             heart_disease_result = "This person does NOT have heart disease"
#             show_result(heart_disease_result, positive=True)
#             if enable_sound:
#                 speak_with_tone(heart_disease_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)


# # ------------------ PARKINSONS UI & LOGIC ------------------
# if selected == "Parkinsons Prediction":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Parkinsons Disease Prediction using ML")
#     st.markdown("<div class='small'>Provide the voice / signal features. All numeric.</div>", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)

#     with st.form("parkinsons_form"):
#         cols = st.columns(4)
#         names = [
#             ("MDVP:Fo(Hz)","MDVPFOHZ"), ("MDVP:Fhi(Hz)","MDVPFHIHZ"),
#             ("MDVP:Flo(Hz)","MDVPFLOHZ"), ("MDVP:Jitter(%)","MDVPJITTERPERCENTAGE"),
#             ("MDVP:Jitter(Abs)","MDVPJITTERPERABS"), ("MDVP:RAP","MDVPRAP"),
#             ("MDVP:PPQ","MDVPPPQ"), ("Jitter:DDP","JITTERDDP"),
#             ("MDVP:Shimmer","MDVPSHIMMER"), ("MDVP:Shimmer(dB)","MDVPSHIMMERDB"),
#             ("Shimmer:APQ3","SHIMMERAPQ3"), ("Shimmer:APQ5","SHIMMERAPQ5"),
#             ("MDVP:APQ","MDVPAPQ"), ("Shimmer:DDA","SHIMMERDDA"),
#             ("NHR","NHR"), ("HNR","HNR"),
#             ("RPDE","RPDE"), ("DFA","DFA"),
#             ("spread1","SPREAD1"), ("spread2","SPREAD2"),
#             ("D2","D2"), ("PPE","PPE")
#         ]
#         values = []
#         for i, (label, key) in enumerate(names):
#             col = cols[i % 4]
#             val = col.text_input(label, placeholder="e.g. 0.0", key=f"par_{i}")
#             values.append(val)

#         submit_parkinsons = st.form_submit_button("Parkinsons Test Result")

#     parkinsons_result = ""
#     if submit_parkinsons:
#         try:
#             vals = to_float_list(values, labels=[n[0] for n in names])
#         except ValueError as e:
#             st.error(str(e))
#             st.stop()

#         try:
#             pred = parkinsons_disease_model.predict(np.array(vals).reshape(1, -1))
#         except Exception as e:
#             st.error("Prediction failed: " + str(e))
#             st.stop()

#         if pred[0] == 1:
#             parkinsons_result = "This person has Parkinsons disease"
#             show_result(parkinsons_result, positive=False)
#             if enable_sound:
#                 speak_with_tone(parkinsons_result, tone="alert")
#         else:
#             parkinsons_result = "This person does NOT have Parkinsons disease"
#             show_result(parkinsons_result, positive=True)
#             if enable_sound:
#                 speak_with_tone(parkinsons_result, tone="ok")

#     st.markdown("</div>", unsafe_allow_html=True)


# # ------------------ ABOUT ------------------
# if selected == "About":
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("About this app")
#     st.write(
#         """
#         - Predictions are produced by pre-trained models loaded from the `notebook/` folder.
#         - Voice and tone playback are performed **in the user's browser** (no audio uploads).
#         - If a prediction errors, check that you supplied the correct number / ordering of features matching your trained models.
#         """
#     )
#     st.markdown("**Developer tips**")
#     st.write(





# app.py - Multi-Disease Prediction (full working file)
# Features:
# - Login / Register / Profile (persistent users.json)
# - Models loaded from notebook/*.pkl
# - Highlighted homepage hero + auto-flash & scroll
# - Prediction result banner that scrolls into view + flash
# - Browser TTS via Web Speech API (toggleable)
# - Defensive input handling & diabetes 8/18 adapt
# - All prior fixes applied



#final h
# app.py - Fixed navigation + styled sidebar + full app
# import streamlit as st
# import pickle
# import os
# import json
# import hashlib
# import numpy as np
# import streamlit.components.v1 as components
# from streamlit_option_menu import option_menu
# import uuid

# st.set_page_config(page_title="Multi-Disease Prediction", layout="wide", page_icon="👨‍⚕️")

# working_dir = os.path.dirname(os.path.abspath(__file__))

# # ------------------ Load models (unchanged) ------------------
# diabetes_model = pickle.load(open(os.path.join(working_dir, "notebook", "diabetes_model.pkl"), "rb"))
# heart_disease_model = pickle.load(open(os.path.join(working_dir, "notebook", "heart_model.pkl"), "rb"))
# parkinsons_disease_model = pickle.load(open(os.path.join(working_dir, "notebook", "parkinsons_model.pkl"), "rb"))

# # ------------------ Small UI utilities ------------------
# def speak(text):
#     safe_text = str(text).replace("'", "\\'").replace("\n", " ")
#     js = f"""
#     <script>
#       const synth = window.speechSynthesis;
#       if (synth) {{
#         const utter = new SpeechSynthesisUtterance('{safe_text}');
#         utter.lang = 'en-US';
#         utter.rate = 1;
#         utter.pitch = 1;
#         synth.cancel();
#         synth.speak(utter);
#       }}
#     </script>
#     """
#     components.html(js, height=0)

# def show_home_hero_and_scroll():
#     hero_id = "home_hero_" + str(uuid.uuid4()).replace("-", "_")
#     html = f"""
#     <div id="{hero_id}" style="
#         width:100%;
#         border-radius:12px;
#         padding:22px 24px;
#         margin-bottom:18px;
#         background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
#         box-shadow: 0 10px 30px rgba(0,0,0,0.12);
#       ">
#       <div style="font-size:28px; font-weight:800; color:#fff; margin-bottom:6px;">
#         Multi-Disease Prediction
#       </div>
#       <div style="font-size:13px; color:#cfcfcf;">
#         Predict Diabetes, Heart Disease and Parkinson's — with voice feedback & improved UI
#       </div>
#     </div>
#     <script>
#       (function(){{
#         const el = document.getElementById("{hero_id}");
#         if (el) {{
#           el.scrollIntoView({{behavior: "smooth", block: "start"}});
#           const orig = el.style.boxShadow;
#           let i = 0;
#           const t = setInterval(() => {{
#             el.style.boxShadow = (i % 2 === 0) ? "0 0 0 10px rgba(0,150,136,0.08)" : orig;
#             i++;
#             if (i > 6) {{ clearInterval(t); el.style.boxShadow = orig; }}
#           }}, 200);
#         }}
#       }})();
#     </script>
#     """
#     components.html(html, height=120)

# def show_highlighted_result_and_scroll(text, positive=True):
#     color = "#1b6b3a" if positive else "#a00000"
#     banner_id = "pred_banner_" + str(uuid.uuid4()).replace("-", "_")
#     html = f"""
#     <div id="{banner_id}" style="
#       border-radius:10px;
#       padding:14px;
#       margin:10px 0 22px 0;
#       display:flex;
#       align-items:center;
#       gap:18px;
#       background: rgba(255,255,255,0.01);
#     ">
#       <div style="flex:1">
#         <div style="font-size:18px; font-weight:700; color:{color}; margin-bottom:6px;">Prediction Result</div>
#         <div style="font-size:15px; color:#e9ecef;">{text}</div>
#       </div>
#       <div style="background:{color}; color:white; padding:10px 14px; border-radius:8px; font-weight:700;">
#         {text}
#       </div>
#     </div>
#     <script>
#       (function(){{
#         const el = document.getElementById("{banner_id}");
#         if (el){{
#           el.scrollIntoView({{behavior: "smooth", block: "center"}});
#           const orig = el.style.boxShadow;
#           let i = 0;
#           const t = setInterval(() => {{
#             el.style.boxShadow = (i % 2 === 0) ? "0 0 0 8px rgba(255,255,0,0.08)" : orig;
#             i++;
#             if (i > 6) {{ clearInterval(t); el.style.boxShadow = orig; }}
#           }}, 180);
#         }}
#       }})();
#     </script>
#     """
#     components.html(html, height=120)

# # ------------------ persistent users ------------------
# USER_DB_FILE = os.path.join(working_dir, "users.json")
# _default_demo_users = {
#     "admin": {"password_hash": hashlib.sha256("admin123".encode()).hexdigest(), "name":"Administrator","email":"admin@example.com"},
#     "vivek": {"password_hash": hashlib.sha256("vivek123".encode()).hexdigest(), "name":"Vivek Kumar","email":"vivek@example.com"}
# }
# def load_users():
#     if os.path.exists(USER_DB_FILE):
#         try:
#             with open(USER_DB_FILE, "r") as f:
#                 return json.load(f)
#         except Exception:
#             return dict(_default_demo_users)
#     else:
#         return dict(_default_demo_users)
# def save_users(users):
#     try:
#         with open(USER_DB_FILE, "w") as f:
#             json.dump(users, f, indent=2)
#     except Exception:
#         pass

# _users = load_users()
# def canonical_username(u): return u.strip().lower() if isinstance(u, str) else u
# def verify_user(username, password):
#     key = canonical_username(username)
#     user = _users.get(key)
#     if not user: return False
#     return user["password_hash"] == hashlib.sha256(password.encode()).hexdigest()
# def create_user(username, password, name="", email=""):
#     key = canonical_username(username)
#     if not username or not password: return False, "Username and password required"
#     if key in _users: return False, "Username already exists"
#     _users[key] = {"password_hash": hashlib.sha256(password.encode()).hexdigest(), "name":name, "email":email}
#     save_users(_users); return True, "User created"
# def get_user_profile(username): return _users.get(canonical_username(username), {})
# def update_user_profile(username, name="", email=""):
#     key = canonical_username(username)
#     if key in _users:
#         _users[key]["name"]=name; _users[key]["email"]=email; save_users(_users); return True
#     return False

# # ------------------ session defaults ------------------
# if "logged_in" not in st.session_state: st.session_state.logged_in = False
# if "user" not in st.session_state: st.session_state.user = None
# if "profile" not in st.session_state: st.session_state.profile = {"name":"","email":""}
# if "enable_tts" not in st.session_state: st.session_state.enable_tts = True

# # ------------------ Sidebar styling (inject CSS) ------------------
# st.markdown(
#     """
#     <style>
#     /* Sidebar background & rounded container */
#     section[data-testid="stSidebar"] > div:first-child {
#       background: linear-gradient(180deg, #222226, #1f1f23);
#       border-top-right-radius: 14px;
#       border-bottom-right-radius: 14px;
#       padding: 18px;
#     }
#     /* Make menu blocks stand out */
#     .sidebar-menu-card {
#       background:#111215;
#       border-radius:10px;
#       padding:10px;
#       margin-bottom:18px;
#       box-shadow: 0 8px 30px rgba(0,0,0,0.35);
#       border: 1px solid rgba(255,255,255,0.02);
#     }
#     /* Style the option_menu active item (works with streamlit-option-menu) */
#     .option-menu .nav-link.active {
#       background: linear-gradient(90deg, #ff6b6b, #ff4d4d);
#       color: white !important;
#       border-radius: 8px;
#       box-shadow: 0 6px 18px rgba(0,0,0,0.25);
#     }
#     /* Tidy up small text */
#     .sidebar .stTextInput>div>label { color: #d1d1d1; }
#     /* Small padding for the bottom area */
#     section[data-testid="stSidebar"] .sb-footer { margin-top: 22px; color:#bdbdbd; font-size:13px; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ------------------ Sidebar content (MENU FIRST) ------------------
# with st.sidebar:
#     # Put menu first to avoid interference from other widgets above it
#     selected = option_menu(
#         menu_title=None,
#         options=["Home", "Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Disease Prediction", "Profile", "About"],
#         icons=["house", "activity", "heart", "person", "person-circle", "info-circle"],
#         menu_icon="cast",
#         default_index=0,
#         orientation="vertical",
#         styles={
#             "container": {"padding": "0!important"},
#             "nav-link": {"font-size": "15px", "text-align": "left", "margin":"4px 0"},
#             "nav-link-selected": {"background":"#ff6b6b", "color":"white"}
#         }
#     )

#     # Decorative boxed section for account & nav helper
#     st.markdown('<div class="sidebar-menu-card">', unsafe_allow_html=True)

#     # Account area
#     if st.session_state.logged_in:
#         st.markdown(f"**Logged in as:** {st.session_state.user}")
#         if st.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.user = None
#             st.session_state.profile = {"name":"","email":""}
#             st.success("Logged out")
#     else:
#         st.markdown("**Account**")
#         acc_choice = st.radio("Choose", ("Login", "Register"), index=0, horizontal=False)
#         if acc_choice == "Login":
#             user_in = st.text_input("Username", key="login_username")
#             pw_in = st.text_input("Password", type="password", key="login_password")
#             if st.button("Sign in"):
#                 if verify_user(user_in, pw_in):
#                     st.session_state.logged_in = True
#                     st.session_state.user = canonical_username(user_in)
#                     data = get_user_profile(user_in)
#                     st.session_state.profile = {"name": data.get("name",""), "email": data.get("email","")}
#                     st.success(f"Welcome, {st.session_state.profile.get('name') or st.session_state.user}!")
#                 else:
#                     st.error("Invalid username or password")
#         else:
#             new_user = st.text_input("Choose Username", key="reg_user")
#             new_pw = st.text_input("Choose Password", type="password", key="reg_pw")
#             new_name = st.text_input("Full name (optional)", key="reg_name")
#             new_email = st.text_input("Email (optional)", key="reg_email")
#             if st.button("Create account"):
#                 ok,msg = create_user(new_user,new_pw,new_name,new_email)
#                 if ok: st.success("Account created. Please log in.")
#                 else: st.error(msg)

#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown('<div class="sidebar-menu-card">', unsafe_allow_html=True)
#     st.write("Enable sound / voice feedback")
#     # widget key "enable_tts" managed by widget itself
#     st.checkbox("Enable sound / voice feedback", key="enable_tts", value=st.session_state.get("enable_tts", True))
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown('<div class="sb-footer">This app uses browser TTS (client-side). For best results use Chrome/Edge.</div>', unsafe_allow_html=True)

# # ------------------ helper to require login ------------------
# def ensure_logged_in():
#     if not st.session_state.logged_in:
#         st.warning("Please log in to use prediction features. Use the sidebar Account -> Login.")
#         st.stop()

# # ------------------ Pages ------------------
# if selected == "Home":
#     show_home_hero_and_scroll()
#     st.write("## Welcome")
#     st.write("Use the sidebar to choose a prediction page. Demo accounts: admin/admin123, vivek/vivek123.")

# elif selected == "Diabetes Prediction":
#     ensure_logged_in()
#     show_home_hero_and_scroll()
#     st.markdown("<h1 style='color:white'>Diabetes Prediction Using Machine Learning</h1>", unsafe_allow_html=True)
#     st.write("Fill in clinical features. Placeholders show example encodings/units.")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         Pregnancies = st.text_input("Number of Pregnancies", placeholder="Enter number of pregnancies (e.g. 2)")
#     with col2:
#         Glucose = st.text_input("Glucose Level", placeholder="Enter glucose level (mg/dl) e.g. 120")
#     with col3:
#         BloodPressure = st.text_input("BloodPressure Value", placeholder="Enter blood pressure (mm Hg) e.g. 70")
#     with col1:
#         SkinThickness = st.text_input("SkinThickness Value", placeholder="Enter skin thickness (mm) e.g. 20")
#     with col2:
#         Insulin = st.text_input("Insulin Value", placeholder="Enter insulin level (IU) e.g. 80")
#     with col3:
#         BMI = st.text_input("BMI Value", placeholder="Enter BMI (kg/m2) e.g. 28.5")
#     with col1:
#         DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value", placeholder="Enter DPF (e.g. 0.5)")
#     with col2:
#         Age = st.text_input("Age", placeholder="Enter age in years (e.g. 45)")

#     if st.button("Diabetes Test Result"):
#         try:
#             raw_Pregnancies = float(Pregnancies); raw_Glucose = float(Glucose)
#             raw_BloodPressure = float(BloodPressure); raw_SkinThickness = float(SkinThickness)
#             raw_Insulin = float(Insulin); raw_BMI = float(BMI)
#             raw_DPF = float(DiabetesPedigreeFunction); raw_Age = float(Age)
#         except Exception as e:
#             st.error("Please fill all diabetes fields with valid numeric values. Error: " + str(e))
#             st.stop()

#         # compute flags
#         NewBMI_Overweight = NewBMI_Underweight = NewBMI_Obesity_1 = NewBMI_Obesity_2 = NewBMI_Obesity_3 = 0
#         NewInsulinScore_Normal = NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0
#         b = raw_BMI
#         if b <= 18.5: NewBMI_Underweight = 1
#         elif 18.5 < b <= 24.9: pass
#         elif 24.9 < b <= 29.9: NewBMI_Overweight = 1
#         elif 29.9 < b <= 34.9: NewBMI_Obesity_1 = 1
#         elif 34.9 < b <= 39.9: NewBMI_Obesity_2 = 1
#         else: NewBMI_Obesity_3 = 1

#         if 16 <= raw_Insulin <= 166: NewInsulinScore_Normal = 1
#         g = raw_Glucose
#         if g <= 70: NewGlucose_Low = 1
#         elif 70 < g <= 99: NewGlucose_Normal = 1
#         elif 99 < g <= 126: NewGlucose_Overweight = 1
#         else: NewGlucose_Secret = 1

#         raw_features = [raw_Pregnancies, raw_Glucose, raw_BloodPressure, raw_SkinThickness, raw_Insulin, raw_BMI, raw_DPF, raw_Age]
#         engineered_flags = [NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1, NewBMI_Obesity_2, NewBMI_Obesity_3,
#                             NewInsulinScore_Normal, NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret]
#         full_input = raw_features + engineered_flags

#         expected = getattr(diabetes_model, "n_features_in__", None)
#         try:
#             if expected is None:
#                 X_try = np.array(raw_features).reshape(1,-1)
#                 try:
#                     pred = diabetes_model.predict(X_try)
#                 except Exception:
#                     X_try = np.array(full_input).reshape(1,-1)
#                     pred = diabetes_model.predict(X_try)
#             else:
#                 if expected == len(raw_features):
#                     X_try = np.array(raw_features).reshape(1,-1); pred = diabetes_model.predict(X_try)
#                 elif expected == len(full_input):
#                     X_try = np.array(full_input).reshape(1,-1); pred = diabetes_model.predict(X_try)
#                 else:
#                     raise ValueError(f"Model expects {expected} features. Raw={len(raw_features)}, Full={len(full_input)}.")
#         except Exception as e:
#             st.error("Prediction failed: " + str(e)); st.stop()

#         if pred[0]==1:
#             res="The person has diabetes"; positive=True
#         else:
#             res="The person does not have diabetes"; positive=False

#         show_highlighted_result_and_scroll(res, positive=positive)
#         if st.session_state.get("enable_tts", True): speak(res)

# elif selected == "Heart Disease Prediction":
#     ensure_logged_in()
#     show_home_hero_and_scroll()
#     st.markdown("<h1 style='color:white'>Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
#     st.write("Fill in clinical features. Use numeric encoding for categorical placeholders.")
#     col1, col2, col3 = st.columns(3)
#     with col1: age = st.text_input("Age", placeholder="Enter age in years e.g. 54")
#     with col2: sex = st.text_input("Sex", placeholder="Enter 1=male 0=female")
#     with col3: cp = st.text_input("Chest Pain Types", placeholder="Enter chest pain type code (e.g. 1)")
#     with col1: trtbps = st.text_input("Resting Blood Pressure", placeholder="Enter resting BP (mm Hg) e.g. 130")
#     with col2: chol = st.text_input("Serum Cholestroal", placeholder="Enter cholesterol mg/dl e.g. 233")
#     with col3: fbs = st.text_input('Fasting Blood Sugar >120', placeholder="Enter 1 if true else 0")
#     with col1: restecg = st.text_input('Resting ECG', placeholder="Enter 0/1/2")
#     with col2: thalachh = st.text_input('Max Heart Rate (thalach)', placeholder="Enter e.g. 150")
#     with col3: exng = st.text_input('Exercise Induced Angina', placeholder="Enter 1 if yes else 0")
#     with col1: oldpeak = st.text_input('ST depression (oldpeak)', placeholder="Enter float e.g. 1.2")
#     with col2: slp = st.text_input('Slope', placeholder="Enter 0/1/2")
#     with col3: caa = st.text_input('Major vessels colored (0-3)', placeholder="Enter 0-3")
#     with col1: thall = st.text_input('thal (0 normal;1 fixed;2 reversible)', placeholder="Enter 0-2")

#     if st.button("Heart Disease Test Result"):
#         try:
#             vals = [float(x) for x in [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]]
#         except Exception as e:
#             st.error("Please fill all heart fields with valid numeric values. Error: " + str(e)); st.stop()
#         Xh = np.array(vals).reshape(1,-1)
#         expected_h = getattr(heart_disease_model,"n_features_in__",None)
#         try:
#             if expected_h is not None and Xh.shape[1] != expected_h:
#                 raise ValueError(f"Model expects {expected_h} features but input has {Xh.shape[1]}.")
#             pred_h = heart_disease_model.predict(Xh)
#         except Exception as e:
#             st.error("Prediction failed: " + str(e)); st.stop()
#         if pred_h[0]==1:
#             res="This person is having heart disease"; pos=True
#         else:
#             res="This person does not have any heart disease"; pos=False
#         show_highlighted_result_and_scroll(res, positive=pos)
#         if st.session_state.get("enable_tts", True): speak(res)

# elif selected == "Parkinsons Disease Prediction":
#     ensure_logged_in()
#     show_home_hero_and_scroll()
#     st.markdown("<h1 style='color:white'>Parkinsons Disease Prediction using ML</h1>", unsafe_allow_html=True)
#     cols = st.columns(4)
#     inputs = []
#     labels = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP",
#               "MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]
#     # draw the 22 inputs in 4 columns roughly
#     i = 0
#     for label in labels:
#         col = cols[i % 4]
#         inputs.append(col.text_input(label, placeholder=f"Enter {label}"))
#         i += 1

#     if st.button("Parkinsons's Test Result"):
#         try:
#             numeric_vals = [float(x) for x in inputs]
#         except Exception as e:
#             st.error("Please fill all Parkinsons fields with valid numeric values. Error: " + str(e)); st.stop()
#         Xp = np.array(numeric_vals).reshape(1,-1)
#         expected_p = getattr(parkinsons_disease_model, "n_features_in__", None)
#         try:
#             if expected_p is not None and Xp.shape[1] != expected_p:
#                 raise ValueError(f"Model expects {expected_p} features but input has {Xp.shape[1]}.")
#             pred_p = parkinsons_disease_model.predict(Xp)
#         except Exception as e:
#             st.error("Prediction failed: " + str(e)); st.stop()
#         if pred_p[0] == 1:
#             res="This person has Parkinsons disease"; pos=True
#         else:
#             res="This person does not have Parkinsons disease"; pos=False
#         show_highlighted_result_and_scroll(res, positive=pos)
#         if st.session_state.get("enable_tts", True): speak(res)

# elif selected == "Profile":
#     if not st.session_state.logged_in:
#         st.warning("You are not logged in. Please login from the sidebar to view/edit profile.")
#     else:
#         st.title("Your Profile")
#         name = st.text_input("Full name", value=st.session_state.profile.get("name",""), key="profile_name")
#         email = st.text_input("Email", value=st.session_state.profile.get("email",""), key="profile_email")
#         if st.button("Save profile"):
#             st.session_state.profile["name"] = name
#             st.session_state.profile["email"] = email
#             update_user_profile(st.session_state.user, name=name, email=email)
#             st.success("Profile updated")
#         st.write("Current session user:", st.session_state.user)

# elif selected == "About":
#     st.title("About")
#     st.write("This app predicts Diabetes, Heart Disease and Parkinson's using pre-trained models.")
#     st.write("For production, replace JSON/in-memory storage with a secure DB and stronger password hashing.")

# app.py
# Multi-Disease Prediction (enhanced)
# - Now imports admin_dashboard.show_admin_dashboard from admin_dashboard.py
# - Language selector shows English/Hindi
# - All other features retained (PDF/TXT reports, history, TTS, models auto-discovery)
# activate your venv first (Windows example)
# .venv\Scripts\activate



import streamlit as st
import pickle
import os
import json
import hashlib
import numpy as np
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from datetime import datetime
import uuid
import io

# Try reportlab for PDF generation (optional). If not available, fallback to TXT.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Import admin dashboard function (separate file)
from admin_dashboard import show_admin_dashboard

st.set_page_config(page_title="Multi-Disease Prediction", layout="wide", page_icon="👨‍⚕️", initial_sidebar_state="expanded")

working_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(working_dir, "notebook")
USER_DB_FILE = os.path.join(working_dir, "users.json")

# ---------------------- Translations (multi-language) ----------------------
TRANSLATIONS = {
    "en": {
        "app_title": "Multi-Disease Prediction",
        "app_sub": "Predict Diabetes, Heart Disease, and Parkinson's — with voice feedback & improved UI",
        "login": "Login",
        "register": "Register",
        "username": "Username",
        "password": "Password",
        "sign_in": "Sign in",
        "create_account": "Create account",
        "logout": "Logout",
        "profile": "Profile",
        "about": "About",
        "home": "Home",
        "diabetes": "Diabetes Prediction",
        "heart": "Heart Disease Prediction",
        "parkinsons": "Parkinsons Disease Prediction",
        "admin_dashboard": "Admin Dashboard",
        "enable_tts": "Enable sound / voice feedback",
        "download_report": "Download Report (PDF/TXT)",
        "generate_report": "Generate Report",
        "prediction_history": "Prediction History",
        "confidence": "Confidence",
        "no_model_proba": "N/A",
        "report_title": "Prediction Report",
        "timestamp": "Timestamp",
        "inputs": "Inputs",
        "result": "Result",
        "model": "Model",
        "ok": "OK",
        "error_fill": "Please fill fields with valid numeric values. Error: ",
        "select_model": "Select disease / model",
        "admin_note": "Admin (view user stats & history)"
    },
    "hi": {
        "app_title": "मल्टी-डिजीज़ प्रेडिक्शन",
        "app_sub": "हृदय रोग, डायबिटीज़ और पार्किंसन का अनुमान — वॉइस फीडबैक व बेहतर UI के साथ",
        "login": "लॉगिन",
        "register": "रजिस्टर",
        "username": "यूज़रनेम",
        "password": "पासवर्ड",
        "sign_in": "साइन इन",
        "create_account": "खाता बनाएँ",
        "logout": "लॉग आउट",
        "profile": "प्रोफ़ाइल",
        "about": "बारे में",
        "home": "होम",
        "diabetes": "डायबिटीज़ प्रेडिक्शन",
        "heart": "हृदय रोग प्रेडिक्शन",
        "parkinsons": "पार्किंसन प्रेडिक्शन",
        "admin_dashboard": "एडमिन डैशबोर्ड",
        "enable_tts": "साउंड / वॉइस फीडबैक सक्षम करें",
        "download_report": "रिपोर्ट डाउनलोड करें (PDF/TXT)",
        "generate_report": "रिपोर्ट बनाएं",
        "prediction_history": "पूर्वानुमान इतिहास",
        "confidence": "विश्वास (Confidence)",
        "no_model_proba": "एन/ए",
        "report_title": "प्रेडिक्शन रिपोर्ट",
        "timestamp": "समय",
        "inputs": "इनपुट्स",
        "result": "परिणाम",
        "model": "मॉडल",
        "ok": "ठीक है",
        "error_fill": "कृपया मान्य संख्यात्मक मान भरें। त्रुटि: ",
        "select_model": "रोग/मॉडल चुनें",
        "admin_note": "एडमिन (यूज़र आँकड़े देखें)"
    }
}

# language helper
if "lang" not in st.session_state:
    st.session_state.lang = "en"
def tr(key):
    return TRANSLATIONS.get(st.session_state.get("lang","en"), TRANSLATIONS["en"]).get(key, key)

# ---------------------- small UI helpers (TTS + banners) ----------------------
def speak(text):
    safe_text = str(text).replace("'", "\\'").replace("\n", " ")
    js = f"""
    <script>
    const synth = window.speechSynthesis;
    if (synth) {{
        const utter = new SpeechSynthesisUtterance('{safe_text}');
        utter.lang = 'en-US';
        utter.rate = 1;
        utter.pitch = 1;
        synth.cancel();
        synth.speak(utter);
    }}
    </script>
    """
    components.html(js, height=0)

def show_home_hero_and_scroll():
    hero_id = "home_hero_" + str(uuid.uuid4()).replace("-", "_")
    html = f"""
    <div id="{hero_id}" style="
        width:100%;
        border-radius:12px;
        padding:24px;
        margin-bottom:18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
        box-shadow: 0 12px 36px rgba(0,0,0,0.12);
      ">
      <div style="font-size:34px; font-weight:800; color:#fff; margin-bottom:6px;">
        {tr('app_title')}
      </div>
      <div style="font-size:15px; color:#cfcfcf;">
        {tr('app_sub')}
      </div>
    </div>
    <script>
      (function(){{
        const el = document.getElementById("{hero_id}");
        if (el){{
          el.scrollIntoView({{behavior: "smooth", block: "start"}});
          const orig = el.style.boxShadow;
          let i = 0;
          const t = setInterval(() => {{
            el.style.boxShadow = (i % 2 === 0) ? "0 0 0 10px rgba(0,150,136,0.08)" : orig;
            i++;
            if (i > 6) {{
              clearInterval(t); el.style.boxShadow = orig;
            }}
          }}, 220);
        }}
      }})();
    </script>
    """
    components.html(html, height=130)

def show_highlighted_result_and_scroll(text, positive=True):
    color = "#1b6b3a" if positive else "#a00000"
    banner_id = "pred_banner_" + str(uuid.uuid4()).replace("-", "_")
    html = f"""
    <div id="{banner_id}" style="
      border-radius:10px;
      padding:14px;
      margin:10px 0 20px 0;
      display:flex;
      align-items:center;
      gap:18px;
      background: rgba(255,255,255,0.01);
    ">
      <div style="flex:1">
        <div style="font-size:18px; font-weight:700; color:{color}; margin-bottom:6px;">{tr('result')}</div>
        <div style="font-size:15px; color:#e9ecef;">{text}</div>
      </div>
      <div style="background:{color}; color:white; padding:10px 14px; border-radius:8px; font-weight:700;">
        {text}
      </div>
    </div>
    <script>
      (function(){{
        const el = document.getElementById("{banner_id}");
        if (el){{
          el.scrollIntoView({{behavior: "smooth", block: "center"}});
          const orig = el.style.boxShadow;
          let i = 0;
          const t = setInterval(() => {{
            el.style.boxShadow = (i % 2 === 0) ? "0 0 0 8px rgba(255,255,0,0.08)" : orig;
            i++;
            if (i > 6) {{ clearInterval(t); el.style.boxShadow = orig; }}
          }}, 180);
        }}
      }})();
    </script>
    """
    components.html(html, height=120)

# ---------------------- user store + history ----------------------
_default_demo_users = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Administrator",
        "email": "admin@example.com",
    },
    "vivek": {
        "password_hash": hashlib.sha256("vivek123".encode()).hexdigest(),
        "name": "Vivek Kumar",
        "email": "vivek@example.com",
    }
}

def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return dict(_default_demo_users)
    else:
        d = dict(_default_demo_users)
        for k in d:
            d[k].setdefault("history", [])
        return d

def save_users(users):
    try:
        with open(USER_DB_FILE, "w") as f:
            json.dump(users, f, indent=2, default=str)
    except Exception as e:
        st.warning("Could not save users to disk: " + str(e))

_users = load_users()

def canonical_username(u):
    return u.strip().lower() if isinstance(u, str) else u

def verify_user(username, password):
    key = canonical_username(username)
    user = _users.get(key)
    if not user:
        return False
    return user["password_hash"] == hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, name="", email=""):
    key = canonical_username(username)
    if not username or not password:
        return False, "Username and password are required"
    if key in _users:
        return False, "Username already exists"
    _users[key] = {
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "name": name,
        "email": email,
        "history": []
    }
    save_users(_users)
    return True, "User created"

def get_user_profile(username):
    key = canonical_username(username)
    return _users.get(key, {})

def update_user_profile(username, name="", email=""):
    key = canonical_username(username)
    if key in _users:
        _users[key]["name"] = name
        _users[key]["email"] = email
        save_users(_users)
        return True
    return False

def add_history_record(username, record):
    key = canonical_username(username)
    if key not in _users:
        return False
    _users[key].setdefault("history", [])
    _users[key]["history"].insert(0, record)
    _users[key]["history"] = _users[key]["history"][:500]
    save_users(_users)
    return True

# ---------------------- auto-detect models in notebook/ ----------------------
def discover_models():
    models = {}
    if not os.path.isdir(MODELS_DIR):
        return models
    for fname in os.listdir(MODELS_DIR):
        if fname.lower().endswith(".pkl") or fname.lower().endswith(".joblib"):
            key = os.path.splitext(fname)[0]
            try:
                model_obj = pickle.load(open(os.path.join(MODELS_DIR, fname), "rb"))
                models[key] = {"model": model_obj, "filename": fname}
            except Exception as e:
                st.warning(f"Could not load model {fname}: {e}")
    return models

MODELS = discover_models()
MODEL_LABELS = {
    "diabetes_model": tr("diabetes"),
    "heart_model": tr("heart"),
    "parkinsons_model": tr("parkinsons")
}
AVAILABLE_MODELS = []
for key in MODELS:
    label = MODEL_LABELS.get(key, key.replace("_", " ").title())
    AVAILABLE_MODELS.append((key, label))

# ---------------------- session defaults ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "", "email": ""}
if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = True
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ---------------------- Sidebar + styling ----------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div:first-child {
      padding: 18px;
      border-top-right-radius: 14px;
      border-bottom-right-radius: 14px;
      background: linear-gradient(180deg,#1f1f23,#19191c);
    }
    .stApp > main { padding-top: 8px; }
    .block-container .stTextInput input, .block-container .stNumberInput input {
      background: #0f1113;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    # language selector: English / Hindi (maps to 'en' / 'hi')
    lang_choice = st.selectbox("Language / भाषा", options=["English", "Hindi"], index=0 if st.session_state.lang=="en" else 1)
    st.session_state.lang = "en" if lang_choice == "English" else "hi"

    # menu (use translated labels)
    menu_items = [
        tr("home"), tr("diabetes"), tr("heart"), tr("parkinsons"),
        tr("prediction_history"), tr("admin_dashboard"), tr("profile"), tr("about")
    ]
    selected = option_menu(
        menu_title=None,
        options=menu_items,
        icons=["house", "activity", "heart", "person", "clock-history", "speedometer", "person-circle", "info-circle"],
        menu_icon="cast",
        default_index=1,
        orientation="vertical"
    )

    st.markdown("---")
    # Account area
    if st.session_state.logged_in:
        st.markdown(f"**{tr('username')}:** {st.session_state.user}")
        if st.button(tr("logout")):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.profile = {"name": "", "email": ""}
            st.success(tr("logout"))
    else:
        st.markdown("**Account**")
        acc_choice = st.radio("", (tr("login"), tr("register")))
        if acc_choice == tr("login"):
            user_in = st.text_input(tr("username"), key="login_username")
            pw_in = st.text_input(tr("password"), type="password", key="login_password")
            if st.button(tr("sign_in")):
                if verify_user(user_in, pw_in):
                    st.session_state.logged_in = True
                    st.session_state.user = canonical_username(user_in)
                    profile_data = get_user_profile(user_in)
                    st.session_state.profile = {"name": profile_data.get("name", ""), "email": profile_data.get("email", "")}
                    st.success(f"Welcome, {st.session_state.profile.get('name') or st.session_state.user}!")
                else:
                    st.error("Invalid username or password")
        else:
            new_user = st.text_input(tr("username"), key="reg_user")
            new_pw = st.text_input(tr("password"), type="password", key="reg_pw")
            new_name = st.text_input(tr("profile"), placeholder="Full name (optional)", key="reg_name")
            new_email = st.text_input("Email (optional)", key="reg_email")
            if st.button(tr("create_account")):
                ok, msg = create_user(new_user, new_pw, new_name, new_email)
                if ok:
                    st.success("Account created. Please log in.")
                else:
                    st.error(msg)

    st.markdown("---")
    st.checkbox(tr("enable_tts"), key="enable_tts", value=st.session_state.get("enable_tts", True))
    st.markdown("---")
    st.write("Available models:")
    for key, label in AVAILABLE_MODELS:
        st.write(f"- **{label}**  (_{MODELS[key]['filename']}_)")
    st.write("Tip: drop more `.pkl` files into the `notebook/` folder and restart the app.")

# ---------------------- helper functions ----------------------
def ensure_logged_in():
    if not st.session_state.logged_in:
        st.warning("Please log in to use prediction features. Use the sidebar Account -> Login.")
        st.stop()

def model_predict_and_confidence(model_obj, X):
    pred = model_obj.predict(X)
    conf = None
    try:
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X)
            conf = float(np.max(probs, axis=1)[0])
        elif hasattr(model_obj, "decision_function"):
            df = model_obj.decision_function(X)
            conf = float(1 / (1 + np.exp(-float(df[0]))))
        else:
            conf = None
    except Exception:
        conf = None
    return int(pred[0]), conf

def generate_pdf_bytes(user, record):
    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        margin = 40
        y = height - margin
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, tr("report_title"))
        y -= 30
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, f"{tr('timestamp')}: {record.get('timestamp')}")
        y -= 20
        c.drawString(margin, y, f"{tr('model')}: {record.get('model')}")
        y -= 20
        c.drawString(margin, y, f"{tr('result')}: {record.get('result')}")
        y -= 20
        conf_text = f"{tr('confidence')}: {record.get('confidence'):.3f}" if record.get("confidence") is not None else f"{tr('confidence')}: {tr('no_model_proba')}"
        c.drawString(margin, y, conf_text)
        y -= 26
        c.drawString(margin, y, f"{tr('inputs')}:")
        y -= 16
        inputs = record.get("inputs", {})
        for k, v in inputs.items():
            s = f" - {k}: {v}"
            if y < 80:
                c.showPage()
                y = height - margin
            c.drawString(margin + 6, y, s)
            y -= 14
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    else:
        buffer = io.BytesIO()
        txt_lines = []
        txt_lines.append(tr("report_title"))
        txt_lines.append("="*len(tr("report_title")))
        txt_lines.append(f"{tr('timestamp')}: {record.get('timestamp')}")
        txt_lines.append(f"{tr('model')}: {record.get('model')}")
        txt_lines.append(f"{tr('result')}: {record.get('result')}")
        if record.get("confidence") is not None:
            txt_lines.append(f"{tr('confidence')}: {record.get('confidence'):.3f}")
        else:
            txt_lines.append(f"{tr('confidence')}: {tr('no_model_proba')}")
        txt_lines.append("")
        txt_lines.append(f"{tr('inputs')}:")
        for k, v in record.get("inputs", {}).items():
            txt_lines.append(f" - {k}: {v}")
        s = "\n".join(txt_lines)
        buffer.write(s.encode("utf-8"))
        buffer.seek(0)
        return buffer

# ---------------------- Pages ----------------------
if selected == tr("home"):
    show_home_hero_and_scroll()
    st.write("## " + tr("home"))
    st.write(tr("app_sub"))
    st.write("---")
    st.write("Demo accounts: `admin` / `admin123`, `vivek` / `vivek123`")
    st.write("To add more disease models, put their `.pkl` files into `notebook/` and restart the app.")

# Prediction pages (unified)
if selected in [tr("diabetes"), tr("heart"), tr("parkinsons")]:
    ensure_logged_in()
    show_home_hero_and_scroll()
    st.markdown(f"<h1 style='color:white'>{selected}</h1>", unsafe_allow_html=True)
    st.write("Fill features below. Numeric placeholders example values are shown.")

    # Determine model key
    desired_label = selected
    model_key = None
    for k, label in AVAILABLE_MODELS:
        if label.lower() == desired_label.lower() or k.lower().find(desired_label.split()[0].lower()) != -1:
            model_key = k
            break
    if model_key is None and AVAILABLE_MODELS:
        model_key = AVAILABLE_MODELS[0][0]

    # DIABETES
    if "diabetes" in model_key:
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
        with col2:
            Glucose = st.number_input("Glucose Level (mg/dl)", min_value=0.0, value=120.0)
        with col3:
            BloodPressure = st.number_input("BloodPressure (mm Hg)", min_value=0.0, value=70.0)
        with col1:
            SkinThickness = st.number_input("SkinThickness (mm)", min_value=0.0, value=20.0)
        with col2:
            Insulin = st.number_input("Insulin (IU)", min_value=0.0, value=80.0)
        with col3:
            BMI = st.number_input("BMI (kg/m2)", min_value=0.0, value=28.5)
        with col1:
            DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.5)
        with col2:
            Age = st.number_input("Age (years)", min_value=0.0, value=45.0)

        if st.button("Diabetes Test Result"):
            try:
                raw_features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                NewBMI_Overweight = NewBMI_Underweight = NewBMI_Obesity_1 = NewBMI_Obesity_2 = NewBMI_Obesity_3 = 0
                NewInsulinScore_Normal = NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0
                b = BMI
                if b <= 18.5:
                    NewBMI_Underweight = 1
                elif 18.5 < b <= 24.9:
                    pass
                elif 24.9 < b <= 29.9:
                    NewBMI_Overweight = 1
                elif 29.9 < b <= 34.9:
                    NewBMI_Obesity_1 = 1
                elif 34.9 < b <= 39.9:
                    NewBMI_Obesity_2 = 1
                else:
                    NewBMI_Obesity_3 = 1
                if 16 <= Insulin <= 166:
                    NewInsulinScore_Normal = 1
                g = Glucose
                if g <= 70:
                    NewGlucose_Low = 1
                elif 70 < g <= 99:
                    NewGlucose_Normal = 1
                elif 99 < g <= 126:
                    NewGlucose_Overweight = 1
                else:
                    NewGlucose_Secret = 1

                engineered_flags = [NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
                                    NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
                                    NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret]
                full_input = raw_features + engineered_flags

                model_obj = MODELS[model_key]["model"]
                expected = getattr(model_obj, "n_features_in__", None)
                if expected is None:
                    try:
                        X = np.array(raw_features, dtype=float).reshape(1, -1)
                        pred, conf = model_predict_and_confidence(model_obj, X)
                    except Exception:
                        X = np.array(full_input, dtype=float).reshape(1, -1)
                        pred, conf = model_predict_and_confidence(model_obj, X)
                else:
                    if expected == len(raw_features):
                        X = np.array(raw_features, dtype=float).reshape(1, -1)
                        pred, conf = model_predict_and_confidence(model_obj, X)
                    elif expected == len(full_input):
                        X = np.array(full_input, dtype=float).reshape(1, -1)
                        pred, conf = model_predict_and_confidence(model_obj, X)
                    else:
                        st.error(f"Model expects {expected} features. Raw={len(raw_features)}, Full={len(full_input)}.")
                        st.stop()

                if pred == 1:
                    res = "The person has diabetes"
                    positive = True
                else:
                    res = "The person does not have diabetes"
                    positive = False

                show_highlighted_result_and_scroll(res, positive=positive)
                if st.session_state.get("enable_tts", True):
                    speak(res)

                record = {
                    "id": str(uuid.uuid4()),
                    "model": MODELS[model_key]["filename"],
                    "model_key": model_key,
                    "inputs": dict(zip(["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age"], raw_features)),
                    "result": res,
                    "confidence": conf,
                    "timestamp": datetime.now().isoformat()
                }
                add_history_record(st.session_state.user, record)
                buf = generate_pdf_bytes(st.session_state.user, record)
                bname = f"report_{record['id']}.pdf" if REPORTLAB_AVAILABLE else f"report_{record['id']}.txt"
                st.download_button(tr("download_report"), data=buf, file_name=bname, mime="application/octet-stream")
            except Exception as e:
                st.error(tr("error_fill") + str(e))

    # HEART
    elif "heart" in model_key:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0.0, value=54.0)
        with col2:
            sex = st.selectbox("Sex (1=male,0=female)", options=[1,0], index=0)
        with col3:
            cp = st.number_input("Chest Pain Types (0-3)", min_value=0.0, max_value=3.0, value=1.0)
        with col1:
            trtbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0.0, value=130.0)
        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=233.0)
        with col3:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1=true,0=false)", options=[0,1], index=0)
        with col1:
            restecg = st.number_input("Resting ECG (0/1/2)", min_value=0.0, max_value=2.0, value=0.0)
        with col2:
            thalachh = st.number_input("Max Heart Rate (thalach)", min_value=0.0, value=150.0)
        with col3:
            exng = st.selectbox("Exercise Induced Angina (1=yes,0=no)", options=[0,1], index=0)
        with col1:
            oldpeak = st.number_input("ST depression (oldpeak)", value=1.2)
        with col2:
            slp = st.number_input("Slope (0/1/2)", min_value=0.0, max_value=2.0, value=1.0)
        with col3:
            caa = st.number_input("Major vessels colored by flourosopy (0-3)", min_value=0.0, max_value=3.0, value=0.0)
        with col1:
            thall = st.number_input("thal (0 normal,1 fixed,2 reversible)", min_value=0.0, max_value=2.0, value=1.0)

        if st.button("Heart Disease Test Result"):
            try:
                vals = [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]
                model_obj = MODELS[model_key]["model"]
                Xh = np.array(vals, dtype=float).reshape(1, -1)
                expected_h = getattr(model_obj, "n_features_in__", None)
                if expected_h is not None and Xh.shape[1] != expected_h:
                    st.error(f"Model expects {expected_h} features but input has {Xh.shape[1]}.")
                    st.stop()
                pred_h, conf = model_predict_and_confidence(model_obj, Xh)
                if pred_h == 1:
                    res = "This person is having heart disease"
                    pos = True
                else:
                    res = "This person does not have any heart disease"
                    pos = False
                show_highlighted_result_and_scroll(res, positive=pos)
                if st.session_state.get("enable_tts", True): speak(res)
                record = {
                    "id": str(uuid.uuid4()), "model": MODELS[model_key]["filename"], "model_key": model_key,
                    "inputs": {"age": age, "sex": sex, "cp": cp, "trtbps": trtbps, "chol": chol, "fbs": fbs,
                               "restecg": restecg, "thalachh": thalachh, "exng": exng, "oldpeak": oldpeak,
                               "slp": slp, "caa": caa, "thall": thall},
                    "result": res, "confidence": conf, "timestamp": datetime.now().isoformat()
                }
                add_history_record(st.session_state.user, record)
                buf = generate_pdf_bytes(st.session_state.user, record)
                bname = f"report_{record['id']}.pdf" if REPORTLAB_AVAILABLE else f"report_{record['id']}.txt"
                st.download_button(tr("download_report"), data=buf, file_name=bname, mime="application/octet-stream")
            except Exception as e:
                st.error(tr("error_fill") + str(e))

    # PARKINSONS OR OTHER
    else:
        if "parkinsons" in model_key:
            st.write("Enter 22 voice-feature numeric fields (see model training spec).")
            cols = st.columns(4)
            labels = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP",
                      "MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]
            inputs = []
            for i, label in enumerate(labels):
                col = cols[i % 4]
                inputs.append(col.number_input(label, value=0.0))
            if st.button("Parkinsons's Test Result"):
                try:
                    numeric_vals = [float(x) for x in inputs]
                    Xp = np.array(numeric_vals).reshape(1, -1)
                    model_obj = MODELS[model_key]["model"]
                    expected_p = getattr(model_obj, "n_features_in__", None)
                    if expected_p is not None and Xp.shape[1] != expected_p:
                        st.error(f"Model expects {expected_p} features but input has {Xp.shape[1]}.")
                        st.stop()
                    pred_p, conf = model_predict_and_confidence(model_obj, Xp)
                    if pred_p == 1:
                        res = "This person has Parkinsons disease"
                        pos = True
                    else:
                        res = "This person does not have Parkinsons disease"
                        pos = False
                    show_highlighted_result_and_scroll(res, positive=pos)
                    if st.session_state.get("enable_tts", True): speak(res)
                    record = {"id": str(uuid.uuid4()), "model": MODELS[model_key]["filename"], "model_key": model_key,
                              "inputs": dict(zip(labels, numeric_vals)), "result": res, "confidence": conf, "timestamp": datetime.now().isoformat()}
                    add_history_record(st.session_state.user, record)
                    buf = generate_pdf_bytes(st.session_state.user, record)
                    bname = f"report_{record['id']}.pdf" if REPORTLAB_AVAILABLE else f"report_{record['id']}.txt"
                    st.download_button(tr("download_report"), data=buf, file_name=bname, mime="application/octet-stream")
                except Exception as e:
                    st.error(tr("error_fill") + str(e))
        else:
            st.write("Custom model detected. Enter comma-separated numeric features matching the model training schema.")
            raw = st.text_area("Comma-separated numeric features (one sample)", placeholder="e.g. 5,124,70,... ")
            if st.button("Predict with custom model"):
                try:
                    vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
                    if len(vals) == 0:
                        st.error("Please enter numeric CSV values.")
                        st.stop()
                    model_obj = MODELS[model_key]["model"]
                    X = np.array(vals, dtype=float).reshape(1, -1)
                    expected = getattr(model_obj, "n_features_in__", None)
                    if expected is not None and X.shape[1] != expected:
                        st.error(f"Model expects {expected} features but input has {X.shape[1]}.")
                        st.stop()
                    pred, conf = model_predict_and_confidence(model_obj, X)
                    res = f"Predicted label: {pred}"
                    show_highlighted_result_and_scroll(res, positive=True)
                    if st.session_state.get("enable_tts", True):
                        speak(res)
                    record = {"id": str(uuid.uuid4()), "model": MODELS[model_key]["filename"], "model_key": model_key,
                              "inputs": {"csv": vals}, "result": res, "confidence": conf, "timestamp": datetime.now().isoformat()}
                    add_history_record(st.session_state.user, record)
                    buf = generate_pdf_bytes(st.session_state.user, record)
                    bname = f"report_{record['id']}.pdf" if REPORTLAB_AVAILABLE else f"report_{record['id']}.txt"
                    st.download_button(tr("download_report"), data=buf, file_name=bname, mime="application/octet-stream")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# Prediction history page
if selected == tr("prediction_history"):
    ensure_logged_in()
    st.title(tr("prediction_history"))
    profile = get_user_profile(st.session_state.user)
    history = profile.get("history", [])
    if not history:
        st.info("No prediction history yet. Run a prediction to create records.")
    else:
        for rec in history:
            st.write(f"**{rec.get('timestamp')}**  —  {rec.get('model')}  —  {rec.get('result')}")
            st.write(f"Confidence: {rec.get('confidence') if rec.get('confidence') is not None else tr('no_model_proba')}")
            with st.expander("Inputs"):
                st.json(rec.get("inputs", {}))
            buf = generate_pdf_bytes(st.session_state.user, rec)
            st.download_button("Download report", data=buf, file_name=f"report_{rec.get('id')}.pdf" if REPORTLAB_AVAILABLE else f"report_{rec.get('id')}.txt", mime="application/octet-stream")

# Admin dashboard: replaced inline page with separate file call
if selected == tr("admin_dashboard"):
    ensure_logged_in()
    # call the external admin_dashboard.show_admin_dashboard
    show_admin_dashboard(st.session_state.user, _users, MODELS)

# Profile & About
if selected == tr("profile"):
    ensure_logged_in()
    st.title(tr("profile"))
    name = st.text_input("Full name", value=st.session_state.profile.get("name",""), key="profile_name")
    email = st.text_input("Email", value=st.session_state.profile.get("email",""), key="profile_email")
    if st.button("Save profile"):
        st.session_state.profile["name"] = name
        st.session_state.profile["email"] = email
        update_user_profile(st.session_state.user, name=name, email=email)
        st.success("Profile updated")
    st.write("Current session user:", st.session_state.user)

if selected == tr("about"):
    st.title(tr("about"))
    st.write("This enhanced app includes: downloadable reports, prediction history, admin dashboard, confidence scores, UI improvements, multi-language support, and more.")
    st.write("To add new models: drop their `.pkl` files into the `notebook/` folder and restart the app.")
