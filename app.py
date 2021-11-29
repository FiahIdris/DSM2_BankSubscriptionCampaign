import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Preprocessing modules
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# Machine Learning metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Machine Learning model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Deployment purposes
import joblib
import pickle

# Sidebar widget
st.sidebar.header('Menu')
# loading our model
model = joblib.load("my_model.pkl")  # DIFF
new_data_test = pd.read_csv('new_test.csv', delimiter=",")


def main():
    page = st.sidebar.selectbox(
        "Select a page", ["Homepage", "Exploration", "Model", "Prediction"])

    if page == "Homepage":
        homepage_screen()
    elif page == "Exploration":
        exploration_screen()
    elif page == "Model":
        model_screen()
    elif page == "Prediction":
        model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('clean_dataset.csv', delimiter=",")
    return data


df = load_data()


def homepage_screen():

    st.title('BANK TERM DEPOSIT CAMPAIGN')
    st.header("Dataset Information")
    st.write("""  
        The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.  
        
        The dataset constains of 41188 instances and 21 columns. More information of the dataset can be accessed [here]("https://archive.ics.uci.edu/ml/datasets/Bank+Marketing").
        
    """)

    if st.checkbox('See dataset'):
        # Load data
        data_load_state = st.text('Loading data...')
        df = load_data()
        st.write(df)
        data_load_state.text('')


def exploration_screen():
    st.title("Data Exploration")
    st.write(""" 
        This page contains general exploratory data analysis in order to get basic insight of the dataset information and get the feeling about what this dataset is about.
    """)

    st.write("""
        ## 📌 Correlational Matrix  
        
    """)
    # Matrix correlation.
    # corr_matrix = np.triu(df.corr())
    fig, ax = plt.subplots()
    plt.figure(figsize=[13, 5])
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax)
    # plt.style.available
    plt.title('Correlation Among Features')
    st.write(fig)

    st.write("""
        ## 📌  Features Correlation Value toward Target Column
        
    """)
    # Display correlation towards target column
    fig, axs = plt.subplots(figsize=(10, 4))
    corr = df.corr()['y'].reset_index()
    # corr.drop( axis=0, inplace=True)
    sns.lineplot(data=corr, x='index', y='y', ax=axs)
    sns.scatterplot(data=corr, x='index', y='y', ax=axs)
    plt.xticks(rotation=70)
    st.write(fig)

    st.write("""
        ## 📌 Target Label Frequency  
        
    """)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
    df['y'].value_counts().plot(kind='bar', ax=axs[0])
    df['y'].value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, ax=axs[1], explode=[0.08]*2, colors=['green', 'teal'])
    st.write(fig)

    st.write("""
        ## 📌  Percentage of subscriber from previous campaign
        
    """)
    # Compare total subscriber from old_campaign.
    fig, axs = plt.subplots(ncols=2, figsize=(7, 4))
    yes = df[(df['old_campaign'] == 1) & (df['y'] == 1)]
    yes_total = df[df['old_campaign'] == 1]
    # yes.shape[0]/yes_total.shape[0]*100

    no = df[(df['old_campaign'] == 1) & (df['y'] == 0)]
    no_total = df[df['old_campaign'] == 1]
    # no.shape[0]/no_total.shape[0]*100

    axs[0].pie(x=[yes.shape[0]/yes_total.shape[0]*100, no.shape[0]/no_total.shape[0]*100],
               autopct="%.1f%%", explode=[0.05]*2, labels=[1, 0], colors=['maroon', 'teal'])
    axs[0].set_title('Attend Previous Campaign')

    yes = df[(df['old_campaign'] == 0) & (df['y'] == 1)]
    yes_total = df[df['old_campaign'] == 0]
    # yes.shape[0]/yes_total.shape[0]*100

    no = df[(df['old_campaign'] == 0) & (df['y'] == 0)]
    no_total = df[df['old_campaign'] == 0]
    # no.shape[0]/no_total.shape[0]*100
    axs[1].pie(x=[yes.shape[0]/yes_total.shape[0]*100, no.shape[0]/no_total.shape[0]*100],
               autopct="%.1f%%", explode=[0.08]*2, labels=[1, 0], colors=['maroon', 'teal'])
    axs[1].set_title('Not Attend Previous Campaign')
    st.write(fig)

    st.write("""
        ## 📌  Correlational features
        
    """)

    fig, axs = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, hue='y', x='month',
                  order=df['month'].value_counts().index)
    st.write(fig)

    fig, axs = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, hue='y', x='nr.employed',
                  order=df['nr.employed'].value_counts().index)
    st.write(fig)

    fig, axs = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, hue='y', x='emp.var.rate',
                  order=df['emp.var.rate'].value_counts().index)
    st.write(fig)

    fig, axs = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, hue='y', x='job',
                  order=df['job'].value_counts().index)
    plt.xticks(rotation=70)
    st.write(fig)


def model_screen():
    # Define model.
    logreg = LogisticRegression(solver="lbfgs", max_iter=1000)
    svc = SVC()
    randfors = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
    ganbay = GaussianNB()
    adaboost = AdaBoostClassifier()
    st.title("Model")
    st.write(""" 
             
             """)
    model_selected = st.selectbox("Select model: ", [
                                  'Logistic Regression', 'SVC', 'KNeighbors Classifier', 'RandomForest Classifier', 'Multinomial NB', 'AdaBoost Classifier'])
    if model_selected == 'Logistic Regression':
        matrix = [[8778, 122], [875, 255]]
        cross_val = 0.8988667701399706
        precision = 0.6763925729442971
        recall = 0.22566371681415928
        F1 = 0.33842070338420704
        train_acc = 0.8990661038917876
        test_acc = 0.9005982053838485
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'SVC':
        matrix = [[8790, 110], [880, 250]]
        cross_val = 0.8978032607622813
        precision = 0.6944444444444444
        recall = 0.22123893805309736
        F1 = 0.33557046979865773
        train_acc = 0.8992655123134701
        test_acc = 0.901296111665005
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'RandomForest Classifier':
        matrix = [[8594, 306], [804, 326]]
        cross_val = 0.8886968091349463
        precision = 0.5158227848101266
        recall = 0.2884955752212389
        F1 = 0.3700340522133938
        train_acc = 0.9422712619229618
        test_acc = 0.8893320039880359
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'Multinomial NB':
        matrix = [[8534, 366], [750, 380]]
        cross_val = 0.88766667147192853
        precision = 0.5093833780160858
        recall = 0.336283185840708
        F1 = 0.4051172707889126
        train_acc = 0.8876998238558942
        test_acc = 0.8887337986041874
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'AdaBoost Classifier':
        matrix = [[8756,  144], [861, 269]]
        cross_val = 0.8986673738686439
        precision = 0.6513317191283293
        recall = 0.336283185840708
        F1 = 0.3486714193130266
        train_acc = 0.8989996344178935
        test_acc = 0.8998005982053838
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)
    if model_selected == 'KNeighbors Classifier':
        matrix = [[8761, 139], [915,  215]]
        cross_val = 0.8941805904650459
        precision = 0.6073446327683616
        recall = 0.1902654867256637
        F1 = 0.28975741239892183
        train_acc = 0.9026886902190169
        test_acc = 0.8949152542372881
        validation(matrix, cross_val, precision,
                   recall, F1, train_acc, test_acc)


def validation(matrix, cross_score, prec_score, rec_score, f1, acc_score_train, acc_score_test):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.set_xlabel('PREDICTED')
    ax.set_ylabel('ACTUAL')
    ax.set_title('Confusion Matrix')
    st.write(fig)
    st.write(f"""
    👉 Cross Validation  mean: {cross_score}   
    👉 Precision : {prec_score}  
    👉 Recall : {rec_score}  
    👉 F1 : {f1}  

    👉 Training Accuracy : {acc_score_train}  
    👉 Validation Accuracy : {acc_score_test}  
             """)


def model_predict():
    st.title("Prediction")
    st.write("### Field this form to predict whether client will subscribe or not !")
    job_value = list(df.job.unique())
    education_value = list(df.education.unique())
    default_value = list(df.default.unique())
    month_value = list(df.month.unique())
    job = st.selectbox("Occupation/Job", job_value)
    education = st.selectbox("Education", education_value)
    default = st.radio("Have credit in default", default_value)
    contact = st.radio("Contact type", ['cellular', 'telephone'])
    month = st.selectbox("Month", month_value)
    campaign = st.slider(
        "Number of contacts performed during this campaign", 0, 20)
    pdays = st.radio(
        "The client was contacted since a previous campaign", ['yes', 'no'])
    emp_var_rate = st.number_input(
        label="Employment variation rate - quarterly indicator", min_value=-4., max_value=2., step=1., format="%.2f")
    nr_employed = st.slider("Number of employees", 4000, 6000)
    has_loan = st.radio("Currently have loan", ['yes', 'no'])
    old_campaign = st.radio("Attend previous campaign", ['yes', 'no'])
    submit_button = st.button("Predict")

    if contact == 'cellular':
        contact = 0
    else:
        contact = 1

    if pdays == 'yes':
        pdays = 1
    else:
        pdays = 0
    if has_loan == 'yes':
        has_loan = 1
    else:
        has_loan = 0

    if old_campaign == 'yes':
        old_campaign = 1
    else:
        old_campaign = 0

    data = {
        'contact': [contact], 'campaign': [campaign], 'pdays': [pdays], 'emp.var.rate': [emp_var_rate], 'nr.employed': [nr_employed],
        'has_loan': [has_loan], 'old_campaign': [old_campaign], 'job': [job], 'education': [education], 'default': [default], 'month': [month]
    }

    new_data = pd.DataFrame(data=data)

    if submit_button:
        result = model.predict(new_data)

        updated_res = result.flatten().astype(float)
        if updated_res[0] == 1:
            updated_res = "True"
            # st.balloons
        else:
            updated_res = 'False'
        st.success(
            'The Probability of getting client subscribed a term deposit is {}'.format(updated_res))


main()
