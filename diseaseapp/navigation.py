import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    st.title('Disease Detection App')
    image=Image.open("C:/Users/Razeen Khan/Documents/diseaseapp/images/bg.jpg")
    st.image(image)
    st.sidebar.title("Detect the disease that you want..")
    st.sidebar.write("This app will use machine learning for the detection of diseases. You can enter your inputs and based on that parameters this app will detect if you have that disease or not.")
    st.sidebar.subheader("We wish you a better health!")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.title("Let's Go!")

elif navigation() == "covid":
    # Create a title and a subtitle
    st.title("Covid Detection")
    st.write(" Detects if someone has covid using machine learning")
    #  and display an image
    image = Image.open('C:/Users/Razeen Khan/Documents/diseaseapp/images/covid.jpg')
    st.image(image, caption="Covid", use_column_width=True)

    # get the data
    df = pd.read_csv('C:/Users/Razeen Khan/Documents/diseaseapp/COVID-19.csv')

    # set a sub header on web app
    st.subheader('Data Information')

    # show the data as a table
    st.dataframe(df)

    # show statistics of the data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # spit the data into independent x and dependant y variables
    x = df.iloc[:, 0:13].values
    y = df.iloc[:, -1].values

    # split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # get the feature input from the user


    def get_user_input():
        age = st.sidebar.slider('Age', 0, 100, 25)
        gender = st.sidebar.slider('Gender', 0, 1, 0)
        # temprature = st.sidebar.slider('Temperature', 60.0, 120.0, 98.0)
        dry_cough = st.sidebar.slider('Dry Cough', 0, 1, 0)
        sour_throat = st.sidebar.slider('Sour Throat', 0, 1, 0)
        weakness = st.sidebar.slider('Weakness', 0, 1, 0)
        breathing_problem = st.sidebar.slider('Breathing Problem', 0, 1, 0)
        drowsiness = st.sidebar.slider('Drowsiness', 0, 1, 0)
        pain_in_chest = st.sidebar.slider('Pain in Chest', 0, 1, 0)
        diabetes = st.sidebar.slider('Diabetes', 0, 1, 0)
        lung_disease = st.sidebar.slider('Lung Disease', 0, 1, 0)
        high_bp = st.sidebar.slider('High Blood Pressure', 0, 1, 0)
        kidney_disease = st.sidebar.slider('Kidney Disease', 0, 1, 0)
        loss_of_smell = st.sidebar.slider('Loss of Smell', 0, 1, 0)

        # store a dictionary into a variable
        user_data = {
            'age': age,
            'gender': gender,
            'dry_cough': dry_cough,
            'sour_throat': sour_throat,
            'weakness': weakness,
            'breathing_problem': breathing_problem,
            'drowsiness': drowsiness,
            'pain_in_chest': pain_in_chest,
            'diabetes': diabetes,
            'lung_disease': lung_disease,
            'high_bp': high_bp,
            'kidney_disease': kidney_disease,
            'loss_of_smell': loss_of_smell,
        }

        # transform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features


    # store the user input in variable
    user_input = get_user_input()

    # set a subheader and display user input
    st.subheader('User Input: ')
    st.write(user_input)

    # create and train the model
    RandomForestClassifier = RandomForestClassifier()
    RandomForestClassifier.fit(x_train, y_train)

    # show the models metrics
    st.subheader('Model test accuracy score in percentage:')
    st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100))

    #  the model prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

    # set a sub-header and display a classification
    st.subheader('Classification: ')
    st.write('0 means you do not have covid. ')
    st.write('1 means you do  have covid. ')
    st.subheader('Prediction is :')
    st.write(prediction)

elif navigation() == "diabetes":
    # Create a title and a subtitle
    st.title("Diabetes Detection")
    st.write(" Detects if someone has diabetes using machine learning")
    #  and display an image
    image = Image.open('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/images/diabetes.png')
    st.image(image, caption="Diabetes", use_column_width=True)

    # get the data
    df = pd.read_csv('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/diabetes.csv')

    # set a sub header on web app
    st.subheader('Data Information')

    # show the data as a table
    st.dataframe(df)

    # show statistics of the data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # spit the data into independent x and dependant y variables
    x = df.iloc[:, 0:8].values
    y = df.iloc[:, -1].values

    # split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # get the feature input from the user
    def get_user_input():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 199, 117)
        bp = st.sidebar.slider('Blood Pressure', 0, 122, 68)
        skint = st.sidebar.slider('Skin Thickness', 0, 99, 23)
        insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
        bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
        age = st.sidebar.slider('Age', 0, 100, 20)

        # store a dictionary into a variable
        user_data = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'bp': bp,
            'skint': skint,
            'insulin': insulin,
            'bmi': bmi,
            'dpf': dpf,
            'age': age
        }

        # transform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features


    # store the user input in variable
    user_input = get_user_input()

    # set a subheader and display user input
    st.subheader('User Input: ')
    st.write(user_input)

    # create and train the model
    RandomForestClassifier = RandomForestClassifier()
    RandomForestClassifier.fit(x_train, y_train)

    # show the models metrics
    st.subheader('Model test accuracy score in percentage:')
    st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100))

    #  the model prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

    # set a sub-header and display a classification
    st.subheader('Classification: ')
    st.write('0 means you do not have diabetes. ')
    st.write('1 means you do  have diabetes. ')
    st.subheader('Prediction is :')
    st.write(prediction)

elif navigation() == "hd":
    # Create a title and a subtitle
    st.title(" Heart Disease Detection")
    st.write(" Detects if someone has heart disease using machine learning")
    # open and display an image
    image = Image.open('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/images/Heart Disease Symptoms.png')
    st.image(image, caption="Heart disease", use_column_width=True)

    # get the data
    df = pd.read_csv('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/Heart_Disease_Prediction.csv')

    # set a sub header on web app
    st.subheader('Data Information')

    # show the data as a table
    st.dataframe(df)

    # show statistics of the data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # spit the data into independant x and dependant y variables
    x = df.iloc[:, 0:11].values
    y = df.iloc[:, -1].values

    # split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # get the feature input from the user
    def get_user_input():
        age = st.sidebar.slider('Age', 0, 100, 27)
        sex = st.sidebar.slider('Sex', 0, 1, 0)
        chest_pain_type = st.sidebar.slider('Chest Pain type', 1, 4, 1)
        bp = st.sidebar.slider('Blood Pressure', 94, 200, 120)
        cholesterol = st.sidebar.slider('Cholesterol', 126, 564, 200)
        hr = st.sidebar.slider('Heart Rate', 71, 202, 100)
        exercise_angina = st.sidebar.slider('Exercise Angina', 0, 1, 0)
        st_depression = st.sidebar.slider('ST Depression', 0.0, 6.2, 1.0)
        slope_of_st = st.sidebar.slider(' Slope of ST', 0, 100, 20)
        number_of_vessels_fluro = st.sidebar.slider('Number of vessels fluro', 0, 100, 34)
        thallium = st.sidebar.slider('Thallium', 0, 100, 45)

        # store a dictionary into a variable
        user_data = {
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'bp': bp,
            'cholesterol': cholesterol,
            'hr': hr,
            'exercise_angina': exercise_angina,
            'st_depression': st_depression,
            'slope_of_st': slope_of_st,
            'number_of_vessels_fluro': number_of_vessels_fluro,
            'thallium': thallium
        }

        # transform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features


    # store the user input in variable
    user_input = get_user_input()

    # set a subheader and display user input
    st.subheader('User Input: ')
    st.write(user_input)

    # create and train the model
    RandomForestClassifier = RandomForestClassifier()
    RandomForestClassifier.fit(x_train, y_train)

    # show the models metrics
    st.subheader('Model test accuracy score in percentage:')
    st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100))

    # store the models prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

    # set a subheader and display a classification
    st.subheader('Classification: ')
    st.write('Absense means you do not have any heart disease.')
    st.write('Presence means you do  have heart disease.')
    st.subheader('Prediction is :')
    st.write(prediction)


elif navigation() == "bc":
    # Create a title and a subtitle
    st.title(" Breast Cancer Detection")
    st.write(" Detects if someone has breast cancer using machine learning")
    # open and display an image
    image = Image.open('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/images/breast_cancer.png')
    st.image(image, caption="Heart disease", use_column_width=True)

    # get the data
    df = pd.read_csv('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/Breast_cancer_data.csv')

    # set a sub header on web app
    st.subheader('Data Information')

    # show the data as a table
    st.dataframe(df)

    # show statistics of the data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # spit the data into independant x and dependant y variables
    x = df.iloc[:, 0:5].values
    y = df.iloc[:, -1].values

    # split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # get the feature input from the user
    def get_user_input():
        mean_radius = st.sidebar.slider('Mean radius', 0, 50, 27)
        mean_texture = st.sidebar.slider('Mean texture', 0, 50, 27)
        mean_perimeter = st.sidebar.slider('Mean perimeter', 0, 100, 27)
        mean_area = st.sidebar.slider('Mean area', 0, 1000, 500)
        mean_smoothness = st.sidebar.slider('Mean Smoothness', 0, 1000, 500)

        # store a dictionary into a variable
        user_data = {
            'mean_radius': mean_radius,
            'mean_texture': mean_texture,
            'mean_perimeter': mean_perimeter,
            'mean_area': mean_area,
            'mean_smoothness': mean_smoothness,
        }

        # tranform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features


    # store the user input in variable
    user_input = get_user_input()

    # set a subheader and display user input
    st.subheader('User Input: ')
    st.write(user_input)

    # create and train the model
    RandomForestClassifier = RandomForestClassifier()
    RandomForestClassifier.fit(x_train, y_train)

    # show the models metrics
    st.subheader('Model test accuracy score in percentage:')
    st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100))

    # store the models prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

    # set a subheader and display a classification
    st.subheader('Classification: ')
    st.write('0 means you do not have breast cancer.')
    st.write('1 means you do  have  breast cancer.')
    st.subheader('Prediction is :')
    st.write(prediction)



elif navigation() == "kid":
    # Create a title and a subtitle
    st.title(" Kidney Disease Detection")
    st.write(" Detects if someone has kidney disease using machine learning")
    # open and display an image
    image = Image.open('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/images/kidney.png')
    st.image(image, caption="Heart disease", use_column_width=True)

    # get the data
    df = pd.read_csv('C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/kidney_disease.csv')

    # set a sub header on web app
    st.subheader('Data Information')

    # show the data as a table
    st.dataframe(df)

    # show statistics of the data
    st.subheader('Data Statistics')
    st.write(df.describe())

    # spit the data into independant x and dependant y variables
    x = df.iloc[:, 0:8].values
    y = df.iloc[:, -1].values

    # split the dataset into 75% training and 25% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # get the feature input from the user
    def get_user_input():
        blood_pressure = st.sidebar.slider('Blood Pressure', 0, 100, 27)
        sugar = st.sidebar.slider('Sugar', 0, 100, 27)
        rbc = st.sidebar.slider('Red Blood Cells', 0, 100, 27)
        blood_urea = st.sidebar.slider('Blood Urea', 0, 100, 27)
        sodium = st.sidebar.slider('Sodium', 0, 100, 27)
        potassium = st.sidebar.slider('Potassium', 0, 100, 27)
        hemoglobin = st.sidebar.slider('Hemoglobin', 0, 100, 27)
        wbc = st.sidebar.slider('White Blood Cells', 0, 100, 27)

        # store a dictionary into a variable
        user_data = {
            'blood_pressure': blood_pressure,
            'sugar': sugar,
            'rbc': rbc,
            'blood_urea': blood_urea,
            'sodium': sodium,
            'potassium': potassium,
            'hemoglobin': hemoglobin,
            'wbc': wbc,
        }

        # tranform the data into a data frame
        features = pd.DataFrame(user_data, index=[0])
        return features


    # store the user input in variable
    user_input = get_user_input()

    # set a subheader and display user input
    st.subheader('User Input: ')
    st.write(user_input)

    # create and train the model
    RandomForestClassifier = RandomForestClassifier()
    RandomForestClassifier.fit(x_train, y_train)

    # show the models metrics
    st.subheader('Model test accuracy score in percentage:')
    st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100))

    # store the models prediction in a variable
    prediction = RandomForestClassifier.predict(user_input)

    # set a subheader and display a classification
    st.subheader('Classification: ')
    st.write('0 means you do not have any Kidney disease.')
    st.write('1 means you do  have Kidney disease.')
    st.subheader('Prediction is :')
    st.write(prediction)


elif navigation() == "bmi":
    st.title("This is My BMI Calculator")

    img = Image.open("C:/Users/Razeen Khan/PycharmProjects/DiseaseDetection/images/bmi.jpg")
    st.image(img)

    # Introduction

    st.subheader("Introduction")

    st.text("""
    BMI is a person’s weight in kilograms divided by the square of height in meters. 
    A high BMI can indicate high body fatness.

    If your BMI is less than 18.5, it falls within the underweight range.
    If your BMI is 18.5 to <25, it falls within the healthy weight range.
    If your BMI is 25.0 to <30, it falls within the overweight range.
    If your BMI is 30.0 or higher, it falls within the obesity range.

    Obesity is frequently subdivided into categories:

    Class 1: BMI of 30 to < 35
    Class 2: BMI of 35 to < 40
    Class 3: BMI of 40 or higher. 
    Class 3 obesity is sometimes categorized as “severe” obesity.
    	""")

    # Input

    weight = st.sidebar.slider("Enter your Weight in KG", 0, 100, 20)

    height = st.sidebar.slider("Enter your Height in feets", 0.0, 7.0, 5.0)
    if height == 0 & weight == 0:
        st.sidebar.write("Height or weight can not be Zero")

    try:
        bmi = weight / (height) ** 2
        st.sidebar.subheader("Your BMI is: ")
        st.sidebar.success(f"Your BMI is {bmi}")
    except ZeroDivisionError:
        print("Zero Division Error occurred")


elif navigation() == "config":
    st.title('Configuration of the app.')
    st.write('Here you can configure the application')
