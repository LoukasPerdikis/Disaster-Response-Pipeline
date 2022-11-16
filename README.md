# Disaster-Response-Pipeline

This project aims to provide all necessary code and functionality in order to create a full Machine Learning Pipeline complete with ETL pipeline, ML pipeline, and web app creation. It surrounds the concept of using Natural Language Processing (NLP) to allow disaster relief organizations to quickly receive and react to incoming distress messages

### Structure
The project is split into several components:

1) Readme file for project details and information
2) Data folder containing initial csv file data files and data processing pipeline code
3) Models folder containing all ML model pipeline code
4) App folder containing html files and python script to create the Disaster Response web app

### Process

#### Extract, Transform, Load
The ETL pipeline aims to read in the disaster and messages csv data files, and performs all cleaning operations to provide a dataset that is fit for machine learning purposes. The cleaning process includes:
   - splitting 'categories' column values to generate targets
   - renaming categories columns
   - replacing target values with 0, 1 for ML purposes
   - dropping unnecessary columns and duplicate rows

Once this has been achieved, the cleaned dataset is saved to a SQLite database for further use.

#### Machine Learning
The ML pipeline aims to retrieve the cleaned data from the SQLite database, and conduct necessary NLP transformations for algorithm training. The data contains multiple potential targets, and to that effect the machine learning problem is a multi-output classification problem. The NLP process includes:
   - normalizing text by applying lower case and punctuation removal 
   - tokenizing text using the nltk library
   - removing stopwords that would not be helpful for ML
   - applying lemmatization

Once this has been achieved, the machine learning pipeline makes use of sklearn's Pipeline class to initialize a vectorizer, TF-IDF transformer, as well as a classification model for the actual machine learning process. This is then conducted with gridsearch capabilities to ensure the best possible model outcome. For this particular project, the gridsearch iteration process is kept to a minimum to preserve computational stress and for efficiency purposes.

The trained model is then tested on unseen data (according to standard ML training and testing techniques), and a classification report supplied complete with accuracy, precision, recall, and F1 scores. The model is then saved to a pickle file for use within the web app.

#### Web App
The web app code provides basic data visualizations which explore and provide insights on the incoming data. The web app allows for the input of a particular message which is then classified according to the model's understanding of which disaster category that message is most aligned to. Such a tool has value for disaster relief organizations to use in understanding where the priority disasters are.

### Running the program
In order to run this project, simply inputting the following commands in a terminal should be sufficient:

1) in the root folder, run the ETL Pipeline with following commands: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2) in the root folder, run the ML Pipeline with the following commands: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3) transfer terminal execution path to the apps folder: cd apps
4) create the web app with the following commands: python run.py
5) You may then preview and test the web app with the created interface.
