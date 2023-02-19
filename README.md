# Disaster Response Pipeline Project
This applications takes in a text message and classifies it against 36 categories. The classification can happen for 
the multiple categories. 

### Motivation:
Motivation is provide a classification model + app that can help in a disaster situation, where the aid teams are
struggling to save every minute in bringing the right aid to the distressed victims. This model can help to filter out
unrelated or non-urgent messages and also direct the messages to the respective queues/teams within the disaster response organization.

### Solution Approach:
This application cosists for 3 major components
1. ETL pipeline (process_data.py): This component takes in the raw datasets (Messages and Categories) and prepares the final datasets by merging and cleansing the data
2. ML pipeline and model (train_classifier.py): This component generates the features using NLP (TFIDF), builds a model and fit it with the training dataset. Finally, stores the final model as pickle file
3. Web app (run.py): This component provies - 1) the overview of the training dataset, 2) An interface to classify the text messages

### Datasets:
| Dataset | Source |
| ------- | ------ |
| disaster_categories.csv | Appen |
| disaster_messages.csv | Appen |

### Foler Structure and Files:
├── app
│   ├── temaplates
│   │   ├──  go.html
│   │   ├──  master.html
│   ├── run.py
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py
├── models
│   ├── classifiier.pkl
│   ├── train_classifier.py
├── README.md


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`


3. Run your web app: `python run.py`

4. Open the browser and connect to the URL to access the application

### License

MIT

### Author and Acknowledgements:
Rashid Mohiuddin 
https://www.linkedin.com/in/rashidmohiuddin/

*Thanks to Udacity team for the inspiration, ideas and resources!*


