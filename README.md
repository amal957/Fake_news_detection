📰 Fake News Detection System

🌟 Project Overview

The Fake News Detection System is a machine learning-powered web application designed to identify and classify fake news articles. 
This project uses Natural Language Processing (NLP) techniques, particularly the TF-IDF Vectorizer, to transform text data and various classification algorithms to predict whether a given news article is Real or Fake.

🚀 Features

🌐 Web Interface: A user-friendly web interface built using Streamlit, enabling users to input news headlines or articles for analysis.

🧠 Machine Learning Models:

XGBoost Classifier: Achieved an accuracy of 96.22%

Passive Aggressive Classifier: Achieved an accuracy of 95.47%

📊 Performance Metrics: Evaluated using metrics like accuracy, precision, recall, and F1-score.

📄 Pre-trained Models: Utilizes models saved in .pkl files for quick predictions.

📚 Dataset: Trained on the WELFake dataset containing over 45,000 articles.

📂 Dataset

The dataset used is the WELFake dataset, containing over 45,000 articles labeled as Real or Fake. The data includes various features such as:

Title
Text content
Label (Real or Fake)


🛠️ How It Works

Text Preprocessing:

Removal of URLs, special characters, and digits.

Conversion to lowercase.

Removal of English stopwords using NLTK.

Feature Extraction:

Uses TF-IDF Vectorizer to transform text data into numerical features.

Model Training:

The dataset was split into training and testing sets.

Multiple classifiers were trained and evaluated for optimal performance.

Prediction:

Users can input news articles via the web interface to get predictions on whether the news is Real or Fake.

📈 Usage
Upload a news article or enter the headline in the input box.

Click on the "Analyze" button.

The model will predict whether the news is Real or Fake.

📸 Project Screenshots

Below are some screenshots of the Fake News Detection System in action:

1. Home Page
 ![image](https://github.com/user-attachments/assets/a5bb21d5-2875-493c-8813-9b595080f61e)



2. News Prediction
    ![image](https://github.com/user-attachments/assets/1ac5335f-9a11-4d9b-aaec-f3cd7442aa83)

🔧 Future Enhancements

 Integrate BERT-based models for better text understanding.
 
 Add support for multilingual fake news detection.
 
 Deploy the application on cloud platforms like AWS or Heroku.
 
 Enable API support for third-party integrations.

 
📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

👤 Author

GitHub: @amal957

Email: amalashok957@gmail.com


🌐 Acknowledgments

Scikit-Learn for machine learning models

Kaggle for providing the WELFake dataset

Inspiration from various open-source projects

