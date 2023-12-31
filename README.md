# Nigeria Language
![App Screenshot](nigerian-language-rdnkws0p2bp.streamlit.app.png)

## Description
The Nigeria Major Languages App is a Streamlit web application that uses Natural Language Processing (NLP) to detect the major languages spoken in Nigeria based on user input text. The app is trained on a dataset containing texts in Yoruba, Igbo, Hausa, and Pidgin languages. The trained model predicts the language of the input text, providing users with insights into the language used.

Note: To get optimum results, please ensure that the input text is written in one of the four languages: Yoruba, Igbo, Hausa, or Pidgin. The accuracy of the language prediction may vary if non-Nigerian languages or mixed languages are used.

## Live Demo
You can access the live demo of the app [here](https://nigerian-language-knbkcvbjluo.streamlit.app/).

## Table of Contents
- [Description](#description)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run the app locally on your machine, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/AbiemwenseMaureenOshobugie/Nigerian-Language.git
    ```
2. Navigate to the project directory:
    ```
    cd Nigerian-Language
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
To run the app, execute the following command in the terminal:

    streamlit run nigerian_language.py

The app will be available at [http://localhost:8501](http://localhost:8501) in your web browser.

## Technologies Used
The Nigeria Major Languages App is built using the following technologies:

- Streamlit: A web framework for creating interactive web applications with Python.
- Python: The programming language used for the app's backend.
- Scikit-learn: A machine learning library used for language detection and text preprocessing.
- Pandas: A library for data manipulation and analysis.
- Matplotlib: A plotting library for creating visualizations.
- NLTK: The Natural Language Toolkit for natural language processing tasks.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Author

Abiemwense Maureen Oshobugie

## License
This project is licensed under the [MIT License](LICENSE).

Please note that the app requires the **nigerian_language.py**, **best_model.pkl**, and **requirements.txt** files to be present in the same directory for it to run successfully. Make sure to download the files from the provided GitHub repository and place them in the appropriate location before running the app.

If you have any questions or need further assistance, feel free to reach out!

