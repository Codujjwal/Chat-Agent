# Titanic Dataset Analysis Chatbot 🚢

An interactive Streamlit application that allows users to explore and analyze the Titanic dataset through natural language queries and data visualizations.

## Features

- Natural language queries about Titanic dataset statistics
- Interactive data visualizations (histograms, pie charts, etc.)
- Pre-built example questions for quick insights
- Responsive chat interface
- Real-time data analysis

## Technologies Used

- Python 3.11
- Streamlit for web interface
- Pandas for data manipulation
- Plotly for interactive visualizations
- Streamlit for the user interface

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/titanic-analysis-chatbot.git
cd titanic-analysis-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

- Click on example questions in the sidebar or type your own questions
- Ask about passenger statistics, demographics, or request visualizations
- Explore survival rates, age distributions, and ticket fare information

## Project Structure

- `app.py` - Main Streamlit application
- `chat_agent.py` - Query processing and response generation
- `data_loader.py` - Dataset loading and preprocessing
- `visualization.py` - Data visualization functions
- `utils.py` - Helper functions

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
