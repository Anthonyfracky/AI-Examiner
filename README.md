# AI Examiner - NLP Course Examination System

## Overview

AI Examiner is an interactive examination system for a Natural Language Processing (NLP) course, designed to conduct oral examinations using AI technology. The system allows students to take an exam by answering randomly selected questions from a predefined set of NLP-related topics.

## Features

- Automated oral examination for NLP course
- Random selection of 3 questions from a comprehensive theme list
- Student authentication using a predefined student list
- AI-powered examination process
- Conversation history and score tracking
- Result logging and JSON export

## Prerequisites

- Python 3.8+
- Gradio
- OpenAI/Groq API
- python-dotenv

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Anthonyfracky/AI-Examiner.git
cd AI-Examiner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. API Configuration
- The `.env` file with API credentials is included in the repository
- No additional configuration is required for this educational project

## Usage

Run the application:
```bash
python main.py
```

## Project Structure

- `main.py`: Core application logic
- `.env`: Environment configuration with API key
- `requirements.txt`: Python dependencies
- `students.txt`: List of eligible students
- `themes.txt`: NLP examination themes
- `assets/`: User interface images
  - `user.png`: User avatar
  - `assistant.png`: AI assistant avatar

## Exam Process

1. Enter your email and full name
2. Answer 3 randomly selected NLP-related questions
3. Receive real-time feedback
4. Get a final score (0-10)

## Exam Results

- Results are saved as JSON files in the `exam_results/` directory
- Includes session details, conversation history, and student score

## Technologies

- Python
- Gradio
- OpenAI/Groq API
- Langchain
- Tool-use AI models