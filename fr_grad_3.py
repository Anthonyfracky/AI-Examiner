import logging
import gradio as gr
import json
from datetime import datetime
from typing import List, Dict, Optional
import random
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_questions(file_path: str = "themes.txt") -> List[str]:
    """Load examination themes from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return ["Sample theme 1", "Sample theme 2", "Sample theme 3"]


class AIExaminer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.conversation_history = []
        self.examination_active = False
        self.current_email = None
        self.current_name = None
        self.questions = load_questions()
        self.current_question_index = 0
        self.max_questions = 5
        self.current_questions = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "start_exam",
                    "description": "Start a new examination session for a student",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Student's email address"
                            },
                            "name": {
                                "type": "string",
                                "description": "Student's full name"
                            }
                        },
                        "required": ["email", "name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end_exam",
                    "description": "End the examination session and record the results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Student's email address"
                            },
                            "score": {
                                "type": "number",
                                "description": "Final score (0-10)"
                            }
                        },
                        "required": ["email", "score"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_next_question",
                    "description": "Get the next question for the examination",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def start_exam(self, email: str, name: str) -> str:
        """Start a new examination session for a student."""
        self.examination_active = True
        self.current_email = email
        self.current_name = name
        self.current_question_index = 0
        self.conversation_history = []
        selected_questions = random.sample(self.questions, self.max_questions)
        self.current_questions = selected_questions

        # Add initial message to conversation history
        initial_message = {
            "role": "system",
            "content": f"Examination started for {name} ({email})",
            "datetime": datetime.now().isoformat()
        }
        self.conversation_history.append(initial_message)

        return f"Starting exam for {name} ({email}). Selected {len(selected_questions)} questions."

    def get_next_question(self) -> Optional[str]:
        """Get the next question in the examination."""
        if not self.examination_active or self.current_question_index >= len(self.current_questions):
            return None
        question = self.current_questions[self.current_question_index]
        self.current_question_index += 1
        return question

    def end_exam(self, email: str, score: float, history: List[Dict]) -> None:
        """End the examination session and record the results."""
        if not self.examination_active:
            return {"error": "No active examination session"}

        if email != self.current_email:
            return {"error": "Email mismatch with current session"}

        # Normalize score to 0-10 scale if needed
        if score > 10:
            score = score / 10

        # Add final message to history
        final_message = {
            "role": "system",
            "content": f"Examination ended for {self.current_name} ({email}). Final score: {score}/10",
            "datetime": datetime.now().isoformat()
        }
        self.conversation_history.append(final_message)

        # Reset the examination state
        self.examination_active = False
        self.current_email = None
        self.current_name = None
        self.current_question_index = 0
        return self.conversation_history

    def process_message(self, message: str, chat_history: List) -> List[Dict]:
        """Process a new message using the OpenAI API and update chat history."""
        # Add user message to conversation history
        user_message = {
            "role": "user",
            "content": message,
            "datetime": datetime.now().isoformat()
        }
        self.conversation_history.append(user_message)

        messages = [
            {
                "role": "system",
                "content": """You are an AI examiner for a Natural Language Processing course. 
                Your role is to conduct oral examinations and evaluate students' knowledge.

                Follow these guidelines:
                1. When a student provides their email and name, use the start_exam tool
                2. Use get_next_question tool to get the next question
                3. Ask one question at a time and wait for the student's response
                4. After each answer, provide feedback
                5. After 5 questions are answered:
                   - Calculate a final score (0-10 scale)
                   - Use the end_exam tool with email and score only

                Keep track of:
                - Number of questions asked
                - Quality of answers
                - Overall performance

                Be professional and supportive while maintaining examination integrity."""
            }
        ]

        # Add conversation history to messages
        for msg in self.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = self.client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            assistant_message = {
                "role": "system",
                "content": response.choices[0].message.content,
                "datetime": datetime.now().isoformat()
            }
            self.conversation_history.append(assistant_message)

            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if function_name == 'start_exam':
                    result = self.start_exam(args['email'], args['name'])
                elif function_name == 'get_next_question':
                    result = self.get_next_question()
                elif function_name == 'end_exam':
                    result = self.end_exam(
                        args['email'],
                        float(args['score']),
                        self.conversation_history
                    )

                tool_message = {
                    "role": "function",
                    "content": str(result),
                    "datetime": datetime.now().isoformat()
                }
                self.conversation_history.append(tool_message)

            # Get final response after tool use
            response = self.client.chat.completions.create(
                model="llama3-groq-70b-8192-tool-use-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

        # Add assistant's response to conversation history
        assistant_message = {
            "role": "system",
            "content": response.choices[0].message.content,
            "datetime": datetime.now().isoformat()
        }
        self.conversation_history.append(assistant_message)

        return [{"content": assistant_message["content"]}]


def create_interface() -> gr.Blocks:
    examiner = AIExaminer()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Examiner - NLP Course")
        gr.Markdown("Вітаємо на іспиті з курсу Natural Language Processing")

        chatbot = gr.Chatbot(
            show_label=False,
            avatar_images=["assets/user.png", "assets/assistant.png"],
            height=600
        )

        msg = gr.Textbox(
            show_label=False,
            placeholder="Введіть повідомлення...",
            container=False
        )

        async def respond(message, chat_history):
            bot_message = examiner.process_message(message, chat_history)
            chat_history.append((message, bot_message[0]["content"]))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()