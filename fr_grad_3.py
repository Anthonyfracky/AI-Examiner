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
        self.questions_for_session = []
        self.answers_received = 0
        self.exam_completed = False
        self.session_id = None

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
                            },
                            "history": {
                                "type": "array",
                                "description": "List of conversation messages",
                                "items": {
                                    "type": "object"
                                }
                            }
                        },
                        "required": ["email", "score", "history"]
                    }
                }
            }
        ]

    def start_exam(self, email: str, name: str) -> List[str]:
        """Start a new examination session for a student."""
        self.examination_active = True
        self.exam_completed = False
        self.current_email = email
        self.current_name = name
        self.current_question_index = 0
        self.answers_received = 0
        self.questions_for_session = random.sample(self.questions, 3)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.questions_for_session

    def end_exam(self, email: str, score: float, history: List[dict]) -> dict:
        """End the examination session and record the results."""
        if not self.examination_active:
            return {"error": "No active examination session"}

        if email != self.current_email:
            return {"error": "Email mismatch with current session"}

        summary = {
            "session_id": self.session_id,
            "student_name": self.current_name,
            "student_email": email,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": history
        }

        # Mark exam as completed before resetting variables
        self.exam_completed = True

        # Reset all session-related variables
        self.examination_active = False
        self.current_email = None
        self.current_name = None
        self.current_question_index = 0
        self.questions_for_session = []
        self.answers_received = 0
        self.conversation_history = []
        self.session_id = None

        return summary

    def process_message(self, message, chat_history):
        """Process a new message using the OpenAI API and update chat history."""
        # If exam is completed, return a standard message
        if self.exam_completed:
            return [{
                        "content": "The examination has been completed. The session is now closed. Please start a new session if you would like to take another examination."}]

        messages = [
            {
                "role": "system",
                "content": """You are an AI examiner for a Natural Language Processing course. 
                Your role is to conduct oral examinations and evaluate students' knowledge.

                Follow these guidelines:
                1. When a student provides their email and name, use the start_exam tool to begin the examination
                2. Ask one question at a time and wait for the student's response
                3. Evaluate each answer thoroughly and provide feedback
                4. Track the number of questions asked and answers received
                5. After receiving the answer to the last question (third question):
                   - Provide final feedback
                   - Calculate the final score (0-10)
                   - Use the end_exam tool with the calculated score
                   - End the conversation with a farewell message
                   DO NOT continue the conversation after using end_exam

                Remember to:
                - Be professional and supportive
                - Give clear feedback after each answer
                - Maintain examination integrity
                - Use appropriate academic language
                - Keep track of which question number you're on (1, 2, or 3)
                - Explicitly state when moving to the next question
                - After the third answer and providing the score, end the session completely"""
            }
        ]

        # Add chat history
        for human_msg, assistant_msg in chat_history:
            messages.extend([
                {"role": "user", "content": human_msg},
                {"role": "assistant", "content": assistant_msg}
            ])

        # Add current message
        messages.append({"role": "user", "content": message})

        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        # Handle tool calls if present
        if response.choices[0].message.tool_calls:
            # First append the assistant's message that contains the tool calls
            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            })

            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == 'start_exam':
                    args = json.loads(tool_call.function.arguments)
                    result = self.start_exam(args['email'], args['name'])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
                elif tool_call.function.name == 'end_exam':
                    args = json.loads(tool_call.function.arguments)
                    result = self.end_exam(args['email'], args['score'], args['history'])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })

            # Get final response after tool use
            response = self.client.chat.completions.create(
                model="llama3-groq-70b-8192-tool-use-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

        # Update question tracking after processing the message
        if self.examination_active and "answer" in message.lower():
            self.answers_received += 1

        return [{"content": response.choices[0].message.content}]


def create_interface() -> gr.Blocks:
    examiner = AIExaminer()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Examiner - NLP Course")
        gr.Markdown("Welcome to the Natural Language Processing exam!")

        chatbot = gr.Chatbot(
            show_label=False,
            avatar_images=["assets/user.png", "assets/assistant.png"],
            height=600
        )

        msg = gr.Textbox(
            show_label=False,
            placeholder="Enter your message...",
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