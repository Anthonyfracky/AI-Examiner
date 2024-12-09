import os
import json
import random
import logging
import gradio as gr
from openai import OpenAI
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

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
        self.reset_exam_state()
        self.questions = load_questions()

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "start_exam",
                    "description": "Start a new examination session for a student. Returns a list of questions for the session, or an error if the student not registered in list of students for this exam.",
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
                        "required": ["email", "score", "history"]
                    }
                }
            }
        ]

    def reset_exam_state(self):
        """Reset all exam-related state variables."""
        self.conversation_history = []
        self.examination_active = False
        self.current_email = None
        self.current_name = None
        self.current_question_index = 0
        self.questions_for_session = []
        self.answers_received = 0
        self.exam_completed = False
        self.session_id = None
        self.last_exam_summary = None

    def start_exam(self, email: str, name: str) -> List[str]:
        """Start a new examination session for a student."""
        with open('students.txt', 'r', encoding='utf-8') as f:
            students = [line.strip() for line in f if line.strip()]

        if name not in students:
            return [f"Student {name} not found in the students list."]

        self.reset_exam_state()
        self.examination_active = True
        self.current_email = email
        self.current_name = name
        self.current_question_index = 0
        self.answers_received = 0
        self.questions_for_session = random.sample(self.questions, 3)
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.questions_for_session

    def end_exam(self, email: str, score: float, history: List[Dict]) -> Dict:
        """
        End the examination session and record the results with robust history processing.

        Args:
            email (str): Student's email address
            score (float): Final exam score
            history (List[Dict]): Conversation history

        Returns:
            Dict: Exam session summary
        """
        logging.basicConfig(
            filename='exam_debug.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        if not self.examination_active:
            logging.warning("Attempt to end non-active examination")
            return {"error": "No active examination session"}

        if email != self.current_email:
            logging.error(f"Email mismatch: {email} vs {self.current_email}")
            return {"error": "Email mismatch with current session"}

        def standardize_history(raw_history):
            """
            Converts different chat history formats to a unified format.

            Args:
                raw_history (List[Any]): Input chat history

            Returns:
                List[Dict]: Standardized history
            """
            standardized_history = []
            current_time = datetime.now().isoformat()

            try:
                for item in raw_history:
                    if isinstance(item, tuple) and len(item) == 2:
                        user_msg, assistant_msg = item
                        standardized_history.extend([
                            {
                                "role": "user",
                                "content": str(user_msg),
                                "timestamp": current_time
                            },
                            {
                                "role": "assistant",
                                "content": str(assistant_msg),
                                "timestamp": current_time
                            }
                        ])

                    elif isinstance(item, dict):
                        if item.get('content'):
                            standardized_history.append({
                                "role": item.get('role', 'unknown'),
                                "content": str(item.get('content', '')),
                                "timestamp": current_time
                            })

                    elif isinstance(item, list):
                        for subitem in item:
                            if isinstance(subitem, dict) and subitem.get('content'):
                                standardized_history.append({
                                    "role": subitem.get('role', 'unknown'),
                                    "content": str(subitem.get('content', '')),
                                    "timestamp": current_time
                                })

            except Exception as e:
                logging.error(f"Error in standardizing history: {e}", exc_info=True)
                standardized_history = [{
                    "role": "system",
                    "content": f"Error processing conversation: {str(e)}",
                    "timestamp": current_time
                }]

            return standardized_history

        try:
            formatted_history = standardize_history(history)

            logging.debug(f"Processed history length: {len(formatted_history)}")

            if not formatted_history:
                logging.warning("No conversation history available")
                formatted_history = [{
                    "role": "system",
                    "content": "No conversation history captured",
                    "timestamp": datetime.now().isoformat()
                }]

        except Exception as e:
            logging.error(f"Unexpected error processing history: {e}", exc_info=True)
            formatted_history = [{
                "role": "system",
                "content": f"Critical error in history processing: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }]

        summary = {
            "session_id": self.session_id or datetime.now().strftime("%Y%m%d%H%M%S"),
            "student_name": self.current_name or "Unknown",
            "student_email": email,
            "score": max(0, min(score, 10)),
            "timestamp": datetime.now().isoformat(),
            "conversation_history": formatted_history
        }

        os.makedirs("exam_results", exist_ok=True)

        filename = f"exam_results/{self.session_id or datetime.now().strftime('%Y%m%d%H%M%S')}_{email.replace('@', '_')}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            logging.info(f"Exam results successfully saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to write exam results: {e}", exc_info=True)
            return {"error": f"Failed to write exam results: {str(e)}"}

        self.exam_completed = True
        self.examination_active = False
        self.last_exam_summary = summary

        return summary

    def process_message(self, message, chat_history):
        """Process a new message using the OpenAI API and update chat history."""
        if self.exam_completed:
            return [{
                "content": "The examination has been completed. The session is now closed. Press Take Exam Again if you would like to take another examination."}]

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
                - Keep track of which question number you're on
                - Explicitly state when moving to the next question
                - After the third answer and providing the score, end the session completely"""
            }
        ]

        for human_msg, assistant_msg in chat_history:
            messages.extend([
                {"role": "user", "content": human_msg},
                {"role": "assistant", "content": assistant_msg}
            ])

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        if response.choices[0].message.tool_calls:
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
                    result = self.end_exam(args['email'], args['score'], messages)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })

            response = self.client.chat.completions.create(
                model="llama3-groq-70b-8192-tool-use-preview",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

        if self.examination_active and "answer" in message.lower():
            self.answers_received += 1

        return [{"content": response.choices[0].message.content}]


def create_interface() -> gr.Blocks:
    examiner = AIExaminer()

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Examiner - NLP Course")
        gr.Markdown("Welcome to the Natural Language Processing exam!")
        gr.Markdown("Enter any message to start the examination ✍️")

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

        retake_btn = gr.Button("Take Exam Again", visible=True)

        async def respond(message, chat_history):
            bot_message = examiner.process_message(message, chat_history)
            chat_history.append((message, bot_message[0]["content"]))

            return "", chat_history

        async def retake_exam():
            examiner.reset_exam_state()
            return "", [], gr.update(visible=True)

        msg.submit(respond, [msg, chatbot], [msg, chatbot], show_progress=False)
        retake_btn.click(retake_exam, None, [msg, chatbot, retake_btn], show_progress=False)

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()