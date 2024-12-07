import gradio as gr
import json
from datetime import datetime
from typing import List, Dict
import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
from datetime import datetime
import random
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Check for Groq API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Groq API key not found! Please set GROQ_API_KEY environment variable.")


def load_themes(file_path: str = "themes.txt") -> List[str]:
    """Load examination themes from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return ["Sample theme 1", "Sample theme 2", "Sample theme 3"]


class AIExaminer:
    def __init__(self):
        # Ініціалізація LLM з нижчою температурою для більш детермінованої поведінки
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.3,  # Знижена температура
            max_tokens=4096
        )

        # Ініціалізація стану екзамену
        self.exam_state = {
            "started": False,
            "name": None,
            "email": None,
            "current_questions": [],
            "current_question_index": 0,
            "scores": []
        }

        self.memory = ConversationBufferWindowMemory(k=10)

    def start_exam(self, email: str, name: str) -> List[str]:
        """Розпочати екзамен"""
        questions = [
            "Поясніть архітектуру моделей Transformer",
            "Що таке механізм уваги (attention) і як він працює?",
            "Опишіть процес токенізації в NLP",
            "Що таке embeddings і як вони використовуються в NLP?",
            "Поясніть концепцію fine-tuning у мовних моделях"
        ]

        self.exam_state["started"] = True
        self.exam_state["name"] = name
        self.exam_state["email"] = email
        self.exam_state["current_questions"] = questions
        self.exam_state["current_question_index"] = 0

        return questions

    def end_exam(self):
        """Завершити екзамен"""
        average_score = sum(self.exam_state["scores"]) / len(self.exam_state["scores"]) if self.exam_state[
            "scores"] else 0

        result = {
            "name": self.exam_state["name"],
            "email": self.exam_state["email"],
            "score": round(average_score, 2),
            "timestamp": datetime.now().isoformat()
        }

        # Скидання стану екзамену
        self.exam_state = {
            "started": False,
            "name": None,
            "email": None,
            "current_questions": [],
            "current_question_index": 0,
            "scores": []
        }

        return result

    async def process_message(self, message: str, history: List[List[str]]) -> tuple:
        # Якщо екзамен не розпочато
        if not self.exam_state["started"]:
            # Перевірка чи є повне ім'я та email
            parts = message.split()
            if len(parts) >= 2 and '@' in message:
                name = ' '.join(parts[:-1])
                email = parts[-1]

                # Розпочати екзамен
                questions = self.start_exam(email, name)
                response = f"Вітаю, {name}! Ми розпочинаємо іспит з NLP.\n\n"
                response += f"Перше питання: {questions[0]}"

                return self._format_message("assistant", response), history
            else:
                # Запит на ім'я та email
                return self._format_message("assistant",
                                            "Будь ласка, введіть ваше повне ім'я та email (наприклад: Іван Петров ivan@example.com)"), history

        # Якщо питання ще є
        if self.exam_state["current_question_index"] < len(self.exam_state["current_questions"]):
            # Оцінювання відповіді на поточне питання
            current_question = self.exam_state["current_questions"][self.exam_state["current_question_index"]]

            # Простий механізм оцінювання (можна замінити на більш складний)
            score = self._evaluate_answer(message, current_question)
            self.exam_state["scores"].append(score)

            # Перехід до наступного питання
            self.exam_state["current_question_index"] += 1

            # Якщо питання закінчилися
            if self.exam_state["current_question_index"] >= len(self.exam_state["current_questions"]):
                result = self.end_exam()
                response = f"Іспит завершено! Ваш бал: {result['score']}/10"
                return self._format_message("assistant", response), history

            # Наступне питання
            next_question = self.exam_state["current_questions"][self.exam_state["current_question_index"]]
            response = f"Оцінка за попереднє питання: {score}/10\n\n"
            response += f"Наступне питання: {next_question}"

            return self._format_message("assistant", response), history

    def _evaluate_answer(self, answer: str, question: str) -> float:
        """Базова функція оцінювання відповіді"""
        # Тимчасова проста логіка оцінювання
        if len(answer.split()) < 5:
            return 2.0
        elif len(answer.split()) < 10:
            return 5.0
        elif len(answer.split()) < 20:
            return 7.0
        else:
            return 9.0

    def _format_message(self, role: str, content: str) -> Dict:
        """Форматування повідомлення"""
        return {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M")
        }


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
            bot_message = await examiner.process_message(message, chat_history)
            chat_history.append((message, bot_message[0]["content"]))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
