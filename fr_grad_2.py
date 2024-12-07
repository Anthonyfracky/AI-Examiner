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
        self.student_info = {"name": "", "email": ""}
        self.chat_history = []
        self.current_themes = []
        self.current_theme_index = 0
        self.examination_started = False

        # Initialize Groq language model
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.7,
            max_tokens=4096
        )

        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )

        # Create examination chain
        self.exam_chain = self._create_exam_chain()

    def _create_exam_chain(self):
        """Create a LangChain chain for examination."""
        template = """Ти є екзаменатором з курсу Natural Language Processing.
        Поточна тема для обговорення: {theme}

        Правила проведення іспиту:
        1. Задавай уточнюючі питання якщо відповідь студента неповна
        2. Оцінюй глибину розуміння теми
        3. Якщо студент не може відповісти або відповідає некоректно, переходь до наступної теми
        4. В кінці кожної теми став оцінку від 0 до 10 за відповідь

        Історія бесіди:
        {chat_history}

        Відповідь студента: {student_input}

        Твоя відповідь має бути в контексті іспиту."""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        return chain

    async def process_message(self, message: str, history: List[List[str]]) -> tuple:
        """Process incoming messages and maintain conversation state."""
        if not self.student_info["name"]:
            self.student_info["name"] = message
            return self._format_message("assistant", "Дякую! Тепер введіть ваш email:"), history

        if not self.student_info["email"]:
            self.student_info["email"] = message
            self.current_themes = self.start_exam(self.student_info["email"], self.student_info["name"])
            self.examination_started = True
            return self._format_message("assistant",
                                        f"Починаємо іспит.\n\nПерше питання: {self.current_themes[0]}"), history

        if self.examination_started:
            try:
                # Process examination dialogue
                chat_history = self.memory.load_memory_variables({})["chat_history"]
                response = await self.exam_chain.ainvoke({
                    "theme": self.current_themes[self.current_theme_index],
                    "student_input": message,
                    "chat_history": chat_history
                })

                # Update memory
                self.memory.save_context({"input": message}, {"output": response.content})

                # Check if we need to move to next theme
                if "оцінка:" in response.content.lower() or "оцінка :" in response.content.lower():
                    self.current_theme_index += 1
                    if self.current_theme_index < len(self.current_themes):
                        response.content += f"\n\nНаступна тема: {self.current_themes[self.current_theme_index]}"
                    else:
                        # Get final evaluation
                        final_template = """На основі всієї бесіди, надай фінальну оцінку студенту.

                        Формат оцінки:
                        1. Загальна оцінка (0-10)
                        2. Сильні сторони
                        3. Області для покращення

                        Історія бесіди:
                        {chat_history}"""

                        final_prompt = ChatPromptTemplate.from_template(final_template)
                        final_chain = final_prompt | self.llm
                        final_response = await final_chain.ainvoke({
                            "chat_history": chat_history
                        })
                        response.content += "\n\nІспит завершено!\n" + final_response.content
                        self.examination_started = False

                return self._format_message("assistant", response.content), history

            except Exception as e:
                error_message = f"Помилка при обробці повідомлення: {str(e)}"
                return self._format_message("assistant", error_message), history

        return self._format_message("assistant", "Іспит завершено. Дякуємо за участь!"), history

    def start_exam(self, email: str, name: str) -> List[str]:
        """Start the examination process."""
        themes = load_themes()
        return random.sample(themes, min(3, len(themes)))

    def _format_message(self, role: str, content: str) -> Dict:
        """Format a message with timestamp."""
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