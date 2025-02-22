import gradio as gr
import os
import openai
from dotenv import load_dotenv
import io
import pypdf
import json

load_dotenv()

# Добавляем ЛОГИРОВАНИЕ для проверки API-ключа
print("=== ЗАПУСК ПРИЛОЖЕНИЯ ===")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"OPENAI_API_KEY найден: {api_key[:5]}... (первые 5 символов)")  # Показываем только начало ключа
else:
    print("ОШИБКА: OPENAI_API_KEY не найден в переменных окружения!")
    gr.Error("OPENAI_API_KEY не найден! Установите переменную окружения.")

openai.api_key = api_key

def chat(message, history, file_obj=None, model_name="gpt-3.5-turbo"):
    print("=== НОВОЕ СООБЩЕНИЕ ===")
    print(f"Входное сообщение: {message}, Модель: {model_name}")

    history = history or []
    history.append({"role": "user", "content": message})
    print(f"История (после добавления сообщения пользователя):\n{json.dumps(history, indent=2)}")

    if file_obj:
        print(f"Загружен файл: {file_obj.name}")
        try:
            if file_obj.name.lower().endswith(".pdf"):
                with open(file_obj.name, "rb") as f:
                    pdf_reader = pypdf.PdfReader(f)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                history[-1]["content"] += f"\n\nСодержимое PDF:\n{pdf_text}"
                print("PDF успешно прочитан.")
            else:
                gr.Warning(f"Файл {file_obj.name} не является PDF. Поддерживаются только PDF.")
                history[-1]["content"] += f"\n\nФайл {file_obj.name} прикреплен, но не обработан (поддерживаются только PDF)."
        except Exception as e:
            gr.Error(f"Ошибка при обработке файла: {e}")
            print(f"Ошибка при обработке файла: {e}")
            history[-1]["content"] += f"\n\nОшибка при обработке файла: {e}"

    try:
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        request_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0,
            "max_tokens": 4096,
        }
        print(f"Запрос к API OpenAI:\n{json.dumps(request_payload, indent=2)}")

        # Добавляем ЛОГИРОВАНИЕ ПЕРЕД отправкой запроса
        print("=== ОТПРАВКА ЗАПРОСА В OPENAI API ===")

        response_stream = openai.chat.completions.create(**request_payload)

        full_response = ""
        for chunk in response_stream:
            print(f"Получен chunk от API: {chunk}")
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content

        history.append({"role": "assistant", "content": full_response})
        print(f"Полный ответ от API: {full_response}")

        messages = []
        for i in range(0, len(history) - 1, 2):
            messages.append((history[i]["content"], history[i+1]["content"]))
        print(f"messages перед yield: {messages}")
        yield "", messages

    except Exception as e:
        gr.Error(f"Ошибка API OpenAI: {e}")
        print(f"Ошибка API OpenAI: {e}")  # Полный текст ошибки
        print(f"Тип ошибки: {type(e)}") # Добавил
        if isinstance(e, openai.APIError): #Добавил
            print(f"Детали ошибки OpenAI: {e.json_body}") #Добавил
        yield f"Error: {e}", []

def clear_history():
    print("=== ОЧИСТКА ИСТОРИИ ===")
    return None, [], None

def get_available_models():
    try:
        models = openai.models.list()
        print(f"Получен список моделей: {[model.id for model in models]}") # Добавил
        return [model.id for model in models]
    except Exception as e:
        print(f"Ошибка при получении списка моделей: {e}")
        return ["gpt-3.5-turbo"]

with gr.Blocks(title="OpenAI Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Chat with OpenAI Models")
    available_models = get_available_models()
    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=available_models,
        value=available_models[0] if available_models else "gpt-3.5-turbo",
    )
    chatbot = gr.Chatbot(label="Chat", value=[], height=550)
    with gr.Row():
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here, or upload a PDF...",
            autofocus=True,
            lines=2,
            scale=4,
        )
        file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1)

    with gr.Row():
        send_button = gr.Button("Send")
        clear = gr.ClearButton([msg, chatbot, file_upload])
        clear_hist_button = gr.Button("Clear History")

    msg.submit(chat, [msg, chatbot, file_upload, model_dropdown], [msg, chatbot])
    file_upload.upload(chat, [msg, chatbot, file_upload, model_dropdown], [msg, chatbot])
    clear_hist_button.click(clear_history, [], [msg, chatbot, file_upload])
    send_button.click(chat, [msg, chatbot, file_upload, model_dropdown], [msg, chatbot])
    demo.load(None, [], [chatbot])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
