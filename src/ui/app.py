import gradio as gr

from src.configs.constants import USERNAME, PASSWORD
from src.services.aws.opensearch.opensearch_utils import list_docs, \
    document_vectorize, get_document
from src.services.llm.llm import get_rag_chain
from src.services.logger.logger_config import logger


def build_history(history: list[str]):
    if history is None or not history:
        return ""
    return "".join(f"User: {q}\nChatbot: {a}\n" for q, a in history)


def fix_auth(username, password):
    if username == USERNAME and password == PASSWORD:
        return True
    return False


def handle_uploaded_file(file):
    logger.info(f"Uploaded file: {file.name}")
    return file.name


def handle_dropdown(doc):
    logger.info(f"Selected document: {doc}")
    return doc


async def handle_question(question, history, uploaded_file, document):
    if uploaded_file:
        doc = document_vectorize(uploaded_file)
    elif document:
        doc = get_document(document)
    else:
        raise gr.Error("Please upload a file or select a document")

    # do rag
    retriever = doc.as_retriever(search_kwargs={"k": 5})
    rag_chain = get_rag_chain(retriever)
    response = await rag_chain.ainvoke(
        {"question": question, "chat_history": build_history(history)})
    history.append((question, response))
    return "", history


with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    file_input = gr.File(label="Upload a file")
    file_input.upload(handle_uploaded_file, file_input)

    with gr.Row(equal_height=True):
        with gr.Column():
            dropdown = gr.Dropdown(list_docs(),
                                   label="Or choose an uploaded document")
        with gr.Column():
            reload_btn = gr.Button("Reload Options")
    dropdown.select(handle_dropdown, dropdown)
    reload_btn.click(lambda: gr.Dropdown(choices=list_docs()), [],
                     [dropdown])

    message = gr.Textbox(lines=1, label="Input your question")
    clean_btn = gr.ClearButton([message, chatbot])

    message.submit(handle_question, [message, chatbot, file_input, dropdown],
                   [message, chatbot])
app.launch(auth=fix_auth,
           share=False,
           auth_message="Please login to access the app",
           server_name="0.0.0.0",
           server_port=8080)
