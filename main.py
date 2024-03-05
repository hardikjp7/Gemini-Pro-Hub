import google.generativeai as genai
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

from gemini import (load_gemini_pro_model,
                            gemini_pro_vision_response,
                            embeddings_model_response)

st.set_page_config(
    page_title="Gemini Hub",
    page_icon="logo.png",
    layout="centered",
)

with st.sidebar:
    API_KEY = st.sidebar.text_input("Gemini API Key", type="password")
    genai.configure(api_key=API_KEY)
    if not API_KEY:
        st.warning("Please enter your Gemini API Key.")


    selected = option_menu('Gemini Hub',
                           ['ChatBot',
                            'Image Captioning',
                            'Embed text',
                            'About'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )
    
    # Guide for obtaining Google API Key if not available
    st.subheader("Don't have a Gemini API Key?")
    st.write("Visit Google [AiStudio](https://aistudio.google.com/app/apikey) and log in with your Google account. Then click on 'Create API Key'.")

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# chatbot page
if selected == 'ChatBot':
    model = load_gemini_pro_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:  # Renamed for clarity
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")  # Renamed for clarity
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)  # Renamed for clarity

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image captioning page
if selected == "Image Captioning":

    st.title("Get Your Image Captioned 📷")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption"):
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((800, 500))
            st.image(resized_img)

        default_prompt = "Please provide a brief and descriptive caption for this image."  # change this prompt as per your requirement

        # get the caption of the image from the gemini-pro-vision LLM
        caption = gemini_pro_vision_response(default_prompt, image)

        with col2:
            st.info(caption)

# text embedding model
if selected == "Embed text":

    st.title("🔡 Embed Text")

    # text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        response = embeddings_model_response(user_prompt)
        st.markdown(response)

if selected == "About":
    st.title("About This GEMINI HUB")
    st.markdown("**Made by 😎 [Hardik](https://www.linkedin.com/in/hardikjp/)**")
    st.write("This GEMINI HUB integrates various features powered by Google's GEMINI models.")
    st.write("Each feature uses a different Gemini model:")
    st.write("- Chatbot: Utilizes the Gemini Pro model for conversational AI.")
    st.write("- Image Captioning: Uses the Gemini-Pro-Vision model for generating captions for images.")
    st.write("- Text Embedding: Utilizes the embeddings-001 model for converting text to embeddings.")
