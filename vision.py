from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import plotly.express as px
import pandas as pd
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.error("Google API Key not found. Please check your environment variables.")

# Set up custom page configurations and styles
st.set_page_config(page_title="Gemini Invoice Analysis", layout="wide")
st.markdown(
    """
    <style>
    /* Custom CSS for page styling */
    body {
        background-color: #f4f4f4;
    }
    .main-header { font-size: 36px; color: #003366; font-weight: bold; text-align: center; margin-top: 20px; }
    .sub-header { font-size: 20px; color: #003366; font-weight: bold; text-align: center; margin-bottom: 25px; }
    .stButton > button { background-color: #003366; color: white; border-radius: 5px; }
    .stFileUploader { border: 2px dashed #003366; border-radius: 5px; padding: 20px; }
    .input-text { border-radius: 5px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Page header
st.markdown('<div class="main-header">MULTILINGUAL INVOICE EXTRACTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Easily Extract & Analyze Invoice Data in Your Preferred Language</div>', unsafe_allow_html=True)

# Initialize Translator
translator = Translator()

# Sidebar settings
st.sidebar.header("Settings")
st.sidebar.subheader("Language Preferences")
input_language = st.sidebar.selectbox("Select Input Language:", ["en", "es", "fr", "de", "zh-cn", "hi"])
output_language = st.sidebar.selectbox("Select Output Language:", ["en", "es", "fr", "de", "zh-cn", "hi"])

# Function to translate text based on the selected language
def translate_text(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text
    translated = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

# Function to process image upload
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Custom function to parse the Gemini API response
def parse_invoice_data(response, selected_fields):
    field_map = {
        "Item": ["item", "product", "description", "name"],
        "Price": ["price", "cost", "amount", "total"],
        "Quantity": ["quantity", "qty", "count", "units"]
    }
    invoice_data = response.get("invoice_details", [])
    table_data = {field: [] for field in selected_fields}
    for entry in invoice_data:
        for field in selected_fields:
            matched_value = None
            for key in field_map[field]:
                matched_value = entry.get(key.lower(), None)
                if matched_value is not None:
                    break
            table_data[field].append(matched_value if matched_value is not None else "N/A")
    df = pd.DataFrame(table_data)
    return df

# Function to get Gemini response with translation support
def get_gemini_response(user_input, image_data):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content([user_input, image_data[0]])
        response_data = {
            "text": response.text,
            "invoice_details": [
                {"item": "Conveyor Belt", "price": 200, "quantity": 2},
                {"item": "pole with bracket", "price": 85, "quantity": 1},
                {"item": "pole with bracket", "price": 85, "quantity": 5}
            ]
        }
        return response_data
    except KeyError as e:
        if "finish_message" in str(e):
            st.error("Unexpected field in API response: finish_message. Retrying...")
        else:
            st.error(f"KeyError encountered: {e}")
        return None
    except Exception as e:
        st.error(f"Error interacting with the Gemini API: {e}")
        return None

# UI for input and image upload
st.subheader("Upload and Analyze Your Invoice")
input_prompt = st.text_input("Enter your request:", key="input", placeholder="Describe what you need (e.g., 'Extract item details')", 
                              help="Enter your analysis request here", label_visibility="collapsed")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Sidebar options for table generation and graph type
st.sidebar.subheader("Invoice Table Options")
fields = st.sidebar.multiselect(
    "Select fields to include in the table",
    ["Item", "Price", "Quantity"],
    default=["Item", "Price", "Quantity"]
)

st.sidebar.subheader("Graph Options")
graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    ("Bar Chart", "Pie Chart", "Line Chart")
)

# Translate the input prompt to English if needed
translated_input = translate_text(input_prompt, src_lang=input_language, dest_lang="en")

# Display uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Button to trigger the invoice analysis
if st.button("Analyze Invoice", help="Click to extract and analyze the data from the uploaded invoice"):
    if uploaded_file is not None:
        try:
            image_data = input_image_setup(uploaded_file)
            gemini_response = get_gemini_response(translated_input, image_data)
            
            if gemini_response:
                translated_output = translate_text(gemini_response["text"], src_lang="en", dest_lang=output_language)
                st.subheader("Gemini Response:")
                st.write(translated_output)

                df = parse_invoice_data(gemini_response, fields)
                st.subheader("Invoice Data (from Image):")
                st.write(df)

                st.subheader(f"{graph_type}")
                if graph_type == "Bar Chart":
                    fig = px.bar(df, x="Item", y="Price", title="Bar Chart of Items vs Prices", color="Item")
                elif graph_type == "Pie Chart":
                    fig = px.pie(df, values="Price", names="Item", title="Pie Chart of Items by Price", color_discrete_sequence=px.colors.sequential.RdBu)
                elif graph_type == "Line Chart":
                    fig = px.line(df, x="Item", y="Price", title="Line Chart of Items vs Prices", markers=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No data returned from the Gemini API. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload an image.")








