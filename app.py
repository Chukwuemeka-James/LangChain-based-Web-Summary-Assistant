import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Set page config with improved page title, icon, and layout
st.set_page_config(
    page_title="LangChain-based Web Summary Assistant",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title and Instructions
st.title("🦜 Web Summary Assistant")
st.markdown(
    "### Summarize content from websites in seconds! "
    "Just enter a valid URL, and let the AI summarize it for you."
)

# Sidebar for Settings
with st.sidebar:
    st.header("Settings")
    st.markdown("🔑 **API Configuration**")
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        help="Enter your Groq API key to access the model."
    )

# Check if the Groq API key is entered before showing the rest of the app
if groq_api_key.strip():
    st.markdown("---")
    st.markdown("📄 **Summarization Parameters**")

    # Input URL (YouTube or Website)
    generic_url = st.text_input(
        "Enter the URL of the content you want summarized:",
        placeholder="e.g. https://www.youtube.com/watch?v=xyz",
        help="Paste the URL of a YouTube video or a webpage for summarization."
    )

    # Groq Model Setup
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    # Summary prompt template
    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Button to trigger the summarization
    if st.button("Summarize Content"):
        if not generic_url.strip():
            st.error("Please provide a valid URL to continue.")
        elif not validators.url(generic_url):
            st.error("Invalid URL. Please enter a valid YouTube or website URL.")
        else:
            try:
                with st.spinner("Summarizing content, please wait..."):
                    # Load content from YouTube or website
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                    docs = loader.load()

                    # Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    # Display the summary
                    st.success("Here is your summary:")
                    st.write(output_summary)

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your Groq API key to continue.")
