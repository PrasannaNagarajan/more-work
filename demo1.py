import os
os.environ["OPENAI_API_KEY"] = '{your_api_key}'

import streamlit as st
from llama_index import download_loader
from llama_index import GPTSimpleKeywordTableIndex

from llama_index import LLMPredictor, GPTSimpleKeywordTableIndex, PromptHelper, ServiceContext
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json
import datetime
from html2image import Html2Image

doc_path = './data/'
transcript_file = './data/transcript.json'
index_file = 'index.json'
youtube_img = 'youtube.png'
youtube_link = ''

if 'video_id' not in st.session_state:
    st.session_state.video_id = ''

def send_click():
    st.session_state.video_id = youtube_link.split("v=")[1][:11]

index = None
st.title("Yeyu's Video Summarizer")

sidebar_placeholder = st.sidebar.container()
youtube_link = st.text_input("Youtube link:")
st.button("Summarize!", on_click=send_click)

if st.session_state.video_id != '':

    progress_bar = st.progress(5, text=f"Summarizing...")

    srt = YouTubeTranscriptApi.get_transcript(st.session_state.video_id, languages=['en'])
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(srt)
    with open(transcript_file, 'w') as f: 
        f.write(json_formatted)

    hti = Html2Image()
    hti.screenshot(url=f"https://www.youtube.com/watch?v={st.session_state.video_id}", save_as=youtube_img)
    
    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    sidebar_placeholder.header('Current Processing Video')
    sidebar_placeholder.image(youtube_img)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=500))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTSimpleKeywordTableIndex.from_documents(
        documents, service_context=service_context
    )
    index.save_to_disk(index_file)

    section_texts = ''
    section_start_s = 0

    with open(transcript_file, 'r') as f:
        transcript = json.load(f)    
    
    start_text = transcript[0]["text"]

    progress_steps = int(transcript[-1]["start"]/300+2)
    progress_period = int(100/progress_steps)
    progress_timeleft = str(datetime.timedelta(seconds=20*progress_steps))
    percent_complete = 5
    progress_bar.progress(percent_complete, text=f"Summarizing...{progress_timeleft} left")

    section_response = ''

    for d in transcript:
    
        if d["start"] <= (section_start_s + 300) and transcript.index(d) != len(transcript) - 1:
            section_texts += ' ' + d["text"]

        else:
            end_text = d["text"]

            prompt = f"summarize this article from \"{start_text}\" to \"{end_text}\", limited in 100 words, start with \"This section of video\""
            #print(prompt)
            response = index.query(prompt)
            start_time = str(datetime.timedelta(seconds=section_start_s))
            end_time = str(datetime.timedelta(seconds=int(d['start'])))

            section_start_s += 300
            start_text = d["text"]
            section_texts = ''
    
            section_response += f"**{start_time} - {end_time}:**\n\r{response}\n\r"      

            percent_complete += progress_period
            progress_steps -= 1
            progress_timeleft = str(datetime.timedelta(seconds=20*progress_steps))
            progress_bar.progress(percent_complete, text=f"Summarizing...{progress_timeleft} left")

    prompt = "Summarize this article of a video, start with \"This Video\", the article is: " + section_response
    #print(prompt)
    response = index.query(prompt)
    
    progress_bar.progress(100, text="Completed!")
    st.subheader("Summary:")
    st.success(response, icon= "🤖")

    with st.expander("Section Details: "):
        st.write(section_response)

    st.session_state.video_id = ''
    st.stop()