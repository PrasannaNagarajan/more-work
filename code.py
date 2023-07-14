import os
os.environ["OPENAI_API_KEY"] = '{your_api_key}'

# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api.formatters import JSONFormatter

# srt = YouTubeTranscriptApi.get_transcript("ttJTCEbyiB4&t", languages=['en'])
# formatter = JSONFormatter()
# json_formatted = formatter.format_transcript(srt)
# print(json_formatted)

from llama_index import SimpleDirectoryReader
SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
documents = loader.load_data()

from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import ChatOpenAI

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=500))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context
)

response = index.query("Summerize the video transcript")
print(response)