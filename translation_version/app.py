# Auther: Kyuwan Choi, Gilead Sciences, October 10, 2024
import os
import openai
import time
from openai import AzureOpenAI
import pyaudio
import streamlit as st
from langchain.memory import ConversationBufferMemory
import wave
#import asyncio

from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper

client = AzureOpenAI(
    api_key="API_KEY",
    api_version="API_version",
    azure_endpoint= "endpoint")

deployment_id = "gilead-whisper-poc"

chunk_file = "C:/temp_audio_chunk.wav"

def main():
    st.markdown('<h1 style="color: darkblue;">OD Voice Assistant️</h1>', unsafe_allow_html=True)

    memory = ConversationBufferMemory(memory_key="chat_history")

    if st.button("Start Recording"):
        while True:
            # Audio Stream Initialization
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

            #Record and save audio chunk
            record_audio_chunk(audio, stream)

            result = client.audio.transcriptions.create(
                file=open(chunk_file, "rb"),
                model=deployment_id,
                response_format='text',
                language = 'Ko'
            )
            text = result

            if text is not None:
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Customer 👤: {text}</div>',
                    unsafe_allow_html=True)

                #os.remove(chunk_file)

                response_llm = get_response_llm(user_question=text, memory=memory)
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">AI Assistant 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)

                play_text_to_speech(text=response_llm)
                #os.remove(chunk_file)
            else:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                break  # Exit the while loop
        print("End Conversation")



if __name__ == "__main__":
    main()
    #asyncio.run(main())