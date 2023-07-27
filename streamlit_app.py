# Bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App Framework
st.title("Poem generator")
prompt = st.text_input('Insert your poem topic here (example: "Cats")')


# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a poem title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], # allows to prompt with wiki as a backup
    template = 'write me a poem based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research}'
) # title and wiki research passed through script_template

# Memory (storing history -- good for chat-based)
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
# set input key, store memory key

# Llms
llm = OpenAI(temperature=0.9) # create instance of OpenAI service
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key='script', memory=script_memory)
# output of title chain gets passed through script_chain (grabs both title and script output)
# sequential_chain = SequentialChain(chains = [title_chain, script_chain], input_variables = ['topic'], output_variables=['title', 'script'], verbose = True) 

wiki = WikipediaAPIWrapper()

# trigger our prompt to llm
# show stuff to screen if there is a prompt
if prompt:
    #response = sequential_chain({'topic':prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research=wiki_research)
    st.write(title) #adds to streamlit app
    st.write(script)

    # streamlit expander (render back to screen)
    with st.expander('Title History'):
        st.info(title_memory.buffer) # creates info box that stores memory coming from each chain
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
