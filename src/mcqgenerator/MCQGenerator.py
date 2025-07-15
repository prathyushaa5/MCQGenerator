from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch,json
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from src.mcqgenerator.logger import logging
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

prompt=PromptTemplate(input_variable=["text","number","subject","tone","response_json"],
                                      template=template)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)
quiz_chain = LLMChain(llm=llm, prompt=prompt, output_key="quiz")

template="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

prompt=PromptTemplate(input_variable=["subject","quiz"],template=template)

llm = HuggingFacePipeline(pipeline=pipe)
review_chain= LLMChain(llm=llm, prompt=prompt, output_key="review")


generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["subject","text", "number", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True)


print("MCQ Generator is ready to use")