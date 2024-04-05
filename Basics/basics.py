from langchain_community.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

load_dotenv()

llm = OpenAI() 


# -----SIMPLE CODE-----

# result = llm("Write a very very short poem")
# print("result")


# -----EMPLOYING CHAINS-----

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very short {language} fuction that will {task}"
)
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="write a test for the following {language} code:\n {code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

result = chain({
    # "langauge": "python",
    # "task": "return a list of numbers"
    "language": args.language,
    "task": args.task
})

# result = llm("Write a very short poem")
# print(result)
# print(result["text"])
print("-----GENERATED CODE-----")
print(result["code"])
print("-----GENERATED TEST-----")
print(result["test"])