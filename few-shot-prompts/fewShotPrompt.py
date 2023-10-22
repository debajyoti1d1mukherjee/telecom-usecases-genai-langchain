from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
#genai
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
import os

template = "You are a helpful assistant that identifies a target Mobile Network Operator for an input plan. The Mobile Network Operators are ABC,DEF,XYZ"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_input_one_text = "Data plan up to 100GB monthly , unlimited voice."
example_input_one = HumanMessagePromptTemplate.from_template(example_input_one_text)

example_output_one_text = "XYZ"
example_output_one = AIMessagePromptTemplate.from_template(example_output_one_text)

example_input_two_text = "Data plan is more than 200GB monthly , 1000 min voice."
example_input_two = HumanMessagePromptTemplate.from_template(example_input_two_text)

example_output_two_text = "ABC"
example_output_two = AIMessagePromptTemplate.from_template(example_output_two_text)

example_input_three_text = "Data plan is more than 200GB monthly , 1000 min voice."
example_input_three= HumanMessagePromptTemplate.from_template(example_input_three_text)

example_output_three_text = "DEF"
example_output_three = AIMessagePromptTemplate.from_template(example_output_three_text)

human_template = "{plan_details}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one, example_output_one, example_input_two,
     example_output_two,example_input_three,example_output_three,human_message_prompt]
)
input_text = "For monthly  data usage of 350GB and 2700 min voice, mno would be"
request = chat_prompt.format_prompt(plan_details=input_text).to_messages()

api_key = "<API Key>"
api_endpoint = "<API Endpoint>"
creds = Credentials(api_key, api_endpoint=api_endpoint)

paramsSummary = GenerateParams(
    decoding_method="sample",
    max_new_tokens=300,
    min_new_tokens=300,
    stream=False,
    temperature=0.05,
    top_k=50,
    top_p=1,
).dict()

#llm = LangChainInterface(model="meta-llama/llama-2-70b-chat", params=paramsSummary, credentials=creds)
os.environ['OPENAI_API_KEY'] = '<API KEY>'
llm = ChatOpenAI(temperature=0)
result = llm(request)

print(result.content)
