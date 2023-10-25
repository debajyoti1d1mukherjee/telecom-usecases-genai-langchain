from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router import MultiPromptChain
#genai
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
import os


telecom_summarization_template = '''You are an expert of content summarization.
Please provide me assistance with summarization.
\n{input}'''

telecom_nba_template = '''You are an expert for suggesting next best action.
Please provide me assistance with next best action.
\n{input}'''

empty_template = 'empty'

prompt_infos = [
    {'name':'empty','description':'Replies to empty questions','prompt_template':empty_template},
    {'name':'Conversation next best action','description': 'Provides next best action from conversation between telecom customer and CSR agent',
     'prompt_template':telecom_nba_template},
    {'name':'Conversation Summarization','description': 'Provides summarization from conversation between telecom customer and CSR agent',
     'prompt_template':telecom_summarization_template},
    
]

api_key = "pak-JBNYgbk8dqHemBnU9pV4qq6VNXNLvlwj7NzAZdhZAU4"
api_endpoint = "https://bam-api.res.ibm.com/v1"
creds = Credentials(api_key, api_endpoint=api_endpoint)

paramsSummary = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=1000,
    min_new_tokens=200
    ).dict()

llm = LangChainInterface(model="meta-llama/llama-2-7b", params=paramsSummary, credentials=creds)
os.environ['OPENAI_API_KEY'] = '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'
llm = ChatOpenAI(temperature=0)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm,prompt=default_prompt)

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, default_chain=default_chain, 
                          verbose=True
                        )

#chain.run("Generate a swagger API definition for telecom customer order management")

document1 = '''
What is summary for the below  conversation ?
User: Hi, I'm having issues with my mobile data. It's not working at all.
Virtual Assistant: Hello! I'm sorry to hear that you're having trouble with your mobile data. Let's get this sorted out for you. Can you please provide me with your mobile number and the name of your mobile carrier?
User: Sure, my mobile number is (555) 123-4567, and my carrier is XYZ Mobile.
Virtual Assistant: Thank you for providing that information. To start, let's try some basic troubleshooting steps. Please turn off your phone and then turn it back on after a few seconds. This can often resolve minor connectivity issues. Let me know if that helps.
User: I've tried that already, but it didn't work.
Virtual Assistant: I see. Let's move on to the next step. Check if you have any outstanding payments on your account. Sometimes, if the bill is overdue, the carrier might restrict data access.
User: I just paid my bill last week, so that shouldn't be the issue.
Virtual Assistant: Understood. In that case, there might be a problem with your network settings. Let's check if your APN (Access Point Name) settings are correct. Could you please navigate to the network settings on your phone and verify the APN settings with the following details: [APN Settings details provided]?
User: I'm not sure where to find the APN settings on my phone.
Virtual Assistant: No worries! The process can vary depending on your phone's make and model. Allow me a moment to look up the specific instructions for your phone.
[Virtual Assistant searches for the instructions and provides step-by-step guidance on finding APN settings.]
User: Found it! The APN settings were incorrect. I've updated them, but my data still isn't working.
Virtual Assistant: Thanks for checking that. If the issue persists, it's possible that there might be a network outage in your area. Let me check for any reported outages.
[Virtual Assistant checks for network outages and confirms if there's one in the user's area.]
Virtual Assistant: I'm sorry to inform you that there is currently a reported network outage in your area. Our technical team is already working on resolving it. The estimated time for restoration is within the next two hours. Please bear with us, and your data should be back up and running soon.
User: Oh, I see. I guess I'll wait for the outage to be fixed then.
Virtual Assistant: Yes, I apologize for the inconvenience caused. If you have any other questions or concerns, feel free to ask. Is there anything else I can assist you with today?
User: That's all for now. Thank you for your help.
Virtual Assistant: You're welcome! If you need any further assistance, don't hesitate to reach out. Have a great day!. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.
'''

document2 = '''
what is the next best action of Virtual Assistant from below conversation?
User: Hi, I'm having issues with my mobile data. It's not working at all.
Virtual Assistant: Hello! I'm sorry to hear that you're having trouble with your mobile data. Let's get this sorted out for you. Can you please provide me with your mobile number and the name of your mobile carrier?
User: Sure, my mobile number is (555) 123-4567, and my carrier is XYZ Mobile.
Virtual Assistant: Thank you for providing that information. To start, let's try some basic troubleshooting steps. Please turn off your phone and then turn it back on after a few seconds. This can often resolve minor connectivity issues. Let me know if that helps.
User: I've tried that already, but it didn't work.
Virtual Assistant: I see. Let's move on to the next step. Check if you have any outstanding payments on your account. Sometimes, if the bill is overdue, the carrier might restrict data access.
User: I just paid my bill last week, so that shouldn't be the issue.
Virtual Assistant: Understood. In that case, there might be a problem with your network settings. Let's check if your APN (Access Point Name) settings are correct. Could you please navigate to the network settings on your phone and verify the APN settings with the following details: [APN Settings details provided]?
User: I'm not sure where to find the APN settings on my phone.
Virtual Assistant: No worries! The process can vary depending on your phone's make and model. Allow me a moment to look up the specific instructions for your phone.
[Virtual Assistant searches for the instructions and provides step-by-step guidance on finding APN settings.]
User: Found it! The APN settings were incorrect. I've updated them, but my data still isn't working.
Virtual Assistant: Thanks for checking that. If the issue persists, it's possible that there might be a network outage in your area. Let me check for any reported outages.
[Virtual Assistant checks for network outages and confirms if there's one in the user's area.]
Virtual Assistant: I'm sorry to inform you that there is currently a reported network outage in your area. Our technical team is already working on resolving it. The estimated time for restoration is within the next two hours. Please bear with us, and your data should be back up and running soon.
User: Oh, I see. I guess I'll wait for the outage to be fixed then.
Virtual Assistant: Yes, I apologize for the inconvenience caused. If you have any other questions or concerns, feel free to ask. Is there anything else I can assist you with today?
User: That's all for now. Thank you for your help.
Virtual Assistant: You're welcome! If you need any further assistance, don't hesitate to reach out. Have a great day!. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.
'''

result = chain.run(document1)
print(result)