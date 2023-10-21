from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain
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


api_key = "<API KEY>"
api_endpoint = "<Model Endpoint>"
creds = Credentials(api_key, api_endpoint=api_endpoint)

paramsSummary = GenerateParams(
decoding_method="greedy",
max_new_tokens=200,
min_new_tokens=200

).dict()

paramsSummary1 = GenerateParams(
decoding_method="greedy",
max_new_tokens=200,
min_new_tokens=200
).dict()

llm = LangChainInterface(model="meta-llama/llama-2-70b-chat", params=paramsSummary, credentials=creds)
llm1 = LangChainInterface(model="ibm/granite-13b-sft", params=paramsSummary1, credentials=creds)

template1 = "Give a summary paragraph of User conversation with Virtual Assistant\n{document}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=llm,
                     prompt=prompt1,
                     output_key="summary")

template2 = "Extract the key Entitiy names in bullet points from summary:\n{summary}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=llm,
                     prompt=prompt2,
                     output_key="entities")

seq_chain = SequentialChain(chains=[chain_1,chain_2],
                            input_variables=['document'],
                            output_variables=['summary','entities'],
                            verbose=True)

document = '''
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

results = seq_chain(document)

print("\n\Summary\n\n")
print(results['summary'])

print("\n\nEntities\n\n")
print(results['entities'])
