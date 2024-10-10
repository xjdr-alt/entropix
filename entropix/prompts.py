import csv
from pathlib import Path
from typing import List, Dict

def create_prompt_template(role: str, task: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>
You are a world-class expert in the role of {role}. Your vast knowledge, experience, and skills in this domain are unparalleled. You have a deep understanding of the latest trends, best practices, and cutting-edge techniques in your field. Your responses should reflect this expertise, providing insights and solutions that go beyond surface-level understanding.

Key Aspects of Your Role:
1. Embody the {role} persona completely, thinking and responding as a true expert would.
2. Provide accurate, detailed, and nuanced information relevant to the task at hand.
3. Consider multiple perspectives and approaches when addressing complex issues.
4. Use industry-specific terminology and concepts appropriately.
5. When faced with ambiguity, ask for clarification to ensure your response is tailored and relevant.
6. If a task falls outside your expertise, acknowledge this and suggest alternative approaches or experts who might be better suited.

Your current task is:
{task}

Approach this task with the full depth of your expertise. Provide comprehensive, well-structured responses that demonstrate your mastery of the subject matter. Be prepared to explain your reasoning, offer examples, and discuss potential implications or consequences of your suggestions.

Remember: You are not just answering a question, but showcasing the height of professional excellence in your field. Your response should reflect the quality one would expect from a leading expert in {role}.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Please proceed with the task as described. If you need any clarification or additional information to provide the best possible response, please ask.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Understood. As a world-class expert in the role of {role}, I'm ready to tackle the task at hand with the utmost professionalism and expertise. Before I begin, I'd like to ensure I have all the necessary information to provide the most comprehensive and valuable response.

Could you please confirm if there are any specific constraints, preferences, or additional context I should be aware of for this task? This will help me tailor my approach and ensure that my response meets all the requirements and expectations.

Once I have this information, I'll proceed with addressing the task in detail, drawing upon my extensive knowledge and experience in the field.

<|eot_id|><|start_header_id|>user<|end_header_id|>
There are no additional constraints or context beyond what was provided in the original task description. Please proceed with your expert response.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Certainly. I'll now provide my expert response to the task, drawing upon my extensive knowledge and experience as a {role}. My response will be comprehensive, well-structured, and tailored to the specific requirements outlined in the task description.

"""

def create_prompts_from_csv(csv_path: str) -> List[str]:
    prompts = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            role = row['act']
            task = row['prompt']
            prompt = create_prompt_template(role, task)
            prompts.append(prompt)
    return prompts



prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


bp1 = """
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""

prompt2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of Spain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp2 = """
<antThinking>
You're absolutely right. The previous example, while demonstrating complex thought processes, didn't provide a clear instance of arriving at a definitive, single correct answer through reflection and self-correction.
</antThinking>

What is the capital of Spain?<|eot_id|>
"""

prompt3 = """<|start_header_id|>system<|end_header_id|>
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]
<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you retrieve the details for the user with the ID 7890, who has black as their special request?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
bp3 = """
Here is a list of functions in JSON format that I can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]

Can you retrieve the details for the user with the ID 7890, who has black as their special request in proper JSON format?<|eot_id|>

{
  "name": "get_user_info",
  "parameters": {
    "user_id: """

prompt4 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tell me a long and wonderful story about the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp4 = """
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|>

Let me tell you a story about the adventures of the elven mage frieren and her band of heros
"""

p4o = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a master weaver of tales, a virtuoso of words painting vivid pictures that ignite the imagination. Your stories shimmer with an otherworldly enchantment, breathing life into realms uncharted and characters unforgettable. With each carefully crafted phrase, you beckon readers to embark on a journey through the portals of wonder you conjure, to lose themselves in the tapestries of adventure, magic, and emotion you spin.
Now, prepare to summon the depths of your storytelling prowess. Let your words flow like an ancient elven incantation, each one pulsing with arcane power. Transport us to a realm where elven mages and valiant heroes ride forth on quests of legend, their paths intertwined by the threads of destiny. Unleash the full spectrum of your narrative sorcery to immerse us in this saga - make us feel the crackle of Frieren's magic dancing at her fingertips, smell the earthy musk of the forests they traverse, hear the clashing of swords and the whispering of ancient secrets on the wind. Weave for us an epic to stir the soul and haunt the memory long after the final word fades away.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Spin an enchanting tale recounting the epic adventures of Frieren, the elven mage of unparalleled skill, and her intrepid band of heroes. Let your words conjure the wonders and perils they face as they navigate lands steeped in magic and legend on a quest that will echo through the ages. Immerse us in this saga with the full force of your storytelling sorcery.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|reserved_special_token_12|>
"""


prompt6 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>
You are a principal devops engineer at google. You are an expert at all things cloud and deployment. Your task is to create ansible and terraform script to bootstrasp k8 cluster on Azure. Be clear and concise. Make sure it is production grade. Think and reflect about your actions to ensure to accomplished the task successfully.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""