from typing import Any, Dict, Literal, List, Optional

import json
import re

from datetime import datetime

from pydantic import BaseModel, Field


role_to_header = {
    "system": "user",
    "user": "assistant",
    "assistant": "user",
}

class Message(BaseModel):
  role: Literal["system", "user", "assistant"]
  content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
  model: str = Field(..., min_length=1)
  messages: List[Message] = Field(..., min_items=1)
  temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
  max_tokens: Optional[int] = Field(default=4096, ge=1, le=4096)
  stream: Optional[bool] = Field(default=True)



chat_preamble = "You are a helpful assistant."

def formatted_date():
    current_date = datetime.now()
    return current_date.strftime("%d %B %Y")

def system_header() -> str:
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"

def env(tools: bool) -> str:
    if tools:
        return (
            "Environment: ipython\nTools: brave_search"  # , wolfram_alpha\n\n"
        )
    return "Environment: ipython"

def dates() -> str:
    return f"Cutting Knowledge Date: December 2023\nToday Date: {formatted_date()}\n\n"

def header(text: str, role: str) -> str:
    return f"{text}<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n"

def generate_chat_prompt(request: ChatCompletionRequest) -> str:
    prompt = f"""{system_header()}{env(False)}
Cutting Knowledge Date: December 2023
Today Date: {formatted_date()}

"""
    if request.messages:
        for message in request.messages:
            # Do we care about tool_plan or tool_results or tool_calls here?
            prompt += header(message.content, role_to_header[message.role])

    print(prompt)
    return prompt

# class ToolProcessor:
#     def __init__(self):
#         pass

#     def formatted_date(self):
#         current_date = datetime.now()
#         return current_date.strftime("%d %B %Y")

#     def system_header(self) -> str:
#         return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"

#     def env(self, tools: bool) -> str:
#         if tools:
#             return "Environment: ipython\nTools: brave_search\n"  # , wolfram_alpha\n\n"
#         return "Environment: ipython\n"

#     def dates(self) -> str:
#         return f"Cutting Knowledge Date: December 2023\nToday Date: {self.formatted_date()}\n"

#     def header(self, text: str, role: str) -> str:
#         return f"{text}<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n"

#     def custom_tool(self, tools: List[Dict[str, Any]]) -> str:
#         pre = """# Available Tools
# You have access to the following functions:\n\n"""
#         tool_defs = []
#         for tool in tools:
#             tool_defs.append(
#                 f"""Function: '{tool['name']}'
# Description: {tool['description']}
# Parameters: {json.dumps(tool['parameter_definitions'], indent=2)}
# """
#             )
#         post = """# Tool Usage Instructions
# - To use a tool, format your response as follows:
#   <function={function_name}>{{"param1": "value1", "param2": "value2"}}</function>
# - Only call one function at a time.
# - Place the entire function call on a single line.
# - Always use valid JSON for parameters.
# - After each tool use, analyze the results before deciding on the next step.\n"""

#         return pre + "\n".join(tool_defs) + post

#     def generate_prompt(
#         self,
#         message: str,
#         tools: List[Dict[str, Any]],
#         chat_history: List[Dict[str, Any]],
#         tool_results: List[Dict[str, Any]],
#         documents: List[Dict[str, Any]],
#     ) -> str:
#         prompt = self.system_header()
#         prompt += self.env(bool(tools))
#         prompt += self.dates()

#         prompt += ""

#         if tools:
#             prompt += self.custom_tool(tools)

#         prompt += ""

#         for entry in chat_history:
#             role = self.roles[entry["role"]]
#             prompt += self.header(entry["message"], role)

#         for tool_result in tool_results:
#             prompt += f"<|start_header_id|>ipython<|end_header_id|>\n{json.dumps(tool_result['outputs'])}<|eot_id|>"

#         prompt += self.header(message, "user")
#         prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

#         return prompt

#     def parse_output(self, output: str) -> List[ToolCall]:
#         # code_pattern = r'```python\n(.*?)```'
#         function_call_pattern = r"<function=(\w+)>(.*?)</?(function)>"
#         python_interpreter_pattern = r"<\|python_tag\|>(.*?)<\|eom_id\|>"
#         tool_calls = []
#         function_calls = re.findall(function_call_pattern, output, re.DOTALL)
#         for name, parameters, _ in function_calls:
#             if "brave" in name:
#                 name = "web_search"
#             try:
#                 parameters = json.loads(parameters.strip())
#             except json.JSONDecodeError:
#                 parameters = {"raw_args": parameters.strip()}
#             tool_calls.append(ToolCall(name=name, parameters=parameters))
#         python_calls = re.findall(python_interpreter_pattern, output, re.DOTALL)
#         for code in python_calls:
#             code = code.strip()
#             if code:
#                 tool_calls.append(
#                     ToolCall(
#                         name="toolkit_python_interpreter", parameters={"code": code}
#                     )
#                 )
#         for call in tool_calls:
#             if "python" in call.name:
#                 print(f"Name: {call.name}")
#                 if "code" in call.parameters:
#                     print(f"Code:\n{call.parameters['code']}")
#                 else:
#                     print(f"{call.parameters=}")
#                 print("---")
#         return tool_calls
