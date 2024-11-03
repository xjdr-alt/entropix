from openai import OpenAI
import time

# Initialize client
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-test-key")


def test_streaming():
  print("\nTesting streaming response:")
  stream = client.chat.completions.create(
    model="entropix-1b",
    messages=[
      {"role": "system", "content": "You are a world class problem solver. You always think step-by-step and come to the proper solutions."},
      {
        "role": "user",
        "content": "Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?",
      },
    ],
    stream=True,
  )

  # stream = client.chat.completions.create(
  #  model="entropix-1b",
  #  messages=[
  #    {"role": "system", "content": "You are a world class problem solver. You always think step-by-step and come to the proper solutions."},
  #    {
  #      "role": "user",
  #      "content": "Think carefully in a step-by-step manner. Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?",
  #    },
  #  ],
  #  stream=True,
  # )

  full_response = ""
  choices = {}
  try:
    for chunk in stream:
      for choice in chunk.choices:
        if choice.delta.content is not None:
          content = choice.delta.content
          #print(content, end="", flush=True)
          if choice.index not in choices:
            choices[choice.index] = ""
          choices[choice.index] += content
          full_response += content
  except Exception as e:
    print(f"Error during testing: {str(e)}")
  print("\n")
  for k,v in choices.items():
    print(f"Choice {k}: {v}")
  return full_response


# def test_non_streaming():
#     print("\nTesting non-streaming response:")
#     completion = client.chat.completions.create(
#         model="entropix-1b",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?"}
#         ],
#         stream=False
#     )
#     print(completion.choices[0].message.content)
#     return completion.choices[0].message.content


def main():
  print("Starting API tests...")

  try:
    # Test streaming
    streaming_response = test_streaming()
    print(f"\nStreaming response length: {len(streaming_response)}")

    # Add a small delay between tests
    # time.sleep(1)

    # Test non-streaming
    # non_streaming_response = test_non_streaming()
    # print(f"\nNon-streaming response length: {len(non_streaming_response)}")

  except Exception as e:
    print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
  main()
