from llama_cpp import Llama

llm = Llama(model_path="E:\Langchain\DocQA\models\llama-7b.ggmlv3.q4_0.bin")

response = llm("who is the richest man on earth ?")

print(response['choices'][0]['text'])