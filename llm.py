import google.generative as genai

genai.configure(api_key="AIzaSyBDx7T58AHb5jukzYGWvPmn08wW9vdkUJ8")

model=genai.GenerativeModel(model_name="gemini-2.0-flash")

chat=model.start_chat(history=[])
while True:
    prmt=input()
    if(prmt=="exit"):
    break

res=chat.send_message(prmt)

print(res.text)