import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title='Meditron Chatbot', layout='wide')
st.title("Meditron Chatbot")

@st.cache(allow_output_mutation=True)
def load_model():
    model_id = "malhajar/meditron-7b-chat"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                 torch_dtype=torch.float16,
                                                 revision="main")
    return model

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    model_id = "malhajar/meditron-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

question = st.text_input("Enter your question:")
prompt = f'''
### Instruction:
{question} 

### Response:'''

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id, top_k=50, do_sample=True, top_p=0.95)
response = tokenizer.decode(output[0], skip_special_tokens=True)

st.markdown(response)