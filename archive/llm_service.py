import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Qween_local:
    def __init__(self, model_name= "Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        quant_config= BitsAndBytesConfig(load_in_4bit=True)
        self.model=AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config= quant_config,
            device_map= 'auto'
        )

    def generate_answer(self, question: str) -> str:
        message= [{
            "role": "user",
            "content": question
        }]
        prompt= self.tokenizer.apply_chat_template(
            message,
            tokenize= False,
            add_generation_prompt= True
        )
        input= self.tokenizer(prompt, return_tensors= "pt",return_dict= True).to(self.model.device)
        output_ids= self.model.generate(
            **input,
            max_new_tokens= 512,
            temperature= 0.1,
            do_sample= True
        )
        generate_ids= output_ids[0][len(input.input_ids[0]):]
        reponse= self.tokenizer.decode(generate_ids, skip_special_tokens= True)
        return reponse
if __name__ == '__main__':
    llm= Qween_local()

    test_question= "Who are you?"
    answer= llm.generate_answer(test_question)
    print (f"\n[Qwen AI]: {answer}")