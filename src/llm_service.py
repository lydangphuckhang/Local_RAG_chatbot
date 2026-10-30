import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

class QwenLocal:
    """
    A service class to load the local Qwen model with 4-bit quantization 
    and wrap it in a Langchain-compatible HuggingFacePipeline
    """
    def __init__(self, model_name:str= "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialize the tokenizer, quantized model, and Langchain wrapper.

        Args:
            model_name(str): The HuggingFace repository ID for the Qwen model.
        Raises:
            RuntimeError: If the model or pipeline fails to load.
        """
        self.model_name= model_name
        self.llm =self._initialize_llm()

    def _initialize_llm(self) -> HuggingFacePipeline:
        try:
            tokenizer= AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True)
            quant_config= BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True, 
                bnb_4bit_compute_dtype=torch.float16,
                load_in_4bit=True)

            model=AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config= quant_config,
                device_map='auto',
                local_files_only=True
            )
            hug_pipe= pipeline(
                "text-generation",
                model= model,
                tokenizer= tokenizer,
                max_new_tokens=512,
                temperature= 0.1,
                do_sample= True,
                return_full_text= False
            )
            llm= HuggingFacePipeline(pipeline= hug_pipe)
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen local LLM: {str(e)}")

    def get_llm(self)-> HuggingFacePipeline:
        """
        Retrieve the langchain-compatible LLM object.

        Returns:
            HuggingFacePipeline: The wrapped language model.

        """
        return self.llm

if __name__ == "__main__":
    try:
        qwen_service= QwenLocal()
        llm= qwen_service.get_llm()

        test_question= "Who are you? Please answer in short sentence."
        print(f"\nUser:{test_question}")
        reponse= llm.invoke(test_question)
        print(f"RAG: {reponse}")
    except Exception as e:
        print(f"Test failed: {str(e)}")