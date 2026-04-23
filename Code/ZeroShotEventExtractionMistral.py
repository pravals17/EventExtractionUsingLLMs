import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_5ws(
    model,
    tokenizer,
    document_text,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True
    ):
    """
    Extract Who, What, When, Where, Why from text.
    """

    # Structured instruction prompt
    extraction_prompt = f"""
    Extract the 5Ws for the main event from the following document:

    Return your answer strictly in the format like this:
      Where: "..."; When: "..."; What:"..."; Who: "..."; Why: "..."

    If any field is missing, return "Not specified".

    Document:
    {document_text}
    """
    messages = [{
        "role": "user",
        "content":  extraction_prompt
    }]    
    
    # Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text=text_input,
        return_tensors="pt"
    ).to(model.device)

    # Generation config
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_args)

    # Remove prompt tokens
    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    return generated_text
    
def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="/Mistral/cache")
		
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/Mistral/cache")
    
    df = pd.read_csv('/Data/test.csv')
    df_preds = pd.DataFrame(columns=['ArticleID', 'Title', 'Content', 'FiveWs'])
    for i, row in df.iterrows():
        fiveWs= extract_5ws(
            model=model,
            tokenizer=tokenizer,
            document_text=row['Title'] + ' ' +  row['Content']
        )
        temp_df = pd.DataFrame([[row['ArticleID'], row['Title'], row['Content'], fiveWs]], columns=df_preds.columns)
        df_preds = pd.concat([df_preds, temp_df])

if __name__ == "__main__":
    main()

