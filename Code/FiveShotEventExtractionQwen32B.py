import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_five_shot_messages(five_shot_df, document_text):
    messages = []

    # Five shot examples
    for _, row in five_shot_df.iterrows():

        example_prompt = f"""
            Extract the 5Ws for the main event.

            Document:
            {row['Title']} {row['Content']}
            """

        example_answer = f"""
            Where: "{row['Where']}"; When: "{row['When']}"; What: "{row['What']}"; Who: "{row['Who']}"; Why: "{row['Why']}";
            """

        messages.append({
            "role": "user",
            "content": example_prompt
        })

        messages.append({
            "role": "assistant",
            "content": example_answer
        })

    # Test document 
    test_prompt = f"""
        Now extract the 5Ws for the following document.

        Return strictly in this format:
        Where: "..."; When: "..."; What: "..."; Who: "..."; Why: "..."

        If any field is missing, return "Not specified".

        Document:
        {document_text}
        """

    messages.append({
        "role": "user",
        "content": test_prompt,
    })

    return messages
    
def extract_5ws(
    model,
    tokenizer,
    messages,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True
):
    """
    Generate 5Ws with few-shot messages.
    """

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text=text_input,
        return_tensors="pt"
    ).to(model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_args)

    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return generated_text



def main():
    model_name = "Qwen/Qwen3-32B"
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir="/Qwen32B/cache")
		
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/Qwen32B/cache")

        
    # Load datasets
    train_df = pd.read_csv('/Data/train.csv')
    test_df = pd.read_csv('/Data/test.csv')
    # Random 5-shot examples
    five_shot_examples = train_df.sample(n=5)
    results_df = pd.DataFrame(columns=['ArticleID', 'Title', 'Content', 'FiveWs'])

    for i, row in test_df.iterrows():

        messages = build_five_shot_messages(
            five_shot_df=five_shot_examples,
            document_text=row['Title'] + " " + row['Content']
        )

        fiveWs = extract_5ws(
            model=model,
            tokenizer=tokenizer,
            messages=messages
        )
        temp_df = pd.DataFrame(
            [[row['ArticleID'], row['Title'], row['Content'], fiveWs]],
            columns=results_df.columns
        )
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
if __name__ == "__main__":
    main()