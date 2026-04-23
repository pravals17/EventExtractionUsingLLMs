import pandas as pd
import re


"""
This code parses the 5Ws from the LLM outputs and stores them in csv format that can be used for evaluation using the evalution.py code.
"""
def extract_5ws(text):

    result = {
        "Where": "",
        "When": "",
        "What": "",
        "Who": "",
        "Why": ""
    }

    if not isinstance(text, str) or text.strip() == "":
        return result

    # remove markdown markers but keep numbers/symbols
    text = text.replace("**", "")

    pattern = re.compile(
        r'(where|when|what|who|why)\s*[:\-]\s*"?([^";\n]+)"?',
        re.IGNORECASE
    )

    matches = pattern.findall(text)

    for label, value in matches:

        label = label.lower().strip()
        value = value.strip()

        if label == "where":
            result["Where"] = value

        elif label == "when":
            result["When"] = value

        elif label == "what":
            result["What"] = value

        elif label == "who":
            result["Who"] = value

        elif label == "why":
            result["Why"] = value

    return result

def parse_llm_predictions(df, prediction_column):

    extracted = []

    for text in df[prediction_column]:

        parsed = extract_5ws(text)

        extracted.append(parsed)
        

    pred_df = pd.DataFrame(extracted)
    #print(extracted)

    return pred_df


def main():

    input_file = "preds.csv"

    df = pd.read_csv(input_file)

    preds = parse_llm_predictions(df, "FiveWs") #FiveWs is the columns in the outputs from the LLMs

    preds.to_csv("formatedPRedictions", index=False)

    #print(preds.head())


if __name__ == "__main__":
    main()