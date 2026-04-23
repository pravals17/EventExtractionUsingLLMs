import pandas as pd
import re
import string
from rouge_score import rouge_scorer

def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(df, predWhere, predWhen, predWhat, predWho, predWhy):
    """For each W, compute the precision, recall, and f1 score. This code is based on the code provided by Tong et al. (2022)
    for comparing event extraction approaches on DocEE dataset."""
    for w in ['Where', 'When', 'What', 'Who', 'Why']:
        predict_number=0
        annotation_number=0
        right_number=0
        for i, row in df.iterrows():
            if w == 'Where':
                if str(predWhere[i]) != 'Not specified':
                    predict_number=predict_number+1
            elif w == 'When':
                if str(predWhen[i]) != 'Not specified':
                    predict_number=predict_number+1
            elif w == 'What':
                if str(predWhat[i]) != 'Not specified':
                    predict_number=predict_number+1 
            elif w == 'Who':
                if str(predWho[i]) != 'Not specified':
                    predict_number=predict_number+1
            else:
                if str(predWhy[i]) != 'Not specified':
                    predict_number=predict_number+1
               
            if str(row[w]).strip().lower() != 'nan':
                annotation_number=annotation_number+1
            if str(row[w]).strip().lower() != 'nan':

                if w == 'Where':
                    if normalize_string(str(row[w]))==normalize_string(str(predWhere[i])):
                        right_number=right_number+1
                elif w == 'When':
                    if normalize_string(str(row[w]))==normalize_string(str(predWhen[i])):
                            right_number=right_number+1
                elif w == 'What':
                    if normalize_string(str(row[w]))==normalize_string(str(predWhat[i])):
                        right_number=right_number+1
                elif w == 'Who':
                    if normalize_string(str(row[w]))==normalize_string(str(predWho[i])):
                        right_number=right_number+1
                else:
                    if normalize_string(str(row[w]))==normalize_string(str(predWhy[i])):
                        right_number=right_number+1

        precision=round(float(right_number)/float(predict_number),3)
        recall=round(float(right_number)/float(annotation_number),3)
        f=round((2*float(precision)*float(recall))/(float(precision)+float(recall)), 3)

def evaluate_with_rouge(df, predWhere, predWhen, predWhat, predWho, predWhy):
    """For each W, compute the precision, recall, and f1 score using ROUGE-L. The computation is based on the concept uses in Tong et al. (2022) 
    to evaluate event extraction approaches on DocEE dataset. The scores for each W are the average of scores obtained for each document across the test dataset."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for w in ['Where', 'When', 'What', 'Who', 'Why']:
        rouge_p_sum = 0
        rouge_r_sum = 0
        rouge_f_sum = 0
        rouge_count = 0

        for i, row in df.iterrows():
            # Select prediction
            if w == 'Where':
                pred = str(predWhere[i])
            elif w == 'When':
                pred = str(predWhen[i])
            elif w == 'What':
                pred = str(predWhat[i])
            elif w == 'Who':
                pred = str(predWho[i])
            else:
                pred = str(predWhy[i])

            gt = str(row[w])

            if gt.strip().lower() != 'nan':
                # ROUGE-L score computation
                if pred != 'Not specified':
                    scores = scorer.score(
                        normalize_string(gt),
                        normalize_string(pred)
                    )

                    rouge_p_sum += scores['rougeL'].precision
                    rouge_r_sum += scores['rougeL'].recall
                    rouge_f_sum += scores['rougeL'].fmeasure
                    rouge_count += 1

        rouge_p_avg = round(rouge_p_sum / rouge_count, 3) if rouge_count else 0
        rouge_r_avg = round(rouge_r_sum / rouge_count, 3) if rouge_count else 0
        rouge_f_avg = round(rouge_f_sum / rouge_count, 3) if rouge_count else 0        
        
def main():
    df_preds= pd.read_csv('predictions.csv') #preds needs to have columns for preds for 5Ws (Where, When, What, Who, Why) and the corresponding grundtruth for each document in the test dataset
    predWhere = df_preds['Where'].to_list()
    predWhen = df_preds['When'].to_list()
    predWhat = df_preds['What'].to_list()
    predWho = df_preds['Who'].to_list()
    predWhy = df_preds['Why'].to_list()
    df_gt = pd.read_csv('/Data/test.csv')
    exact_match(df_gt, predWhere, predWhen, predWhat, predWho, predWhy)
    evaluate_with_rouge(df_gt, predWhere, predWhen, predWhat, predWho, predWhy)

if __name__ == "__main__":
    main()