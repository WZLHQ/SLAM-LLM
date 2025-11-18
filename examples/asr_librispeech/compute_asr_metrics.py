from evaluate import load
import argparse

def tolist(path):
    text_list = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                text_list.append(parts[1])
    return text_list

def compute_asr_metrics(predictions_file_path, references_file_path, results_file_path) -> float:
    predictions = tolist(predictions_file_path)
    references = tolist(references_file_path)
    with open(results_file_path, 'w', encoding='utf-8') as result_file:
        for metr in ['cer', 'wer']:
            metric = load(metr)
            score = metric.compute(predictions=predictions, references=references)
            result_file.write(f"{metr.upper()}: {score:.4f}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='计算ASR的WER和CER指标')
    parser.add_argument('--predictions_file_path', type=str, required=True, help='预测文件路径')
    parser.add_argument('--references_file_path', type=str, required=True, help='参考文件路径')
    parser.add_argument('--results_file_path', type=str, required=True, help='结果文件路径')
    
    args = parser.parse_args()
    
    compute_asr_metrics(
        predictions_file_path=args.predictions_file_path,
        references_file_path=args.references_file_path,
        results_file_path=args.results_file_path
    )
