import csv
import glob
import numpy as np
import sklearn.metrics as metrics

def combine_folds(model_name, results_dir, output_file):
    result_files = glob.glob(f"{results_dir}/eval_fold*.csv")
    combined_results = []
    for file in result_files:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                combined_results.append(row)
    if combined_results:
        fieldnames = combined_results[0].keys()
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_results)
    
    # after combining, sort the rows by 'Night' - 2020/09-15
    with open(output_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        sorted_results = sorted(reader, key=lambda x: x['Night'])
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_results)

def add_true_ahi_column(input_csv, reference_csv, output_csv):
    reference_dict = {}
    with open(reference_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reference_dict[row['Night']] = row['Scored AHI']
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    fieldnames = reader.fieldnames + ['True AHI']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            night = row['Night']
            row['True AHI'] = reference_dict.get(night, 'N/A')
            writer.writerow(row)

def csv_to_cutoff_results(input_csv, output_txt, cutoffs=[5,10,15,20,25,30]):
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # reference_ahi = [float(row['Scored AHI']) for row in rows]
    reference_ahi = [float(row['True AHI']) for row in rows]
    predicted_ahi = [float(row['Pred AHI']) for row in rows]
    with open(output_txt, 'w') as txtfile:
        for cutoff in cutoffs:
            tp = sum(1 for ref, pred in zip(reference_ahi, predicted_ahi) if ref >= cutoff and pred >= cutoff)
            tn = sum(1 for ref, pred in zip(reference_ahi, predicted_ahi) if ref < cutoff and pred < cutoff)
            fp = sum(1 for ref, pred in zip(reference_ahi, predicted_ahi) if ref < cutoff and pred >= cutoff)
            fn = sum(1 for ref, pred in zip(reference_ahi, predicted_ahi) if ref >= cutoff and pred < cutoff)
            # print(tp, tn, fp, fn)
            under_cutoff = sum(1 for ref in reference_ahi if ref < cutoff)
            over_cutoff = sum(1 for ref in reference_ahi if ref >= cutoff)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            cm = metrics.confusion_matrix(
                [1 if ref>= cutoff else 0 for ref in reference_ahi],
                [1 if pred >= cutoff else 0 for pred in predicted_ahi]
            )
            # cm count to percent
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # auc for different cutoffs can be approximated as average of sensitivity and specificity
            # auc = metrics.roc_auc_score(
            #     [1 if ref >= cutoff else 0 for ref in reference_ahi],
            #     [1 if pred >= cutoff else 0 for pred in predicted_ahi],
            #     average='weighted'
            # ) if len(set([1 if ref >= cutoff else 0 for ref in reference_ahi])) > 1 else 0
            ref = [1 if r >= cutoff else 0 for r in reference_ahi]
            pred = [1 if p >= cutoff else 0 for p in predicted_ahi]
            auc = metrics.roc_auc_score(ref, pred) if len(set(ref)) > 1 else 0
            # macro average f1
            mf1 = metrics.f1_score(
                [1 if ref >= cutoff else 0 for ref in reference_ahi],
                [1 if pred >= cutoff else 0 for pred in predicted_ahi],
                average='macro'
            )

            txtfile.write(f"Cutoff: {cutoff}\n")
            txtfile.write(f"People below cutoff: {under_cutoff}\n")
            txtfile.write(f"People above cutoff: {over_cutoff}\n")
            txtfile.write(f"Sensitivity: {sensitivity:.4f}\n")
            txtfile.write(f"Specificity: {specificity:.4f}\n")
            # txtfile.write(f"Precision: {precision:.4f}\n")
            txtfile.write(f"PPV: {ppv:.4f}\n")
            txtfile.write(f"NPV: {npv:.4f}\n")
            txtfile.write(f"AUC: {auc:.4f}\n")    
            txtfile.write(f"Macro F1: {mf1:.4f}\n\n")
            txtfile.write(f"Confusion Matrix:\n{cm}\n\n")
    # general measures
    ahi_differences = [pred - ref for ref, pred in zip(reference_ahi, predicted_ahi)]
    mae = np.mean(np.abs(ahi_differences))
    rmse = np.sqrt(np.mean([diff**2 for diff in ahi_differences]))

    with open(output_txt, 'a') as txtfile:
        txtfile.write(f"Overall MAE: {mae:.4f}\n")
        txtfile.write(f"Overall RMSE: {rmse:.4f}\n")


if __name__ == "__main__":
    # model_name = "cnnt"
    # model_name = "fbank_best_auc_cnnt"
    model_name = "fbank_best_auc_weighted_cnnt"
    results_dir = f"./results/{model_name}"
    output_file = f"./results/eval_{model_name}_combined.csv"
    combine_folds(model_name, results_dir, output_file)
    reference_csv = "./results/AHI_correct.csv"
    output_with_true_ahi = f"./results/eval_{model_name}_combined_with_true_ahi.csv"
    add_true_ahi_column(output_file, reference_csv, output_with_true_ahi)

    output_txt = f"./results/{model_name}/eval_{model_name}_cutoff_results.txt"
    csv_to_cutoff_results(output_with_true_ahi, output_txt)