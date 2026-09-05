import json
import os


def calculate_forgetting(auc_dir):
    auc_dir = os.path.join(auc_dir, f'auc')
    num_tasks = len(os.listdir(auc_dir))  # Count task_X.json files.

    if num_tasks < 4:
        return None  # Forgetting is undefined for fewer than four tasks.

    # Load results from all rounds.
    task_data = []
    for t in range(num_tasks):
        with open(os.path.join(auc_dir, f'task_{t}.json'), 'r') as f:
            task_data.append(json.load(f))

    # Store the average forgetting score for each stage.
    F_k_stages = []

    # Start after the first task; k denotes the current evaluation stage.
    for k in range(1, num_tasks):
        forgetting_at_k = []
        for j in range(k):  # Iterate over previous tasks.
            dataset_name = list(task_data[k].keys())[j]
            classes_forgetting = []

            # AUC for task j at the end of stage k.
            current_classes = task_data[k][dataset_name]['AUC per class']
            for cls_name, current_auc in current_classes.items():
                if current_auc is None:
                    continue

                # Find the historical maximum AUC for task j.
                max_auc = -1.0
                for t in range(j, k):
                    auc_t = task_data[t][dataset_name]['AUC per class'].get(cls_name)
                    if auc_t is not None and auc_t > max_auc:
                        max_auc = auc_t

                if max_auc != -1.0:
                    classes_forgetting.append(max_auc - current_auc)

            if classes_forgetting:
                f_j_k = sum(classes_forgetting) / len(classes_forgetting)
                forgetting_at_k.append(f_j_k)

        if forgetting_at_k:
            avg_F_k = sum(forgetting_at_k) / len(forgetting_at_k)
            F_k_stages.append(avg_F_k)

    if not F_k_stages:
        return None

    # Compute the final score as the mean of intermediate scores.
    final_F_k = sum(F_k_stages) / len(F_k_stages)
    return final_F_k


if __name__ == '__main__':
    auc_directory = "result_new"
    for fdir in sorted(os.listdir(auc_directory)):
        F_k = calculate_forgetting(os.path.join(auc_directory, fdir))
        # print(f"Final forgetting for {fdir}: {F_k:.5f}")
        if F_k is not None:
            json_path = os.path.join(auc_directory, fdir, 'forgetting_average.json')
            with open(json_path, 'w') as f:
                json.dump({'F_k(%)': round(F_k * 100, 4)}, f, indent=4)
        else:
            print(f"Directory {fdir} has fewer than four tasks; forgetting is undefined.")
