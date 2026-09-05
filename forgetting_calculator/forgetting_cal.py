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

    task_names = list(task_data[-1].keys())
    forgetting_list = []

    # Iterate over all previous tasks.
    for j in range(num_tasks - 1):
        dataset_name = task_names[j]  # Name of each task except the final one.
        classes_forgetting = []

        # Read class names from the final task result.
        final_classes = task_data[-1][dataset_name]['AUC per class']
        for cls_name, final_auc in final_classes.items():
            if final_auc is None:
                continue  # Skip invalid classes.

            # Find the historical maximum AUC after the task was introduced.
            max_auc = -1.0
            for t in range(j, num_tasks - 1):
                auc_t = task_data[t][dataset_name]['AUC per class'].get(cls_name)
                if auc_t is not None and auc_t > max_auc:
                    max_auc = auc_t

            if max_auc != -1.0:
                cls_forgetting = max_auc - final_auc
                classes_forgetting.append(cls_forgetting)

        # Compute f_j^k.
        if classes_forgetting:
            f_j = sum(classes_forgetting) / len(classes_forgetting)
            # print(f"Task {j} ({dataset_name}) mean forgetting: {f_j:.5f}")
            forgetting_list.append(f_j)

    # Compute the final F_k.
    F_k = sum(forgetting_list) / len(forgetting_list)
    return F_k


if __name__ == '__main__':
    auc_directory = "result_new"
    for fdir in sorted(os.listdir(auc_directory)):
        F_k = calculate_forgetting(os.path.join(auc_directory, fdir))
        # print(f"Final forgetting for {fdir}: {F_k:.5f}")
        if F_k is not None:
            json_path = os.path.join(auc_directory, fdir, 'forgetting.json')
            with open(json_path, 'w') as f:
                json.dump({'F_k(%)': round(F_k * 100, 4)}, f, indent=4)
        else:
            print(f"Directory {fdir} has fewer than four tasks; forgetting is undefined.")
