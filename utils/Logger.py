import json
import os

class Logger:
    def __init__(self, args, loss_logger=0):
        os.makedirs(args.logdir, exist_ok=True)
        self.result_file = os.path.join(args.logdir, 'result.txt')
        if os.path.exists(self.result_file):
            os.remove(self.result_file)

        self.args_json_file = os.path.join(args.logdir, 'args.json')
        # print(self.args_json_file)
        with open(self.args_json_file, 'w', encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        if loss_logger:
            self.loss_file = os.path.join(args.logdir, 'loss.txt')
            print("Loss will be logged to:", self.loss_file)
            if os.path.exists(self.loss_file):
                os.remove(self.loss_file)            
        
    def log(self, message):
        if not isinstance(message, str):
            message = str(message)

        with open(self.result_file, 'a') as f:
            f.write(message)
            f.write('\n')

    def log_loss(self, loss):
        if not isinstance(loss, str):
            loss = str(loss)

        with open(self.loss_file, 'a') as f:
            f.write(loss)
            f.write('\n')