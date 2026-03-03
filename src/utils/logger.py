import csv
from pathlib import Path

class Logger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.log_path, "w", newline="")
        self.writer = None

    def log(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file, fieldnames=row.keys()
            )
            self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()