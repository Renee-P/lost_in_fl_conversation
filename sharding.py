from config import config
from segment import Segmentation
# from rephrase import Rephrasing
# from verify import Verification

if __name__ == "__main__":
  for task in self.config["tasks"]:
    segmenter = Segmentation(config.config)
    segments = segmenter.process_task(task)
    
    # rephraser = Rephrasing(config.config)
    # shards_init = rephraser.process_task(task)

    # verifier = Verification(config.config)
    # shards = verifier.process_task(task)
