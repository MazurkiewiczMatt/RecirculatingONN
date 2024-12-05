from datasets import get_dataset

if __name__ == "__main__":

    dataset_name = "iris"
    # available: "iris" (150 datapoints, 4 features); "tic-tac-toe" (958, 9); "car evaluation" (1728, 6)

    dataset = get_dataset(dataset_name)