from src.config import DATA_FILE, TEST_INDEX, EPOCHS
from src.data_loader import load_all_samples, make_train_val_test_split, make_tf_datasets
from src.model import MultiTaskPINN
from src.trainer import train_model
from src.evaluation import evaluate_test_sample, plot_training_samples


def main():
    data = load_all_samples(DATA_FILE)
    split = make_train_val_test_split(data, test_index=TEST_INDEX)
    train_ds, val_ds = make_tf_datasets(split)

    model = MultiTaskPINN(hemo_dim=split["H_train"].shape[1])
    model, history = train_model(model, train_ds, val_ds, epochs=EPOCHS)

    evaluate_test_sample(model, split)
    plot_training_samples(model, data, split["test_index"])


if __name__ == "__main__":
    main()
