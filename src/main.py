from dataset import DatasetHandler


def main():
    # Download the datasets
    ds_handler = DatasetHandler()
    ds_handler.download_dataset()
    X_train, y_train, X_valid, y_valid = ds_handler.split_train_valid()

    img_resize = (128, 128)
    batch_size = 16
    train_it = ds_handler.get_train_generator(X_train, y_train, img_resize, batch_size)
    valid_it = ds_handler.get_train_generator(X_valid, y_valid, img_resize, batch_size, is_validation_set=True)
    t = next(train_it)
    v = next(valid_it)
    d = 0

if __name__ == "__main__":
    main()
