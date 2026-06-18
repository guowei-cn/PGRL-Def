import argparse
import os, shutil, sys

from torchvision.datasets.utils import download_and_extract_archive

parent_path = os.path.join(os.getcwd().split('PGRL-Def')[0], 'PGRL-Def')
sys.path.append(parent_path)
from lib.rawDataProcessing import download_raw, upsample_raw, createBenign, createPoisonTest, createPoisonTrain, \
    createBenignCifar, extractCifar, cifar_load_meta, createPoisonTestCifar, createPoisonTrainCifar

from lib.dataLoader import target_cifar


def create_data(poison_ratio, target_class):
    # download raw data
    temp_folder = 'temp'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    download_and_extract_archive(url, temp_folder, filename=filename, md5=tgz_md5)

    # extract data
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    tr_data, tr_targets = extractCifar(train_list, temp_folder)
    ts_data, ts_targets = extractCifar(test_list, temp_folder)
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    classes, class_to_idx = cifar_load_meta(temp_folder, meta)
    # # build the benign dataset
    createBenignCifar(tr_data, tr_targets, ts_data, ts_targets, classes)
    # # generate the poisoned test
    createPoisonTestCifar(ts_data, ts_targets, 'poisonTest', classes)
    # create he poisoned dataset
    createPoisonTrainCifar(tr_data, tr_targets, 'poisonTrain_pr_{}_t_{}'.format(poison_ratio, target_cifar), poison_ratio, target_class, classes)

    # remove temp and download file
    shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisoning data generation.")

    # Adding an argument for poison_ratio
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        choices=[0.003, 0.05],
                        help="Poison ratio value (choose between 0.003 and 0.05)")

    # Parse the arguments
    args = parser.parse_args()

    poison_ratio, target_class = args.poison_ratio, target_cifar
    print(poison_ratio, target_class)
    create_data(poison_ratio, target_class)