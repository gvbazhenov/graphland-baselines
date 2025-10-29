from helpers.arguments import parse_arguments
from experiment import Experiment

if __name__ == '__main__':
    args = parse_arguments()
    if args.is_train:
        dataset_list = args.train_test_setup.get_train_datasets(train_size=args.train_size)
        Experiment(args=args).run_multiple_data(is_train=True, dataset_list=dataset_list)
    else:
        Experiment(args=args).test_single_data()
