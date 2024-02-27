from argparse import ArgumentParser

# Flag to decide if run in local, set to True for run locally False otherwise
RUN_LOCALLY = False

def _clear_args(parsed_args):
    parsed_args.experiment_args = eval(parsed_args.experiment_args)
    parsed_args.dataset_args = eval(parsed_args.dataset_args)
    return parsed_args

if RUN_LOCALLY:
    def parse_arguments():
        # Arguments to edit for changing the domain and the experiment, then parsed
        experiment = 'baseline'
        target_domain = 'cartoon'
        experiment_name = experiment + '/' + target_domain
        dataset_args = {'root': 'data\PACS', 'source_domain': 'art_painting', 'target_domain': target_domain}
        dataset_args_string = f"{{'root': '{dataset_args['root']}', 'source_domain': '{dataset_args['source_domain']}', 'target_domain': '{dataset_args['target_domain']}'}}"

        parser = ArgumentParser()

        parser.add_argument('--seed', type=int, default=0, help='Seed used for deterministic behavior')
        parser.add_argument('--test_only', action='store_true', help='Whether to skip training')
        parser.add_argument('--cpu', action='store_true', help='Whether to force the usage of CPU')

        parser.add_argument('--experiment', type=str, default = experiment)
        parser.add_argument('--experiment_name', type=str, default = experiment_name)
        parser.add_argument('--experiment_args', type=str, default='{}')
        parser.add_argument('--dataset_args', type=str, default = dataset_args_string)

        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--num_workers', type=int, default=5)
        parser.add_argument('--grad_accum_steps', type=int, default=1)

        return _clear_args(parser.parse_args())
else:
    def parse_arguments():
        parser = ArgumentParser()

        parser.add_argument('--seed', type=int, default=0, help='Seed used for deterministic behavior')
        parser.add_argument('--test_only', action='store_true', help='Whether to skip training')
        parser.add_argument('--cpu', action='store_true', help='Whether to force the usage of CPU')

        parser.add_argument('--experiment', type=str, default='baseline')
        parser.add_argument('--experiment_name', type=str, default='baseline')
        parser.add_argument('--experiment_args', type=str, default='{}')
        parser.add_argument('--dataset_args', type=str, default='{}')

        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--num_workers', type=int, default=5)
        parser.add_argument('--grad_accum_steps', type=int, default=1)

        return _clear_args(parser.parse_args())