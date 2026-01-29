import argparse
import yaml

class Flags:
    """Parse CLI args and merge with YAML config."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MatPlat inputs")
        self.add_required_args()
        args, unknown = self.get_args()
        default_config = self.parse_yml(args.config_file)
        updated_config = self.update_from_args(unknown, default_config)

        for key, val in vars(args).items():
            setattr(updated_config, key, val)
        self.updated_config = updated_config

    def add_required_args(self):
        self.parser.add_argument(
            "--task_type",
            choices=["train", "test", "predict", "visualize", "hyperparameter", "CV"],
            required=True,
            type=str,
            help="Type of task to perform: train, test, predict, visualize, hyperparameter",)
        self.parser.add_argument(
            "--config_file",
            required=True,
            type=str,
            help="Default parameters for training",
            default='./config.yml'
        )

    def get_args(self):
        """Parse known args and keep the rest for overrides."""
        args, unknown = self.parser.parse_known_args()
        return args, unknown

    def parse_yml(self, yml_file):
        """
        解析YAML配置文件并将嵌套结构扁平化
        
        Args:
            yml_file: Path to a YAML config.

        Returns:
            config: Namespace with all config parameters.
        """
        
        def recursive_flatten(nestedConfig, parent_key=''):
            """Flatten a nested dict with preserved subtrees."""
            for k, v in nestedConfig.items():
                current_key = f"{parent_key}.{k}" if parent_key else k
                
                if isinstance(v, dict):
                    if (k.split('_')[-1] == 'args' or 
                        k == 'hyperparameter_search' or 
                        k == 'optuna' or
                        k == 'search_space' or 
                        parent_key in ['hyperparameter_search', 'optuna', 'search_space']):
                        flattenConfig[k] = dict_to_namespace(v)
                    else:
                        recursive_flatten(v, current_key)
                else:
                    flattenConfig[k] = v
        
        def dict_to_namespace(d):
            """Recursively convert a dict to Namespace."""
            if isinstance(d, dict):
                ns = argparse.Namespace()
                for k, v in d.items():
                    setattr(ns, k, dict_to_namespace(v))
                return ns
            else:
                return d

        flattenConfig = {}
        with open(yml_file, 'r', encoding='utf-8') as f:
            nestedConfig = yaml.load(f, Loader=yaml.FullLoader)
        recursive_flatten(nestedConfig, '')

        config = argparse.Namespace()
        for key, val in flattenConfig.items():
            setattr(config, key, val)
            
        return config
    
    def update_from_args(self, unknown_args, ymlConfig):
        """Apply CLI overrides to the YAML config."""
        assert len(unknown_args) % 2 == 0, f"Please Check Arguments, {' '.join(unknown_args)}"
        
        for key, val in zip(unknown_args[0::2], unknown_args[1::2]):
            key = key.strip("--")
            val = self.parse_value(val)
            setattr(ymlConfig, key, val)
            
        return ymlConfig

    def parse_value(self, value):
        """Parse a string into a Python literal when possible."""
        import ast
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
