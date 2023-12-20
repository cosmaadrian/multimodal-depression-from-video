import os
import pprint
import socket

from collections import defaultdict
import argparse
import sys
import warnings

import yaml
import easydict


def extend_config(cfg_path, child = None):
    if not os.path.exists(cfg_path):
        warnings.warn(f'::: File {cfg_path} was not found!')
        return child

    with open(cfg_path, 'rt', encoding="utf8") as fd:
        parent_cfg = yaml.load(fd, Loader = yaml.FullLoader)

    if child is not None:
        parent_cfg.update(child)

    if '$extends$' in parent_cfg:
        path = parent_cfg['$extends$']
        del parent_cfg['$extends$']
        parent_cfg = extend_config(child = parent_cfg, cfg_path = path)

    return parent_cfg

def load_args(args):
    cfg = extend_config(cfg_path = f'{args.config_file}', child = None)

    if cfg is None:
        return args, cfg

    if '$includes$' in cfg:
        included_cfg = {}
        for included_cfg_path in cfg['$includes$']:
            included_cfg = {**included_cfg, ** extend_config(cfg_path = included_cfg_path, child = None)}

        cfg = {**cfg, **included_cfg}
        del cfg['$includes$']

    for key, value in cfg.items():
        if key in args and args.__dict__[key] is not None:
            continue

        if not isinstance(args, dict):
            args.__dict__[key] = value
        else:
            args[key] = value

    return args, cfg

def flatten_dict_keys(original_key, values):
    new_values = []
    for key, value in values.items():
        new_key = original_key + '.' + key
        if isinstance(value, list):
            new_values.append((new_key, value, type(value)))
            continue

        if isinstance(value, dict):
            extra_new_values = flatten_dict_keys(new_key, value)
            new_values.extend(extra_new_values)

        if not isinstance(value, dict):
            new_values.append((new_key, value, type(value)))

    return new_values

def unflatten_dict_keys(dict_args, args):
    dict_values = dict_args.copy()

    for key, value in args.items():
        if '.' in key:
            continue

        if key in dict_values:
            dict_values[key] = {**dict_values[key], **value}
        else:
            dict_values[key] = value

    unnested = defaultdict(dict)
    for key, value in args.items():
        if '.' not in key:
            continue

        root = '.'.join(key.split('.')[:-1])
        value = {key.split('.')[-1]: value}
        unnested[root].update(value)

    if len(unnested):
        dict_values = unflatten_dict_keys(dict_values.copy(), unnested)

    return dict_values

def update_parser(parser, args):
    for key, value in args.__dict__.items():
        if isinstance(value, dict):
            new_values = flatten_dict_keys(key, value)

            for arg_name, default_value, arg_type in new_values:
                parser.add_argument(f'--{arg_name}', type = arg_type, default = default_value, required = False)

            continue

        if isinstance(value, list):
            parser.add_argument(f'--{key}', type = type(value[0]), default = value, nargs = '+', required = False)
            continue

        if key == 'config_file':
            continue

        parser.add_argument(f'--{key}', type = type(value), default = value)

    return parser

def instantiate_references(flattened_args):
    refs = {}
    for name, value in flattened_args.items():
        if not isinstance(value, str):
            continue

        if value.startswith('${') and value.endswith('}'):
            refs[name] = flattened_args[value[2:-1]]

    for name, value in refs.items():
        flattened_args[name] = value

    return flattened_args

def find_config_file():
    config_path = None
    has_equals = False
    for i in range(len(sys.argv)):
        if '--config_file' in sys.argv[i]:
            if '=' in sys.argv[i]:
                config_path = sys.argv[i].split('=')[-1]
                has_equals = True
            else:
                config_path = sys.argv[i + 1] if len(sys.argv) > i + 1  else None
            break

    if config_path is None:
        raise Exception('::: --config_file is required!')

    sys.argv.pop(i)
    if not has_equals:
        sys.argv.pop(i)

    return config_path

def define_args(extra_args = None, verbose = True, require_config_file = True):

    if require_config_file:
        config_path = find_config_file()
        cfg_args, _ = load_args(argparse.Namespace(config_file = config_path))
    else:
        cfg_args = argparse.Namespace(_={})

    parser = argparse.ArgumentParser(description='Do stuff.')
    parser.add_argument('--name', type = str, default = 'test')
    parser.add_argument('--group', type = str, default = 'default')
    parser.add_argument('--notes', type = str, default = '')
    parser.add_argument("--mode", type = str, default = 'dryrun')
    parser.add_argument("--debug", type = int, default = 0)

    parser.add_argument('--use_amp', type = int, default = 1, required = False)

    parser.add_argument('--env', type = str, default = socket.gethostname())

    if extra_args is not None:
        for name, arguments in extra_args:
            parser.add_argument(name, **arguments)

    parser = update_parser(parser = parser, args = cfg_args)
    flattened_args = parser.parse_args()
    flattened_args = flattened_args.__dict__

    flattened_args = instantiate_references(flattened_args)

    nested_args = unflatten_dict_keys({}, flattened_args)
    args = easydict.EasyDict(nested_args)

    if os.path.exists('configs/env_config.yaml'):
        with open('configs/env_config.yaml', 'rt', encoding="utf8") as fd:
            env_cfg = yaml.load(fd, Loader = yaml.FullLoader)
        args.environment = env_cfg[args.env]

    if args.debug:
        output_string =  "#############################\n"
        output_string += '#############################\n'
        output_string += '########🐞DEBUG MODE🐞########\n'
        output_string += '#############################\n'
        output_string += '#############################\n'

        print(output_string)

    if args.debug:
        print("[🐞DEBUG MODE🐞] Changing name to:", args.name + '-DEBUG')
        print("[🐞DEBUG MODE🐞] Changing WANDB_MODE to 'dryrun'",)
        args.mode = 'dryrun'
        args.name = args.name + '-DEBUG'

    os.environ['WANDB_MODE'] = args.mode
    os.environ['WANDB_NAME'] = args.name
    os.environ['WANDB_NOTES'] = args.notes

    if verbose:
        pprint.pprint(args)

    return args
