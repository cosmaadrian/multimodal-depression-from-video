import yaml
import pprint

import nomenclature
from lib.arg_utils import define_args
from lib.utils import load_model
from lib.loggers import NoLogger

args = define_args(
    extra_args = [
        ('--eval_config', {'default': '', 'type': str, 'required': True}),
        ('--output_dir', {'default': '', 'type': str, 'required': True}),
        ('--checkpoint_kind', {'default': 'best', 'type': str, 'required': False})
    ])

with open(args.eval_config, 'rt') as f:
    eval_cfg = yaml.load(f, Loader = yaml.FullLoader)

architecture = nomenclature.MODELS[args.model](args)

state_dict = load_model(args)
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

architecture.load_state_dict(state_dict)
architecture.eval()
architecture.train(False)
architecture.to(nomenclature.device)

evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args, logger = NoLogger())
    for evaluator_args in eval_cfg['evaluators']
]

for evaluator in evaluators:
    results = evaluator.evaluate(save = True)
    print(evaluator.__class__.__name__)
    pprint.pprint(results)
