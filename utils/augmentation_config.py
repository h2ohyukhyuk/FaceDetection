"""Configuration for the original WIDER FACE training pipeline."""
from copy import deepcopy
import math

DEFAULTS = {
    'SRC_FACE_MIN': 24, 'MIN_FACE_RATIO': .07, 'BACKGROUND_PROB': .2,
    'FILL_VALUE': 128, 'PIXEL_NOISE': .4, 'GAMMA': .3,
    'BLUR_PROB': .1, 'BLUR_KERNEL_RANGE': [3, 11],
    'TRANSLATION_MARGIN_RATIO': .1, 'TRANSLATION_MIN_MARGIN': 5,
    'TRANSLATION_MAX_MARGIN_RATIO': .45,
    'FOREGROUND': {'MIN_FACE_SIZE': 10, 'SCALE_RANGE': [.5, 2.],
                   'SCALE_MIN_GAP': 10, 'ROTATION_DEGREES': 10,
                   'ROTATION_PROB': .5, 'FLIP_PROB': .5,
                   'CROP_SCALE_RANGE': [.2, 1.]},
    'BACKGROUND': {'SCALE_RANGE': [.5, 2.], 'ROTATION_DEGREES': 30,
                   'FLIP_PROB': .5, 'CROP_SCALE_RANGE': [.2, .95]},
}


def resolve_augmentation(options=None):
    def merge(defaults, supplied):
        if not isinstance(supplied, dict):
            raise ValueError('AUGMENTATION sections must be mappings')
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown AUGMENTATION keys: {sorted(unknown)}')
        result = deepcopy(defaults)
        for key, value in supplied.items():
            result[key] = merge(defaults[key], value) if isinstance(defaults[key], dict) else value
        return result
    c = merge(DEFAULTS, options or {})
    def number(value, name, low, high=math.inf):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'AUGMENTATION.{name} must be in [{low}, {high}]')
    for key in ('BACKGROUND_PROB', 'BLUR_PROB', 'MIN_FACE_RATIO'):
        number(c[key], key, 0, 1)
    number(c['SRC_FACE_MIN'], 'SRC_FACE_MIN', 1e-6)
    number(c['FILL_VALUE'], 'FILL_VALUE', 0, 255)
    if int(c['FILL_VALUE']) != c['FILL_VALUE']:
        raise ValueError('FILL_VALUE must be an integer')
    number(c['PIXEL_NOISE'], 'PIXEL_NOISE', 0, 1)
    number(c['GAMMA'], 'GAMMA', 0, .999999)
    number(c['TRANSLATION_MARGIN_RATIO'], 'TRANSLATION_MARGIN_RATIO', 0)
    number(c['TRANSLATION_MIN_MARGIN'], 'TRANSLATION_MIN_MARGIN', 0)
    number(c['TRANSLATION_MAX_MARGIN_RATIO'], 'TRANSLATION_MAX_MARGIN_RATIO', 0, .499999)
    for section in ('FOREGROUND', 'BACKGROUND'):
        item = c[section]
        for key in ('FLIP_PROB', 'ROTATION_PROB'):
            if key in item:
                number(item[key], section+'.'+key, 0, 1)
        number(item['ROTATION_DEGREES'], section+'.ROTATION_DEGREES', 0)
        for key in ('SCALE_RANGE', 'CROP_SCALE_RANGE'):
            pair = item[key]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f'{section}.{key} requires [min, max]')
            for value in pair:
                number(value, section+'.'+key, 1e-6, 1 if key == 'CROP_SCALE_RANGE' else math.inf)
            if pair[0] > pair[1]:
                raise ValueError(f'{section}.{key}: min must be <= max')
    number(c['FOREGROUND']['MIN_FACE_SIZE'], 'FOREGROUND.MIN_FACE_SIZE', 1e-6)
    number(c['FOREGROUND']['SCALE_MIN_GAP'], 'FOREGROUND.SCALE_MIN_GAP', 0)
    pair = c['BLUR_KERNEL_RANGE']
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(v) is not int or v < 1 for v in pair) or pair[0] > pair[1]:
        raise ValueError('BLUR_KERNEL_RANGE requires positive integer [min, max]')
    return c


def input_hw(input_size):
    if (not isinstance(input_size, (list, tuple)) or len(input_size) != 2 or
            any(type(v) is not int or v <= 0 or v % 32 for v in input_size)):
        raise ValueError('MODEL.INPUT_SIZE must be [H, W], positive multiples of 32 for APAG FPN2')
    return tuple(input_size)
