def gen_BIO(_line, labels):
    label = len(_line) * ['O']
    for _preditc in labels:
        label[_preditc['start']] = 'B-' + _preditc['labels']
        label[_preditc['start'] + 1: _preditc['end']] = (_preditc['end'] - _preditc[
            'start'] - 1) * [('I-' + _preditc['labels'])]

    return label