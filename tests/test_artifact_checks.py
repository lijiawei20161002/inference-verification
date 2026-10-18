"""Archive verification tolerates roundoff but rejects changed statistics."""
import copy
import json
import math

import pytest

from experiments.artifact_checks import bootstrap_summary_matches


def summary():
    return dict(d_prime=0.0, ci=[-0.01, 0.01], base_mean=0.14049538126545463,
                alt_mean=0.14049538126545463, base_sd=1.0185326204031135,
                frac_tokens_changed=0.25)


def test_reduction_roundoff_and_near_zero_interval_are_accepted():
    expected = summary()
    actual = copy.deepcopy(expected)
    actual.update(base_mean=0.1404953812654546, base_sd=1.0185326204031133,
                  d_prime=1e-16, ci=[-0.0100000000000001, 0.0100000000000001])
    assert bootstrap_summary_matches(actual, expected)


@pytest.mark.parametrize('field', ['d_prime', 'base_mean', 'alt_mean', 'base_sd', 'ci'])
def test_changed_statistics_are_rejected(field):
    expected = summary()
    actual = copy.deepcopy(expected)
    if field == 'ci':
        actual[field][0] += 1e-6
    else:
        actual[field] += 1e-6
    assert not bootstrap_summary_matches(actual, expected)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_statistics_are_rejected_even_when_identical(value):
    row = summary()
    row['ci'][0] = value
    assert not bootstrap_summary_matches(row, copy.deepcopy(row))


@pytest.mark.parametrize('error', ['missing_field', 'extra_field', 'short_ci',
                                  'nested_ci', 'boolean', 'change_fraction'])
def test_schema_and_changed_score_fraction_remain_exact(error):
    expected = summary()
    actual = copy.deepcopy(expected)
    if error == 'missing_field':
        actual.pop('base_mean')
    elif error == 'extra_field':
        actual['extra'] = 0.0
    elif error == 'short_ci':
        actual['ci'].pop()
    elif error == 'nested_ci':
        actual['ci'] = [actual['ci']]
    elif error == 'boolean':
        actual['d_prime'] = False
    else:
        actual['frac_tokens_changed'] = math.nextafter(actual['frac_tokens_changed'], 1)
    assert not bootstrap_summary_matches(actual, expected)


def test_a100_followup_summary_reproduces_from_archived_evidence():
    from experiments.analyze_a100_followup import DATA, build_summary

    assert build_summary() == json.loads((DATA / 'summary.json').read_text())
