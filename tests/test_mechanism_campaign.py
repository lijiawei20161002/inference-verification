import copy

import pytest

from experiments.numerical.run_mechanism_campaign import compare_records


def test_repeat_comparison_rejects_coverage_and_state_drift():
    records = [dict(case=dict(case_id=name), states=dict(kv="original")) for name in ("a", "b")]
    assert compare_records(records, copy.deepcopy(records), ["a", "b"])["exact"]
    for changed in (records[:-1], records[::-1], records + records[:1]):
        with pytest.raises(ValueError, match="mechanism cases"):
            compare_records(records, changed, ["a", "b"])
    changed = copy.deepcopy(records)
    changed[1]["states"]["kv"] = "stale"
    with pytest.raises(ValueError, match="replay differs"):
        compare_records(records, changed, ["a", "b"])
